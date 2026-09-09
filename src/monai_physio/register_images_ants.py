"""ANTs-based image registration implementation.

This module provides the RegisterImagesANTS class, a concrete implementation of
RegisterImagesBase that uses the Advanced Normalization Tools (ANTs) algorithm
for image registration. It supports both affine and deformable (SyN) registration
for aligning medical images, particularly useful for 4D cardiac CT registration.

The module uses the antspyx package which provides Python bindings to the ANTs
C++ library, offering robust and well-established registration algorithms.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Union

import ants
import itk
import numpy as np
from numpy.typing import NDArray

from .register_images_base import RegisterImagesBase
from .transform_tools import TransformTools

if TYPE_CHECKING:  # typed for mypy; never imported at runtime
    # antspyx re-exports these at the top level in some releases and not in
    # others (0.6.1 has ants.ANTsImage, 0.5.3 does not), so they are named from
    # the module they are defined in, which both carry.  ``ants.ants_image`` is
    # that module, not the class -- annotating with it says nothing.
    from ants.core.ants_image import ANTsImage
    from ants.core.ants_transform import ANTsTransform


class RegisterImagesANTS(RegisterImagesBase):
    """ANTs-based deformable image registration implementation.

    This class extends RegisterImagesBase to provide deformable image registration
    using the Advanced Normalization Tools (ANTs) algorithm. It supports various
    registration types including affine, deformable (SyN), and elastic registration
    for aligning medical images.

    ANTs is a well-established image registration framework with proven accuracy
    and robustness for medical imaging applications. The SyN (Symmetric
    Normalization) algorithm provides diffeomorphic registration with inverse
    consistency.

    ANTs-specific features:
    - Multiple transform types: Rigid, Affine, SyN, ElasticSyN
    - Robust optimization algorithms
    - Support for multi-resolution registration
    - Symmetric normalization for inverse consistency
    - Comprehensive metric options (MI, CC, Mattes)

    Inherits from RegisterImagesBase:
    - Fixed and moving image management
    - Binary mask processing with optional dilation
    - Modality-specific parameter configuration
    - Standardized registration interface

    Attributes:
        type_of_transform (str): The registration transform type (default: 'SyN')
        grad_step (float): Gradient step size for SyN (default: 0.2)
        flow_sigma (float): Smoothing parameter for regularization (default: 3.0)
        total_sigma (float): Total field smoothing (default: 0.0)
        syn_metric (str): Similarity metric for SyN (default: 'CC')
        syn_sampling (int): Sampling strategy (default: 2)
        reg_iterations (tuple): Iterations per resolution level (default: (40, 20, 0))
        metric (str): Similarity metric to use ('CC', 'Mattes', or 'MeanSquares', default: 'CC')

    Example:
        >>> registrar = RegisterImagesANTS()
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(reference_image)
        >>> registrar.set_transform_type('Affine')
        >>> registrar.set_metric('Mattes')
        >>> result = registrar.register(moving_image)
        >>> moving_to_fixed_transform = result['moving_to_fixed_transform']
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the ANTs image registration class.

        Calls the parent RegisterImagesBase constructor to set up common parameters.
        Default ANTs registration parameters are set to work well for medical images.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.number_of_iterations: list[int] = [40, 20, 10]
        self.transform_type = "Deformable"
        self.metric = "CC"

    def set_number_of_iterations(self, number_of_iterations: list[int]) -> None:
        """Set the number of iterations for ANTs registration.

        Args:
            number_of_iterations: List of iterations for multi-resolution registration
                (e.g., [40, 20, 10] for three resolution levels)
        """
        self.number_of_iterations = number_of_iterations

    def set_transform_type(self, transform_type: str) -> None:
        """Set the type of transform to use for registration.

        Args:
            transform_type (str): Type of transform to use for registration.
                Options: 'Deformable', 'Affine', 'Rigid'
        """
        self.transform_type = transform_type
        if transform_type not in ["Deformable", "Affine", "Rigid"]:
            self.log_error("Invalid transform type: %s", transform_type)
            raise ValueError(f"Invalid transform type: {transform_type}")

    def set_metric(self, metric: str) -> None:
        """Set the similarity metric to use for registration.

        Args:
            metric (str): Similarity metric to use for registration.
                Options: 'CC' (cross-correlation), 'Mattes' (Mattes mutual information),
                'MeanSquares' (mean squares difference)
        """
        self.metric = metric
        if metric not in ["CC", "Mattes", "MeanSquares"]:
            self.log_error("Invalid metric: %s", metric)
            raise ValueError(f"Invalid metric: {metric}")

    def _ants_to_itk_image(self, ants_image: ANTsImage) -> itk.Image:
        """Convert ANTs image back to ITK format.

        Args:
            ants_image (ANTsImage): ANTs image to convert
            reference_itk_image (itk.image): Reference ITK image for metadata

        Returns:
            itk.image: Converted ITK image
        """
        data = ants_image.numpy()

        image_dimension = ants_image.dimension
        if image_dimension not in (2, 3, 4):
            raise ValueError(f"Unsupported ANTs image dimension: {image_dimension}")

        is_vector = ants_image.components > 1

        data_reshaped: NDArray[np.float64]
        if is_vector:
            # Vector images: ANTs gives (components, z, y, x) or (components, y, x)
            data_reshaped = data.transpose(
                list(range(image_dimension - 1, -1, -1)) + [image_dimension]
            ).astype(np.float64)
        else:
            data_reshaped = data.transpose(list(range(image_dimension - 1, -1, -1)))

        img_itk: itk.Image
        if is_vector:
            img_itk = itk.GetImageFromArray(data_reshaped, is_vector=True)
        else:
            img_itk = itk.GetImageFromArray(data_reshaped)

        spacing = ants_image.spacing
        origin = ants_image.origin
        direction_reshaped: NDArray[np.floating] = np.asarray(
            ants_image.direction
        ).reshape((image_dimension, image_dimension))

        img_itk.SetSpacing(spacing)
        img_itk.SetOrigin(origin)
        img_itk.SetDirection(direction_reshaped)

        return img_itk

    def _itk_to_ants_image(
        self, itk_image: itk.Image, dtype: str = "float"
    ) -> ANTsImage:
        """Convert ITK image to ANTs format.

        Args:
            itk_image (itk.image): ITK image to convert

        Returns:
            ANTsImage: Converted ANTs image
        """
        ndim = itk_image.GetImageDimension()
        if ndim not in (2, 3, 4):
            raise ValueError(f"Unsupported ITK image dimension: {ndim}")

        is_vector = itk_image.GetNumberOfComponentsPerPixel() > 1

        if dtype == "float":
            data = itk.GetArrayFromImage(itk_image).astype(np.float32)
        elif dtype == "double":
            data = itk.GetArrayFromImage(itk_image).astype(np.float64)
        elif dtype == "int":
            data = itk.GetArrayFromImage(itk_image).astype(np.int32)
        elif dtype == "uint":
            data = itk.GetArrayFromImage(itk_image).astype(np.uint32)
        elif dtype == "uchar":
            data = itk.GetArrayFromImage(itk_image).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if is_vector:
            spatial_shape = data.shape[:-1]  # drop components
        else:
            spatial_shape = data.shape

        image_dimension = len(spatial_shape)

        direction = itk.array_from_matrix(itk_image.GetDirection())
        spacing = list(itk_image.GetSpacing())
        origin = list(itk_image.GetOrigin())

        # Reshape the array properly for ANTsPy
        if is_vector:
            data_reshaped = data.transpose(
                list(range(image_dimension - 1, -1, -1)) + [image_dimension]
            )
        else:
            data_reshaped = data.transpose(list(range(image_dimension - 1, -1, -1)))

        ants_image: ANTsImage = ants.from_numpy(
            data=data_reshaped,
            origin=origin,
            spacing=spacing,
            direction=direction,
            has_components=is_vector,
        )

        return ants_image

    def _antsfile_to_itk_affine_transform(
        self, ants_transform_file: str
    ) -> itk.Transform:
        """Convert ANTs affine transform to ITK affine transform.

        ANTs affine transform has 12 parameters for 3D:
        - parameters[0:9]: 3x3 transformation matrix in row-major order
        - parameters[9:12]: translation vector
        - fixed_parameters[0:3]: center of rotation

        Args:
            ants_transform_file (str): Path to ANTs transform file

        Returns:
            itk.AffineTransform: Converted ITK affine transform
        """
        ants_tfm = ants.read_transform(ants_transform_file)

        params = np.array(ants_tfm.parameters)
        fixed_params = np.array(ants_tfm.fixed_parameters)

        # Parameters structure for 3D affine:
        # params[0:9] = 3x3 matrix in row-major order
        # params[9:12] = translation vector
        # fixed_params[0:3] = center of rotation

        # Create ITK affine transform
        affine_tfm = itk.AffineTransform[itk.D, 3].New()

        # Set the center of rotation (fixed parameters)
        center = itk.Point[itk.D, 3]()
        for i in range(3):
            center[i] = fixed_params[i]
        affine_tfm.SetCenter(center)

        # Set the 3x3 transformation matrix (first 9 parameters in row-major order)
        mat = np.zeros((3, 3), dtype=np.float64)
        for row in range(3):
            for col in range(3):
                mat[row, col] = params[row * 3 + col]
        mat_itk = itk.GetMatrixFromArray(mat)
        affine_tfm.SetMatrix(mat_itk)

        # Set the translation vector (last 3 parameters)
        translation = itk.Vector[itk.D, 3]()
        for i in range(3):
            translation[i] = params[9 + i]
        affine_tfm.SetTranslation(translation)

        return affine_tfm

    def _antsfile_to_itk_displacement_field_transform(
        self, ants_transform_file: str, ref_image: itk.Image
    ) -> itk.Transform:
        """Create ITK displacement field from ANTs transform.

        Args:
            ants_transform_file (str): Path to ANTs transform file
            reference_image (itk.image): Reference image for field generation

        Returns:
            itk.DisplacementFieldTransform: ITK displacement field transform
        """
        disp_field_tfm_ANTS = ants.read_transform(
            ants_transform_file, precision="double"
        )
        disp_field_ANTS = ants.transform_to_displacement_field(
            disp_field_tfm_ANTS,
            self._itk_to_ants_image(ref_image, dtype="float"),
        )

        disp_field_itk_raw = self._ants_to_itk_image(disp_field_ANTS)

        # Convert to the correct Image[Vector[D, 3], 3] type for DisplacementFieldTransform
        # Use ImageTools helper to convert array to vector image with correct type
        from .image_tools import ImageTools

        image_tools = ImageTools()

        disp_array = itk.array_from_image(disp_field_itk_raw)
        disp_field_itk = image_tools.convert_array_to_image_of_vectors(
            disp_array, ref_image, itk.D
        )

        # Create displacement field transform
        disp_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
        disp_tfm.SetDisplacementField(disp_field_itk)

        return disp_tfm

    def itk_affine_transform_to_ANTS_transform(
        self, itk_tfm: itk.Transform
    ) -> ANTsTransform:
        """Convert ITK affine/rigid transform to ANTs affine transform.

        Converts an ITK MatrixOffsetTransformBase-derived transform (such as
        AffineTransform or Rigid3DTransform) to an ANTs affine transform object.

        The conversion extracts:
        - 3x3 transformation matrix (converted to row-major order for ANTs)
        - Translation vector
        - Center of rotation (fixed parameters)

        Args:
            itk_tfm (itk.Transform): ITK affine or rigid transform derived from
                itkMatrixOffsetTransformBase (e.g., itk.AffineTransform[itk.D, 3],
                itk.Rigid3DTransform, etc.)

        Returns:
            ANTsTransform: ANTs affine transform object

        Raises:
            ValueError: If transform dimension is not 3D

        Example:
            >>> # Create ITK affine transform
            >>> affine_itk = itk.AffineTransform[itk.D, 3].New()
            >>> affine_itk.SetIdentity()
            >>> # Convert to ANTs
            >>> affine_ANTS = registrar.itk_affine_transform_to_ANTS_transform(affine_itk)
            >>> # Use in ANTs operations
            >>> result = ants.apply_ants_transform(affine_ANTS, moving_image)
        """
        # Get dimension of the transform
        dimension = itk_tfm.GetInputSpaceDimension()
        if dimension != 3:
            raise ValueError(
                f"Only 3D transforms are supported, got dimension: {dimension}"
            )

        # Check if transform has a matrix (Translation transforms don't)
        matrix_itk: NDArray[np.float64]
        if hasattr(itk_tfm, "GetMatrix"):
            # Extract matrix (ITK matrix is row-major)
            matrix_itk = (
                np.asarray(itk_tfm.GetMatrix()).reshape(3, 3).astype(np.float64)
            )
        else:
            # For transforms without matrix (e.g., TranslationTransform), use identity matrix
            matrix_itk = np.eye(3, dtype=np.float64)

        # Extract translation and center based on transform type:
        # - MatrixOffsetTransformBase (Affine, Rigid): use GetTranslation() WITH GetCenter()
        # - TranslationTransform: use GetOffset() WITHOUT GetCenter()
        if hasattr(itk_tfm, "GetTranslation"):
            # MatrixOffsetTransformBase-derived transforms
            translation_itk = np.asarray(itk_tfm.GetTranslation())
            center_itk = np.asarray(itk_tfm.GetCenter())
        elif hasattr(itk_tfm, "GetOffset"):
            # TranslationTransform - use GetOffset() WITHOUT GetCenter()
            translation_itk = np.asarray(itk_tfm.GetOffset())
            center_itk = np.zeros(3, dtype=np.float64)  # No center for translation
        else:
            # Fallback for unknown transform types
            translation_itk = np.zeros(3, dtype=np.float64)
            center_itk = np.zeros(3, dtype=np.float64)

        # ANTs affine transform parameters structure:
        # For 3D: 12 parameters
        # parameters[0:9]: 3x3 matrix in row-major order
        # parameters[9:12]: translation vector
        # fixed_parameters[0:3]: center of rotation

        # Flatten matrix to row-major order for ANTs
        params = np.zeros(12, dtype=np.float64)
        params[0:9] = matrix_itk.flatten()  # Already row-major
        params[9:12] = translation_itk

        # Ensure fixed_params is also float64
        fixed_params = center_itk.astype(np.float64)

        # Create ANTs affine transform
        # Note: dimension must be integer 3, not float
        ants_tfm = ants.new_ants_transform(
            precision="double",
            transform_type="AffineTransform",
            dimension=3,
            parameters=params.tolist(),  # Convert to list to ensure proper type
            fixed_parameters=fixed_params.tolist(),
        )

        return ants_tfm

    def itk_transform_to_ANTSfile(
        self,
        itk_tfm: itk.Transform,
        reference_image: itk.Image,
        output_filename: str,
    ) -> list[str]:
        """Convert ITK transform to ANTs transform file.

        This method converts any ITK transform (Affine, Rigid, DisplacementField, etc.)
        to an ANTs transform file that can be used as initial_transform in
        ants.registration() or ants.label_image_registration().

        The conversion process:
        1. Uses TransformTools to convert the ITK transform to a displacement field
        2. Converts the displacement field image from ITK to ANTs format
        3. Creates an ANTsPy transform object from the displacement field
        4. Writes the ANTs transform to a file

        Args:
            itk_tfm (itk.Transform): Input ITK transform to convert. Can be any
                ITK transform type (AffineTransform, DisplacementFieldTransform,
                CompositeTransform, etc.)
            reference_image (itk.Image): Reference image that defines the spatial
                domain for the displacement field (spacing, size, origin, direction)
            output_filename (str): Path where the ANTs transform file will be written.
                Typically should have .mat extension for ANTs transforms.

        Returns:
            list[str]: List containing the path to the written ANTs transform file

        Example:
            >>> # Convert ITK affine transform to ANTs file
            >>> affine_itk = itk.AffineTransform[itk.D, 3].New()
            >>> affine_itk.SetIdentity()
            >>> transform_files = registrar.itk_transform_to_ANTSfile(
            ...     affine_itk, reference_image, 'initial_transform.mat'
            ... )
            >>>
            >>> # Use in registration
            >>> result = ants.registration(
            ...     fixed=fixed_ANTS, moving=moving_ANTS, initial_transform=transform_files
            ... )
        """
        if isinstance(itk_tfm, itk.DisplacementFieldTransform) or isinstance(
            itk_tfm, itk.CompositeTransform
        ):
            transform_tools = TransformTools()
            disp_field_itk = transform_tools.convert_transform_to_displacement_field(
                tfm=itk_tfm,
                reference_image=reference_image,
                np_component_type=np.float32,  # Use float32 for compatibility with ANTs
                use_reference_image_as_mask=False,
            )

            if "nii.gz" not in output_filename:
                output_filename = os.path.splitext(output_filename)[0] + ".nii.gz"

            # Write displacement field directly as nifti (ANTs can read this)
            itk.imwrite(disp_field_itk, output_filename, compression=True)
            self.log_info("Wrote ANTs displacement field to: %s", output_filename)

            return [output_filename]
        ants_tfm = self.itk_affine_transform_to_ANTS_transform(itk_tfm)
        if ".mat" not in output_filename:
            output_filename = os.path.splitext(output_filename)[0] + ".mat"

        # Write transform to file
        ants.write_transform(ants_tfm, output_filename)
        self.log_info("Wrote ANTs transform to: %s", output_filename)

        return [output_filename]

    def _antsfiles_to_itk_transforms(
        self,
        ants_transforms: list[str],
        reference_image: itk.Image,
        inverse: bool = False,
    ) -> itk.Transform:
        phi = itk.CompositeTransform[itk.D, 3].New()
        for ants_tfm_filename in ants_transforms:
            tfm = ants.read_transform(ants_tfm_filename)
            if tfm.transform_type == "AffineTransform":
                affine_tfm_itk = self._antsfile_to_itk_affine_transform(
                    ants_tfm_filename
                )
                if inverse:
                    affine_tfm_itk = affine_tfm_itk.GetInverseTransform()
                phi.AddTransform(affine_tfm_itk)
            elif tfm.transform_type == "DisplacementFieldTransform":
                disp_tfm_itk = self._antsfile_to_itk_displacement_field_transform(
                    ants_tfm_filename, reference_image
                )
                phi.AddTransform(disp_tfm_itk)
            else:
                raise ValueError(
                    f"Unsupported ANTs transform type: {tfm.transform_type}"
                )

        return phi

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Register moving image to fixed image using ANTs registration algorithm.

        Implementation of the abstract register() method from RegisterImagesBase.
        Performs deformable registration to align the moving image with the
        fixed image using ANTs SyN or other specified algorithms.

        Args:
            moving_image (itk.image): The 3D image to be registered/aligned.
            moving_mask (itk.image, optional): Binary mask defining the
                region of interest in the moving image
            moving_image_pre (itk.Image, optional): Pre-processed moving image.
             If None, preprocessing is performed automatically

        Returns:
            dict: Dictionary containing:
                - "fixed_to_moving_transform": Warps the moving image onto the fixed
                  grid (warping moving points/landmarks into fixed space uses
                  "moving_to_fixed_transform" instead -- image and point warps use
                  opposite transforms; see
                  docs/developer/transform_conventions)
                - "moving_to_fixed_transform": Warps the fixed image onto the moving grid
                - "loss": Loss value from the registration

        Note:
            For SyN registration, the transformations are approximately inverse
            consistent. The forward and inverse transforms are stored separately
            by ANTs.

            To seed the registration with a known alignment, use
            RegisterImagesBase.register_from(), which handles the pre-warp and
            the composition.

        Implementation details:
            - Uses ANTs registration with configurable transform types
            - Supports multi-resolution optimization
            - Handles masked and unmasked registration
            - Returns ITK-compatible displacement field transforms

        Example:
            >>> # Basic registration
            >>> result = registrar.register(moving_image)
            >>> moving_to_fixed_transform = result['moving_to_fixed_transform']
            >>> fixed_to_moving_transform = result['fixed_to_moving_transform']
            >>>
            >>> # Masked registration for cardiac structures
            >>> registrar.set_fixed_mask(heart_mask_fixed)
            >>> result = registrar.register(moving_image, moving_mask=heart_mask_moving)
            >>>
            >>> # Registration seeded with a known alignment
            >>> initial_tfm = itk.AffineTransform[itk.D, 3].New()
            >>> result = registrar.register_from(initial_tfm, moving_image)
        """
        if moving_image is not None:
            self.moving_image = moving_image

        if moving_image_pre is not None:
            self.moving_image_pre = moving_image_pre
        elif self.moving_image is not None:
            self.moving_image_pre = self.preprocess(self.moving_image, self.modality)

        if moving_mask is not None:
            self.moving_mask = moving_mask

        if self.fixed_image_pre is None:
            self.fixed_image_pre = self.preprocess(self.fixed_image, self.modality)

        transform_type = None
        if self.transform_type == "Deformable":
            transform_type = "antsRegistrationSyNQuick[so]"
        elif self.transform_type == "Affine":
            transform_type = "Affine"
        elif self.transform_type == "Rigid":
            transform_type = "Rigid"
        else:
            self.log_error("Invalid transform type: %s", self.transform_type)
            raise ValueError(f"Invalid transform type: {self.transform_type}")

        # Determine the appropriate metric based on transform type and user-specified metric
        aff_metric = None
        syn_metric = None

        if self.transform_type in ["Affine", "Rigid"]:
            # For Affine/Rigid transforms, set aff_metric
            if self.metric == "CC":
                aff_metric = "GC"
            elif self.metric == "Mattes":
                aff_metric = "mattes"
            elif self.metric == "MeanSquares":
                aff_metric = "meansquares"
        elif self.transform_type == "Deformable":
            # For Deformable transforms, set syn_metric
            if self.metric == "CC":
                syn_metric = "CC"
            elif self.metric == "Mattes":
                syn_metric = "mattes"
            elif self.metric == "MeanSquares":
                syn_metric = "meansquares"

        # antsRegistration --dimensionality 3 --float 0 \
        # --output [$thisfolder/pennTemplate_to_${sub}_,$thisfolder/pennTemplate_to_${sub}_Warped.nii.gz] \
        # --interpolation Linear \
        # --winsorize-image-intensities [0.005,0.995] \
        # --use-histogram-matching 0 \
        # --initial-moving-transform [$t1brain,$template,1] \
        # --transform Rigid[0.1] \
        # --metric MI[$t1brain,$template,1,32,Regular,0.25] \
        # --convergence [1000x500x250x100,1e-6,10] \
        # --shrink-factors 8x4x2x1 \
        # --smoothing-sigmas 3x2x1x0vox \
        # --transform Affine[0.1] \
        # --metric MI[$t1brain,$template,1,32,Regular,0.25] \
        # --convergence [1000x500x250x100,1e-6,10] \
        # --shrink-factors 8x4x2x1 \
        # --smoothing-sigmas 3x2x1x0vox \
        # --transform SyN[0.1,3,0] \
        # --metric CC[$t1brain,$template,1,4] \
        # --convergence [100x70x50x20,1e-6,10] \
        # --shrink-factors 8x4x2x1 \
        # --smoothing-sigmas 3x2x1x0vox \
        # -x $brainlesionmask

        if self.fixed_mask is not None and self.moving_mask is not None:
            # mask_all_stages=True re-applies the mask at every pyramid
            # level and is significantly more expensive than masking only
            # the final stage. fast_mode trades that extra precision for
            # speed (e.g. in automated tests).
            registration_result = ants.registration(
                fixed=self._itk_to_ants_image(self.fixed_image_pre),
                mask=self._itk_to_ants_image(self.fixed_mask),
                moving=self._itk_to_ants_image(self.moving_image_pre),
                moving_mask=self._itk_to_ants_image(self.moving_mask),
                initial_transform=["identity"],
                type_of_transform=transform_type,
                aff_metric=aff_metric,
                syn_metric=syn_metric,
                use_histogram_matching=False,
                mask_all_stages=not self.fast_mode,
                verbose=False,
                reg_iterations=self.number_of_iterations,
            )
        else:
            registration_result = ants.registration(
                fixed=self._itk_to_ants_image(self.fixed_image_pre),
                moving=self._itk_to_ants_image(self.moving_image_pre),
                initial_transform=["identity"],
                type_of_transform=transform_type,
                aff_metric=aff_metric,
                syn_metric=syn_metric,
                use_histogram_matching=False,
                verbose=False,
                reg_iterations=self.number_of_iterations,
            )

        # Convert ANTs transforms to ITK
        forward_reg = self._antsfiles_to_itk_transforms(
            registration_result["fwdtransforms"],
            inverse=False,
            reference_image=self.fixed_image,
        )
        inverse_reg = self._antsfiles_to_itk_transforms(
            registration_result["invtransforms"],
            inverse=True,
            reference_image=self.moving_image,
        )

        fixed_to_moving_transform = forward_reg
        moving_to_fixed_transform = inverse_reg
        moving_image_reg = registration_result["warpedmovout"]
        loss = ants.image_similarity(
            self._itk_to_ants_image(self.fixed_image),
            moving_image_reg,
        )

        return {
            "fixed_to_moving_transform": fixed_to_moving_transform,
            "moving_to_fixed_transform": moving_to_fixed_transform,
            "loss": loss,
        }
