"""
Tools for transforming and manipulating ITK transforms.

This module provides the TransformTools class with utilities for working with
ITK transforms, including transforming images and contours, generating
deformation fields, interpolating between transforms, and correcting spatial
folding artifacts.

The tools support various transform operations needed for medical image
analysis, particularly in the context of 4D cardiac imaging where transforms
are used to track anatomical motion over time.
"""

import logging
from typing import Optional, Type, Union, cast

import itk
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk

from .image_tools import ImageTools
from .monai_physio_base import MONAIPhysioBase


class TransformTools(MONAIPhysioBase):
    """
    Utilities for transforming and manipulating ITK transforms.

    This class provides a comprehensive set of tools for working with ITK
    transforms in medical image analysis. It supports transforming various
    data types (images, contours), generating visualization aids, and
    performing advanced operations like transform interpolation and spatial
    folding correction.

    The class is particularly useful for 4D cardiac imaging workflows where
    transforms are used to track anatomical motion over time, requiring
    operations like transform chaining, interpolation, and quality control.

    Key capabilities:
    - Transform PyVista contours and ITK images
    - Generate deformation fields from transforms
    - Interpolate between transforms temporally
    - Smooth transforms to reduce noise
    - Combine transforms with spatial masks
    - Detect and correct spatial folding
    - Generate visualization grids

    Example:
        >>> transform_tools = TransformTools()
        >>> # Transform a contour mesh
        >>> transformed_contour = transform_tools.transform_pvcontour(
        ...     contour, transform, with_deformation_magnitude=True
        ... )
        >>> # Generate deformation field
        >>> field = transform_tools.generate_field(transform, reference_image)
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the TransformTools class.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    def combine_displacement_field_transforms(
        self,
        tfm1: itk.Transform,
        tfm2: itk.Transform,
        reference_image: itk.Image,
        tfm1_weight: float = 1.0,
        tfm2_weight: float = 1.0,
        mode: str = "compose",
        tfm1_blur_sigma: float = 0.0,
        tfm2_blur_sigma: float = 0.0,
    ) -> itk.Transform:
        """
        Compose two displacement field transforms.

        In ``add`` mode, returns a single displacement field transform with
        weighted summed vectors. In ``compose`` mode, returns a composite
        transform containing both weighted displacement field transforms.

        ``compose`` follows ITK's CompositeTransform convention, where the
        last-added transform is applied first: the result evaluates
        ``tfm1(tfm2(x))``, so ``tfm2`` is the stage that runs first.

        In ``compose`` mode at unit weights with no blurring there is nothing to
        apply to the fields, so the inputs are chained as they are and no field
        is rasterized at all.  See the comment on that branch below for what
        that saves.
        """
        assert mode in ["add", "compose"], "Invalid mode"

        if (
            mode == "compose"
            and tfm1_weight == 1.0
            and tfm2_weight == 1.0
            and tfm1_blur_sigma == 0.0
            and tfm2_blur_sigma == 0.0
        ):
            # A CompositeTransform chains its members lazily, so rasterizing
            # them first only reproduces what it would compute anyway -- less
            # accurately, because sampling onto *reference_image* and
            # interpolating back introduces error that evaluating the originals
            # does not.
            #
            # It is also what dominates memory here.  The distance-map caller
            # composes on a grid padded to 2.5 * mask_dilation_mm per side; on
            # the lung chest CT that is 758 x 758 x 664, where one
            # ``Vector<double, 3>`` field is 9.2 GB.  Rasterizing both sides of
            # both directions retained 36.6 GB and peaked far above that, to
            # hold an affine and a 175-cubed field that together are under
            # 130 MB.
            #
            # Weighting or blurring a transform genuinely needs its field, so
            # those keep the path below.
            passthrough_tfm = itk.CompositeTransform[itk.D, 3].New()
            passthrough_tfm.AddTransform(tfm1)
            passthrough_tfm.AddTransform(tfm2)
            return passthrough_tfm

        dtfm1 = self.convert_transform_to_displacement_field_transform(
            tfm1, reference_image
        )
        dtfm2 = self.convert_transform_to_displacement_field_transform(
            tfm2, reference_image
        )
        dfield1 = dtfm1.GetDisplacementField()
        dfield2 = dtfm2.GetDisplacementField()
        dfield1_arr = itk.array_from_image(dfield1)
        dfield2_arr = itk.array_from_image(dfield2)
        if tfm1_blur_sigma > 0.0:
            for dim in range(dfield1.GetNumberOfComponentsPerPixel()):
                tmp_field = dfield1_arr[:, :, :, dim]
                tmp_image = itk.image_from_array(tmp_field)
                tmp_image.CopyInformation(dfield1)
                tmp_image = itk.smoothing_recursive_gaussian_image_filter(
                    tmp_image, Sigma=tfm1_blur_sigma
                )
                tmp_field = itk.array_from_image(tmp_image)
                dfield1_arr[:, :, :, dim] = tmp_field
        if tfm2_blur_sigma > 0.0:
            for dim in range(dfield2.GetNumberOfComponentsPerPixel()):
                tmp_field = dfield2_arr[:, :, :, dim]
                tmp_image = itk.image_from_array(tmp_field)
                tmp_image.CopyInformation(dfield2)
                tmp_image = itk.smoothing_recursive_gaussian_image_filter(
                    tmp_image, Sigma=tfm2_blur_sigma
                )
                tmp_field = itk.array_from_image(tmp_image)
                dfield2_arr[:, :, :, dim] = tmp_field
        if mode == "add":
            dfield_composed_arr = tfm1_weight * dfield1_arr + tfm2_weight * dfield2_arr
            image_tools = ImageTools()
            dfield_composed = image_tools.convert_array_to_image_of_vectors(
                dfield_composed_arr,
                ptype=itk.D,
                reference_image=dfield1,
            )
            new_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
            new_tfm.SetDisplacementField(dfield_composed)
            return new_tfm
        # compose
        image_tools = ImageTools()

        dfield1_arr = tfm1_weight * dfield1_arr
        dfield2_arr = tfm2_weight * dfield2_arr
        new_dfield1 = image_tools.convert_array_to_image_of_vectors(
            dfield1_arr,
            ptype=itk.D,
            reference_image=dfield1,
        )
        new_dfield2 = image_tools.convert_array_to_image_of_vectors(
            dfield2_arr,
            ptype=itk.D,
            reference_image=dfield2,
        )
        new_tfm1 = itk.DisplacementFieldTransform[itk.D, 3].New()
        new_tfm1.SetDisplacementField(new_dfield1)
        new_tfm2 = itk.DisplacementFieldTransform[itk.D, 3].New()
        new_tfm2.SetDisplacementField(new_dfield2)
        composite_tfm = itk.CompositeTransform[itk.D, 3].New()
        composite_tfm.AddTransform(new_tfm1)
        composite_tfm.AddTransform(new_tfm2)
        return composite_tfm

    def convert_transform_to_displacement_field(
        self,
        tfm: itk.Transform,
        reference_image: itk.image,
        np_component_type: type[np.float32] | type[np.float64] = np.float64,
        use_reference_image_as_mask: bool = False,
    ) -> itk.image:
        """
        Generate a dense deformation field from an ITK transform.

        Converts any ITK transform into a dense displacement field that
        explicitly stores the displacement vector at each voxel. This is
        useful for visualization, analysis, and storage of transforms.

        Args:
            tfm (itk.Transform): Input transform to convert. Can be any ITK
                transform type (Affine, BSpline, DisplacementField, etc.)
            reference_image (itk.image): Defines the spatial grid for the
                output deformation field (spacing, size, origin, direction)
            use_reference_image_as_mask (bool): If True, applies the reference
                image as a mask to zero out displacement vectors outside
                the image domain

        Returns:
            itk.image: Vector image where each voxel contains a displacement
                vector [dx, dy, dz] in physical coordinates

        Example:
            >>> # Generate deformation field for visualization
            >>> field = transform_tools.generate_field(registration_transform,
                reference_ct)
            >>> # Use as mask to limit field to anatomical regions
            >>> masked_field = transform_tools.generate_field(
            ...     transform, reference_ct, use_reference_image_as_mask=True
            ... )
        """
        # Handle case where tfm is a list (e.g., from itk.transformread)
        if isinstance(tfm, (list, tuple)):
            if len(tfm) == 1:
                tfm = tfm[0]
            else:
                raise ValueError(
                    f"Expected single transform, got list with {len(tfm)} transforms"
                )

        TfmPrecision = itk.template(tfm)[1][0]

        # Create and configure filter
        field = None
        if "DisplacementFieldTransform" not in str(type(tfm)):
            field_filter = itk.TransformToDisplacementFieldFilter[
                itk.Image[itk.Vector[itk.F, 3], 3], TfmPrecision
            ].New()
            field_filter.SetTransform(tfm)
            field_filter.SetReferenceImage(reference_image)
            field_filter.SetUseReferenceImage(True)
            field_filter.Update()
            field = field_filter.GetOutput()
        else:
            field = tfm.GetDisplacementField()
            field_arr = itk.array_view_from_image(tfm.GetDisplacementField())
            reference_image_arr = itk.array_view_from_image(reference_image)
            if field_arr.shape[:3] != reference_image_arr.shape:
                field_filter = itk.TransformToDisplacementFieldFilter[
                    itk.Image[itk.Vector[itk.F, 3], 3], TfmPrecision
                ].New()
                field_filter.SetTransform(tfm)
                field_filter.SetReferenceImage(reference_image)
                field_filter.SetUseReferenceImage(True)
                field_filter.Update()
                field = field_filter.GetOutput()

        field_arr = itk.array_from_image(field)
        field_arr = field_arr.astype(np_component_type)

        image_tools = ImageTools()
        field = image_tools.convert_array_to_image_of_vectors(
            field_arr,
            ptype=np_component_type,
            reference_image=reference_image,
        )

        if use_reference_image_as_mask:
            mask = reference_image
            field = itk.MaskImageFilter(field, mask)

        return field

    def convert_transform_to_displacement_field_transform(
        self, tfm: itk.Transform, reference_image: itk.Image
    ) -> itk.DisplacementFieldTransform:
        """
        Convert an ITK transform to a displacement field transform.
        """
        # TransformToDisplacementFieldFilter only supports float precision
        # so we need to cast the transform to float and then convert
        # to double since most other transform filters require double precision
        field = self.convert_transform_to_displacement_field(
            tfm, reference_image, np_component_type=np.float64
        )

        new_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
        new_tfm.SetDisplacementField(field)
        return new_tfm

    def invert_displacement_field_transform(
        self,
        tfm: itk.Transform,
        max_iterations: int = 20,
        max_error_tolerance: float = 0.05,
        mean_error_tolerance: float = 0.0005,
    ) -> itk.Transform:
        """
        Invert a displacement field transform.

        Uses SimpleITK's fixed-point iterative inversion on the input field's
        own grid. The defaults are tighter than SimpleITK's own (10 iterations,
        0.1 mm max error) because the fields produced here are not smooth
        everywhere and converge slowly near their support boundary.

        Args:
            tfm: Displacement field transform to invert.
            max_iterations: Fixed-point iterations per voxel.
            max_error_tolerance: Convergence threshold on the maximum error, in
                the field's units.
            mean_error_tolerance: Convergence threshold on the mean error.

        Returns:
            The inverted displacement field transform.
        """
        assert "DisplacementFieldTransform" in str(type(tfm)), (
            "Input transform must be a displacement field transform"
        )
        image_tools = ImageTools()

        field_itk = tfm.GetDisplacementField()

        field_sitk = image_tools.convert_itk_image_to_sitk(field_itk)

        field_sitk_inv = sitk.InvertDisplacementField(
            field_sitk,
            maximumNumberOfIterations=max_iterations,
            maxErrorToleranceThreshold=max_error_tolerance,
            meanErrorToleranceThreshold=mean_error_tolerance,
        )

        field_itk_inv = image_tools.convert_sitk_image_to_itk(field_sitk_inv)

        new_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
        new_tfm.SetDisplacementField(field_itk_inv)

        return new_tfm

    def invert_transform(
        self, tfm: itk.Transform, reference_image: itk.Image
    ) -> itk.Transform:
        """Invert any transform, analytically when the type supports it.

        Prefers ITK's analytic inverse (available for translation, rigid, affine
        and composites of them) and falls back to rasterizing a displacement
        field over ``reference_image`` and inverting that numerically.

        The analytic inverse is preferred because the fallback is only defined on
        ``reference_image``'s grid: outside it the field is zero, so the inverse
        silently degrades to the identity there.

        Args:
            tfm (itk.Transform): Transform to invert.
            reference_image (itk.Image): Grid used by the displacement-field
                fallback.

        Returns:
            itk.Transform: The inverse transform.
        """
        try:
            analytic = tfm.GetInverseTransform()
        except Exception:  # pragma: no cover - transform type dependent
            analytic = None
        if analytic is not None:
            return cast(itk.Transform, analytic)

        return self.invert_displacement_field_transform(
            self.convert_transform_to_displacement_field_transform(tfm, reference_image)
        )

    def transform_pvcontour(
        self,
        contour: pv.PolyData,
        tfm: itk.Transform,
        with_deformation_magnitude: bool = False,
    ) -> pv.PolyData:
        """
        Transform PyVista contour meshes using an ITK transform.

        Applies an ITK transform to all points in a PyVista PolyData mesh,
        useful for deforming anatomical contours according to computed
        registration transforms. Optionally computes deformation magnitude
        at each point.

        Args:
            contour (pv.PolyData): The input contour mesh to transform
            tfm (itk.Transform): ITK transform to apply. Can be a single
                transform or a list/array containing one transform
            with_deformation_magnitude (bool): If True, adds a
                "DeformationMagnitude" point data array containing the
                Euclidean distance each point moved

        Returns:
            pv.PolyData: The transformed contour mesh with updated point
                coordinates and optionally deformation magnitude data

        Example:
            >>> # Transform cardiac contour with deformation tracking
            >>> transformed_heart = transform_tools.transform_pvcontour(
            ...     heart_contour, cardiac_transform, with_deformation_magnitude=True
            ... )
            >>> # Access deformation magnitudes
            >>> deformation = transformed_heart['DeformationMagnitude']
        """

        return cast(
            pv.PolyData,
            self.transform_dataset(
                contour,
                tfm,
                with_deformation_magnitude=with_deformation_magnitude,
            ),
        )

    def transform_dataset(
        self,
        mesh: pv.DataSet,
        tfm: itk.Transform,
        with_deformation_magnitude: bool = False,
    ) -> pv.DataSet:
        """Transform a PyVista dataset while preserving mesh topology and data arrays.

        Applies an ITK point transform to every point in the input dataset and
        returns a deep copy with the original cells, cell data, and point data
        preserved. This is appropriate for non-contour datasets such as
        UnstructuredGrid inputs where casting to PolyData would lose topology.
        """

        new_mesh = mesh.copy(deep=True)
        pnts = np.array(new_mesh.points, dtype=float)

        # Handle case where tfm is a list (e.g., from itk.transformread)
        if isinstance(tfm, (list, tuple)):
            if len(tfm) == 1:
                tfm = tfm[0]
            else:
                raise ValueError(
                    f"Expected single transform, got list with {len(tfm)} transforms"
                )

        pnts = np.array(pnts)
        new_pnts = [
            np.array(tfm.TransformPoint((float(p[0]), float(p[1]), float(p[2]))))
            for p in pnts
        ]
        new_mesh.points = np.asarray(new_pnts, dtype=float).reshape(-1, 3)

        if with_deformation_magnitude:
            try:
                import cupy as cp  # noqa: PLC0415
            except (ImportError, OSError):
                cp = None
            if cp is not None:
                try:
                    import cupy_backends.cuda.api.runtime as _cuda_rt  # noqa: PLC0415

                    _CUDARuntimeError: Type[BaseException] = _cuda_rt.CUDARuntimeError
                except ImportError:
                    _CUDARuntimeError = OSError
                try:
                    new_pnts_cp = cp.array(new_pnts)
                    pnts_cp = cp.array(pnts)
                    new_mesh.point_data["DeformationMagnitude"] = cp.linalg.norm(
                        new_pnts_cp - pnts_cp, axis=1
                    ).get()
                except (OSError, _CUDARuntimeError):
                    cp = None
            if cp is None:
                new_mesh.point_data["DeformationMagnitude"] = np.linalg.norm(
                    np.asarray(new_pnts) - np.asarray(pnts), axis=1
                )

        return new_mesh

    def transform_image(
        self,
        img: itk.image,
        tfm: itk.Transform,
        reference_image: itk.image,
        interpolation_method: str = "linear",
        background_value: float = 0.0,
    ) -> itk.image:
        """
        Transform an ITK image using a specified transform and interpolation.

        Resamples an image according to a geometric transform, using the
        reference image to define the output grid properties. Different
        interpolation methods are available depending on data type and
        quality requirements.

        Args:
            img (itk.image): The input image to transform
            tfm (itk.Transform): The ITK transform to apply
            reference_image (itk.image): Defines output spacing, size, origin,
                and direction for the transformed image
            tfm_type (str): Interpolation method. Options:
                - "linear": Linear interpolation (default, good for CT/MR)
                - "nearest": Nearest neighbor (preserves discrete values)
                - "sinc": Sinc interpolation (highest quality, slower)
            background_value (float): Value written where the reference grid
                samples outside the input image. Default 0.0, which is right for
                labelmaps and masks; intensity images need the value that means
                "no tissue" in their own units -- for CT that is -1000 HU (air),
                not 0 HU (water).

        Returns:
            itk.image: The transformed image resampled to reference grid

        Raises:
            ValueError: If tfm_type is not one of the supported options

        Example:
            >>> # Transform CT image with linear interpolation
            >>> warped_ct = transform_tools.transform_image(
            ...     ct_image, deformation_transform, reference_ct
            ... )
            >>> # Transform label map preserving discrete values
            >>> warped_labels = transform_tools.transform_image(
            ...     labelmap, transform, reference, interpolation_method='nearest'
            ... )
        """
        # Handle case where tfm is a list (e.g., from itk.transformread)
        if isinstance(tfm, (list, tuple)):
            if len(tfm) == 1:
                tfm = tfm[0]
            else:
                raise ValueError(
                    "Expected single transform or list with one transform, got list"
                    f"with {len(tfm)} transforms"
                )

        interpolator = None
        if interpolation_method == "linear":
            interpolator = itk.LinearInterpolateImageFunction.New(img)
        elif interpolation_method == "nearest":
            interpolator = itk.NearestNeighborInterpolateImageFunction.New(img)
        elif interpolation_method == "sinc":
            interpolator = itk.WindowedSincInterpolateImageFunction.New(img)
        else:
            raise ValueError(f"Invalid transform type: {interpolation_method}")

        # This shouldn't be needed, but for certain itk.CompositeTransform types,
        # the resample_image_filter will silently fail and apply the identity
        # transform instead of the one passed.
        dftfm = self.convert_transform_to_displacement_field_transform(
            tfm, reference_image
        )

        # ITK's wrapping types DefaultPixelValue to the image's pixel type, and
        # rejects a Python float for a discrete image.
        dtype = itk.GetArrayViewFromImage(img).dtype
        default_pixel_value: Union[int, float]
        if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
            default_pixel_value = int(round(background_value))
            low, high = (
                (0, 1)
                if np.issubdtype(dtype, np.bool_)
                else (int(np.iinfo(dtype).min), int(np.iinfo(dtype).max))
            )
            if not low <= default_pixel_value <= high:
                raise ValueError(
                    f"background_value {background_value} is outside the range "
                    f"[{low}, {high}] of the image's {dtype} pixel type"
                )
        else:
            default_pixel_value = float(background_value)

        img_reg = itk.resample_image_filter(
            Input=img,
            Transform=dftfm,
            Interpolator=interpolator,
            ReferenceImage=reference_image,
            UseReferenceImage=True,
            DefaultPixelValue=default_pixel_value,
        )
        return img_reg

    def convert_vtk_matrix_to_itk_transform(
        self, vtk_mat: vtk.vtkMatrix4x4
    ) -> itk.Transform:
        """
        Convert a VTK matrix to an ITK transform.

        Converts a VTK matrix object into an equivalent ITK transform.
        This is useful for interoperability between VTK-based processing
        (e.g., mesh manipulation) and ITK-based image processing and
        registration.

        Args:
            vtk_mat (itk.vtkMatrix): The input VTK transform to convert
        Returns:
            itk.Transform: The equivalent ITK transform

        Example:
            >>> # Convert VTK transform from mesh processing
            >>> itk_transform = transform_tools.get_itk_transform_from_vtk_transform
                vtk_transform)
        """
        mat = np.eye(3).astype(np.float64)
        vec = itk.Vector[itk.D, 3]()
        for i in range(3):
            vec[i] = vtk_mat.GetElement(i, 3)
            for j in range(3):
                mat[i, j] = vtk_mat.GetElement(i, j)
        itkmat = itk.Matrix[itk.D, 3, 3](itk.GetVnlMatrixFromArray(mat))
        itk_tfm = itk.AffineTransform[itk.D, 3].New()
        itk_tfm.SetIdentity()
        itk_tfm.SetMatrix(itkmat)
        itk_tfm.SetOffset(vec)

        return itk_tfm

    def smooth_transform(
        self, tfm: itk.Transform, sigma: float, reference_image: itk.image
    ) -> itk.Transform:
        """
        Smooth a transform using Gaussian filtering to reduce noise.

        Applies Gaussian smoothing to the displacement field representation
        of a transform to reduce noise and create more regularized
        deformations. This is useful for improving transform quality and
        reducing artifacts.

        Args:
            tfm (itk.Transform): Input transform to smooth
            sigma (float): Standard deviation of Gaussian smoothing kernel
                in physical units (millimeters). Larger values create
                more smoothing
            reference_image (itk.image): Defines spatial grid for field
                generation and smoothing

        Returns:
            itk.Transform: DisplacementFieldTransform with smoothed
                deformation field

        Example:
            >>> # Smooth noisy registration transform
            >>> smooth_transform = transform_tools.smooth_transform(
            ...     noisy_transform, sigma=2.0, reference_ct
            ... )
            >>> # Light smoothing for artifact reduction
            >>> refined_transform = transform_tools.smooth_transform(
            ...     transform, sigma=0.5, reference_image
            ... )
        """
        field = self.convert_transform_to_displacement_field(tfm, reference_image)

        field_arr = itk.array_from_image(field)
        for dim in range(field.GetNumberOfComponentsPerPixel()):
            tmp_field_arr = field_arr[:, :, :, dim]
            tmp_image = itk.image_from_array(tmp_field_arr)
            tmp_image.CopyInformation(field)
            tmp_image = itk.smoothing_recursive_gaussian_image_filter(
                tmp_image, Sigma=sigma
            )
            tmp_field_arr = itk.array_from_image(tmp_image)
            field_arr[:, :, :, dim] = tmp_field_arr
        image_tools = ImageTools()
        field = image_tools.convert_array_to_image_of_vectors(
            field_arr,
            ptype=itk.D,
            reference_image=field,
        )

        tfm_smooth = itk.DisplacementFieldTransform[
            itk.D, field.GetImageDimension()
        ].New()
        tfm_smooth.SetDisplacementField(field)

        return tfm_smooth

    def smooth_deformation_field_transform(
        self,
        field: itk.Image,
        sigma: float,
        weight_image: Optional[itk.Image] = None,
        normal_image: Optional[itk.Image] = None,
        interior_mask: Optional[itk.Image] = None,
        exterior_sigma: Optional[float] = None,
    ) -> itk.DisplacementFieldTransform:
        """Spread a sparsely sampled deformation field into a continuous one.

        ``field`` is treated as a weighted set of displacement *samples* rather
        than as an image: the weighted samples and their weights are each
        Gaussian-smoothed by ``sigma`` (physical millimeters) and then divided,
        which is a Gaussian-weighted average of the nearby samples. A thin
        surface shell therefore becomes a continuous deformation that keeps the
        displacement magnitude the samples carried, instead of being diluted by
        the empty voxels a plain blur would average in. Far from every sample
        the smoothed weight vanishes and the field decays to zero.

        That spread is otherwise isotropic, and carries the whole displacement
        vector outward. Giving ``normal_image`` and ``interior_mask`` splits each
        sample into the component along the surface normal, which expansion and
        contraction live in, and the tangential remainder, which sliding lives
        in, and spreads only the normal component outside the mask. Tissue
        beyond an organ is then pushed and pulled by it without being dragged
        along it, which is how a slip interface such as the pleura or the
        pericardium behaves. Inside the mask the full vector is spread, so the
        organ's own contents still follow its surface. ``exterior_sigma`` sets
        how far that outward push and pull carries, independently of the sigma
        filling the organ itself.
        ``exterior_normal_scale`` sets how much of that normal component the
        surrounding tissue actually receives.

        Args:
            field (itk.Image): Input vector deformation field, sampled where
                ``weight_image`` is non-zero.
            sigma (float): Standard deviation of the Gaussian smoothing kernel
                in physical units (millimeters).
            weight_image (Optional[itk.Image]): Per-voxel sample weight, such as
                the vertex count
                :meth:`WorkflowInferMovement.create_deformation_field` returns.
                Omit to weight every voxel holding a non-zero displacement
                equally, which cannot tell an empty voxel from a genuinely
                zero-displacement one.
            normal_image (Optional[itk.Image]): Per-voxel unit surface normal on
                ``field``'s grid, as
                :meth:`WorkflowInferMovement.create_deformation_field` returns
                alongside the field. Samples whose normal is zero are spread
                whole, having no direction to project onto.
            interior_mask (Optional[itk.Image]): Scalar image on ``field``'s
                grid, 1 where the full displacement should be spread and 0 where
                only its normal component should be. Soften its edge to set the
                width of the band the tangential motion dies out over; a binary
                mask makes the boundary a discontinuity.
            exterior_sigma (Optional[float]): Smoothing sigma (millimeters) for
                the normal component spread outside ``interior_mask``, in place
                of ``sigma``. This is how far the organ reaches into the tissue
                around it: a smaller value confines its push and pull to a
                narrower shell without weakening the displacement at the
                surface, and without touching the spread inside the mask.
                Defaults to ``sigma``. Ignored when no mask is given.

        Returns:
            itk.DisplacementFieldTransform: Smoothed field transform.

        Raises:
            ValueError: If only one of ``normal_image`` and ``interior_mask`` is
                given, if either does not lie on ``field``'s grid, or if the
                field holds no non-zero samples to spread.
        """
        if (normal_image is None) != (interior_mask is None):
            raise ValueError(
                "normal_image and interior_mask must be given together: the "
                "normals say what to project onto, the mask says where to."
            )

        field_arr = itk.array_from_image(field).astype(np.float64)
        if weight_image is not None:
            weights = itk.array_from_image(weight_image).astype(np.float64)
        else:
            weights = (np.linalg.norm(field_arr, axis=3) > 0.0).astype(np.float64)

        # Outside the mask only the normal component of each sample is spread,
        # and it may be spread by a sigma of its own. Each set therefore carries
        # the sigma that both its samples and the weights normalizing them are
        # smoothed by, so a narrower exterior spread stays normalized against
        # the weight that reached the same distance.
        sample_sets = [(field_arr, sigma)]
        mask: Optional[np.ndarray] = None
        if normal_image is not None and interior_mask is not None:
            normals = itk.array_from_image(normal_image).astype(np.float64)
            mask = itk.array_from_image(interior_mask).astype(np.float64)
            if normals.shape != field_arr.shape or mask.shape != field_arr.shape[:3]:
                raise ValueError(
                    f"normal_image {normals.shape} and interior_mask "
                    f"{mask.shape} must lie on the field's grid "
                    f"{field_arr.shape}."
                )
            projected = (field_arr * normals).sum(axis=3, keepdims=True) * normals
            # A vertex interior to a volumetric template carries a zero normal.
            # Projecting it would delete a sample the weights still count in the
            # denominator, biasing the result toward zero rather than leaving the
            # sample unprojected, so those keep their full displacement.
            unoriented = np.linalg.norm(normals, axis=3) == 0.0
            projected[unoriented] = field_arr[unoriented]
            sample_sets.append(
                (projected, sigma if exterior_sigma is None else exterior_sigma)
            )

        smoothed_sets: list[np.ndarray] = []
        for samples, set_sigma in sample_sets:
            spread = np.zeros_like(field_arr)
            for dim in range(field_arr.shape[3]):
                spread[:, :, :, dim] = self._smooth_scalar_array(
                    samples[:, :, :, dim] * weights, set_sigma, field
                )
            smoothed_weights = self._smooth_scalar_array(weights, set_sigma, field)

            # Add a floor to the denominator rather than clamping to it. ITK's
            # recursive Gaussian is an IIR approximation, so far from every
            # sample both smoothed arrays ring around zero; clamping a
            # denominator that small turns that ringing into displacements
            # several times larger than any the samples carried, while adding to
            # it lets the quotient fall off to zero there, which is what a field
            # with no nearby sample should do.
            weight_floor = 1.0e-3 * float(smoothed_weights.max())
            if weight_floor <= 0.0:
                raise ValueError("Deformation field has no non-zero samples to spread.")
            spread /= (np.maximum(smoothed_weights, 0.0) + weight_floor)[..., None]
            smoothed_sets.append(spread)

        smoothed = smoothed_sets[0]
        if mask is not None:
            inside = np.clip(mask, 0.0, 1.0)[..., None]
            smoothed = inside * smoothed_sets[0] + (1.0 - inside) * smoothed_sets[1]

        smoothed_field = ImageTools().convert_array_to_image_of_vectors(
            smoothed, reference_image=field, ptype=itk.D
        )
        field_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
        field_transform.SetDisplacementField(smoothed_field)
        return field_transform

    @staticmethod
    def _smooth_scalar_array(
        array: np.ndarray, sigma: float, reference_image: itk.Image
    ) -> np.ndarray:
        """Gaussian-smooth a scalar array on ``reference_image``'s grid.

        The array is wrapped with the reference geometry before filtering, so
        ``sigma`` is in millimeters rather than in voxels.
        """
        image = itk.image_from_array(np.ascontiguousarray(array))
        image.CopyInformation(reference_image)
        smoothed: np.ndarray = itk.array_from_image(
            itk.smoothing_recursive_gaussian_image_filter(image, Sigma=sigma)
        )
        return smoothed

    def combine_transforms_with_masks(
        self,
        transform1: itk.Transform,
        transform2: itk.Transform,
        mask1: itk.Image,
        mask2: itk.Image,
        reference_image: itk.Image,
        max_iter: int = 10,
        jacobian_threshold: float = 0.1,
    ) -> itk.Transform:
        """
        Combine two transforms using spatial masks with folding correction.

        Merges two transforms by weighting their displacement fields according
        to provided masks, then iteratively corrects any spatial folding
        (negative Jacobian determinant) that may result from the combination.

        This is useful for combining transforms computed for different
        anatomical regions (e.g., separate heart and lung registration)
        into a single coherent transform.

        Args:
            transform1 (itk.Transform): First transform to combine
            transform2 (itk.Transform): Second transform to combine
            mask1 (itk.Image): Float mask defining spatial influence of
                transform1 (0.0 = no influence, 1.0 = full influence)
            mask2 (itk.Image): Float mask defining spatial influence of
                transform2
            reference_image (itk.Image): Defines output grid properties
            max_iter (int): Maximum iterations for folding correction
            jacobian_threshold (float): Jacobian determinant threshold below
                which folding is detected and corrected

        Returns:
            itk.Transform: DisplacementFieldTransform with combined and
                corrected transformation

        Example:
            >>> # Combine heart and lung transforms
            >>> combined_transform = transform_tools.combine_transforms_with_masks(
            ...     heart_transform, lung_transform, heart_mask, lung_mask, reference_ct
            ... )
        """
        # Generate displacement fields
        field1 = self.convert_transform_to_displacement_field(
            transform1, reference_image
        )
        field2 = self.convert_transform_to_displacement_field(
            transform2, reference_image
        )

        # Weight fields by masks
        mask1_arr = itk.array_from_image(mask1)
        mask2_arr = itk.array_from_image(mask2)

        field1_arr = itk.array_from_image(field1)
        field2_arr = itk.array_from_image(field2)

        # Expand mask dimensions to match vector field (add dimension for vector
        #     components)
        mask1_arr = mask1_arr[..., np.newaxis]
        mask2_arr = mask2_arr[..., np.newaxis]

        sum_fields_arr = mask1_arr * field1_arr + mask2_arr * field2_arr

        denom = mask1_arr + mask2_arr
        denom[denom == 0] = 1.0

        combined_field_arr = sum_fields_arr / denom

        # Copy array data to ITK image
        combined_field = ImageTools().convert_array_to_image_of_vectors(
            combined_field_arr, field1, itk.F
        )

        # Correct spatial folding iteratively
        for _ in range(max_iter):
            jacobian_det = self.compute_jacobian_determinant_from_field(combined_field)
            if not self.detect_folding_in_field(
                jacobian_det, threshold=jacobian_threshold
            ):
                break
            combined_field = self.reduce_folding_in_field(combined_field, jacobian_det)

        # Get dimension and create transform with correct types
        Dimension = combined_field.GetImageDimension()
        tfm_combined = itk.DisplacementFieldTransform[itk.F, Dimension].New()
        tfm_combined.SetDisplacementField(combined_field)

        return tfm_combined

    def compute_jacobian_determinant_from_field(self, field: itk.Image) -> itk.Image:
        """Compute Jacobian determinant of a displacement field.

        Calculates the Jacobian determinant at each voxel of a displacement
        field, which indicates local volume change. Values less than 0
        indicate spatial folding, values between 0-1 indicate compression,
        and values greater than 1 indicate expansion.

        Args:
            field (itk.Image): Vector displacement field image

        Returns:
            itk.Image: Scalar image containing Jacobian determinant values

        Example:
            >>> jacobian = transform_tools.compute_jacobian_determinant_from_field(
                    deformation_field
                )
        """
        if "VF" not in str(type(field)):
            field_arr = itk.array_from_image(field)
            field = ImageTools().convert_array_to_image_of_vectors(
                field_arr, field, itk.F
            )
        jac_filter = itk.DisplacementFieldJacobianDeterminantFilter.New(field)
        jac_filter.SetUseImageSpacing(True)
        jac_filter.Update()
        return jac_filter.GetOutput()

    def detect_folding_in_field(
        self, jacobian_det: itk.Image, threshold: float = 0.1
    ) -> bool:
        """Detect spatial folding in a transform.

        Checks for spatial folding by examining the minimum Jacobian
        determinant value. Folding occurs when the Jacobian determinant
        becomes negative or very small, indicating non-invertible regions.

        Args:
            jacobian_det (itk.Image): Jacobian determinant image
            threshold (float): Threshold below which folding is detected

        Returns:
            bool: True if folding is detected, False otherwise

        Example:
            >>> if transform_tools.detect_folding_in_field(jacobian, 0.1):
            ...     print('Spatial folding detected - transform needs correction')
        """
        stats = itk.StatisticsImageFilter.New(jacobian_det)
        stats.Update()
        return float(stats.GetMinimum()) < threshold

    def reduce_folding_in_field(
        self,
        field: itk.Image,
        jacobian_det: itk.Image,
        reduction_factor: float = 0.8,
        threshold: float = 0.1,
    ) -> itk.Image:
        """Reduce folding by scaling displacement field in problematic regions.

        Corrects spatial folding by reducing the magnitude of displacement
        vectors in regions where the Jacobian determinant is below the
        threshold. This is a simple but effective approach to maintaining
        transform invertibility.

        Args:
            field (itk.Image): Input displacement field to correct
            jacobian_det (itk.Image): Jacobian determinant image
            reduction_factor (float): Factor to multiply displacements in
                folding regions (0.8 = 20% reduction)
            threshold (float): Jacobian threshold for identifying folding

        Returns:
            itk.Image: Corrected displacement field with reduced folding

        Example:
            >>> corrected_field = transform_tools.reduce_folding_in_field(
            ...     folded_field, jacobian, reduction_factor=0.7
            ... )
        """
        # Create correction mask
        thresholder = itk.BinaryThresholdImageFilter.New(jacobian_det)
        thresholder.SetLowerThreshold(-1000)
        thresholder.SetUpperThreshold(threshold)
        thresholder.SetInsideValue(reduction_factor)
        thresholder.SetOutsideValue(1.0)
        thresholder.Update()

        thresh_arr = itk.array_from_image(thresholder.GetOutput())
        field_arr = itk.array_from_image(field)
        for i in range(field_arr.shape[3]):
            field_arr[:, :, :, i] *= thresh_arr
        corrected_field = ImageTools().convert_array_to_image_of_vectors(
            field_arr, field, itk.F
        )
        return corrected_field

    def generate_grid_image(
        self, reference_image: itk.image, grid_size: int = 60, line_width: int = 3
    ) -> itk.image:
        """
        Generate a grid image.
        """
        img_arr = itk.array_from_image(reference_image)
        img_arr_max = np.max(img_arr)
        img_shape = list(img_arr.shape)
        grid_spacing = [s / grid_size for s in img_shape]
        if line_width <= 0:
            line_width = 1
        width_min = line_width // 2
        width_max = width_min + line_width
        for i in range(grid_size):
            for j in range(grid_size):
                min_idx0 = max(0, int(i * grid_spacing[0]) - width_min)
                max_idx0 = min(img_arr.shape[0], int(i * grid_spacing[0]) + width_max)
                min_idx1 = max(0, int(j * grid_spacing[1]) - width_min)
                max_idx1 = min(img_arr.shape[1], int(j * grid_spacing[1]) + width_max)
                if min_idx0 < max_idx0 and min_idx1 < max_idx1:
                    img_arr[min_idx0:max_idx0, min_idx1:max_idx1, :] = img_arr_max

                min_idx2 = max(0, int(j * grid_spacing[2]) - width_min)
                max_idx2 = min(img_arr.shape[2], int(j * grid_spacing[2]) + width_max)
                if min_idx0 < max_idx0 and min_idx2 < max_idx2:
                    img_arr[min_idx0:max_idx0, :, min_idx2:max_idx2] = img_arr_max

                min_idx1 = max(0, int(i * grid_spacing[1]) - width_min)
                max_idx1 = min(img_arr.shape[1], int(i * grid_spacing[1]) + width_max)
                if min_idx1 < max_idx1 and min_idx2 < max_idx2:
                    img_arr[:, min_idx1:max_idx1, min_idx2:max_idx2] = img_arr_max

        grid_image = itk.image_from_array(img_arr)
        grid_image.CopyInformation(reference_image)

        return grid_image

    def convert_field_to_grid_visualization(
        self,
        tfm: itk.Transform,
        reference_image: itk.image,
        grid_size: int = 60,
        line_width: int = 3,
    ) -> itk.image:
        """
        Generate a visual deformation grid for transform visualization.

        Creates a regular grid pattern in the reference image space, then
        applies the transform to visualize the deformation. The resulting
        warped grid shows how the transform deforms space and can reveal
        areas of compression, expansion, or folding.

        Args:
            tfm (itk.Transform): Transform to visualize
            reference_image (itk.image): Defines spatial domain and grid properties
            grid_size (int): Number of grid lines in each dimension

        Returns:
            itk.image: Binary image containing the transformed grid pattern

        Example:
            >>> # Create deformation visualization grid
            >>> grid = transform_tools.generate_visual_grid_from_field(
            ...     cardiac_transform, reference_ct, grid_size=20
            ... )
            >>> # Overlay on original image for visualization
        """
        grid_image = self.generate_grid_image(reference_image, grid_size, line_width)

        grid_image_tfm = self.transform_image(grid_image, tfm, reference_image)

        return grid_image_tfm
