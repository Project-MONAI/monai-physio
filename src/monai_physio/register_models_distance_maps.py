"""Distance-map-based model-to-model registration for anatomical models.

This module provides the RegisterModelsDistanceMaps class for aligning anatomical
models using distance-map-based deformable registration. The workflow includes:
1. Generate distance maps from moving and fixed models
2. Generate binary registration masks with dilation
3. Progressive registration stages:
   - rigid: Greedy rigid registration
   - affine: Greedy affine registration
   - deformable: Greedy affine → ICON deformable registration

The registration is particularly useful for aligning anatomical models where
shape differences require deformable transformations beyond rigid/affine ICP.

Key Features:
    - Automatic mask generation from PyVista models
    - Multi-stage Greedy/ICON registration (rigid/affine/deformable)
    - Automatic transform composition
    - Support for PyVista models

Example:
    >>> import itk
    >>> import pyvista as pv
    >>> from monai_physio import RegisterModelsDistanceMaps
    >>>
    >>> # Load models and reference image
    >>> moving_model = pv.read('generic_model.vtu').extract_surface(algorithm="dataset_surface")
    >>> fixed_model = pv.read('patient_surface.stl')
    >>> reference_image = itk.imread('patient_ct.nii.gz')
    >>>
    >>> # Run deformable registration (Greedy affine + ICON deformable)
    >>> registrar = RegisterModelsDistanceMaps(
    ...     moving_model=moving_model,
    ...     fixed_model=fixed_model,
    ...     reference_image=reference_image,
    ...     mask_dilation_mm=20,
    ... )
    >>> result = registrar.register(transform_type='Deformable')
    >>>
    >>> # Access results
    >>> aligned_model = result['registered_model']
    >>> fixed_to_moving_transform = result['fixed_to_moving_transform']  # warps moving image -> fixed grid
"""

import logging
from typing import Optional

import itk
import pyvista as pv

from .contour_tools import ContourTools
from .labelmap_tools import LabelmapTools
from .monai_physio_base import MONAIPhysioBase
from .register_images_greedy import RegisterImagesGreedy
from .register_images_icon import RegisterImagesICON
from .transform_tools import TransformTools


class RegisterModelsDistanceMaps(MONAIPhysioBase):
    """Register anatomical models using distance-map-based deformable registration.

    This class provides distance-map-based alignment of 3D surface models with support
    for rigid, affine, and deformable transformation modes. The registration pipeline
    generates signed distance maps from models, applies optional binary mask dilation,
    and uses Greedy for rigid/affine stages and ICON for deformable registration.

    **Registration Pipelines:**
        - **None mode**: No registration (identity transform)
        - **Rigid mode**: Greedy rigid registration
        - **Affine mode**: Greedy affine registration
        - **Deformable mode**: Greedy affine → ICON deformable registration

    **Transform Convention:**
        These are the underlying image-registration (Greedy/ICON) transforms, so
        they follow the image convention (see
        docs/developer/transform_conventions):

        - fixed_to_moving_transform: warps the moving image/mask onto the fixed grid.
          Warping the moving MODEL points/landmarks onto the fixed model uses
          moving_to_fixed_transform instead (image and point warps use opposite
          transforms).
        - moving_to_fixed_transform: warps the fixed image/mask onto the moving grid.

    Attributes:
        moving_model (pv.PolyData): Surface model to be aligned
        fixed_model (pv.PolyData): Target surface model
        reference_image (itk.Image): Reference image for coordinate frame
        mask_dilation_mm (float): Dilation amount in mm for binary registration masks
        distance_squared_max (float): Maximum squared distance for distance map normalization
        transform_tools (TransformTools): Transform utility instance
        contour_tools (ContourTools): Model utility instance
        registrar_Greedy (RegisterImagesGreedy): Greedy registration instance
        registrar_ICON (RegisterImagesICON): ICON registration instance
        fixed_to_moving_transform (itk.CompositeTransform): Optimized fixed-to-moving transform
        moving_to_fixed_transform (itk.CompositeTransform): Optimized moving-to-fixed transform
        registered_model (pv.PolyData): Aligned moving model

    Example:
        >>> # Initialize with models and reference image
        >>> registrar = RegisterModelsDistanceMaps(
        ...     moving_model=model_surface,
        ...     fixed_model=patient_surface,
        ...     reference_image=patient_ct,
        ...     mask_dilation_mm=20,
        ... )
        >>>
        >>> # Run rigid registration
        >>> result = registrar.register(transform_type='Rigid')
        >>>
        >>> # Or run affine registration
        >>> result = registrar.register(transform_type='Affine')
        >>>
        >>> # Or run deformable (Greedy affine + ICON)
        >>> result = registrar.register(transform_type='Deformable')
        >>>
        >>> # Get aligned model and transforms
        >>> aligned_model = result['registered_model']
        >>> fixed_to_moving_transform = result['fixed_to_moving_transform']
    """

    def __init__(
        self,
        moving_model: pv.PolyData,
        fixed_model: pv.PolyData,
        reference_image: itk.Image,
        distance_squared_max: float = 50.0,
        mask_dilation_mm: float = 20,
        log_level: int | str = logging.INFO,
    ):
        """Initialize distance-map-based model registration.

        Args:
            moving_model: PyVista surface model to be aligned to fixed model
            fixed_model: PyVista target surface model
            reference_image: ITK image providing coordinate frame (origin, spacing, direction)
                for mask generation. Typically the patient CT/MRI image.
            distance_squared_max: Maximum squared distance, in squared millimeters,
                that the distance maps are normalized against. It fixes their
                intensity distribution, so it must match the value the ICON
                weights in use were finetuned at. Default: 50.0
            mask_dilation_mm: Dilation amount in millimeters for binary registration
                mask generation. Default: 20mm
            log_level: Logging level (default: logging.INFO)

        Note:
            The moving_model and fixed_model are typically extracted from VTU models
            using model.extract_surface(algorithm="dataset_surface") before passing to this class.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.moving_model = moving_model
        self.fixed_model = fixed_model
        self.reference_image = reference_image
        self.distance_squared_max = distance_squared_max
        self.mask_dilation_mm = mask_dilation_mm

        # Utilities
        self.transform_tools = TransformTools()
        self.contour_tools = ContourTools()
        self.labelmap_tools = LabelmapTools(log_level=log_level)

        # Registration instances
        self.registrar_Greedy = RegisterImagesGreedy(log_level=log_level)
        self.registrar_ICON = RegisterImagesICON(log_level=log_level)
        self.registrar_ICON.set_modality("ct")
        self.registrar_ICON.set_multi_modality(False)

        # Generated distance maps and binary registration masks (created during registration)
        self.fixed_distance_map_image: Optional[itk.Image] = None
        self.fixed_mask_image: Optional[itk.Image] = None
        self.moving_distance_map_image: Optional[itk.Image] = None
        self.moving_mask_image: Optional[itk.Image] = None

        # Registration results
        self.fixed_to_moving_transform: Optional[itk.CompositeTransform] = (
            None  # Fixed-to-moving
        )
        self.moving_to_fixed_transform: Optional[itk.CompositeTransform] = (
            None  # Moving-to-fixed
        )
        self.registered_model: Optional[pv.PolyData] = None

    def set_icon_weights_path(self, weights_path: str) -> None:
        """Use a finetuned uniGradICON checkpoint for the deformable stage.

        The distance maps this class registers are not CT intensities, so stock
        uniGradICON weights are out of distribution for them.  Weights finetuned
        on distance maps, e.g. by
        ``tutorials/tutorial_02_lung_distancemap_finetune_icon.py``, are
        supplied here.

        Args:
            weights_path: Path to an existing uniGradICON checkpoint.

        Raises:
            FileNotFoundError: If weights_path does not exist.
        """
        self.registrar_ICON.set_weights_path(weights_path)

    @staticmethod
    def _require_non_constant(
        image: itk.Image, description: str, greedy_loss: Optional[float] = None
    ) -> None:
        """Raise if *image* carries no contrast, naming what went wrong.

        A constant image reaches ICON as a uniform volume and trips a bare
        assertion inside ``icon_registration``'s ``register_pair`` that names
        neither which side degenerated nor why.  There are two ways to get one: a
        rasterization that caught no surface, and a registration stage that moved
        every sample outside the image it was resampling, leaving the resampler
        to fill the whole grid with its background value.

        Args:
            image: Image to check.
            description: What the image is, named the way the caller would
                recognize it.
            greedy_loss: Loss of the Greedy stage that produced *image*, when it
                came through one.  Supplied so the message can distinguish a
                diverged registration from an empty rasterization.

        Raises:
            RuntimeError: If every voxel carries the same value.
        """
        array = itk.GetArrayViewFromImage(image)
        minimum, maximum = float(array.min()), float(array.max())
        if minimum != maximum:
            return

        message = [
            f"The {description} is constant ({minimum:.6g} everywhere), so there "
            "is nothing for a registration metric to align."
        ]
        if greedy_loss is not None:
            message.append(
                f"The Greedy stage that produced it reported a loss of "
                f"{greedy_loss:.6g}. A loss at or near zero means that stage "
                "diverged, so every sample landed outside the moving image and "
                "the resampler filled the grid with its background value."
            )
            message.append(
                "Greedy is seeded nondeterministically, so this is usually "
                "transient rather than a property of the data. Re-running is the "
                "first thing to try; the workflows that call this cache their "
                "artifacts, so a re-run resumes at the failed item rather than "
                "starting over."
            )
        else:
            message.append(
                "Check that the model falls inside the reference image and that "
                "it has surface geometry to rasterize."
            )
        raise RuntimeError(" ".join(message))

    def _create_masks_from_models(self) -> None:
        """Generate distance maps and binary registration masks from moving and fixed models.

        Creates:
            - fixed_distance_map_image: Signed distance map from fixed model
            - fixed_mask_image: Dilated binary registration mask from fixed model
            - moving_distance_map_image: Signed distance map from moving model
            - moving_mask_image: Dilated binary registration mask from moving model

        Uses self.reference_image for coordinate frame (origin, spacing, direction).
        """
        self.log_info("Generating distance maps and registration masks from models...")

        # Create fixed distance map
        self.fixed_distance_map_image = self.contour_tools.create_distance_map(
            self.fixed_model,
            self.reference_image,
            squared_distance=True,
            negative_inside=True,
            zero_inside=False,
            norm_to_max_distance=self.distance_squared_max,
        )

        if self.mask_dilation_mm > 0:
            # Create fixed binary registration mask with dilation
            self.log_info(
                "Dilating fixed mask by %.1fmm for registration mask...",
                self.mask_dilation_mm,
            )
            binary_mask = self.contour_tools.create_mask_from_mesh(
                self.fixed_model, self.reference_image
            )
            self.fixed_mask_image = self.labelmap_tools.convert_labelmap_to_mask(
                binary_mask, dilation_in_mm=self.mask_dilation_mm
            )
        else:
            self.fixed_mask_image = None

        # Create moving distance map
        self.moving_distance_map_image = self.contour_tools.create_distance_map(
            self.moving_model,
            self.reference_image,
            squared_distance=True,
            negative_inside=True,
            zero_inside=False,
            norm_to_max_distance=self.distance_squared_max,
        )

        # Emulate CT intensity range by multiplying by 1000
        tmp_arr = itk.GetArrayViewFromImage(self.fixed_distance_map_image)
        tmp_arr *= 1000

        tmp_arr = itk.GetArrayViewFromImage(self.moving_distance_map_image)
        tmp_arr *= 1000

        if self.mask_dilation_mm > 0:
            # Create moving binary registration mask with dilation
            self.log_info(
                "Dilating moving mask by %.1fmm for registration mask...",
                self.mask_dilation_mm,
            )
            binary_mask = self.contour_tools.create_mask_from_mesh(
                self.moving_model, self.reference_image
            )
            self.moving_mask_image = self.labelmap_tools.convert_labelmap_to_mask(
                binary_mask, dilation_in_mm=self.mask_dilation_mm
            )
        else:
            self.moving_mask_image = None

        # Caught here rather than several stages later inside ICON, where the
        # same degeneracy surfaces as an assertion naming neither side nor cause.
        self._require_non_constant(
            self.fixed_distance_map_image, "fixed model's distance map"
        )
        self._require_non_constant(
            self.moving_distance_map_image, "moving model's distance map"
        )

        self.log_info("Distance map and mask generation complete")

    def register(
        self,
        transform_type: str = "Deformable",
    ) -> dict:
        """Perform mask-based registration of moving model to fixed model.

        This method executes progressive multi-stage registration:

        **None transform type:**
            1. No registration (identity transform)

        **Rigid transform type:**
            1. Greedy rigid registration

        **Affine transform type:**
            1. Greedy affine registration

        **Deformable transform type:**
            1. Greedy affine registration
            2. ICON deformable registration on the affine-pre-aligned masks

        Args:
            transform_type: Registration transform type - 'None', 'Rigid', 'Affine', or 'Deformable'. Default: 'Deformable'

        Returns:
            Dictionary containing:
                - 'registered_model': Aligned moving model (PyVista PolyData)
                - 'fixed_to_moving_transform': Fixed-to-moving transform (ITK CompositeTransform)
                - 'moving_to_fixed_transform': Moving-to-fixed transform (ITK CompositeTransform)

        Raises:
            ValueError: If transform_type is not 'None', 'Rigid', 'Affine', or 'Deformable'

        Example:
            >>> # Rigid registration
            >>> result = registrar.register(transform_type='Rigid')
            >>>
            >>> # Affine registration
            >>> result = registrar.register(transform_type='Affine')
            >>>
            >>> # Deformable registration (Greedy affine + ICON)
            >>> result = registrar.register(transform_type='Deformable')
        """
        if transform_type not in ["None", "Rigid", "Affine", "Deformable"]:
            raise ValueError(
                f"Invalid transform type '{transform_type}'. Must be 'None', 'Rigid', 'Affine', or 'Deformable'."
            )

        self.log_section("%s Distance-Map-based Registration", transform_type.upper())

        # Step 1: Generate distance maps and registration masks from models
        self._create_masks_from_models()

        # Step 2: Greedy rigid or affine stage (skipped for None/Deformable uses Affine)
        greedy_type = "Affine" if transform_type == "Deformable" else transform_type

        fixed_to_moving_transform_Greedy = None
        moving_to_fixed_transform_Greedy = None
        greedy_loss: Optional[float] = None
        if greedy_type != "None":
            self.log_info("Performing Greedy %s registration...", greedy_type)
            self.registrar_Greedy.set_fixed_image(self.fixed_distance_map_image)
            self.registrar_Greedy.set_fixed_mask(self.fixed_mask_image)
            self.registrar_Greedy.set_transform_type(greedy_type)
            self.registrar_Greedy.set_metric("CC")

            result_Greedy = self.registrar_Greedy.register(
                moving_image=self.moving_distance_map_image,
                moving_mask=self.moving_mask_image,
            )
            fixed_to_moving_transform_Greedy = result_Greedy[
                "fixed_to_moving_transform"
            ]
            moving_to_fixed_transform_Greedy = result_Greedy[
                "moving_to_fixed_transform"
            ]
            greedy_loss = result_Greedy.get("loss")
        else:
            identity_transform = itk.AffineTransform[itk.D, 3].New()
            identity_transform.SetIdentity()
            fixed_to_moving_transform_Greedy = identity_transform
            moving_to_fixed_transform_Greedy = identity_transform

        self.fixed_to_moving_transform = fixed_to_moving_transform_Greedy
        self.moving_to_fixed_transform = moving_to_fixed_transform_Greedy

        # Step 3: ICON deformable stage (only for Deformable mode)
        if transform_type == "Deformable":
            self.log_info("Performing ICON deformable registration...")

            # Pre-align moving distance map and binary mask into the fixed grid using the Greedy affine result
            moving_distance_map_affine_transformed = (
                self.transform_tools.transform_image(
                    self.moving_distance_map_image,
                    fixed_to_moving_transform_Greedy,
                    self.reference_image,
                    interpolation_method="linear",
                )
            )
            # A diverged Greedy affine puts every sample outside the moving
            # image, and transform_image then fills the grid with its background
            # value. Catch that here, where the Greedy loss is still in hand to
            # explain it, rather than inside ICON where it is a bare assertion.
            self._require_non_constant(
                moving_distance_map_affine_transformed,
                "moving distance map after the Greedy affine",
                greedy_loss=greedy_loss,
            )
            # moving_mask_affine_transformed = self.transform_tools.transform_image(
            # self.moving_mask_image,
            # fixed_to_moving_transform_Greedy,
            # self.reference_image,
            # interpolation_method="nearest",
            # )

            # Configure and run ICON. Iteration count and any other ICON tuning
            # come from registrar_ICON itself, configured by the caller.
            self.registrar_ICON.set_fixed_image(self.fixed_distance_map_image)
            # self.registrar_ICON.set_fixed_mask(self.fixed_mask_image)

            result_ICON = self.registrar_ICON.register(
                moving_image=moving_distance_map_affine_transformed,
                # moving_mask=moving_mask_affine_transformed,
            )
            fixed_to_moving_transform_ICON = result_ICON["fixed_to_moving_transform"]
            moving_to_fixed_transform_ICON = result_ICON["moving_to_fixed_transform"]

            # Compose Greedy affine + ICON deformable.
            # ICON runs on images already resampled to the patient (fixed) grid,
            # so its transforms are deformations within patient space.
            # fixed_to_moving_transform (image pull-back): apply ICON first
            # (patient-space delta), then Greedy (patient-to-ICP-template).
            # moving_to_fixed_transform (point push-forward): apply Greedy first
            # (ICP-template-to-patient), then ICON (patient-space refinement).
            # combine_displacement_field_transforms(a, b) evaluates b then a, so
            # the stage that runs first is the second argument.
            self.fixed_to_moving_transform = (
                self.transform_tools.combine_displacement_field_transforms(
                    fixed_to_moving_transform_Greedy,
                    fixed_to_moving_transform_ICON,
                    reference_image=self.reference_image,
                    mode="compose",
                )
            )
            self.moving_to_fixed_transform = (
                self.transform_tools.combine_displacement_field_transforms(
                    moving_to_fixed_transform_ICON,
                    moving_to_fixed_transform_Greedy,
                    reference_image=self.reference_image,
                    mode="compose",
                )
            )

        # Apply final transform to moving model
        self.log_info("Transforming moving model...")
        self.registered_model = self.transform_tools.transform_pvcontour(
            self.moving_model,
            self.moving_to_fixed_transform,
            with_deformation_magnitude=True,
        )

        self.log_info(
            "%s distance-map-based registration complete.", transform_type.upper()
        )

        self._release_intermediates()

        # Return results as dictionary
        return {
            "fixed_to_moving_transform": self.fixed_to_moving_transform,
            "moving_to_fixed_transform": self.moving_to_fixed_transform,
            "registered_model": self.registered_model,
        }

    def _release_intermediates(self) -> None:
        """Drop the working images once the result no longer depends on them.

        A registration builds four full-grid images here and hands two more
        preprocessed copies plus a pair of dense transforms to each sub-registrar.
        Together that is close to a gigabyte, held for as long as this object
        lives -- and callers construct one of these per frame, so with a cohort
        of any size the peak is set by how much is still reachable rather than by
        how much any one registration needs.

        Only the working set goes.  ``fixed_to_moving_transform``, ``moving_to_fixed_transform``
        and ``registered_model`` are the result and are left alone.
        """
        self.fixed_distance_map_image = None
        self.moving_distance_map_image = None
        self.fixed_mask_image = None
        self.moving_mask_image = None
        for registrar in (self.registrar_Greedy, self.registrar_ICON):
            registrar.fixed_image = None
            registrar.fixed_image_pre = None
            registrar.fixed_mask = None
            registrar.fixed_labelmap = None
            registrar.moving_image = None
            registrar.moving_image_pre = None
            registrar.moving_mask = None
            registrar.moving_labelmap = None
            registrar.moving_image_registered = None
            # The composed result above no longer refers to these, and each is a
            # dense field on the reference grid.
            registrar.fixed_to_moving_transform = None
            registrar.moving_to_fixed_transform = None
