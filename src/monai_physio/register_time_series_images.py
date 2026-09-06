"""Time series image registration implementation.

This module provides the RegisterTimeSeriesImages class for registering an
ordered sequence of images (time series) to a fixed image using a
caller-supplied RegisterImagesBase backend (e.g. RegisterImagesGreedy,
RegisterImagesICON, or RegisterImagesGreedyICON for Greedy-then-ICON
refinement).

The class is particularly useful for 4D medical imaging applications such as cardiac
CT where sequential frames need to be registered to a common frame.
"""

import logging
from typing import Literal, Optional, Union, cast

import itk
import numpy as np

from .register_images_base import RegisterImagesBase
from .register_images_greedy import RegisterImagesGreedy
from .transform_tools import TransformTools


class RegisterTimeSeriesImages(RegisterImagesBase):
    """Register a time series of images to a fixed image.

    This class extends RegisterImagesBase to provide registration of multiple
    images (time series) to a fixed image, using a caller-supplied registration
    backend. Every frame is registered to the fixed image independently.

    Key features:

    - Sequential registration of ordered image lists
    - Supports any RegisterImagesBase backend, including RegisterImagesChain
      / RegisterImagesGreedyICON for multi-stage registration
    - Configurable starting point in the time series
    - Returns all transforms and loss values for the entire series

    Attributes:
        registrar (RegisterImagesBase): The registration backend in use.
        transform_tools (TransformTools): Utility for transform operations.

    Example:
        >>> # Register a cardiac CT time series
        >>> registrar = RegisterTimeSeriesImages()
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(fixed_image)
        >>>
        >>> # Register all time points to fixed image
        >>> result = registrar.register_time_series(
        ...     moving_images=time_series_images,
        ...     reference_frame=5,  # Start from middle of cardiac cycle
        ...     register_reference=True,
        ... )
        >>>
        >>> # warp moving images -> fixed grid
        >>> f2m_tfms = result['fixed_to_moving_transforms']
        >>> # warp fixed image -> moving grids
        >>> m2f_tfms = result['moving_to_fixed_transforms']
        >>> losses = result['losses']
        >>>
        >>> # Reconstruct time series with optional upsampling
        >>> reconstructed = registrar.reconstruct_time_series(
        ...     moving_images=time_series_images,
        ...     moving_to_fixed_transforms=m2f_tfms,
        ...     upsample_to_fixed_resolution=True,
        ... )
    """

    def __init__(
        self,
        registration_method: Optional[RegisterImagesBase] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """Initialize the time series image registration class.

        Args:
            registration_method: Registration backend instance to use.
                Defaults to a new RegisterImagesGreedy when None.
            log_level: Logging level (default: logging.INFO)

        Raises:
            TypeError: If registration_method is neither None nor a
                RegisterImagesBase instance.
        """
        super().__init__(log_level=log_level)

        if registration_method is None:
            registration_method = RegisterImagesGreedy(log_level=log_level)
        elif not isinstance(registration_method, RegisterImagesBase):
            raise TypeError(
                "registration_method must be a RegisterImagesBase instance or None"
            )
        self.registrar: RegisterImagesBase = registration_method

        self.transform_tools: TransformTools = TransformTools()

    def set_mask_dilation(self, mask_dilation_mm: float) -> None:
        """Set the dilation of the fixed and moving image masks.

        This passes through to the underlying registration method.

        Args:
            mask_dilation_mm (float): The dilation in millimeters.
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_modality(self, modality: str) -> None:
        """Set the imaging modality for registration optimization.

        This passes through to the underlying registration method.

        Args:
            modality (str): The imaging modality (e.g., 'ct', 'mri')
        """
        self.modality = modality

    def set_fixed_image(self, fixed_image: itk.Image) -> None:
        """Set the fixed image for registration.

        All moving images in the time series will be registered to this
        fixed image.

        Args:
            fixed_image (itk.Image): The 3D fixed image
        """
        self.fixed_image = fixed_image

    def set_fixed_mask(self, fixed_mask: Optional[itk.Image]) -> None:
        """Set a binary mask for the fixed image region of interest.

        This passes through to the underlying registration method.

        Args:
            fixed_mask (itk.Image): Binary mask defining ROI
        """
        self.fixed_mask = fixed_mask

    def set_fixed_labelmap(self, fixed_labelmap: Optional[itk.Image]) -> None:
        """Set a labelmap for the fixed image region of interest.

        This passes through to the underlying registration method.

        Args:
            fixed_labelmap (Optional[itk.Image]): Labelmap defining ROI
        """
        self.fixed_labelmap = fixed_labelmap

    def register_time_series(
        self,
        moving_images: list[itk.Image],
        moving_masks: Optional[list[Optional[itk.Image]]] = None,
        moving_labelmaps: Optional[list[Optional[itk.Image]]] = None,
        reference_frame: int = 0,
        register_reference: bool = True,
    ) -> dict[str, list[itk.Transform] | list[float]]:
        """Register a time series of images to the fixed image.

        This method registers an ordered sequence of images to a common fixed
        frame. The reference frame is registered first, then every other frame,
        each independently of the others.

        Args:
            moving_images (list[itk.Image]): List of 3D images to register
            moving_masks (list[itk.Image], optional): List of binary masks,
                one for each moving image. If None, no masks are used. If provided,
                must have the same length as moving_images. Default: None
            moving_labelmaps (list[itk.Image], optional): Per-frame multi-label
                segmentations, one for each moving image. If None, no labelmaps are
                used. If provided, must have the same length as moving_images. Default: None
            reference_frame (int, optional): Index of the reference image, which
                is registered first. Default: 0
            register_reference (bool, optional): If True, register the
                reference image to the fixed image. If False, use identity transform
                for the reference image. Default: True

        Returns:
            dict: Dictionary containing results:
                - "fixed_to_moving_transforms" (list[itk.Transform]): one per image;
                  each warps its moving image onto the fixed grid (warping
                  moving points/landmarks into fixed space uses the matching
                  moving_to_fixed_transforms entry instead -- see
                  docs/developer/transform_conventions)
                - "moving_to_fixed_transforms" (list[itk.Transform]): one per image;
                  each warps the fixed image onto that moving image's grid
                  (used by reconstruct_time_series)
                - "losses" (list[float]): Registration loss value for each image

        Raises:
            ValueError: If fixed_image is not set
            ValueError: If reference_frame is out of range
            ValueError: If moving_masks length doesn't match moving_images length

        Note:
            Every frame is registered independently, so an error in one frame
            cannot propagate along the series.

            The fixed image mask can be set using set_fixed_mask() before
            calling this method.

        Example:
            >>> greedy = RegisterImagesGreedy()
            >>> registrar = RegisterTimeSeriesImages(registration_method=greedy)
            >>> registrar.set_fixed_image(fixed_image)
            >>> registrar.set_fixed_mask(fixed_mask)  # Optional
            >>>
            >>> result = registrar.register_time_series(
            ...     moving_images=image_list,
            ...     moving_masks=mask_list,  # Optional
            ...     moving_labelmaps=labelmap_list,  # Optional
            ...     reference_frame=5,
            ...     register_reference=True,
            ... )
            >>>
            >>> # Access results using new intuitive names
            >>> for i, (f2m_tfm, loss) in enumerate(
            ...     zip(result['fixed_to_moving_transforms'], result['losses'])
            ... ):
            ...     # Apply fixed_to_moving_transform to align moving image i
            ...     # to fixed
            ...     registered = transform_tools.transform_image(
            ...         moving_images[i], f2m_tfm, fixed_image
            ...     )
        """
        if self.fixed_image is None:
            raise ValueError("Fixed image must be set before registering time series")

        self.registrar.set_fixed_image(self.fixed_image)
        self.registrar.set_modality(self.modality)
        self.registrar.set_mask_dilation(self.mask_dilation_mm)
        self.registrar.set_fixed_mask(self.fixed_mask)
        self.registrar.set_fixed_labelmap(self.fixed_labelmap)

        num_images = len(moving_images)

        if reference_frame < 0 or reference_frame >= num_images:
            raise ValueError(
                f"reference_frame {reference_frame} out of range [0, {num_images - 1}]"
            )

        if moving_masks is not None and len(moving_masks) != num_images:
            raise ValueError(
                f"moving_masks length ({len(moving_masks)}) must match "
                f"moving_images length ({num_images})"
            )

        if moving_labelmaps is not None and len(moving_labelmaps) != num_images:
            raise ValueError(
                f"moving_labelmaps length ({len(moving_labelmaps)}) must match "
                f"moving_images length ({num_images})"
            )

        # Initialize result lists
        fixed_to_moving_transforms: list[Optional[itk.Transform]] = [None] * num_images
        moving_to_fixed_transforms: list[Optional[itk.Transform]] = [None] * num_images
        losses = [0.0] * num_images

        # Create identity transform for fixed image
        identity_tfm = itk.IdentityTransform[itk.D, 3].New()
        identity_tfm = (
            self.transform_tools.convert_transform_to_displacement_field_transform(
                identity_tfm, self.fixed_image
            )
        )

        # Register the reference frame image
        if register_reference:
            reference_mask = (
                moving_masks[reference_frame] if moving_masks is not None else None
            )
            reference_labelmap = (
                moving_labelmaps[reference_frame]
                if moving_labelmaps is not None
                else None
            )
            result = self.registrar.register(
                moving_images[reference_frame],
                moving_mask=reference_mask,
                moving_labelmap=reference_labelmap,
            )
            fixed_to_moving_transform = result["fixed_to_moving_transform"]
            moving_to_fixed_transform = result["moving_to_fixed_transform"]
            loss = result["loss"]
        else:
            # Use identity transform for reference frame
            fixed_to_moving_transform = identity_tfm
            moving_to_fixed_transform = identity_tfm
            loss = 0.0

        fixed_to_moving_transforms[reference_frame] = fixed_to_moving_transform
        moving_to_fixed_transforms[reference_frame] = moving_to_fixed_transform
        losses[reference_frame] = loss

        # Register every remaining frame; each is independent of the others.
        for img_idx in range(num_images):
            if img_idx == reference_frame:
                continue
            moving_image = moving_images[img_idx]
            moving_mask = moving_masks[img_idx] if moving_masks is not None else None
            moving_labelmap = (
                moving_labelmaps[img_idx] if moving_labelmaps is not None else None
            )

            result = self.registrar.register(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_labelmap=moving_labelmap,
            )

            fixed_to_moving_transforms[img_idx] = result["fixed_to_moving_transform"]
            moving_to_fixed_transforms[img_idx] = result["moving_to_fixed_transform"]
            losses[img_idx] = cast(float, result["loss"])

        assert all(t is not None for t in fixed_to_moving_transforms)
        assert all(t is not None for t in moving_to_fixed_transforms)
        return {
            "fixed_to_moving_transforms": [
                t for t in fixed_to_moving_transforms if t is not None
            ],
            "moving_to_fixed_transforms": [
                t for t in moving_to_fixed_transforms if t is not None
            ],
            "losses": losses,
        }

    def reconstruct_time_series(
        self,
        moving_images: list[itk.Image],
        moving_to_fixed_transforms: list[itk.Transform],
        upsample_to_fixed_resolution: bool = False,
        fixed_to_moving_transforms: Optional[list[itk.Transform]] = None,
        composite_mode: Literal["reference", "mean", "max"] = "reference",
    ) -> list[itk.Image]:
        """Reconstruct time series images using moving_to_fixed_transforms.

        This method applies the moving_to_fixed_transforms to reconstruct each
        moving image in the fixed image space. If upsample_to_fixed_resolution
        is enabled, the reconstructed images will use isotropic spacing (mean
        of fixed image's X and Y spacing) while maintaining each moving
        image's original origin and direction.

        By default (composite_mode="reference"), the fixed/reference image is
        warped back to each time point. When composite_mode is "mean" or
        "max", a single composite image is built first -- the pixel-by-pixel
        mean or max across the fixed image and every moving image warped onto
        the fixed grid via fixed_to_moving_transforms -- and that composite is warped
        back to each time point instead. This lets anatomy or contrast only
        visible in some frames propagate into every reconstructed time point.

        Args:
            moving_images (list[itk.Image]): List of moving images to reconstruct
            moving_to_fixed_transforms (list[itk.Transform]): List of
                moving-to-fixed transforms (one per moving image), each used
                to warp the fixed image onto that moving image's grid
            upsample_to_fixed_resolution (bool, optional): If True, reconstructed
                images will be upsampled to isotropic resolution (mean of fixed image's
                X and Y spacing) while maintaining their original origin and direction.
                Default: False
            fixed_to_moving_transforms (list[itk.Transform], optional): List of
                fixed-to-moving transforms (one per moving image), each used to
                warp that moving image onto the fixed grid. Required when
                composite_mode is "mean" or "max". Default: None
            composite_mode (Literal["reference", "mean", "max"], optional):
                Which image to warp back to each time point. "reference" uses
                the fixed image as-is (default). "mean"/"max" build a composite
                of the fixed image and all registered moving images first.

        Returns:
            list[itk.Image]: List of reconstructed images in fixed image space

        Raises:
            ValueError: If fixed_image is not set
            ValueError: If lengths of moving_images and
                moving_to_fixed_transforms don't match
            ValueError: If composite_mode is "mean"/"max" and
                fixed_to_moving_transforms is not provided or its length
                doesn't match moving_images

        Example:
            >>> greedy = RegisterImagesGreedy()
            >>> registrar = RegisterTimeSeriesImages(registration_method=greedy)
            >>> registrar.set_fixed_image(fixed_image)
            >>>
            >>> result = registrar.register_time_series(
            ...     moving_images=time_series_images,
            ...     reference_frame=0,
            ... )
            >>>
            >>> reconstructed_images = registrar.reconstruct_time_series(
            ...     moving_images=time_series_images,
            ...     moving_to_fixed_transforms=result['moving_to_fixed_transforms'],
            ...     upsample_to_fixed_resolution=True,
            ... )
        """
        if self.fixed_image is None:
            raise ValueError(
                "Fixed image must be set before reconstructing time series"
            )

        if len(moving_images) != len(moving_to_fixed_transforms):
            raise ValueError(
                f"Number of moving images ({len(moving_images)}) must match "
                f"number of moving_to_fixed_transforms "
                f"({len(moving_to_fixed_transforms)})"
            )

        if composite_mode == "reference":
            source_image = self.fixed_image
        elif composite_mode in ("mean", "max"):
            if fixed_to_moving_transforms is None or len(
                fixed_to_moving_transforms
            ) != len(moving_images):
                raise ValueError(
                    "fixed_to_moving_transforms must be provided and match "
                    "moving_images length when composite_mode is "
                    f"{composite_mode!r}"
                )
            source_image = self._compute_composite_reference(
                moving_images, fixed_to_moving_transforms, composite_mode
            )
        else:
            raise ValueError(
                "composite_mode must be 'reference', 'mean', or 'max', "
                f"got {composite_mode!r}"
            )

        reconstructed_images: list[itk.Image] = []

        for moving_image, moving_to_fixed_transform in zip(
            moving_images, moving_to_fixed_transforms
        ):
            if upsample_to_fixed_resolution:
                # Create a reference image with isotropic spacing (mean of fixed image's
                # X and Y spacing) and moving image's origin and direction
                reference_image = self._create_upsampled_reference(
                    moving_image, self.fixed_image
                )
            else:
                # Use the moving image's own grid as the output space
                reference_image = moving_image

            # Transform the source image to the reference space.  The source
            # image is an intensity image, so voxels sampled outside it take the
            # modality's "no tissue" value, not 0.
            reconstructed = self.transform_tools.transform_image(
                source_image,
                moving_to_fixed_transform,
                reference_image,
                background_value=self._prewarp_background_value(source_image),
            )
            reconstructed_images.append(reconstructed)

        return reconstructed_images

    def _compute_composite_reference(
        self,
        moving_images: list[itk.Image],
        fixed_to_moving_transforms: list[itk.Transform],
        mode: Literal["mean", "max"],
    ) -> itk.Image:
        """Build a composite reference image from the fixed image and moving images.

        Warps every moving image onto the fixed grid using its
        fixed_to_moving_transform, then combines those registered images with the
        fixed image pixel-by-pixel using the given reduction. Moving images
        whose extent does not fully cover the fixed grid contribute only
        where they actually have data -- voxels resampled from outside a
        moving image's bounds (extrapolated fill) are excluded from the
        reduction rather than treated as real samples. The fixed image
        counts as one valid sample at every voxel.

        Args:
            moving_images (list[itk.Image]): Moving images to warp and combine
            fixed_to_moving_transforms (list[itk.Transform]): One
                fixed_to_moving transform per moving image, warping it onto
                the fixed grid
            mode (Literal["mean", "max"]): Pixel-wise reduction to apply

        Returns:
            itk.Image: Composite image on the fixed image's grid
        """
        assert self.fixed_image is not None
        fixed_arr = itk.GetArrayViewFromImage(self.fixed_image)
        dtype = fixed_arr.dtype

        # float32 keeps peak memory bounded for large volumes; only widen to
        # float64 when the source data already needs it. A float accumulator
        # (rather than the fixed image's own dtype) also keeps np.maximum
        # from raising when a moving image's pixel type is floating point
        # but the fixed image's is integer.
        accumulator_dtype = np.float64 if dtype == np.float64 else np.float32
        accumulator = fixed_arr.astype(accumulator_dtype)
        if mode == "mean":
            valid_count = np.ones_like(accumulator)

        for moving_image, fixed_to_moving_transform in zip(
            moving_images, fixed_to_moving_transforms
        ):
            registered = self.transform_tools.transform_image(
                moving_image,
                fixed_to_moving_transform,
                self.fixed_image,
                background_value=self._prewarp_background_value(moving_image),
            )
            registered_arr = itk.GetArrayViewFromImage(registered)

            moving_shape = itk.GetArrayViewFromImage(moving_image).shape
            coverage_image = itk.image_from_array(np.ones(moving_shape, dtype=np.uint8))
            coverage_image.CopyInformation(moving_image)
            registered_coverage = self.transform_tools.transform_image(
                coverage_image,
                fixed_to_moving_transform,
                self.fixed_image,
                interpolation_method="nearest",
                background_value=0,
            )
            valid_mask = itk.GetArrayViewFromImage(registered_coverage) != 0

            if mode == "mean":
                accumulator += np.where(valid_mask, registered_arr, 0)
                valid_count += valid_mask
            else:
                masked = np.where(valid_mask, registered_arr, accumulator)
                np.maximum(accumulator, masked, out=accumulator)

        if mode == "mean":
            accumulator /= valid_count
            if np.issubdtype(dtype, np.integer):
                accumulator = np.round(accumulator)
        reduced = accumulator.astype(dtype)

        composite = itk.image_from_array(np.ascontiguousarray(reduced))
        composite.CopyInformation(self.fixed_image)
        return composite

    def _create_upsampled_reference(
        self, moving_image: itk.Image, fixed_image: itk.Image
    ) -> itk.Image:
        """Create a reference image with isotropic spacing and moving image origin/direction.

        The spacing is calculated as the mean of the fixed image's X and Y spacing,
        applied to all three dimensions (X, Y, Z) for isotropic resolution.

        Args:
            moving_image (itk.Image): Image providing origin and direction
            fixed_image (itk.Image): Image providing spacing for X and Y dimensions

        Returns:
            itk.Image: Reference image with isotropic spacing and moving image's
                origin and direction
        """
        # Get properties from both images
        moving_origin = moving_image.GetOrigin()
        moving_direction = moving_image.GetDirection()
        moving_spacing = moving_image.GetSpacing()
        moving_size = moving_image.GetLargestPossibleRegion().GetSize()

        fixed_spacing = fixed_image.GetSpacing()

        # Calculate mean of X and Y spacing for isotropic resolution
        mean_xy_spacing = (fixed_spacing[0] + fixed_spacing[1]) / 2.0

        # Create ITK Vector for spacing
        isotropic_spacing = itk.Vector[itk.D, 3]()
        isotropic_spacing[0] = mean_xy_spacing
        isotropic_spacing[1] = mean_xy_spacing
        isotropic_spacing[2] = mean_xy_spacing

        # Calculate new size to cover the same physical extent with isotropic spacing
        new_size = itk.Size[3]()
        for i in range(3):
            new_size[i] = int(
                round((moving_size[i] * moving_spacing[i]) / isotropic_spacing[i])
            )

        # Create reference image with combined properties
        ImageType = type(moving_image)
        reference_image = ImageType.New()
        reference_image.SetOrigin(moving_origin)
        reference_image.SetDirection(moving_direction)
        reference_image.SetSpacing(isotropic_spacing)

        region = itk.ImageRegion[3]()
        region.SetSize(new_size)
        reference_image.SetRegions(region)
        reference_image.Allocate()

        return reference_image

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Registration method required by RegisterImagesBase.

        Delegates to the configured ``registrar``. This method is not
        typically called directly; use register_time_series() instead for
        time series registration.

        Args:
            moving_image (itk.Image): Image to register
            moving_mask (itk.Image, optional): Binary mask
            moving_labelmap (itk.Image, optional): Multi-label segmentation
            moving_image_pre (itk.Image, optional): Ignored - the registrar
                computes its own preprocessing from the raw moving_image

        Returns:
            dict: Registration result with fixed_to_moving_transform,
                moving_to_fixed_transform, and loss
        """
        self._delegate_to(self.registrar, moving_image, moving_mask, moving_labelmap)
        result = self.registrar.registration_method(
            moving_image=moving_image,
            moving_mask=moving_mask,
            moving_labelmap=moving_labelmap,
            moving_image_pre=None,
        )
        self._capture_delegate_result(self.registrar, result)
        return {
            "fixed_to_moving_transform": cast(
                itk.Transform, result["fixed_to_moving_transform"]
            ),
            "moving_to_fixed_transform": cast(
                itk.Transform, result["moving_to_fixed_transform"]
            ),
            "loss": float(cast(float, result["loss"])),
        }
