"""High-resolution 4D CT reconstruction workflow.

This module provides the WorkflowReconstructHighres4DCT class for reconstructing
a high-resolution 4D CT time series from lower-resolution time-series images and
a single high-resolution reference image.

The workflow uses a caller-supplied RegisterImagesBase backend to:
1. Register each time-series image to the high-resolution reference
2. Apply inverse transforms to reconstruct high-resolution time series
3. Optionally upsample to the reference image resolution

This is particularly useful for cardiac CT where you have:
- Multiple low-resolution time-series images capturing cardiac motion
- One high-resolution static image for anatomical detail
- Goal: Combine both to create high-resolution dynamic images

Key Features:
    - Sequential time-series registration using RegisterTimeSeriesImages
    - Any RegisterImagesBase backend, defaulting to RegisterImagesGreedy
    - Bidirectional registration from reference frame
    - Optional temporal smoothing with prior transforms
    - High-resolution reconstruction with optional upsampling
    - No I/O operations (data passed in-memory)
"""

import logging
from typing import Literal, Optional

import itk

from .monai_physio_base import MONAIPhysioBase
from .register_images_base import RegisterImagesBase
from .register_time_series_images import RegisterTimeSeriesImages


class WorkflowReconstructHighres4DCT(MONAIPhysioBase):
    """Reconstruct high-resolution 4D CT from time series and reference image.

    This class implements a workflow for reconstructing high-resolution dynamic
    CT images by registering low-resolution time-series images to a high-resolution
    reference image using a caller-supplied registration backend.

    **Registration Pipeline:**
        1. **Time Series Registration**: Register each time-series image to the
           high-resolution reference using RegisterTimeSeriesImages
        2. **Reconstruction**: Apply inverse transforms to reconstruct high-resolution
           time series
        3. **Optional Upsampling**: Resample to isotropic high resolution

    **Input Requirements:**
        - time_series_images: Ordered list of 3D images (typically lower resolution)
        - reference_image: High-resolution reference image
        - All images should be in the same anatomical coordinate system

    ``registration_method`` accepts a pre-configured
    :class:`RegisterImagesBase` instance. Configure backend-specific
    parameters (iteration counts, etc.) on the instance before passing it
    in. Defaults to a new :class:`RegisterImagesGreedy` when omitted.

    Attributes:
        time_series_images (list[itk.Image]): Ordered list of time-series images
        reference_image (itk.Image): High-resolution reference image
        reference_time_frame (int): Index of reference time frame in time series
        register_reference_time_frame_to_reference_image (bool): Whether to register
            the reference time frame to the reference image
        upsample_to_fixed_resolution (bool): Whether to upsample reconstruction
        composite_mode (Literal["reference", "mean", "max"]): Which image is
            warped back to each time point during reconstruction: the
            reference image as-is, or a pixel-by-pixel mean/max composite of
            the reference image and all registered time-series images
        registrar (RegisterTimeSeriesImages): Internal registration object
        forward_transforms (list[itk.Transform]): one per frame; each warps its
            moving image onto the reference grid
        inverse_transforms (list[itk.Transform]): one per frame; each warps the
            reference image onto that frame's moving grid (used for reconstruction)
        losses (list[float]): Registration loss values
        reconstructed_images (list[itk.Image]): Reconstructed high-resolution images

    Example:
        >>> # Initialize workflow with data
        >>> workflow = WorkflowReconstructHighres4DCT(
        ...     time_series_images=lowres_images,
        ...     reference_image=highres_reference,
        ...     reference_time_frame=7,
        ... )
        >>>
        >>> # Access results
        >>> reconstructed = result['reconstructed_images']
        >>> transforms = result['forward_transforms']
        >>> losses = result['losses']
    """

    def __init__(
        self,
        time_series_images: list[itk.Image],
        reference_image: itk.Image,
        reference_time_frame: int = 0,
        register_reference_time_frame_to_reference_image: bool = True,
        registration_method: Optional[RegisterImagesBase] = None,
        log_level: int | str = logging.INFO,
    ):
        """Initialize the high-resolution 4D CT reconstruction workflow.

        Args:
            time_series_images (list[itk.Image]): Ordered list of 3D time-series images
                to be registered and reconstructed
            reference_image (itk.Image): High-resolution 3D reference image
            reference_time_frame (int, optional): Index of the reference time frame in
                the time series. Registration proceeds bidirectionally from this frame.
                Default: 0
            register_reference_time_frame_to_reference_image (bool, optional): If True,
                register the reference time frame to the reference image. If False, use
                an identity transform for that frame. Default: True
            registration_method (Optional[RegisterImagesBase]): Registration
                backend instance. Defaults to a new
                :class:`RegisterImagesGreedy` when None.
            log_level: Logging level (logging.DEBUG, logging.INFO, etc.).
                Default: logging.INFO

        Raises:
            ValueError: If time_series_images is empty
            ValueError: If reference_time_frame is out of range
            TypeError: If registration_method is neither None nor a
                RegisterImagesBase instance
        """
        # Initialize base class with logging
        super().__init__(
            class_name="WorkflowReconstructHighres4DCT", log_level=log_level
        )

        # Validate inputs
        if not time_series_images:
            raise ValueError("time_series_images cannot be empty")

        if reference_time_frame < 0 or reference_time_frame >= len(time_series_images):
            raise ValueError(
                f"reference_time_frame {reference_time_frame} out of range "
                f"[0, {len(time_series_images) - 1}]"
            )

        if registration_method is not None and not isinstance(
            registration_method, RegisterImagesBase
        ):
            raise TypeError(
                "registration_method must be a RegisterImagesBase instance or None"
            )

        # Store input data
        self.time_series_images = time_series_images
        self.reference_image = reference_image
        self.reference_time_frame = reference_time_frame
        self.register_reference_time_frame_to_reference_image = (
            register_reference_time_frame_to_reference_image
        )

        # Initialize parameters with defaults
        self.upsample_to_fixed_resolution: bool = True
        self.composite_mode: Literal["reference", "mean", "max"] = "reference"
        self.modality: str = "ct"
        self.mask_dilation_mm: float = 0.0
        self.fixed_mask: Optional[itk.Image] = None
        self.moving_masks: Optional[list[Optional[itk.Image]]] = None

        # Initialize registrar
        self.registrar = RegisterTimeSeriesImages(
            registration_method=registration_method, log_level=log_level
        )

        # Results storage
        self.forward_transforms: Optional[list[itk.Transform]] = None
        self.inverse_transforms: Optional[list[itk.Transform]] = None
        self.losses: Optional[list[float]] = None
        self.reconstructed_images: Optional[list[itk.Image]] = None

    def set_modality(self, modality: str) -> None:
        """Set the imaging modality for registration optimization.

        Args:
            modality (str): The imaging modality (e.g., 'ct', 'mri')
        """
        self.modality = modality

    def set_mask_dilation(self, mask_dilation_mm: float) -> None:
        """Set the dilation of the fixed and moving image masks.

        Args:
            mask_dilation_mm (float): The dilation in millimeters
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_fixed_mask(self, fixed_mask: Optional[itk.Image]) -> None:
        """Set a binary mask for the fixed image region of interest.

        Args:
            fixed_mask (itk.Image): Binary mask defining ROI in fixed image
        """
        self.fixed_mask = fixed_mask

    def set_moving_masks(
        self, moving_masks: Optional[list[Optional[itk.Image]]]
    ) -> None:
        """Set binary masks for the moving images.

        Args:
            moving_masks (list[itk.Image] | None): List of binary masks,
                one for each moving image. If None, no masks are used.
                Must have same length as time_series_images if provided.

        Raises:
            ValueError: If moving_masks length doesn't match time_series_images
        """
        if moving_masks is not None and len(moving_masks) != len(
            self.time_series_images
        ):
            raise ValueError(
                f"moving_masks length ({len(moving_masks)}) must match "
                f"time_series_images length ({len(self.time_series_images)})"
            )
        self.moving_masks = moving_masks

    def register_time_series(self) -> dict:
        """Register time series images to the fixed image.

        Performs sequential registration of all time-series images to the
        high-resolution reference image using the configured parameters.

        Returns:
            dict: Dictionary containing:
                - 'forward_transforms' (list[itk.Transform]): one per frame;
                  each warps its moving image onto the reference grid
                - 'inverse_transforms' (list[itk.Transform]): one per frame;
                  each warps the reference image onto that frame's moving grid
                  (see docs/developer/transform_conventions)
                - 'losses' (list[float]): Registration loss value for each image

        Raises:
            RuntimeError: If registration fails
        """
        self.log_section(
            "Stage 1: Time Series Registration (RegisterTimeSeriesImages)", width=70
        )

        # Configure registrar
        self.registrar.set_fixed_image(self.reference_image)
        self.registrar.set_modality(self.modality)
        self.registrar.set_mask_dilation(self.mask_dilation_mm)
        self.registrar.set_fixed_mask(self.fixed_mask)

        self.log_info(f"Registration method: {type(self.registrar.registrar).__name__}")
        self.log_info(f"Number of time points: {len(self.time_series_images)}")
        self.log_info(f"Reference time frame: {self.reference_time_frame}")
        self.log_info(
            "Register reference time frame to reference image: "
            f"{self.register_reference_time_frame_to_reference_image}"
        )

        # Perform registration
        result = self.registrar.register_time_series(
            moving_images=self.time_series_images,
            moving_masks=self.moving_masks,
            reference_frame=self.reference_time_frame,
            register_reference=self.register_reference_time_frame_to_reference_image,
        )

        # Store results
        self.forward_transforms = result["forward_transforms"]
        self.inverse_transforms = result["inverse_transforms"]
        self.losses = result["losses"]

        self.log_info("Stage 1 complete: Time series registration finished.")
        self.log_info(f"  Average loss: {sum(self.losses) / len(self.losses):.6f}")
        self.log_info(f"  Min loss: {min(self.losses):.6f}")
        self.log_info(f"  Max loss: {max(self.losses):.6f}")

        return {
            "forward_transforms": self.forward_transforms,
            "inverse_transforms": self.inverse_transforms,
            "losses": self.losses,
        }

    def set_upsample_to_fixed_resolution(
        self, upsample_to_fixed_resolution: bool
    ) -> None:
        """Set whether to upsample the reconstructed time series to the fixed
        resolution.

        Args:
            upsample_to_fixed_resolution (bool): Whether to upsample the reconstructed
                time series to the fixed resolution.
        """
        self.upsample_to_fixed_resolution = upsample_to_fixed_resolution

    def set_composite_mode(
        self, composite_mode: Literal["reference", "mean", "max"]
    ) -> None:
        """Set which image is warped back to each time point during reconstruction.

        Args:
            composite_mode (Literal["reference", "mean", "max"]): "reference"
                warps the reference image back to each time point (default).
                "mean"/"max" first build a composite of the reference image
                and every registered time-series image (pixel-by-pixel mean
                or max on the reference grid), then warp that composite back
                to each time point instead.

        Raises:
            ValueError: If composite_mode is not one of "reference", "mean",
                or "max"
        """
        if composite_mode not in ("reference", "mean", "max"):
            raise ValueError(
                f"composite_mode must be 'reference', 'mean', or 'max', "
                f"got {composite_mode!r}"
            )
        self.composite_mode = composite_mode

    def reconstruct_time_series(self) -> dict:
        """Reconstruct high-resolution time series using inverse transforms.

        Applies the inverse transforms from registration to reconstruct each
        time-series image in the high-resolution fixed image space.

        Returns:
            dict: Dictionary containing:
                - 'reconstructed_images' (list[itk.Image]): Reconstructed high-resolution
                  time-series images

        Raises:
            RuntimeError: If reconstruction fails
            ValueError: If inverse_transforms is not set (call register_time_series first)
        """
        if self.inverse_transforms is None:
            raise ValueError(
                "inverse_transforms not set. Call register_time_series() first."
            )

        self.log_section(
            "Stage 2: High-Resolution Time Series Reconstruction", width=70
        )

        self.log_info(
            f"Upsampling to fixed resolution: {self.upsample_to_fixed_resolution}"
        )
        self.log_info(f"Composite mode: {self.composite_mode}")

        # Reconstruct time series
        self.reconstructed_images = self.registrar.reconstruct_time_series(
            moving_images=self.time_series_images,
            inverse_transforms=self.inverse_transforms,
            upsample_to_fixed_resolution=self.upsample_to_fixed_resolution,
            forward_transforms=self.forward_transforms,
            composite_mode=self.composite_mode,
        )

        self.log_info("Stage 2 complete: Time series reconstruction finished.")
        self.log_info(f"  Reconstructed {len(self.reconstructed_images)} images")

        # Log image properties for first reconstructed image
        if self.reconstructed_images:
            img = self.reconstructed_images[0]
            self.log_info(f"  Reconstructed image size: {itk.size(img)}")
            self.log_info(f"  Reconstructed image spacing: {itk.spacing(img)}")

        return {"reconstructed_images": self.reconstructed_images}

    def process(self) -> dict:
        """Execute the complete high-resolution 4D CT reconstruction workflow.

        Runs the full pipeline:
        1. Register time series to high-resolution reference
        2. Reconstruct high-resolution time series using inverse transforms

        Returns:
            dict: Dictionary containing all results:
                - 'forward_transforms' (list[itk.Transform]): Registration transforms
                - 'inverse_transforms' (list[itk.Transform]): Inverse transforms
                - 'losses' (list[float]): Registration loss values
                - 'reconstructed_images' (list[itk.Image]): Reconstructed high-res images

        Raises:
            RuntimeError: If any stage of the workflow fails
        """
        self.log_section(
            "STARTING HIGH-RESOLUTION 4D CT RECONSTRUCTION WORKFLOW", width=70
        )

        self.log_info("Input configuration:")
        self.log_info(f"  Number of time points: {len(self.time_series_images)}")
        registrar_type = type(self.registrar.registrar).__name__
        self.log_info(f"  Registration method: {registrar_type}")
        self.log_info(f"  Reference time frame: {self.reference_time_frame}")
        self.log_info(
            f"  Upsample to fixed resolution: {self.upsample_to_fixed_resolution}"
        )

        # Stage 1: Register time series
        _ = self.register_time_series()

        # Stage 2: Reconstruct high-resolution time series
        _ = self.reconstruct_time_series()

        self.log_section("RECONSTRUCTION WORKFLOW COMPLETE", width=70)
        assert self.reconstructed_images is not None, "Reconstructed images must be set"
        self.log_info(
            f"Successfully reconstructed {len(self.reconstructed_images)} "
            "high-resolution images"
        )

        return {
            "forward_transforms": self.forward_transforms,
            "inverse_transforms": self.inverse_transforms,
            "losses": self.losses,
            "reconstructed_images": self.reconstructed_images,
        }
