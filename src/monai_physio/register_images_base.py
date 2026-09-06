"""Base class for image registration algorithms.

This module provides the RegisterImagesBase class that serves as a foundation
for implementing different image registration algorithms. It defines a common
interface and shared functionality for deformable image registration, particularly
designed for medical imaging applications such as 4D cardiac CT registration.

The base class handles common operations including:
- Fixed and moving image management
- Binary mask processing and dilation
- Modality-specific parameter settings
- Standardized registration interface

Concrete implementations should inherit from RegisterImagesBase and implement
the register() method with their specific algorithm (e.g., Icon, ANTs, etc.).
"""

import logging
from typing import Any, Optional, Union, cast

import itk
import numpy as np

from .labelmap_tools import LabelmapTools
from .monai_physio_base import MONAIPhysioBase
from .transform_tools import TransformTools


class RegisterImagesBase(MONAIPhysioBase):
    """Base class for deformable image registration algorithms.

    This class provides a common interface and shared functionality for
    implementing different image registration algorithms. It handles standard
    operations like image and mask management, preprocessing, and parameter
    configuration that are common across registration methods.

    The base class is designed to support various registration algorithms
    including deep learning-based methods (Icon, UniGradIcon) and traditional
    methods (ANTs, ITK). Concrete implementations should inherit from this
    class and implement the register() method.

    Key features:
    - Standardized interface for different registration algorithms
    - Fixed and moving image management
    - Binary mask processing with optional dilation
    - Modality-specific parameter configuration
    - Support for region-of-interest registration

    Attributes:
        net (object): Algorithm-specific network or registration object
        modality (str): Image modality ('ct', 'mri', etc.) for parameter optimization
        fixed_image (itk.image): The target/reference image
        fixed_image_pre (itk.image): Preprocessed fixed image
        fixed_mask (itk.image): Binary mask for fixed image ROI
        mask_dilation_mm (float): Mask dilation amount in millimeters
        fast_mode (bool): When True, subclasses may use cheaper/less-accurate
            registration settings to trade quality for speed (e.g. in
            automated tests). Defaults to False.

    Example:
        >>> class MyRegistration(RegisterImagesBase):
        ...     def registration_method(self, moving_image, **kwargs):
        ...         # Implement specific registration algorithm
        ...         return {
        ...             'fixed_to_moving_transform': tfm_f2m,  # warps moving image -> fixed grid
        ...             'moving_to_fixed_transform': tfm_m2f,  # warps fixed image -> moving grid
        ...             'loss': 0.0,
        ...         }
        >>>
        >>> registrar = MyRegistration()
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(reference_image)
        >>> result = registrar.register(moving_image)
        >>> f2m_tfm = result['fixed_to_moving_transform']  # warps moving image -> fixed grid
        >>> m2f_tfm = result['moving_to_fixed_transform']  # warps fixed image -> moving grid

    See :class:`RegisterImagesChain` to combine multiple registrars into a
    multi-stage pipeline (e.g. a fast coarse registrar followed by a
    refinement stage), and :class:`RegisterImagesGreedyICON` for the common
    Greedy-then-ICON case.
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the base image registration class.

        Sets up the common registration parameters with default values. Algorithm-specific
        components (like neural networks or optimization objects) should be initialized
        in the concrete implementation to avoid unnecessary resource allocation.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.labelmap_tools = LabelmapTools(log_level=log_level)

        self.net: Any = None

        self.modality: str = "ct"

        self.fixed_image: Optional[itk.Image] = None
        self.fixed_image_pre: Optional[itk.Image] = None
        self.fixed_mask: Optional[itk.Image] = None
        self.fixed_labelmap: Optional[itk.Image] = None

        self.moving_image: Optional[itk.Image] = None
        self.moving_image_pre: Optional[itk.Image] = None
        self.moving_mask: Optional[itk.Image] = None
        self.moving_labelmap: Optional[itk.Image] = None

        self.mask_dilation_mm: float = 5.0
        self.prewarp_background_value: Optional[float] = None

        self.fast_mode: bool = False

        self.fixed_to_moving_transform: Optional[itk.Transform] = None
        self.moving_to_fixed_transform: Optional[itk.Transform] = None
        self.loss: Optional[float] = None
        self.moving_image_registered: Optional[itk.Image] = None

    def set_modality(self, modality: str) -> None:
        """Set the imaging modality for registration optimization.

        Different imaging modalities benefit from different registration
        parameters. CT images.

        Args:
            modality (str): The imaging modality.
                Supported values: 'ct', 'mri'

        Example:
            >>> registrar.set_modality('ct')
            >>> registrar.set_modality('mri')
        """
        self.modality = modality

    def set_prewarp_background_value(self, background_value: float) -> None:
        """Override the value a seeded registration's pre-warp writes off-grid.

        Args:
            background_value: Intensity written where the fixed grid samples
                outside the moving image. Leave unset to derive it from the
                modality; see :meth:`_prewarp_background_value`.
        """
        self.prewarp_background_value = background_value

    def _prewarp_background_value(self, moving_image: itk.Image) -> float:
        """Return the intensity that means "no tissue" for the moving image.

        Pre-warping onto the fixed grid samples outside the moving image
        wherever the two extents disagree. ITK's default fill of 0 is wrong for
        an intensity image: in CT it is water, so the filled region reads as
        soft tissue rather than air and any downstream similarity metric treats
        it as structure to match. -1000 HU is also uniGradICON's ``ct_window``
        lower bound, so it normalizes to exactly the same value as true air.

        Args:
            moving_image: Image being pre-warped.

        Returns:
            The explicit override when set, -1000.0 for CT, otherwise the
            moving image's own minimum intensity.
        """
        if self.prewarp_background_value is not None:
            return self.prewarp_background_value
        if self.modality == "ct":
            return -1000.0
        return float(np.min(itk.GetArrayViewFromImage(moving_image)))

    def set_fixed_image(self, fixed_image: itk.Image) -> None:
        """Set the fixed/target image for registration.

        The fixed image serves as the reference coordinate system to which
        all moving images will be aligned. Setting a new fixed image clears
        any preprocessed data to ensure consistency.

        Args:
            fixed_image (itk.image): The 3D reference image that serves as
                the target for registration

        Example:
            >>> registrar.set_fixed_image(reference_frame)
        """
        self.fixed_image = fixed_image
        self.fixed_image_pre = None
        self.fixed_to_moving_transform = None
        self.moving_to_fixed_transform = None
        self.loss = None
        self.moving_image_registered = None

    def set_mask_dilation(self, mask_dilation_mm: float) -> None:
        """Set the dilation of the fixed and moving image masks.

        Args:
            mask_dilation_mm (float): The dilation in millimeters.
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_fixed_mask(self, fixed_mask: Optional[itk.Image]) -> None:
        """Set a binary mask for the fixed image region of interest.

        The mask constrains registration to focus on specific anatomical
        regions, improving accuracy and reducing computation time. The mask
        is automatically converted to binary format. If mask_dilation_mm is set,
        the mask is dilated by the specified amount.

        Args:
            fixed_mask (itk.image): Binary or label mask defining the
                region of interest in the fixed image. Non-zero values are
                treated as foreground

        Example:
            >>> # Use heart mask to focus registration on cardiac structures
            >>> registrar.set_fixed_mask(heart_mask)
        """
        self.fixed_image_pre = None
        self.fixed_to_moving_transform = None
        self.moving_to_fixed_transform = None
        self.loss = None
        self.moving_image_registered = None

        if fixed_mask is None:
            self.fixed_mask = None
            return

        if self.fixed_image is None:
            raise ValueError("Fixed image must be set before setting a fixed mask.")

        self.fixed_mask = self.labelmap_tools.convert_labelmap_to_mask(
            fixed_mask, dilation_in_mm=self.mask_dilation_mm
        )
        self.fixed_mask.CopyInformation(self.fixed_image)

    def set_fixed_labelmap(self, fixed_labelmap: Optional[itk.Image]) -> None:
        """Set the fixed image labelmap (multi-label segmentation).

        Args:
            fixed_labelmap (itk.Image, optional): Multi-label segmentation
                co-registered with the fixed image, or None to clear.
        """
        self.fixed_labelmap = fixed_labelmap
        self.fixed_to_moving_transform = None
        self.moving_to_fixed_transform = None
        self.loss = None
        self.moving_image_registered = None

    def preprocess(self, image: itk.Image, modality: str = "ct") -> itk.Image:
        """Preprocess the image based on modality-specific requirements.

        This method applies preprocessing steps such as intensity normalization,
        histogram equalization, or noise reduction tailored to the specified
        imaging modality. Preprocessing enhances image quality and improves
        registration accuracy.

        Args:
            image (itk.image): The 3D image to preprocess
            modality (str): The imaging modality ('ct', 'mri', etc.)

        Returns:
            itk.image: The preprocessed image

        Example:
            >>> preprocessed_image = registrar.preprocess(raw_image, modality='ct')
        """
        # Placeholder implementation - override in subclass if needed
        return image

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Main registration method to align moving image to fixed image.

        This method serves as the primary interface for performing image
        registration. It takes a moving image and optional mask and
        preprocessed image, and returns the forward and backward transformations.

        Note: This is an internal method that should be implemented by subclasses.
        The public API is register() which wraps this method.

        Args:
            moving_image (itk.image): The 3D image to be registered to the fixed image
            moving_mask (itk.image, optional): Binary mask for moving image ROI
            moving_labelmap (itk.image, optional): Multi-label segmentation for the moving image
            moving_image_pre (itk.image, optional): Preprocessed moving image

        Returns:
            dict: Dictionary containing:
                - "fixed_to_moving_transform": Warps the moving image onto the
                  fixed grid. Warping moving points/landmarks into fixed space
                  uses "moving_to_fixed_transform" instead (see register() and
                  docs/developer/transform_conventions).
                - "moving_to_fixed_transform": Warps the fixed image onto the
                  moving grid
                - "loss": Registration loss/metric value

        Raises:
            ValueError: If fixed image is not set
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def register(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Register a moving image to the fixed image.

        This is the main registration method that must be implemented by
        concrete subclasses. It should align the moving image to the fixed
        image using the specific algorithm implemented by the subclass.

        To start from a known alignment, use :meth:`register_from` rather than
        seeding the backend directly.

        Args:
            moving_image (itk.image): The 3D image to be registered to the fixed image
            moving_mask (itk.image, optional): Binary mask for moving image ROI
            moving_labelmap (itk.image, optional): Multi-label segmentation for the moving image
            moving_image_pre (itk.image, optional): Preprocessed moving image

        Returns:
            dict: Dictionary containing transformation results:
                - "fixed_to_moving_transform": Warps the moving IMAGE onto the
                  fixed grid, i.e.
                  transform_image(moving, fixed_to_moving_transform, fixed).
                - "moving_to_fixed_transform": Warps the fixed IMAGE onto the
                  moving grid, i.e.
                  transform_image(fixed, moving_to_fixed_transform, moving).
                - "loss": Registration loss/metric value

        Note:
            Image warps and point/landmark warps use OPPOSITE members of the
            transform pair, because ITK image resampling pulls back (it maps a
            fixed-grid sample to the moving image) while point transforms push
            forward (they map a point to its corresponding location):

            - Warp the moving image into fixed space  -> fixed_to_moving_transform
            - Warp moving points/landmarks into fixed  -> moving_to_fixed_transform
            - Warp the fixed image into moving space   -> moving_to_fixed_transform
            - Warp fixed points/landmarks into moving   -> fixed_to_moving_transform

            See docs/developer/transform_conventions for the full discussion.

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        self.moving_image_registered = None
        self.fixed_to_moving_transform = None
        self.moving_to_fixed_transform = None
        self.loss = None

        if self.fixed_image_pre is None:
            self.fixed_image_pre = self.preprocess(
                self.fixed_image,
                modality=self.modality,
            )

        if moving_image_pre is None:
            moving_image_pre = self.preprocess(
                moving_image,
                modality=self.modality,
            )

        new_moving_mask = moving_mask
        if moving_mask is not None:
            new_moving_mask = self.labelmap_tools.convert_labelmap_to_mask(
                moving_mask, dilation_in_mm=self.mask_dilation_mm
            )
            new_moving_mask.CopyInformation(moving_image)

        self.moving_image = moving_image
        self.moving_image_pre = moving_image_pre
        self.moving_mask = new_moving_mask
        self.moving_labelmap = moving_labelmap

        result = self.registration_method(
            moving_image,
            moving_mask=new_moving_mask,
            moving_labelmap=moving_labelmap,
            moving_image_pre=moving_image_pre,
        )

        self.fixed_to_moving_transform = result["fixed_to_moving_transform"]
        self.moving_to_fixed_transform = result["moving_to_fixed_transform"]
        self.loss = result["loss"]

        return {
            "fixed_to_moving_transform": self.fixed_to_moving_transform,
            "moving_to_fixed_transform": self.moving_to_fixed_transform,
            "loss": self.loss,
        }

    def register_from(
        self,
        initial_fixed_to_moving_transform: itk.Transform,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Register starting from a known alignment.

        The moving data is warped onto the fixed grid by
        ``initial_fixed_to_moving_transform`` first, :meth:`register` then
        measures only the residual misalignment, and the two are composed.
        This is the single supported way to seed a registration: doing it
        here rather than inside each backend keeps the pre-warp, the
        composition and the inversion identical for every algorithm.

        The image, the mask and the labelmap are all pre-warped, so they stay in
        the same frame as each other; the mask and labelmap use nearest-neighbor
        interpolation to preserve their discrete values.

        Args:
            initial_fixed_to_moving_transform: Starting alignment, in the same
                convention as the returned ``fixed_to_moving_transform`` -- it
                warps the moving image onto the fixed grid.
            moving_image: The 3D image to be registered to the fixed image.
            moving_mask: Binary mask for the moving image ROI.
            moving_labelmap: Multi-label segmentation for the moving image.

        Returns:
            dict: Same keys as :meth:`register`, with the transforms composed so
            they map between the *original* moving image and the fixed image.

        Raises:
            ValueError: If the fixed image has not been set.
        """
        warped_image, warped_mask, warped_labelmap = self._prewarp_moving(
            initial_fixed_to_moving_transform,
            moving_image,
            moving_mask,
            moving_labelmap,
        )
        result = self.register(
            warped_image,
            moving_mask=warped_mask,
            moving_labelmap=warped_labelmap,
        )
        composed = self._compose_with_initial(
            initial_fixed_to_moving_transform, result, moving_image
        )

        # register() left the pre-warped image on self; the composed transforms
        # are defined against the original, so restore it and drop any
        # registered-image cache built for the pre-warped one.
        self.moving_image = moving_image
        self.moving_image_registered = None

        self.fixed_to_moving_transform = composed["fixed_to_moving_transform"]
        self.moving_to_fixed_transform = composed["moving_to_fixed_transform"]
        self.loss = composed["loss"]
        return composed

    def _prewarp_moving(
        self,
        initial_fixed_to_moving_transform: itk.Transform,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image],
        moving_labelmap: Optional[itk.Image],
    ) -> tuple[itk.Image, Optional[itk.Image], Optional[itk.Image]]:
        """Warp the moving image, mask and labelmap onto the fixed grid.

        Args:
            initial_fixed_to_moving_transform: Alignment to apply, in the
                image-warp convention.
            moving_image: Raw moving image.
            moving_mask: Moving mask, or None.
            moving_labelmap: Moving labelmap, or None.

        Returns:
            Tuple of the warped ``(image, mask, labelmap)``, the latter two None
            when not supplied. The mask and labelmap are warped with
            nearest-neighbor interpolation to keep their discrete values, and
            filled with 0 off-grid; the image is filled with
            :meth:`_prewarp_background_value` instead, since 0 is a tissue
            intensity rather than an absence of tissue.

        Raises:
            ValueError: If the fixed image has not been set.
        """
        if self.fixed_image is None:
            raise ValueError("Fixed image must be set before registration.")

        transform_tools = TransformTools()
        background_value = self._prewarp_background_value(moving_image)
        self.log_info(
            "Pre-warping moving data with the initial transform (background %.1f)...",
            background_value,
        )

        def _warp(image: Optional[itk.Image], nearest: bool) -> Optional[itk.Image]:
            if image is None:
                return None
            return transform_tools.transform_image(
                image,
                initial_fixed_to_moving_transform,
                self.fixed_image,
                interpolation_method="nearest" if nearest else "linear",
                background_value=0.0 if nearest else background_value,
            )

        return (
            _warp(moving_image, nearest=False),
            _warp(moving_mask, nearest=True),
            _warp(moving_labelmap, nearest=True),
        )

    def _compose_with_initial(
        self,
        initial_fixed_to_moving_transform: itk.Transform,
        result: dict[str, Union[itk.Transform, float]],
        moving_image: itk.Image,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Compose a residual registration result onto its initial transform.

        Args:
            initial_fixed_to_moving_transform: The alignment the moving data
                was pre-warped by.
            result: Result of registering the pre-warped data.
            moving_image: Raw moving image, whose grid defines the domain the
                initial transform is inverted over.

        Returns:
            The result dict with both transforms mapping between the *original*
            moving image and the fixed image. ``loss`` is passed through
            unchanged, so it is the residual stage's loss measured against the
            already pre-warped data -- not a loss for the composed transform,
            and not comparable to the loss of a stage that started from scratch.
        """
        transform_tools = TransformTools()

        # The registration measured the residual from the pre-warped position,
        # so the total is the initial transform followed by that residual. An
        # itk.CompositeTransform applies its transforms in reverse order of
        # addition, so adding the initial first makes the residual apply first --
        # which is what the image-warp direction needs: a fixed-grid sample is
        # mapped by the residual, then by the initial transform, to land in the
        # original moving image.
        fixed_to_moving_transform = itk.CompositeTransform[itk.D, 3].New()
        self._add_transform_flattened(
            fixed_to_moving_transform, initial_fixed_to_moving_transform
        )
        self._add_transform_flattened(
            fixed_to_moving_transform,
            cast(itk.Transform, result["fixed_to_moving_transform"]),
        )

        # The inverse runs the other way -- a moving-grid sample is mapped by the
        # initial transform's inverse into the pre-warped frame, then by the
        # residual's inverse into the fixed image -- so the additions are
        # reversed too.
        initial_inverse = transform_tools.invert_transform(
            initial_fixed_to_moving_transform, moving_image
        )
        moving_to_fixed_transform = itk.CompositeTransform[itk.D, 3].New()
        self._add_transform_flattened(
            moving_to_fixed_transform,
            cast(itk.Transform, result["moving_to_fixed_transform"]),
        )
        self._add_transform_flattened(moving_to_fixed_transform, initial_inverse)

        return {
            "fixed_to_moving_transform": fixed_to_moving_transform,
            "moving_to_fixed_transform": moving_to_fixed_transform,
            "loss": result["loss"],
        }

    @staticmethod
    def _add_transform_flattened(
        composite: itk.CompositeTransform, transform: itk.Transform
    ) -> None:
        """Append a transform to a composite, splicing in nested composites.

        itk.HDF5TransformIO refuses to write a CompositeTransform that holds
        another CompositeTransform ("Composite Transform can only be 1st
        transform in a file"), which every multi-stage registration would
        otherwise produce: RegisterImagesGreedy already returns an affine+warp
        composite, and composing a residual onto it would nest that composite.

        Splicing the sub-transforms in at the position their composite occupied
        leaves the mapping unchanged, since itk.CompositeTransform applies its
        queue back to front either way.

        The down_cast is required: ITK hands back base-typed ``itkTransformD33``
        Python objects from ``GetInverseTransform()`` and ``GetNthTransform()``,
        which carry none of CompositeTransform's methods.
        """
        transform = itk.down_cast(transform)
        if isinstance(transform, itk.CompositeTransform[itk.D, 3]):
            for i in range(transform.GetNumberOfTransforms()):
                composite.AddTransform(transform.GetNthTransform(i))
        else:
            composite.AddTransform(transform)

    def _delegate_to(
        self,
        other: "RegisterImagesBase",
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image],
        moving_labelmap: Optional[itk.Image],
    ) -> None:
        """Prepare ``other`` to run a standalone ``registration_method()`` call.

        Uses direct attribute assignment for fixed_image/fixed_mask/
        fixed_labelmap (not the public setters), since ``self.fixed_mask``
        is already the dilated/converted mask produced by
        :meth:`set_fixed_mask` -- calling it again on ``other`` would
        re-dilate it. ``moving_mask`` is similarly already converted by the
        outer :meth:`register` call and is passed through unchanged, since
        mask conversion is backend-independent.

        ``fixed_image_pre``/``moving_image_pre`` are deliberately NOT copied
        from this instance: unlike mask conversion, intensity preprocessing
        is backend-specific (e.g. ``RegisterImagesICON.preprocess()`` runs
        uniGradICON preprocessing; ``RegisterImagesGreedy`` does not), so a
        "pre" value this instance computed with its own (possibly no-op)
        ``preprocess()`` cannot be trusted for ``other``. Instead, ``other``
        computes its own ``fixed_image_pre`` via its own ``preprocess()``
        (cached, matching :meth:`register`'s own caching), and
        ``moving_image_pre`` is left unset so ``other.registration_method()``
        preprocesses the moving image itself.

        Args:
            other: The registrar to prepare for a delegated call.
            moving_image: Raw moving image for the delegated call.
            moving_mask: Already-converted moving mask, or None.
            moving_labelmap: Moving labelmap, or None.
        """
        other.modality = self.modality
        other.mask_dilation_mm = self.mask_dilation_mm
        # Recompute other.fixed_image_pre whenever the fixed image changes.
        # The identity check preserves per-frame caching (many moving frames
        # against one fixed image reuse the same pre) while preventing a stale
        # pre from a previous, different fixed image - which would silently
        # register against the wrong reference - when ``other`` is reused.
        if other.fixed_image is not self.fixed_image:
            other.fixed_image = self.fixed_image
            other.fixed_image_pre = None
        if other.fixed_image_pre is None:
            other.fixed_image_pre = other.preprocess(
                other.fixed_image, modality=other.modality
            )
        other.fixed_mask = self.fixed_mask
        other.fixed_labelmap = self.fixed_labelmap
        other.moving_image = moving_image
        other.moving_image_pre = None
        other.moving_mask = moving_mask
        other.moving_labelmap = moving_labelmap

    def _capture_delegate_result(
        self,
        other: "RegisterImagesBase",
        result: dict[str, Union[itk.Transform, float]],
    ) -> None:
        """Mirror a delegate's registration result back onto it as state.

        Matches what :meth:`register` would have set had it been called on
        ``other`` directly, so ``other.get_registered_image()`` still works
        afterward.

        Args:
            other: The registrar whose ``registration_method()`` produced
                ``result``.
            result: The dict returned by ``other.registration_method(...)``.
        """
        other.fixed_to_moving_transform = cast(
            itk.Transform, result["fixed_to_moving_transform"]
        )
        other.moving_to_fixed_transform = cast(
            itk.Transform, result["moving_to_fixed_transform"]
        )
        other.loss = cast(float, result["loss"])
        other.moving_image_registered = None

    def get_registered_image(self) -> itk.Image:
        """Get the registered image.

        The moving image is an intensity image, so voxels of the fixed grid that
        fall outside it are filled with :meth:`_prewarp_background_value` rather
        than 0, which is a tissue intensity rather than an absence of tissue.

        Returns:
            itk.Image: The registered image
        """
        if self.moving_image_registered is None:
            TfmTools = TransformTools()
            self.moving_image_registered = TfmTools.transform_image(
                self.moving_image,
                self.fixed_to_moving_transform,
                self.fixed_image,
                background_value=self._prewarp_background_value(self.moving_image),
            )
        return self.moving_image_registered
