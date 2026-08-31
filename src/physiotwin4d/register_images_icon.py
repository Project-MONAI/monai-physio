# ruff: noqa: F821  -- torch/icon_registration lazy-loaded; "torch.Size" annotations are intentional
"""Icon-based image registration implementation.

This module provides the RegisterImagesICON class, a concrete implementation of
RegisterImagesBase that uses the Icon (Inverse Consistent Image Registration)
algorithm with deep learning models. It supports both masked and unmasked
registration for aligning medical images, particularly useful for 4D cardiac CT registration.

The module uses the unigradicon package which provides GPU-accelerated
deformable registration with mass preservation constraints.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import itk
import numpy as np

from .register_images_base import RegisterImagesBase


def _load_icon():
    """Lazy-load icon_registration, torch, and unigradicon to avoid
    initializing GPU/CUDA resources at import time."""
    import icon_registration as icon
    import icon_registration.itk_wrapper
    import torch
    import torch.nn.functional as F
    from unigradicon import get_multigradicon, get_unigradicon
    from unigradicon import preprocess as unigradicon_preprocess

    return (
        icon,
        icon.itk_wrapper,
        torch,
        F,
        get_multigradicon,
        get_unigradicon,
        unigradicon_preprocess,
    )


class RegisterImagesICON(RegisterImagesBase):
    """ICON-based deformable image registration implementation.

    This class extends RegisterImagesBase to provide GPU-accelerated deformable
    image registration using the ICON (Inverse Consistent Image Registration)
    algorithm implemented with deep learning models. It supports both full image
    registration and mask-constrained registration for specific anatomical regions.

    The ICON algorithm ensures inverse consistency, meaning the forward and
    backward transformations are true inverses of each other. This is important
    for maintaining spatial relationships and avoiding registration artifacts.

    ICON-specific features:

    - GPU acceleration using UniGradIcon framework
    - Mass preservation
    - LNCC (Local Normalized Cross Correlation) similarity metric
    - Inverse consistent transformations
    - Per-registration finetuning, 50 optimization steps by default
      (override with set_number_of_iterations())

    Inherits from RegisterImagesBase:

    - Fixed and moving image management
    - Binary mask processing with optional dilation
    - Modality-specific parameter configuration
    - Standardized registration interface

    Attributes:
        net (unigradicon model): The ICON deep learning registration network

    Example:
        >>> registrar = RegisterImagesICON()
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(reference_image)
        >>> result = registrar.register(moving_image)
        >>> forward_transform = result['forward_transform']
    """

    # Networks already built in this process, keyed by everything that changes
    # what the network *is*.  Shared across instances rather than held per
    # instance: building one costs four 3D U-Nets constructed on the CPU, a
    # second host copy of the checkpoint through ``torch.load``, and recursive
    # identity maps, all before the move to the GPU.  Workflows construct a
    # registrar per frame, so without this the same network is rebuilt hundreds
    # of times in a cohort run, and the freed host arenas are not necessarily
    # returned to the OS -- which is how a long run walks into the Linux OOM
    # killer.
    _net_cache: dict[tuple[Optional[str], bool, bool], Any] = {}

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the ICON image registration class.

        Calls the parent RegisterImagesBase constructor to set up common parameters.
        The ICON deep learning network is initialized lazily on first use to avoid
        unnecessary GPU memory allocation.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.net = None
        self.number_of_iterations: Optional[int] = 50
        self.use_multi_modality: bool = False
        self.use_mass_preservation: bool = False
        self.weights_path: Optional[str] = None

    @classmethod
    def clear_net_cache(cls) -> None:
        """Drop every network built so far in this process.

        Needed only to reclaim the GPU memory they hold, or to force a rebuild
        after a checkpoint file is overwritten in place.
        """
        cls._net_cache.clear()

    def set_weights_path(self, weights_path: str) -> None:
        """Set a custom weights file for the uniGradICON network.

        Use this to load a finetuned checkpoint instead of the default
        pretrained weights. Clears any previously loaded network so the new
        weights are applied on the next call to register().

        The file must already exist.  uniGradICON treats a missing
        ``weights_location`` as a download destination and silently fetches the
        stock pretrained weights into it, so an unvalidated path yields a
        stock-weights registration that looks like a finetuned one.

        Args:
            weights_path: Path to an existing uniGradICON checkpoint, e.g.
                "results/duke_4d_finetune/checkpoints/network_weights_final.trch"

        Raises:
            FileNotFoundError: If weights_path does not exist.
        """
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"uniGradICON weights not found: {weights_path}.  Leave the "
                "weights path unset to use the stock pretrained weights."
            )
        self.weights_path = weights_path
        self.net = None  # force reload on next register() call

    def set_number_of_iterations(self, number_of_iterations: Optional[int]) -> None:
        """Set the number of iterations for ICON registration.

        Args:
            number_of_iterations: Number of finetuning steps for ICON registration
        """
        self.number_of_iterations = number_of_iterations

    def set_multi_modality(self, enable: bool) -> None:
        """Enable or disable multi-modality registration.

        Multi-modality registration is useful when aligning images from different
        imaging modalities (e.g., CT to MRI). Enabling this option adjusts the
        registration parameters to better handle differences in intensity
        distributions and contrast between modalities.

        Args:
            enable (bool): True to enable multi-modality registration, False to disable

        Example:
            >>> registrar.set_multi_modality(True)  # Enable for CT to MRI
            >>> registrar.set_multi_modality(False)  # Disable for CT to CT
        """
        self.use_multi_modality = enable

    def set_mass_preservation(self, enable: bool) -> None:
        """Enable or disable mass preservation constraint.

        Mass preservation is particularly useful for CT images where the
        intensity values correspond to physical tissue densities. Enabling
        this constraint helps maintain realistic intensity distributions
        during registration.

        Args:
            enable (bool): True to enable mass preservation, False to disable

        Example:
            >>> registrar.set_mass_preservation(True)  # Enable for CT
            >>> registrar.set_mass_preservation(False)  # Disable for MRI
        """
        self.use_mass_preservation = enable

    def preprocess(self, image: itk.Image, modality: str = "ct") -> itk.Image:
        """Preprocess the image for ICON registration.

        Applies modality-specific preprocessing steps to prepare the image
        for registration. This may include intensity normalization, bias
        field correction, and resampling.

        Args:
            image (itk.image): The input 3D image to preprocess
            modality (str): The imaging modality ('ct', 'mri', etc.)

        Returns:
            itk.image: The preprocessed image

        Example:
            >>> preprocessed_image = registrar.preprocess(raw_image, modality='ct')
        """
        _, _, _, _, _, _, unigradicon_preprocess = _load_icon()
        return unigradicon_preprocess(image, modality=modality)

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Register moving image to fixed image using ICON registration algorithm.

        Implementation of the abstract register() method from RegisterImagesBase.
        Performs deformable registration to align the moving image with the
        fixed image using the ICON algorithm. The method automatically handles
        preprocessing, network initialization, and applies the computed transformation.

        Args:
            moving_image (itk.image): The 3D image to be registered/aligned
            moving_mask (itk.image, optional): Binary mask defining the
                region of interest in the moving image. If provided along with
                fixed_mask, enables mask-constrained registration
            moving_image_pre (itk.image, optional): Pre-processed moving image.
                If None, preprocessing is performed automatically

        Returns:
            dict: Dictionary containing:
                - "forward_transform": Warps the moving image onto the fixed
                  grid (warping moving points/landmarks into fixed space uses
                  "inverse_transform" instead -- image and point warps use
                  opposite transforms; see
                  docs/developer/transform_conventions)
                - "inverse_transform": Warps the fixed image onto the moving grid
                - "loss": Loss value from the registration

        Note:
            The transformations are inverse consistent, meaning
            forward_transform is approximately inverse(inverse_transform).
            Use forward_transform to warp the moving image onto the fixed grid,
            and inverse_transform to warp the fixed image onto the moving grid.
            Point/landmark warps use the opposite transform from image warps
            (see docs/developer/transform_conventions).

        Implementation details:
            - Uses UniGradIcon with LNCC loss function
            - Optionally applies mass preservation
            - Performs number_of_iterations finetuning steps per registration
              (passed to unigradicon as finetune_steps; 50 by default)
            - Supports both masked and unmasked registration modes

        Example:
            >>> # Basic registration
            >>> result = registrar.register(moving_image)
            >>> forward_transform = result['forward_transform']
            >>> inverse_transform = result['inverse_transform']
            >>>
            >>> # Masked registration for cardiac structures
            >>> registrar.set_fixed_mask(heart_mask_fixed)
            >>> result = registrar.register(moving_image, moving_mask=heart_mask_moving)
        """

        if moving_image_pre is None:
            moving_image_pre = self.preprocess(moving_image, self.modality)

        new_moving_image_pre = moving_image_pre

        # Prefer labelmap over binary mask when both sides have a labelmap.
        use_labelmaps = moving_labelmap is not None and self.fixed_labelmap is not None
        moving_effective_mask = moving_labelmap if use_labelmaps else moving_mask
        fixed_effective_mask = self.fixed_labelmap if use_labelmaps else self.fixed_mask

        self._ensure_net()

        inverse_transform = None
        forward_transform = None
        loss_artifacts = None
        _, icon_itk_wrapper, _, _, _, _, _ = _load_icon()
        if fixed_effective_mask is not None and moving_effective_mask is not None:
            inverse_transform, forward_transform, loss_artifacts = (
                icon_itk_wrapper.register_pair_with_mask(
                    self.net,
                    self.fixed_image_pre,
                    new_moving_image_pre,
                    fixed_effective_mask,
                    moving_effective_mask,
                    finetune_steps=self.number_of_iterations,
                    return_artifacts=True,
                )
            )
        else:
            inverse_transform, forward_transform, loss_artifacts = (
                icon_itk_wrapper.register_pair(
                    self.net,
                    self.fixed_image_pre,
                    new_moving_image_pre,
                    finetune_steps=self.number_of_iterations,
                    return_artifacts=True,
                )
            )

        loss = loss_artifacts[0]

        return {
            "forward_transform": forward_transform,
            "inverse_transform": inverse_transform,
            "loss": loss,
        }

    def _ensure_net(self) -> None:
        """Lazily instantiate the ICON network using current configuration.

        Honors set_weights_path() if a custom checkpoint has been requested,
        otherwise loads the default UniGradICON / MultiGradICON pretrained
        weights.
        """
        if self.net is not None:
            return

        # Reusing a network is sound because a registration leaves it as it
        # found it: ``finetune_execute`` deep-copies the state dict on entry and
        # restores it on exit, ``identity_map`` is a non-persistent buffer that
        # finetuning never touches, and the network stays in ``eval()`` so
        # BatchNorm statistics do not drift.
        #
        # Measured rather than assumed.  Over five runs each, the worst-probe
        # displacement between a reused network and a freshly built one had a
        # median of 0.0068 mm -- indistinguishable from the 0.0068 mm median
        # between two *freshly built* networks, whose own spread reaches
        # 0.0092 mm.  Reuse sits inside the method's own nondeterminism, and
        # ``tests/test_register_images_icon.py`` re-measures both distributions
        # rather than trusting a fixed tolerance.
        key = (self.weights_path, self.use_multi_modality, self.use_mass_preservation)
        cached = self._net_cache.get(key)
        if cached is not None:
            self.net = cached
            return

        icon, _, _, _, get_multigradicon, get_unigradicon, _ = _load_icon()
        build = get_multigradicon if self.use_multi_modality else get_unigradicon
        self.net = build(
            loss_fn=icon.LNCC(sigma=5),
            apply_intensity_conservation_loss=self.use_mass_preservation,
            weights_location=self.weights_path,
        )
        self._net_cache[key] = self.net

    def _image_to_resized_tensor(
        self, image: itk.Image, shape: "torch.Size"
    ) -> "torch.Tensor":
        """Convert an itk image to a torch tensor resized to the net's input grid.

        Mirrors the trilinear preprocessing path used by
        ``icon_registration.itk_wrapper.register_pair`` exactly.
        ``[None, None]`` prepends batch and channel singletons; ``shape`` is
        ``self.net.identity_map.shape`` (5D NCDHW).

        Notes:
            - Single-channel scalar inputs only; vector ``itk.Image`` inputs
              are not supported, matching ICON's own preprocessing.
            - 4D series must be split into 3D timepoints by the caller.
        """
        icon, _, torch, F, _, _, _ = _load_icon()
        arr = np.array(image)
        tensor = torch.Tensor(arr).to(icon.config.device)[None, None]
        return F.interpolate(
            tensor, size=shape[2:], mode="trilinear", align_corners=False
        )

    def _mask_to_resized_tensor(
        self, mask: itk.Image, shape: "torch.Size"
    ) -> "torch.Tensor":
        """Convert an itk mask image to a torch tensor resized via nearest-neighbor.

        Mirrors the mask preprocessing used by
        ``icon_registration.itk_wrapper.register_pair_with_mask`` exactly.
        ``[None, None]`` prepends batch and channel singletons; ``shape`` is
        ``self.net.identity_map.shape`` (5D NCDHW).

        Notes:
            - Single-channel mask inputs only; multi-label masks are scalar
              integer values, not channels. Vector ``itk.Image`` inputs are
              not supported.
            - ``mode='nearest'`` preserves label identities.
        """
        icon, _, torch, F, _, _, _ = _load_icon()
        arr = np.array(mask)
        tensor = torch.Tensor(arr).to(icon.config.device)[None, None]
        return F.interpolate(tensor, size=shape[2:], mode="nearest")
