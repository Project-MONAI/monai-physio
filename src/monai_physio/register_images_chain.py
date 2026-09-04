"""Composite registration: run multiple registrars in sequence.

This module provides the RegisterImagesChain class, which combines an
ordered list of RegisterImagesBase instances into a single multi-stage
registration pipeline.
"""

import logging
from typing import Optional, Union, cast

import itk

from .register_images_base import RegisterImagesBase


class RegisterImagesChain(RegisterImagesBase):
    """Run an ordered list of registrars in sequence, each stage refining the
    previous stage's forward_transform via
    :meth:`RegisterImagesBase.register_from`.

    Use this to combine independent registration backends into a multi-stage
    pipeline (e.g. a fast coarse registrar followed by a refinement stage).
    Every element of ``registrars`` must be a :class:`RegisterImagesBase`
    instance. ``registrars`` (plural, a list) is distinct from the singular
    ``registrar`` attribute used by classes like
    :class:`RegisterTimeSeriesImages`.

    See :class:`RegisterImagesGreedyICON` for a named 2-stage convenience
    subclass (Greedy followed by ICON refinement).

    Chaining is not free accuracy. Every stage's result is applied
    unconditionally, so a refinement stage helps only when its own accuracy
    floor is below the error the previous stage has already reached. A stage
    whose deformation model is coarser than that error cannot resolve what is
    left and acts as a low-pass perturbation, giving a slightly worse answer for
    strictly more runtime. Compare each stage against the one before it on a
    held-out metric rather than assuming the chain wins.

    ``result["loss"]`` is the *last* stage's loss, measured against data the
    earlier stages already warped; it is not comparable to a single-stage loss.

    Example:
        >>> chain = RegisterImagesChain([RegisterImagesGreedy(), RegisterImagesICON()])
        >>> chain.set_fixed_image(fixed_image)
        >>> result = chain.register(moving_image)
    """

    def __init__(
        self, registrars: list[RegisterImagesBase], log_level: int | str = logging.INFO
    ) -> None:
        """Initialize the registration chain.

        Args:
            registrars: Ordered, non-empty list of RegisterImagesBase
                instances to run in sequence.
            log_level: Logging level (default: logging.INFO)

        Raises:
            ValueError: If registrars is empty.
            TypeError: If any element of registrars is not a
                RegisterImagesBase instance.
        """
        super().__init__(log_level=log_level)

        if not registrars:
            raise ValueError("registrars must not be empty")
        for registrar in registrars:
            if not isinstance(registrar, RegisterImagesBase):
                raise TypeError(
                    "Every element of registrars must be a RegisterImagesBase "
                    f"instance, got {type(registrar).__name__}"
                )

        self.registrars = registrars

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Run each registrar in ``self.registrars`` in order.

        The first stage registers the raw moving image; every later stage sees
        the moving data pre-warped by the running result and contributes only a
        refinement, which is composed back on -- the same mechanics as
        :meth:`RegisterImagesBase.register_from`, run through the delegated
        ``registration_method`` path so masks are not re-converted per stage.

        Note:
            ``moving_image_pre`` is ignored: each stage may need different
            intensity preprocessing (e.g. ICON's uniGradICON preprocessing
            vs. Greedy's no-op), so every stage computes its own
            preprocessing from the raw ``moving_image`` rather than reuse a
            value computed for a different backend.

        Args:
            moving_image (itk.image): The 3D image to be registered
            moving_mask (itk.image, optional): Binary mask for moving image ROI
            moving_labelmap (itk.image, optional): Multi-label segmentation
                for the moving image
            moving_image_pre (itk.image, optional): Ignored - see Note above

        Returns:
            dict: The last stage's result dict (see :meth:`RegisterImagesBase.register`)
        """
        current_forward: Optional[itk.Transform] = None
        result: dict[str, Union[itk.Transform, float]] = {}
        for registrar in self.registrars:
            if current_forward is None:
                stage_image, stage_mask, stage_labelmap = (
                    moving_image,
                    moving_mask,
                    moving_labelmap,
                )
            else:
                stage_image, stage_mask, stage_labelmap = self._prewarp_moving(
                    current_forward, moving_image, moving_mask, moving_labelmap
                )

            self._delegate_to(registrar, stage_image, stage_mask, stage_labelmap)
            stage_result = registrar.registration_method(
                stage_image,
                moving_mask=stage_mask,
                moving_labelmap=stage_labelmap,
                moving_image_pre=None,
            )
            self._capture_delegate_result(registrar, stage_result)

            result = (
                stage_result
                if current_forward is None
                else self._compose_with_initial(
                    current_forward, stage_result, moving_image
                )
            )
            current_forward = cast(itk.Transform, result["forward_transform"])

        return result
