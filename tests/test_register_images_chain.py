"""Unit tests for RegisterImagesChain's sequencing and state-propagation.

These are pure plumbing tests using stub RegisterImagesBase subclasses -
no real registration is performed. RegisterImagesChain composes independent
backends that may each need different intensity preprocessing, so the
chain's job is dict/attribute routing, not numeric accuracy; per this
project's testing conventions, that's exactly the kind of behavior
appropriate for synthetic/stub-based unit tests rather than real data.
"""

from __future__ import annotations

from typing import Any, Optional, Union, cast

import itk
import numpy as np
import pytest

from monai_physio.register_images_base import RegisterImagesBase
from monai_physio.register_images_chain import RegisterImagesChain


def _small_image(value: float = 0.0) -> itk.Image:
    """A tiny constant-valued synthetic image, shape (Z, Y, X) = (4, 4, 4)."""
    arr = np.full((4, 4, 4), value, dtype=np.float32)
    return itk.image_from_array(arr)


class _RecordingRegistrar(RegisterImagesBase):
    """Stub registrar that records the state visible to it when
    registration_method() is called, and returns a sentinel transform."""

    def __init__(self, name: str, sentinel_value: float) -> None:
        """Args: name: label used in the sentinel transform strings.
        sentinel_value: loss value this stub's registration_method() returns.
        """
        super().__init__()
        self.name = name
        self.sentinel_value = sentinel_value
        self.seen_fixed_image_pre: Optional[itk.Image] = None
        self.seen_moving_image: Optional[itk.Image] = None
        self.seen_moving_image_pre: Optional[itk.Image] = None
        self.seen_moving_mask: Optional[itk.Image] = None
        self.preprocess_call_count = 0

    def preprocess(self, image: itk.Image, modality: str = "ct") -> itk.Image:
        """No-op preprocessing that records how many times it was called."""
        self.preprocess_call_count += 1
        return image

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[object, float]]:
        """Record the state visible at call time; return a sentinel result.

        The sentinel transforms are real translations along x of
        ``sentinel_value`` mm, so a chained result can be checked numerically.
        """
        self.seen_fixed_image_pre = self.fixed_image_pre
        self.seen_moving_image = self.moving_image
        self.seen_moving_image_pre = moving_image_pre
        self.seen_moving_mask = moving_mask
        forward = itk.TranslationTransform[itk.D, 3].New()
        forward.SetOffset([self.sentinel_value, 0.0, 0.0])
        inverse = itk.TranslationTransform[itk.D, 3].New()
        inverse.SetOffset([-self.sentinel_value, 0.0, 0.0])
        return {
            "forward_transform": forward,
            "inverse_transform": inverse,
            "loss": self.sentinel_value,
        }


def test_chain_refines_previous_stage_result() -> None:
    """Stage 2 must refine stage 1's alignment rather than start over.

    Stage 1 is called on the raw moving image; stage 2 sees a moving image
    pre-warped by stage 1's forward transform, and the chain's composed
    forward transform is the sum of both stages' translations.
    """
    stage1 = _RecordingRegistrar("stage1", 1.0)
    stage2 = _RecordingRegistrar("stage2", 2.0)
    chain = RegisterImagesChain([stage1, stage2])
    chain.set_fixed_image(_small_image())
    moving = _small_image()

    result = chain.register(moving)

    assert stage1.seen_moving_image is moving
    # Stage 2 registers the pre-warped image, not the caller's image.
    assert stage2.seen_moving_image is not moving
    composed = cast(itk.Transform, result["forward_transform"])
    assert list(composed.TransformPoint([0.0, 0.0, 0.0])) == [3.0, 0.0, 0.0]
    assert result["loss"] == 2.0


class _CompositeRegistrar(_RecordingRegistrar):
    """Stub registrar returning a CompositeTransform, as RegisterImagesGreedy
    does (its result is an affine plus a displacement field)."""

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[object, float]]:
        """Wrap the parent's translations in single-entry composites."""
        result = super().registration_method(
            moving_image, moving_mask, moving_labelmap, moving_image_pre
        )
        for key in ("forward_transform", "inverse_transform"):
            composite = itk.CompositeTransform[itk.D, 3].New()
            composite.AddTransform(cast(itk.Transform, result[key]))
            result[key] = composite
        return result


def test_chain_result_holds_no_nested_composite(tmp_path: Any) -> None:
    """A stage returning a CompositeTransform must not end up nested.

    itk.HDF5TransformIO refuses to write a CompositeTransform that holds
    another one, so a chain over Greedy (which returns affine+warp composites)
    would produce transforms that cannot be saved.
    """
    stage1 = _CompositeRegistrar("stage1", 1.0)
    stage2 = _CompositeRegistrar("stage2", 2.0)
    chain = RegisterImagesChain([stage1, stage2])
    chain.set_fixed_image(_small_image())

    result = chain.register(_small_image())

    for key in ("forward_transform", "inverse_transform"):
        composed = cast(itk.Transform, result[key])
        assert isinstance(composed, itk.CompositeTransform[itk.D, 3])
        for i in range(composed.GetNumberOfTransforms()):
            sub = composed.GetNthTransform(i)
            assert "Composite" not in sub.GetNameOfClass()
        itk.transformwrite(composed, str(tmp_path / f"{key}.hdf"))

    # Splicing the sub-transforms in must leave the mapping unchanged.
    forward = cast(itk.Transform, result["forward_transform"])
    assert list(forward.TransformPoint([0.0, 0.0, 0.0])) == [3.0, 0.0, 0.0]


def test_chain_propagates_fixed_and_moving_state_to_each_child() -> None:
    """Every child must see a non-None fixed_image_pre (computed via its own
    preprocess(), not copied from the chain's no-op preprocess()) and the
    raw moving_image, before its registration_method() runs."""
    stage1 = _RecordingRegistrar("stage1", 1.0)
    stage2 = _RecordingRegistrar("stage2", 2.0)
    chain = RegisterImagesChain([stage1, stage2])
    fixed = _small_image(value=5.0)
    moving = _small_image(value=7.0)
    chain.set_fixed_image(fixed)

    chain.register(moving)

    for stage in (stage1, stage2):
        assert stage.seen_fixed_image_pre is not None
        assert stage.seen_moving_image is not None
        # Each stage must compute its own preprocessing (moving_image_pre is
        # not inherited from the chain, which has no meaningful preprocess()
        # of its own).
        assert stage.seen_moving_image_pre is None
        assert stage.preprocess_call_count == 1
    # The first stage gets the caller's image; later stages get it pre-warped
    # by the running result.
    assert stage1.seen_moving_image is moving
    assert stage2.seen_moving_image is not moving


def test_chain_recomputes_fixed_image_pre_when_fixed_image_changes() -> None:
    """A reused chain must recompute each child's fixed_image_pre when the
    chain's fixed image changes, not reuse a stale pre from the prior image;
    but it must NOT recompute across frames that share the same fixed image."""
    stage = _RecordingRegistrar("stage", 1.0)
    chain = RegisterImagesChain([stage])

    fixed_a = _small_image(value=1.0)
    chain.set_fixed_image(fixed_a)
    chain.register(_small_image(value=9.0))
    # The stub's preprocess() is identity, so fixed_image_pre is fixed_a.
    assert stage.fixed_image_pre is fixed_a
    assert stage.preprocess_call_count == 1

    # Second frame, same fixed image: pre is cached, not recomputed.
    chain.register(_small_image(value=8.0))
    assert stage.preprocess_call_count == 1

    # New fixed image: the stale pre for fixed_a must be discarded and recomputed.
    fixed_b = _small_image(value=2.0)
    chain.set_fixed_image(fixed_b)
    chain.register(_small_image(value=7.0))
    assert stage.fixed_image_pre is fixed_b
    assert stage.preprocess_call_count == 2


def test_chain_rejects_empty_list() -> None:
    """An empty registrars list is not a valid chain."""
    with pytest.raises(ValueError, match="registrars must not be empty"):
        RegisterImagesChain([])


def test_chain_rejects_non_register_images_base_element() -> None:
    """Every element must be a RegisterImagesBase instance."""
    with pytest.raises(TypeError, match="RegisterImagesBase"):
        bad_registrars: list[Any] = [_RecordingRegistrar("a", 0.0), "not-a-registrar"]
        RegisterImagesChain(bad_registrars)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
