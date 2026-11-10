"""Tests for high-resolution 4D CT reconstruction configuration and composites."""

from __future__ import annotations

import itk
import numpy as np
import pytest

from monai_physio.register_images_base import RegisterImagesBase
from monai_physio.register_images_greedy import RegisterImagesGreedy
from monai_physio.register_images_icon import RegisterImagesICON
from monai_physio.workflow_reconstruct_highres_4d_ct import (
    WorkflowReconstructHighres4DCT,
)


def _small_image() -> itk.Image:
    """A tiny synthetic image, shape (Z, Y, X) = (3, 3, 3)."""
    return itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))


def test_default_registration_method_is_greedy() -> None:
    """Omitting registration_method defaults to RegisterImagesGreedy."""
    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=[_small_image(), _small_image()],
        reference_image=_small_image(),
    )
    assert isinstance(workflow.registrar.registrar, RegisterImagesGreedy)


def test_registration_method_rejects_wrong_type() -> None:
    """A non-RegisterImagesBase registration_method raises TypeError."""
    with pytest.raises(TypeError, match="registration_method must be"):
        WorkflowReconstructHighres4DCT(
            time_series_images=[_small_image(), _small_image()],
            reference_image=_small_image(),
            registration_method="ICON",  # type: ignore[arg-type]
        )


def test_caller_supplied_instance_is_used_as_is() -> None:
    """A caller-supplied registration_method instance is stored unmodified."""
    registrar: RegisterImagesBase = RegisterImagesICON()
    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=[_small_image(), _small_image()],
        reference_image=_small_image(),
        registration_method=registrar,
    )
    assert workflow.registrar.registrar is registrar


@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize(("percentile", "expected"), [(95.0, 29.0), (50.0, 20.0)])
def test_percentile_reconstruction(
    precompute: bool, percentile: float, expected: float
) -> None:
    """Synthetic 3-cubed images test cached and direct percentile reconstruction."""
    images = [itk.image_from_array(np.full((3, 3, 3), v, np.float32)) for v in (20, 30)]
    reference = itk.image_from_array(np.full((3, 3, 3), 10, np.float32))
    workflow = WorkflowReconstructHighres4DCT(images, reference)
    assert workflow.composite_mode == "reference"
    assert workflow.composite_percentile == 95.0
    if percentile == 95.0:
        workflow.set_composite_mode("percentile")
    else:
        workflow.set_composite_mode("percentile", percentile=percentile)
    workflow.set_upsample_to_fixed_resolution(False)
    workflow.registrar.set_fixed_image(reference)
    workflow.fixed_to_moving_transforms = [itk.IdentityTransform[itk.D, 3].New()] * 2
    workflow.moving_to_fixed_transforms = [itk.IdentityTransform[itk.D, 3].New()] * 2
    if precompute:
        composite = workflow.get_composite_reference_image()
        assert composite is not None
        np.testing.assert_allclose(itk.array_from_image(composite), expected)
    result = workflow.reconstruct_time_series()
    assert len(result["reconstructed_images"]) == 2
    for image in result["reconstructed_images"]:
        np.testing.assert_allclose(itk.array_from_image(image), expected)

    # Reconfiguring the percentile must discard the cached source image.
    workflow.set_composite_mode("percentile", percentile=0.0)
    assert workflow.composite_reference_image is None
    result = workflow.reconstruct_time_series()
    for image in result["reconstructed_images"]:
        np.testing.assert_allclose(itk.array_from_image(image), 10.0)


@pytest.mark.parametrize("percentile", [-1.0, 101.0, np.nan, np.inf, -np.inf])
def test_percentile_configuration_rejects_invalid_values(percentile: float) -> None:
    """Invalid percentiles fail at configuration time on synthetic 3-cubed images."""
    workflow = WorkflowReconstructHighres4DCT([_small_image()], _small_image())
    with pytest.raises(ValueError, match="percentile"):
        workflow.set_composite_mode("percentile", percentile=percentile)
