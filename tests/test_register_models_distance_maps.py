"""Synthetic tests for the distance-map registration guard.

A constant image reaches ICON as a uniform volume and trips a bare assertion
inside ``icon_registration``'s ``register_pair`` that names neither which side
degenerated nor why. That happened on a real cardiac run: the Greedy affine
diverged, every sample landed outside the moving image, and the resampler filled
the grid with its background value.

These exercise the guard that turns that into a diagnosis. They call the check
directly on synthetic 4x4x4 ITK images, so they need no GPU, no ICON and no
data, and run in the default fast suite.
"""

from __future__ import annotations

import itk
import numpy as np
import pytest

from monai_physio.register_models_distance_maps import RegisterModelsDistanceMaps


def _image(values: np.ndarray) -> itk.Image:
    """Return a small ITK image carrying *values*."""
    return itk.GetImageFromArray(np.ascontiguousarray(values, dtype=np.float32))


def _varying_image() -> itk.Image:
    """Return an image with contrast, which is what a healthy pair looks like."""
    return _image(np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4))


def test_an_image_with_contrast_passes_untouched() -> None:
    """The guard is silent on a healthy image, so nothing else changes."""
    RegisterModelsDistanceMaps._require_non_constant(_varying_image(), "test image")


def test_a_constant_image_is_refused_and_named() -> None:
    """A uniform image raises, naming the value and which image it was."""
    constant = _image(np.full((4, 4, 4), 7.5, dtype=np.float32))

    with pytest.raises(RuntimeError, match="moving model's distance map") as caught:
        RegisterModelsDistanceMaps._require_non_constant(
            constant, "moving model's distance map"
        )

    assert "7.5" in str(caught.value), "The message should report the value found"


def test_a_diverged_affine_is_blamed_by_its_loss() -> None:
    """With a Greedy loss in hand, the guard explains the divergence.

    A loss at or near zero is the signature: no overlap left, so the resampler
    filled the grid with its background value. The message has to say that, and
    say that re-running is reasonable, because the stage is seeded
    nondeterministically and the callers cache their work.
    """
    background = _image(np.zeros((4, 4, 4), dtype=np.float32))

    with pytest.raises(RuntimeError) as caught:
        RegisterModelsDistanceMaps._require_non_constant(
            background,
            "moving distance map after the Greedy affine",
            greedy_loss=-0.0,
        )

    message = str(caught.value)
    assert "Greedy" in message, "The message should name the stage that diverged"
    assert "diverged" in message, "The message should say what a zero loss means"
    assert "Re-running" in message or "re-run" in message, (
        "The message should say the run resumes, since the failure is transient"
    )


def test_a_rasterization_failure_is_not_blamed_on_greedy() -> None:
    """Without a Greedy loss the guard points at the geometry instead.

    The same constant image means something different before any registration
    has run: nothing was rasterized. Blaming Greedy there would send the reader
    the wrong way.
    """
    constant = _image(np.zeros((4, 4, 4), dtype=np.float32))

    with pytest.raises(RuntimeError) as caught:
        RegisterModelsDistanceMaps._require_non_constant(
            constant, "fixed model's distance map"
        )

    message = str(caught.value)
    assert "Greedy" not in message, "Nothing diverged; do not name a stage"
    assert "reference image" in message, "Point at the geometry that produced it"
