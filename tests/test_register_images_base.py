"""Tests for the background value RegisterImagesBase writes off-grid.

The images here are synthetic: a moving image shifted off the fixed grid, so
every resampling leaves a region the moving image does not cover.
"""

import itk
import numpy as np

from monai_physio.register_images_base import RegisterImagesBase

_SIZE = 10
_SHIFT_MM = 5.0


def _uniform_image(value: float, dtype: type) -> itk.Image:
    """Return a cubic image whose voxels all hold ``value``."""
    return itk.image_from_array(np.full((_SIZE, _SIZE, _SIZE), value, dtype=dtype))


def _shift_transform() -> itk.Transform:
    """Return a translation large enough to push half the grid off the moving image."""
    transform = itk.TranslationTransform[itk.D, 3].New()
    transform.SetOffset([_SHIFT_MM, 0.0, 0.0])
    return transform


def _off_grid_voxel(image: itk.Image) -> float:
    """Return the voxel that the shift maps outside the moving image."""
    return float(itk.GetArrayViewFromImage(image)[0, 0, _SIZE - 1])


def _on_grid_voxel(image: itk.Image) -> float:
    """Return a voxel the shift keeps inside the moving image."""
    return float(itk.GetArrayViewFromImage(image)[0, 0, 0])


def test_prewarp_background_value_uses_override() -> None:
    """An explicit override wins over the modality default."""
    registrar = RegisterImagesBase()
    registrar.set_modality("ct")
    registrar.set_prewarp_background_value(42.0)

    assert (
        registrar._prewarp_background_value(_uniform_image(100.0, np.float32)) == 42.0
    )


def test_prewarp_background_value_ct_is_air() -> None:
    """CT falls back to -1000 HU rather than the image's own minimum."""
    registrar = RegisterImagesBase()
    registrar.set_modality("ct")

    assert (
        registrar._prewarp_background_value(_uniform_image(100.0, np.float32))
        == -1000.0
    )


def test_prewarp_background_value_non_ct_uses_image_minimum() -> None:
    """Other modalities have no fixed air value, so the image's minimum is used."""
    registrar = RegisterImagesBase()
    registrar.set_modality("mri")
    moving_image = _uniform_image(100.0, np.float32)
    itk.GetArrayViewFromImage(moving_image)[0, 0, 0] = 7.0

    assert registrar._prewarp_background_value(moving_image) == 7.0


def test_prewarp_moving_fills_image_with_modality_background() -> None:
    """The pre-warped intensity image is filled with air, not 0 HU."""
    registrar = RegisterImagesBase()
    registrar.set_modality("ct")
    registrar.set_fixed_image(_uniform_image(0.0, np.float32))

    warped_image, warped_mask, warped_labelmap = registrar._prewarp_moving(
        _shift_transform(), _uniform_image(100.0, np.float32), None, None
    )

    assert _on_grid_voxel(warped_image) == 100.0
    assert _off_grid_voxel(warped_image) == -1000.0
    assert warped_mask is None
    assert warped_labelmap is None


def test_prewarp_moving_fills_mask_and_labelmap_with_zero() -> None:
    """Masks and labelmaps keep their discrete values and are filled with 0."""
    registrar = RegisterImagesBase()
    registrar.set_modality("ct")
    registrar.set_fixed_image(_uniform_image(0.0, np.float32))

    _, warped_mask, warped_labelmap = registrar._prewarp_moving(
        _shift_transform(),
        _uniform_image(100.0, np.float32),
        _uniform_image(1, np.uint8),
        _uniform_image(3, np.uint8),
    )

    assert _on_grid_voxel(warped_mask) == 1
    assert _off_grid_voxel(warped_mask) == 0
    assert _on_grid_voxel(warped_labelmap) == 3
    assert _off_grid_voxel(warped_labelmap) == 0


def test_get_registered_image_fills_with_modality_background() -> None:
    """The registered CT is filled with air where the moving image ends."""
    registrar = RegisterImagesBase()
    registrar.set_modality("ct")
    registrar.set_fixed_image(_uniform_image(0.0, np.float32))
    registrar.moving_image = _uniform_image(100.0, np.float32)
    registrar.forward_transform = _shift_transform()

    registered = registrar.get_registered_image()

    assert _on_grid_voxel(registered) == 100.0
    assert _off_grid_voxel(registered) == -1000.0
