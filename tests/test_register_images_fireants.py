"""Tests for FireANTs-based image registration (GPU)."""

import hashlib
from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import monai_physio
from monai_physio.process_images import ProcessImages
from monai_physio.process_transforms import ProcessTransforms
from monai_physio.register_images_base import RegisterImagesBase
from monai_physio.register_images_fireants import RegisterImagesFireants
from monai_physio.workflow_reconstruct_highres_4d_ct import (
    WorkflowReconstructHighres4DCT,
)

from .conftest import KnownAffineCase, KnownShiftCase

T_SHIFT = 0.5
T_AFFINE = 1.0


def test_s1_ac1_subclass_and_public_export() -> None:
    """The class subclasses RegisterImagesBase and is exported."""
    assert issubclass(RegisterImagesFireants, RegisterImagesBase)
    assert monai_physio.RegisterImagesFireants is RegisterImagesFireants
    assert "RegisterImagesFireants" in monai_physio.__all__


def test_s1_ac2_invalid_transform_type_and_metric_raise_valueerror() -> None:
    """Invalid settings raise ValueError with Greedy's messages."""
    registrar = RegisterImagesFireants()
    for transform_type in ("Rigid", "Affine", "Deformable"):
        registrar.set_transform_type(transform_type)
        assert registrar.transform_type == transform_type
    for metric in ("CC", "Mattes", "MeanSquares"):
        registrar.set_metric(metric)
        assert registrar.metric == metric
    with pytest.raises(ValueError, match="Invalid transform type: Invalid"):
        registrar.set_transform_type("Invalid")
    with pytest.raises(ValueError, match="Invalid metric: Invalid"):
        registrar.set_metric("Invalid")


@pytest.mark.requires_gpu
@pytest.mark.parametrize("transform_type", ["Rigid", "Affine"])
def test_s1_ac3_known_shift_recovered(
    known_shift_case: KnownShiftCase, transform_type: str
) -> None:
    """Rigid and Affine recover a known shift within T_SHIFT."""
    registrar = RegisterImagesFireants()
    registrar.set_modality("ct")
    registrar.set_transform_type(transform_type)
    registrar.set_number_of_iterations([200, 100, 50])
    registrar.set_fixed_image(known_shift_case.fixed)

    result = registrar.register(moving_image=known_shift_case.moving)

    error_mm = known_shift_case.center_error_mm(result["fixed_to_moving_transform"])
    print(f"\nFireANTs {transform_type} known-shift error: {error_mm:.3f} mm")
    assert error_mm <= T_SHIFT


@pytest.mark.requires_gpu
@pytest.mark.parametrize("offset_z", [0.0, 1800.0])
def test_s1_ac4_known_affine_near_origin_and_z1800(offset_z: float) -> None:
    """Affine recovers a known affine, near the origin and at z ~ 1800 mm."""
    case = KnownAffineCase(origin_offset_mm=(0.0, 0.0, offset_z))
    registrar = RegisterImagesFireants()
    registrar.set_modality("ct")
    registrar.set_transform_type("Affine")
    # The blocky synthetic volume has no local contrast for CC to correlate.
    registrar.set_metric("MeanSquares")
    registrar.set_number_of_iterations([100, 50, 25])
    registrar.set_fixed_image(case.fixed)

    result = registrar.register(moving_image=case.moving)

    errors = case.probe_errors_mm(result["fixed_to_moving_transform"])
    print(f"\nFireANTs known-affine z={offset_z}: {np.round(errors, 3).tolist()} mm")
    assert float(errors.max()) <= T_AFFINE


@pytest.mark.requires_gpu
def test_s1_ac5_return_keys_and_ncc_not_worse(
    known_shift_case: KnownShiftCase,
) -> None:
    """Return dict has the three keys and registration does not lower NCC."""
    registrar = RegisterImagesFireants()
    registrar.set_modality("ct")
    registrar.set_transform_type("Affine")
    registrar.set_number_of_iterations([200, 100, 50])
    registrar.set_fixed_image(known_shift_case.fixed)

    result = registrar.register(moving_image=known_shift_case.moving)

    assert set(result) == {
        "fixed_to_moving_transform",
        "moving_to_fixed_transform",
        "loss",
    }
    assert isinstance(result["loss"], float)
    ncc = known_shift_case.foreground_ncc(result["fixed_to_moving_transform"])
    initial_ncc = known_shift_case.unregistered_ncc()
    assert ncc >= initial_ncc
    inverse = result["moving_to_fixed_transform"]
    assert isinstance(inverse, itk.Transform)


T_DEF_GAIN = 0.30
T_INV = 0.5
T_POINT_MM = 1.0


def _smooth_volume(size: int = 64, spacing_mm: float = 2.0) -> itk.Image:
    """Smoothed random texture: local contrast everywhere for CC."""
    rng = np.random.default_rng(0)
    array = gaussian_filter(rng.standard_normal((size,) * 3), sigma=2.0)
    array = (1000.0 * (array - array.min()) / np.ptp(array)).astype(np.float32)
    image = itk.image_from_array(array)
    image.SetSpacing([spacing_mm] * 3)
    return image


def _smooth_warp(image: itk.Image, amplitude_mm: float = 4.0) -> itk.Transform:
    """Smooth sinusoidal displacement field on the grid of ``image``."""
    shape = itk.array_from_image(image).shape
    z, y, x = np.meshgrid(*[np.linspace(0, np.pi, n) for n in shape], indexing="ij")
    field = np.stack(
        [
            amplitude_mm * np.sin(y) * np.sin(z),
            amplitude_mm * np.sin(x) * np.sin(z),
            -amplitude_mm * np.sin(x) * np.sin(y),
        ],
        axis=-1,
    )
    vectors = ProcessImages().convert_array_to_image_of_vectors(field, image, itk.D)
    transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    transform.SetDisplacementField(vectors)
    return transform


def _translation(offset: tuple[float, float, float]) -> itk.Transform:
    transform = itk.TranslationTransform[itk.D, 3].New()
    transform.SetOffset(list(offset))
    return transform


def _one_minus_ncc(a: itk.Image, b: itk.Image) -> float:
    x = itk.array_from_image(a).ravel().astype(np.float64)
    y = itk.array_from_image(b).ravel().astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    return 1.0 - float((x * y).sum() / np.sqrt((x**2).sum() * (y**2).sum()))


def _warp_pair() -> tuple[itk.Image, itk.Image]:
    fixed = _smooth_volume()
    moving = ProcessTransforms().transform_image(
        fixed, _smooth_warp(fixed), fixed, interpolation_method="linear"
    )
    return fixed, moving


def _deformable_registrar(fixed: itk.Image) -> RegisterImagesFireants:
    registrar = RegisterImagesFireants()
    registrar.set_modality("ct")
    registrar.set_transform_type("Deformable")
    registrar.set_number_of_iterations([100, 50, 25])
    registrar.set_fixed_image(fixed)
    return registrar


@pytest.mark.requires_gpu
def test_s2_ac1_deformable_gain_on_smooth_warp() -> None:
    """Deformable lowers (1 - NCC) by at least T_DEF_GAIN on a smooth warp."""
    fixed, moving = _warp_pair()
    registrar = _deformable_registrar(fixed)

    result = registrar.register(moving_image=moving)

    warped = ProcessTransforms().transform_image(
        moving,
        result["fixed_to_moving_transform"],
        fixed,
        interpolation_method="linear",
    )
    before = _one_minus_ncc(fixed, moving)
    after = _one_minus_ncc(fixed, warped)
    print(f"\nFireANTs deformable 1-NCC: {before:.4f} -> {after:.4f}")
    assert after <= (1.0 - T_DEF_GAIN) * before


@pytest.mark.requires_gpu
def test_s2_ac2_transforms_roundtrip_hdf_and_inverse_consistency(
    tmp_path: Path,
) -> None:
    """Both transforms survive an .hdf round trip and invert each other."""
    fixed, moving = _warp_pair()
    registrar = _deformable_registrar(fixed)
    result = registrar.register(moving_image=moving)

    size = np.array(itk.size(fixed))
    lo, hi = (size * 0.25).astype(int), (size * 0.75).astype(int)
    points = [
        fixed.TransformIndexToPhysicalPoint([int(i), int(j), int(k)])
        for i in range(lo[0], hi[0], 6)
        for j in range(lo[1], hi[1], 6)
        for k in range(lo[2], hi[2], 6)
    ]

    loaded = {}
    for key in ("fixed_to_moving_transform", "moving_to_fixed_transform"):
        transform = result[key]
        assert isinstance(transform, itk.Transform)
        path = tmp_path / f"{key}.hdf"
        itk.transformwrite([transform], str(path), compression=True)
        loaded[key] = itk.transformread(str(path))[0]
        for point in points[:20]:
            expected = np.array(list(transform.TransformPoint(point)))
            actual = np.array(list(loaded[key].TransformPoint(point)))
            assert np.abs(expected - actual).max() < 1e-3

    forward = loaded["fixed_to_moving_transform"]
    inverse = loaded["moving_to_fixed_transform"]
    residual = np.mean(
        [
            np.linalg.norm(
                np.array(list(inverse.TransformPoint(forward.TransformPoint(p))))
                - np.array(list(p))
            )
            for p in points
        ]
    )
    print(f"\nFireANTs inverse-consistency residual: {residual:.4f} mm")
    assert residual <= T_INV


@pytest.mark.requires_gpu
def test_s2_ac3_workflow_on_slicer_heart_small(test_images: list[Any]) -> None:
    """Injected into the 4D CT workflow, one image per phase on the reference grid."""
    registrar = RegisterImagesFireants()
    registrar.set_modality("ct")
    registrar.set_transform_type("Deformable")
    registrar.set_number_of_iterations([1, 0])
    reference = test_images[0]
    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=test_images[:3],
        reference_image=reference,
        reference_time_frame=0,
        registration_method=registrar,
    )

    result = workflow.process()

    reconstructed = result["reconstructed_images"]
    assert len(reconstructed) == 3
    for image in reconstructed:
        assert tuple(itk.size(image)) == tuple(itk.size(reference))
        assert np.allclose(itk.spacing(image), itk.spacing(reference))
        assert np.allclose(itk.origin(image), itk.origin(reference))


def _conflicting_pair(case: KnownShiftCase) -> tuple[itk.Image, itk.Image]:
    """Moving image shifted by the case's shift inside a box and by a conflicting
    amount outside it. Returns (moving, box mask)."""
    conflicting = ProcessTransforms().transform_image(
        case.fixed,
        _translation((-14.0, 10.0, -9.0)),
        case.fixed,
        interpolation_method="linear",
    )
    shape = itk.array_from_image(case.fixed).shape
    box = np.zeros(shape, dtype=np.uint8)
    box[
        shape[0] // 4 : 3 * shape[0] // 4,
        shape[1] // 4 : 3 * shape[1] // 4,
        shape[2] // 4 : 3 * shape[2] // 4,
    ] = 1
    moving_array = np.where(
        box.astype(bool),
        itk.array_from_image(case.moving),
        itk.array_from_image(conflicting),
    )
    moving = itk.image_from_array(moving_array.astype(np.float32))
    moving.CopyInformation(case.fixed)
    mask = itk.image_from_array(box)
    mask.CopyInformation(case.fixed)
    return moving, mask


@pytest.mark.requires_gpu
def test_s2_ac4_mask_labelmap_policy(known_shift_case: KnownShiftCase) -> None:
    """A mask changes the result and ignores out-of-mask structure; labelmaps raise."""
    moving, mask = _conflicting_pair(known_shift_case)

    def run(masked: bool) -> float:
        registrar = RegisterImagesFireants()
        registrar.set_modality("ct")
        registrar.set_transform_type("Affine")
        registrar.set_mask_dilation(0.0)
        registrar.set_number_of_iterations([200, 100, 50])
        registrar.set_fixed_image(known_shift_case.fixed)
        if masked:
            registrar.set_fixed_mask(mask)
        result = registrar.register(
            moving_image=moving, moving_mask=mask if masked else None
        )
        return known_shift_case.center_error_mm(result["fixed_to_moving_transform"])

    unmasked_error = run(masked=False)
    masked_error = run(masked=True)
    print(
        f"\nFireANTs center error unmasked {unmasked_error:.3f} mm, "
        f"masked {masked_error:.3f} mm"
    )
    assert masked_error <= T_SHIFT
    assert unmasked_error > masked_error

    registrar = RegisterImagesFireants()
    registrar.set_fixed_image(known_shift_case.fixed)
    with pytest.raises(NotImplementedError, match="moving_labelmap"):
        registrar.register(moving_image=moving, moving_labelmap=mask)
    registrar.set_fixed_labelmap(mask)
    with pytest.raises(NotImplementedError, match="fixed_labelmap"):
        registrar.register(moving_image=moving)


@pytest.mark.requires_gpu
def test_s2_ac6_inputs_not_mutated() -> None:
    """Registration leaves the fixed, moving, and mask images unchanged."""
    fixed, moving = _warp_pair()
    mask = itk.image_from_array(np.ones((64, 64, 64), dtype=np.uint8))
    mask.CopyInformation(fixed)

    def digest(image: itk.Image) -> tuple[str, tuple[Any, ...]]:
        pixels = hashlib.sha256(itk.array_from_image(image).tobytes()).hexdigest()
        return pixels, (tuple(image.GetSpacing()), tuple(image.GetOrigin()))

    images = (fixed, moving, mask)
    before = [digest(image) for image in images]
    registrar = _deformable_registrar(fixed)
    registrar.set_number_of_iterations([10, 5])
    registrar.set_fixed_mask(mask)
    registrar.register(moving_image=moving, moving_mask=mask)
    assert [digest(image) for image in images] == before


def test_s4_invalid_deformable_options_scales_and_init_affine_raise() -> None:
    """Unknown deformable options, bad scales and bad init_affine are rejected."""
    registrar = RegisterImagesFireants()
    with pytest.raises(ValueError, match="Invalid deformable options"):
        registrar.set_deformable_options(init_scale=2)
    with pytest.raises(ValueError, match="equal length"):
        registrar.set_scales([2.0, 1.0])
    with pytest.raises(ValueError, match="Invalid init_affine"):
        registrar.set_init_affine("Moments")


@pytest.mark.requires_gpu
def test_s4_tuned_options_deformable_gain_on_smooth_warp() -> None:
    """Identity init, explicit scales and tuned options still reduce 1 - NCC."""
    fixed, moving = _warp_pair()
    registrar = _deformable_registrar(fixed)
    registrar.set_number_of_iterations([40, 20, 10])
    registrar.set_scales([4.0, 2.0, 1.0])
    registrar.set_init_affine("Identity")
    registrar.set_deformable_options(
        smooth_warp_sigma=1.0, cc_kernel_size=9, optimizer_lr=1.0
    )

    result = registrar.register(moving_image=moving)

    warped = ProcessTransforms().transform_image(
        moving,
        result["fixed_to_moving_transform"],
        fixed,
        interpolation_method="linear",
    )
    assert _one_minus_ncc(fixed, warped) <= (1.0 - T_DEF_GAIN) * _one_minus_ncc(
        fixed, moving
    )


@pytest.mark.parametrize("preset", ["configure_for_heart_ct", "configure_for_lung_ct"])
def test_scales_length_validated_after_preset_iteration_change(preset: str) -> None:
    """Changing the iteration count after a preset cannot pair mismatched scales."""
    registrar = RegisterImagesFireants()
    getattr(registrar, preset)()
    assert len(registrar._scales()) == len(registrar.number_of_iterations)

    registrar.set_number_of_iterations([1, 0])
    with pytest.raises(ValueError, match="equal length"):
        registrar._scales()

    registrar.set_scales([2.0, 1.0])
    assert registrar._scales() == [2.0, 1.0]


def test_mattes_rescale_keeps_negative_intensities_distinct() -> None:
    """An all-negative CT keeps its contrast after the unit-range rescale."""
    array = np.linspace(-1000.0, -900.0, 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    image = itk.image_from_array(array)
    image.SetSpacing((2.0, 2.0, 2.0))

    rescaled = RegisterImagesFireants._rescale_unit_range(image)

    values = itk.array_from_image(rescaled)
    assert values.min() == 0.0
    assert values.max() == 1.0
    assert len(np.unique(values)) == len(np.unique(array))
    assert tuple(rescaled.GetSpacing()) == (2.0, 2.0, 2.0)


@pytest.mark.requires_gpu
def test_mattes_fireants_image_is_unit_range() -> None:
    """The Mattes metric hands FireANTs intensities inside [0, 1]."""
    registrar = RegisterImagesFireants()
    registrar.set_metric("Mattes")
    array = np.linspace(-1000.0, -900.0, 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)

    tensor = registrar._to_fireants_image(itk.image_from_array(array))()

    assert float(tensor.min()) == 0.0
    assert float(tensor.max()) == 1.0


def _known_geometry_pair() -> tuple[itk.Image, itk.Image, itk.Transform]:
    """Fixed and moving images on different grids related by a known transform.

    The moving grid differs from the fixed one in size, spacing and direction.
    The returned transform maps fixed physical points to moving physical points,
    and the moving image samples the fixed content through its inverse.
    """
    fixed = _smooth_volume()
    center = [63.0, 63.0, 63.0]
    known = itk.Euler3DTransform[itk.D].New()
    known.SetCenter(center)
    known.SetRotation(np.deg2rad(5.0), 0.0, np.deg2rad(8.0))
    known.SetTranslation([6.0, -4.0, 3.0])
    inverse = known.GetInverseTransform()

    size = np.array([96, 60, 80])
    spacing = np.array([1.5, 2.5, 2.0])
    direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moving_center = np.array(known.TransformPoint(center))
    origin = moving_center - direction @ (spacing * (size - 1) / 2.0)
    grid = itk.image_from_array(np.zeros(size[::-1], dtype=np.float32))
    grid.SetSpacing(spacing.tolist())
    grid.SetOrigin(origin.tolist())
    grid.SetDirection(itk.matrix_from_array(direction))
    moving = ProcessTransforms().transform_image(
        fixed, inverse, grid, interpolation_method="linear"
    )
    return fixed, moving, known


@pytest.mark.requires_gpu
def test_deformable_composition_matches_known_point_mapping_across_geometries() -> None:
    """Fixed-to-moving and its inverse map physical points as the known transform."""
    fixed, moving, known = _known_geometry_pair()
    registrar = _deformable_registrar(fixed)
    registrar.set_number_of_iterations([100, 50, 25])
    registrar.set_init_affine("Affine")

    result = registrar.register(moving_image=moving)

    margin = 0.25
    axis = np.linspace(margin * 126.0, (1.0 - margin) * 126.0, 6)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    expected = np.array([known.TransformPoint(p.tolist()) for p in points])
    forward = result["fixed_to_moving_transform"]
    backward = result["moving_to_fixed_transform"]
    mapped = np.array([forward.TransformPoint(p.tolist()) for p in points])
    returned = np.array([backward.TransformPoint(q.tolist()) for q in expected])

    assert np.linalg.norm(expected - points, axis=1).mean() > 5.0
    assert np.linalg.norm(mapped - expected, axis=1).mean() < T_POINT_MM
    assert np.linalg.norm(returned - points, axis=1).mean() < T_POINT_MM
