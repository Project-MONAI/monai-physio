"""Synthetic tests for PCA model registration helpers."""

from __future__ import annotations

from typing import Any, cast

import itk
import numpy as np
import pytest
import pyvista as pv
from scipy.optimize import approx_fprime

from monai_physio.register_models_pca import RegisterModelsPCA


def _make_registrar(**kwargs: Any) -> RegisterModelsPCA:
    """Create a small PCA registrar with a three-point template surface."""
    template_model = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    pca_eigenvectors = np.zeros((1, template_model.n_points * 3), dtype=np.float64)
    pca_std_deviations = np.ones(1, dtype=np.float64)
    fixed_distance_map = itk.image_from_array(np.zeros((4, 4, 4), dtype=np.float32))
    kwargs.setdefault("symmetric_weight", 0.0)
    return RegisterModelsPCA(
        pca_template_model=template_model,
        pca_eigenvectors=pca_eigenvectors,
        pca_std_deviations=pca_std_deviations,
        pca_number_of_modes=1,
        fixed_distance_map=fixed_distance_map,
        **kwargs,
    )


def _sphere_registrar(
    radius: float,
    modes: int = 1,
    **kwargs: Any,
) -> tuple[RegisterModelsPCA, np.ndarray]:
    """Build a registrar whose single mode inflates a sphere radially.

    Returns the registrar and the per-point unit-norm eigenvector it was given.
    """
    template = pv.Sphere(radius=radius, theta_resolution=24, phi_resolution=24)
    directions = np.asarray(template.points, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    eigenvectors = np.zeros((modes, template.n_points * 3), dtype=np.float64)
    eigenvectors[0] = directions.reshape(-1) / np.linalg.norm(directions)
    for mode in range(1, modes):
        # Orthogonal filler modes: displace a disjoint slab of points along x.
        filler = np.zeros((template.n_points, 3), dtype=np.float64)
        filler[mode::modes, 0] = 1.0
        eigenvectors[mode] = filler.reshape(-1) / np.linalg.norm(filler)

    reference_image = itk.image_from_array(np.zeros((48, 48, 48), dtype=np.float32))
    reference_image.SetSpacing([1.0, 1.0, 1.0])
    reference_image.SetOrigin([-24.0, -24.0, -24.0])

    registrar = RegisterModelsPCA(
        pca_template_model=template,
        pca_eigenvectors=eigenvectors,
        pca_std_deviations=np.full(modes, 5.0),
        pca_template_model_point_subsample=1,
        fixed_model=pv.Sphere(radius=radius, theta_resolution=24, phi_resolution=24),
        reference_image=reference_image,
        **kwargs,
    )
    return registrar, eigenvectors[0].reshape(-1, 3)


def test_set_fixed_model_requires_reference_image() -> None:
    """set_fixed_model fails clearly when reference_image is None."""
    registrar = _make_registrar()

    with pytest.raises(ValueError, match="reference_image must not be None"):
        registrar.set_fixed_model(
            cast(pv.UnstructuredGrid, registrar.pca_template_model), None
        )


def test_mode_count_mismatch_is_rejected() -> None:
    """Eigenvector and standard-deviation counts must agree."""
    template_model = pv.PolyData(np.zeros((3, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="Mode count mismatch"):
        RegisterModelsPCA(
            pca_template_model=template_model,
            pca_eigenvectors=np.zeros((2, 9), dtype=np.float64),
            pca_std_deviations=np.ones(3, dtype=np.float64),
            fixed_distance_map=itk.image_from_array(
                np.zeros((4, 4, 4), dtype=np.float32)
            ),
        )


def test_compute_pca_deformation_scales_eigenvectors_by_std() -> None:
    """Deformation is exactly sum(b_i * std_i * eigenvector_i), reshaped (N, 3)."""
    template_model = pv.PolyData(np.zeros((2, 3), dtype=np.float64))
    eigenvectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    std_deviations = np.array([2.0, 5.0], dtype=np.float64)
    registrar = RegisterModelsPCA(
        pca_template_model=template_model,
        pca_eigenvectors=eigenvectors,
        pca_std_deviations=std_deviations,
        fixed_distance_map=itk.image_from_array(np.zeros((4, 4, 4), dtype=np.float32)),
    )

    deformation = registrar._compute_pca_deformation(np.array([1.5, -1.0]))

    # Point 0: 1.5*2*[1,0,0] + (-1)*5*[0,0,1]; point 1: 1.5*2*[0,1,0].
    assert np.allclose(deformation, [[3.0, 0.0, -5.0], [0.0, 3.0, 0.0]])

    # A shorter coefficient vector uses only the leading modes.
    leading = registrar._compute_pca_deformation(np.array([1.5]))
    assert np.allclose(leading, [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])


def test_transform_template_model_applies_post_pca_transform_after_deformation() -> (
    None
):
    """Post-PCA transform is applied after PCA deformation."""
    registrar = _make_registrar()
    registrar.registered_model_pca_coefficients = np.array([1.0], dtype=np.float64)
    registrar.registered_model_pca_deformation = np.tile(
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        (registrar.pca_template_model.n_points, 1),
    )
    transform = itk.ScaleTransform[itk.D, 3].New()
    transform.SetScale([2.0, 2.0, 2.0])
    registrar.post_pca_transform = transform

    registered_model: Any = registrar.transform_template_model()

    assert np.allclose(registered_model.points[0], [2.0, 0.0, 0.0])
    assert np.allclose(registered_model.points[1], [4.0, 0.0, 0.0])


def test_modes_are_deformed_in_the_template_frame_then_transformed() -> None:
    """Regression: modes must be rotated with the template, not added after it.

    The registered model must equal ``A @ (template + deformation)``, never
    ``A @ template + deformation``. The two differ whenever the post-PCA
    transform contains a rotation, which is the case for every ICP alignment.
    """
    registrar = _make_registrar()
    registrar.registered_model_pca_coefficients = np.array([1.0], dtype=np.float64)
    deformation = np.tile(
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        (registrar.pca_template_model.n_points, 1),
    )
    registrar.registered_model_pca_deformation = deformation

    # 90 degrees about z, so x-displacements must come out along y.
    matrix = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    offset = np.array([10.0, -3.0, 2.0], dtype=np.float64)
    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetMatrix(itk.matrix_from_array(matrix))
    transform.SetTranslation(offset)
    registrar.post_pca_transform = transform

    registered_model: Any = registrar.transform_template_model()

    template_points = np.asarray(registrar.pca_template_model.points, dtype=np.float64)
    correct = (template_points + deformation) @ matrix.T + offset
    wrong = template_points @ matrix.T + offset + deformation

    assert np.allclose(registered_model.points, correct)
    assert not np.allclose(registered_model.points, wrong)


def test_analytic_gradient_matches_finite_differences() -> None:
    """The supplied Jacobian agrees with a finite-difference gradient."""
    registrar, _ = _sphere_registrar(radius=8.0, modes=3, symmetric_weight=0.5)
    registrar.pca_prior_weight = 0.1
    registrar._prepare_sampling()

    params = np.array([0.4, -0.25, 0.15], dtype=np.float64)
    _, analytic = registrar._objective_and_gradient(params)
    numeric = approx_fprime(params, registrar._mean_distance_metric, 1e-5)

    assert np.allclose(analytic, numeric, atol=2e-3)


def test_register_recovers_known_coefficients() -> None:
    """Fitting to a target built from known coefficients recovers them."""
    registrar, mode = _sphere_registrar(radius=8.0, modes=1, symmetric_weight=0.0)

    # Target = template inflated by b = 1.0 along the single radial mode.
    truth = 1.0
    target = registrar.pca_template_model.copy(deep=True)
    target.points = (
        np.asarray(registrar.pca_template_model.points, dtype=np.float64)
        + truth * registrar.pca_std_deviations[0] * mode
    )

    reference_image = itk.image_from_array(np.zeros((64, 64, 64), dtype=np.float32))
    reference_image.SetSpacing([0.75, 0.75, 0.75])
    reference_image.SetOrigin([-24.0, -24.0, -24.0])
    registrar.set_fixed_model(cast(pv.UnstructuredGrid, target), reference_image)

    result = registrar.register(pca_number_of_modes=1, max_iterations=60)

    assert result["pca_coefficients"][0] == pytest.approx(truth, abs=0.1)
    assert result["mean_distance"] < 0.5


def test_symmetric_term_penalizes_partial_coverage() -> None:
    """The target-to-model term sees coverage the model-to-target term misses.

    A hemisphere sitting on a full sphere scores near-perfectly one-way: every
    model point lies on the target surface. Only the target-to-model term
    notices that half the target has no model near it.
    """
    target = pv.Sphere(radius=8.0, theta_resolution=24, phi_resolution=24)
    points = np.asarray(target.points, dtype=np.float64)
    hemisphere = pv.PolyData(points[points[:, 2] > 0.0])

    reference_image = itk.image_from_array(np.zeros((64, 64, 64), dtype=np.float32))
    reference_image.SetSpacing([0.75, 0.75, 0.75])
    reference_image.SetOrigin([-24.0, -24.0, -24.0])

    registrar = RegisterModelsPCA(
        pca_template_model=hemisphere,
        pca_eigenvectors=np.zeros((1, hemisphere.n_points * 3), dtype=np.float64),
        pca_std_deviations=np.ones(1, dtype=np.float64),
        pca_template_model_point_subsample=1,
        fixed_model=target,
        reference_image=reference_image,
        symmetric_weight=0.0,
    )

    registrar._prepare_sampling()
    one_way = registrar._mean_distance_metric(np.zeros(1))

    registrar.symmetric_weight = 0.5
    registrar._prepare_sampling()
    symmetric = registrar._mean_distance_metric(np.zeros(1))

    # Model points all lie on the target surface, so the one-way term is small.
    assert one_way < 0.5
    assert symmetric > 1.0


def test_prior_shrinks_coefficients() -> None:
    """Raising pca_prior_weight pulls the solution toward the mean shape."""
    registrar, mode = _sphere_registrar(radius=8.0, modes=1, symmetric_weight=0.0)

    target = registrar.pca_template_model.copy(deep=True)
    target.points = (
        np.asarray(registrar.pca_template_model.points, dtype=np.float64)
        + 1.0 * registrar.pca_std_deviations[0] * mode
    )

    reference_image = itk.image_from_array(np.zeros((64, 64, 64), dtype=np.float32))
    reference_image.SetSpacing([0.75, 0.75, 0.75])
    reference_image.SetOrigin([-24.0, -24.0, -24.0])
    registrar.set_fixed_model(cast(pv.UnstructuredGrid, target), reference_image)

    unregularized = registrar.register(pca_number_of_modes=1, max_iterations=60)

    registrar.pca_prior_weight = 5.0
    registrar._sampling_ready = False
    regularized = registrar.register(pca_number_of_modes=1, max_iterations=60)

    assert abs(regularized["pca_coefficients"][0]) < abs(
        unregularized["pca_coefficients"][0]
    )


def test_transform_point_requires_computed_transforms() -> None:
    """transform_point raises instead of silently returning the input."""
    registrar = _make_registrar()
    point = itk.Point[itk.D, 3]()
    point[0], point[1], point[2] = 1.0, 2.0, 3.0

    with pytest.raises(ValueError, match="compute_pca_transforms"):
        registrar.transform_point(point)


def test_pca_transforms_round_trip() -> None:
    """forward reproduces the deformation and inverse undoes it."""
    registrar, mode = _sphere_registrar(radius=8.0, modes=1, symmetric_weight=0.0)
    registrar.registered_model_pca_coefficients = np.array([1.0], dtype=np.float64)
    registrar.registered_model_pca_deformation = (
        1.0 * registrar.pca_std_deviations[0] * mode
    )

    reference_image = itk.image_from_array(np.zeros((64, 64, 64), dtype=np.float32))
    reference_image.SetSpacing([0.75, 0.75, 0.75])
    reference_image.SetOrigin([-24.0, -24.0, -24.0])

    transforms = registrar.compute_pca_transforms(reference_image, blur_sigma=1.5)
    forward = transforms["forward_point_transform"]
    inverse = transforms["inverse_point_transform"]

    template_points = np.asarray(registrar.pca_template_model.points, dtype=np.float64)
    expected = template_points + registrar.registered_model_pca_deformation

    point = itk.Point[itk.D, 3]()
    mapped = np.empty_like(template_points)
    back = np.empty_like(template_points)
    for i, source in enumerate(template_points):
        point[0], point[1], point[2] = (float(v) for v in source)
        forward_point = forward.TransformPoint(point)
        mapped[i] = (forward_point[0], forward_point[1], forward_point[2])
        inverse_point = inverse.TransformPoint(forward_point)
        back[i] = (inverse_point[0], inverse_point[1], inverse_point[2])

    # The field is splatted and blurred, so it only approximates the deformation.
    field_rms = np.sqrt(np.mean(np.sum((mapped - expected) ** 2, axis=1)))
    round_trip_rms = np.sqrt(np.mean(np.sum((back - template_points) ** 2, axis=1)))

    assert field_rms < 1.5
    assert round_trip_rms < 1.0

    # An identity field would satisfy both bounds above, so require that the
    # forward transform actually moves the points a comparable distance.
    displacement_rms = np.sqrt(np.mean(np.sum((mapped - template_points) ** 2, axis=1)))
    expected_rms = np.sqrt(np.mean(np.sum((expected - template_points) ** 2, axis=1)))
    assert displacement_rms > 0.5 * expected_rms
