"""Synthetic tests for the unbiased mean-surface workflow."""

from __future__ import annotations

from typing import Any, Optional

import itk
import numpy as np
import pytest
import pyvista as pv

from monai_physio import WorkflowCreateMeanSurface
from monai_physio import workflow_create_mean_surface as wcms

# Spheres of differing radii AND differing point counts: the population this
# workflow exists for. Radii mean is 12.0 mm.
_SPHERES = (
    (8.0, 20, 20),
    (12.0, 30, 30),
    (16.0, 24, 24),
)


def _sphere(radius: float, theta: int, phi: int) -> pv.PolyData:
    """Sphere at the origin with an explicit tessellation density."""
    return pv.Sphere(
        radius=radius,
        theta_resolution=theta,
        phi_resolution=phi,
        center=(0.0, 0.0, 0.0),
    )


def _mean_radius(surface: pv.PolyData) -> float:
    """Mean distance from the surface centroid to its points."""
    points = np.asarray(surface.points)
    return float(np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1)))


def _make_workflow(template: pv.PolyData, iterations: int) -> WorkflowCreateMeanSurface:
    """Workflow over the sphere population, in the CPU-only Affine mode."""
    workflow = WorkflowCreateMeanSurface(
        surfaces=[_sphere(*spec) for spec in _SPHERES],
        template_surface=template,
        spatial_resolution=1.0,
    )
    # Affine keeps the test off the GPU/ICON path; scale differences are then
    # carried by the distance-map stage rather than the ICP alignment.
    workflow.set_registration_transform_type("Affine")
    workflow.set_number_of_iterations(iterations)
    return workflow


def test_requires_at_least_two_surfaces() -> None:
    """A single surface has no population to average."""
    with pytest.raises(ValueError, match="At least 2 surfaces"):
        WorkflowCreateMeanSurface(surfaces=[_sphere(*_SPHERES[0])])


def test_invalid_transform_types_rejected() -> None:
    """Transform-type setters validate against the registrars' vocabularies."""
    workflow = _make_workflow(_sphere(*_SPHERES[1]), iterations=1)
    with pytest.raises(ValueError, match="Invalid alignment transform"):
        workflow.set_alignment_transform_type("Deformable")
    with pytest.raises(ValueError, match="Invalid registration transform"):
        workflow.set_registration_transform_type("Elastic")


@pytest.mark.slow
def test_mean_keeps_template_topology_and_averages_size() -> None:
    """The mean has template topology and the population's mean size."""
    template = _sphere(*_SPHERES[1])
    result = _make_workflow(template, iterations=2).process()

    mean_surface = result["mean_surface"]
    assert mean_surface.n_points == template.n_points
    assert mean_surface.n_cells == template.n_cells
    assert len(result["corresponded_surfaces"]) == len(_SPHERES)

    expected = float(np.mean([spec[0] for spec in _SPHERES]))
    assert _mean_radius(mean_surface) == pytest.approx(expected, abs=1.5)


@pytest.mark.slow
def test_mean_is_independent_of_template_choice() -> None:
    """The mean does not inherit the size of whichever template started it.

    Spheres differ only by scale, which the scale normalization pins to the
    population, so the two runs agree without needing several iterations; the
    iterations matter for non-affine shape differences, which the Affine mode
    used here cannot express.
    """
    iterated_small = _make_workflow(_sphere(*_SPHERES[0]), iterations=3).process()
    iterated_large = _make_workflow(_sphere(*_SPHERES[2]), iterations=3).process()

    radius_small = _mean_radius(iterated_small["mean_surface"])
    radius_large = _mean_radius(iterated_large["mean_surface"])

    # Templates 8 mm apart in radius; the means must not be.
    assert abs(radius_small - radius_large) < 0.5
    expected = float(np.mean([spec[0] for spec in _SPHERES]))
    for radius in (radius_small, radius_large):
        assert radius == pytest.approx(expected, abs=1.5)


@pytest.mark.slow
def test_converged_run_stops_early() -> None:
    """A converged iteration reports fewer runs than requested."""
    workflow = _make_workflow(_sphere(*_SPHERES[1]), iterations=5)
    workflow.set_convergence_tolerance(1.0e3)  # trivially satisfied
    result = workflow.process()

    assert result["number_of_iterations_run"] == 1
    assert len(result["iteration_rms_mm"]) == 1


def test_distance_squared_max_defaults_to_the_mask_radius() -> None:
    """An unset saturation radius is sized to the mask, as the fit workflow does."""
    workflow = _make_workflow(_sphere(*_SPHERES[1]), iterations=1)
    workflow.set_mask_dilation_mm(10.0)
    assert workflow._distance_squared_max() == (1.25 * 10.0) ** 2

    workflow.set_distance_squared_max(50.0)
    assert workflow._distance_squared_max() == 50.0


def test_correspondence_tuning_reaches_the_registrar(monkeypatch: Any) -> None:
    """Stock distance maps and stock ICON weights under-fit, so both are tunable."""
    seen: list[_Registrar] = []

    class _Registrar:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.weights_path: Optional[str] = None
            seen.append(self)

        def set_icon_weights_path(self, weights_path: str) -> None:
            self.weights_path = weights_path

        def register(self, transform_type: str) -> dict[str, Any]:
            identity = itk.AffineTransform[itk.D, 3].New()
            identity.SetIdentity()
            return {"forward_transform": identity}

    monkeypatch.setattr(wcms, "RegisterModelsDistanceMaps", _Registrar)

    workflow = _make_workflow(_sphere(*_SPHERES[1]), iterations=1)
    workflow.set_mask_dilation_mm(10.0)
    workflow.set_icon_weights_path("finetuned.trch")
    workflow.process()

    assert len(seen) == len(_SPHERES)
    for registrar in seen:
        assert registrar.kwargs["mask_dilation_mm"] == 10.0
        assert registrar.kwargs["distance_squared_max"] == (1.25 * 10.0) ** 2
        assert registrar.weights_path == "finetuned.trch"
