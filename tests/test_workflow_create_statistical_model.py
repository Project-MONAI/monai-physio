"""Synthetic tests for the statistical-model workflow's PCA inputs.

The deformable registration is replaced by an identity transform, which is the
worst case of the under-fit these tests are about: the correspondence lands the
template exactly on itself and records none of the subject's shape.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import itk
import numpy as np
import pytest
import pyvista as pv

from monai_physio import workflow_create_statistical_model as wcsm
from monai_physio.workflow_create_statistical_model import (
    WorkflowCreateStatisticalModel,
)


def _bumpy_sphere(amplitude: float) -> pv.PolyData:
    """Return a sphere whose radius varies quadratically along Z.

    The modulation is quadratic so that it survives the workflow's affine ICP:
    a linear one would be a shear, which the alignment removes, leaving nothing
    for the PCA to find.
    """
    sphere = pv.Sphere(radius=10.0, theta_resolution=20, phi_resolution=20)
    points = np.asarray(sphere.points, dtype=np.float64)
    scale = 1.0 + amplitude * (points[:, 2] / 10.0) ** 2
    sphere.points = points * scale[:, np.newaxis]
    return sphere


class _IdentityRegistrar:
    """Stand-in for RegisterModelsDistanceMaps that barely deforms anything.

    Its forward transform picks up a millionth of the size difference between
    the sample it was handed and the template, rather than none: bumpy
    spheres of different amplitude have different bounding-box diagonals, so
    this ties the correspondence to a real, per-sample shape difference
    instead of an arbitrary, shape-unrelated fudge -- and unlike a pure
    identity it does not leave every corresponded sample bit-identical to the
    template, which is a degenerate population sklearn's PCA warns about. The
    fraction is small enough that it stays well under the near-zero variance
    thresholds the projection tests assert.
    """

    reference_images: list[itk.Image] = []

    def __init__(
        self,
        moving_model: pv.PolyData,
        fixed_model: pv.PolyData,
        reference_image: itk.Image,
        distance_squared_max: float = 50.0,
        mask_dilation_mm: float = 20.0,
    ) -> None:
        self.moving_model = moving_model
        self.fixed_model = fixed_model
        self.distance_squared_max = distance_squared_max
        self.mask_dilation_mm = mask_dilation_mm
        _IdentityRegistrar.reference_images.append(reference_image)

    def set_icon_weights_path(self, weights_path: str) -> None:
        self.weights_path = weights_path

    def register(self, transform_type: str) -> dict[str, Any]:
        transform = itk.AffineTransform[itk.D, 3].New()
        transform.SetIdentity()
        scale = 1.0 + 1.0e-6 * (
            self.moving_model.length / self.fixed_model.length - 1.0
        )
        transform.SetMatrix(itk.GetMatrixFromArray(np.eye(3) * scale))
        inverse_transform = itk.AffineTransform[itk.D, 3].New()
        assert transform.GetInverse(inverse_transform), "transform not invertible"
        return {
            "fixed_to_moving_transform": transform,
            "moving_to_fixed_transform": inverse_transform,
            "registered_model": self.moving_model,
        }


def _run(
    monkeypatch: Any, registrar: type = _IdentityRegistrar, **kwargs: Any
) -> WorkflowCreateStatisticalModel:
    """Run the workflow over three bumpy spheres with registration stubbed out."""
    monkeypatch.setattr(wcsm, "RegisterModelsDistanceMaps", registrar)
    _IdentityRegistrar.reference_images = []
    samples = [_bumpy_sphere(a) for a in (-0.15, 0.0, 0.15)]
    workflow = WorkflowCreateStatisticalModel(
        sample_meshes=list(samples),
        reference_mesh=_bumpy_sphere(0.0),
        number_of_pca_components=2,
        **kwargs,
    )
    workflow.process()
    return workflow


def test_distance_squared_max_defaults_to_the_mask_radius() -> None:
    """An unset saturation radius is sized to the mask, as the fit workflow does."""
    workflow = WorkflowCreateStatisticalModel(
        sample_meshes=[pv.Sphere()],
        reference_mesh=pv.Sphere(),
        mask_dilation_mm=10.0,
    )
    assert workflow.distance_squared_max == (1.25 * 10.0) ** 2

    explicit = WorkflowCreateStatisticalModel(
        sample_meshes=[pv.Sphere()],
        reference_mesh=pv.Sphere(),
        mask_dilation_mm=10.0,
        distance_squared_max=50.0,
    )
    assert explicit.distance_squared_max == 50.0


def test_projection_is_on_by_default() -> None:
    """The measured surfaces, not the registration's output, define the modes."""
    default = (
        inspect.signature(WorkflowCreateStatisticalModel.__init__)
        .parameters["project_to_measured_surfaces"]
        .default
    )
    assert default is True


def test_residual_to_the_measured_surface_is_reported(monkeypatch: Any) -> None:
    """A registration that does not deform must be reported as a full residual."""
    workflow = _run(monkeypatch)

    assert len(workflow.pca_input_residual_rms) == 3
    # The outer samples differ from the template, so the identity correspondence
    # leaves a residual; the middle one is the template itself.
    assert (
        min(workflow.pca_input_residual_rms[0], workflow.pca_input_residual_rms[2])
        > 0.01
    )
    assert workflow.pca_input_residual_rms[1] < 1.0e-6


def test_projection_recovers_the_population_variance(monkeypatch: Any) -> None:
    """Without projection an under-fitting registration collapses the modes."""
    unprojected = _run(monkeypatch, project_to_measured_surfaces=False)
    projected = _run(monkeypatch, project_to_measured_surfaces=True)

    assert unprojected.pca_fitted is not None and projected.pca_fitted is not None
    # Every corresponded shape is the template, so there is nothing to explain.
    assert unprojected.pca_fitted.explained_variance_.sum() < 1.0e-6
    assert projected.pca_fitted.explained_variance_.sum() > 1.0

    # The residual reports what the registration achieved, so projecting must
    # not flatter it.
    np.testing.assert_allclose(
        projected.pca_input_residual_rms, unprojected.pca_input_residual_rms
    )


def test_projection_threshold_leaves_distant_points_alone(monkeypatch: Any) -> None:
    """Points further than the threshold keep their registered position."""
    projected = _run(monkeypatch, projection_max_distance_mm=0.0)

    assert projected.pca_fitted is not None
    assert projected.pca_fitted.explained_variance_.sum() < 1.0e-6


def test_reference_image_covers_every_aligned_sample(monkeypatch: Any) -> None:
    """A sample clipped by the grid would register as if it were smaller."""
    workflow = _run(monkeypatch)

    image = _IdentityRegistrar.reference_images[0]
    origin = np.asarray(image.GetOrigin())
    spacing = np.asarray(image.GetSpacing())
    size = np.asarray(image.GetLargestPossibleRegion().GetSize())
    upper = origin + (size - 1) * spacing

    for aligned in workflow.aligned_models:
        points = np.asarray(aligned.points)
        assert np.all(points.min(axis=0) >= origin)
        assert np.all(points.max(axis=0) <= upper)


def test_aligned_models_stay_the_measured_inputs(monkeypatch: Any) -> None:
    """Step 4 measures against these, so the deformable stage must not eat them.

    The registrar here has to *move* its input: one that hands the same object
    back would satisfy this test whether the workflow kept the ICP-aligned
    model or replaced it with the registration output.
    """
    offset = np.array([3.0, -2.0, 1.0])
    handed_in: list[pv.PolyData] = []
    handed_back: list[pv.PolyData] = []

    class _DisplacingRegistrar(_IdentityRegistrar):
        def register(self, transform_type: str) -> dict[str, Any]:
            result = super().register(transform_type)
            moved = self.moving_model.copy(deep=True)
            moved.points = np.asarray(moved.points) + offset
            result["registered_model"] = moved
            handed_in.append(self.moving_model)
            handed_back.append(moved)
            return result

    workflow = _run(
        monkeypatch,
        registrar=_DisplacingRegistrar,
    )

    assert len(workflow.aligned_models) == 3
    assert len(handed_back) == 3
    for aligned, sample, given, returned in zip(
        workflow.aligned_models, workflow.sample_models, handed_in, handed_back
    ):
        assert aligned.n_points == sample.n_points
        # The ICP-aligned model is what the registrar was handed...
        np.testing.assert_allclose(aligned.points, given.points)
        # ...and it is not what the registrar handed back.
        np.testing.assert_allclose(
            np.asarray(returned.points) - np.asarray(aligned.points),
            np.broadcast_to(offset, (aligned.n_points, 3)),
        )


def test_icp_transform_type_defaults_to_affine_and_validates() -> None:
    """Affine strips size and gross proportion; the fit side must match it."""
    workflow = WorkflowCreateStatisticalModel(
        sample_meshes=[pv.Sphere()],
        reference_mesh=pv.Sphere(),
    )
    assert workflow.icp_transform_type == "Affine"

    workflow.set_icp_transform_type("Rigid")
    assert workflow.icp_transform_type == "Rigid"

    with pytest.raises(ValueError, match="Invalid ICP transform"):
        workflow.set_icp_transform_type("Deformable")


def test_the_constructor_validates_what_the_setters_validate() -> None:
    """A constructor that skips the setters accepts what the setter refuses."""
    with pytest.raises(ValueError, match="Invalid ICP transform"):
        WorkflowCreateStatisticalModel(
            sample_meshes=[pv.Sphere()],
            reference_mesh=pv.Sphere(),
            icp_transform_type="Deformable",
        )

    with pytest.raises(ValueError, match="distance_squared_max must be positive"):
        WorkflowCreateStatisticalModel(
            sample_meshes=[pv.Sphere()],
            reference_mesh=pv.Sphere(),
            distance_squared_max=0.0,
        )

    # None still means "derive it from the dilation", not "reject it".
    derived = WorkflowCreateStatisticalModel(
        sample_meshes=[pv.Sphere()],
        reference_mesh=pv.Sphere(),
        mask_dilation_mm=20.0,
        distance_squared_max=None,
    )
    assert derived.distance_squared_max == pytest.approx((1.25 * 20.0) ** 2)


def test_icp_transform_type_reaches_the_registrar(monkeypatch: Any) -> None:
    """The alignment the caller asked for is the one every sample gets."""
    seen: list[str] = []
    real = wcsm.RegisterModelsICP

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            self._inner = real(**kwargs)

        def register(self, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs["transform_type"])
            return cast(dict[str, Any], self._inner.register(**kwargs))

    monkeypatch.setattr(wcsm, "RegisterModelsICP", _Recorder)
    _run(monkeypatch, icp_transform_type="Rigid")

    assert seen == ["Rigid", "Rigid", "Rigid"]
