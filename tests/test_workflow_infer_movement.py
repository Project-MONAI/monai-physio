"""Tests for WorkflowInferMovement.process_time_series progress reporting.

A stub stands in for the PhysicsNeMo inference workflow, so the test exercises
the time-series loop, the image warp, and the USD export without a trained
network.
"""

import json
import weakref
from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pytest
import pyvista as pv

from monai_physio import WorkflowInferMovement
from monai_physio.process_transforms import ProcessTransforms


class _StubInference:
    """Predicts a uniform displacement proportional to the stage."""

    n_target = 3

    def __init__(self, template_mesh: pv.PolyData, model_directory: Path) -> None:
        self.template_mesh = template_mesh
        self.model_directory = model_directory

    def predict(self, coeffs: np.ndarray, stage: float) -> np.ndarray:
        displacement = np.zeros((self.template_mesh.n_points, 3), dtype=np.float32)
        displacement[:, 2] = 2.0 * stage
        return displacement


@pytest.mark.parametrize("restrict_exterior", [False, True])
def test_time_series_can_release_dense_fields_between_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restrict_exterior: bool
) -> None:
    """A 16-cubed warp checks outputs and reference lifetimes without a network."""
    sphere = pv.Sphere(radius=3, theta_resolution=8, phi_resolution=8)
    mesh_file = tmp_path / "fitted.vtp"
    sphere.save(str(mesh_file))
    coeff_file = tmp_path / "subject.json"
    coeff_file.write_text(json.dumps([0.0, 0.0]), encoding="utf-8")
    values = np.indices((16, 16, 16)).sum(axis=0).astype(np.float32)
    image = itk.image_from_array(values)
    image.SetOrigin([-8.0, -8.0, -8.0])
    mask = itk.image_from_array((values < 24).astype(np.uint8))
    mask.CopyInformation(image)
    workflow = WorkflowInferMovement(cast(Any, _StubInference(sphere, tmp_path)))
    options: dict[str, Any] = {
        "shape_parameters": coeff_file,
        "stages": [0.0, 0.5, 1.0],
        "fitted_reference_mesh": mesh_file,
        "reference_image": image,
        "smoothing_sigma_mm": 2.0,
        "exterior_mask": mask if restrict_exterior else None,
    }
    kept_progress: list[tuple[float, str]] = []
    workflow.set_progress_callback(lambda f, m: kept_progress.append((f, m)))
    retained = workflow.process_time_series(
        output_directory=tmp_path / "retained", **options
    )
    assert len(retained["transforms"]) == 3

    references: list[Any] = []
    create_field = workflow.create_deformation_field
    transform_image = ProcessTransforms.transform_image

    def track_field(**kwargs: Any) -> dict[str, Any]:
        assert all(ref() is None for ref in references)
        field = create_field(**kwargs)
        references.extend(
            weakref.ref(value)
            for value in field.values()
            if isinstance(value, itk.Image)
        )
        return field

    def track_warp(
        self: ProcessTransforms,
        image: itk.Image,
        transform: Any,
        *args: Any,
        **kwargs: Any,
    ) -> itk.Image:
        result = transform_image(self, image, transform, *args, **kwargs)
        references.extend((weakref.ref(transform), weakref.ref(result)))
        return result

    monkeypatch.setattr(workflow, "create_deformation_field", track_field)
    monkeypatch.setattr(ProcessTransforms, "transform_image", track_warp)
    released_progress: list[tuple[float, str]] = []
    workflow.set_progress_callback(lambda f, m: released_progress.append((f, m)))
    released = workflow.process_time_series(
        output_directory=tmp_path / "released", return_transforms=False, **options
    )
    assert released["transforms"] == []
    assert all(ref() is None for ref in references)
    assert released_progress == kept_progress
    for kept_mesh, released_mesh in zip(
        retained["stage_meshes"], released["stage_meshes"]
    ):
        np.testing.assert_array_equal(kept_mesh.points, released_mesh.points)
    for kept_path, released_path in zip(
        retained["warped_images"], released["warped_images"]
    ):
        kept_image = itk.imread(str(kept_path))
        released_image = itk.imread(str(released_path))
        np.testing.assert_array_equal(
            itk.array_from_image(kept_image), itk.array_from_image(released_image)
        )
        np.testing.assert_allclose(kept_image.GetOrigin(), released_image.GetOrigin())
        from monai_physio.process_tests import ProcessTests

        checks = ProcessTests(
            class_name=(
                "movement_time_series_exterior"
                if restrict_exterior
                else "movement_time_series_plain"
            ),
            results_dir=released_path.parent,
        )
        assert checks.compare_result_to_baseline_image(
            released_path.name,
            per_pixel_absolute_error_tol=1e-5,
            total_absolute_error_tol=0.05,
        )


def test_process_time_series_reports_each_stage_and_usd(tmp_path: Path) -> None:
    sphere = pv.Sphere(radius=6.0, theta_resolution=16, phi_resolution=16)
    mesh_file = tmp_path / "fitted.vtp"
    sphere.save(str(mesh_file))
    coeff_file = tmp_path / "subject.json"
    coeff_file.write_text(json.dumps([0.0, 0.0]), encoding="utf-8")

    image = itk.image_from_array(np.zeros((20, 20, 20), dtype=np.float32))
    image.SetOrigin([-10.0, -10.0, -10.0])

    calls: list[tuple[float, str]] = []
    stub = cast(Any, _StubInference(sphere, tmp_path))
    workflow = WorkflowInferMovement(stub)
    workflow.set_progress_callback(
        lambda fraction, message: calls.append((fraction, message))
    )
    result = workflow.process_time_series(
        shape_parameters=coeff_file,
        stages=[0.0, 0.5],
        output_directory=tmp_path / "out",
        fitted_reference_mesh=mesh_file,
        reference_image=image,
        smoothing_sigma_mm=2.0,
        usd_project_name="progress",
    )

    assert result["usd_file"] is not None
    fractions = [fraction for fraction, _ in calls]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    messages = [message for _, message in calls]
    assert any(m.startswith("Stage 0.000 (1/2): predicting") for m in messages)
    assert any(m.startswith("Stage 0.500 (2/2): warping image") for m in messages)
    assert any(m.startswith("USD: ") for m in messages)
    # The USD export is reported after every stage.
    first_usd = next(i for i, m in enumerate(messages) if m.startswith("USD: "))
    assert all(not m.startswith("Stage ") for m in messages[first_usd:])
