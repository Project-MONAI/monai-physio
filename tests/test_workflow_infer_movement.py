"""Tests for WorkflowInferMovement.process_time_series progress reporting.

A stub stands in for the PhysicsNeMo inference workflow, so the test exercises
the time-series loop, the image warp, and the USD export without a trained
network.
"""

import json
from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pyvista as pv

from monai_physio import WorkflowInferMovement


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
