"""Progress reporting of WorkflowFitStatisticalModelToPatient on real meshes.

The KCL heart average mesh is fitted to one KCL patient mesh, both decimated so
the fit is quick.  The KCL data ships no statistical model, so a one-mode PCA
model is built here whose mode inflates the template radially; the fit only
needs the PCA stage to run its optimizer, not to explain a population.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pytest
import pyvista as pv

from monai_physio.process_surface_mesh import ProcessSurfaceMesh
from monai_physio.workflow_fit_statistical_model_to_patient import (
    WorkflowFitStatisticalModelToPatient,
)


def _surface(path: Path, reduction: float) -> pv.PolyData:
    mesh = pv.read(str(path))
    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    return cast(pv.PolyData, surface.decimate(reduction).clean())


@pytest.fixture(scope="module")
def fit_inputs(
    download_kcl_heart_model: Path,
) -> FitInputs:
    template = _surface(download_kcl_heart_model / "average_mesh.vtk", 0.95)
    patient = _surface(download_kcl_heart_model / "input_meshes" / "01.vtk", 0.95)

    radial = np.asarray(template.points, dtype=np.float64) - np.asarray(template.center)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    component = radial.reshape(-1) / np.linalg.norm(radial)
    pca_model = {"eigenvalues": [25.0], "components": [component.tolist()]}
    # A 2 mm grid rather than the 1 mm one the workflow derives by default
    # keeps the distance-map stages quick.
    patient_image = ProcessSurfaceMesh().create_reference_image(
        mesh=patient, spatial_resolution=2.0, buffer_factor=0.25, ptype=itk.F
    )
    return template, patient, pca_model, patient_image


FitInputs = tuple[pv.PolyData, pv.PolyData, dict[str, Any], itk.Image]


def _workflow(fit_inputs: FitInputs) -> WorkflowFitStatisticalModelToPatient:
    template, patient, pca_model, patient_image = fit_inputs
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=template, patient_models=[patient], patient_image=patient_image
    )
    workflow.set_use_pca_registration(
        True, pca_model=pca_model, number_of_pca_components=1, use_surface=True
    )
    return workflow


def _check_progress(calls: list[tuple[float, str]], stages: list[str]) -> None:
    fractions = [fraction for fraction, _ in calls]
    assert fractions == sorted(fractions), "progress moved backwards"
    assert fractions[0] == pytest.approx(0.0, abs=0.01)
    assert fractions[-1] == 1.0
    messages = [message for _, message in calls]
    for stage in stages:
        assert any(m.startswith(f"{stage}: ") for m in messages), stage


def test_fit_progress_with_greedy_deformable(
    fit_inputs: FitInputs,
) -> None:
    workflow = _workflow(fit_inputs)
    workflow.set_l2l_deformable_engine("greedy")
    calls: list[tuple[float, str]] = []
    workflow.set_progress_callback(
        lambda fraction, message: calls.append((fraction, message))
    )
    workflow.process()

    _check_progress(calls, ["ICP", "PCA", "Labelmap-to-labelmap", "Apply transforms"])
    messages = [message for _, message in calls]
    assert not any(m.startswith("Labelmap-to-image") for m in messages)
    assert any(m.startswith("PCA: PCA iteration: 1/") for m in messages)


@pytest.mark.requires_gpu
def test_fit_progress_reports_icon_steps_without_printing(
    fit_inputs: FitInputs,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workflow = _workflow(fit_inputs)
    calls: list[tuple[float, str]] = []
    workflow.set_progress_callback(
        lambda fraction, message: calls.append((fraction, message))
    )
    workflow.process()

    _check_progress(calls, ["ICP", "PCA", "Labelmap-to-labelmap", "Apply transforms"])
    messages = [message for _, message in calls]
    icon_steps = [m for m in messages if "ICON step" in m]
    assert icon_steps[-1] == "Labelmap-to-labelmap: ICON deformable: ICON step: 50/50"
    # icon_registration prints each step's loss; with a callback it is logged.
    assert "ICONLoss" not in capsys.readouterr().out


def test_omitting_the_callback_changes_nothing(
    fit_inputs: FitInputs,
) -> None:
    """ICP and PCA are deterministic, so with and without reporting agree."""
    results = []
    for callback in (None, lambda fraction, message: None):
        workflow = _workflow(fit_inputs)
        workflow.set_use_labelmap_to_labelmap_registration(False)
        workflow.set_progress_callback(callback)
        results.append(workflow.process())

    without, with_callback = results
    assert np.array_equal(
        np.asarray(without["fitted_reference_mesh"].points),
        np.asarray(with_callback["fitted_reference_mesh"].points),
    )
