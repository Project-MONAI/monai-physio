"""
Tutorial 7 (Duke Heart): Fit Statistical Shape Model to Patient Data

Purpose
-------
Duke counterpart of ``tutorial_07_heart_fit_statistical_model_to_patient.py``.
The heart PCA model built by ``tutorial_06_duke_heart_create_statistical_model.py``
is fitted to one Duke-Heart-4DLabelmaps case, whose reference frame plays the
patient.

Duke-Heart-4DLabelmaps ships labelmaps rather than CT, so there is nothing to
segment: the patient surface is contoured straight from the case's labelmap,
with the same labels dropped that Tutorial 4 drops, so the patient and the model
describe the same structure.  Nothing being segmented also means there
is no patient intensity image; the workflow then rasterizes its own reference
grid from the patient surface.

The patient is ``ParametersDukeHeartLabelmaps.hold_out_case``, which Tutorial 6
leaves out of the population it builds the model from, so this fit measures
generalization rather than reconstruction.

Data Required
-------------
PCA model: Tutorial 6 output (``output/tutorial_06_duke_heart/pca_model.json``,
``pca_mean_surface.vtp``)
Patient: ``data/Duke-Heart-4DLabelmaps/<patient_case>/*_ref_labelmap.nii.gz``
ICON weights: ``tutorial_02_duke_heart_distancemap_finetune_icon.py`` output
(``network_weights/icon_duke_heart_distancemap/
icon_duke_heart_distancemap_model/checkpoints/network_weights_final.trch``),
optional -- the stock uniGradICON weights are used when it is absent.
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from monai_physio import (
    ContourTools,
    TestTools,
    WorkflowFitStatisticalModelToPatient,
)

# Only run if this script is not imported as a module
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    project_name = "tutorial_07_duke_heart"

    test_mode = TestTools.running_as_test()

    output_dir = DUKE_HEART.output_directory(test_mode) / project_name
    weights_dir = DUKE_HEART.weights_directory(test_mode)

    baselines_dir = repo_root / "tests" / "baselines"

    # PCA model + mean surface produced by Tutorial 6 (Duke Heart).
    pca_json = DUKE_HEART.pca_model_file(test_mode)
    pca_mean_file = DUKE_HEART.pca_mean_surface_file(test_mode)

    # The case whose reference frame plays the patient: the one Tutorial 6
    # leaves out of the model, so this fit is out of sample.
    patient_case = DUKE_HEART.hold_out_case

    number_of_pca_components = DUKE_HEART.pca_components(test_mode)
    data_dir = DUKE_HEART.hold_out_directory(test_mode)

    # Labels left out of the whole-heart structure, the same ones Tutorial 4
    # drops when it builds the surfaces the model was trained on, so the patient
    # and the model describe the same structure.
    interior_object_ids = DUKE_HEART.interior_object_ids

    # Contouring grid, shared with Tutorial 4 so the patient surface carries the
    # same level of detail as the model's training surfaces.
    surface_spacing_mm = DUKE_HEART.surface_spacing_mm
    smoothing_iterations = DUKE_HEART.surface_smoothing_iterations

    # Distance-map weights finetuned by
    # tutorial_02_duke_heart_distancemap_finetune_icon.py.  The heart has its
    # own finetuning run rather than reusing the lung one's: the heart
    # registration mask is far tighter, so heart distance maps saturate over a
    # shorter radius and do not share an intensity distribution with lung ones.
    icon_weights_path = (
        weights_dir
        / "icon_duke_heart_distancemap"
        / "icon_duke_heart_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    log_level = logging.INFO

    contour_tools = ContourTools(log_level=log_level)

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    if not pca_mean_file.exists():
        raise FileNotFoundError(
            f"Tutorial 6 PCA mean surface not found: {pca_mean_file}\n"
            "Run tutorials/tutorial_06_duke_heart_create_statistical_model.py first."
        )
    pca_mean = cast(pv.DataSet, pv.read(str(pca_mean_file)))

    # The fit is a PCA fit: without the model there is nothing to fit, and the
    # workflow's PCA outputs read below would never be set.
    if not pca_json.exists():
        raise FileNotFoundError(
            f"Tutorial 6 PCA model not found: {pca_json}\n"
            "Run tutorials/tutorial_06_duke_heart_create_statistical_model.py first."
        )
    with pca_json.open(encoding="utf-8") as f:
        pca_model: dict[str, Any] = json.load(f)

    patient_labelmap_files = sorted(
        (data_dir / patient_case).glob("*_ref_labelmap.nii.gz")
    )
    if not patient_labelmap_files:
        raise FileNotFoundError(
            f"No reference-frame labelmap for {patient_case} under {data_dir}.\n"
            "See data/Duke-Heart-4DLabelmaps/README.md."
        )
    patient_labelmap = itk.imread(str(patient_labelmap_files[0]))

    # The patient surface, contoured once and cached: the whole heart minus its
    # chamber cavities, which is the structure the model describes.
    heart_surface_file = output_dir / f"{project_name}_heart_surface.vtp"
    if not heart_surface_file.exists():
        labels = itk.GetArrayViewFromImage(patient_labelmap)
        heart_ids = [
            int(value)
            for value in np.unique(labels)
            if value != 0 and int(value) not in interior_object_ids
        ]
        heart_mask = itk.GetImageFromArray(np.isin(labels, heart_ids).astype(np.uint8))
        heart_mask.CopyInformation(patient_labelmap)
        heart_surface = contour_tools.extract_label_surfaces(
            heart_mask,
            isotropic_spacing_mm=surface_spacing_mm,
            smoothing_iterations=smoothing_iterations,
        )[1]
        heart_surface.save(str(heart_surface_file))
    heart_surface = cast(pv.PolyData, pv.read(str(heart_surface_file)))

    # Workflow initialization

    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=pca_mean,
        patient_models=[heart_surface],
        # This dataset carries no intensity image, so the workflow rasterizes
        # its own reference grid from the patient surface.
        patient_image=None,
        patient_labelmap=patient_labelmap,
        log_level=log_level,
        # The labels the whole-heart surface leaves out are the ones a distance
        # map must not measure to either.
        labelmap_interior_object_ids=interior_object_ids,
    )
    workflow.set_icp_transform_type(DUKE_HEART.icp_transform_type)
    workflow.set_mask_dilation_mm(DUKE_HEART.mask_dilation_mm)
    workflow.set_distancemap_squared_max(DUKE_HEART.distancemap_squared_max)
    workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model=pca_model,
        number_of_pca_components=number_of_pca_components,
        use_surface=False,
    )

    # The labelmap-to-labelmap stage registers distance maps, not intensities,
    # so it uses the distance-map-finetuned weights when they exist; without
    # them the tutorial still runs, on the stock uniGradICON weights.
    if icon_weights_path.exists():
        workflow.set_labelmap_to_labelmap_icon_weights_path(str(icon_weights_path))
    else:
        workflow.log_warning(
            "Finetuned distance-map ICON weights not found at %s; fitting with "
            "the stock uniGradICON weights. Run "
            "tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py to "
            "create them.",
            icon_weights_path,
        )

    # Workflow execution
    workflow_results = workflow.process()

    # Result saving
    registered_coefficients = workflow.pca_coefficients
    if registered_coefficients is not None:
        registered_coefficients_path = (
            output_dir / f"{project_name}_registered_coefficients.json"
        )
        with registered_coefficients_path.open(mode="w", encoding="utf-8") as f:
            json.dump(registered_coefficients.tolist(), f)

    # The Duke heart PCA model from Tutorial 6 is built from surfaces only, so
    # the model *is* a surface: the volume mesh and its bounding surface are the
    # same geometry and only the .vtp surfaces are written.
    template_surface = workflow.pca_template_model_surface
    assert template_surface is not None, (
        "pca_template_model_surface must be set after process()"
    )
    template_surface.save(str(output_dir / f"{project_name}_template_surface.vtp"))

    registered_surface = workflow_results["fitted_reference_mesh"]
    registered_surface.save(
        str(output_dir / f"{project_name}_template_surface_registered.vtp")
    )

    # Testing
    TestTools(
        class_name=project_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    try:
        pv.start_xvfb()
    except Exception:
        pass

    screenshots: list[Path] = []

    before_path = output_dir / f"{project_name}_model_before_registration.png"
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(pca_mean, color="dodgerblue", opacity=0.6)
    plotter.add_mesh(heart_surface, color="tomato", opacity=0.6)
    plotter.camera_position = "iso"
    plotter.screenshot(str(before_path))
    plotter.close()
    screenshots.append(before_path)

    after_path = output_dir / f"{project_name}_model_after_registration.png"
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(registered_surface, color="limegreen", opacity=0.7)
    plotter.add_mesh(heart_surface, color="tomato", opacity=0.4)
    plotter.camera_position = "iso"
    plotter.screenshot(str(after_path))
    plotter.close()
    screenshots.append(after_path)

    tutorial_results = {
        "registered_surface": registered_surface,
        "registered_coefficients": registered_coefficients,
        "screenshots": screenshots,
    }
