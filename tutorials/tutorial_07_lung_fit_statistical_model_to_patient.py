"""
Tutorial 7 (Lung): Fit Statistical Shape Model to Patient Data

Purpose
-------
Lung counterpart of ``tutorial_07_heart_fit_statistical_model_to_patient.py``.
The lung PCA model built by ``tutorial_06_lung_create_statistical_model.py`` is
fitted to the lungs segmented from a single patient chest CT.

Data Required
-------------
PCA model: Tutorial 6 output (``output/tutorial_06_lung/pca_model.json``,
``pca_mean_surface.vtp``)
Patient image: an ungated 3D chest CT,
``data/Chest-CT/Chest-CT.mha``, downloaded with
``monai-physio-download-data Chest-CT --directory data/Chest-CT``
(see ``data/Chest-CT/README.md`` for the data source and required citation)
ICON weights: ``tutorial_02_lung_distancemap_finetune_icon.py`` output
(``network_weights/icon_dirlab_4dct_distancemap/
icon_dirlab_4dct_distancemap_model/checkpoints/network_weights_final.trch``),
optional -- the stock uniGradICON weights are used when it is absent.
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import pyvista as pv
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from monai_physio import (
    ContourTools,
    SegmentNVSegmentCTMRI,
    TestTools,
    WorkflowConvertImageToVTK,
    WorkflowFitStatisticalModelToPatient,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    project_name = "tutorial_07_lung"

    test_mode = TestTools.running_as_test()

    output_dir = LUNG_CT_DIRLAB.output_directory(test_mode) / project_name
    weights_dir = LUNG_CT_DIRLAB.weights_directory(test_mode)

    baselines_dir = repo_root / "tests" / "baselines"

    # PCA model + mean surface produced by Tutorial 6 (lung).
    pca_json = LUNG_CT_DIRLAB.pca_model_file(test_mode)
    pca_mean_file = LUNG_CT_DIRLAB.pca_mean_surface_file(test_mode)

    number_of_pca_components = LUNG_CT_DIRLAB.pca_components(test_mode)

    # The study Tutorial 6 leaves out of the model, so this fit is out of sample.
    patient_image_file = (
        LUNG_CT_DIRLAB.hold_out_directory(test_mode) / LUNG_CT_DIRLAB.hold_out_case
    )

    # Distance-map weights finetuned on DIR-Lab by
    # tutorial_02_lung_distancemap_finetune_icon.py; see
    # WorkflowFinetuneICONRegistration.expected_weights_path().
    icon_weights_path = (
        weights_dir
        / "icon_dirlab_4dct_distancemap"
        / "icon_dirlab_4dct_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    log_level = logging.INFO

    # The same segmenter and surface-extraction workflow used by Tutorial 6, so
    # the patient surface matches the topology the PCA model was built from.
    segmentation_method = SegmentNVSegmentCTMRI(log_level=log_level)
    segmentation_workflow = WorkflowConvertImageToVTK(
        segmentation_method=segmentation_method, log_level=log_level
    )
    contour_tools = ContourTools(log_level=log_level)

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    if not pca_mean_file.exists():
        raise FileNotFoundError(
            f"Tutorial 6 PCA mean surface not found: {pca_mean_file}\n"
            "Run tutorials/tutorial_06_lung_create_statistical_model.py first."
        )
    pca_mean = cast(pv.DataSet, pv.read(str(pca_mean_file)))

    pca_model: Optional[dict[str, Any]] = None
    if pca_json.exists():
        with pca_json.open(encoding="utf-8") as f:
            pca_model = json.load(f)

    if not patient_image_file.exists():
        raise FileNotFoundError(
            f"Patient chest CT not found: {patient_image_file}\n"
            "Run: monai-physio-download-data Chest-CT --directory data/Chest-CT"
        )
    patient_image = itk.imread(str(patient_image_file))

    lung_surface_file = output_dir / f"{project_name}_lung_surface.vtp"
    lung_labelmap_file = output_dir / f"{project_name}_lung_labelmap.nii.gz"
    if not (lung_surface_file.exists() and lung_labelmap_file.exists()):
        segmentation_result = segmentation_workflow.process(
            input_image=patient_image,
            anatomy_groups=["lung"],
            surface_reduction_rate=LUNG_CT_DIRLAB.surface_reduction_rate,
            extract_label_surfaces=True,
        )
        contour_tools.save_combined_surfaces(
            segmentation_result["label_surfaces"], str(lung_surface_file)
        )
        itk.imwrite(
            segmentation_result["labelmap"], str(lung_labelmap_file), compression=True
        )
    lung_surface = cast(pv.PolyData, pv.read(str(lung_surface_file)))
    lung_labelmap = itk.imread(str(lung_labelmap_file))

    # Workflow initialization

    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=pca_mean,
        patient_models=[lung_surface],
        patient_image=patient_image,
        patient_labelmap=lung_labelmap,
        log_level=log_level,
    )
    if pca_model is not None:
        workflow.set_use_pca_registration(
            use_pca_registration=True,
            pca_model=pca_model,
            number_of_pca_components=number_of_pca_components,
            use_surface=False,
        )

    workflow.set_icp_transform_type(LUNG_CT_DIRLAB.icp_transform_type)
    workflow.set_mask_dilation_mm(LUNG_CT_DIRLAB.mask_dilation_mm)
    workflow.set_distancemap_squared_max(LUNG_CT_DIRLAB.distancemap_squared_max)

    # The labelmap-to-labelmap stage registers distance maps, not intensities,
    # so it uses the distance-map-finetuned weights when they exist; without
    # them the tutorial still runs, on the stock uniGradICON weights.
    if icon_weights_path.exists():
        workflow.set_labelmap_to_labelmap_icon_weights_path(str(icon_weights_path))
    else:
        workflow.log_warning(
            "Finetuned distance-map ICON weights not found at %s; fitting with "
            "the stock uniGradICON weights. Run "
            "tutorials/tutorial_02_lung_distancemap_finetune_icon.py to create "
            "them.",
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

    # The lung PCA model from Tutorial 6 is built from surfaces only, so the
    # model *is* a surface: the volume mesh and its bounding surface are the
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

    registered_pca_surface = workflow.pca_template_model_surface
    assert registered_pca_surface is not None, (
        "pca_template_model_surface must be set after process()"
    )
    registered_pca_surface.save(
        str(output_dir / f"{project_name}_pca_surface_registered.vtp")
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
    plotter.add_mesh(lung_surface, color="tomato", opacity=0.6)
    plotter.camera_position = "iso"
    plotter.screenshot(str(before_path))
    plotter.close()
    screenshots.append(before_path)

    after_path = output_dir / f"{project_name}_model_after_registration.png"
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(registered_surface, color="limegreen", opacity=0.7)
    plotter.add_mesh(lung_surface, color="tomato", opacity=0.4)
    plotter.camera_position = "iso"
    plotter.screenshot(str(after_path))
    plotter.close()
    screenshots.append(after_path)

    tutorial_results = {
        "registered_surface": registered_surface,
        "screenshots": screenshots,
    }
