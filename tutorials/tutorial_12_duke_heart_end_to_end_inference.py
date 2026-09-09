"""
Tutorial 12 (Duke Heart, MGN): Predict Heart Motion Straight From Gated Labelmaps

Purpose
-------
Duke counterpart of ``tutorial_12_lung_end_to_end_inference.py``: the whole
inference pipeline for one case in one script, starting at the labelmaps in
``data/`` rather than at another tutorial's output.  Tutorial 10 predicts the
same motion but reads the fit Tutorial 8 wrote for the case; this one performs
that fit itself, so the only inputs are the case's gated frames, the shape model
and the trained network.

The case is ``ParametersDukeHeartLabelmaps.hold_out_case``, held out of every fit
in this chain, so the prediction measures generalization rather than recall.

1. Read the case's gated frames.  No segmentation is needed: this cohort ships
   one labelmap per frame.  The cardiac stages come from the ``g{PPP}`` gate
   tags, so the acquisition itself says what is predicted.

2. Contour the reference frame's heart --- the whole heart minus its chamber
   cavities, the structure the shape model describes.  The chambers and the
   great vessels are left out by ``ParametersDukeHeartLabelmaps``.

3. Fit the heart PCA model to that frame with
   :class:`monai_physio.WorkflowFitStatisticalModelToPatient` and PCA-based
   registration.  This is what puts the model in this patient: it yields the
   case's PCA coefficients, which the network is conditioned on, and the fitted
   SSM surface, whose points the predicted displacements are added to.

4. Predict every stage with the Tutorial 9 MeshGraphNet, carry the reference
   frame's labelmap through each stage's deformation, and write the series as
   VTP surfaces and one animated USD ---
   :meth:`WorkflowInferMovement.process_time_series`.

No frame registration happens anywhere here.  Tutorial 8 registers every frame
to build the training targets a network needs; a trained network predicts the
motion instead, which is the point of the chain.

Data Required
-------------
  * ``data/Duke-Heart-4DLabelmaps/<case>/*_labelmap.nii.gz`` - gated frames
  * ``output/tutorial_06_duke_heart/`` - heart PCA model + mean surface
  * ``network_weights/physicsnemo_mgn_duke_heart_motion/`` - Tutorial 9 checkpoint
  * ``network_weights/icon_duke_heart_distancemap/`` - Tutorial 2 weights,
    optional; the stock uniGradICON weights are used without them

Outputs (under ``output/tutorial_12_duke_heart/<case>/``)
---------------------------------------------------------
The directory is deleted and rebuilt on every run: nothing is reused, so the
runtimes below are the cost of the whole pipeline from scratch.

  * ``<case>_heart_surface.vtp``         - contoured reference-frame heart
  * ``<case>_ssm_pca_coefficients.json`` - this patient's shape parameters
  * ``<case>_ssm_surface.vtp``           - the model fitted to the reference frame
  * ``<case>_ssm_pca_coefficients_s{TTT}_pred.vtp``   - predicted surface per stage
  * ``<case>_ssm_pca_coefficients_s{TTT}_warped.mha`` - labelmap at that stage
  * ``<case>_mgn_motion.usd``            - animated predicted motion
  * ``<case>_runtimes.csv``              - wall-clock seconds per pipeline step
"""

# Imports
from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from monai_physio import (
    ContourTools,
    TestTools,
    WorkflowFitStatisticalModelToPatient,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)

LABELMAP_SUFFIX = "_labelmap.nii.gz"


def _cardiac_stage_from_filename(labelmap_file: Path) -> float:
    """Extract the normalized cardiac stage [0, 1] from a ``g{PPP}`` filename stem."""
    for part in labelmap_file.name.split("_"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse cardiac gate from filename: {labelmap_file}")


def _record_step(times_s: dict[str, float], step: str, started: float) -> float:
    """Store the seconds elapsed since ``started`` under ``step``; return a new mark."""
    now = time.perf_counter()
    times_s[step] = now - started
    return now


# Only run if this script is not imported as a module

# The registration backends and torch spawn worker processes. On Windows the
# spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the whole pipeline in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_12_duke_heart_end_to_end_inference"

    # Case to predict: the case held out of every fit in this chain.
    case_id = DUKE_HEART.hold_out_case

    test_mode = TestTools.running_as_test()
    # Keep a test run out of the directories a full run reads and writes.
    weights_dir = DUKE_HEART.weights_directory(test_mode)

    # PCA model + mean surface produced by Tutorial 6 (Duke Heart).
    pca_model_file = DUKE_HEART.pca_model_file(test_mode)
    pca_mean_file = DUKE_HEART.pca_mean_surface_file(test_mode)
    # Weights Tutorial 9 trained, and the checkpoint epoch to infer with; None
    # uses the final weights.
    model_dir = DUKE_HEART.mgn_weights_directory(test_mode)
    epoch: Optional[int] = None

    # Distance-map weights finetuned by
    # tutorial_02_duke_heart_distancemap_finetune_icon.py, used by the
    # labelmap-to-labelmap stage of the SSM fit.
    icon_weights_path = (
        weights_dir
        / "icon_duke_heart_distancemap"
        / "icon_duke_heart_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the labelmap is resampled through.
    smoothing_sigma_mm = 10.0

    output_dir = (
        DUKE_HEART.output_directory(test_mode) / "tutorial_12_duke_heart" / case_id
    )
    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # Wall-clock per step. Every run starts from an empty output directory, so
    # these are the cost of computing each step, not of re-reading it.
    step_times_s: dict[str, float] = {}
    step_start = time.perf_counter()

    labelmap_dir = DUKE_HEART.hold_out_directory(test_mode) / case_id
    number_of_pca_components = DUKE_HEART.pca_components(test_mode)

    # Directory setup and data reading

    # The run is from scratch: the output directory is emptied first so every
    # step below is computed and timed, never read back from a previous run.
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    for required_file, hint in (
        (pca_model_file, "tutorial_06_duke_heart_create_statistical_model.py"),
        (pca_mean_file, "tutorial_06_duke_heart_create_statistical_model.py"),
        (
            model_dir / "mgn_stage_model.pt",
            "tutorial_09_duke_heart_train_physicsnemo_mgn.py",
        ),
    ):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required input not found: {required_file}\n"
                f"Run tutorials/{hint} first."
            )

    frame_files = sorted(labelmap_dir.glob(f"*{LABELMAP_SUFFIX}"))
    if not frame_files:
        raise FileNotFoundError(
            f"No gated labelmaps found in {labelmap_dir}.\n"
            "See data/Duke-Heart-4DLabelmaps/README.md."
        )
    reference_files = [
        path for path in frame_files if path.name.endswith(f"_ref{LABELMAP_SUFFIX}")
    ]
    if not reference_files:
        raise FileNotFoundError(
            f"No *_ref{LABELMAP_SUFFIX} frame in {labelmap_dir}; that frame is the "
            "one the shape model is fitted to and the predicted motion starts from."
        )
    reference_file = reference_files[0]

    pca_mean_surface = cast(pv.DataSet, pv.read(str(pca_mean_file)))
    with pca_model_file.open(encoding="utf-8") as f:
        pca_model = json.load(f)

    # The Tutorial 2 distance-map weights are used when they exist; without them
    # the tutorial still runs, on the stock uniGradICON weights.
    use_finetuned_weights = icon_weights_path.exists()
    if not use_finetuned_weights:
        logger.warning(
            "Finetuned distance-map ICON weights not found at %s; fitting the SSM "
            "with the stock uniGradICON weights. Run "
            "tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py to "
            "create them.",
            icon_weights_path,
        )

    # Step 1: the acquisition says which stages there are to predict.
    stages = [_cardiac_stage_from_filename(path) for path in frame_files]
    reference_labelmap = itk.imread(str(reference_file))
    logger.info("Case %s: %d gated frames", case_id, len(stages))
    step_start = _record_step(step_times_s, "read_inputs", step_start)

    # Step 2: contour the reference frame's heart, minus the cavities and the
    # vessels the shape model leaves out, on the pitch Tutorial 4 contoured the
    # model's training surfaces at.
    heart_surface_file = output_dir / f"{case_id}_heart_surface.vtp"
    contour_tools = ContourTools(log_level=log_level)
    labels = itk.GetArrayViewFromImage(reference_labelmap)
    wall_ids = [
        int(value)
        for value in np.unique(labels)
        if value != 0 and int(value) not in DUKE_HEART.interior_object_ids
    ]
    heart_mask = itk.GetImageFromArray(np.isin(labels, wall_ids).astype(np.uint8))
    heart_mask.CopyInformation(reference_labelmap)
    heart_surface = contour_tools.extract_label_surfaces(
        heart_mask,
        isotropic_spacing_mm=DUKE_HEART.surface_spacing_mm,
        smoothing_iterations=DUKE_HEART.surface_smoothing_iterations,
    )[1]
    heart_surface.save(str(heart_surface_file))
    step_start = _record_step(step_times_s, "contour_reference", step_start)

    # Step 3: fit the shape model to the reference frame. The coefficients are
    # what the network is conditioned on; the fitted surface is what its
    # displacements are added to. This data carries no intensity image, so the
    # workflow rasterizes its own reference grid from the patient surface.
    fit_workflow = WorkflowFitStatisticalModelToPatient(
        template_model=pca_mean_surface,
        patient_models=[heart_surface],
        patient_image=None,
        patient_labelmap=reference_labelmap,
        labelmap_interior_object_ids=DUKE_HEART.interior_object_ids,
        log_level=log_level,
    )
    fit_workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model=pca_model,
        number_of_pca_components=number_of_pca_components,
        use_surface=False,
    )
    fit_workflow.set_icp_transform_type(DUKE_HEART.icp_transform_type)
    fit_workflow.set_mask_dilation_mm(DUKE_HEART.mask_dilation_mm)
    fit_workflow.set_distancemap_squared_max(DUKE_HEART.distancemap_squared_max)
    if use_finetuned_weights:
        fit_workflow.set_labelmap_to_labelmap_icon_weights_path(str(icon_weights_path))
    fit_result = fit_workflow.process()

    pca_coefficients = fit_workflow.pca_coefficients
    assert pca_coefficients is not None
    pca_coefficients_file = output_dir / f"{case_id}_ssm_pca_coefficients.json"
    with pca_coefficients_file.open(mode="w", encoding="utf-8") as f:
        json.dump(pca_coefficients.tolist(), f)

    # The heart PCA model from Tutorial 6 (Duke Heart) is built from surfaces
    # only, so the model *is* a surface here: only the .vtp is written.
    fitted_reference_mesh_file = output_dir / f"{case_id}_ssm_surface.vtp"
    fit_result["fitted_reference_mesh"].save(str(fitted_reference_mesh_file))
    logger.info("Fitted the heart model to %s", reference_file.name)
    step_start = _record_step(step_times_s, "fit_shape_model", step_start)

    # Step 4: predict the whole cycle, warp the reference frame's labelmap
    # through each stage's deformation, and write the animated USD. Nearest
    # neighbor keeps the label values discrete. The SSM is one structure, so the
    # USD surface is kept whole rather than split by connectivity. There is no
    # ground truth to score against: no frame was registered, which is what the
    # network replaces.
    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir, epoch=epoch, log_level=log_level
    )
    infer_result = WorkflowInferMovement(
        infer_workflow, log_level=log_level
    ).process_time_series(
        shape_parameters=pca_coefficients_file,
        stages=stages,
        output_directory=output_dir,
        fitted_reference_mesh=fitted_reference_mesh_file,
        reference_image=reference_labelmap,
        warp_interpolation="nearest",
        warp_background_value=0.0,
        smoothing_sigma_mm=smoothing_sigma_mm,
        usd_project_name=f"{case_id}_mgn_motion",
        anatomy_type="heart",
        separate_by_connectivity=False,
    )
    logger.info("USD: %s", infer_result["usd_file"])
    step_start = _record_step(step_times_s, "predict_and_warp", step_start)

    tutorial_results: dict[str, Any] = dict(infer_result)
    tutorial_results["pca_coefficients_file"] = pca_coefficients_file
    tutorial_results["fitted_reference_mesh_file"] = fitted_reference_mesh_file

    # Testing: the fitted reference surface beside the first predicted stage.
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=repo_root / "tests" / "baselines" / class_name,
        log_level=log_level,
    )
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(fitted_reference_mesh_file))),
            "fitted_reference_surface.png",
            camera_position="iso",
            color="steelblue",
        ),
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(infer_result["predicted_surfaces"][0]))),
            "predicted_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
    ]
    step_start = _record_step(step_times_s, "screenshots", step_start)

    # Reporting: the per-step wall clock, on screen and as a CSV row per step.
    step_times_s["total"] = sum(step_times_s.values())
    runtime_file = output_dir / f"{case_id}_runtimes.csv"
    with runtime_file.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "step", "seconds"])
        writer.writerows(
            [case_id, step, f"{seconds:.3f}"] for step, seconds in step_times_s.items()
        )

    logger.info("Runtime, seconds")
    for step, seconds in step_times_s.items():
        logger.info("  %-18s %8.1f", step, seconds)
    logger.info("Runtime CSV: %s", runtime_file)

    tutorial_results["step_times_s"] = step_times_s
    tutorial_results["runtime_file"] = runtime_file
