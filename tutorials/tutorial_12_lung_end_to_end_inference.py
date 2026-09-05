"""
Tutorial 12 (Lung, MGN): Predict Lung Motion Straight From a 4D CT

Purpose
-------
The whole inference pipeline for one case in one script, starting at the images
in ``data/`` rather than at another tutorial's output.  Tutorial 10 predicts the
same motion but reads the fit Tutorial 8 wrote for the case; this one performs
that fit itself, so the only inputs are the case's CT, the shape model and the
trained network.

The case is ``ParametersLungCTDirLab.mgn_hold_out_case``, held out of the
Tutorial 9 training, so the prediction measures generalization rather than
recall.

1. Read the case's gated CT sequence.  The respiratory stages come from the
   ``T{PP}`` filenames, so the acquisition itself says what is predicted.

2. Segment the reference phase (``T70``) with ``SegmentNVSegmentCTMRI``, the
   segmenter the lung shape model was built with, and extract its lung surface.

3. Fit the lung PCA model to that phase with
   :class:`monai_physio.WorkflowFitStatisticalModelToPatient` and PCA-based
   registration.  This is what puts the model in this patient: it yields the
   case's PCA coefficients, which the network is conditioned on, and the fitted
   SSM surface, whose points the predicted displacements are added to.

4. Predict every stage with the Tutorial 9 MeshGraphNet, carry the reference
   CT through each stage's deformation, and write the series as VTP surfaces
   and one animated USD --- :meth:`WorkflowInferMovement.process_time_series`.

No phase registration happens anywhere here.  Tutorial 8 registers every phase
to build the training targets a network needs; a trained network predicts the
motion instead, which is the point of the chain.

Data Required
-------------
  * ``data/DirLab-4DCT/<case>_T??.mha`` - the gated CT sequence
  * ``output/tutorial_06_lung/`` - lung PCA model + mean surface
  * ``network_weights/physicsnemo_mgn_lung_motion/`` - Tutorial 9 checkpoint
  * ``network_weights/icon_dirlab_4dct_distancemap/`` - Tutorial 2 weights,
    optional; the stock uniGradICON weights are used without them

Outputs (under ``output/tutorial_12_lung/<case>/``)
---------------------------------------------------
The directory is deleted and rebuilt on every run: nothing is reused, so the
runtimes below are the cost of the whole pipeline from scratch.

  * ``<case>_T70.vtp``, ``<case>_T70_labelmap.nii.gz`` - reference segmentation
  * ``<case>_ssm_pca_coefficients.json`` - this patient's shape parameters
  * ``<case>_ssm_surface.vtp``           - the model fitted to the reference phase
  * ``<case>_ssm_pca_coefficients_s{TTT}_pred.vtp``   - predicted surface per stage
  * ``<case>_ssm_pca_coefficients_s{TTT}_warped.mha`` - CT carried to that stage
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
import pyvista as pv
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from monai_physio import (
    ContourTools,
    SegmentNVSegmentCTMRI,
    TestTools,
    WorkflowConvertImageToVTK,
    WorkflowFitStatisticalModelToPatient,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)


def _respiratory_stage_from_filename(image_file: Path) -> float:
    """Extract the normalized respiratory stage [0, 1] from a ``T{PP}`` filename stem."""
    for part in image_file.stem.split("_"):
        if part.startswith("T") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse respiratory phase from filename: {image_file}")


def _record_step(times_s: dict[str, float], step: str, started: float) -> float:
    """Store the seconds elapsed since ``started`` under ``step``; return a new mark."""
    now = time.perf_counter()
    times_s[step] = now - started
    return now


# Only run if this script is not imported as a module

# nnUNetv2 and torch spawn worker processes. On Windows the spawn start method
# re-imports this script in each child; without the __name__ == "__main__" guard
# around top-level work, that re-import would restart the whole pipeline in every
# worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_12_lung_end_to_end_inference"

    # Case to predict: the case Tutorial 9 held out of training.
    case_id = LUNG_CT_DIRLAB.mgn_hold_out_case
    # Phase the shape model is fitted to. Tutorial 6 builds the lung PCA model
    # from the T70 surfaces and Tutorial 9 trained on displacements from that
    # phase, so the network's reference frame is this one.
    reference_phase = "T70"

    test_mode = TestTools.running_as_test()
    # Keep a test run out of the directories a full run reads and writes.
    weights_dir = LUNG_CT_DIRLAB.weights_directory(test_mode)

    # PCA model + mean surface produced by Tutorial 6 (lung).
    pca_model_file = LUNG_CT_DIRLAB.pca_model_file(test_mode)
    pca_mean_file = LUNG_CT_DIRLAB.pca_mean_surface_file(test_mode)
    # Weights Tutorial 9 trained, and the checkpoint epoch to infer with; None
    # uses the final weights.
    model_dir = LUNG_CT_DIRLAB.mgn_weights_directory(test_mode)
    epoch: Optional[int] = None

    # Distance-map weights finetuned on DIR-Lab by
    # tutorial_02_lung_distancemap_finetune_icon.py, used by the
    # labelmap-to-labelmap stage of the SSM fit.
    icon_distancemap_weights_path = (
        weights_dir
        / "icon_dirlab_4dct_distancemap"
        / "icon_dirlab_4dct_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the CT is resampled through.
    smoothing_sigma_mm = 10.0

    output_dir = (
        LUNG_CT_DIRLAB.output_directory(test_mode) / "tutorial_12_lung" / case_id
    )
    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # Wall-clock per step. Every run starts from an empty output directory, so
    # these are the cost of computing each step, not of re-reading it.
    step_times_s: dict[str, float] = {}
    step_start = time.perf_counter()

    data_dir = LUNG_CT_DIRLAB.input_directory(test_mode)
    number_of_pca_components = LUNG_CT_DIRLAB.pca_components(test_mode)

    # Directory setup and data reading

    # The run is from scratch: the output directory is emptied first so every
    # step below is computed and timed, never read back from a previous run.
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    for required_file, hint in (
        (pca_model_file, "tutorial_06_lung_create_statistical_model.py"),
        (pca_mean_file, "tutorial_06_lung_create_statistical_model.py"),
        (model_dir / "mgn_stage_model.pt", "tutorial_09_lung_train_physicsnemo_mgn.py"),
    ):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required input not found: {required_file}\n"
                f"Run tutorials/{hint} first."
            )

    frame_files = sorted(data_dir.glob(f"{case_id}_T??.mha"))
    if not frame_files:
        raise FileNotFoundError(
            f"No {case_id}_T??.mha frames found under {data_dir}.\n"
            "See data/DirLab-4DCT/README.md for download instructions."
        )
    reference_file = data_dir / f"{case_id}_{reference_phase}.mha"
    if not reference_file.exists():
        raise FileNotFoundError(
            f"Reference phase not found: {reference_file}; it is the phase the "
            "shape model is fitted to and the predicted motion starts from."
        )

    pca_mean_surface = cast(pv.DataSet, pv.read(str(pca_mean_file)))
    with pca_model_file.open(encoding="utf-8") as f:
        pca_model = json.load(f)

    # The Tutorial 2 distance-map weights are used when they exist; without them
    # the tutorial still runs, on the stock uniGradICON weights.
    use_finetuned_distancemap_weights = icon_distancemap_weights_path.exists()
    if not use_finetuned_distancemap_weights:
        logger.warning(
            "Finetuned distance-map ICON weights not found at %s; fitting the SSM "
            "with the stock uniGradICON weights. Run "
            "tutorials/tutorial_02_lung_distancemap_finetune_icon.py to create "
            "them.",
            icon_distancemap_weights_path,
        )

    # Step 1: the acquisition says which stages there are to predict.
    stages = [_respiratory_stage_from_filename(path) for path in frame_files]
    reference_image = itk.imread(str(reference_file))
    logger.info("Case %s: %d respiratory phases", case_id, len(stages))
    step_start = _record_step(step_times_s, "read_inputs", step_start)

    # Step 2: segment the lungs in the reference phase. This is the segmenter the
    # Tutorial 6 shape model was built with, so the surface the fit sees is the
    # kind of surface the model describes.
    lung_surface_file = output_dir / f"{reference_file.stem}.vtp"
    lung_labelmap_file = output_dir / f"{reference_file.stem}_labelmap.nii.gz"
    logger.info("Segmenting the reference phase %s", reference_file.name)
    contour_tools = ContourTools(log_level=log_level)
    segmentation_result = WorkflowConvertImageToVTK(
        segmentation_method=SegmentNVSegmentCTMRI(log_level=log_level),
        log_level=log_level,
    ).process(
        input_image=reference_image,
        anatomy_groups=[LUNG_CT_DIRLAB.anatomy_group],
        surface_reduction_rate=LUNG_CT_DIRLAB.surface_reduction_rate,
        extract_label_surfaces=True,
    )
    contour_tools.save_combined_surfaces(
        segmentation_result["label_surfaces"], str(lung_surface_file)
    )
    lung_labelmap = segmentation_result["labelmap"]
    itk.imwrite(lung_labelmap, str(lung_labelmap_file), compression=True)
    # Read back rather than combining in memory: save_combined_surfaces is what
    # merges the per-label surfaces into the one surface the fit is given.
    lung_surface = cast(pv.PolyData, pv.read(str(lung_surface_file)))
    step_start = _record_step(step_times_s, "segment_reference", step_start)

    # Step 3: fit the shape model to the reference phase. The coefficients are
    # what the network is conditioned on; the fitted surface is what its
    # displacements are added to.
    fit_workflow = WorkflowFitStatisticalModelToPatient(
        template_model=pca_mean_surface,
        patient_models=[lung_surface],
        patient_image=reference_image,
        patient_labelmap=lung_labelmap,
        log_level=log_level,
    )
    fit_workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model=pca_model,
        number_of_pca_components=number_of_pca_components,
        use_surface=False,
    )
    fit_workflow.set_icp_transform_type(LUNG_CT_DIRLAB.icp_transform_type)
    fit_workflow.set_mask_dilation_mm(LUNG_CT_DIRLAB.mask_dilation_mm)
    fit_workflow.set_distancemap_squared_max(LUNG_CT_DIRLAB.distancemap_squared_max)
    if use_finetuned_distancemap_weights:
        fit_workflow.set_labelmap_to_labelmap_icon_weights_path(
            str(icon_distancemap_weights_path)
        )
    fit_result = fit_workflow.process()

    pca_coefficients = fit_workflow.pca_coefficients
    assert pca_coefficients is not None
    pca_coefficients_file = output_dir / f"{case_id}_ssm_pca_coefficients.json"
    with pca_coefficients_file.open(mode="w", encoding="utf-8") as f:
        json.dump(pca_coefficients.tolist(), f)

    # The lung PCA model from Tutorial 6 is built from surfaces only, so the
    # model *is* a surface here: only the .vtp is written.
    fitted_reference_mesh_file = output_dir / f"{case_id}_ssm_surface.vtp"
    fit_result["fitted_reference_mesh"].save(str(fitted_reference_mesh_file))
    logger.info("Fitted the lung model to %s", reference_file.name)
    step_start = _record_step(step_times_s, "fit_shape_model", step_start)

    # Step 4: predict the whole cycle, warp the reference CT through each stage's
    # deformation, and write the animated USD. -1000 HU is air, the value a CT
    # grid samples outside itself. There is no ground truth to score against: no
    # phase was registered, which is what the network replaces.
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
        reference_image=reference_image,
        warp_interpolation="linear",
        warp_background_value=-1000.0,
        smoothing_sigma_mm=smoothing_sigma_mm,
        usd_project_name=f"{case_id}_mgn_motion",
        anatomy_type="lung",
        separate_by_connectivity=True,
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
