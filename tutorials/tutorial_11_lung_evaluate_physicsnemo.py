"""
Tutorial 11 (Lung, MGN): Score Predicted Lung Motion Per Lobe

Purpose
-------
Measures how close the size and shape of the lung inferred by Tutorial 10 are to
the lung actually imaged, one respiratory phase at a time and one lobe at a
time.  The case is ``ParametersLungCTDirLab.mgn_hold_out_case``, held out of the
Tutorial 9 training, so this scores generalization rather than recall.

1. Build the ground truth: segment every gated CT frame of the case
   independently, giving one labelmap per respiratory phase whose lobes were
   never seen by the shape model or by the network.

2. Score the prediction: :class:`monai_physio.WorkflowEvaluateMovement` carries
   the reference phase's labelmap into every other phase with the network's own
   deformation, and compares the result to that phase's segmentation --- volume
   difference and surface RMSE per lobe.  Dice is left out: a lobe barely
   changes shape over a breath compared to how big it is, so the overlap
   fraction stays above 0.96 however well or badly the motion is predicted, and
   describes the lobe rather than the motion.

3. Score the motion point by point: with Tutorial 8's per-phase SSM surfaces as
   the ground truth, the report also carries the distance between where the
   network puts each mesh point and where the shape model fitted it --- RMS,
   95th percentile and maximum, per lobe and per phase.  The lung shape model
   tags every triangle with its lobe, so a mesh point is scored under the lobe
   the model itself says it belongs to.  A lobe whose predicted motion is right on average can
   still be wrong everywhere, and only this measure says so.

4. Write ``evaluation_report.md`` and ``evaluation_metrics.csv``, both carrying
   the hold-out case name, the case's shape parameters, and the network weights
   path with its dates.  Each metric is reported both averaged over the phases
   and at the phase it is worst at.

Step 1 costs one segmentation pass per phase on first run and is cached
afterwards; Steps 2 to 4 are the workflow, so this script only chooses the case,
the lobes and the ground truth.

Data Required
-------------
  * ``data/DirLab-4DCT/<case>_T??.mha``  - the gated CT sequence
  * ``output/tutorial_08_lung/<case>/``  - Tutorial 8 SSM surface + coefficients
  * ``network_weights/physicsnemo_mgn_lung_motion/`` - Tutorial 9 checkpoint

Outputs (under ``output/tutorial_11_lung/<case>/``)
---------------------------------------------------
  * ``evaluation_report.md``    - per-lobe accuracy of the prediction, mean and
    worst case, with the per-point displacement error per phase
  * ``evaluation_metrics.csv``  - one row per stage and lobe, each carrying
    that lobe's displacement error (RMS, 95th percentile, maximum)
  * ``volume_vs_stage.png``     - each lobe's volume across the stages
  * ``ground_truth/<case>_T{PP}_labelmap.nii.gz`` - cached per-phase segmentation
  * ``<case>_ssm_pca_coefficients_s{TTT}_pred.vtp`` - predicted surface per stage,
    carrying the displacement point-data arrays the ``include_*`` switches ask for
  * ``displacement_per_point.csv`` - every mesh point's predicted and true
    displacement at every phase; written only when ``report_displacement_data``
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import pyvista as pv
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from monai_physio import (
    EvaluateMovementLung,
    TestTools,
    WorkflowEvaluateMovement,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)


# Only run if this script is not imported as a module

# nnUNetv2 and torch spawn worker processes. On Windows the spawn start method
# re-imports this script in each child; without the __name__ == "__main__" guard
# around top-level work, that re-import would restart the whole evaluation in
# every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_11_lung_evaluate_physicsnemo"

    # Case to score: the case Tutorial 9 held out of training.
    case_id = LUNG_CT_DIRLAB.mgn_hold_out_case
    # Phase Tutorial 8 fitted the SSM to, and therefore the phase whose anatomy
    # the predicted deformations carry into every other phase.
    reference_phase = "T70"

    # Fitted SSM surface and PCA coefficients written by Tutorial 8 (lung).
    test_mode = TestTools.running_as_test()
    # Keep a test run out of the directories a full run reads and writes.
    case_dir = LUNG_CT_DIRLAB.output_directory(test_mode) / "tutorial_08_lung" / case_id
    # Weights Tutorial 9 trained, and the checkpoint epoch Tutorial 10 infers
    # with; None uses the final weights.
    model_dir = LUNG_CT_DIRLAB.mgn_weights_directory(test_mode)
    epoch: Optional[int] = None

    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the labelmap is resampled through.
    smoothing_sigma_mm = 10.0
    # Isotropic pitch every metric is measured on.  Coarser than the CT, whose
    # in-plane pitch is finer than the accuracy being reported, and fine enough
    # that a lobe boundary is not quantized away.
    evaluation_spacing_mm = 2.0

    # Per-point displacement reporting, all off by default.  The first writes
    # one CSV row per mesh point per phase; the rest carry the same quantities
    # as point data on each phase's predicted surface.  Every one of them except
    # the predicted displacement is measured against Tutorial 8's per-phase SSM
    # surfaces, the only geometry that shares this mesh's point ordering.
    report_displacement_data = False
    include_predicted_displacements = False
    include_true_displacements = False
    # On: the point-by-point error is the one measure a displacement predicted
    # in the wrong direction cannot hide in, and it costs one mesh read a phase.
    include_displacement_error = True

    output_dir = (
        LUNG_CT_DIRLAB.output_directory(test_mode) / "tutorial_11_lung" / case_id
    )
    ground_truth_dir = output_dir / "ground_truth"
    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    data_dir = LUNG_CT_DIRLAB.input_directory(test_mode)

    # Directory setup and data reading

    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    fitted_reference_mesh_file = case_dir / f"{case_id}_ssm_surface.vtp"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    for required_file in (fitted_reference_mesh_file, pca_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 8 output not found: {required_file}\n"
                "Run tutorials/tutorial_08_lung_fit_model_to_4d_patients.py first."
            )

    # Step 1: the cohort assembles what this case is scored against.  Every
    # gated frame is segmented on its own, so the lobes each phase is scored
    # against came from that phase's image rather than from a registration or a
    # shape-model fit; segmentation dominates the runtime, so each labelmap is
    # cached in ``ground_truth_dir`` and reused on a re-run.  Tutorial 8's
    # per-phase fits come along too: they are the only geometry that shares the
    # fitted reference mesh's point ordering, so they are what the
    # point-by-point error is measured against.
    cohort = EvaluateMovementLung(reference_phase=reference_phase, log_level=log_level)
    ground_truth = cohort.assemble_ground_truth(
        case_id=case_id,
        frame_directory=data_dir,
        fit_directory=case_dir,
        cache_directory=ground_truth_dir,
    )

    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir, epoch=epoch, log_level=log_level
    )
    evaluate_workflow = WorkflowEvaluateMovement(
        movement_workflow=WorkflowInferMovement(infer_workflow, log_level=log_level),
        cohort=cohort,
        log_level=log_level,
    )
    result = evaluate_workflow.process(
        case_id=case_id,
        shape_parameters=pca_file,
        fitted_reference_mesh=fitted_reference_mesh_file,
        ground_truth=ground_truth,
        output_directory=output_dir,
        smoothing_sigma_mm=smoothing_sigma_mm,
        evaluation_spacing_mm=evaluation_spacing_mm,
        report_displacement_data=report_displacement_data,
        include_predicted_displacements=include_predicted_displacements,
        include_true_displacements=include_true_displacements,
        include_displacement_error=include_displacement_error,
    )

    # Step 3: the report and the CSV are written by the workflow.
    logger.info("Report: %s", result["report_file"])
    logger.info("Metrics: %s", result["csv_file"])
    if result["displacement_data_file"] is not None:
        logger.info("Displacements: %s", result["displacement_data_file"])
    logger.info(
        "Displacement error: rms=%.3f mm  95th=%.3f mm  max=%.3f mm",
        result["displacement_rms_mm"],
        result["displacement_95th_mm"],
        result["displacement_max_mm"],
    )

    tutorial_results: dict[str, Any] = dict(result)
    tutorial_results["ground_truth_labelmap_dir"] = ground_truth_dir

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=repo_root / "tests" / "baselines" / class_name,
        log_level=log_level,
    )
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(result["predicted_surfaces"][0]))),
            "predicted_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
        tt.save_screenshot_image_slice(
            itk.imread(str(result["warped_labelmaps"][0])),
            "warped_labelmap.png",
            axis=0,
            slice_fraction=0.5,
            colormap="viridis",
        ),
    ]
