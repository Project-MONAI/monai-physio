"""
Tutorial 11 (Duke Heart, MGN): Score Predicted Heart Motion Per Chamber

Purpose
-------
Duke counterpart of ``tutorial_11_lung_evaluate_physicsnemo.py``.  Measures how
close the size and shape of the heart inferred by Tutorial 10 are to the heart
actually imaged, one gated frame at a time and one chamber at a time.  The case
is ``ParametersDukeHeartLabelmaps.hold_out_case``, held out of every fit in this
chain, so this scores generalization rather than recall.

1. Read the ground truth.  Unlike the lung chain, no segmentation is needed:
   this cohort ships one labelmap per gated frame, each already carrying the
   four chambers, the myocardium and the whole heart.

2. Score the prediction: :class:`monai_physio.WorkflowEvaluateMovement` carries
   the reference frame's labelmap into every other frame with the network's own
   deformation, and compares the result to that frame's labelmap --- volume
   difference, Dice and surface RMSE per chamber.

   The shape model this network moves is one structure, the whole heart minus
   its chamber cavities, so the chambers exist only in the acquired labelmaps.
   Going through the labelmaps rather than through the model's surface is what
   makes per-chamber scores possible at all.

3. Score the motion point by point: with Tutorial 8's per-frame SSM surfaces as
   the ground truth, the report also carries the distance between where the
   network puts each mesh point and where the shape model fitted it --- RMS,
   95th percentile and maximum, per structure and per frame.  The shape model
   has no chamber geometry of its own, so a mesh point counts toward the
   structure whose reference-frame surface is nearest: a chamber is scored on
   the piece of wall that bounds it.  A chamber whose predicted motion is right on average can
   still be wrong everywhere, and only this measure says so.

4. Write ``evaluation_report.md`` and ``evaluation_metrics.csv``, both carrying
   the hold-out case name, the case's shape parameters, and the network weights
   path with its dates.  Each metric is reported both averaged over the frames
   and at the frame it is worst at.

Data Required
-------------
  * ``data/Duke-Heart-4DLabelmaps/<case>/*_labelmap.nii.gz`` - gated frames
  * ``output/tutorial_08_duke_heart/<case>/`` - Tutorial 8 SSM surface + coefficients
  * ``network_weights/physicsnemo_mgn_duke_heart_motion/`` - Tutorial 9 checkpoint

Outputs (under ``output/tutorial_11_duke_heart/<case>/``)
---------------------------------------------------------
  * ``evaluation_report.md``    - per-chamber accuracy of the prediction, mean
    and worst case, with the per-point displacement error per frame
  * ``evaluation_metrics.csv``  - one row per stage and structure, each
    carrying that structure's displacement error (RMS, 95th percentile, maximum)
  * ``volume_vs_stage.png``     - each structure's volume across the stages
  * ``<case>_ssm_pca_coefficients_s{TTT}_pred.vtp`` - predicted surface per stage,
    carrying the displacement point-data arrays the ``include_*`` switches ask for
  * ``displacement_per_point.csv`` - every mesh point's predicted and true
    displacement at every frame; written only when ``report_displacement_data``
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from monai_physio import (
    EvaluateMovementDukeHeart,
    TestTools,
    WorkflowEvaluateMovement,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)


# Only run if this script is not imported as a module

# PhysicsNeMo and torch spawn worker processes. On Windows the spawn start
# method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the whole evaluation in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_11_duke_heart_evaluate_physicsnemo"

    # Case to score: the case held out of every fit in this chain.
    case_id = DUKE_HEART.hold_out_case

    # Fitted SSM surface and PCA coefficients written by Tutorial 8 (Duke Heart).
    test_mode = TestTools.running_as_test()
    # Keep a test run out of the directories a full run reads and writes.
    case_dir = (
        DUKE_HEART.output_directory(test_mode) / "tutorial_08_duke_heart" / case_id
    )
    # Weights Tutorial 9 trained, and the checkpoint epoch Tutorial 10 infers
    # with; None uses the final weights.
    model_dir = DUKE_HEART.mgn_weights_directory(test_mode)
    epoch: Optional[int] = None

    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the labelmap is resampled through.
    smoothing_sigma_mm = 10.0
    # Isotropic pitch every metric is measured on.  Coarser than these
    # labelmaps, whose in-plane pitch is finer than the accuracy being reported,
    # and still below the thinnest wall of the heart.
    evaluation_spacing_mm = 1.0

    # Per-point displacement reporting, all off by default.  The first writes
    # one CSV row per mesh point per frame; the rest carry the same quantities
    # as point data on each frame's predicted surface.  Every one of them except
    # the predicted displacement is measured against Tutorial 8's per-frame SSM
    # surfaces, the only geometry that shares this mesh's point ordering.
    report_displacement_data = False
    include_predicted_displacements = False
    include_true_displacements = False
    # On: the point-by-point error is the one measure a displacement predicted
    # in the wrong direction cannot hide in, and it costs one mesh read a frame.
    include_displacement_error = True

    output_dir = (
        DUKE_HEART.output_directory(test_mode) / "tutorial_11_duke_heart" / case_id
    )
    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    labelmap_dir = DUKE_HEART.hold_out_directory(test_mode) / case_id

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    fitted_reference_mesh_file = case_dir / f"{case_id}_ssm_surface.vtp"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    for required_file in (fitted_reference_mesh_file, pca_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 8 output not found: {required_file}\n"
                "Run tutorials/tutorial_08_duke_heart_fit_model_to_4d_patients.py "
                "first."
            )

    # Step 1: the cohort assembles what this case is scored against -- every
    # gated frame's labelmap, the reference frame among them, and Tutorial 8's
    # per-frame fits, which are the only geometry that shares the fitted
    # reference mesh's point ordering.
    cohort = EvaluateMovementDukeHeart(log_level=log_level)
    ground_truth = cohort.assemble_ground_truth(
        case_id=case_id,
        frame_directory=labelmap_dir,
        fit_directory=case_dir,
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
