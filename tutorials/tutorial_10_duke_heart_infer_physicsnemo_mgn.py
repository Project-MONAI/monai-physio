"""
Tutorial 10 (Duke Heart, MGN): Predict Heart Motion Across the Cardiac Cycle

Purpose
-------
Final inference stage of the Duke heart 4D deep-learning pipeline (Tutorials 8
-> 9 -> 10), the counterpart of ``tutorial_10_lung_infer_physicsnemo_mgn.py``.
A thin driver over :class:`monai_physio.WorkflowInferPhysicsNeMo` and its
displacement decoder :class:`monai_physio.WorkflowInferMovement`:

1. Discover the per-frame SSM surfaces that Tutorial 8 (Duke Heart)
   (``tutorial_08_duke_heart_fit_model_to_4d_patients.py``) wrote for the
   held-out case.  Stages are parsed from the ``g{PPP}`` gate tag of the frame
   filenames.

2. Predict that case's surface at *every* cardiac stage with the MeshGraphNet
   trained by Tutorial 9 (``tutorial_09_duke_heart_train_physicsnemo_mgn.py``).
   The network predicts per-vertex displacements, so the decoder adds them to
   the case's fitted reference SSM surface.  Scoring the result is Tutorial
   11's job, not this one's; here the acquired frame surface is only rendered
   beside the prediction so the two can be compared by eye.

3. Rasterize each stage's displacements into a deformation field and carry the
   reference frame through it, and write the whole series as one animated USD.
   This cohort ships segmented labelmaps rather than CT, so the image carried
   through the deformation is the reference frame's labelmap, resampled with
   nearest-neighbor interpolation to keep its label values discrete.

Steps 2 and 3 are :meth:`WorkflowInferMovement.process_time_series`; this script
only chooses the case, the stages and the image to warp.

For command-line use with path arguments, use the installed
``monai-physio-infer-physicsnemo`` CLI instead of editing this script.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "monai-physio[physicsnemo]"

Data Required
-------------
  * ``output/tutorial_08_duke_heart/<case>/`` - Tutorial 8 SSM surfaces
  * ``data/Duke-Heart-4DLabelmaps/<case>/*_ref_labelmap.nii.gz``
    - reference frame that is warped
  * ``network_weights/physicsnemo_mgn_duke_heart_motion/mgn_stage_model.pt``
    - Tutorial 9 checkpoint
    (``ParametersDukeHeartLabelmaps.mgn_weights_directory``)

Outputs (under ``output/tutorial_10_duke_heart_mgn/<case>/``)
------------------------------------------------------------
  * ``<case>_ssm_pca_coefficients_s{TTT}_pred.vtp``   - predicted surface
  * ``<case>_ssm_pca_coefficients_s{TTT}_warped.mha`` - labelmap at that stage
  * ``<case>_mgn_motion.usd``                         - animated predicted motion
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
    TestTools,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)

# Gated frames carry a ``g{PPP}`` tag naming their percentage of the R-R
# interval; this is what a per-frame SSM surface is matched and staged by.
PHASE_SURFACE_PATTERN = "*_g[0-9][0-9][0-9]_*_ssm_surface.vtp"
LABELMAP_SUFFIX = "_labelmap.nii.gz"


def _cardiac_stage_from_filename(surface_file: Path) -> float:
    """Extract the normalized cardiac stage [0, 1] from a ``g{PPP}`` filename stem."""
    for part in surface_file.stem.split("_"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse cardiac gate from filename: {surface_file}")


# Only run if this script is not imported as a module

# PhysicsNeMo and torch spawn worker processes. On Windows the spawn start
# method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the prediction in every worker.
if __name__ == "__main__":
    # Data directory specification
    tutorials_dir = Path(__file__).resolve().parent
    test_mode = TestTools.running_as_test()
    # Keep a test run out of the directories a full run reads and writes.
    # Fitted SSM surfaces and PCA coefficients written by Tutorial 8 (Duke Heart).
    data_dir = DUKE_HEART.output_directory(test_mode) / "tutorial_08_duke_heart"
    # The network Tutorial 9 (Duke Heart) trained.
    model_dir = DUKE_HEART.mgn_weights_directory(test_mode)
    # Intermittent-checkpoint epoch to load; None uses the final weights.
    epoch: Optional[int] = None

    # Case to predict; the held-out test case of Tutorial 9 (Duke Heart).
    case_id = DUKE_HEART.hold_out_case
    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the reference frame is resampled through.
    smoothing_sigma_mm = 10.0

    output_dir = (
        DUKE_HEART.output_directory(test_mode) / "tutorial_10_duke_heart_mgn" / case_id
    )
    log_level = logging.INFO

    class_name = "tutorial_10_duke_heart_infer_physicsnemo_mgn"
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    labelmap_dir = DUKE_HEART.hold_out_directory(test_mode) / case_id

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = model_dir / "mgn_stage_model.pt"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Tutorial 9 checkpoint not found: {checkpoint_file}\n"
            "Run tutorials/tutorial_09_duke_heart_train_physicsnemo_mgn.py first."
        )

    case_dir = data_dir / case_id
    fitted_reference_mesh_file = case_dir / f"{case_id}_ssm_surface.vtp"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    phase_files = sorted(case_dir.glob(PHASE_SURFACE_PATTERN))
    for required_file in (fitted_reference_mesh_file, pca_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 8 output not found: {required_file}\n"
                "Run tutorials/tutorial_08_duke_heart_fit_model_to_4d_patients.py "
                "first."
            )
    if not phase_files:
        raise FileNotFoundError(f"No gated frame surfaces found in {case_dir}")

    reference_labelmaps = sorted(labelmap_dir.glob(f"*_ref{LABELMAP_SUFFIX}"))
    if not reference_labelmaps:
        raise FileNotFoundError(
            f"No *_ref{LABELMAP_SUFFIX} frame found in {labelmap_dir}.\n"
            "See data/Duke-Heart-4DLabelmaps/README.md."
        )

    # Step 1: read every gated frame of the case and its ground-truth surface,
    # in gate order.
    stages = [_cardiac_stage_from_filename(f) for f in phase_files]
    logger.info("Case %s: predicting %d gated frames", case_id, len(stages))

    # Step 2 and 3: predict the whole cycle, warp the reference frame through
    # each stage's deformation, and write the animated USD.  The SSM is one
    # structure, the whole heart minus its chamber cavities, so the USD surface
    # is kept whole rather than split by connectivity.
    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir, epoch=epoch, log_level=log_level
    )
    infer_result = WorkflowInferMovement(
        infer_workflow, log_level=log_level
    ).process_time_series(
        shape_parameters=pca_file,
        stages=stages,
        output_directory=output_dir,
        fitted_reference_mesh=fitted_reference_mesh_file,
        reference_image=itk.imread(str(reference_labelmaps[0])),
        warp_interpolation="nearest",
        warp_background_value=0.0,
        smoothing_sigma_mm=smoothing_sigma_mm,
        usd_project_name=f"{case_id}_mgn_motion",
        anatomy_type="heart",
        separate_by_connectivity=False,
    )

    tutorial_results: dict[str, Any] = dict(infer_result)
    tutorial_results["ground_truth_files"] = phase_files

    # Testing: render the first predicted stage beside the ground-truth frame it
    # is scored against.
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=tutorials_dir.parent / "tests" / "baselines" / class_name,
        log_level=log_level,
    )
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(infer_result["predicted_surfaces"][0]))),
            "predicted_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(phase_files[0]))),
            "ground_truth_surface.png",
            camera_position="iso",
            color="steelblue",
        ),
    ]
