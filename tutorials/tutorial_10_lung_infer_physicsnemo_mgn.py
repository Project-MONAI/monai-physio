"""
Tutorial 10 (Lung, MGN): Predict Lung Motion Across the Respiratory Cycle

Purpose
-------
Final inference stage of the lung 4D deep-learning pipeline (Tutorials 8 -> 9 ->
10). A thin driver over :class:`physiotwin4d.WorkflowInferPhysicsNeMo` and its
displacement decoder :class:`physiotwin4d.WorkflowInferMovement`:

1. Discover the per-phase SSM surfaces that Tutorial 8
   (``tutorial_08_lung_fit_model_to_4d_patients.py``) wrote for
   ``ParametersLungCTDirLab.mgn_hold_out_case`` -- the case Tutorial 9 held out
   of training, so this scores generalization rather than recall. Stages are
   parsed from the ``T{PP}`` phase filenames.

2. Predict that case's surface at *every* respiratory stage with the
   MeshGraphNet trained by Tutorial 9
   (``tutorial_09_lung_train_physicsnemo_mgn.py``). The network predicts
   per-vertex displacements, so the decoder adds them to the case's fitted
   reference SSM surface.  Scoring the result is Tutorial 11's job, not this
   one's; here the acquired phase surface is only rendered beside the
   prediction so the two can be compared by eye.

3. Rasterize each stage's displacements into a deformation field and carry the
   reference-phase CT through it, giving one warped CT per stage, and write the
   whole series as one animated USD.

Steps 2 and 3 are :meth:`WorkflowInferMovement.process_time_series`; this script
only chooses the case, the stages and the image to warp.

For command-line use with path arguments, use the installed
``physiotwin4d-infer-physicsnemo`` CLI instead of editing this script.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiotwin4d[physicsnemo]"

Data Required
-------------
  * ``output/tutorial_08_lung/<case>/``  - Tutorial 8 SSM surfaces
  * ``data/DirLab-4DCT/<case>_T70.mha``  - reference-phase CT that is warped
  * ``network_weights/physicsnemo_mgn_lung_motion/mgn_stage_model.pt``
    - Tutorial 9 checkpoint (``ParametersLungCTDirLab.mgn_weights_dir``)

Outputs (under ``output/tutorial_10_lung_mgn/<case>/``)
-------------------------------------------------------
  * ``<case>_ssm_pca_coefficients_s{TTT}_pred.vtp``   - predicted surface
  * ``<case>_ssm_pca_coefficients_s{TTT}_warped.mha`` - CT carried to that stage
  * ``<case>_mgn_motion.usd``                         - animated predicted motion
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import pyvista as pv
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from physiotwin4d import (
    TestTools,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)


def _respiratory_stage_from_filename(surface_file: Path) -> float:
    """Extract the normalized respiratory stage [0, 1] from a ``T{PP}`` filename stem."""
    for part in surface_file.stem.split("_"):
        if part.startswith("T") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse respiratory phase from filename: {surface_file}")


# Only run if this script is not imported as a module

# PhysicsNeMo and torch spawn worker processes. On Windows the spawn start
# method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the prediction in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent
    # Fitted SSM surfaces and PCA coefficients written by Tutorial 8 (lung).
    data_dir = tutorials_dir / "output" / "tutorial_08_lung"
    # Weights Tutorial 9 trained. A resumed Tutorial 9 run writes to a numbered
    # sibling of this directory, which is what would be evaluated instead.
    model_dir = LUNG_CT_DIRLAB.mgn_weights_dir
    # Intermittent-checkpoint epoch to load; None uses the final weights.
    epoch: Optional[int] = None

    # Case to predict: the case Tutorial 9 held out of training.
    case_id = LUNG_CT_DIRLAB.mgn_hold_out_case
    # Phase the SSM was fitted to by Tutorial 8, and therefore the phase whose
    # CT the predicted deformations carry into every other stage.
    reference_phase = "T70"
    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the CT is resampled through.
    smoothing_sigma_mm = 10.0

    output_dir = tutorials_dir / "output" / "tutorial_10_lung_mgn" / case_id
    log_level = logging.INFO

    class_name = "tutorial_10_lung_infer_physicsnemo_mgn"
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    if epoch is not None:
        checkpoint_file = model_dir / f"mgn_stage_model_epoch_{epoch:05d}.pt"
    else:
        checkpoint_file = model_dir / "mgn_stage_model.pt"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Tutorial 9 checkpoint not found: {checkpoint_file}\n"
            "Run tutorials/tutorial_09_lung_train_physicsnemo_mgn.py first."
        )

    case_dir = data_dir / case_id
    fitted_reference_mesh_file = case_dir / f"{case_id}_ssm_surface.vtp"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    reference_ct_file = (
        repo_root / "data" / "DirLab-4DCT" / (f"{case_id}_{reference_phase}.mha")
    )
    phase_files = sorted(case_dir.glob(f"{case_id}_T??_ssm_surface.vtp"))
    for required_file in (fitted_reference_mesh_file, pca_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 8 output not found: {required_file}\n"
                "Run tutorials/tutorial_08_lung_fit_model_to_4d_patients.py first."
            )
    if not phase_files:
        raise FileNotFoundError(f"No respiratory phase surfaces found in {case_dir}")
    if not reference_ct_file.exists():
        raise FileNotFoundError(
            f"Reference-phase CT not found: {reference_ct_file}\n"
            "See data/DirLab-4DCT/README.md for download instructions."
        )

    # Step 1: read every respiratory phase of the case and its ground-truth
    # surface, in phase order.
    stages = [_respiratory_stage_from_filename(f) for f in phase_files]
    logger.info("Case %s: predicting %d respiratory phases", case_id, len(stages))

    # Step 2 and 3: predict the whole cycle, warp the reference CT through each
    # stage's deformation, and write the animated USD. The network predicts
    # displacements, so the decoder adds them to the case's reference SSM
    # surface. -1000 HU is air, the value a CT grid samples outside itself.
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
        reference_image=itk.imread(str(reference_ct_file)),
        warp_interpolation="linear",
        warp_background_value=-1000.0,
        smoothing_sigma_mm=smoothing_sigma_mm,
        usd_project_name=f"{case_id}_mgn_motion",
        anatomy_type="lung",
        separate_by_connectivity=True,
    )

    tutorial_results: dict[str, Any] = dict(infer_result)
    tutorial_results["ground_truth_files"] = phase_files

    # Testing: render the first predicted stage beside the ground-truth phase it
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
