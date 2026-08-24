"""
Tutorial 9 (Duke Heart, MGN): Train a PhysicsNeMo MeshGraphNet on the Fitted SSM

Purpose
-------
Duke counterpart of ``tutorial_09_lung_train_physicsnemo_mgn.py``, run on the
Duke-Heart-4DLabelmaps cohort.  A thin driver over the reusable
:class:`physiotwin4d.WorkflowTrainPhysicsNeMo` workflow:

1. Discover the per-frame SSM surfaces produced by Tutorial 8 (Duke Heart)
   (``tutorial_08_duke_heart_fit_model_to_4d_patients.py``), write the training
   target for each frame, and write one JSON manifest per case.  The target here
   is the per-vertex displacement from the case's reference surface, stored as a
   ``displacement`` point-data array -- the workflow reads targets verbatim and
   never derives them.  Cardiac stages are parsed from the ``g{PPP}`` gate tag
   of the labelmap filenames and written explicitly into the manifest (the
   workflow never parses filenames).

2. Split the cases into train and held-out test -- plus an optional validation
   set, empty by default, which is what makes the intermittent validation RMSE
   read ``n/a`` -- and train the MeshGraphNet (``WorkflowTrainPhysicsNeMo``
   driving ``TrainPhysicsNeMoMGN``).

3. Evaluate the held-out test cases against their ground-truth frames with
   :class:`physiotwin4d.WorkflowInferPhysicsNeMo` wrapped in
   :class:`physiotwin4d.WorkflowInferMovement`.

Why a GNN?
----------
The SSM surface has a fixed topology across all cases and the myocardium is a
continuum: adjacent vertices co-vary smoothly.  MeshGraphNet encodes that prior
directly by passing messages along mesh edges, giving an explicit
continuum-deformation inductive bias the MLP must infer from coordinates alone.

Node features (per vertex):   [mean_shape_x, mean_shape_y, mean_shape_z, pca_c1 ... pca_cN, stage]
Edge features (per edge):     [rel_x, rel_y, rel_z, distance]   (from the mean shape)
Output (per vertex):          [dx, dy, dz]  (displacement in mm)

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiotwin4d[physicsnemo]"

Data Required
-------------
SSM surfaces: Tutorial 8 (Duke Heart) output
(``output/tutorial_08_duke_heart/pm????/``)
PCA mean surface: Tutorial 6 (Duke Heart) output
(``output/tutorial_06_duke_heart/pca_mean_surface.vtp``, alongside
``pca_model.json``)

Outputs
-------
Manifests and per-frame targets are written under
``output/tutorial_09_duke_heart_mgn/manifests_mgn/``:

  * ``pm????_manifest.json``            - per-case training manifest
  * ``<frame_stem>_ssm_surface_target.vtp`` - per-frame displacement targets

The evaluation of the held-out cases lands in
``output/tutorial_09_duke_heart_mgn/eval_mgn/pm????/``.

The model itself is written to ``ParametersDukeHeartLabelmaps.mgn_weights_dir``
(``network_weights/physicsnemo_mgn_duke_heart_motion/``), or to a fresh ``..._1``
sibling when resuming (see ``resume_from``), which is what ``tutorial_results``
reports as ``model_directory``:

  * ``mgn_stage_model.pt``  - trained MeshGraphNet checkpoint
  * ``mgn_stage_model_epoch_#####.pt`` - intermittent checkpoints
  * ``pca_mean_surface.vtp``, ``pca_mean_template.vtp``, ``pca_model.json``,
    ``shared_edge_index.pt``, ``shared_edge_features.pt`` and the metadata JSON
    - everything inference needs beside the weights

Everything but the checkpoints is written before the first epoch, so
``tutorial_10_duke_heart_infer_physicsnemo_mgn.py`` can be pointed at this
directory with its ``epoch`` set to an intermittent checkpoint while training
is still running.
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from physiotwin4d import (
    TestTools,
    TrainPhysicsNeMoMGN,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
    WorkflowTrainPhysicsNeMo,
)

# Point-data array the tutorial writes its targets into and the manifests name.
TARGET_ARRAY = "displacement"

# Gated frames carry a ``g{PPP}`` tag naming their percentage of the R-R
# interval; this is what a per-frame SSM surface is matched and staged by.
PHASE_SURFACE_PATTERN = "*_g[0-9][0-9][0-9]_*_ssm_surface.vtp"


def _cardiac_stage_from_filename(surface_file: Path) -> float:
    """Extract the normalized cardiac stage [0, 1] from a ``g{PPP}`` filename stem."""
    for part in surface_file.stem.split("_"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse cardiac gate from filename: {surface_file}")


def _write_target_mesh(
    phase_file: Path, ref_points: np.ndarray, targets_dir: Path
) -> Path:
    """Write one frame's training target and return the mesh path.

    The target is the per-vertex displacement from the case's reference surface,
    stored as the ``TARGET_ARRAY`` point-data array on a copy of the frame
    surface.  Any other per-vertex quantity could be written here instead -- the
    training workflow reads whatever array the manifest names.
    """
    phase_mesh = pv.read(str(phase_file))
    phase_points = np.asarray(phase_mesh.points, dtype=np.float32)
    phase_mesh.point_data[TARGET_ARRAY] = phase_points - ref_points
    target_path = targets_dir / f"{phase_file.stem}_target.vtp"
    phase_mesh.save(str(target_path))
    return target_path


def _write_case_manifest(
    case_dir: Path, manifests_dir: Path, logger: logging.Logger
) -> Optional[Path]:
    """Write a per-case manifest JSON; return its path (or None if incomplete).

    A case needs a reference SSM surface, a PCA coefficient file, and at least
    two gated-frame surfaces.  A case that is missing any of them is skipped
    with the reason logged, so a half-finished Tutorial 8 run is distinguishable
    from one that never ran.
    """
    case_id = case_dir.name
    fitted_reference_mesh_file = case_dir / f"{case_id}_ssm_surface.vtp"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    phase_files = sorted(case_dir.glob(PHASE_SURFACE_PATTERN))
    missing = []
    if not fitted_reference_mesh_file.exists():
        missing.append(f"reference surface {fitted_reference_mesh_file.name}")
    if not pca_file.exists():
        missing.append(f"PCA coefficients {pca_file.name}")
    if len(phase_files) < 2:
        missing.append(f"at least 2 frame surfaces (found {len(phase_files)})")
    if missing:
        logger.warning("Skipping %s: missing %s", case_id, "; ".join(missing))
        return None

    manifests_dir.mkdir(parents=True, exist_ok=True)
    ref_points = np.asarray(
        pv.read(str(fitted_reference_mesh_file)).points, dtype=np.float32
    )
    manifest = {
        "subject_id": case_id,
        "fitted_reference_mesh": str(fitted_reference_mesh_file),
        "pca_coefficients": str(pca_file),
        "target_array": TARGET_ARRAY,
        "phases": [
            {
                "mesh": str(_write_target_mesh(phase_file, ref_points, manifests_dir)),
                "stage": _cardiac_stage_from_filename(phase_file),
            }
            for phase_file in phase_files
        ],
    }
    manifest_path = manifests_dir / f"{case_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


# Only run if this script is not imported as a module

# PhysicsNeMo and torch spawn worker processes for data loading. On Windows the
# spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart training in every worker.
if __name__ == "__main__":
    # Data directory specification
    tutorials_dir = Path(__file__).resolve().parent
    # Fitted SSM surfaces and PCA coefficients written by Tutorial 8 (Duke Heart).
    data_dir = tutorials_dir / "output" / "tutorial_08_duke_heart"
    # PCA mean surface written by Tutorial 6 (Duke Heart); pca_model.json must
    # sit beside it, which is how Tutorial 6 writes them.
    ssm_mean_surface_file = DUKE_HEART.pca_mean_file
    # Manifests, targets and the held-out evaluation are written here.
    output_dir = tutorials_dir / "output" / "tutorial_09_duke_heart_mgn"
    manifests_dir = output_dir / "manifests_mgn"
    eval_dir = output_dir / "eval_mgn"
    # The trained network is kept with the other trained networks instead.
    model_output_dir = DUKE_HEART.mgn_weights_dir

    # Warm-start from a previous run's checkpoint; None trains from scratch.
    # When resuming, training writes to a fresh sibling directory (``..._1``).
    resume_from: Optional[Path] = None

    # Training hyperparameters.  The heart template carries
    # ParametersDukeHeartLabelmaps.model_points vertices, an order of magnitude
    # fewer than the lung template of Tutorial 9 (lung), so a larger mini-batch
    # fits; lower batch_size, or call
    # training_method.set_num_processor_checkpoint_segments(...), on a smaller
    # card.
    epochs = 1500
    batch_size = 8  # mini-batch measured in (case, frame) graphs
    learning_rate = 1.0e-3
    processor_size = 3  # message-passing hops
    hidden_dim = 128
    num_layers = 2  # MLP layers inside each encoder / processor / decoder block

    # Explicit held-out splits; every other discovered case is used for
    # training.  The held-out case is the one Tutorials 2, 6 and 7 also keep
    # out, so one patient stays unseen by everything in the pipeline.  Adding a
    # case to val_cases spends it on the intermittent validation RMSE instead of
    # training; empty means that RMSE is reported as "n/a".
    test_cases = [DUKE_HEART.hold_out_case]
    val_cases: list[str] = []
    log_level = logging.INFO

    class_name = "tutorial_09_duke_heart_train_physicsnemo_mgn"
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # In test mode, train for a couple of epochs to keep the run tractable.
    test_mode = TestTools.running_as_test()
    if test_mode:
        epochs = 2

    if not ssm_mean_surface_file.exists():
        raise FileNotFoundError(
            f"Tutorial 6 PCA mean surface not found: {ssm_mean_surface_file}\n"
            "Run tutorials/tutorial_06_duke_heart_create_statistical_model.py first."
        )

    # Step 1: build one manifest per valid case and partition into splits.
    manifests: dict[str, Path] = {}
    for case_dir in sorted(
        p for p in data_dir.glob("pm[0-9][0-9][0-9][0-9]") if p.is_dir()
    ):
        manifest_path = _write_case_manifest(case_dir, manifests_dir, logger)
        if manifest_path is not None:
            manifests[case_dir.name] = manifest_path

    if len(manifests) < 3:
        raise RuntimeError(
            f"Found only {len(manifests)} valid case(s) under {data_dir}; need at "
            "least 3 to hold one out and still train on a population. See the "
            "skip reasons logged above, and run "
            "tutorials/tutorial_08_duke_heart_fit_model_to_4d_patients.py first."
        )

    unknown = [
        case_id for case_id in test_cases + val_cases if case_id not in manifests
    ]
    if unknown:
        raise ValueError(f"Split cases not found: {unknown}")

    test_manifests = [manifests[case_id] for case_id in test_cases]
    val_manifests = [manifests[case_id] for case_id in val_cases]
    train_manifests = [
        manifest_path
        for case_id, manifest_path in manifests.items()
        if case_id not in test_cases and case_id not in val_cases
    ]
    logger.info(
        "Case split - train: %d, val: %d, test: %d",
        len(train_manifests),
        len(val_manifests),
        len(test_manifests),
    )

    # Step 2: train the MeshGraphNet. The training method carries the network and
    # its hyper-parameters; the workflow feeds it manifests and saves the results.
    training_method = TrainPhysicsNeMoMGN(log_level=log_level)
    training_method.set_epochs(epochs)
    training_method.set_batch_size(batch_size)
    training_method.set_learning_rate(learning_rate)
    training_method.set_processor_size(processor_size)
    training_method.set_hidden_dim(hidden_dim)
    training_method.set_num_layers(num_layers)

    train_workflow = WorkflowTrainPhysicsNeMo(
        train_manifests=train_manifests,
        val_manifests=val_manifests,
        pca_mean_mesh=ssm_mean_surface_file,
        output_directory=model_output_dir,
        resume_from=resume_from,
        training_method=training_method,
        log_level=log_level,
    )
    train_result = train_workflow.process()

    # Step 3: evaluate held-out test cases against their ground-truth frames.
    # When resuming, training writes to a fresh sibling directory, so evaluate
    # the model from the directory training actually used.
    model_directory = train_result["output_directory"]
    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_directory, log_level=log_level
    )
    # The targets are displacements from each case's reference surface, so the
    # raw predictions are turned back into surfaces by the displacement decoder.
    displacement_workflow = WorkflowInferMovement(infer_workflow, log_level=log_level)

    tutorial_results: dict[str, Any] = {
        "model_directory": model_directory,
        "cases": {},
    }
    for case_id in test_cases:
        logger.info("Evaluating held-out case %s", case_id)
        tutorial_results["cases"][case_id] = displacement_workflow.process(
            manifests[case_id],
            output_directory=eval_dir / case_id,
        )

    # Testing: render the first predicted surface of the last held-out case.
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=tutorials_dir.parent / "tests" / "baselines" / class_name,
        log_level=log_level,
    )
    last_case = tutorial_results["cases"][test_cases[-1]]
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(last_case["predicted_surfaces"][0]))),
            "predicted_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
    ]
