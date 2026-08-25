"""
Tutorial 14 (Lung, MGN): How Much Do the Shape Parameters Move the Motion?

Purpose
-------
Tutorial 11 scores the inferred motion of the hold-out case at the shape
parameters its statistical-model fit happened to land on --- one point in shape
space.  This tutorial sweeps that point.  It perturbs the first few PCA
coefficients over a grid, re-infers the whole respiratory cycle at every
combination, and scores each one the way Tutorial 11 scores its single fit, so
the CSV says how far Dice, volume and surface RMSE move when the shape
parameters move.

The case is ``ParametersLungCTDirLab.mgn_hold_out_case``, held out of the
Tutorial 9 training, so this measures the sensitivity of a generalizing
prediction rather than of a recalled one.

**Only the coefficients handed to the network change.**  The reference anatomy
stays the Tutorial 8 fitted surface at every point of the grid; what the
perturbed coefficients change is the displacement field the MeshGraphNet infers,
and those displacements are applied to that one unmodified surface.  So the
sweep isolates the network's sensitivity to its shape conditioning, with no
shape difference of the patient's own anatomy mixed in.

Because the reference surface, the reference labelmap and the acquired frames
are the same for every combination, every combination is scored on the same
evaluation grid, and the Dice and volume figures are directly comparable across
the grid.

1. Build the ground truth: segment every gated CT frame of the case
   independently, giving one labelmap per respiratory phase whose lobes were
   never seen by the shape model or by the network.  It does not depend on the
   perturbation, so it is built once and cached for the whole sweep.

2. Read the Tutorial 8 fit: the fitted SSM surface, held fixed for the whole
   sweep, and the case's PCA coefficient vector, which is the center of the
   grid.

3. Build the grid: every combination of ``number_of_modes_to_vary`` offsets,
   each running from ``-perturbation_range`` to ``+perturbation_range`` in steps
   of ``perturbation_step``, in units of standard deviations.  The all-zero
   combination is in the grid, so the unperturbed score comes out of the same
   code path as every perturbed one.

4. Score every combination with :class:`physiotwin4d.WorkflowEvaluateMovement`
   and write ``shape_sweep_metrics.csv`` (one row per combination, phase and
   lobe) and ``shape_sweep_summary.csv`` (one row per combination, averaged over
   phases and lobes).

Unlike Tutorial 11, Dice is reported here.  The caveat it states still holds ---
a lobe barely changes shape over a breath compared to how big it is, so the
overlap fraction stays above 0.96 whatever the prediction does --- so expect the
Dice column to move far less across the grid than the volume and surface RMSE
columns do.

Every combination also carries ``displacement_rms_mm``,
``displacement_95th_mm`` and ``displacement_max_mm``, the point-by-point
distance between where the network puts each mesh point and where Tutorial 8
fitted it in that phase.  That is the
column to read the sweep by: a perturbed coefficient can leave a lobe the same
size in the same place and still move every point of it wrong, which the
labelmap metrics cannot see and this one cannot miss.  Those three are
pooled over the whole model; ``mean_displacement_rms_mm`` and
``mean_displacement_95th_mm`` are the same error averaged over the
lobes, each scored on the mesh points nearest to it.

Cost
----
Every combination is a full time-series inference, deformation-field
rasterization, warp, contour and metric pass, and the per-combination
directories hold the warped labelmaps of every phase.  The default grid is
``5 ** 2 = 25`` combinations, each costing one Tutorial 11 run: tens of
minutes to hours in total, depending on the case, and a few gigabytes on disk.
``number_of_modes_to_vary``, ``perturbation_step`` and ``evaluation_spacing_mm``
are the knobs.  The acquired labelmaps are resampled and contoured once per
combination even though they never change; that is the price of calling the
scoring workflow unchanged.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiotwin4d[physicsnemo]"

Data Required
-------------
  * ``data/DirLab-4DCT/<case>_T??.mha``  - the gated CT sequence
  * ``output/tutorial_08_lung/<case>/``  - Tutorial 8 SSM surface + coefficients
  * ``network_weights/physicsnemo_mgn_lung_motion/`` - Tutorial 9 checkpoint

Outputs (under ``output/tutorial_14_lung/<case>/``)
---------------------------------------------------
  * ``shape_sweep_metrics.csv`` - one row per combination, phase and lobe
  * ``shape_sweep_summary.csv`` - one row per combination, with that
    combination's pooled displacement error
  * ``ground_truth/<case>_T{PP}_labelmap.nii.gz`` - cached per-phase segmentation
  * ``combo_{NNN}/shape_parameters.json`` - that combination's coefficients
  * ``combo_{NNN}/evaluation_report.md``, ``evaluation_metrics.csv``,
    ``volume_vs_stage.png`` - that combination's own Tutorial 11 style report
  * ``combo_{NNN}/shape_parameters_s{TTT}_pred.vtp`` - predicted surface per phase
  * ``combo_{NNN}/shape_parameters_s{TTT}_warped.mha`` - reference labelmap
    carried to that phase
"""

# Imports
from __future__ import annotations

import csv
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from physiotwin4d import (
    EvaluateMovementLung,
    TestTools,
    WorkflowEvaluateMovement,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)


def _offset_grid(
    number_of_modes: int, perturbation_range: float, perturbation_step: float
) -> list[tuple[float, ...]]:
    """Every combination of per-mode offsets, in standard deviations.

    Half a step is added to the stop so that ``+perturbation_range`` itself is on
    the grid rather than falling off its floating-point edge.
    """
    offsets = np.arange(
        -perturbation_range,
        perturbation_range + perturbation_step / 2.0,
        perturbation_step,
    )
    return [
        tuple(float(offset) for offset in combination)
        for combination in itertools.product(offsets, repeat=number_of_modes)
    ]


def _mean_of_measured(values: list[float]) -> float:
    """Mean of the values that could be measured; ``nan`` when none could."""
    measured = [value for value in values if not math.isnan(value)]
    return float(np.mean(measured)) if measured else float("nan")


# Only run if this script is not imported as a module

# nnUNetv2 and torch spawn worker processes. On Windows the spawn start method
# re-imports this script in each child; without the __name__ == "__main__" guard
# around top-level work, that re-import would restart the whole sweep in every
# worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    class_name = "tutorial_14_lung_shape_parameter_sweep"

    # Case to sweep: the case Tutorial 9 held out of training.
    case_id = LUNG_CT_DIRLAB.mgn_hold_out_case
    # Phase Tutorial 8 fitted the SSM to, and therefore the phase whose anatomy
    # the predicted deformations carry into every other phase.
    reference_phase = "T70"

    # Fitted SSM surface and PCA coefficients written by Tutorial 8 (lung).
    case_dir = tutorials_dir / "output" / "tutorial_08_lung" / case_id
    # Weights Tutorial 9 trained, and the checkpoint epoch to infer with; None
    # uses the final weights.
    model_dir = LUNG_CT_DIRLAB.mgn_weights_dir
    epoch: Optional[int] = None

    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the labelmap is resampled through.
    smoothing_sigma_mm = 10.0
    # Isotropic pitch every metric is measured on.  Coarser than the CT, whose
    # in-plane pitch is finer than the accuracy being reported, and fine enough
    # that a lobe boundary is not quantized away.
    evaluation_spacing_mm = 2.0

    output_dir = tutorials_dir / "output" / "tutorial_14_lung" / case_id
    ground_truth_dir = output_dir / "ground_truth"
    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    test_mode = TestTools.running_as_test()
    data_dir = LUNG_CT_DIRLAB.input_directory(test_mode)

    # The sweep itself.  The first modes carry the most variance, so varying
    # them is what a shape-parameter study is about; the offsets are in standard
    # deviations, the units the coefficients themselves are in.  The grid is
    # every combination of them, so its size is
    # (2 * range / step + 1) ** number_of_modes_to_vary and its cost grows the
    # same way.
    number_of_modes_to_vary = 2
    perturbation_range = 2.0
    perturbation_step = 1.0
    if test_mode:
        # Three combinations rather than twenty-five: the test asserts the sweep
        # runs and reports, not that the grid is finely sampled.
        number_of_modes_to_vary = 1
        perturbation_range = 1.0

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

    # The cohort assembles what every combination is scored against.  It does
    # not depend on the perturbation, so it is read once for the whole sweep.
    cohort = EvaluateMovementLung(reference_phase=reference_phase, log_level=log_level)
    ground_truth = cohort.assemble_ground_truth(
        case_id=case_id,
        frame_directory=data_dir,
        fit_directory=case_dir,
        cache_directory=ground_truth_dir,
    )

    baseline_coefficients = np.asarray(
        json.loads(pca_file.read_text(encoding="utf-8")), dtype=np.float32
    )
    if number_of_modes_to_vary > len(baseline_coefficients):
        raise ValueError(
            f"Asked to vary {number_of_modes_to_vary} modes, but the fit has only "
            f"{len(baseline_coefficients)} coefficients."
        )

    # Step 3: the grid of coefficient offsets.
    combinations = _offset_grid(
        number_of_modes_to_vary, perturbation_range, perturbation_step
    )
    logger.info(
        "Sweeping the first %d mode(s) from %+.2f to %+.2f in steps of %.2f "
        "standard deviations: %d combinations",
        number_of_modes_to_vary,
        -perturbation_range,
        perturbation_range,
        perturbation_step,
        len(combinations),
    )

    # Step 4: score every combination.  The network is loaded once and reused,
    # so the per-combination cost is inference and scoring alone.
    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir, epoch=epoch, log_level=log_level
    )
    evaluate_workflow = WorkflowEvaluateMovement(
        movement_workflow=WorkflowInferMovement(infer_workflow, log_level=log_level),
        cohort=cohort,
        log_level=log_level,
    )

    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    combination_dirs: list[Path] = []
    first_surfaces: list[Path] = []

    for index, offsets in enumerate(combinations):
        combination_dir = output_dir / f"combo_{index:03d}"
        combination_dir.mkdir(parents=True, exist_ok=True)
        combination_dirs.append(combination_dir)

        coefficients = baseline_coefficients.copy()
        coefficients[:number_of_modes_to_vary] += np.asarray(offsets, dtype=np.float32)
        coefficients_file = combination_dir / "shape_parameters.json"
        with coefficients_file.open(mode="w", encoding="utf-8") as f:
            json.dump(coefficients.tolist(), f)

        result = evaluate_workflow.process(
            case_id=case_id,
            shape_parameters=coefficients_file,
            fitted_reference_mesh=fitted_reference_mesh_file,
            ground_truth=ground_truth,
            output_directory=combination_dir,
            smoothing_sigma_mm=smoothing_sigma_mm,
            evaluation_spacing_mm=evaluation_spacing_mm,
            # Kept on, unlike Tutorial 11: a sweep over the shape parameters is
            # asked to report Dice, so it does. It is the least responsive of
            # the three columns for the reason Tutorial 11 leaves it out.
            report_dice=True,
            # The one metric measured point by point, and the most responsive of
            # them to a perturbed coefficient: the labelmap metrics see a lobe
            # that stayed the same size, this sees every point that moved wrong.
            include_displacement_error=True,
        )
        first_surfaces.append(Path(result["predicted_surfaces"][0]))

        # The offsets say where in the grid the row is; the coefficients the
        # workflow already stamped on it as pca_c01.. say what was actually fed
        # to the network.
        grid_position = {
            f"mode_{mode:02d}_offset": offset for mode, offset in enumerate(offsets)
        }
        rows = [
            {"combination": index, **grid_position, **row} for row in result["rows"]
        ]
        metric_rows.extend(rows)

        summary: dict[str, Any] = {
            "combination": index,
            **grid_position,
            **{
                f"mode_{mode:02d}_coefficient": float(coefficients[mode])
                for mode in range(number_of_modes_to_vary)
            },
            "mean_dice": _mean_of_measured([float(row["dice"]) for row in rows]),
            "mean_volume_difference_percent": _mean_of_measured(
                [float(row["volume_difference_percent"]) for row in rows]
            ),
            "mean_abs_volume_difference_percent": _mean_of_measured(
                [abs(float(row["volume_difference_percent"])) for row in rows]
            ),
            "mean_surface_rmse_mm": _mean_of_measured(
                [float(row["surface_rmse_mm"]) for row in rows]
            ),
            # Averaged over the structures, beside the same error pooled over
            # every point of the whole model: a mean of structures and a pooled
            # figure answer different questions, so both are kept.
            "mean_displacement_rms_mm": _mean_of_measured(
                [float(row["displacement_rms_mm"]) for row in rows]
            ),
            "mean_displacement_95th_mm": _mean_of_measured(
                [float(row["displacement_95th_mm"]) for row in rows]
            ),
            "displacement_rms_mm": result["displacement_rms_mm"],
            "displacement_95th_mm": result["displacement_95th_mm"],
            "displacement_max_mm": result["displacement_max_mm"],
            "n_rows": len(rows),
        }
        summary_rows.append(summary)

        logger.info(
            "combination %d/%d  offsets %s  dice=%.4f  dV=%+.2f%%  rmse=%.3f mm  "
            "displacement rms=%.3f mm  95th=%.3f mm  max=%.3f mm  "
            "(per structure: 95th=%.3f mm)",
            index + 1,
            len(combinations),
            ", ".join(f"{offset:+.2f}" for offset in offsets),
            summary["mean_dice"],
            summary["mean_volume_difference_percent"],
            summary["mean_surface_rmse_mm"],
            summary["displacement_rms_mm"],
            summary["displacement_95th_mm"],
            summary["displacement_max_mm"],
            summary["mean_displacement_95th_mm"],
        )

    # Step 5: the whole grid as two CSVs.
    metrics_csv_file = output_dir / "shape_sweep_metrics.csv"
    with metrics_csv_file.open(mode="w", newline="", encoding="utf-8") as f:
        metrics_writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        metrics_writer.writeheader()
        metrics_writer.writerows(metric_rows)

    summary_csv_file = output_dir / "shape_sweep_summary.csv"
    with summary_csv_file.open(mode="w", newline="", encoding="utf-8") as f:
        summary_writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        summary_writer.writeheader()
        summary_writer.writerows(summary_rows)

    logger.info("Metrics: %s", metrics_csv_file)
    logger.info("Summary: %s", summary_csv_file)

    tutorial_results: dict[str, Any] = {
        "rows": metric_rows,
        "summary_rows": summary_rows,
        "metrics_csv_file": metrics_csv_file,
        "summary_csv_file": summary_csv_file,
        "combination_directories": combination_dirs,
        "ground_truth_labelmap_dir": ground_truth_dir,
    }

    # Testing: the unperturbed prediction beside the most perturbed one, which
    # is the difference the sweep is measuring.
    grid_distances = [float(np.abs(offsets).sum()) for offsets in combinations]
    baseline_index = int(np.argmin(grid_distances))
    extreme_index = int(np.argmax(grid_distances))

    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=repo_root / "tests" / "baselines" / class_name,
        log_level=log_level,
    )
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(first_surfaces[baseline_index]))),
            "unperturbed_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(first_surfaces[extreme_index]))),
            "perturbed_surface.png",
            camera_position="iso",
            color="orange",
        ),
    ]
