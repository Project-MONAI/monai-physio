"""
Tutorial 15 (Duke Heart, MGN): Leave-One-Out Cross-Validation of the Whole Chain

Purpose
-------
Duke counterpart of ``tutorial_15_lung_leave_one_out.py``.  Tutorials 6 -> 8 ->
9 -> 11 report accuracy for one fixed held-out case; this runs the same chain
once per fold, holding out a different patient each time, and reports the mean
and spread of the metrics across folds.

Each fold is a full re-run, not a re-score of a shared model:

1. Build a PCA statistical shape model from the population *without* the
   held-out case (``WorkflowCreateMeanSurface`` then
   ``WorkflowCreateStatisticalModel``).  Rebuilding is the whole point --- a
   model built once from everyone has already seen every case, so scoring
   against it measures recall rather than generalization.

2. Fit that fold's model to every case, held-out one included, and propagate
   each fit through the gated cardiac frames.  This cohort ships labelmaps
   rather than CT, so there is no intensity image to register: each frame's
   heart surface is contoured and the fitted SSM surface is registered to it
   with ``RegisterModelsDistanceMaps``, which warps the SSM while keeping its
   topology.

3. Train a MeshGraphNet on the other cases' frames
   (``WorkflowTrainPhysicsNeMo`` driving ``TrainPhysicsNeMoMGN``).

4. Infer the held-out case at every gated frame and score it against that
   frame's acquired labelmap --- Dice, volume difference and surface RMSE per
   chamber (``WorkflowEvaluateMovement``), plus, also per chamber, the
   distance between where the network puts each mesh point and where this
   fold's own fit put it.

Then every fold's rows are pooled into ``loo_metrics.csv`` and summarized per
chamber in ``loo_report.md``, which also carries each fold's displacement error, its
RMS, 95th percentile and maximum.
That last measure is the point-by-point one: a fold can keep every chamber the
right size and still put all of them in the wrong place, and only it says so.

The shape model this network moves is one structure, the whole heart minus its
chamber cavities, so the chambers exist only in the acquired labelmaps.  Going
through the labelmaps rather than through the model's surface is what makes
per-chamber scores possible at all.

What is *not* recomputed per fold
---------------------------------
The per-frame heart contours and the remeshed model-population surfaces do not
depend on which case is held out, so they are contoured once into ``shared/``
and reused by every fold.  Tutorial 4's surfaces are read in place of the cache
when they exist.

Unlike the lung variant, the *phase propagation* here cannot be hoisted: it
registers the fold's own fitted SSM surface to each frame, and that surface
changes with the fold.  This is why a Duke fold costs materially more than a
lung fold.

Multi-GPU
---------
Launch under ``torchrun`` and every rank runs this script:

    torchrun --standalone --nproc_per_node=8 \\
        tutorials/tutorial_15_duke_heart_leave_one_out.py

Training is then data-parallel across the ranks (``DistributedDataParallel``
inside ``WorkflowTrainPhysicsNeMo``), and the per-case contouring, fitting and
frame-registration loops are split across the ranks a case at a time.  Run
without a launcher and the same script runs as a single process.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric must be installed::

    pip install "physiotwin4d[physicsnemo]"

Data Required
-------------
  * ``data/Duke-Heart-4DLabelmaps/pm????/*_labelmap.nii.gz`` - the whole cohort,
    one directory per case with exactly one ``*_ref_labelmap.nii.gz`` frame
  * Nothing from Tutorials 4, 6, 8 or 9.  Their outputs are reused as a cache
    when present, but this tutorial contours everything it needs.

Outputs (under ``tutorials/output/tutorial_15_duke_heart/``)
------------------------------------------------------------
  * ``loo_metrics.csv``           - every metric row of every fold
  * ``loo_report.md``             - per-chamber mean +/- std across folds,
    displacement 95th percentile included, plus each fold's whole-model error
  * ``loo_metrics_by_label.png``  - the same, as a per-chamber box plot
  * ``fold_<case>/``              - that fold's shape model, fits and weights
  * ``shared/``                   - the fold-independent contours

Runtime
-------
The long pole is the per-fold frame registration: roughly one distance-map ICON
registration per case per gated frame, across the whole cohort, repeated for
every fold.  Splitting the cases across ranks divides that by the rank count.
Training adds one MeshGraphNet run per fold on top.  Expect five folds to run
overnight even on eight GPUs.
"""

# Imports
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from physiotwin4d import (
    ContourTools,
    DistributedContext,
    RegisterModelsDistanceMaps,
    EvaluateMovementDukeHeart,
    TestTools,
    TrainPhysicsNeMoMGN,
    WorkflowCreateMeanSurface,
    WorkflowCreateStatisticalModel,
    WorkflowEvaluateMovement,
    WorkflowFitStatisticalModelToPatient,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
    WorkflowTrainPhysicsNeMo,
    distributed_context,
)

# Structure name Tutorial 4 (Duke Heart) writes its whole-heart surfaces under.
WHOLE_HEART_NAME = "heart_minus_interior_chambers"
LABELMAP_SUFFIX = "_labelmap.nii.gz"

# The four chambers, plus the myocardium and the whole heart for context: 5 and
# 6 are what the shape model itself represents, 1-4 are the cavities it does
# not.  The great vessels and coronaries (7-10) are left out; they come and go
# between frames and are not part of the model.
HEART_LABEL_IDS = [1, 2, 3, 4, 5, 6]

# Point-data array the tutorial writes its targets into and the manifests name.
TARGET_ARRAY = "displacement"

# Gated frames carry a ``g{PPP}`` tag naming their percentage of the R-R
# interval; this is what a per-frame SSM surface is matched and staged by.
PHASE_SURFACE_PATTERN = "*_g[0-9][0-9][0-9]_*_ssm_surface.vtp"


def _cardiac_stage_from_filename(frame_file: Path) -> float:
    """Extract the normalized cardiac stage [0, 1] from a ``g{PPP}`` filename stem."""
    for part in frame_file.name.split("_"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse cardiac gate from filename: {frame_file}")


def _rank_share(items: list[Any], context: DistributedContext) -> list[Any]:
    """Return the slice of ``items`` this rank is responsible for.

    Every rank writes to its own cases' directories and the caller waits on a
    barrier afterwards, so the ranks never contend for a file.
    """
    return items[context.rank :: context.world_size]


def _write_target_mesh(
    phase_file: Path, ref_points: np.ndarray, targets_dir: Path
) -> Path:
    """Write one frame's training target and return the mesh path.

    The target is the per-vertex displacement from the case's reference surface,
    stored as the ``TARGET_ARRAY`` point-data array on a copy of the frame
    surface.  The training workflow reads whatever array the manifest names and
    never derives one.
    """
    phase_mesh = pv.read(str(phase_file))
    phase_points = np.asarray(phase_mesh.points, dtype=np.float32)
    phase_mesh.point_data[TARGET_ARRAY] = phase_points - ref_points
    target_path = targets_dir / f"{phase_file.stem}_target.vtp"
    phase_mesh.save(str(target_path))
    return target_path


def _write_case_manifest(case_dir: Path, manifests_dir: Path) -> Path:
    """Write one case's training manifest and return its path."""
    case_id = case_dir.name
    fitted_reference_mesh_file = case_dir / f"{case_id}_ssm_surface.vtp"
    phase_files = sorted(case_dir.glob(PHASE_SURFACE_PATTERN))
    manifests_dir.mkdir(parents=True, exist_ok=True)
    ref_points = np.asarray(
        pv.read(str(fitted_reference_mesh_file)).points, dtype=np.float32
    )
    manifest = {
        "subject_id": case_id,
        "fitted_reference_mesh": str(fitted_reference_mesh_file),
        "pca_coefficients": str(case_dir / f"{case_id}_ssm_pca_coefficients.json"),
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


def _summarize(values: list[float]) -> tuple[float, float, int]:
    """Return ``(mean, sample standard deviation, count)`` ignoring NaNs.

    One fold cannot have a spread, so its standard deviation is reported as 0.0
    rather than left undefined.
    """
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mean, std, len(finite)


def _write_loo_report(
    rows: list[dict[str, Any]],
    held_out_cases: list[str],
    label_names: dict[int, str],
    report_file: Path,
) -> None:
    """Write the per-chamber cross-fold summary.

    Each fold contributes one number per structure --- the mean over that fold's
    frames --- and the reported spread is across those per-fold numbers.
    Pooling every frame of every fold instead would let a case with more gated
    frames weigh more heavily than the others.
    """
    metrics = [
        ("dice", "Dice", "{:.4f}"),
        ("volume_difference_percent", "|volume difference| (%)", "{:.2f}"),
        ("surface_rmse_mm", "surface RMSE (mm)", "{:.3f}"),
        ("displacement_95th_mm", "displacement 95th percentile (mm)", "{:.3f}"),
    ]
    lines = [
        "# Leave-One-Out Cross-Validation: Duke Heart Motion",
        "",
        f"Folds: {len(held_out_cases)}",
        f"Held-out cases: {', '.join(held_out_cases)}",
        "",
        "Each fold rebuilt the shape model without its held-out case, refitted "
        "the cohort, retrained the MeshGraphNet and scored that case against "
        "its own acquired labelmaps. Every cell is the mean over folds of the "
        "fold's own mean over frames, plus or minus the standard deviation "
        "across folds.",
        "",
        "| Structure | " + " | ".join(title for _, title, _ in metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for label in sorted(label_names):
        cells = []
        for key, _, fmt in metrics:
            per_fold = []
            for case_id in held_out_cases:
                fold_values = [
                    abs(float(row[key]))
                    for row in rows
                    if row["held_out_case"] == case_id
                    and int(row["label_id"]) == label
                    and np.isfinite(float(row[key]))
                ]
                if fold_values:
                    per_fold.append(float(np.mean(fold_values)))
            mean, std, count = _summarize(per_fold)
            cells.append(
                "n/a" if count == 0 else f"{fmt.format(mean)} +/- {fmt.format(std)}"
            )
        lines.append(f"| {label_names[label]} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Pooled over all structures", ""])
    for key, title, fmt in metrics:
        values = [abs(float(row[key])) for row in rows if np.isfinite(float(row[key]))]
        mean, std, count = _summarize(values)
        lines.append(
            f"- {title}: "
            + ("n/a" if count == 0 else f"{fmt.format(mean)} +/- {fmt.format(std)}")
            + f"  (n={count} rows)"
        )

    lines.extend(_displacement_error_lines(rows, held_out_cases))
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _displacement_error_lines(
    rows: list[dict[str, Any]], held_out_cases: list[str]
) -> list[str]:
    """Per-fold displacement error, one number per fold rather than per structure.

    The error is measured on the whole shape model at once, so every row of a
    fold carries the same ``fold_displacement_*`` figures; the fold is the unit
    here, while each row's own ``displacement_*`` columns are its stage's.
    """
    per_fold: list[tuple[str, float, float, float]] = []
    for case_id in held_out_cases:
        fold_rows = [row for row in rows if row["held_out_case"] == case_id]
        if fold_rows and np.isfinite(float(fold_rows[0]["fold_displacement_rms_mm"])):
            per_fold.append(
                (
                    case_id,
                    float(fold_rows[0]["fold_displacement_rms_mm"]),
                    float(fold_rows[0]["fold_displacement_95th_mm"]),
                    float(fold_rows[0]["fold_displacement_max_mm"]),
                )
            )
    if not per_fold:
        return []

    mean, std, _ = _summarize([rms for _, rms, _, _ in per_fold])
    p95_mean, p95_std, _ = _summarize([p95 for _, _, p95, _ in per_fold])
    lines = [
        "",
        "## Displacement error, per fold",
        "",
        "Distance between where the network puts each mesh point of the held-out "
        "case and where that fold's own fit put it, pooled over every point and "
        "frame. Measured point by point, so a chamber predicted the right size "
        "in the wrong place is scored here and nowhere above.",
        "",
        f"- RMS across folds: {mean:.3f} +/- {std:.3f} mm",
        f"- 95th percentile across folds: {p95_mean:.3f} +/- {p95_std:.3f} mm",
        f"- Worst fold: {max(maximum for _, _, _, maximum in per_fold):.3f} mm",
        "",
        "| Held-out case | RMS (mm) | 95th percentile (mm) | Max (mm) |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {case_id} | {rms:.3f} | {p95:.3f} | {maximum:.3f} |"
        for case_id, rms, p95, maximum in per_fold
    ]
    return lines


def _plot_metrics_by_label(
    rows: list[dict[str, Any]], label_names: dict[int, str], plot_file: Path
) -> Path:
    """Box-plot each metric per structure, one box over every scored frame."""
    import matplotlib.pyplot as plt

    metrics = [
        ("dice", "Dice"),
        ("volume_difference_percent", "|volume difference| (%)"),
        ("surface_rmse_mm", "surface RMSE (mm)"),
        ("displacement_95th_mm", "displacement 95th percentile (mm)"),
    ]
    labels = sorted(label_names)
    fig, axes = plt.subplots(1, len(metrics), figsize=(17.0, 4.5))
    try:
        for ax, (key, title) in zip(axes, metrics):
            data = [
                [
                    abs(float(row[key]))
                    for row in rows
                    if int(row["label_id"]) == label and np.isfinite(float(row[key]))
                ]
                for label in labels
            ]
            # A structure absent from every acquired frame has no box to draw;
            # give it an empty slot so the tick labels stay aligned.
            ax.boxplot([values or [np.nan] for values in data], showmeans=True)
            ax.set_xticklabels(
                [label_names[label] for label in labels], rotation=30, ha="right"
            )
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(plot_file), dpi=110)
    finally:
        plt.close(fig)
    return plot_file


# Only run if this script is not imported as a module

# The registration backends and torch spawn worker processes. On Windows the
# spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the whole study in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_15_duke_heart_leave_one_out"

    # Number of folds. Each holds out one patient and re-runs the entire chain
    # without them, so the runtime is linear in this.
    number_of_leave_one_out_runs = 5

    # Atlas iterations used to build each fold's reference surface; 1 is a
    # single template-biased pass.
    mean_surface_iterations = 3

    # Training hyperparameters, matching Tutorial 9 (duke heart) so a fold's
    # network is comparable to the one that tutorial trains.
    epochs = 1500
    batch_size = 8  # mini-batch measured in (case, frame) graphs
    learning_rate = 1.0e-3
    processor_size = 3  # message-passing hops
    hidden_dim = 128
    num_layers = 2  # MLP layers inside each encoder / processor / decoder block

    # Pitch of the grid the frame distance maps are rasterized on. Coarser than
    # the contouring pitch: it carries a distance field, not a boundary.
    registration_spacing_mm = 1.0

    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the labelmap is resampled through.
    smoothing_sigma_mm = 10.0
    # Isotropic pitch every metric is measured on. Coarser than these
    # labelmaps, whose in-plane pitch is finer than the accuracy being
    # reported, and still below the thinnest wall of the heart.
    evaluation_spacing_mm = 1.0

    test_mode = TestTools.running_as_test()
    # Keep a test run out of the directories a full run reads and writes.
    weights_dir = DUKE_HEART.weights_directory(test_mode)

    output_dir = DUKE_HEART.output_directory(test_mode) / "tutorial_15_duke_heart"
    shared_dir = output_dir / "shared"
    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # Under torchrun / SLURM / mpirun this is one rank of many; started plainly
    # it reports a world of one and every branch below collapses to a single
    # process.
    context = distributed_context()

    data_dir = DUKE_HEART.hold_out_directory(test_mode)
    tutorial_04_dir = DUKE_HEART.input_directory(test_mode)
    number_of_pca_components = DUKE_HEART.pca_components(test_mode)

    # Labels left out of the whole-heart structure, the same ones Tutorials 4,
    # 6 and 8 drop, so the frames and the model describe the same structure.
    interior_object_ids = DUKE_HEART.interior_object_ids

    # In test mode, run the smallest study that is still a cross-validation and
    # train for a couple of epochs, to keep the run inside the pytest timeout.
    if test_mode:
        number_of_leave_one_out_runs = 2
        mean_surface_iterations = 1
        epochs = 2

    # Distance-map weights finetuned by
    # tutorial_02_duke_heart_distancemap_finetune_icon.py, used both by the
    # labelmap-to-labelmap stage of the SSM fit and by the frame registrations.
    icon_weights_path = (
        weights_dir
        / "icon_duke_heart_distancemap"
        / "icon_duke_heart_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    # Directory setup and data reading

    contour_dir = shared_dir / "contours"
    samples_dir = shared_dir / "model_samples"
    if context.is_main:
        for directory in (contour_dir, samples_dir):
            directory.mkdir(parents=True, exist_ok=True)
    context.barrier()

    case_dirs = sorted(
        path for path in data_dir.glob("pm[0-9][0-9][0-9][0-9]") if path.is_dir()
    )
    if not case_dirs:
        raise FileNotFoundError(
            f"No pm???? case directories found under {data_dir}.\n"
            "See data/Duke-Heart-4DLabelmaps/README.md."
        )

    def reference_frame_file(case_dir: Path) -> Path:
        """Return the case's reference frame, the one every fit starts from."""
        matches = sorted(case_dir.glob(f"*_ref{LABELMAP_SUFFIX}"))
        if not matches:
            raise FileNotFoundError(
                f"No *_ref{LABELMAP_SUFFIX} frame in {case_dir}; the SSM is "
                "fitted to that frame, so it is the one every deformation "
                "starts from."
            )
        return matches[0]

    case_dirs = [
        path for path in case_dirs if sorted(path.glob(f"*_ref{LABELMAP_SUFFIX}"))
    ]
    case_ids = [path.name for path in case_dirs]
    case_dir_by_id = dict(zip(case_ids, case_dirs))
    if len(case_ids) < 3:
        raise RuntimeError(
            f"Found only {len(case_ids)} case(s) with a reference frame under "
            f"{data_dir}; a fold needs one case to hold out and at least two to "
            "build a shape model from."
        )
    if number_of_leave_one_out_runs > len(case_ids):
        raise ValueError(
            f"number_of_leave_one_out_runs is {number_of_leave_one_out_runs} but "
            f"only {len(case_ids)} cases are available under {data_dir}."
        )
    held_out_cases = case_ids[:number_of_leave_one_out_runs]
    logger.info(
        "%d cases, %d folds; holding out %s",
        len(case_ids),
        len(held_out_cases),
        ", ".join(held_out_cases),
    )

    use_finetuned_weights = icon_weights_path.exists()
    if not use_finetuned_weights:
        logger.warning(
            "Finetuned distance-map ICON weights not found at %s; registering "
            "with the stock uniGradICON weights. Run "
            "tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py to "
            "create them.",
            icon_weights_path,
        )

    contour_tools = ContourTools(log_level=log_level)

    # The cohort names the structures every fold reports and, in Step 5, reads
    # each fold's ground truth: the acquired labelmaps and this fold's own fits.
    cohort = EvaluateMovementDukeHeart(log_level=log_level)
    heart_names = cohort.label_names()

    def heart_surface_for(labelmap_file: Path) -> Path:
        """Return the path to one frame's whole heart, minus its chamber cavities.

        Tutorial 4's ``"full"`` pass contours this surface for every gated
        frame, so its output is used when present.  Otherwise the surface is
        contoured into the shared cache, which no fold recomputes: the contour
        of an acquired frame does not depend on which case was held out.
        """
        stem = labelmap_file.name[: -len(LABELMAP_SUFFIX)]
        tutorial_04_file = tutorial_04_dir / f"{stem}_{WHOLE_HEART_NAME}.vtp"
        if tutorial_04_file.exists():
            return tutorial_04_file

        surface_file = contour_dir / f"{stem}_heart_surface.vtp"
        if not surface_file.exists():
            labelmap = itk.imread(str(labelmap_file))
            labels = itk.GetArrayViewFromImage(labelmap)
            heart_ids = [
                int(value)
                for value in np.unique(labels)
                if value != 0 and int(value) not in interior_object_ids
            ]
            heart_mask = itk.GetImageFromArray(
                np.isin(labels, heart_ids).astype(np.uint8)
            )
            heart_mask.CopyInformation(labelmap)
            contour_tools.extract_label_surfaces(
                heart_mask,
                isotropic_spacing_mm=DUKE_HEART.surface_spacing_mm,
                smoothing_iterations=DUKE_HEART.surface_smoothing_iterations,
            )[1].save(str(surface_file))
        return surface_file

    # Step 1: the shared cache. Contouring a frame and remeshing a case's
    # reference surface to the model's point budget are both independent of
    # which case is held out, so they happen once. Cases are split across the
    # ranks; the barrier below is what lets every rank read every case's files.
    for case_id in _rank_share(case_ids, context):
        case_dir = case_dir_by_id[case_id]
        for frame_file in sorted(case_dir.glob(f"*{LABELMAP_SUFFIX}")):
            heart_surface_for(frame_file)

        # The population surface: the model's fixed point budget, which the
        # contours carry thirty times over. See the parameters module.
        sample_file = samples_dir / f"{case_id}_ref_model_surface.vtp"
        if not sample_file.exists():
            surface = cast(
                pv.PolyData,
                pv.read(str(heart_surface_for(reference_frame_file(case_dir)))),
            )
            contour_tools.remesh_and_smooth_surface(
                surface, 1.0 - DUKE_HEART.model_points / surface.n_points, 0
            ).save(str(sample_file))
    context.barrier()

    # Steps 2-5: one fold per held-out case.
    all_rows: list[dict[str, Any]] = []
    fold_results: dict[str, Any] = {}

    for fold_index, held_out_case in enumerate(held_out_cases):
        logger.info("%s", "=" * 48)
        logger.info(
            "Fold %d/%d: holding out %s",
            fold_index + 1,
            len(held_out_cases),
            held_out_case,
        )
        logger.info("%s", "=" * 48)

        fold_dir = output_dir / f"fold_{held_out_case}"
        fits_dir = fold_dir / "fits"
        weights_dir = fold_dir / "weights"
        pca_model_file = fold_dir / "pca_model.json"
        pca_mean_file = fold_dir / "pca_mean_surface.vtp"
        if context.is_main:
            fits_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: the fold's shape model, built without the held-out case.
        # Groupwise and not parallel over cases, so rank 0 builds it and the
        # others pick it up off disk after the barrier.
        if context.is_main and not (pca_model_file.exists() and pca_mean_file.exists()):
            sample_surfaces = [
                pv.read(str(samples_dir / f"{case_id}_ref_model_surface.vtp"))
                for case_id in case_ids
                if case_id != held_out_case
            ]
            reference_surface_file = fold_dir / "reference_mean_surface.vtp"
            if not reference_surface_file.exists():
                # The reference surface defines the topology every PCA input is
                # expressed in, so picking one case would make the model inherit
                # that case's shape. Use the unbiased mean of the fold's own
                # population instead.
                mean_workflow = WorkflowCreateMeanSurface(
                    surfaces=sample_surfaces,
                    template_surface=sample_surfaces[len(sample_surfaces) // 2],
                    log_level=log_level,
                )
                mean_workflow.set_number_of_iterations(mean_surface_iterations)
                mean_workflow.set_mask_dilation_mm(DUKE_HEART.mask_dilation_mm)
                mean_workflow.set_distance_squared_max(
                    DUKE_HEART.distancemap_squared_max
                )
                if use_finetuned_weights:
                    mean_workflow.set_icon_weights_path(str(icon_weights_path))
                mean_workflow.process()["mean_surface"].save(
                    str(reference_surface_file)
                )
            model_workflow = WorkflowCreateStatisticalModel(
                sample_meshes=sample_surfaces,
                reference_mesh=pv.read(str(reference_surface_file)),
                number_of_pca_components=number_of_pca_components,
                icp_transform_type=DUKE_HEART.icp_transform_type,
                mask_dilation_mm=DUKE_HEART.mask_dilation_mm,
                distance_squared_max=DUKE_HEART.distancemap_squared_max,
                log_level=log_level,
            )
            if use_finetuned_weights:
                model_workflow.set_icon_weights_path(str(icon_weights_path))
            model_result = model_workflow.process()
            pca_model_file.write_text(
                json.dumps(model_result["pca_model"], indent=2), encoding="utf-8"
            )
            model_result["pca_mean_surface"].save(str(pca_mean_file))
        context.barrier()

        pca_mean_surface = cast(pv.DataSet, pv.read(str(pca_mean_file)))
        with pca_model_file.open(encoding="utf-8") as f:
            pca_model = json.load(f)

        # Step 3: fit the fold's model to every case, held-out one included, and
        # carry each fit onto every gated frame.
        for case_id in _rank_share(case_ids, context):
            case_dir = case_dir_by_id[case_id]
            case_fit_dir = fits_dir / case_id
            case_fit_dir.mkdir(parents=True, exist_ok=True)
            fitted_reference_mesh_file = case_fit_dir / f"{case_id}_ssm_surface.vtp"
            pca_coefficients_file = (
                case_fit_dir / f"{case_id}_ssm_pca_coefficients.json"
            )

            reference_file = reference_frame_file(case_dir)
            reference_surface = cast(
                pv.PolyData, pv.read(str(heart_surface_for(reference_file)))
            )
            if not (
                fitted_reference_mesh_file.exists() and pca_coefficients_file.exists()
            ):
                logger.info("Fold %s: fitting %s", held_out_case, case_id)
                # This data carries no intensity image, so the workflow
                # rasterizes its own reference grid from the patient surface.
                fit_workflow = WorkflowFitStatisticalModelToPatient(
                    template_model=pca_mean_surface,
                    patient_models=[reference_surface],
                    patient_image=None,
                    patient_labelmap=itk.imread(str(reference_file)),
                    labelmap_interior_object_ids=interior_object_ids,
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
                fit_workflow.set_distancemap_squared_max(
                    DUKE_HEART.distancemap_squared_max
                )
                if use_finetuned_weights:
                    fit_workflow.set_labelmap_to_labelmap_icon_weights_path(
                        str(icon_weights_path)
                    )
                fit_result = fit_workflow.process()

                coefficients = fit_workflow.pca_coefficients
                assert coefficients is not None
                pca_coefficients_file.write_text(
                    json.dumps(coefficients.tolist()), encoding="utf-8"
                )
                # The heart PCA model is built from surfaces only, so the fitted
                # model is itself a surface and only the .vtp is written.
                fit_result["fitted_reference_mesh"].save(
                    str(fitted_reference_mesh_file)
                )
            fitted_reference_mesh = cast(
                pv.PolyData, pv.read(str(fitted_reference_mesh_file))
            )

            # One grid built around the reference frame's heart and reused by
            # every frame, so the whole case is registered in a common space;
            # its buffer holds the frames the heart moves into.
            registration_grid = contour_tools.create_reference_image(
                mesh=reference_surface,
                spatial_resolution=registration_spacing_mm,
                buffer_factor=0.25,
                ptype=itk.F,
            )
            for frame_file in sorted(case_dir.glob(f"*{LABELMAP_SUFFIX}")):
                stem = frame_file.name[: -len(LABELMAP_SUFFIX)]
                frame_surface_file = case_fit_dir / f"{stem}_ssm_surface.vtp"
                if frame_surface_file.exists():
                    continue
                if frame_file == reference_file:
                    # The fit already placed the SSM on this frame.
                    fitted_reference_mesh.save(str(frame_surface_file))
                    continue
                logger.info("Fold %s: %s warping to %s", held_out_case, case_id, stem)
                registrar = RegisterModelsDistanceMaps(
                    moving_model=fitted_reference_mesh,
                    fixed_model=cast(
                        pv.PolyData, pv.read(str(heart_surface_for(frame_file)))
                    ),
                    reference_image=registration_grid,
                    distance_squared_max=DUKE_HEART.distancemap_squared_max,
                    mask_dilation_mm=DUKE_HEART.mask_dilation_mm,
                    log_level=log_level,
                )
                if use_finetuned_weights:
                    registrar.set_icon_weights_path(str(icon_weights_path))
                registrar.register(transform_type="Deformable")[
                    "registered_model"
                ].save(str(frame_surface_file))

        context.barrier()

        # Step 4: train on every case but the held-out one. Manifest paths are
        # derived from the case ids, so every rank names the same files without
        # having to be told which ones the others wrote.
        manifests_dir = fold_dir / "manifests_mgn"
        for case_id in _rank_share(case_ids, context):
            _write_case_manifest(fits_dir / case_id, manifests_dir)
        context.barrier()

        manifests = {
            case_id: manifests_dir / f"{case_id}_manifest.json" for case_id in case_ids
        }
        training_method = TrainPhysicsNeMoMGN(log_level=log_level)
        training_method.set_epochs(epochs)
        training_method.set_batch_size(batch_size)
        training_method.set_learning_rate(learning_rate)
        training_method.set_processor_size(processor_size)
        training_method.set_hidden_dim(hidden_dim)
        training_method.set_num_layers(num_layers)

        train_result = WorkflowTrainPhysicsNeMo(
            train_manifests=[
                path for case_id, path in manifests.items() if case_id != held_out_case
            ],
            # Spending a case on validation would take it out of a training set
            # that is already one case short, so the intermittent validation
            # RMSE reads "n/a" here as it does in Tutorial 9.
            val_manifests=[],
            pca_mean_mesh=pca_mean_file,
            output_directory=weights_dir,
            training_method=training_method,
            log_level=log_level,
        ).process()

        # Step 5: score the held-out case against its acquired labelmaps. No
        # segmentation is needed: this cohort ships one labelmap per gated
        # frame, each already carrying the four chambers, the myocardium and
        # the whole heart.
        if context.is_main:
            held_out_dir = case_dir_by_id[held_out_case]
            case_fit_dir = fits_dir / held_out_case
            # The fits scored against are this fold's own, not Tutorial 8's,
            # which is what makes the fold's number honest.
            fold_ground_truth = cohort.assemble_ground_truth(
                case_id=held_out_case,
                frame_directory=held_out_dir,
                fit_directory=case_fit_dir,
            )
            evaluate_workflow = WorkflowEvaluateMovement(
                movement_workflow=WorkflowInferMovement(
                    WorkflowInferPhysicsNeMo(
                        model_directory=train_result["output_directory"],
                        log_level=log_level,
                    ),
                    log_level=log_level,
                ),
                cohort=cohort,
                log_level=log_level,
            )
            fold_result = evaluate_workflow.process(
                case_id=held_out_case,
                shape_parameters=case_fit_dir
                / f"{held_out_case}_ssm_pca_coefficients.json",
                fitted_reference_mesh=case_fit_dir / f"{held_out_case}_ssm_surface.vtp",
                ground_truth=fold_ground_truth,
                output_directory=fold_dir / "evaluation",
                smoothing_sigma_mm=smoothing_sigma_mm,
                evaluation_spacing_mm=evaluation_spacing_mm,
                # The point-by-point error, which is what separates a fold that
                # placed the chambers right from one that merely kept their size.
                include_displacement_error=True,
            )
            for row in fold_result["rows"]:
                all_rows.append(
                    {
                        "fold_index": fold_index,
                        "held_out_case": held_out_case,
                        **row,
                        "fold_displacement_rms_mm": fold_result["displacement_rms_mm"],
                        "fold_displacement_95th_mm": fold_result[
                            "displacement_95th_mm"
                        ],
                        "fold_displacement_max_mm": fold_result["displacement_max_mm"],
                    }
                )
            fold_results[held_out_case] = {
                "model_directory": train_result["output_directory"],
                "report_file": fold_result["report_file"],
                "csv_file": fold_result["csv_file"],
                "displacement_rms_mm": fold_result["displacement_rms_mm"],
                "displacement_95th_mm": fold_result["displacement_95th_mm"],
                "displacement_max_mm": fold_result["displacement_max_mm"],
            }
        context.barrier()

    # Step 6: pool the folds. Only rank 0 has the rows.
    tutorial_results: dict[str, Any] = {
        "held_out_cases": held_out_cases,
        "folds": fold_results,
        "screenshots": [],
    }
    if context.is_main:
        metrics_file = output_dir / "loo_metrics.csv"
        with metrics_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        report_file = output_dir / "loo_report.md"
        _write_loo_report(all_rows, held_out_cases, heart_names, report_file)
        plot_file = _plot_metrics_by_label(
            all_rows, heart_names, output_dir / "loo_metrics_by_label.png"
        )
        logger.info("Metrics: %s", metrics_file)
        logger.info("Report: %s", report_file)

        tutorial_results["metrics_file"] = metrics_file
        tutorial_results["report_file"] = report_file
        tutorial_results["rows"] = all_rows

        # Testing
        tt = TestTools(
            class_name=class_name,
            results_dir=output_dir,
            baselines_dir=repo_root / "tests" / "baselines" / class_name,
            log_level=log_level,
        )
        last_case = held_out_cases[-1]
        tutorial_results["screenshots"] = [
            plot_file,
            tt.save_screenshot_mesh(
                cast(
                    pv.DataSet,
                    pv.read(
                        str(
                            output_dir
                            / f"fold_{last_case}"
                            / "fits"
                            / last_case
                            / f"{last_case}_ssm_surface.vtp"
                        )
                    ),
                ),
                "held_out_fitted_surface.png",
                camera_position="iso",
                color="steelblue",
            ),
        ]
        logger.info("Last fold report: %s", fold_results[last_case]["report_file"])
