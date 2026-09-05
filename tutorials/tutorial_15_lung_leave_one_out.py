"""
Tutorial 15 (Lung, MGN): Leave-One-Out Cross-Validation of the Whole Chain

Purpose
-------
Tutorials 6 -> 8 -> 9 -> 11 report accuracy for one fixed held-out case. That
is a single observation: it says nothing about how much the number would move
had a different patient been held out. This tutorial runs the same chain once
per fold, holding out a different DIR-Lab case each time, and reports the mean
and spread of the metrics across folds.

Each fold is a full re-run, not a re-score of a shared model:

1. Build a PCA statistical shape model from the population *without* the
   held-out case (``WorkflowCreateMeanSurface`` then
   ``WorkflowCreateStatisticalModel``). Rebuilding is the whole point --- a
   model built once from everyone has already seen every case, so scoring
   against it measures recall rather than generalization.

2. Fit that fold's model to every case, held-out one included, and propagate
   each fit through the respiratory phases
   (``WorkflowFitStatisticalModelToPatient``, then the cached phase transforms).
   The held-out case is fitted by a model that never saw it, which is what makes
   its shape parameters an honest input to inference.

3. Train a MeshGraphNet on the other cases' phases
   (``WorkflowTrainPhysicsNeMo`` driving ``TrainPhysicsNeMoMGN``).

4. Infer the held-out case at every acquired phase and score it against that
   phase's own segmentation --- Dice, volume difference and surface RMSE per
   lobe (``WorkflowEvaluateMovement``), plus, also per lobe, the distance
   between where the network puts each mesh point and where this fold's own
   fit put it.

Then every fold's rows are pooled into ``loo_metrics.csv`` and summarized per
lobe in ``loo_report.md``, which also carries each fold's displacement error, its
RMS, 95th percentile and maximum.
That last measure is the point-by-point one: a fold can keep every lobe the
right size and still put all of them in the wrong place, and only it says so.

On Dice for lobes: ``tutorial_11_lung_evaluate_physicsnemo.py`` deliberately
leaves it out, because a lobe barely deforms over a breath compared to how big
it is, so the overlap fraction mostly describes the lobe. It is reported here
because a leave-one-out study is about comparing folds, and a metric that is
insensitive in absolute terms can still separate a fold that went wrong. Read it
alongside surface RMSE, which is what resolves the motion.

What is *not* recomputed per fold
---------------------------------
Two things do not depend on which case is held out, so they are computed once
into ``shared/`` and reused by every fold:

  * the T70 lung segmentations, and
  * the phase-to-reference-phase image registrations.

The registrations are image-to-image and never touch the shape model, so one
set is correct for every fold. Hoisting them is what makes this tutorial
tractable; they otherwise dominate its runtime. Tutorial 6 and Tutorial 8
outputs are read in place of the cache when they exist.

Multi-GPU
---------
Launch under ``torchrun`` and every rank runs this script:

    torchrun --standalone --nproc_per_node=8 \\
        tutorials/tutorial_15_lung_leave_one_out.py

Training is then data-parallel across the ranks (``DistributedDataParallel``
inside ``WorkflowTrainPhysicsNeMo``), and the per-case segmentation, fitting and
registration loops are split across the ranks a case at a time. Run without a
launcher and the same script runs as a single process.

Data Required
-------------
  * ``data/DirLab-4DCT/Case*_T??.mha`` - the whole cohort; see
    ``data/DirLab-4DCT/README.md``
  * Nothing from Tutorials 6, 8 or 9. Their outputs are reused as a cache when
    present, but this tutorial builds everything it needs.

Outputs (under ``tutorials/output/tutorial_15_lung/``)
------------------------------------------------------
  * ``loo_metrics.csv``           - every metric row of every fold
  * ``loo_report.md``             - per-lobe mean +/- std across folds,
    displacement 95th percentile included, plus each fold's whole-model error
  * ``loo_metrics_by_label.png``  - the same, as a per-lobe box plot
  * ``fold_<case>/``              - that fold's shape model, fits and weights
  * ``shared/``                   - the fold-independent segmentations and
    transforms

Runtime
-------
Dominated by training: Tutorial 9 measures the full lung template (179k points,
1.07M mesh-graph edges) at roughly four hours for 1500 epochs on one GPU, and
that is paid once per fold. Eight ranks bring a fold's training well under an
hour, so five folds is an afternoon rather than a week. The shared cache is
built once and costs one segmentation and one phase registration per case.
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
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from monai_physio import (
    ContourTools,
    DistributedContext,
    RegisterImagesGreedy,
    EvaluateMovementLung,
    SegmentNVSegmentCTMRI,
    TestTools,
    TrainPhysicsNeMoMGN,
    TransformTools,
    WorkflowConvertImageToVTK,
    WorkflowCreateMeanSurface,
    WorkflowCreateStatisticalModel,
    WorkflowEvaluateMovement,
    WorkflowFitStatisticalModelToPatient,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
    WorkflowReconstructHighres4DCT,
    WorkflowTrainPhysicsNeMo,
    distributed_context,
)

# The five lobes of ``SegmentNVSegmentCTMRI``.  Its "lung" group also carries
# whole-lung, tumor and airway labels, which are not lobes.
LOBE_LABEL_IDS = [28, 29, 30, 31, 32]

# Point-data array the tutorial writes its targets into and the manifests name.
TARGET_ARRAY = "displacement"


def _respiratory_stage_from_filename(surface_file: Path) -> float:
    """Extract the normalized respiratory stage [0, 1] from a ``T{PP}`` filename stem."""
    for part in surface_file.stem.split("_"):
        if part.startswith("T") and part[1:].isdigit():
            return int(part[1:]) / 100.0
    raise ValueError(f"Cannot parse respiratory phase from filename: {surface_file}")


def _rank_share(items: list[Any], context: DistributedContext) -> list[Any]:
    """Return the slice of ``items`` this rank is responsible for.

    Every rank writes to its own cases' directories and the caller waits on a
    barrier afterwards, so the ranks never contend for a file.
    """
    return items[context.rank :: context.world_size]


def _write_target_mesh(
    phase_file: Path, ref_points: np.ndarray, targets_dir: Path
) -> Path:
    """Write one phase's training target and return the mesh path.

    The target is the per-vertex displacement from the case's reference surface,
    stored as the ``TARGET_ARRAY`` point-data array on a copy of the phase
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
    phase_files = sorted(case_dir.glob(f"{case_id}_T??_ssm_surface.vtp"))
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
                "stage": _respiratory_stage_from_filename(phase_file),
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
    """Write the per-lobe cross-fold summary.

    Each fold contributes one number per lobe --- the mean over that fold's
    phases --- and the reported spread is across those per-fold numbers.
    Pooling every phase of every fold instead would let a case with more
    acquired phases weigh more heavily than the others.
    """
    metrics = [
        ("dice", "Dice", "{:.4f}"),
        ("volume_difference_percent", "|volume difference| (%)", "{:.2f}"),
        ("surface_rmse_mm", "surface RMSE (mm)", "{:.3f}"),
        ("displacement_95th_mm", "displacement 95th percentile (mm)", "{:.3f}"),
    ]
    lines = [
        "# Leave-One-Out Cross-Validation: Lung Motion",
        "",
        f"Folds: {len(held_out_cases)}",
        f"Held-out cases: {', '.join(held_out_cases)}",
        "",
        "Each fold rebuilt the shape model without its held-out case, refitted "
        "the cohort, retrained the MeshGraphNet and scored that case against "
        "its own per-phase segmentations. Every cell is the mean over folds of "
        "the fold's own mean over phases, plus or minus the standard deviation "
        "across folds.",
        "",
        "Dice on a lobe is dominated by the size of the lobe rather than by how "
        "far it moved; surface RMSE is what resolves the motion here.",
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

    lines.extend(["", "## Pooled over all lobes", ""])
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
    """Per-fold displacement error, one number per fold rather than per lobe.

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
        "phase. Measured point by point, so a lobe predicted the right size in "
        "the wrong place is scored here and nowhere above.",
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
    """Box-plot each metric per lobe, one box per lobe over every scored phase."""
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
            # A lobe absent from every acquired frame has no box to draw; give
            # it an empty slot so the tick labels stay aligned with the lobes.
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

# nnUNetv2 (inside SegmentNVSegmentCTMRI), the registration backends and torch
# all spawn worker processes. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import would restart the whole study in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_15_lung_leave_one_out"

    # Number of folds. Each holds out one DIR-Lab case and re-runs the entire
    # chain without it, so the runtime is linear in this.
    number_of_leave_one_out_runs = 5

    # Atlas iterations used to build each fold's reference surface; 1 is a
    # single template-biased pass.
    mean_surface_iterations = 3

    # Training hyperparameters, matching Tutorial 9 (lung) so a fold's network
    # is comparable to the one that tutorial trains.
    epochs = 1500
    batch_size = 4
    learning_rate = 1.0e-3
    processor_size = 3  # message-passing hops
    hidden_dim = 128
    num_layers = 2  # MLP layers inside each encoder / processor / decoder block

    # Phase the SSM is fitted to, and therefore the phase whose anatomy the
    # predicted deformations carry into every other phase.
    reference_phase = "T70"

    # Gaussian sigma, in mm, that spreads the predicted surface displacements
    # into the continuous field the labelmap is resampled through.
    smoothing_sigma_mm = 10.0
    # Isotropic pitch every metric is measured on. Coarser than the CT, whose
    # in-plane pitch is finer than the accuracy being reported, and fine enough
    # that a lobe boundary is not quantized away.
    evaluation_spacing_mm = 2.0

    test_mode = TestTools.running_as_test()
    # Keep a test run out of the directories a full run reads and writes.
    weights_dir = LUNG_CT_DIRLAB.weights_directory(test_mode)

    output_dir = LUNG_CT_DIRLAB.output_directory(test_mode) / "tutorial_15_lung"
    shared_dir = output_dir / "shared"
    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # Under torchrun / SLURM / mpirun this is one rank of many; started plainly
    # it reports a world of one and every branch below collapses to a single
    # process.
    context = distributed_context()

    data_dir = LUNG_CT_DIRLAB.input_directory(test_mode)
    number_of_pca_components = LUNG_CT_DIRLAB.pca_components(test_mode)

    # In test mode, run the smallest study that is still a cross-validation and
    # train for a couple of epochs, to keep the run inside the pytest timeout.
    if test_mode:
        number_of_leave_one_out_runs = 2
        mean_surface_iterations = 1
        epochs = 2

    # Tutorial 6 caches one segmentation per case beside its model, and
    # Tutorial 8 one set of phase transforms per case; both are reused when
    # present because neither depends on which case is held out.
    tutorial_06_dir = LUNG_CT_DIRLAB.pca_model_file(test_mode).parent
    tutorial_08_dir = LUNG_CT_DIRLAB.output_directory(test_mode) / "tutorial_08_lung"

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

    # Directory setup and data reading

    if context.is_main:
        shared_dir.mkdir(parents=True, exist_ok=True)
    context.barrier()

    # DIR-Lab names case 8 "Case8Deploy" while every other case is "Case*Pack",
    # so match on "Case*" to avoid silently dropping it.
    reference_files = sorted(data_dir.glob(f"Case*_{reference_phase}.mha"))
    if not reference_files:
        raise FileNotFoundError(
            f"No DirLab {reference_phase} images found under {data_dir}.\n"
            "See data/DirLab-4DCT/README.md for download instructions."
        )
    case_ids = [path.name.split("_")[0] for path in reference_files]
    if len(case_ids) < 3:
        raise RuntimeError(
            f"Found only {len(case_ids)} case(s) under {data_dir}; a fold needs "
            "one case to hold out and at least two to build a shape model from."
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

    use_finetuned_distancemap_weights = icon_distancemap_weights_path.exists()
    if not use_finetuned_distancemap_weights:
        logger.warning(
            "Finetuned distance-map ICON weights not found at %s; fitting the "
            "SSM with the stock uniGradICON weights. Run "
            "tutorials/tutorial_02_lung_distancemap_finetune_icon.py to create "
            "them.",
            icon_distancemap_weights_path,
        )

    segmentation_method = SegmentNVSegmentCTMRI(log_level=log_level)
    segmentation_workflow = WorkflowConvertImageToVTK(
        segmentation_method=segmentation_method, log_level=log_level
    )
    contour_tools = ContourTools(log_level=log_level)
    transform_tools = TransformTools(log_level=log_level)

    # The cohort names the structures every fold reports and, in Step 5, reads
    # each fold's ground truth back out of the shared cache below.
    cohort = EvaluateMovementLung(reference_phase=reference_phase, log_level=log_level)
    lobe_names = cohort.label_names()

    # Step 1: the shared cache. Neither the reference-phase segmentations nor
    # the phase registrations depend on which case is held out, so they are
    # built once here and every fold reads them. Cases are split across the
    # ranks; the barrier below is what lets every rank read every case's files.
    segmentation_dir = shared_dir / "segmentations"
    transform_dir = shared_dir / "transforms"
    ground_truth_dir = shared_dir / "ground_truth"
    if context.is_main:
        for directory in (segmentation_dir, transform_dir, ground_truth_dir):
            directory.mkdir(parents=True, exist_ok=True)
    context.barrier()

    def shared_segmentation_files(case_id: str) -> tuple[Path, Path]:
        """Return this case's cached reference-phase surface and labelmap paths.

        Tutorial 6 wrote the same two files for its own population, so they are
        read from there when they exist rather than segmented again.
        """
        stem = f"{case_id}_{reference_phase}"
        tutorial_06_surface = tutorial_06_dir / f"{stem}.vtp"
        tutorial_06_labelmap = tutorial_06_dir / f"{stem}_labelmap.nii.gz"
        if tutorial_06_surface.exists() and tutorial_06_labelmap.exists():
            return tutorial_06_surface, tutorial_06_labelmap
        return (
            segmentation_dir / f"{stem}.vtp",
            segmentation_dir / f"{stem}_labelmap.nii.gz",
        )

    for case_id in _rank_share(case_ids, context):
        surface_file, labelmap_file = shared_segmentation_files(case_id)
        if not (surface_file.exists() and labelmap_file.exists()):
            logger.info("Segmenting %s %s", case_id, reference_phase)
            segmentation_result = segmentation_workflow.process(
                input_image=itk.imread(
                    str(data_dir / f"{case_id}_{reference_phase}.mha")
                ),
                anatomy_groups=["lung"],
                surface_reduction_rate=LUNG_CT_DIRLAB.surface_reduction_rate,
                extract_label_surfaces=True,
            )
            contour_tools.save_combined_surfaces(
                segmentation_result["label_surfaces"], str(surface_file)
            )
            itk.imwrite(
                segmentation_result["labelmap"], str(labelmap_file), compression=True
            )

        # Every phase registered to the reference phase. This is image-to-image
        # and never sees the shape model, so one set of transforms serves every
        # fold. Tutorial 8 wrote the same transforms; reuse them when present.
        case_transform_dir = transform_dir / case_id
        case_transform_dir.mkdir(parents=True, exist_ok=True)
        phase_files = sorted(data_dir.glob(f"{case_id}_T??.mha"))
        phase_ids = [path.stem.split("_")[1] for path in phase_files]
        wanted = [
            case_transform_dir / f"{case_id}_{p}_forward_tfm.hdf" for p in phase_ids
        ]
        if not all(path.exists() for path in wanted):
            tutorial_08_case_dir = tutorial_08_dir / case_id
            existing = [
                tutorial_08_case_dir / f"{case_id}_{p}_forward_tfm.hdf"
                for p in phase_ids
            ]
            if all(path.exists() for path in existing):
                logger.info("Reusing Tutorial 8 phase transforms for %s", case_id)
                for source, destination in zip(existing, wanted):
                    destination.write_bytes(source.read_bytes())
            else:
                logger.info("Registering %d phases of %s", len(phase_files), case_id)
                reg_workflow = WorkflowReconstructHighres4DCT(
                    time_series_images=[itk.imread(str(path)) for path in phase_files],
                    reference_image=itk.imread(
                        str(data_dir / f"{case_id}_{reference_phase}.mha")
                    ),
                    reference_time_frame=phase_ids.index(reference_phase),
                    register_reference_time_frame_to_reference_image=False,
                    registration_method=RegisterImagesGreedy(log_level=log_level),
                    log_level=log_level,
                )
                reg_workflow.set_modality("ct")
                reg_result = reg_workflow.process()
                for phase_index, destination in enumerate(wanted):
                    itk.transformwrite(
                        reg_result["forward_transforms"][phase_index], str(destination)
                    )

    # Ground truth for the folds' held-out cases: every gated frame segmented on
    # its own, so the lobes a phase is scored against came from that phase's
    # image rather than from a registration or a shape-model fit.
    for case_id in _rank_share(held_out_cases, context):
        case_ground_truth_dir = ground_truth_dir / case_id
        case_ground_truth_dir.mkdir(parents=True, exist_ok=True)
        for frame_file in sorted(data_dir.glob(f"{case_id}_T??.mha")):
            labelmap_file = case_ground_truth_dir / f"{frame_file.stem}_labelmap.nii.gz"
            if labelmap_file.exists():
                continue
            logger.info("Segmenting ground-truth frame %s", frame_file.name)
            itk.imwrite(
                segmentation_method.segment(itk.imread(str(frame_file)))["labelmap"],
                str(labelmap_file),
                compression=True,
            )
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
            training_case_ids = [c for c in case_ids if c != held_out_case]
            sample_surfaces = [
                pv.read(str(shared_segmentation_files(c)[0])) for c in training_case_ids
            ]
            reference_surface_file = fold_dir / "reference_mean_surface.vtp"
            if not reference_surface_file.exists():
                # The reference surface defines the topology every PCA input is
                # expressed in, so picking one case would make the model inherit
                # that case's shape. Use the unbiased mean of the fold's own
                # population instead.
                mean_workflow = WorkflowCreateMeanSurface(
                    surfaces=sample_surfaces, log_level=log_level
                )
                mean_workflow.set_number_of_iterations(mean_surface_iterations)
                mean_workflow.set_mask_dilation_mm(LUNG_CT_DIRLAB.mask_dilation_mm)
                mean_workflow.set_distance_squared_max(
                    LUNG_CT_DIRLAB.distancemap_squared_max
                )
                if use_finetuned_distancemap_weights:
                    mean_workflow.set_icon_weights_path(
                        str(icon_distancemap_weights_path)
                    )
                mean_workflow.process()["mean_surface"].save(
                    str(reference_surface_file)
                )
            model_workflow = WorkflowCreateStatisticalModel(
                sample_meshes=sample_surfaces,
                reference_mesh=pv.read(str(reference_surface_file)),
                number_of_pca_components=number_of_pca_components,
                icp_transform_type=LUNG_CT_DIRLAB.icp_transform_type,
                mask_dilation_mm=LUNG_CT_DIRLAB.mask_dilation_mm,
                distance_squared_max=LUNG_CT_DIRLAB.distancemap_squared_max,
                log_level=log_level,
            )
            if use_finetuned_distancemap_weights:
                model_workflow.set_icon_weights_path(str(icon_distancemap_weights_path))
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
        # carry each fit through the phases with the cached transforms.
        for case_id in _rank_share(case_ids, context):
            case_fit_dir = fits_dir / case_id
            case_fit_dir.mkdir(parents=True, exist_ok=True)
            fitted_reference_mesh_file = case_fit_dir / f"{case_id}_ssm_surface.vtp"
            pca_coefficients_file = (
                case_fit_dir / f"{case_id}_ssm_pca_coefficients.json"
            )
            if not (
                fitted_reference_mesh_file.exists() and pca_coefficients_file.exists()
            ):
                logger.info("Fold %s: fitting %s", held_out_case, case_id)
                surface_file, labelmap_file = shared_segmentation_files(case_id)
                fit_workflow = WorkflowFitStatisticalModelToPatient(
                    template_model=pca_mean_surface,
                    patient_models=[cast(pv.PolyData, pv.read(str(surface_file)))],
                    patient_image=itk.imread(
                        str(data_dir / f"{case_id}_{reference_phase}.mha")
                    ),
                    patient_labelmap=itk.imread(str(labelmap_file)),
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
                fit_workflow.set_distancemap_squared_max(
                    LUNG_CT_DIRLAB.distancemap_squared_max
                )
                if use_finetuned_distancemap_weights:
                    fit_workflow.set_labelmap_to_labelmap_icon_weights_path(
                        str(icon_distancemap_weights_path)
                    )
                fit_result = fit_workflow.process()

                coefficients = fit_workflow.pca_coefficients
                assert coefficients is not None
                pca_coefficients_file.write_text(
                    json.dumps(coefficients.tolist()), encoding="utf-8"
                )
                fit_result["fitted_reference_mesh"].save(
                    str(fitted_reference_mesh_file)
                )

            # The lung PCA model is built from surfaces only, so the fitted
            # model is itself a surface and only the .vtp is written.
            fitted_reference_mesh = cast(
                pv.PolyData, pv.read(str(fitted_reference_mesh_file))
            )
            for phase_file in sorted(data_dir.glob(f"{case_id}_T??.mha")):
                phase_id = phase_file.stem.split("_")[1]
                phase_surface_file = (
                    case_fit_dir / f"{case_id}_{phase_id}_ssm_surface.vtp"
                )
                if phase_surface_file.exists():
                    continue
                # itk.transformread returns a list; a composite is written with
                # its sub-transforms behind it, so the first entry is the whole
                # transform either way.
                forward_transform = itk.transformread(
                    str(
                        transform_dir
                        / case_id
                        / f"{case_id}_{phase_id}_forward_tfm.hdf"
                    )
                )[0]
                transform_tools.transform_pvcontour(
                    fitted_reference_mesh,
                    forward_transform,
                    with_deformation_magnitude=True,
                ).save(str(phase_surface_file))

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

        # Step 5: score the held-out case against its own segmentations. The
        # metrics are read off one process; the other ranks wait at the barrier.
        if context.is_main:
            case_fit_dir = fits_dir / held_out_case
            # Every labelmap this needs was segmented into the shared cache
            # above, so the cohort reads them straight back; the fits it scores
            # against are this fold's own, which is what makes the fold honest.
            fold_ground_truth = cohort.assemble_ground_truth(
                case_id=held_out_case,
                frame_directory=data_dir,
                fit_directory=case_fit_dir,
                cache_directory=ground_truth_dir / held_out_case,
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
                # Reported for both anatomies here, unlike Tutorial 11 (lung):
                # a study that compares folds can use a metric that is
                # insensitive in absolute terms.
                report_dice=True,
                # The point-by-point error, which is what separates a fold that
                # placed the lobes right from one that merely kept their size.
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
        _write_loo_report(all_rows, held_out_cases, lobe_names, report_file)
        plot_file = _plot_metrics_by_label(
            all_rows, lobe_names, output_dir / "loo_metrics_by_label.png"
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
        last_fold = fold_results[held_out_cases[-1]]
        tutorial_results["screenshots"] = [
            plot_file,
            tt.save_screenshot_mesh(
                cast(
                    pv.DataSet,
                    pv.read(
                        str(
                            output_dir
                            / f"fold_{held_out_cases[-1]}"
                            / "fits"
                            / held_out_cases[-1]
                            / f"{held_out_cases[-1]}_ssm_surface.vtp"
                        )
                    ),
                ),
                "held_out_fitted_surface.png",
                camera_position="iso",
                color="steelblue",
            ),
        ]
        logger.info("Last fold report: %s", last_fold["report_file"])
