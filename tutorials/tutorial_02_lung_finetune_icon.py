"""
Tutorial 2: Finetune uniGradICON on TCIA-4DLung

Purpose
-------
Finetune uniGradICON on every TCIA-4DLung case except ``100_HM10395``, then
register ``100_HM10395_g000.nii.gz`` (moving) to ``100_HM10395_g050.nii.gz``
(fixed) several ways: ``RegisterImagesGreedy`` alone, deformable, with its
default iteration schedule; ``RegisterImagesICON`` with the stock uniGradICON
weights and with the finetuned weights; and ``RegisterImagesGreedyICON`` --
the same Greedy stage initializing an ICON stage that carries the finetuned
weights -- which separates what the finetuned network adds on its own from
what it adds on top of a classical affine-plus-deformable initialization.
``100_HM10395`` is never seen during finetuning, so it is a held-out
evaluation pair.

Accuracy is measured two ways.  The primary metric is image similarity:
TCIA-4DLung ships no expert landmarks (unlike DIR-Lab), so each method's
warped image is compared to the fixed image directly -- normalized
cross-correlation (1.0 is a perfect match) and RMSE in HU, both restricted to
the fixed image's segmented foreground.  The secondary metric is label
overlap: ``SegmentNVSegmentCTMRI`` segments the fixed and moving images once
each, and the moving labelmap is warped onto the fixed grid by every
transform, so the Dice scores reflect the transform rather than segmentation
variability on re-segmented warped volumes.  The moving image and labelmap
resampled onto the fixed grid without registration supply the "before
registration" reference row for both metrics.

Reported per method: normalized cross-correlation and RMSE (HU) between the
warped and fixed image; the mean, 5th percentile, median, 95th percentile,
minimum and maximum of the per-class Dice scores; the number of mislabeled
voxels; and the wall-clock registration time.  The ``loss`` column is *not*
comparable across rows: each backend reports its own metric, and a chain
reports only its last stage's loss, measured against data the earlier stage
already warped.

Why the chain does not win here
-------------------------------
On this pair ``RegisterImagesGreedyICON`` scores no better than Greedy alone, and
the extra rows exist to show why rather than to hide it.

Compare a chain row against ``greedy_icon_stage0`` -- that row is the chain's own
Greedy stage scored on its own transform, and it is the right comparator because
Greedy is not bit-reproducible run to run (its scatter is a few thousandths of a
millimeter, the same size as the effect being measured).  ``icon_residual_*`` is
how far the ICON stage moved sampled foreground points and ``ncc_delta``/
``rmse_delta`` how much the similarity metrics changed as a result.

What that shows: the ICON residual is small, and its size at a sampled point is
essentially uncorrelated with how much it helps or hurts -- it is a random
perturbation of an already-better transform, not a correction.  Raising ICON's
test-time optimization steps shrinks the residual and the damage together, so the
chain converges toward simply reproducing its Greedy stage, at several times the
runtime.  The reason is resolution: ICON's residual deformation lives on
uniGradICON's fixed 175^3 network grid, which over this roughly 250mm field of
view is about 1.4mm between nodes -- coarser than the error Greedy has already
reached.  A refinement stage that cannot resolve the error it is asked to remove
has nothing to contribute, and the chain applies its residual unconditionally.

Ruled out as causes: the two stages are configured identically; the transform
composition was verified; every row is scored the same way in the same
direction; and the pre-warp between stages leaves under a percent of the fixed
grid without moving data.

Finetuning artifacts (dataset JSON, YAML config, checkpoint tree) are written
under ``tutorials/network_weights/icon_tcia_4dlung``.  The final checkpoint is
``tutorials/network_weights/icon_tcia_4dlung/icon_tcia_4dlung_model/checkpoints/
network_weights_final.trch``, the path returned by
``WorkflowFinetuneICONRegistration.expected_weights_path()``.  ``run_finetuning``
is off by default so runs reuse that checkpoint; turning it on deletes the
directory and finetunes from scratch.

Data Required
-------------
Full data: ``data/TCIA-4DLung`` (all cases, ``*_HM10395/*_HM10395_g0??.nii.gz``)
Test data: ``data/test/TCIA-4DLung``
"""

# Imports
from __future__ import annotations

import csv
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import itk
import numpy as np
from parameters_tcia_4d_lung import TCIA_4D_LUNG

from monai_physio import (
    MONAIPhysioBase,
    ProcessTests,
    ProcessTransforms,
    RegisterImagesBase,
    RegisterImagesGreedy,
    RegisterImagesGreedyICON,
    RegisterImagesICON,
    SegmentNVSegmentCTMRI,
    WorkflowFinetuneICONRegistration,
)

# Only run if this script is not imported as a module

# unigradicon finetuning is launched as a subprocess and torch spawns worker
# processes; on Windows the spawn start method re-imports this script in each
# child, so all top-level work stays under the __name__ == "__main__" guard.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_02_lung_finetune_icon"

    test_mode = ProcessTests.running_as_test()

    output_dir = TCIA_4D_LUNG.output_directory(test_mode) / "tutorial_02_lung"
    # The workflow writes its dataset JSON, YAML config, and checkpoint tree
    # under ``weights_dir / finetune_name``.
    weights_dir = TCIA_4D_LUNG.weights_directory(test_mode)
    cases_dir = TCIA_4D_LUNG.cases_directory(test_mode)
    hold_out_case = TCIA_4D_LUNG.mgn_hold_out_case

    # Segmented labelmaps, cached so re-runs skip the segmentation.
    labelmaps_dir = output_dir / "labelmaps"
    baselines_dir = repo_root / "tests" / "baselines"

    finetune_name = "icon_tcia_4dlung"

    # Set True to finetune from scratch.  That deletes experiment_dir below,
    # including any checkpoint a previous run left there.
    run_finetuning = True

    if test_mode:
        number_of_iterations_greedy: Optional[list[int]] = [1, 0]
        number_of_iterations_icon = None  # [1]
        epochs = 1
    else:
        number_of_iterations_greedy = None  # [60, 30, 20]
        number_of_iterations_icon = None  # [10]
        epochs = 100

    log_level = logging.INFO
    reporter = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    labelmaps_dir.mkdir(parents=True, exist_ok=True)

    # Held-out evaluation pair (the hold-out case is excluded from
    # finetuning).  g000 and g050 are the extreme inhale/exhale phases used
    # here for the largest deformation available.
    fixed_file = cases_dir / hold_out_case / f"{hold_out_case}_g050.nii.gz"
    moving_file = cases_dir / hold_out_case / f"{hold_out_case}_g000.nii.gz"
    missing = [str(p) for p in (fixed_file, moving_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing TCIA-4DLung phase images: {missing}.\n"
            "See data/TCIA-4DLung/README.md for download instructions."
        )

    # Finetuning cohort: every case except the held-out one.
    training_files = sorted(
        path
        for path in cases_dir.glob("*_HM10395/*_HM10395_g0??.nii.gz")
        if path.parent.name != hold_out_case
    )
    subject_image_files: dict[str, list[str]] = {}
    for path in training_files:
        subject_image_files.setdefault(path.parent.name, []).append(str(path))
    if not subject_image_files:
        raise FileNotFoundError(
            f"No non-{hold_out_case} TCIA-4DLung phase images found under "
            f"{cases_dir}.\nSee data/TCIA-4DLung/README.md for download "
            "instructions."
        )
    reporter.log_info(
        "Finetuning cohort: %d cases, %d frames",
        len(subject_image_files),
        len(training_files),
    )

    # Always finetune from scratch.  uniGradICON refuses to overwrite an
    # existing experiment directory: it appends "-N" to the name instead
    # (``icon_tcia_4dlung_model-5``, ...), while expected_weights_path() keeps
    # pointing at the original, never-written path.  Deleting the tree up front
    # keeps the two in agreement.
    #
    # To reuse a previous run instead, delete the shutil.rmtree call below and
    # guard the process() call:
    #     weights_path = workflow.expected_weights_path()
    #     if not weights_path.exists():
    #         weights_path = workflow.process()
    experiment_dir = weights_dir / finetune_name
    if run_finetuning:
        if experiment_dir.exists():
            reporter.log_info(
                "Removing previous finetuning outputs: %s", experiment_dir
            )
            shutil.rmtree(experiment_dir)

        # No labelmaps or masks are supplied here, so the Dice loss must be
        # disabled: uniGradICON requires a ``segmentation`` field on every
        # dataset entry when dice_loss_weight > 0.
        #
        # lncc_sigma matches the sigma RegisterImagesICON uses at inference, so
        # finetuning optimizes the similarity this comparison scores.
        workflow = WorkflowFinetuneICONRegistration(
            subject_image_files=list(subject_image_files.values()),
            output_dir=weights_dir,
            finetune_name=finetune_name,
            subject_ids=list(subject_image_files.keys()),
            epochs=epochs,
            dice_loss_weight=0.0,
            lncc_sigma=5,
            log_level=log_level,
        )
        weights_path = workflow.process()
    else:
        weights_path = (
            experiment_dir
            / f"{finetune_name}_model"
            / "checkpoints"
            / "network_weights_final.trch"
        )
        # Checked here rather than at the first set_weights_path() call, which
        # only happens after the greedy and stock-ICON rows have already run.
        if not weights_path.exists():
            raise FileNotFoundError(
                f"run_finetuning is False but no checkpoint at {weights_path}.  "
                "Set run_finetuning = True to finetune from scratch."
            )

    # Registration comparison
    fixed_image = itk.imread(str(fixed_file), pixel_type=itk.F)
    moving_image = itk.imread(str(moving_file), pixel_type=itk.F)
    transform_tools = ProcessTransforms()

    # Each image is segmented once and the moving labelmap is warped by every
    # transform, so Dice reflects the transform rather than what the segmenter
    # does differently on each interpolated volume.
    segmenter = SegmentNVSegmentCTMRI(log_level=log_level)

    def segment_phase(image_file: Path, image: itk.Image) -> itk.Image:
        """Segment one phase, caching the labelmap under ``labelmaps_dir``.

        An existing labelmap short-circuits the segmentation, which dominates
        this tutorial's runtime outside of finetuning.
        """
        # ``.stem`` only strips ``.gz``, leaving a stray ``.nii`` in the name,
        # since these are ``.nii.gz`` (TCIA) rather than ``.mha`` (DIR-Lab).
        image_stem = image_file.name.removesuffix(".nii.gz")
        labelmap_file = labelmaps_dir / f"{image_stem}_labelmap.mha"
        if labelmap_file.exists():
            reporter.log_info("Reusing cached labelmap: %s", labelmap_file.name)
            return itk.imread(str(labelmap_file))

        labelmap = segmenter.segment(image)["labelmap"]
        itk.imwrite(labelmap, str(labelmap_file), compression=True)
        return labelmap

    fixed_labelmap = segment_phase(fixed_file, fixed_image)
    moving_labelmap = segment_phase(moving_file, moving_image)
    fixed_labels = itk.array_from_image(fixed_labelmap)

    # TCIA-4DLung ships no expert landmarks, unlike DIR-Lab, so accuracy is
    # read off image similarity instead of target registration error: how well
    # each method's warped image matches the fixed one, within the fixed
    # segmentation's footprint.
    fixed_roi = fixed_labels != 0

    def similarity_metrics(warped_image: itk.Image) -> dict[str, Any]:
        """NCC and RMSE (HU) between a warped and the fixed image, in the ROI."""
        warped_vals = itk.array_from_image(warped_image)[fixed_roi].astype(np.float64)
        fixed_vals = itk.array_from_image(fixed_image)[fixed_roi].astype(np.float64)
        ncc = float(np.corrcoef(fixed_vals, warped_vals)[0, 1])
        rmse = float(np.sqrt(np.mean((fixed_vals - warped_vals) ** 2)))
        return {"ncc": ncc, "rmse": rmse}

    def overlap_metrics(labelmap: itk.Image) -> dict[str, Any]:
        """Per-class Dice summary against the fixed labelmap.

        Classes are the union of the two labelmaps' non-zero ids, so a class
        found in only one of them scores 0 rather than being dropped.
        """
        labels = itk.array_from_image(labelmap)
        classes = np.union1d(np.unique(fixed_labels), np.unique(labels))
        classes = classes[classes != 0]
        dice = np.array(
            [
                2.0
                * np.count_nonzero((fixed_labels == c) & (labels == c))
                / (np.count_nonzero(fixed_labels == c) + np.count_nonzero(labels == c))
                for c in classes
            ]
        )
        return {
            "n_classes": int(dice.size),
            "dice_mean": float(dice.mean()),
            "dice_p5": float(np.percentile(dice, 5)),
            "dice_median": float(np.median(dice)),
            "dice_p95": float(np.percentile(dice, 95)),
            "dice_min": float(dice.min()),
            "dice_max": float(dice.max()),
            "mislabeled_voxels": int(np.count_nonzero(fixed_labels != labels)),
        }

    # Reference row: the moving image and its labelmap on the fixed grid,
    # unregistered.
    unregistered_image = itk.resample_image_filter(
        moving_image,
        ReferenceImage=fixed_image,
        UseReferenceImage=True,
        # Air, not ITK's default 0, which in CT is water.
        DefaultPixelValue=-1000.0,
    )
    unregistered_labelmap = itk.resample_image_filter(
        moving_labelmap,
        Interpolator=itk.NearestNeighborInterpolateImageFunction.New(moving_labelmap),
        ReferenceImage=fixed_image,
        UseReferenceImage=True,
    )

    # Diagnostic columns describing what the ICON stage of the chain added on
    # top of its Greedy stage.  They are empty on rows where they do not apply,
    # but csv.DictWriter takes its field names from the first row, so every row
    # has to carry the keys.  The residual is sampled at a strided subset of
    # the fixed ROI's voxels, in place of DIR-Lab's expert landmarks.
    empty_chain_diagnostics: dict[str, Any] = {
        "icon_residual_mean": None,
        "icon_residual_p95": None,
        "icon_residual_max": None,
        "ncc_delta": None,
        "rmse_delta": None,
    }

    # Every 500th ROI voxel, in physical space -- a subsample of a few
    # thousand points spread over the whole lung, not a handful of landmarks.
    roi_indices = np.argwhere(fixed_roi)[::500]
    sample_points = np.array(
        [
            fixed_image.TransformIndexToPhysicalPoint([int(v) for v in index[::-1]])
            for index in roi_indices
        ]
    )

    def warp_moving(transform: itk.Transform) -> tuple[itk.Image, itk.Image]:
        """Warp the moving image and labelmap onto the fixed grid."""
        return (
            transform_tools.transform_image(
                moving_image, transform, fixed_image, background_value=-1000.0
            ),
            transform_tools.transform_image(
                moving_labelmap,
                transform,
                fixed_image,
                interpolation_method="nearest",
            ),
        )

    registered_images: dict[str, itk.Image] = {"unregistered": unregistered_image}
    labelmaps: dict[str, itk.Image] = {"unregistered": unregistered_labelmap}
    rows: list[dict[str, Any]] = [
        {
            "method": "unregistered",
            "weights": "-",
            "registration_time_s": None,
            "loss": None,
            **similarity_metrics(unregistered_image),
            **overlap_metrics(unregistered_labelmap),
            **empty_chain_diagnostics,
        }
    ]

    # Freezing Greedy and sweeping only the ICON stage's test-time optimization
    # steps makes the residual the sole difference between the chain rows, which
    # bounds how much test-time optimization could recover.
    icon_step_sweep: list[Optional[int]] = [None, 10, 50]
    methods: list[tuple[str, Optional[Path], Optional[int]]] = [
        ("greedy", None, None),
        ("icon_stock", None, number_of_iterations_icon),
        ("icon_finetuned", weights_path, number_of_iterations_icon),
    ]
    methods += [
        (f"greedy_icon_finetuned_steps_{steps}", weights_path, steps)
        for steps in icon_step_sweep
    ]

    greedy_stage_transform: Optional[itk.Transform] = None
    greedy_stage_metrics: Optional[dict[str, Any]] = None
    for method_name, method_weights, icon_steps in methods:
        registrar: RegisterImagesBase
        chain: Optional[RegisterImagesGreedyICON] = None
        if method_name == "greedy":
            registrar = RegisterImagesGreedy(log_level=log_level)
            registrar.set_transform_type("Deformable")
            if number_of_iterations_greedy is not None:
                registrar.set_number_of_iterations(number_of_iterations_greedy)
        elif method_name.startswith("greedy_icon"):
            # Both stages are configured exactly as the standalone "greedy" and
            # "icon_finetuned" rows above, so these rows differ from
            # "icon_finetuned" only by the Greedy transform ICON starts from.
            chain = RegisterImagesGreedyICON(log_level=log_level)
            chain.greedy.set_transform_type("Deformable")
            if number_of_iterations_greedy is not None:
                chain.greedy.set_number_of_iterations(number_of_iterations_greedy)
            chain.icon.set_number_of_iterations(icon_steps)
            chain.icon.set_mass_preservation(True)  # For non-contrast CT
            chain.icon.set_weights_path(str(method_weights))
            registrar = chain
        else:
            registrar = RegisterImagesICON(log_level=log_level)
            # None, not 0: icon_registration rejects 0 and takes None to mean
            # "no test-time finetuning steps", so the comparison reflects what
            # each set of weights predicts rather than per-pair optimization.
            registrar.set_number_of_iterations(icon_steps)
            registrar.set_mass_preservation(True)  # For non-contrast CT
            if method_weights is not None:
                registrar.set_weights_path(str(method_weights))
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)

        start_time = time.perf_counter()
        result = registrar.register(moving_image)
        elapsed_s = time.perf_counter() - start_time

        registered_images[method_name], labelmaps[method_name] = warp_moving(
            result["fixed_to_moving_transform"]
        )
        composed_metrics = similarity_metrics(registered_images[method_name])
        chain_diagnostics = dict(empty_chain_diagnostics)
        if chain is not None:
            # RegisterImagesChain mirrors each stage's own result onto the
            # sub-registrar before composing, so chain.greedy.fixed_to_moving_transform
            # is the stage-only Greedy result and chain.icon.fixed_to_moving_transform
            # is the residual ICON added on top of it.  Both are exact
            # transforms, scored the same way as every other row.
            icon_stage_transform: itk.Transform = chain.icon.fixed_to_moving_transform
            residual_mm = np.array(
                [
                    np.linalg.norm(
                        np.asarray(icon_stage_transform.TransformPoint(tuple(point)))
                        - point
                    )
                    for point in sample_points
                ]
            )

            # The Greedy stage is identical across the sweep, so score it once.
            if greedy_stage_transform is None:
                greedy_stage_transform = chain.greedy.fixed_to_moving_transform
                stage_image, stage_labelmap = warp_moving(greedy_stage_transform)
                greedy_stage_metrics = similarity_metrics(stage_image)
                registered_images["greedy_icon_stage0"] = stage_image
                labelmaps["greedy_icon_stage0"] = stage_labelmap
                rows.append(
                    {
                        "method": "greedy_icon_stage0",
                        "weights": "-",
                        "registration_time_s": None,
                        "loss": None,
                        **greedy_stage_metrics,
                        **overlap_metrics(stage_labelmap),
                        **empty_chain_diagnostics,
                    }
                )
            assert greedy_stage_metrics is not None

            chain_diagnostics = {
                "icon_residual_mean": float(residual_mm.mean()),
                "icon_residual_p95": float(np.percentile(residual_mm, 95)),
                "icon_residual_max": float(residual_mm.max()),
                "ncc_delta": composed_metrics["ncc"] - greedy_stage_metrics["ncc"],
                "rmse_delta": composed_metrics["rmse"] - greedy_stage_metrics["rmse"],
            }

        rows.append(
            {
                "method": method_name,
                "weights": str(method_weights) if method_weights else "-",
                "registration_time_s": elapsed_s,
                # Not comparable across rows: each backend reports its own
                # metric, and a chain reports only its last stage's loss,
                # measured against data the earlier stage already warped.
                "loss": float(result["loss"]),
                **composed_metrics,
                **overlap_metrics(labelmaps[method_name]),
                **chain_diagnostics,
            }
        )

    # How much of the fixed grid the pre-warp has no moving data for.  Those
    # voxels used to be filled with ITK's default 0 -- water in CT -- which the
    # ICON stage then saw as tissue where the moving image had nothing.
    if greedy_stage_transform is not None:
        coverage_input = itk.image_from_array(
            np.ones(list(itk.size(moving_image))[::-1], dtype=np.float32)
        )
        coverage_input.CopyInformation(moving_image)
        coverage = transform_tools.transform_image(
            coverage_input, greedy_stage_transform, fixed_image, background_value=0.0
        )
        itk.imwrite(
            coverage, str(output_dir / "prewarp_coverage.mha"), compression=True
        )
        coverage_arr = itk.GetArrayFromImage(coverage)
        uncovered = coverage_arr < 0.999
        fixed_hu = itk.GetArrayFromImage(fixed_image)
        uncovered_in_air = uncovered & (fixed_hu < -500.0)
        reporter.log_info(
            "Pre-warp coverage: %d/%d fixed voxels uncovered (%.4f%%), "
            "%d of them where the fixed image is air",
            int(uncovered.sum()),
            uncovered.size,
            100.0 * uncovered.sum() / uncovered.size,
            int(uncovered_in_air.sum()),
        )
        if uncovered.any():
            k_counts = uncovered.sum(axis=(1, 2))
            reporter.log_info(
                "Uncovered voxels per slice along k: min %d, max %d, "
                "first slice %d, last slice %d",
                int(k_counts.min()),
                int(k_counts.max()),
                int(k_counts[0]),
                int(k_counts[-1]),
            )
            # ROI voxels affected by missing data are the ones a bad fill
            # value could plausibly have moved.
            roi_uncovered = int((uncovered & fixed_roi).sum())
            reporter.log_info(
                "ROI voxels within uncovered data: %d/%d",
                roi_uncovered,
                int(fixed_roi.sum()),
            )

    # Result saving
    itk.imwrite(
        fixed_labelmap, str(output_dir / "fixed_labelmap.mha"), compression=True
    )
    # Difference against the fixed image, not the resampled result: residual
    # structure is what distinguishes the methods, and it is invisible in the
    # registered images themselves.
    fixed_arr = itk.GetArrayFromImage(fixed_image).astype(np.float32)
    for method_name, image in registered_images.items():
        difference = itk.GetImageFromArray(
            fixed_arr - itk.GetArrayFromImage(image).astype(np.float32)
        )
        difference.CopyInformation(fixed_image)
        itk.imwrite(
            difference,
            str(output_dir / f"difference_{method_name}.mha"),
            compression=True,
        )
    for method_name, labelmap in labelmaps.items():
        itk.imwrite(
            labelmap,
            str(output_dir / f"labelmap_{method_name}.mha"),
            compression=True,
        )

    summary_file = output_dir / "registration_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Reporting
    reporter.log_info(
        "%s_g000 -> %s_g050, image similarity in the fixed segmented ROI",
        hold_out_case,
        hold_out_case,
    )
    reporter.log_info("  %-21s %7s %9s %9s", "method", "ncc", "rmse", "time_s")
    for row in rows:
        elapsed = row["registration_time_s"]
        reporter.log_info(
            "  %-21s %7.4f %9.2f %9s",
            row["method"],
            row["ncc"],
            row["rmse"],
            "-" if elapsed is None else f"{float(elapsed):.1f}",
        )

    reporter.log_info("Per-class Dice of the warped moving labelmap against the fixed")
    reporter.log_info(
        "  %-21s %7s %7s %7s %7s %7s %7s %7s %12s",
        "method",
        "classes",
        "mean",
        "p5",
        "median",
        "p95",
        "min",
        "max",
        "mislabeled",
    )
    for row in rows:
        reporter.log_info(
            "  %-21s %7d %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %12d",
            row["method"],
            row["n_classes"],
            row["dice_mean"],
            row["dice_p5"],
            row["dice_median"],
            row["dice_p95"],
            row["dice_min"],
            row["dice_max"],
            row["mislabeled_voxels"],
        )
    reporter.log_info("Wrote summary: %s", summary_file)

    # Testing
    tt = ProcessTests(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = [
        tt.save_screenshot_image_slice(
            fixed_image,
            "fixed_frame.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
        )
    ]
    for method_name, image in registered_images.items():
        screenshots.append(
            tt.save_screenshot_image_slice(
                image,
                f"registered_{method_name}.png",
                axis=0,
                slice_fraction=0.5,
                colormap="gray",
            )
        )

    tutorial_results = {
        "weights_path": weights_path,
        "registration_metrics": rows,
        "labelmaps": labelmaps,
        "summary_file": summary_file,
        "registered_images": registered_images,
        "screenshots": screenshots,
    }
