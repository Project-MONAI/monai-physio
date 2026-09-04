"""
Tutorial 2 (Lung, distance maps): Finetune uniGradICON on lung distance maps

Purpose
-------
``RegisterModelsDistanceMaps`` -- the labelmap-to-labelmap stage of
``WorkflowFitStatisticalModelToPatient`` -- does not register CT intensities.
It rasterizes a signed squared distance map from each surface, normalizes it to
[-1, 1] and scales it by 1000 so that it fills the CT window uniGradICON
preprocesses with, then hands that pair to ICON.  Stock uniGradICON has never
seen such an image, so this tutorial finetunes it on exactly that
representation: distance maps rasterized from the lung surfaces segmented out of
the DIR-Lab 4D CT cases.

The finetuning cohort is every DIR-Lab case except Case 1, and within each case
only every other respiratory phase (``T00``, ``T20``, ``T40``, ``T60``, ``T80``)
-- half the time points, spanning the full breathing cycle at half the
segmentation cost.  Each selected phase is segmented once with
``SegmentNVSegmentCTMRI``; the lung labelmap is kept for uniGradICON's Dice
loss and the lung surfaces are combined and rasterized into the distance map
that serves as the training "image".  Segmentation outputs are cached on disk,
so a second run of this tutorial only re-runs the finetuning.

Accuracy is measured on Case 1, which is never seen during finetuning, by
registering ``Case1Pack_T00.mha`` (moving) to ``Case1Pack_T50.mha`` (fixed) five
ways: ``RegisterImagesGreedy`` deformable on the distance maps,
``RegisterImagesICON`` on the distance maps with the stock and with the
finetuned weights, ``RegisterImagesGreedyICON`` on the distance maps with the
finetuned weights in its ICON stage -- which separates what the finetuned
network adds on its own from what it adds on top of the Greedy result -- and,
as the intensity-based reference point, stock ``RegisterImagesICON`` on the CT
images themselves.  No registration masks are used, so the comparison isolates
the method and the weights.

The two metrics match ``tutorial_02_lung_finetune_icon.py``.  The primary metric
is target registration error: DIR-Lab ships 300 expert landmarks for the extreme
phases (T00 and T50) of every case, so each fixed-image landmark is mapped
through the registration transform and compared, in millimeters, against its
moving-image counterpart.  The secondary metric is label overlap: the moving
lung labelmap is warped onto the fixed grid by every transform, so the Dice
scores reflect the transform rather than segmentation variability.  The moving
image and labelmap resampled onto the fixed grid without registration supply the
"before registration" reference row for both metrics.

Reported per method: the mean, standard deviation, 95th percentile and maximum
landmark error in millimeters; the mean, 5th percentile, median, 95th
percentile, minimum and maximum of the per-class Dice scores; the number of
mislabeled voxels; and the wall-clock registration time.

Finetuning artifacts (dataset JSON, YAML config, checkpoint tree) are written
under ``tutorials/network_weights/icon_dirlab_4dct_distancemap``.  The final
checkpoint is ``tutorials/network_weights/icon_dirlab_4dct_distancemap/
icon_dirlab_4dct_distancemap_model/checkpoints/network_weights_final.trch``, the
path returned by ``WorkflowFinetuneICONRegistration.expected_weights_path()``
and the path ``tutorial_07_lung_fit_statistical_model_to_patient.py`` and
``tutorial_08_lung_fit_model_to_4d_patients.py`` look for.  That directory is
deleted at the start of every run, so each run finetunes from scratch; see the
comment above the ``shutil.rmtree`` call for how to reuse a previous run.

Data Required
-------------
Full data: ``data/DirLab-4DCT`` (all 10 cases, converted to HU ``.mha`` by
``data/DirLab-4DCT/fix_downloaded_data.py``), including the raw
``downloaded_data/Case1Pack/ExtremePhases`` landmark files
Test data: ``data/test/DirLab-4DCT``
"""

# Imports
from __future__ import annotations

import csv
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv

from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from monai_physio import (
    ContourTools,
    MONAIPhysioBase,
    RegisterImagesBase,
    RegisterImagesGreedy,
    RegisterImagesGreedyICON,
    RegisterImagesICON,
    SegmentNVSegmentCTMRI,
    TestTools,
    TransformTools,
    WorkflowConvertImageToVTK,
    WorkflowFinetuneICONRegistration,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used inside the segmenter) spawns a multiprocessing.Pool and
# unigradicon finetuning is launched as a subprocess that spawns torch workers;
# on Windows the spawn start method re-imports this script in each child, so all
# top-level work stays under the __name__ == "__main__" guard.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_02_lung_distancemap_finetune_icon"

    test_mode = TestTools.running_as_test()

    output_dir = (
        LUNG_CT_DIRLAB.output_directory(test_mode) / "tutorial_02_lung_distancemap"
    )
    # The workflow writes its dataset JSON, YAML config, and checkpoint tree
    # under ``weights_dir / finetune_name``.
    weights_dir = LUNG_CT_DIRLAB.weights_directory(test_mode)

    # Segmented labelmaps, lung surfaces, and rasterized distance maps.  Cached
    # so re-runs skip the segmentation, which dominates this tutorial's runtime.
    derived_dir = output_dir / "distance_maps"
    baselines_dir = repo_root / "tests" / "baselines"

    finetune_name = "icon_dirlab_4dct_distancemap"

    # Distance-map normalization, shared with every lung tutorial that later
    # registers these maps, so this run trains on the same image distribution
    # they infer on.
    distance_squared_max = LUNG_CT_DIRLAB.distancemap_squared_max

    run_finetuning = True

    # Half the respiratory phases per case, taken as every other phase so the
    # selection still spans inhale through exhale.
    phase_stride = 2

    if test_mode:
        data_dir = LUNG_CT_DIRLAB.data_directory(test_mode) / "DirLab-4DCT"
        number_of_iterations_icon: Optional[int] = 1
        epochs = 1
    else:
        data_dir = LUNG_CT_DIRLAB.data_directory(test_mode) / "DirLab-4DCT"
        number_of_iterations_icon = 10
        epochs = 200
    number_of_iterations_greedy = LUNG_CT_DIRLAB.greedy_iterations(test_mode)

    log_level = logging.INFO
    reporter = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    derived_dir.mkdir(parents=True, exist_ok=True)

    # Held-out evaluation pair (Case 1 is excluded from finetuning).  T00 and
    # T50 are the extreme inhale/exhale phases, the only pair DIR-Lab supplies
    # expert landmarks for.
    fixed_file = data_dir / "Case1Pack_T50.mha"
    moving_file = data_dir / "Case1Pack_T00.mha"
    landmark_dir = data_dir / "downloaded_data" / "Case1Pack" / "ExtremePhases"
    fixed_landmark_file = landmark_dir / "Case1_300_T50_xyz.txt"
    moving_landmark_file = landmark_dir / "Case1_300_T00_xyz.txt"
    missing = [
        str(p)
        for p in (fixed_file, moving_file, fixed_landmark_file, moving_landmark_file)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing DirLab phase images or landmarks: {missing}.\n"
            "See data/DirLab-4DCT/README.md for download instructions."
        )

    # Segmentation and distance-map generation
    segmenter = SegmentNVSegmentCTMRI(log_level=log_level)
    segmentation_workflow = WorkflowConvertImageToVTK(
        segmentation_method=segmenter,
        log_level=log_level,
    )
    # WorkflowConvertImageToVTK returns the segmenter's whole-body labelmap
    # regardless of anatomy_groups -- that argument only selects which surfaces
    # are extracted.  uniGradICON's Dice loss one-hots every class the two
    # labelmaps share, on its 175^3 grid at batch 4, so handing it ~95
    # whole-body classes costs tens of gigabytes per step; the lung labels alone
    # cost a twentieth of that and are the only ones the distance maps describe.
    lung_label_ids = np.array(
        sorted(segmenter.taxonomy.labels_in_group("lung")), dtype=np.uint16
    )
    contour_tools = ContourTools(log_level=log_level)
    transform_tools = TransformTools()

    def segment_phase(image_file: Path) -> tuple[Path, Path]:
        """Segment one phase's lungs and rasterize their distance map.

        Returns the distance map and lung labelmap paths.  Both, plus the
        combined lung surface, are cached under ``derived_dir``; an existing
        pair short-circuits the segmentation.
        """
        distance_map_file = derived_dir / f"{image_file.stem}_distance_map.mha"
        labelmap_file = derived_dir / f"{image_file.stem}_lung_labelmap.nii.gz"
        surface_file = derived_dir / f"{image_file.stem}_lung_surface.vtp"
        if distance_map_file.exists() and labelmap_file.exists():
            return distance_map_file, labelmap_file

        reporter.log_info("Segmenting lungs in %s", image_file.name)
        image = itk.imread(str(image_file), pixel_type=itk.F)
        segmentation_result = segmentation_workflow.process(
            input_image=image,
            anatomy_groups=["lung"],
            extract_label_surfaces=True,
        )
        contour_tools.save_combined_surfaces(
            segmentation_result["label_surfaces"], str(surface_file)
        )
        whole_body_labelmap = segmentation_result["labelmap"]
        whole_body_arr = itk.GetArrayViewFromImage(whole_body_labelmap)
        lung_labelmap = itk.GetImageFromArray(
            np.where(np.isin(whole_body_arr, lung_label_ids), whole_body_arr, 0)
        )
        lung_labelmap.CopyInformation(whole_body_labelmap)
        itk.imwrite(lung_labelmap, str(labelmap_file), compression=True)
        surface = cast(pv.PolyData, pv.read(str(surface_file)))
        # Rasterize the lung surface into ICON's distance-map representation,
        # mirroring ``RegisterModelsDistanceMaps._create_masks_from_models`` so
        # the finetuning inputs match what that class feeds ICON at inference:
        # a signed squared distance normalized to [-1, 1] by
        # ``distance_squared_max``, then multiplied by 1000 to fill the
        # [-1000, 1000] HU window uniGradICON's CT preprocessing expects.
        distance_map = contour_tools.create_distance_map(
            surface,
            image,
            squared_distance=True,
            negative_inside=True,
            zero_inside=False,
            norm_to_max_distance=distance_squared_max,
        )
        itk.GetArrayViewFromImage(distance_map)[...] *= 1000
        itk.imwrite(distance_map, str(distance_map_file), compression=True)
        return distance_map_file, labelmap_file

    # Finetuning cohort: every case except Case1Pack, every other phase.
    # ``Case10Pack_*`` is kept because only the exact ``Case1Pack_`` prefix is
    # excluded.
    case_phase_files: dict[str, list[Path]] = {}
    for path in sorted(data_dir.glob("Case*_T??.mha")):
        if path.name.startswith("Case1Pack_"):
            continue
        case_phase_files.setdefault(path.name.split("_")[0], []).append(path)
    if not case_phase_files:
        raise FileNotFoundError(
            f"No non-Case1 DirLab phase images found under {data_dir}.\n"
            "See data/DirLab-4DCT/README.md for download instructions."
        )
    case_phase_files = {
        case_id: files[::phase_stride] for case_id, files in case_phase_files.items()
    }
    reporter.log_info(
        "Finetuning cohort: %d cases, %d frames (every %d-th phase)",
        len(case_phase_files),
        sum(len(files) for files in case_phase_files.values()),
        phase_stride,
    )

    subject_distance_map_files: list[list[str]] = []
    subject_labelmap_files: list[list[Optional[str]]] = []
    for case_id, phase_files in case_phase_files.items():
        segmented = [segment_phase(path) for path in phase_files]
        subject_distance_map_files.append([str(pair[0]) for pair in segmented])
        subject_labelmap_files.append([str(pair[1]) for pair in segmented])

    # Always finetune from scratch.  uniGradICON refuses to overwrite an
    # existing experiment directory: it appends "-N" to the name instead
    # (``icon_dirlab_4dct_distancemap_model-5``, ...), while
    # expected_weights_path() keeps pointing at the original, never-written
    # path.  Deleting the tree up front keeps the two in agreement.
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

        # Unlike tutorial_02_lung_finetune_icon.py, the lungs are segmented here
        # anyway, so the labelmaps are supplied and uniGradICON's Dice loss stays
        # enabled at its default weight.
        #
        # lncc_sigma matches the sigma RegisterImagesICON uses at inference, so
        # finetuning optimizes the similarity this comparison scores.  The
        # distance maps are already scaled into [-1000, 1000], so the default CT
        # window passes them through unclipped.
        workflow = WorkflowFinetuneICONRegistration(
            subject_image_files=subject_distance_map_files,
            output_dir=weights_dir,
            finetune_name=finetune_name,
            subject_ids=list(case_phase_files.keys()),
            subject_labelmap_files=subject_labelmap_files,
            epochs=epochs,
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

    fixed_distance_map_file, fixed_labelmap_file = segment_phase(fixed_file)
    moving_distance_map_file, moving_labelmap_file = segment_phase(moving_file)
    fixed_distance_map = itk.imread(str(fixed_distance_map_file), pixel_type=itk.F)
    moving_distance_map = itk.imread(str(moving_distance_map_file), pixel_type=itk.F)
    fixed_labelmap = itk.imread(str(fixed_labelmap_file))
    moving_labelmap = itk.imread(str(moving_labelmap_file))
    fixed_labels = itk.array_from_image(fixed_labelmap)

    def read_landmarks(landmark_file: Path, image: itk.Image) -> np.ndarray:
        """Read a DIR-Lab landmark file as an (N, 3) array of world points.

        Each line holds one 1-based voxel index as ``x y z``.
        """
        indices = np.loadtxt(landmark_file, dtype=int) - 1
        return np.array(
            [
                image.TransformIndexToPhysicalPoint([int(v) for v in index])
                for index in indices
            ]
        )

    fixed_landmarks = read_landmarks(fixed_landmark_file, fixed_image)
    moving_landmarks = read_landmarks(moving_landmark_file, moving_image)

    def landmark_metrics(errors_mm: np.ndarray) -> dict[str, Any]:
        """Summarize per-landmark target registration errors, in millimeters."""
        return {
            "tre_mean": float(errors_mm.mean()),
            "tre_std": float(errors_mm.std()),
            "tre_p95": float(np.percentile(errors_mm, 95)),
            "tre_max": float(errors_mm.max()),
        }

    def landmark_errors(transform: itk.Transform) -> np.ndarray:
        """Distance from each mapped fixed landmark to its moving counterpart.

        ``forward_transform`` is the resampling transform: it maps points on the
        fixed grid back into moving space, which is the direction the landmark
        correspondences are defined in.
        """
        mapped = np.array(
            [transform.TransformPoint(tuple(point)) for point in fixed_landmarks]
        )
        return np.asarray(np.linalg.norm(mapped - moving_landmarks, axis=1))

    def overlap_metrics(labelmap: itk.Image) -> dict[str, Any]:
        """Per-class Dice summary against the fixed lung labelmap.

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

    # Reference row: the moving distance map and labelmap on the fixed grid,
    # unregistered.
    unregistered_distance_map = itk.resample_image_filter(
        moving_distance_map,
        ReferenceImage=fixed_distance_map,
        UseReferenceImage=True,
    )
    unregistered_labelmap = itk.resample_image_filter(
        moving_labelmap,
        Interpolator=itk.NearestNeighborInterpolateImageFunction.New(moving_labelmap),
        ReferenceImage=fixed_labelmap,
        UseReferenceImage=True,
    )

    registered_distance_maps: dict[str, itk.Image] = {
        "unregistered": unregistered_distance_map
    }
    labelmaps: dict[str, itk.Image] = {"unregistered": unregistered_labelmap}
    rows: list[dict[str, Any]] = [
        {
            "method": "unregistered",
            "input": "-",
            "weights": "-",
            "registration_time_s": None,
            "loss": None,
            **landmark_metrics(
                np.linalg.norm(fixed_landmarks - moving_landmarks, axis=1)
            ),
            **overlap_metrics(unregistered_labelmap),
        }
    ]
    # (method, registration input, ICON weights).  The last row registers the CT
    # images themselves, as the intensity-based reference point for what the
    # distance-map methods achieve from surfaces alone.
    methods: list[tuple[str, str, Optional[Path]]] = [
        ("greedy_dmap", "distance_map", None),
        ("icon_stock_dmap", "distance_map", None),
        ("icon_finetuned_dmap", "distance_map", weights_path),
        ("greedy_icon_finetuned_dmap", "distance_map", weights_path),
        ("icon_stock_ct", "ct", None),
    ]
    for method_name, method_input, method_weights in methods:
        if method_input == "distance_map":
            method_fixed, method_moving = fixed_distance_map, moving_distance_map
        else:
            method_fixed, method_moving = fixed_image, moving_image

        registrar: RegisterImagesBase
        if method_name == "greedy_icon_finetuned_dmap":
            # Both stages are configured exactly as the standalone "greedy_dmap"
            # and "icon_finetuned_dmap" rows above, so this row differs from
            # "icon_finetuned_dmap" only by the Greedy transform ICON starts
            # from.
            chain = RegisterImagesGreedyICON(log_level=log_level)
            chain.greedy.set_transform_type("Deformable")
            chain.greedy.set_metric("CC")
            chain.greedy.set_number_of_iterations(number_of_iterations_greedy)
            chain.icon.set_number_of_iterations(number_of_iterations_icon)
            chain.icon.set_mass_preservation(False)
            chain.icon.set_weights_path(str(method_weights))
            registrar = chain
        elif method_name.startswith("greedy"):
            registrar = RegisterImagesGreedy(log_level=log_level)
            registrar.set_transform_type("Deformable")
            # CC is what RegisterModelsDistanceMaps uses on distance maps.
            registrar.set_metric("CC")
            registrar.set_number_of_iterations(number_of_iterations_greedy)
        else:
            registrar = RegisterImagesICON(log_level=log_level)
            # None, not 0: icon_registration rejects 0 and takes None to mean
            # "no test-time finetuning steps", so the comparison reflects what
            # each set of weights predicts rather than per-pair optimization.
            registrar.set_number_of_iterations(number_of_iterations_icon)
            # Mass preservation models CT density; a distance map carries no
            # mass, so it is enabled only on the CT reference row.
            registrar.set_mass_preservation(method_input == "ct")
            if method_weights is not None:
                registrar.set_weights_path(str(method_weights))
        registrar.set_modality("ct")
        registrar.set_fixed_image(method_fixed)

        start_time = time.perf_counter()
        result = registrar.register(method_moving)
        elapsed_s = time.perf_counter() - start_time

        registered_distance_maps[method_name] = transform_tools.transform_image(
            moving_distance_map, result["forward_transform"], fixed_distance_map
        )
        labelmaps[method_name] = transform_tools.transform_image(
            moving_labelmap,
            result["forward_transform"],
            fixed_labelmap,
            interpolation_method="nearest",
        )
        rows.append(
            {
                "method": method_name,
                "input": method_input,
                "weights": str(method_weights) if method_weights else "-",
                "registration_time_s": elapsed_s,
                "loss": float(result["loss"]),
                **landmark_metrics(landmark_errors(result["forward_transform"])),
                **overlap_metrics(labelmaps[method_name]),
            }
        )

    # Result saving
    itk.imwrite(
        fixed_distance_map, str(output_dir / "fixed_distance_map.mha"), compression=True
    )
    # Difference against the fixed distance map, not the resampled result:
    # residual structure is what distinguishes the methods, and it is invisible
    # in the registered distance maps themselves.
    fixed_arr = itk.GetArrayFromImage(fixed_distance_map).astype(np.float32)
    for method_name, distance_map in registered_distance_maps.items():
        difference = itk.GetImageFromArray(
            fixed_arr - itk.GetArrayFromImage(distance_map).astype(np.float32)
        )
        difference.CopyInformation(fixed_distance_map)
        itk.imwrite(
            difference,
            str(output_dir / f"difference_distance_map_{method_name}.mha"),
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
        "Case1Pack_T00 -> Case1Pack_T50, error at %d expert landmarks, mm",
        len(fixed_landmarks),
    )
    reporter.log_info(
        "  %-26s %7s %7s %7s %7s %9s",
        "method",
        "mean",
        "std",
        "p95",
        "max",
        "time_s",
    )
    for row in rows:
        elapsed = row["registration_time_s"]
        reporter.log_info(
            "  %-26s %7.2f %7.2f %7.2f %7.2f %9s",
            row["method"],
            row["tre_mean"],
            row["tre_std"],
            row["tre_p95"],
            row["tre_max"],
            "-" if elapsed is None else f"{float(elapsed):.1f}",
        )

    reporter.log_info("Per-class Dice of the warped moving labelmap against the fixed")
    reporter.log_info(
        "  %-26s %7s %7s %7s %7s %7s %7s %7s %12s",
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
            "  %-26s %7d %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %12d",
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
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = [
        tt.save_screenshot_image_slice(
            fixed_distance_map,
            "fixed_distance_map.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
        )
    ]
    for method_name, distance_map in registered_distance_maps.items():
        screenshots.append(
            tt.save_screenshot_image_slice(
                distance_map,
                f"registered_distance_map_{method_name}.png",
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
        "registered_distance_maps": registered_distance_maps,
        "screenshots": screenshots,
    }
