"""
Tutorial 2 (Heart): Finetune uniGradICON on heart distance maps

Purpose
-------
The heart counterpart of ``tutorial_02_lung_distancemap_finetune_icon.py``.
Both finetune uniGradICON on *distance maps* rather than on CT intensities,
because that is what ``RegisterModelsDistanceMaps`` -- the labelmap-to-labelmap
stage of ``WorkflowFitStatisticalModelToPatient`` -- actually registers.  Stock
uniGradICON weights are out of distribution for those images.

The heart needs its own run rather than reusing the lung weights.  A distance
map's appearance is set by the radius it saturates at, and the heart is
registered with a far tighter mask than the lungs
(``ParametersDukeHeartLabelmaps.mask_dilation_mm`` versus
``ParametersLungCTDirLab.mask_dilation_mm``), so the two organs' distance maps
do not share an intensity distribution.  Every value that fixes that
distribution comes from ``parameters_duke_heart_labelmaps.py``, so this run
trains on exactly what Tutorial 7 (Duke Heart) later infers on.

Duke-Heart-4DLabelmaps ships segmented labelmaps, not CT, which is all this
tutorial needs.  Each labelmap serves twice: with the chamber-interior labels
dropped it yields the surface a distance map is measured to, and whole it
supplies uniGradICON's Dice loss.  Those labelmaps were produced by
``SegmentHeartSimplewareTrimmedBranches``, whose ventricle and atrium labels
cover the cavities rather than the walls, so measuring distance to them would
measure distance to the inside of the heart; the ids to drop come from
``parameters_duke_heart_labelmaps.py`` alongside every other value this data
needs.

``ParametersDukeHeartLabelmaps.hold_out_case`` is held out -- the same case
Tutorials 6 and 7 keep out of the shape model -- and two of its gated phases are
registered three ways -- Greedy on the distance maps, stock ICON, and the finetuned ICON -- and
scored by target registration error over the anatomical landmarks Slicer markups
files supply for every frame, plus per-class Dice.

Data Required
-------------
``data/Duke-Heart-4DLabelmaps/pm????/`` with, per gated frame:
``*_labelmap.nii.gz`` (multi-label) and ``*_landmark.mrk.json`` (Slicer markups,
LPS).
"""

# Imports
from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import itk
import numpy as np
from parameters_duke_heart_labelmaps import DUKE_HEART

from monai_physio import (
    ContourTools,
    MONAIPhysioBase,
    RegisterImagesBase,
    RegisterImagesGreedy,
    RegisterImagesICON,
    TestTools,
    TransformTools,
    WorkflowFinetuneICONRegistration,
)

# Only run if this script is not imported as a module

# unigradicon finetuning is launched as a subprocess and torch spawns worker
# processes; on Windows the spawn start method re-imports this script in each
# child, so all top-level work stays under the __name__ == "__main__" guard.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_02_duke_heart_distancemap_finetune_icon"

    test_mode = TestTools.running_as_test()

    output_dir = (
        DUKE_HEART.output_directory(test_mode) / "tutorial_02_heart_distancemap"
    )
    # The workflow writes its dataset JSON, YAML config, and checkpoint tree
    # under ``weights_dir / finetune_name``.
    weights_dir = DUKE_HEART.weights_directory(test_mode)

    # Rasterized distance maps, cached so re-runs skip the rasterization.
    derived_dir = output_dir / "distance_maps"
    baselines_dir = repo_root / "tests" / "baselines"

    finetune_name = "icon_duke_heart_distancemap"

    # Distance-map normalization, shared with every heart tutorial that later
    # registers these maps, so this run trains on the same image distribution
    # they infer on.
    distance_squared_max = DUKE_HEART.distancemap_squared_max

    # Set True to finetune from scratch.  That deletes experiment_dir below,
    # including any checkpoint a previous run left there.
    run_finetuning = True

    data_dir = DUKE_HEART.hold_out_directory(test_mode)
    if test_mode:
        number_of_iterations_icon: Optional[int] = 1
        epochs = 1
    else:
        number_of_iterations_icon = 10
        epochs = 100
    number_of_iterations_greedy = DUKE_HEART.greedy_iterations(test_mode)

    log_level = logging.INFO
    reporter = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    derived_dir.mkdir(parents=True, exist_ok=True)

    contour_tools = ContourTools(log_level=log_level)
    transform_tools = TransformTools()

    # Labels left out of the surface a distance map is measured to; see the
    # parameters module.
    interior_object_ids = DUKE_HEART.interior_object_ids

    # Cohort discovery
    case_dirs = sorted(path for path in data_dir.glob("pm*") if path.is_dir())
    if len(case_dirs) < 2:
        raise FileNotFoundError(
            f"Need at least 2 Duke heart cases under {data_dir}; found "
            f"{len(case_dirs)}.\nSee data/README.md for download instructions."
        )

    def frames_for_case(case_dir: Path) -> list[Path]:
        """Return one path stem per gated frame, as its multi-label labelmap."""
        return sorted(case_dir.glob("*_labelmap.nii.gz"))

    def companion(labelmap_file: Path, suffix: str) -> Path:
        """Return the file that shares *labelmap_file*'s frame stem."""
        stem = labelmap_file.name[: -len("_labelmap.nii.gz")]
        return labelmap_file.parent / f"{stem}{suffix}"

    def distance_map_for(labelmap_file: Path) -> Path:
        """Rasterize one frame's heart distance map, caching it under derived_dir.

        Mirrors ``RegisterModelsDistanceMaps._create_masks_from_models`` so the
        finetuning inputs match what that class feeds ICON at inference: a
        signed squared distance to the heart surface, normalized to [-1, 1] by
        ``distance_squared_max``, then multiplied by 1000 to fill the
        [-1000, 1000] window uniGradICON's CT preprocessing expects.
        """
        stem = labelmap_file.name[: -len("_labelmap.nii.gz")]
        # Frame stems repeat across cases, so the case directory name is part of
        # the cache key; without it one case's map would be read for another's.
        distance_map_file = (
            derived_dir / f"{labelmap_file.parent.name}_{stem}_distance_map.mha"
        )
        if distance_map_file.exists():
            return distance_map_file

        reporter.log_info("Rasterizing distance map for %s", stem)
        labelmap_image = itk.imread(str(labelmap_file))
        labels = itk.GetArrayViewFromImage(labelmap_image)
        # The chamber labels cover the cavities rather than their walls, so a
        # distance measured to them would be a distance to the inside of the
        # heart.  Dropping them leaves the myocardium and vessels, whose
        # boundary is the surface RegisterModelsDistanceMaps measures to at
        # inference.
        heart_mask = itk.GetImageFromArray(
            np.where((labels > 0) & ~np.isin(labels, interior_object_ids), 1, 0).astype(
                np.uint8
            )
        )
        heart_mask.CopyInformation(labelmap_image)
        surface = contour_tools.extract_contours(heart_mask)
        distance_map = contour_tools.create_distance_map(
            surface,
            labelmap_image,
            squared_distance=True,
            negative_inside=True,
            zero_inside=False,
            norm_to_max_distance=distance_squared_max,
        )
        itk.GetArrayViewFromImage(distance_map)[...] *= 1000
        itk.imwrite(distance_map, str(distance_map_file), compression=True)
        return distance_map_file

    # Held-out patient, excluded from finetuning entirely: the case Tutorials 6
    # and 7 also keep out, so one patient is unseen by everything downstream.
    held_out_dir = next(
        (path for path in case_dirs if path.name == DUKE_HEART.hold_out_case),
        case_dirs[0],
    )
    training_dirs = [path for path in case_dirs if path != held_out_dir]

    held_out_frames = frames_for_case(held_out_dir)
    if len(held_out_frames) < 2:
        raise FileNotFoundError(
            f"Held-out case {held_out_dir.name} has {len(held_out_frames)} frames; "
            "at least 2 are needed to form an evaluation pair."
        )
    # The 30% and 70% gated phases are the most separated, so they are the
    # hardest pair in the case and the most informative to score.
    moving_labelmap_file = held_out_frames[int(len(held_out_frames) * 0.3)]
    fixed_labelmap_file = held_out_frames[int(len(held_out_frames) * 0.7)]

    subject_distance_map_files: list[list[str]] = []
    subject_labelmap_files: list[list[Optional[str]]] = []
    subject_ids: list[str] = []
    for case_dir in training_dirs:
        frames = frames_for_case(case_dir)
        if len(frames) < 2:
            reporter.log_warning(
                "Case %s has %d frame(s); skipping (paired training needs 2+)",
                case_dir.name,
                len(frames),
            )
            continue
        subject_ids.append(case_dir.name)
        subject_distance_map_files.append(
            [str(distance_map_for(frame)) for frame in frames]
        )
        subject_labelmap_files.append([str(frame) for frame in frames])
    if not subject_ids:
        raise FileNotFoundError(
            f"No training case under {data_dir} has the 2+ frames paired "
            "finetuning needs; every case but the held-out one was skipped."
        )
    reporter.log_info(
        "Finetuning cohort: %d cases, %d frames (held out %s)",
        len(subject_ids),
        sum(len(files) for files in subject_distance_map_files),
        held_out_dir.name,
    )

    # Always finetune from scratch.  uniGradICON refuses to overwrite an
    # existing experiment directory: it appends "-N" to the name instead
    # (``icon_duke_heart_distancemap_model-5``, ...), while
    # expected_weights_path() keeps pointing at the original, never-written
    # path.  Deleting the tree up front keeps the two in agreement.
    experiment_dir = weights_dir / finetune_name
    if run_finetuning:
        if experiment_dir.exists():
            reporter.log_info(
                "Removing previous finetuning outputs: %s", experiment_dir
            )
            shutil.rmtree(experiment_dir)

        # The labelmaps are supplied, so uniGradICON's Dice loss stays enabled
        # at its default weight.  They hold only the heart's own structures,
        # which keeps the one-hot encoding the Dice term builds small.
        #
        # lncc_sigma matches the sigma RegisterImagesICON uses at inference, so
        # finetuning optimizes the similarity this comparison scores.  The
        # distance maps are already scaled into [-1000, 1000], so the default CT
        # window passes them through unclipped.
        workflow = WorkflowFinetuneICONRegistration(
            subject_image_files=subject_distance_map_files,
            output_dir=weights_dir,
            finetune_name=finetune_name,
            subject_ids=subject_ids,
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
                "Set run_finetuning = True to create one."
            )

    # Registration comparison on the held-out patient
    fixed_distance_map = itk.imread(
        str(distance_map_for(fixed_labelmap_file)), pixel_type=itk.F
    )
    moving_distance_map = itk.imread(
        str(distance_map_for(moving_labelmap_file)), pixel_type=itk.F
    )
    fixed_labelmap = itk.imread(str(fixed_labelmap_file))
    moving_labelmap = itk.imread(str(moving_labelmap_file))

    def read_landmarks(labelmap_file: Path) -> dict[str, np.ndarray]:
        """Read a frame's Slicer markups file as ``{label: LPS point}``.

        The markups files declare ``coordinateSystem: LPS``, the frame this
        project works in, so the control points are used as written.  A file
        written in Slicer's RAS default would flip X and Y, so the declaration
        is checked rather than assumed.
        """
        landmark_file = companion(labelmap_file, "_landmark.mrk.json")
        with landmark_file.open(encoding="utf-8") as f:
            markups = json.load(f)["markups"]
        for markup in markups:
            coordinate_system = markup.get("coordinateSystem")
            if coordinate_system != "LPS":
                raise ValueError(
                    f"{landmark_file.name} declares coordinateSystem "
                    f"{coordinate_system!r}; this tutorial reads LPS markups."
                )
        return {
            point["label"]: np.asarray(point["position"], dtype=np.float64)
            for markup in markups
            for point in markup["controlPoints"]
        }

    fixed_landmarks_by_name = read_landmarks(fixed_labelmap_file)
    moving_landmarks_by_name = read_landmarks(moving_labelmap_file)
    shared_landmarks = sorted(
        set(fixed_landmarks_by_name) & set(moving_landmarks_by_name)
    )
    if not shared_landmarks:
        raise ValueError(
            f"No landmark names shared between {fixed_labelmap_file.name} and "
            f"{moving_labelmap_file.name}."
        )
    fixed_landmarks = np.array(
        [fixed_landmarks_by_name[name] for name in shared_landmarks]
    )
    moving_landmarks = np.array(
        [moving_landmarks_by_name[name] for name in shared_landmarks]
    )
    reporter.log_info(
        "Scoring %d shared landmarks on held-out case %s",
        len(shared_landmarks),
        held_out_dir.name,
    )

    def landmark_metrics(errors_mm: np.ndarray) -> dict[str, Any]:
        """Summarize per-landmark target registration errors, in millimeters."""
        return {
            "tre_mean": float(errors_mm.mean()),
            "tre_std": float(errors_mm.std()),
            "tre_max": float(errors_mm.max()),
        }

    def landmark_errors(transform: itk.Transform) -> np.ndarray:
        """Distance from each mapped fixed landmark to its moving counterpart.

        ``fixed_to_moving_transform`` is the resampling transform: it maps points on the
        fixed grid back into moving space, which is the direction the landmark
        correspondences are defined in.
        """
        mapped = np.array(
            [transform.TransformPoint(tuple(point)) for point in fixed_landmarks]
        )
        return np.asarray(np.linalg.norm(mapped - moving_landmarks, axis=1))

    fixed_labels = itk.array_from_image(fixed_labelmap)

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
            "dice_min": float(dice.min()),
            "dice_max": float(dice.max()),
            "mislabeled_voxels": int(np.count_nonzero(fixed_labels != labels)),
        }

    # Reference row: the moving labelmap on the fixed grid, unregistered.
    unregistered_labelmap = itk.resample_image_filter(
        moving_labelmap,
        Interpolator=itk.NearestNeighborInterpolateImageFunction.New(moving_labelmap),
        ReferenceImage=fixed_labelmap,
        UseReferenceImage=True,
    )
    labelmaps: dict[str, itk.Image] = {"unregistered": unregistered_labelmap}
    rows: list[dict[str, Any]] = [
        {
            "method": "unregistered",
            "weights": "-",
            "registration_time_s": None,
            **landmark_metrics(
                np.linalg.norm(fixed_landmarks - moving_landmarks, axis=1)
            ),
            **overlap_metrics(unregistered_labelmap),
        }
    ]
    for method_name, method_weights in (
        ("greedy", None),
        ("icon_stock", None),
        ("icon_finetuned", weights_path),
    ):
        registrar: RegisterImagesBase
        if method_name == "greedy":
            registrar = RegisterImagesGreedy(log_level=log_level)
            registrar.set_transform_type("Deformable")
            # CC is what RegisterModelsDistanceMaps uses on distance maps.
            registrar.set_metric("CC")
            registrar.set_number_of_iterations(number_of_iterations_greedy)
        else:
            registrar = RegisterImagesICON(log_level=log_level)
            # None, not 0: icon_registration rejects 0 and takes None to mean
            # "no test-time finetuning steps".
            registrar.set_number_of_iterations(number_of_iterations_icon)
            # Mass preservation models CT density; a distance map carries no
            # mass, so it stays off here.
            registrar.set_mass_preservation(False)
            if method_weights is not None:
                registrar.set_weights_path(str(method_weights))
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_distance_map)

        start_time = time.perf_counter()
        result = registrar.register(moving_distance_map)
        elapsed_s = time.perf_counter() - start_time

        labelmaps[method_name] = transform_tools.transform_image(
            moving_labelmap,
            result["fixed_to_moving_transform"],
            fixed_labelmap,
            interpolation_method="nearest",
        )
        rows.append(
            {
                "method": method_name,
                "weights": str(method_weights) if method_weights else "-",
                "registration_time_s": elapsed_s,
                **landmark_metrics(
                    landmark_errors(result["fixed_to_moving_transform"])
                ),
                **overlap_metrics(labelmaps[method_name]),
            }
        )

    # Result saving
    output_dir.mkdir(parents=True, exist_ok=True)
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

    for row in rows:
        reporter.log_info(
            "%-16s TRE mean %s mm, Dice mean %.4f",
            row["method"],
            f"{row['tre_mean']:.4f}",
            row["dice_mean"],
        )

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

    tutorial_results = {
        "weights_path": weights_path,
        "registration_metrics": rows,
        "labelmaps": labelmaps,
        "summary_file": summary_file,
        "screenshots": screenshots,
    }
