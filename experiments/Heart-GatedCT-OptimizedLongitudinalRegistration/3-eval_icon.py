# %% [markdown]
# # Evaluate ICON default vs finetuned weights on held-out longitudinal CT
#
# Enumerates the Duke patient cohort from ``timepoint_base_dir`` subdirectories
# and uses the *last 20%* of patients as the held-out test set - the same fixed
# split applied by ``2-finetune_icon.py`` (first 80% train, last 20% test).
# Each subject directory must contain exactly one file whose name ends with
# ``ref.nii.gz``; that file is the fixed image for all registration methods.
# All gated frames (``_g[0-9]{3}.nii.gz``) are registered to it as moving
# images.  The resampler-convention inverse transform (which maps moving-grid
# points back to reference-grid points) is applied to each time-point's
# precomputed landmarks to land them in reference space, and the Euclidean
# error against the reference landmarks is recorded.
#
# Run interactively cell-by-cell; all paths are hard-coded.

# %%
import csv
import re
from pathlib import Path

import itk
import numpy as np

from monai_physio import (
    RegisterImagesBase,
    RegisterImagesGreedy,
    RegisterImagesGreedyICON,
    RegisterImagesICON,
    RegisterTimeSeriesImages,
    SegmentHeartSimpleware,
)
from monai_physio.labelmap_tools import LabelmapTools
from monai_physio.landmark_tools import LandmarkTools
from monai_physio.transform_tools import TransformTools


def _build_registrar(
    reg_method: str,
    greedy_iters,
    icon_iterations,
    weights_path,
) -> RegisterImagesBase:
    """Build a configured registrar instance for one of "Greedy", "ICON", or
    "Greedy_ICON", matching this experiment's per-method config entries."""
    if reg_method == "Greedy":
        greedy = RegisterImagesGreedy()
        if greedy_iters is not None:
            greedy.set_number_of_iterations(greedy_iters)
        return greedy
    if reg_method == "ICON":
        icon = RegisterImagesICON()
        icon.set_number_of_iterations(icon_iterations)
        if weights_path is not None:
            icon.set_weights_path(str(weights_path))
        return icon
    if reg_method == "Greedy_ICON":
        greedy_icon = RegisterImagesGreedyICON()
        if greedy_iters is not None:
            greedy_icon.greedy.set_number_of_iterations(greedy_iters)
        greedy_icon.icon.set_number_of_iterations(icon_iterations)
        if weights_path is not None:
            greedy_icon.icon.set_weights_path(str(weights_path))
        return greedy_icon
    raise ValueError(f"Unknown registration method: {reg_method}")


# %% [markdown]
# ## 1. Hard-coded paths and configuration

# %%
timepoint_base_dir = Path("d:/MONAI-Physio/duke_data/gated_nii")
segmentation_base_dir = Path("d:/MONAI-Physio/duke_data/simple_ascardio")

_HERE = Path(__file__).parent
output_dir = _HERE / "results_icon_eval"
finetuned_weights_path = (
    _HERE
    / "results_finetuning"
    / "icon_finetuning"
    / "icon_finetuning_model-2"
    / "checkpoints"
    / "network_weights_final.trch"
)

train_fraction = 0.8
icon_iterations = None
exclude_tokens = ["nop"]
timepoint_re = re.compile(r"_g(?P<timepoint>[0-9]{3})")

# Each entry: name, reg_method, weights_path, use_mask, greedy_iters
# All methods register gated frames to the per-subject reference scan (*ref.nii.gz).
all_methods = [
    {
        "name": "icon_default",
        "reg_method": "ICON",
        "weights_path": None,
        "use_mask": True,
        "greedy_iters": None,
    },
    {
        "name": "icon_finetuned",
        "reg_method": "ICON",
        "weights_path": finetuned_weights_path,
        "use_mask": True,
        "greedy_iters": None,
    },
    {
        "name": "Greedy",
        "reg_method": "Greedy",
        "weights_path": None,
        "use_mask": True,
        "greedy_iters": [80, 40, 5],
    },
    {
        "name": "Greedy_ICON",
        "reg_method": "Greedy_ICON",
        "weights_path": finetuned_weights_path,
        "use_mask": True,
        "greedy_iters": [80, 40, 5],
    },
]

all_patient_ids = sorted(
    p.name
    for p in timepoint_base_dir.iterdir()
    if p.is_dir() and p.name.startswith("pm00")
)
if len(all_patient_ids) < 2:
    raise ValueError(
        f"Expected at least 2 subjects under {timepoint_base_dir}, "
        f"found {len(all_patient_ids)}."
    )
n_train = max(
    1, min(len(all_patient_ids) - 1, round(train_fraction * len(all_patient_ids)))
)
test_subjects = all_patient_ids[n_train:]
print(
    f"Cohort: {len(all_patient_ids)} patients; "
    f"first {n_train} train, last {len(test_subjects)} test."
)
print(f"Held-out test subjects: {test_subjects}")

output_dir.mkdir(parents=True, exist_ok=True)
detail_file = output_dir / "landmark_errors_by_point.csv"
summary_file = output_dir / "registration_summary.csv"
warped_ref_detail_file = output_dir / "warped_ref_landmark_errors_by_point.csv"
if detail_file.exists():
    detail_file.unlink()
if warped_ref_detail_file.exists():
    warped_ref_detail_file.unlink()

# %% [markdown]
# ## 2. Validate that every test subject has exactly one reference file

# %%
missing = []
for subject_id in test_subjects:
    ref_candidates = list((timepoint_base_dir / subject_id).glob("*ref.nii.gz"))
    if len(ref_candidates) != 1:
        missing.append(
            f"{subject_id}: found {len(ref_candidates)} ref file(s)"
            + (f" {ref_candidates}" if ref_candidates else "")
        )
if missing:
    raise FileNotFoundError(
        "Missing or ambiguous ref.nii.gz for test subjects:\n"
        + "\n".join(f"  {m}" for m in missing)
    )
print("All test subjects have exactly one ref.nii.gz")

# %% [markdown]
# ## 3. Reader instance used in the per-frame inner loop
#
# Landmarks are read with :meth:`LandmarkTools.read_landmarks_3dslicer` -
# they were written as ``<stem>_landmark.mrk.json`` (3D Slicer Markups JSON,
# LPS) by ``0-cardiacGatedCT_segment_and_landmark.py``.  Binary registration
# masks come from :meth:`LabelmapTools.convert_labelmap_to_mask` (``>0``
# threshold plus 5 mm dilation), matching the loss-function masks used
# during finetuning in ``1-finetune_icon.py``.

# %%
landmark_tools = LandmarkTools()
labelmap_tools = LabelmapTools()
transform_tools = TransformTools()
segmenter = SegmentHeartSimpleware()


# %% [markdown]
# ## 4. Register and score every test subject under all methods
#
# All methods register each gated frame to the 70th-percentile gated frame as
# the fixed image.  The per-frame metric pipeline is shared: landmark error and
# warped-reference re-segmentation.

# %%
summary_rows: list[dict[str, object]] = []

for subject_id in test_subjects:
    seg_dir = segmentation_base_dir / subject_id
    source_dir = timepoint_base_dir / subject_id
    print(f"\nSubject {subject_id}")

    # --- Per-subject reference scan (fixed image for all methods) ---
    ref_file = next(source_dir.glob("*ref.nii.gz"))
    ref_stem = ref_file.name[:-7]
    fixed_image = itk.imread(str(ref_file), pixel_type=itk.F)
    fixed_mask_path = seg_dir / f"{ref_stem}_labelmap_mask.nii.gz"
    fixed_labelmap_path = seg_dir / f"{ref_stem}_labelmap.nii.gz"
    if fixed_mask_path.exists():
        fixed_mask = itk.imread(str(fixed_mask_path))
    else:
        fixed_labelmap = itk.imread(str(fixed_labelmap_path))
        fixed_mask = labelmap_tools.convert_labelmap_to_mask(
            fixed_labelmap, dilation_in_mm=3.0
        )
        itk.imwrite(fixed_mask, str(fixed_mask_path), compression=True)
    fixed_landmarks = landmark_tools.read_landmarks_3dslicer(
        str(seg_dir / f"{ref_stem}_landmark.mrk.json")
    )
    print(f"  Fixed: {ref_file.name}")

    # --- Gated frames (moving images, shared across all methods) ---
    image_files = [
        p
        for p in sorted(source_dir.glob("*.nii.gz"))
        if not any(t in p.name for t in exclude_tokens)
        and timepoint_re.search(p.name) is not None
    ]
    stems = [p.name[:-7] for p in image_files]
    timepoints = [timepoint_re.search(p.name).group("timepoint") for p in image_files]
    labelmap_files = [seg_dir / f"{s}_labelmap.nii.gz" for s in stems]
    mask_files = [seg_dir / f"{s}_labelmap_mask.nii.gz" for s in stems]
    landmark_files = [seg_dir / f"{s}_landmark.mrk.json" for s in stems]
    print(f"  {len(image_files)} gated frames")

    moving_images = [itk.imread(str(p), pixel_type=itk.F) for p in image_files]
    moving_labelmaps = [itk.imread(str(p)) for p in labelmap_files]
    moving_landmarks = [
        landmark_tools.read_landmarks_3dslicer(str(p)) for p in landmark_files
    ]
    moving_masks = []
    for i, p in enumerate(mask_files):
        if not p.exists():
            mask = labelmap_tools.convert_labelmap_to_mask(
                moving_labelmaps[i], dilation_in_mm=3.0
            )
            itk.imwrite(mask, str(p), compression=True)
            moving_masks.append(mask)
        else:
            moving_masks.append(itk.imread(str(p)))

    # --- Per-method registration and scoring ---
    for method_cfg in all_methods:
        method_name = str(method_cfg["name"])
        reg_method = str(method_cfg["reg_method"])
        weights_path = method_cfg["weights_path"]
        use_mask = bool(method_cfg["use_mask"])
        greedy_iters = method_cfg["greedy_iters"]

        print(f"  Method: {method_name}")
        registrar = RegisterTimeSeriesImages(
            registration_method=_build_registrar(
                reg_method, greedy_iters, icon_iterations, weights_path
            )
        )
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        if use_mask:
            registrar.set_fixed_mask(fixed_mask)

        result = registrar.register_time_series(
            moving_images=moving_images,
            moving_masks=moving_masks if use_mask else None,
            moving_labelmaps=moving_labelmaps if use_mask else None,
            register_reference=True,
            reference_frame=0,  # Not used
            prior_weight=0.0,  # Not used
        )

        method_dir = output_dir / method_name / subject_id
        method_dir.mkdir(parents=True, exist_ok=True)

        for index in range(len(image_files)):
            timepoint = timepoints[index]
            fixed_to_moving_transform = result["fixed_to_moving_transforms"][index]
            moving_to_fixed_transform = result["moving_to_fixed_transforms"][index]
            loss = float(result["losses"][index])

            itk.transformwrite(
                fixed_to_moving_transform,
                str(method_dir / f"{subject_id}_g{timepoint}_forward_tfm.hdf"),
                compression=True,
            )
            itk.transformwrite(
                moving_to_fixed_transform,
                str(method_dir / f"{subject_id}_g{timepoint}_inverse_tfm.hdf"),
                compression=True,
            )

            # Landmark error: warp gated landmarks into fixed-image space.
            timepoint_landmarks = moving_landmarks[index]
            shared = sorted(timepoint_landmarks.keys() & fixed_landmarks.keys())
            errors = []
            for name in shared:
                warped = moving_to_fixed_transform.TransformPoint(
                    timepoint_landmarks[name]
                )
                err = float(
                    np.linalg.norm(
                        np.asarray(warped, dtype=np.float64)
                        - np.asarray(fixed_landmarks[name], dtype=np.float64)
                    )
                )
                errors.append((name, err))

            with detail_file.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if fh.tell() == 0:
                    writer.writerow(
                        ["subject_id", "method", "timepoint", "name", "error_mm"]
                    )
                for name, err in errors:
                    writer.writerow([subject_id, method_name, timepoint, name, err])

            values = np.asarray([e for _, e in errors], dtype=np.float64)
            summary_rows.append(
                {
                    "subject_id": subject_id,
                    "method": method_name,
                    "reference_timepoint": ref_stem,
                    "timepoint": timepoint,
                    "loss": loss,
                    "n_landmarks": int(values.size),
                    "mean_mm": float(np.mean(values)) if values.size else "",
                    "median_mm": float(np.median(values)) if values.size else "",
                    "max_mm": float(np.max(values)) if values.size else "",
                }
            )

            # Warp fixed image back onto the gated frame's grid, re-segment, and
            # compare the resulting landmarks against the gated frame's own
            # precomputed landmarks.
            warped_ref = transform_tools.transform_image(
                fixed_image,
                moving_to_fixed_transform,
                moving_images[index],
                interpolation_method="linear",
            )
            itk.imwrite(
                warped_ref,
                str(method_dir / f"{subject_id}_g{timepoint}_warped_ref.mha"),
                compression=True,
            )

            seg_result = segmenter.segment(warped_ref)
            warped_ref_labelmap = seg_result["labelmap"]
            warped_ref_landmarks = segmenter.get_landmarks()

            itk.imwrite(
                warped_ref_labelmap,
                str(method_dir / f"{subject_id}_g{timepoint}_warped_ref_labelmap.mha"),
                compression=True,
            )
            landmark_tools.write_landmarks_3dslicer(
                warped_ref_landmarks,
                str(
                    method_dir
                    / f"{subject_id}_g{timepoint}_warped_ref_landmarks.mrk.json"
                ),
            )

            tp_landmarks = moving_landmarks[index]
            shared_warp = sorted(warped_ref_landmarks.keys() & tp_landmarks.keys())
            warp_errors = []
            for name in shared_warp:
                err = float(
                    np.linalg.norm(
                        np.asarray(warped_ref_landmarks[name], dtype=np.float64)
                        - np.asarray(tp_landmarks[name], dtype=np.float64)
                    )
                )
                warp_errors.append((name, err))
                print(f"    Warped-ref landmark {name}: {err:.3f} mm")

            with warped_ref_detail_file.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if fh.tell() == 0:
                    writer.writerow(
                        ["subject_id", "method", "timepoint", "name", "error_mm"]
                    )
                for name, err in warp_errors:
                    writer.writerow([subject_id, method_name, timepoint, name, err])

            warp_vals = np.asarray([e for _, e in warp_errors], dtype=np.float64)
            if warp_vals.size:
                print(
                    f"  Warped-ref landmark errors ({timepoint}): "
                    f"mean={float(np.mean(warp_vals)):.3f} mm  "
                    f"median={float(np.median(warp_vals)):.3f} mm  "
                    f"max={float(np.max(warp_vals)):.3f} mm"
                )

# %% [markdown]
# ## 5. Write the wide-form per-timepoint summary CSV

# %%
if not summary_rows:
    print("No summary rows to write")
else:
    with summary_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary: {summary_file}")
print(f"Wrote landmark details: {detail_file}")

# %% [markdown]
# ## 6. Per-method aggregate table over all test subjects

# %%
groups: dict[str, list[float]] = {}
with detail_file.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        groups.setdefault(row["method"], []).append(float(row["error_mm"]))

header = (
    f"{'Method':<18}{'N':>8}{'Mean (mm)':>12}"
    f"{'Median (mm)':>14}{'P95 (mm)':>12}{'Max (mm)':>12}"
)
print()
print("=" * len(header))
print(f"Landmark error summary ({len(test_subjects)} test subjects)")
print("=" * len(header))
print(header)
print("-" * len(header))
for method_cfg in all_methods:
    method_name = str(method_cfg["name"])
    arr = np.asarray(groups.get(method_name, []), dtype=np.float64)
    if arr.size == 0:
        print(f"{method_name:<18}{0:>8}{'':>12}{'':>14}{'':>12}{'':>12}")
        continue
    print(
        f"{method_name:<18}"
        f"{arr.size:>8}"
        f"{float(np.mean(arr)):>12.3f}"
        f"{float(np.median(arr)):>14.3f}"
        f"{float(np.percentile(arr, 95)):>12.3f}"
        f"{float(np.max(arr)):>12.3f}"
    )
print("=" * len(header))

# %% [markdown]
# ## 7. Per-method aggregate table: warped-reference landmark errors
#
# Compares landmarks extracted from the reference image warped back to each
# time-point's grid (via ``moving_to_fixed_transform``) against that time-point's own
# precomputed landmarks.  Both sets are in the moving (time-point) image space,
# so errors are Euclidean distances without any additional transform.

# %%
if warped_ref_detail_file.exists():
    warp_groups: dict[str, list[float]] = {}
    with warped_ref_detail_file.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            warp_groups.setdefault(row["method"], []).append(float(row["error_mm"]))

    warp_header = (
        f"{'Method':<18}{'N':>8}{'Mean (mm)':>12}"
        f"{'Median (mm)':>14}{'P95 (mm)':>12}{'Max (mm)':>12}"
    )
    print()
    print("=" * len(warp_header))
    print(
        f"Warped-reference landmark error summary ({len(test_subjects)} test subjects)"
    )
    print("=" * len(warp_header))
    print(warp_header)
    print("-" * len(warp_header))
    for method_cfg in all_methods:
        method_name = str(method_cfg["name"])
        arr = np.asarray(warp_groups.get(method_name, []), dtype=np.float64)
        if arr.size == 0:
            print(f"{method_name:<18}{0:>8}{'':>12}{'':>14}{'':>12}{'':>12}")
            continue
        print(
            f"{method_name:<18}"
            f"{arr.size:>8}"
            f"{float(np.mean(arr)):>12.3f}"
            f"{float(np.median(arr)):>14.3f}"
            f"{float(np.percentile(arr, 95)):>12.3f}"
            f"{float(np.max(arr)):>12.3f}"
        )
    print("=" * len(warp_header))
else:
    print(
        "No warped-reference landmark errors written (all frames were reference frames)."
    )
