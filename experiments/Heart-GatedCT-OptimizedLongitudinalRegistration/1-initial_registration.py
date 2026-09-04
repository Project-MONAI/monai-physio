# %% [markdown]
# Initial registration: compare ANTS vs Greedy vs ICON on the Duke gated CT cohort
#
#   * :class:`RegisterImagesANTS` (CPU, SyN deformable)
#   * :class:`RegisterImagesGreedy` (CPU, deformable)
#   * :class:`RegisterImagesICON` (GPU, uniGradICON deformable)
#
# %%
import csv
import shutil
import time
from pathlib import Path
from typing import Optional

import itk
import numpy as np

from monai_physio.labelmap_tools import LabelmapTools
from monai_physio.landmark_tools import LandmarkTools
from monai_physio.register_images_ants import RegisterImagesANTS
from monai_physio.register_images_greedy import RegisterImagesGreedy
from monai_physio.register_images_icon import RegisterImagesICON
from monai_physio.transform_tools import TransformTools

# %%
ref_data_dir = Path("d:/MONAI-Physio/duke_data/ref_images")
src_data_dir_base = Path("d:/MONAI-Physio/duke_data/gated_nii")
segmentation_dir_base = Path("d:/MONAI-Physio/duke_data/simple_ascardio")

use_mask_list = [False, False]
use_labelmap_list = [False, True]

# ICON only
use_mass_list = [False, False]

methods_list = [
    ["Greedy"],
    ["Greedy"],
]
number_of_iterations_ANTS_list = [
    [40, 20, 10],
    [40, 20, 10],
]
number_of_iterations_greedy_list = [[80, 40, 5], [40, 40, 10]]
number_of_iterations_ICON_list = [100, 100]

exclude_tokens = ["nop"]
ref_suffix = "_ref"
icon_weights_path: Optional[Path] = None
mask_dilation_mm = 3.0
use_crop = False
fixed_image_resolution_mm = 0.0

debug_subjects = []  # ["pm0002", "pm0003", "pm0004"]

labelmap_tools = LabelmapTools()
landmark_tools = LandmarkTools()
transform_tools = TransformTools()

# %%
ref_files = sorted(
    p
    for p in ref_data_dir.iterdir()
    if p.name.startswith("pm00") and p.suffixes[-2:] == [".nii", ".gz"]
)
all_patient_ids = [p.name[:6] for p in ref_files]
print(f"Found {len(all_patient_ids)} patients under {ref_data_dir}")

if debug_subjects:
    cohort = [pid for pid in all_patient_ids if pid in debug_subjects]
    print(
        f"DEBUG: restricting cohort to {debug_subjects} -> "
        f"{len(cohort)} matching patients"
    )
else:
    cohort = all_patient_ids


def per_label_dice(
    fixed_labelmap: itk.Image, warped_labelmap: itk.Image
) -> dict[int, float]:
    """Return ``{label_id: Dice}`` for every positive label present in
    either the fixed or the warped labelmap.

    Arrays come back from :func:`itk.array_from_image` in shape
    ``(Z, Y, X)`` (numpy reverses ITK's index order); we compare element-wise
    so the axis convention does not matter as long as both labelmaps live
    on the same reference grid (guaranteed because ``warped_labelmap`` was
    resampled with ``fixed_labelmap`` as the reference image).
    """
    fixed_array = itk.array_from_image(fixed_labelmap)
    warped_array = itk.array_from_image(warped_labelmap)
    labels = sorted(
        {int(v) for v in np.unique(fixed_array)}
        | {int(v) for v in np.unique(warped_array)}
    )
    labels = [label for label in labels if label > 0]

    dice_by_label: dict[int, float] = {}
    for label in labels:
        a = fixed_array == label
        b = warped_array == label
        denom = int(a.sum()) + int(b.sum())
        if denom == 0:
            continue
        intersection = int(np.logical_and(a, b).sum())
        dice_by_label[label] = 2.0 * intersection / denom
    return dice_by_label


def warp_landmarks(
    inverse_transform: itk.Transform,
    moving_landmarks: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    """Warp every moving landmark into reference space.

    Point/landmark warping uses ``inverse_transform`` -- the moving-space ->
    fixed-space point map -- which is the opposite of the transform used to
    warp the moving image onto the fixed grid (images pull back; points push
    forward). Returns a ``{label: (x, y, z)}`` dict in LPS. See
    docs/developer/transform_conventions.
    """
    new_landmarks = {}
    for name, point in moving_landmarks.items():
        new_point = inverse_transform.TransformPoint(np.array(point))
        new_landmarks[name] = tuple(np.array(new_point).tolist())
    return new_landmarks


def landmark_rms_errors(
    warped_landmarks: dict[str, tuple[float, float, float]],
    fixed_landmarks: dict[str, tuple[float, float, float]],
) -> list[tuple[str, float]]:
    """Return per-landmark RMS Euclidean error in mm between the
    reference-space ``warped_landmarks`` and the matching reference
    landmarks, in sorted-name order.
    """
    errors: list[tuple[str, float]] = []
    for name in fixed_landmarks.keys():
        if name not in warped_landmarks:
            errors.append((name, float("nan")))
            continue
        diff = 0
        for i in range(3):
            diff += (warped_landmarks[name][i] - fixed_landmarks[name][i]) ** 2
        errors.append((name, float(np.sqrt(diff))))
        print(f"Landmark {name} RMS error: {errors[-1][1]:.4f} mm")
    return errors


def load_or_derive_mask(labelmap: itk.Image, mask_path: Path) -> itk.Image:
    """Return the cached ``<stem>_labelmap_mask.nii.gz`` next to the
    labelmap, or derive it via
    :meth:`LabelmapTools.convert_labelmap_to_mask` (threshold ``>0`` plus
    3 mm physical-radius dilation) and write it out so subsequent runs and
    the ICON eval reuse the same mask.
    """
    if mask_path.exists():
        return itk.imread(str(mask_path))
    mask = labelmap_tools.convert_labelmap_to_mask(
        labelmap,
        dilation_in_mm=mask_dilation_mm,
        exclude_labels=[1, 2, 3, 4],
        # Interior chambers of the heart: LV, RV, LA, RA
    )
    itk.imwrite(mask, str(mask_path), compression=True)
    return mask


def crop_image_to_mask(
    image: itk.Image,
    mask: Optional[itk.Image] = None,
    labelmap: Optional[itk.Image] = None,
    margin_fraction: float = 0.1,
) -> dict[str, itk.Image]:
    if mask is None:
        mask_arr = itk.array_from_image(image)
        print("No mask provided, using image as mask")
    else:
        mask_arr = itk.array_from_image(mask)
    shape_z, shape_y, shape_x = mask_arr.shape
    bounding_box = np.where(mask_arr > 0)
    if not mask_arr.any() or bounding_box[0].size == 0:
        min_x, max_x = 0, shape_x - 1
        min_y, max_y = 0, shape_y - 1
        min_z, max_z = 0, shape_z - 1
    else:
        min_x = int(np.min(bounding_box[2]))
        max_x = int(np.max(bounding_box[2]))
        min_y = int(np.min(bounding_box[1]))
        max_y = int(np.max(bounding_box[1]))
        min_z = int(np.min(bounding_box[0]))
        max_z = int(np.max(bounding_box[0]))
    margin_x = int((max_x - min_x) * margin_fraction)
    margin_y = int((max_y - min_y) * margin_fraction)
    margin_z = int((max_z - min_z) * margin_fraction)
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y
    min_z -= margin_z
    max_z += margin_z
    min_x = max(0, min(min_x, shape_x - 1))
    min_y = max(0, min(min_y, shape_y - 1))
    min_z = max(0, min(min_z, shape_z - 1))
    max_x = max(0, min(max_x, shape_x - 1))
    max_y = max(0, min(max_y, shape_y - 1))
    max_z = max(0, min(max_z, shape_z - 1))
    print(f"array shape: {mask_arr.shape}")
    print(
        f"min_x: {min_x}, max_x: {max_x}, min_y: {min_y}, max_y: {max_y}, min_z: {min_z}, max_z: {max_z}"
    )
    new_image_arr = itk.array_from_image(image)
    new_image_arr = new_image_arr[
        min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1
    ]
    new_origin = image.TransformIndexToPhysicalPoint(
        [int(min_x), int(min_y), int(min_z)]
    )
    new_image = itk.image_from_array(new_image_arr)
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetDirection(image.GetDirection())
    new_image.SetOrigin(new_origin)

    if labelmap is not None:
        new_labelmap_arr = itk.array_from_image(labelmap)
        new_labelmap_arr = new_labelmap_arr[
            min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1
        ]
        new_labelmap = itk.image_from_array(new_labelmap_arr)
        new_labelmap.CopyInformation(new_image)
    else:
        new_labelmap = None

    if mask is not None:
        new_mask_arr = itk.array_from_image(mask)
        new_mask_arr = new_mask_arr[
            min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1
        ]
        new_mask = itk.image_from_array(new_mask_arr)
        new_mask.CopyInformation(new_image)
    else:
        new_mask = None

    return {
        "image": new_image,
        "labelmap": new_labelmap,
        "mask": new_mask,
    }


# %%
_HERE = Path(__file__).parent
for subject_index, subject_id in enumerate(cohort):
    print(f"\n=== Subject {subject_index + 1}/{len(cohort)}: {subject_id} ===")
    src_dir = src_data_dir_base / subject_id
    seg_dir = segmentation_dir_base / subject_id

    ref_file = next((p for p in ref_files if p.name.startswith(subject_id)), None)
    ref_stem = ref_file.name[:-7]
    ref_labelmap_path = seg_dir / f"{ref_stem}_labelmap.nii.gz"
    ref_mask_path = seg_dir / f"{ref_stem}_labelmap_mask.nii.gz"
    ref_landmark_path = seg_dir / f"{ref_stem}_landmark.mrk.json"

    fixed_output_dir = _HERE / "fixed"
    fixed_output_dir.mkdir(parents=True, exist_ok=True)

    fixed_image = itk.imread(str(ref_file), pixel_type=itk.F)

    fixed_labelmap = None
    if ref_labelmap_path.exists():
        fixed_labelmap = itk.imread(str(ref_labelmap_path))

    fixed_mask = None
    if ref_mask_path.exists():
        fixed_mask = load_or_derive_mask(fixed_labelmap, ref_mask_path)

    fixed_landmarks = None
    if ref_landmark_path.exists():
        fixed_landmarks = landmark_tools.read_landmarks_3dslicer(ref_landmark_path)

    if use_crop:
        cropped = crop_image_to_mask(
            fixed_image, mask=fixed_mask, labelmap=fixed_labelmap
        )
        fixed_image = cropped["image"]
        fixed_labelmap = cropped["labelmap"]
        fixed_mask = cropped["mask"]

    if fixed_image_resolution_mm > 0.0:
        fixed_image_size = fixed_image.GetLargestPossibleRegion().GetSize()
        fixed_image_size[0] = int(
            fixed_image_size[0]
            * fixed_image.GetSpacing()[0]
            / fixed_image_resolution_mm
        )
        fixed_image_size[1] = int(
            fixed_image_size[1]
            * fixed_image.GetSpacing()[1]
            / fixed_image_resolution_mm
        )
        fixed_image_size[2] = int(
            fixed_image_size[2]
            * fixed_image.GetSpacing()[2]
            / fixed_image_resolution_mm
        )
        fixed_image = itk.resample_image_filter(
            fixed_image,
            output_direction=fixed_image.GetDirection(),
            output_origin=fixed_image.GetOrigin(),
            size=fixed_image_size,
            output_spacing=[
                fixed_image_resolution_mm,
                fixed_image_resolution_mm,
                fixed_image_resolution_mm,
            ],
            default_pixel_value=-1000,
        )
        if fixed_labelmap is not None:
            fixed_labelmap = itk.resample_image_filter(
                fixed_labelmap,
                output_parameters_from_image=fixed_image,
                default_pixel_value=0,
                interpolator=itk.NearestNeighborInterpolateImageFunction.New(
                    fixed_labelmap
                ),
            )
        if fixed_mask is not None:
            fixed_mask = itk.resample_image_filter(
                fixed_mask,
                output_parameters_from_image=fixed_image,
                default_pixel_value=0,
                interpolator=itk.NearestNeighborInterpolateImageFunction.New(
                    fixed_mask
                ),
            )

    print(f"Writing reference image to {f'{subject_id}_ref.nii.gz'}")
    itk.imwrite(
        fixed_image,
        str(fixed_output_dir / f"{subject_id}_ref.nii.gz"),
        compression=True,
    )
    if fixed_labelmap is not None:
        print(f"Writing reference labelmap to {f'{subject_id}_ref_labelmap.nii.gz'}")
        itk.imwrite(
            fixed_labelmap,
            str(fixed_output_dir / f"{subject_id}_ref_labelmap.nii.gz"),
            compression=True,
        )
    if fixed_mask is not None:
        print(f"Writing reference mask to {f'{subject_id}_ref_mask.nii.gz'}")
        itk.imwrite(
            fixed_mask,
            str(fixed_output_dir / f"{subject_id}_ref_mask.nii.gz"),
            compression=True,
        )
    if fixed_landmarks is not None:
        print(
            f"Writing reference landmarks to {f'{subject_id}_ref_landmarks.mrk.json'}"
        )
        landmark_tools.write_landmarks_3dslicer(
            fixed_landmarks,
            str(fixed_output_dir / f"{subject_id}_ref_landmarks.mrk.json"),
        )

    gated_files = sorted(
        p
        for p in src_dir.glob("*.nii.gz")
        if not any(token in p.name for token in exclude_tokens)
        and not p.name.endswith(f"{ref_suffix}.nii.gz")
    )

    print(f"Found {len(gated_files)} gated images under {src_dir}")

    for image_index, image_path in enumerate(gated_files):
        stem = image_path.name[:-7]

        print(f"\n\n *** Processing {stem} ***\n\n")

        labelmap_path = seg_dir / f"{stem}_labelmap.nii.gz"
        mask_path = seg_dir / f"{stem}_labelmap_mask.nii.gz"
        landmark_path = seg_dir / f"{stem}_landmark.mrk.json"

        moving_image_name = str(image_path)
        moving_image = itk.imread(moving_image_name, pixel_type=itk.F)
        moving_labelmap = None
        if fixed_labelmap is not None and labelmap_path.exists():
            moving_labelmap = itk.imread(str(labelmap_path))
        moving_mask = None
        if fixed_mask is not None and mask_path.exists():
            moving_mask = load_or_derive_mask(moving_labelmap, mask_path)
        moving_landmarks = None
        if fixed_landmarks is not None and landmark_path.exists():
            moving_landmarks = landmark_tools.read_landmarks_3dslicer(landmark_path)

        if use_crop:
            cropped = crop_image_to_mask(
                moving_image, mask=moving_mask, labelmap=moving_labelmap
            )
            moving_image = cropped["image"]
            moving_labelmap = cropped["labelmap"]
            moving_mask = cropped["mask"]

        for cond_index in range(len(use_mask_list)):
            methods = methods_list[cond_index]

            use_mask = use_mask_list[cond_index]
            use_labelmap = use_labelmap_list[cond_index]
            use_mass = use_mass_list[cond_index]

            number_of_iterations_ANTS = number_of_iterations_ANTS_list[cond_index]
            number_of_iterations_greedy = number_of_iterations_greedy_list[cond_index]
            number_of_iterations_ICON = number_of_iterations_ICON_list[cond_index]

            cond = "_"
            if use_mask:
                cond += "m"
            if use_labelmap:
                cond += "l"
            if use_mass:
                cond += "p"
            if use_crop:
                cond += "c"
            if cond == "_":
                cond += "raw"

            print(
                f"\n\n ***** {cond_index + 1}/{len(use_mask_list)}: results{cond} *****\n\n"
            )

            output_dir = _HERE / f"results{cond}"
            detail_landmarks_file = output_dir / "registration_landmarks_init.csv"
            detail_dice_file = output_dir / "registration_dice_init.csv"
            if subject_index == 0 and image_index == 0 and cond_index == 0:
                if output_dir.exists():
                    shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for method_index, method_name in enumerate(methods):
                print(f"\n\n  --- {method_name} --- \n\n")
                if method_name == "ANTS":
                    reg = RegisterImagesANTS()
                    reg.set_number_of_iterations(number_of_iterations_ANTS)
                    num_iters_str = ".".join(str(n) for n in number_of_iterations_ANTS)
                    reg.set_transform_type("Deformable")
                    # NCC ("CC") beats MeanSquares for same-modality CT registration.
                    if use_labelmap:
                        reg.set_metric("MeanSquares")
                    else:
                        reg.set_metric("CC")
                elif method_name == "Greedy":
                    reg = RegisterImagesGreedy()
                    reg.set_number_of_iterations(number_of_iterations_greedy)
                    print(f"Number of iterations: {number_of_iterations_greedy}")
                    num_iters_str = ".".join(
                        str(n) for n in number_of_iterations_greedy
                    )
                    reg.set_transform_type("Deformable")
                    # NCC ("CC") beats MeanSquares for same-modality CT registration.
                    if use_labelmap:
                        reg.set_metric("MeanSquares")
                    else:
                        reg.set_metric("CC")
                else:  # ICON: GPU deep-learning deformable registration.
                    reg = RegisterImagesICON()
                    reg.set_number_of_iterations(number_of_iterations_ICON)
                    num_iters_str = str(number_of_iterations_ICON)
                    reg.set_multi_modality(False)
                    reg.set_mass_preservation(use_mass)
                    if icon_weights_path is not None:
                        reg.set_weights_path(str(icon_weights_path))
                reg.set_modality("ct")
                reg.set_mask_dilation(0)  # Already dilated
                reg.set_fixed_image(fixed_image)
                if use_mask:
                    reg.set_fixed_mask(fixed_mask)
                if use_labelmap:
                    reg.set_fixed_labelmap(fixed_labelmap)

                method_dir_name = str(method_name.lower()) + "_" + num_iters_str
                method_dir = output_dir / method_dir_name / subject_id
                print(f"Method directory: {method_dir}")
                method_dir.mkdir(parents=True, exist_ok=True)
                print(f"Registering {stem} with {method_name}...")
                time_start = time.perf_counter()
                reg_result = reg.register(
                    moving_image=moving_image,
                    moving_mask=moving_mask if use_mask else None,
                    moving_labelmap=moving_labelmap if use_labelmap else None,
                )
                time_elapsed = time.perf_counter() - time_start
                print(f"   ...finished registration in {time_elapsed:.1f}s")

                forward_transform = reg_result["forward_transform"]
                inverse_transform = reg_result["inverse_transform"]
                loss = float(reg_result["loss"])

                print(f"Writing results to {method_dir / f'{stem}_init_*.*'}")

                itk.transformwrite(
                    forward_transform,
                    str(method_dir / f"{stem}_init_fwd.hdf"),
                    compression=True,
                )
                itk.transformwrite(
                    inverse_transform,
                    str(method_dir / f"{stem}_init_inv.hdf"),
                    compression=True,
                )

                warped_image = transform_tools.transform_image(
                    moving_image,
                    forward_transform,
                    fixed_image,
                    interpolation_method="linear",
                )
                itk.imwrite(
                    warped_image,
                    str(method_dir / f"{stem}_init.mha"),
                    compression=True,
                )

                average_dice = float("nan")
                if fixed_labelmap is not None and moving_labelmap is not None:
                    warped_labelmap = transform_tools.transform_image(
                        moving_labelmap,
                        forward_transform,
                        fixed_labelmap,
                        interpolation_method="nearest",
                    )
                    itk.imwrite(
                        warped_labelmap,
                        str(method_dir / f"{stem}_init_labelmap.mha"),
                        compression=True,
                    )
                    dice_by_label = per_label_dice(fixed_labelmap, warped_labelmap)
                    with detail_dice_file.open("a", newline="", encoding="utf-8") as fh:
                        writer = csv.writer(fh)
                        if fh.tell() == 0:
                            writer.writerow(
                                ["subject_id", "method", "stem", "label", "dice"]
                            )
                        for label, dice in dice_by_label.items():
                            writer.writerow(
                                [
                                    subject_id,
                                    method_name + "_" + num_iters_str,
                                    stem,
                                    label,
                                    dice,
                                ]
                            )
                    if dice_by_label:
                        average_dice = float(np.mean(list(dice_by_label.values())))

                if fixed_mask is not None and moving_mask is not None:
                    warped_mask = transform_tools.transform_image(
                        moving_mask,
                        forward_transform,
                        fixed_mask,
                        interpolation_method="nearest",
                    )
                    itk.imwrite(
                        warped_mask,
                        str(method_dir / f"{stem}_init_mask.mha"),
                        compression=True,
                    )

                average_rms_errors = float("nan")
                if fixed_landmarks is not None and moving_landmarks is not None:
                    # Landmarks live in LPS world space, unaffected by cropping, so
                    # the uncropped moving_landmarks are warped here.
                    warped_landmarks = warp_landmarks(
                        inverse_transform, moving_landmarks
                    )
                    landmark_tools.write_landmarks_3dslicer(
                        warped_landmarks,
                        str(method_dir / f"{stem}_init_landmarks.mrk.json"),
                    )
                    rms_errors = landmark_rms_errors(warped_landmarks, fixed_landmarks)
                    with detail_landmarks_file.open(
                        "a", newline="", encoding="utf-8"
                    ) as fh:
                        writer = csv.writer(fh)
                        if fh.tell() == 0:
                            writer.writerow(
                                [
                                    "subject_id",
                                    "method",
                                    "stem",
                                    "name",
                                    "rms_err_mm",
                                ]
                            )
                        for name, error in rms_errors:
                            writer.writerow(
                                [
                                    subject_id,
                                    method_name + "_" + num_iters_str,
                                    stem,
                                    name,
                                    error,
                                ]
                            )
                    rms_values = np.asarray(
                        [e for _, e in rms_errors], dtype=np.float64
                    )
                    if rms_values.size and not np.all(np.isnan(rms_values)):
                        average_rms_errors = float(np.nanmean(rms_values))

                print(
                    f"Method: {method_name}_{num_iters_str}, Subject: {subject_id}, "
                    f"Timepoint: {stem}, time: {time_elapsed:.1f}s, "
                    f"loss: {loss:.4f}, Dice: {average_dice:.4f}, "
                    f"RMS(mm): {average_rms_errors:.4f}"
                )

# %%
