# %% [markdown]
# # Segment and Landmark Duke 4D Gated CT Data
#
# Runs Simpleware ASCardio segmentation on each gated time-point image and
# stores the labelmap plus extracted landmarks for later registration
# accuracy experiments.
#
# Each `gated_nii/<subject_id>/` directory contains one patient's time-point
# images. The script maps each `ref_images/pm00*.nii.gz` file to the matching
# `gated_nii/<subject_id>/` directory by the first six filename characters.
#
# Segmentation labelmaps and landmarks are written to:
# `d:/MONAI-Physio/duke_data/simple_ascardio/<subject_id>/`
#
# Output files follow the input stem:
# - `<stem>_labelmap.nii.gz`
# - `<stem>_landmark.mrk.json`
#

# %%
import os
from pathlib import Path

import itk

from monai_physio import SegmentHeartSimpleware
from monai_physio.landmark_tools import LandmarkTools

# %%
# Discover data (mirrors recon_4d.py)
########################################################

ref_data_dir = "d:/MONAI-Physio/duke_data/ref_images"
src_data_dir_base = "d:/MONAI-Physio/duke_data/gated_nii"
segmentation_dir_base = "d:/MONAI-Physio/duke_data/simple_ascardio"

ref_files = [
    os.path.join(ref_data_dir, f)
    for f in sorted(os.listdir(ref_data_dir))
    if f.startswith("pm00") and f.endswith(".nii.gz")
]

print(f"Found {len(ref_files)} reference images")

src_data_dirs = []
src_data_files = []
for ref_file in ref_files:
    src_dir = os.path.join(src_data_dir_base, os.path.basename(ref_file)[:6])
    src_data_dirs.append(src_dir)

    file_list = sorted(os.listdir(src_dir))
    valid_file_list = [f for f in file_list if "nop" not in f and f.endswith(".nii.gz")]
    src_data_files.append(valid_file_list)

print(f"Found {len(src_data_dirs)} source data directories")
for d, fs in zip(src_data_dirs, src_data_files, strict=True):
    print(f"  {d}: {len(fs)} files")

# %%
# Function to segment images and save labelmaps and landmarks
########################################################


def segment_images(
    src_data_dirs: list[str],
    src_data_files: list[list[str]],
) -> dict[str, str]:
    """Segment each image with SegmentHeartSimpleware and save labelmaps.

    Skips images whose labelmap file already exists. Returns a mapping from
    image path to labelmap path for all images.

    Args:
        src_data_dirs: List of per-patient source directories.
        src_data_files: List of per-patient filename lists.

    Returns:
        Dict mapping absolute image path -> absolute labelmap path.
    """
    segmenter = SegmentHeartSimpleware()
    image_to_labelmap: dict[str, str] = {}

    for src_dir, files in zip(src_data_dirs, src_data_files, strict=True):
        print(f"Segmenting {src_dir}...")
        subject_id = os.path.basename(src_dir)
        labelmap_dir = Path(segmentation_dir_base) / subject_id
        labelmap_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            print(f"  Segmenting {f}...")
            labelmap_path = labelmap_dir / f.replace(".nii.gz", "_labelmap.nii.gz")
            landmark_path = labelmap_dir / f.replace(".nii.gz", "_landmark.mrk.json")

            if not os.path.exists(labelmap_path) or not os.path.exists(landmark_path):
                image_path = os.path.join(src_dir, f)
                input_image = itk.imread(image_path, pixel_type=itk.F)
                results = segmenter.segment(input_image)
                labelmap = results["labelmap"]
                itk.imwrite(labelmap, str(labelmap_path), compression=True)

                landmarks = segmenter.get_landmarks()
                LandmarkTools().write_landmarks_3dslicer(landmarks, landmark_path)

            image_to_labelmap[os.path.join(src_dir, f)] = str(labelmap_path)

    return image_to_labelmap


# %%
# Segment each image and save labelmaps
########################################################

print("\nSegmenting images...")
image_to_labelmap = segment_images(src_data_dirs, src_data_files)
print(f"Segmentation complete. {len(image_to_labelmap)} labelmaps available.\n")

# %%
