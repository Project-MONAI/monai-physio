"""
Tutorial 1 (Slicer Heart): Heart-Gated CT to Animated USD

Purpose
-------
Convert a 4D cardiac CT scan (multiple gated time frames) into an animated USD
model suitable for visualization in NVIDIA Omniverse. The workflow segments the
heart and surrounding anatomy from a reference frame, registers all other frames
to that reference using deep learning or classical registration, and assembles
the resulting time-varying surface meshes into a single USD file with anatomical
materials applied.

Inputs
------
- A set of 3D CT volumes (``slice_???.mha``) representing successive cardiac
  phases.
  Expected location: ``data/Slicer-Heart-CT/slice_???.mha``
  (``data/test/slicer_heart_small/`` under ``ProcessTests.running_as_test``).
- One of those phases (``reference_image``) is the segmentation and
  registration reference.

Outputs
-------
Under ``tutorials/output/tutorial_01_convert_to_usd_slicer_heart_4d_ct/``:

- The animated USD named after ``usd_project_name`` (``cardiac_model``); the
  workflow returns the dynamic variant when one was produced, otherwise the
  combined one.
- ``slice_<n>_all_registered.mha`` - each phase registered to the reference,
  and ``reference_labelmap.mha`` - the reference phase's segmentation
- Screenshots (PNG) for documentation and regression testing:
  - ``slice_<n>_registered_test.png`` - the phase 70% through the series,
    registered
  - ``cardiac_model_test.png`` - the assembled contours

Strengths
---------
- Single call (``WorkflowConvertImageToUSD.process()``) runs the full pipeline.
- Registers on the CPU with ``RegisterImagesGreedy``; no GPU needed for this stage.
- Automatically detects contrast enhancement and adjusts segmentation thresholds.
- Output is Omniverse-ready with anatomical materials (ProcessUSDAnatomy).

Weaknesses / Limitations
------------------------
- Segmentation quality depends on TotalSegmentator's training distribution;
  unusual pathologies or pediatric anatomy may degrade results.
- Large 4D datasets (>20 phases, high resolution) can require 32 GB+ RAM.

Classes Used
------------
- WorkflowConvertImageToUSD (workflow_convert_image_to_usd.py):
    Orchestrates the full pipeline: CT phases -> segmentation -> registration
    -> contour extraction -> USD export.
- SegmentChestTotalSegmentatorWithContrast
  (segment_chest_total_segmentator_with_contrast.py):
    Deep-learning anatomy segmentation (``SLICER_HEART_4D_CT.segmenter``).
- RegisterImagesGreedy (register_images_greedy.py):
    Frame-to-frame image registration (``SLICER_HEART_4D_CT.registrar``).
- ProcessSurfaceMesh (process_surface_mesh.py):
    Extracts and transforms surface meshes from segmentation masks (used internally).
- ProcessUSDAnatomy (process_usd_anatomy.py):
    Applies clinical material colours to USD prims (used internally).

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: Slicer-Heart-CT - https://github.com/SlicerHeart/SlicerHeart
This script expects the ``slice_???.mha`` phase volumes to already exist under
``data/Slicer-Heart-CT/``; ``monai-physio-download-data Slicer-Heart-CT``
downloads the 4D sequence and splits it into those volumes.
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk
from parameters_slicer_heart_4d_ct import SLICER_HEART_4D_CT

from monai_physio import (
    ProcessTests,
    WorkflowConvertImageToUSD,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used by TotalSegmentator inside WorkflowConvertImageToUSD)
# spawns a multiprocessing.Pool. On Windows the spawn start method re-imports
# this script in each child; without the __name__ == "__main__" guard around
# the top-level work, that re-import fires workflow.process() again and
# Python's spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # Data directory specification

    class_name = "tutorial_01_convert_to_usd_slicer_heart_4d_ct"

    test_mode = ProcessTests.running_as_test()

    output_dir = (
        SLICER_HEART_4D_CT.output_directory(test_mode)
        / "tutorial_01_convert_to_usd_slicer_heart_4d_ct"
    )

    data_dir = SLICER_HEART_4D_CT.input_directory(test_mode)

    if test_mode:
        frame_files = sorted(data_dir.glob("slice_???.mha"))[0:2]
    else:
        frame_files = sorted(data_dir.glob("slice_???.mha"))

    log_level = logging.INFO

    registration_method = SLICER_HEART_4D_CT.registrar(test_mode, log_level=log_level)
    segmentation_method = SLICER_HEART_4D_CT.segmenter(test_mode, log_level=log_level)

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    input_filenames = [str(path) for path in frame_files]
    if not input_filenames:
        raise FileNotFoundError(
            "Slicer-Heart-CT data not found. Checked:\n"
            + f"  - {data_dir}"
            + "\n"
            + "See data/README.md for download instructions."
        )

    time_series_images = [itk.imread(str(path)) for path in input_filenames]
    # Frame 7 of the full series; test mode loads fewer frames than that.
    reference_image = time_series_images[min(7, len(time_series_images) - 1)]

    print("Number of time-series images:", len(time_series_images))

    # Workflow initialization

    workflow = WorkflowConvertImageToUSD(
        time_series_images=time_series_images,
        reference_image=reference_image,
        output_directory=str(output_dir),
        usd_project_name="cardiac_model",
        registration_method=registration_method,
        segmentation_method=segmentation_method,
        surface_reduction_rate=SLICER_HEART_4D_CT.surface_reduction_rate,
        distance_sigma_mm=SLICER_HEART_4D_CT.surface_distance_sigma_mm,
        log_level=log_level,
        save_assets=True,
        save_transforms=False,
    )

    # Workflow execution
    workflow_results = workflow.process()

    # if dynamic_labelmap_ids is not None, there are two USD files
    if len(workflow.dynamic_labelmap_ids) > 0:
        usd_file = output_dir / workflow_results["dynamic"]
    else:
        usd_file = output_dir / workflow_results["all"]

    # Result saving
    tt = ProcessTests(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []

    test_image_num = int(0.7 * len(input_filenames))
    test_image_path = output_dir / f"slice_{test_image_num:03d}_all_registered.mha"
    if test_image_path.exists():
        test_image = itk.imread(str(test_image_path))
        screenshots.append(
            tt.save_screenshot_image_slice(
                test_image,
                f"slice_{test_image_num:03d}_registered_test.png",
                axis=0,
                slice_fraction=0.5,
                colormap="gray",
                vmin=-200,
                vmax=600,
            )
        )

    if usd_file.exists():
        screenshots.append(
            tt.save_screenshot_openusd(
                usd_file,
                "cardiac_model_test.png",
            )
        )

    tutorial_results = {"usd_file": str(usd_file), "screenshots": screenshots}
