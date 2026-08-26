"""
Tutorial 1: Heart-Gated CT to Animated USD

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
- A 4D NRRD sequence file (``*.seq.nrrd``) **or** a list of 3D CT volumes
  (``*.mha`` / ``*.nrrd``) representing successive cardiac phases.
  Expected location: ``data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd``
- Optional: a reference frame image to fix the cardiac phase used as the
  segmentation source.

Outputs (under ``tutorials/output/tutorial_01_heart/``)
------------------------------------------------------
- The animated USD named after ``usd_project_name`` (``cardiac_model``); the
  workflow returns the dynamic variant when one was produced, otherwise the
  combined one.
- Screenshots (PNG) for documentation and regression testing:
  - ``slice_<n>_registered_test.png`` - each registered phase
  - ``slice_<n>_labelmap_test.png`` - each phase's labelmap
  - ``cardiac_model_test.png`` - the assembled contours

Strengths
---------
- Single call (``WorkflowConvertImageToUSD.process()``) runs the full pipeline.
- Registers on the CPU with ``RegisterImagesGreedy``; no GPU needed for this stage.
- Automatically detects contrast enhancement and adjusts segmentation thresholds.
- Output is Omniverse-ready with anatomical materials (USDAnatomyTools).

Weaknesses / Limitations
------------------------
- Segmentation quality depends on TotalSegmentator's training distribution;
  unusual pathologies or pediatric anatomy may degrade results.
- Large 4D datasets (>20 phases, high resolution) can require 32 GB+ RAM.

Classes Used
------------
- WorkflowConvertImageToUSD (workflow_convert_image_to_usd.py):
    Orchestrates the full pipeline: 4D NRRD -> segmentation -> registration ->
    contour extraction -> USD export.
- SegmentChestTotalSegmentator (segment_chest_total_segmentator.py):
    Deep-learning segmentation of 117 anatomical structures (used internally).
- RegisterImagesGreedy (register_images_greedy.py):
    Frame-to-frame image registration (used internally).
- ContourTools (contour_tools.py):
    Extracts and transforms surface meshes from segmentation masks (used internally).
- USDAnatomyTools (usd_anatomy_tools.py):
    Applies clinical material colours to USD prims (used internally).

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: Slicer-Heart-CT - https://github.com/SlicerHeart/SlicerHeart
This script expects the data to already exist at
``data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd``. Run the repository data
download notebook or download the file manually before running this tutorial.
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk
from parameters_heart_ct_kcl import HEART_CT_KCL

from physiotwin4d import (
    RegisterImagesGreedy,
    SegmentChestTotalSegmentatorWithContrast,
    TestTools,
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

    class_name = "tutorial_01_heart_gated_ct_to_usd"

    test_mode = TestTools.running_as_test()

    output_dir = HEART_CT_KCL.output_directory(test_mode) / "tutorial_01_heart"

    if test_mode:
        data_dir = HEART_CT_KCL.data_directory(test_mode) / "slicer_heart_small"
        number_of_iterations_greedy = [1, 0]
        frame_files = sorted(data_dir.glob("slice_???.mha"))[0:2]
    else:
        data_dir = HEART_CT_KCL.data_directory(test_mode) / "Slicer-Heart-CT"
        number_of_iterations_greedy = [30, 15, 7, 3]
        frame_files = sorted(data_dir.glob("slice_???.mha"))

    log_level = logging.INFO

    registration_method = RegisterImagesGreedy(log_level=log_level)
    registration_method.set_number_of_iterations(number_of_iterations_greedy)

    segmentation_method = SegmentChestTotalSegmentatorWithContrast(log_level=log_level)
    segmentation_method.set_has_academic_license(True)

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
    reference_image = time_series_images[int(0.7 * len(time_series_images))]

    print("Number of time-series images:", len(time_series_images))

    # Workflow initialization

    workflow = WorkflowConvertImageToUSD(
        time_series_images=time_series_images,
        reference_image=reference_image,
        output_directory=str(output_dir),
        usd_project_name="cardiac_model",
        registration_method=registration_method,
        segmentation_method=segmentation_method,
        surface_reduction_rate=HEART_CT_KCL.surface_reduction_rate,
        log_level=log_level,
        save_assets=True,
    )

    # Workflow execution
    workflow_results = workflow.process()

    # if dynamic_labelmap_ids is not None, there are two USD files
    if len(workflow.dynamic_labelmap_ids) > 0:
        usd_file = output_dir / workflow_results["dynamic"]
    else:
        usd_file = output_dir / workflow_results["all"]

    # Result saving
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []

    test_image_num = int(0.7 * len(input_filenames))
    test_image_path = output_dir / f"slice_{test_image_num:03d}_registered.mha"
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

        test_labelmap_path = output_dir / f"slice_{test_image_num:03d}_labelmap.mha"
        if test_labelmap_path.exists():
            test_labelmap = itk.imread(str(test_labelmap_path))
            screenshots.append(
                tt.save_screenshot_image_slice(
                    test_image,
                    f"slice_{test_image_num:03d}_labelmap_test.png",
                    axis=0,
                    slice_fraction=0.5,
                    colormap="gray",
                    vmin=-200,
                    vmax=600,
                    overlay_mask=test_labelmap,
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
