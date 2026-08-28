"""
Tutorial 1: Lung-Gated 4D CT to Animated USD

Purpose
-------
Convert a respiratory-gated 4D lung CT scan (multiple breathing phases) into an
animated USD model suitable for visualization in NVIDIA Omniverse. The workflow
segments the lungs and surrounding chest anatomy from a reference phase,
registers all other respiratory phases to that reference using deep-learning
registration, and assembles the resulting time-varying surface meshes into a
single USD file with anatomical materials applied.

Inputs
------
- A set of 3D CT volumes (``*.nii.gz``) representing successive respiratory
  phases of one TCIA 4D-Lung case.
  Expected location: ``data/TCIA-4DLung/100_HM10395/100_HM10395_g0??.nii.gz``.
- The mid-inspiration phase (index ~0.7 through the series) is used as the
  reference frame for segmentation and registration.

Outputs (under ``tutorials/output/tutorial_01_lung/``)
-----------------------------------------------------
- An animated USD file with anatomy materials, named after the workflow's
  ``usd_project_name`` (``lung_model``).
- Screenshots (PNG) for documentation and regression testing:
  - ``slice_<n>_registered_test.png`` - each registered respiratory phase
  - ``lung_model_test.png`` - a rendered view of the exported USD model

Strengths
---------
- Single call (``WorkflowConvertImageToUSD.process()``) runs the full pipeline.
- Registers on the CPU with ``RegisterImagesICON``; no GPU needed for this stage.
- Output is Omniverse-ready with anatomical materials (ProcessUSDAnatomy).

Weaknesses / Limitations
------------------------
- Segmentation quality depends on TotalSegmentator's training distribution;
  unusual pathologies or pediatric anatomy may degrade results.
- Large 4D datasets (>20 phases, high resolution) can require 32 GB+ RAM.

Classes Used
------------
- WorkflowConvertImageToUSD (workflow_convert_image_to_usd.py):
    Orchestrates the full pipeline: CT phases -> segmentation -> registration ->
    contour extraction -> USD export.
- SegmentChestTotalSegmentator (segment_chest_total_segmentator.py):
    Deep-learning segmentation of 117 anatomical structures (used internally).
- RegisterImagesICON (register_images_icon.py):
    Frame-to-frame image registration (used internally).
- ProcessContours (process_contours.py):
    Extracts and transforms surface meshes from segmentation masks (used internally).
- ProcessUSDAnatomy (process_usd_anatomy.py):
    Applies clinical material colours to USD prims (used internally).

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: TCIA 4D-Lung - see ``data/TCIA-4DLung/README.md``.
This script expects the ``100_HM10395_g0??.nii.gz`` phase volumes to already
exist under ``data/TCIA-4DLung/100_HM10395/``.

Segmentation Models
-------------------
By default this tutorial uses only TotalSegmentator tasks permitted without a
separate model license. Lung lobes, vessels, airways, and the body task remain
available; high-resolution heart chambers and tissue classes are omitted. Set
``use_totalsegmentator_licensed_tasks`` to ``True`` only after configuring the
corresponding TotalSegmentator license.
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk
from parameters_tcia_4d_lung import TCIA_4D_LUNG

from monai_physio import (
    ProcessTests,
    RegisterImagesGreedy,
    SegmentChestTotalSegmentator,
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

    class_name = "tutorial_01_lung_gated_ct_to_usd"

    test_mode = ProcessTests.running_as_test()

    output_dir = TCIA_4D_LUNG.output_directory(test_mode) / "tutorial_01_lung"

    data_dir = TCIA_4D_LUNG.input_directory(test_mode)

    if test_mode:
        number_of_iterations_greedy = TCIA_4D_LUNG.number_of_iterations_greedy_test
        frame_files = sorted(data_dir.glob("100_HM10395_g0??.nii.gz"))[0:2]
    else:
        number_of_iterations_greedy = TCIA_4D_LUNG.number_of_iterations_greedy
        frame_files = sorted(data_dir.glob("100_HM10395_g0??.nii.gz"))

    log_level = logging.INFO
    use_totalsegmentator_licensed_tasks = False

    registration_method = RegisterImagesGreedy(log_level=log_level)
    registration_method.set_number_of_iterations(number_of_iterations_greedy)
    registration_method.set_metric(TCIA_4D_LUNG.greedy_metric)

    segmentation_method = SegmentChestTotalSegmentator(log_level=log_level)
    segmentation_method.set_has_academic_license(use_totalsegmentator_licensed_tasks)

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    input_filenames = [str(path) for path in frame_files]
    if not input_filenames:
        raise FileNotFoundError(
            "TCIA-4DLung data not found. Checked:\n"
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
        usd_project_name="lung_model",
        registration_method=registration_method,
        segmentation_method=segmentation_method,
        surface_reduction_rate=TCIA_4D_LUNG.surface_reduction_rate,
        log_level=log_level,
        frames_per_second=1,
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
                "lung_model_test.png",
            )
        )

    tutorial_results = {"usd_file": str(usd_file), "screenshots": screenshots}
