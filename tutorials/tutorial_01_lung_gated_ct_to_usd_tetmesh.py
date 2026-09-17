"""
Tutorial 1 (Tetmesh Variant): Lung-Gated 4D CT to Animated Tetrahedral Mesh

Purpose
-------
Convert a respiratory-gated 4D lung CT scan (multiple breathing phases) into a
time series of tetrahedral volume meshes of the lung, one per respiratory
phase. Unlike Tutorial 1, which builds a surface-only animated USD model, this
variant fills the reference-frame lung surface with tetrahedra and warps that
single volume mesh through every phase's registration transform, giving a
4D tetmesh suitable for physics-based simulation (e.g. finite-element lung
motion) rather than visualization alone.

Inputs
------
- A set of 3D CT volumes (``*.nii.gz``) representing successive respiratory
  phases of one TCIA 4D-Lung case.
  Expected location: ``data/TCIA-4DLung/100_HM10395/100_HM10395_g0??.nii.gz``.
- The mid-inspiration phase (index ~0.7 through the series) is used as the
  reference frame for segmentation, meshing, and registration.

Outputs (under ``tutorials/output/tutorial_01_lung_tetmesh/``)
----------------------------------------------------------------
- One tetrahedral mesh (``*.vtu``) per respiratory phase, named
  ``lung_tetmesh_<phase>.vtu``, all sharing the reference mesh's connectivity.
- A screenshot (PNG) of the reference-phase lung surface for documentation and
  regression testing: ``lung_surface_test.png``.

Strengths
---------
- Single reference tetmesh, warped rather than re-meshed per phase, so every
  frame shares the same connectivity -- required for a 4D finite-element
  simulation to track individual elements across phases.
- Registers on the CPU with ``RegisterImagesGreedy``; no GPU needed for this
  stage.
- Inverted or degenerate elements introduced by large deformations are
  repaired per frame via ``ProcessContours.repair_inverted_tetrahedra``.

Weaknesses / Limitations
------------------------
- Segmentation quality depends on TotalSegmentator's training distribution;
  unusual pathologies or pediatric anatomy may degrade results.
- Large 4D datasets (>20 phases, high resolution) can require 32 GB+ RAM.

Classes Used
------------
- SegmentChestTotalSegmentator (segment_chest_total_segmentator.py):
    Deep-learning segmentation of the lung from the reference phase.
- RegisterImagesGreedy (register_images_greedy.py):
    Frame-to-frame image registration.
- ProcessContours (process_contours.py):
    Extracts the reference lung surface and tetmesh, and warps/repairs the
    tetmesh per phase.

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: TCIA 4D-Lung - see ``data/TCIA-4DLung/README.md``.
This script expects the ``100_HM10395_g0??.nii.gz`` phase volumes to already
exist under ``data/TCIA-4DLung/100_HM10395/``.
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk
from parameters_tcia_4d_lung import TCIA_4D_LUNG

from monai_physio import (
    MONAIPhysioBase,
    ProcessContours,
    ProcessTests,
    RegisterImagesGreedy,
    SegmentChestTotalSegmentator,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used by TotalSegmentator) spawns a multiprocessing.Pool. On
# Windows the spawn start method re-imports this script in each child;
# without the __name__ == "__main__" guard around the top-level work, that
# re-import fires the pipeline again and Python's spawn-cascade detector
# raises RuntimeError.
if __name__ == "__main__":
    # Data directory specification

    class_name = "tutorial_01_lung_gated_ct_to_usd_tetmesh"

    test_mode = ProcessTests.running_as_test()

    output_dir = TCIA_4D_LUNG.output_directory(test_mode) / "tutorial_01_lung_tetmesh"

    data_dir = TCIA_4D_LUNG.input_directory(test_mode)

    if test_mode:
        number_of_iterations_greedy = [1, 0]
        frame_files = sorted(data_dir.glob("100_HM10395_g0??.nii.gz"))[0:2]
    else:
        number_of_iterations_greedy = [30, 15, 7, 3]
        frame_files = sorted(data_dir.glob("100_HM10395_g0??.nii.gz"))

    log_level = logging.INFO
    reporter = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    registration_method = RegisterImagesGreedy(log_level=log_level)
    registration_method.set_number_of_iterations(number_of_iterations_greedy)

    segmentation_method = SegmentChestTotalSegmentator(log_level=log_level)
    segmentation_method.set_has_academic_license(True)

    contour_tools = ProcessContours(log_level=log_level)

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
    reference_index = int(0.7 * len(time_series_images))
    reference_image = time_series_images[reference_index]

    reporter.log_info("Number of time-series images: %d", len(time_series_images))

    # Reference-frame segmentation and tetmesh construction

    reporter.log_section("Segmenting reference frame")
    seg_result = segmentation_method.segment(reference_image)
    lung_mask = seg_result["lung"]

    reporter.log_section("Building reference lung tetmesh")
    reference_surface = contour_tools.extract_watertight_surface(
        lung_mask,
        surface_reduction_rate=TCIA_4D_LUNG.surface_reduction_rate,
        anatomy_names=["lung"],
    )
    reference_tetmesh = contour_tools.extract_tetrahedra(
        lung_mask,
        element_size_mm=TCIA_4D_LUNG.mesh_element_size_mm,
        anatomy_names=["lung"],
    )
    reference_tetmesh = contour_tools.trim_tetrahedra_to_surface(
        reference_tetmesh, reference_surface
    )
    reporter.log_info(
        "Reference tetmesh: %d points, %d cells",
        reference_tetmesh.n_points,
        reference_tetmesh.n_cells,
    )

    # Registration and per-phase tetmesh warping

    registration_method.set_fixed_image(reference_image)

    tetmesh_files: list[Path] = []
    for i, moving_image in enumerate(time_series_images):
        reporter.log_progress(
            i + 1, len(time_series_images), prefix="Registering and warping phases"
        )
        if i == reference_index:
            phase_tetmesh = reference_tetmesh
        else:
            reg_results = registration_method.register(moving_image)
            phase_tetmesh = contour_tools.transform_contours(
                reference_tetmesh, reg_results["fixed_to_moving_transform"]
            )
            phase_tetmesh = contour_tools.repair_inverted_tetrahedra(phase_tetmesh)

        tetmesh_file = output_dir / f"lung_tetmesh_{i:03d}.vtu"
        phase_tetmesh.save(tetmesh_file)
        tetmesh_files.append(tetmesh_file)

    # Result saving
    tt = ProcessTests(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = [
        tt.save_screenshot_mesh(
            reference_surface,
            "lung_surface_test.png",
            camera_position="iso",
            color="lightblue",
            opacity=0.85,
        )
    ]

    tutorial_results = {
        "tetmesh_files": [str(path) for path in tetmesh_files],
        "screenshots": screenshots,
    }
