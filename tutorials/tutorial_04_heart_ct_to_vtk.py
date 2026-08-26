"""
Tutorial 4 (Heart): CT Segmentation to VTK Surfaces

Purpose
-------
Segment one 3D CT frame into anatomical groups and save a combined VTK
surface file. The output can be inspected directly in PyVista or used as
input for Tutorial 5.

Data Required
-------------
Full data: ``data/Slicer-Heart-CT/slice_???.mha``
Test data: ``data/test/slicer_heart_small/slice_???.mha``
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk
import pyvista as pv
from parameters_heart_ct_kcl import HEART_CT_KCL

from physiotwin4d import (
    ContourTools,
    SegmentAnatomyBase,
    SegmentChestTotalSegmentatorWithContrast,
    SegmentHeartSimpleware,
    TestTools,
    WorkflowConvertImageToVTK,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # Data directory specification

    class_name = "tutorial_04_heart_ct_to_vtk"

    test_mode = TestTools.running_as_test()

    output_dir = HEART_CT_KCL.output_directory(test_mode) / "tutorial_04_heart"

    # In addition to the combined surface file always saved below, also
    # save one VTP per anatomy group (e.g. heart.vtp, lung.vtp) and/or one
    # VTP per individual anatomical structure (e.g. left_ventricle.vtp).
    save_group_surfaces = True
    save_label_surfaces = True

    use_simpleware = False
    use_totalsegmentator_academic_license = True

    if test_mode:
        data_dir = HEART_CT_KCL.data_directory(test_mode) / "slicer_heart_small"
    else:
        data_dir = HEART_CT_KCL.data_directory(test_mode) / "Slicer-Heart-CT"

    frame_files = sorted(data_dir.glob("slice_???.mha"))

    log_level = logging.INFO

    if use_simpleware:
        segmentation_method: SegmentAnatomyBase = SegmentHeartSimpleware(
            log_level=log_level
        )
    else:
        total_segmentation_method = SegmentChestTotalSegmentatorWithContrast(
            log_level=log_level
        )
        total_segmentation_method.set_has_academic_license(
            use_totalsegmentator_academic_license
        )
        segmentation_method = total_segmentation_method

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    if not frame_files:
        raise FileNotFoundError(
            "Slicer-Heart-CT frame data not found. Checked:\n"
            + f"  - {data_dir}\n"
            + "See data/README.md for download instructions."
        )

    ct_file = frame_files[0]
    ct_image = itk.imread(str(ct_file))

    # Workflow initialization

    workflow = WorkflowConvertImageToVTK(
        segmentation_method=segmentation_method,
        log_level=log_level,
    )

    # Workflow execution
    #
    # surface_reduction_rate decimates each exported VTP surface.
    result = workflow.process(
        input_image=ct_image,
        surface_reduction_rate=HEART_CT_KCL.surface_reduction_rate,
        extract_label_surfaces=save_label_surfaces,
    )

    # Result saving
    #
    # Merging the per-structure surfaces, rather than the per-group ones, lets
    # the combined file carry a per-cell SegmentationLabelIds array: structure
    # identity survives the merge, so the file can still be split per structure
    # downstream. Per-group surfaces are contoured from a merged binary mask
    # and have no per-cell identity to record.
    combined_input = (
        result["label_surfaces"] if save_label_surfaces else result["surfaces"]
    )
    surface_file = Path(
        ContourTools.save_combined_surfaces(
            combined_input,
            str(output_dir / "patient_surfaces.vtp"),
        )
    )
    if save_group_surfaces:
        ContourTools.save_surfaces(
            result["surfaces"], str(output_dir), prefix="patient"
        )
    if save_label_surfaces:
        ContourTools.save_surfaces(
            result["label_surfaces"], str(output_dir), prefix="patient"
        )
    labelmap_file = output_dir / "patient_labelmap.mha"
    itk.imwrite(result["labelmap"], str(labelmap_file), compression=True)

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    screenshots.append(
        tt.save_screenshot_image_slice(
            ct_image,
            "segmentation_overlay.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
            vmin=-200,
            vmax=600,
            overlay_mask=result["labelmap"],
        )
    )

    surfaces = [
        surface for surface in result["surfaces"].values() if surface is not None
    ]
    if surfaces:
        combined_surface = pv.merge(surfaces) if len(surfaces) > 1 else surfaces[0]
        screenshots.append(
            tt.save_screenshot_mesh(
                combined_surface,
                "vtk_surfaces.png",
                camera_position="iso",
                color="lightblue",
                opacity=0.85,
            )
        )

    tutorial_results = {
        "result": result,
        "surface_file": surface_file,
        "labelmap_file": labelmap_file,
        "screenshots": screenshots,
    }
