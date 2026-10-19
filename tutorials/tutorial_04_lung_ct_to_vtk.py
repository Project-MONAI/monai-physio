"""
Tutorial 4 (Lung): CT Segmentation to VTK Surfaces

Purpose
-------
Segment one 3D CT frame into anatomical groups and save its VTK surfaces: one
file per anatomy group, one per structure, and every structure combined in one
file. The output can be inspected directly in PyVista or used as input for
Tutorial 5.

Data Required
-------------
Full data: ``data/TCIA-4DLung/100_HM10395/100_HM10395_g0??.nii.gz``
Test data: ``data/test/TCIA-4DLung/100_HM10395/100_HM10395_g0??.nii.gz``
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk
from parameters_tcia_4d_lung import TCIA_4D_LUNG

from monai_physio import (
    ProcessSurfaceMesh,
    ProcessTests,
    WorkflowConvertImageToVTK,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used by the segmenters inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # Data directory specification

    class_name = "tutorial_04_lung"

    test_mode = ProcessTests.running_as_test()

    output_dir = TCIA_4D_LUNG.output_directory(test_mode) / class_name

    data_dir = TCIA_4D_LUNG.input_directory(test_mode)

    frame_files = sorted(data_dir.glob("100_HM10395_g0??.nii.gz"))

    log_level = logging.INFO

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    if not frame_files:
        raise FileNotFoundError(
            "TCIA-4DLung frame data not found. Checked:\n"
            + f"  - {data_dir}\n"
            + "See data/README.md for download instructions."
        )

    ct_image = itk.imread(str(frame_files[0]))

    # Workflow execution
    #
    # surface_reduction_rate remeshes each surface to fewer triangles.
    workflow = WorkflowConvertImageToVTK(
        segmentation_method=TCIA_4D_LUNG.segmenter(test_mode, log_level=log_level),
        log_level=log_level,
    )
    result = workflow.process(
        input_image=ct_image,
        surface_reduction_rate=TCIA_4D_LUNG.surface_reduction_rate,
        extract_label_surfaces=True,
    )

    # Result saving
    #
    # The combined file merges the per-structure surfaces, so each cell keeps
    # the id of the structure it came from and the file can be split per
    # structure downstream.
    surface_file = Path(
        WorkflowConvertImageToVTK.save_combined_surfaces(
            result["label_surfaces"], str(output_dir / "patient_surfaces.vtp")
        )
    )
    WorkflowConvertImageToVTK.save_surfaces(
        result["surfaces"], str(output_dir), prefix="patient"
    )
    WorkflowConvertImageToVTK.save_surfaces(
        result["label_surfaces"], str(output_dir), prefix="patient"
    )
    labelmap_file = output_dir / "patient_labelmap.mha"
    itk.imwrite(result["labelmap"], str(labelmap_file), compression=True)

    # Testing
    tt = ProcessTests(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = [
        tt.save_screenshot_image_slice(
            ct_image,
            f"{class_name}_segmentation_overlay.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
            vmin=-200,
            vmax=600,
            overlay_mask=result["labelmap"],
        )
    ]
    if result["surfaces"]:
        screenshots.append(
            tt.save_screenshot_mesh(
                ProcessSurfaceMesh.combine_surfaces(result["surfaces"]),
                f"{class_name}_vtk_surfaces.png",
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
