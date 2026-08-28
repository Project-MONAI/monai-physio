"""Tutorial 4 (Lung): CT segmentation to VTK surfaces with NV-Segment-CTMR.

Purpose
-------
Convert one CT frame into a detailed labelmap, a lung-only labelmap, and VTK
surfaces. This exposes the image-to-anatomy boundary that participants replace
when adapting the pipeline to another organ.

Data Required
-------------
Full data: ``data/DirLab-4DCT/Case1Pack_T??.mha``
Test data: ``data/test/DirLab-4DCT/Case1Pack_T??.mha``

Segmentation Models
-------------------
This tutorial uses the shared lung segmenter, ``SegmentNVSegmentCTMRI``. Its
weights use the NVIDIA OneWay Non-Commercial License. The output keeps the
model's published label IDs; the lung-only labelmap retains the five lobe IDs
and any other predicted lung labels while clearing non-lung voxels.
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk
import numpy as np
import pyvista as pv

from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from monai_physio import (
    ContourTools,
    TestTools,
    WorkflowConvertImageToVTK,
)

# Only run if this script is not imported as a module

# Keep GPU segmentation and file writes below the main guard so importing this
# tutorial for documentation or test discovery has no runtime side effects.
if __name__ == "__main__":
    # Data directory specification

    project_name = "tutorial_04_lung"
    output_prefix = "patient_nvsegmentctmri"

    test_mode = TestTools.running_as_test()

    output_dir = LUNG_CT_DIRLAB.output_directory(test_mode) / project_name

    # In addition to the combined surface file always saved below, also
    # save one VTP per anatomy group (e.g. heart.vtp, lung.vtp) and/or one
    # VTP per individual anatomical structure (e.g. left_ventricle.vtp).
    save_group_surfaces = True
    save_label_surfaces = True

    if test_mode:
        data_dir = LUNG_CT_DIRLAB.data_directory(test_mode) / "DirLab-4DCT"
    else:
        data_dir = LUNG_CT_DIRLAB.data_directory(test_mode) / "DirLab-4DCT"

    frame_files = sorted(data_dir.glob("Case1Pack_T??.mha"))

    log_level = logging.INFO
    segmentation_method = LUNG_CT_DIRLAB.segmenter_class(log_level=log_level)

    # Directory setup and data reading
    output_dir.mkdir(parents=True, exist_ok=True)

    if not frame_files:
        raise FileNotFoundError(
            "DirLab-4DCT frame data not found. Checked:\n"
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
        anatomy_groups=[LUNG_CT_DIRLAB.anatomy_group],
        surface_reduction_rate=LUNG_CT_DIRLAB.surface_reduction_rate,
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
            str(output_dir / f"{output_prefix}_surfaces.vtp"),
        )
    )
    if save_group_surfaces:
        ContourTools.save_surfaces(
            result["surfaces"], str(output_dir), prefix=output_prefix
        )
    if save_label_surfaces:
        ContourTools.save_surfaces(
            result["label_surfaces"], str(output_dir), prefix=output_prefix
        )
    labelmap_file = output_dir / f"{output_prefix}_labelmap.mha"
    itk.imwrite(result["labelmap"], str(labelmap_file), compression=True)

    # Keep the model's native lobe ids but clear every non-lung label. This is
    # the labelmap participants inspect or replace for another organ.
    lung_label_ids = np.array(
        sorted(
            segmentation_method.taxonomy.labels_in_group(LUNG_CT_DIRLAB.anatomy_group)
        ),
        dtype=np.uint16,
    )
    full_labelmap_arr = itk.GetArrayViewFromImage(result["labelmap"])
    lung_labelmap = itk.GetImageFromArray(
        np.where(np.isin(full_labelmap_arr, lung_label_ids), full_labelmap_arr, 0)
    )
    lung_labelmap.CopyInformation(result["labelmap"])
    lung_labelmap_file = output_dir / f"{output_prefix}_lung_labelmap.mha"
    itk.imwrite(lung_labelmap, str(lung_labelmap_file), compression=True)

    # Testing
    tt = TestTools(
        class_name=project_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    screenshots.append(
        tt.save_screenshot_image_slice(
            ct_image,
            f"{project_name}_segmentation_overlay.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
            vmin=-200,
            vmax=600,
            overlay_mask=lung_labelmap,
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
                f"{project_name}_vtk_surfaces.png",
                camera_position="iso",
                color="lightblue",
                opacity=0.85,
            )
        )

    tutorial_results = {
        "result": result,
        "surface_file": surface_file,
        "labelmap_file": labelmap_file,
        "lung_labelmap_file": lung_labelmap_file,
        "screenshots": screenshots,
    }
