"""
Tutorial 5 (Duke Heart): 4D VTK Surfaces to Animated USD

Purpose
-------
Assemble the per-frame, per-structure surfaces Tutorial 4 (Duke Heart) wrote
into one animated USD per case: every gated frame becomes a time sample, every
label its own prim under ``/World/<case>/<anatomy_group>/<structure>``, painted
with that structure's OmniSurface material.

Tutorial 4 saves one VTP per frame holding every structure, each cell tagged
with its originating label id in ``SegmentationLabelIds``.
``WorkflowConvertVTKToUSD`` splits on that array when given ``label_names``,
which is what keeps structure identity -- and therefore per-structure
materials -- through the time series; without it a time series is split by
connectivity instead, and a component's index is no guarantee of which
structure it holds from one frame to the next.  The whole-heart surface
Tutorial 4 writes beside those files is skipped: its geometry is already
covered by the per-structure ones and would be exported twice.

Each frame is contoured from its own labelmap, so the frames agree on neither
point count nor triangulation.  The stage therefore carries time-sampled
topology, which USD holds rather than interpolates, so playback snaps from
frame to frame; a surface propagated through a registration (Tutorial 1) keeps
one topology and does interpolate.

Data Required
-------------
``tutorials/output/tutorial_04_duke_heart_labelmap/pm????_*_surfaces.vtp``
(run ``tutorial_04_duke_heart_labelmap_to_vtk.py`` first)
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pyvista as pv

from parameters_base import ParametersBase
from monai_physio import (
    MONAIPhysioBase,
    SegmentHeartSimplewareTrimmedBranches,
    TestTools,
    WorkflowConvertVTKToUSD,
)

# Only run if this script is not imported as a module
if __name__ == "__main__":
    # Data directory specification

    class_name = "tutorial_05_duke_heart_vtk_to_usd"

    # Only the shared directory roots are needed here; no dataset-specific
    # parameters module applies to this tutorial.
    tutorial_paths = ParametersBase()
    test_mode = TestTools.running_as_test()

    input_dir = (
        tutorial_paths.output_directory(test_mode) / "tutorial_04_duke_heart_labelmap"
    )
    output_dir = tutorial_paths.output_directory(test_mode) / "tutorial_05_duke_heart"

    log_level = logging.INFO
    reporter = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    # The labelmaps were produced by Simpleware ASCardio, so its taxonomy is
    # what maps each structure onto an anatomy group and a material.
    segmenter = SegmentHeartSimplewareTrimmedBranches(log_level=logging.WARNING)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Tutorial 4 writes every case's frames into one directory, and every frame
    # stem starts with its case id, so the cases are the distinct prefixes.
    # Sorting by name puts each case's frames in gating order.
    frame_files: dict[str, list[Path]] = {}
    for vtp_file in sorted(input_dir.glob("pm[0-9][0-9][0-9][0-9]_*_surfaces.vtp")):
        frame_files.setdefault(vtp_file.name.split("_")[0], []).append(vtp_file)
    if not frame_files:
        raise FileNotFoundError(
            f"No pm????_*_surfaces.vtp frame surfaces found under {input_dir}.\n"
            "Run tutorial_04_duke_heart_labelmap_to_vtk.py first."
        )

    # Conversion
    usd_files: list[Path] = []
    last_time_codes: list[float] = []
    all_structures: set[str] = set()
    label_names = segmenter.taxonomy.all_labels()
    for case_id, case_files in frame_files.items():
        # One file per frame, already holding every structure.
        frame_meshes = [
            cast(pv.PolyData, pv.read(str(vtp_file))) for vtp_file in case_files
        ]

        # The merged files carry label ids but not names, which come from the
        # taxonomy of the segmenter that produced the labelmaps.
        mask_ids = {
            int(label_id): label_names.get(int(label_id), f"label_{label_id}")
            for mesh in frame_meshes
            for label_id in np.unique(mesh.cell_data["SegmentationLabelIds"])
        }
        all_structures.update(mask_ids.values())
        reporter.log_section(
            f"{case_id}: {len(frame_meshes)} frames, {len(mask_ids)} structures"
        )

        # One frame per time code and one cardiac cycle per second of playback.
        workflow = WorkflowConvertVTKToUSD(
            input_meshes=frame_meshes,
            usd_project_name=case_id,
            output_directory=output_dir,
            separate_by_connectivity=False,
            appearance="anatomy",
            label_names=mask_ids,
            segmenter=segmenter,
            frames_per_second=float(len(frame_meshes)),
            log_level=log_level,
        )
        results = workflow.process()
        usd_files.append(Path(results["usd_file"]))
        last_time_codes.append(float(len(frame_meshes) - 1))

    reporter.log_section(f"Wrote {len(usd_files)} animated USD files to {output_dir}")

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    # The first frame and the last one: a stage whose later frames were written
    # against the first frame's topology renders as garbage in the second shot
    # while the first still looks right.
    screenshots: list[Path] = []
    if usd_files:
        screenshots.append(
            tt.save_screenshot_openusd(
                usd_files[0],
                f"{usd_files[0].stem}_usd_mesh_rendering.png",
            )
        )
        screenshots.append(
            tt.save_screenshot_openusd(
                usd_files[0],
                f"{usd_files[0].stem}_usd_mesh_rendering_last_frame.png",
                time_code=last_time_codes[0],
            )
        )

    tutorial_results = {
        "usd_files": usd_files,
        "structures": sorted(all_structures),
        "screenshots": screenshots,
    }
