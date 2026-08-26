"""
Tutorial 5: VTK Surface Series to USD

Purpose
-------
Convert the VTK surface output from Tutorial 4, or another VTK-compatible mesh,
into a USD file with anatomy materials.

Tutorial 4 writes one VTP per anatomical structure, each annotated with its
structure name. Feeding those files in individually — rather than the single
combined surface — keeps that name attached through the conversion, so each
structure becomes its own named USD prim and gets its own material: bright
oxygenated red for the left chambers, darker deoxygenated red for the right,
red-brown myocardium, and so on, instead of one uniform heart material.

Data Required
-------------
Preferred input: ``tutorials/output/tutorial_04_heart/patient_*.vtp``
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import pyvista as pv

from parameters_base import ParametersBase
from physiotwin4d import (
    TestTools,
    WorkflowConvertVTKToUSD,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_05_heart_vtk_to_usd"

    # Only the shared directory roots are needed here; no dataset-specific
    # parameters module applies to this tutorial.
    tutorial_paths = ParametersBase()
    test_mode = TestTools.running_as_test()

    output_dir = tutorial_paths.output_directory(test_mode) / "tutorial_05_heart"
    baselines_dir = repo_root / "tests" / "baselines"

    project_name = "tutorial_04_heart"

    # Preferred input: the per-structure surfaces saved by Tutorial 4.
    input_dir = tutorial_paths.output_directory(test_mode) / "tutorial_04_heart"

    log_level = logging.INFO

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    # Tutorial 4 writes both per-group and per-structure surfaces into one
    # directory. A per-structure surface is the one carrying exactly one name
    # in SegmentationLabelNames, and its filename ends with that name; the
    # per-group surfaces list every structure in the group and are skipped, as
    # their geometry would otherwise be exported twice.
    meshes: list[pv.PolyData] = []
    structure_names: list[str] = []
    for vtk_file in sorted(input_dir.glob("patient_*.vtp")):
        mesh = pv.read(str(vtk_file))
        label_names = mesh.field_data.get("SegmentationLabelNames")
        if label_names is None or len(label_names) != 1:
            continue
        if not vtk_file.stem.endswith(str(label_names[0])):
            continue
        meshes.append(mesh)
        structure_names.append(str(label_names[0]))

    if not meshes:
        raise FileNotFoundError(
            "No per-structure surfaces found. Checked:\n"
            + f"  - {input_dir}/patient_*.vtp\n"
            + "Run tutorial_04_heart_ct_to_vtk.py with save_label_surfaces=True."
        )

    # Workflow initialization
    #
    # static_merge=True treats the meshes as separate objects in one scene
    # rather than as frames of a time series. Leaving anatomy_type unset lets
    # each object's name select its material, and object names default to the
    # structure names read from field_data above.
    workflow = WorkflowConvertVTKToUSD(
        input_meshes=meshes,
        usd_project_name=project_name,
        output_directory=output_dir,
        appearance="anatomy",
        static_merge=True,
        separate_by_connectivity=True,
        log_level=log_level,
    )

    # Workflow execution
    results = workflow.process()

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    screenshots = [
        tt.save_screenshot_openusd(
            results["usd_file"],
            f"{project_name}_usd_mesh_rendering.png",
        )
    ]

    tutorial_results = {
        "usd_file": results["usd_file"],
        "structures": structure_names,
        "screenshots": screenshots,
    }
