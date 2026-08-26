"""
Tutorial 6 (Heart): Create a PCA Statistical Shape Model

Purpose
-------
Build a PCA statistical shape model from a reference mesh and a small population
of sample meshes, less ``ParametersHeartCTKCL.hold_out_case``, which Tutorial 7
fits the model to. Tutorial 7 reuses the saved ``pca_model.json``.

Data Required
-------------
Full data: ``data/KCL-Heart-Model``
Test data: ``data/test/KCL-Heart-Model``
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pv
from parameters_heart_ct_kcl import HEART_CT_KCL

from physiotwin4d import (
    ContourTools,
    TestTools,
    WorkflowCreateStatisticalModel,
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

    class_name = "tutorial_06_heart_create_statistical_model"

    test_mode = TestTools.running_as_test()

    output_dir = HEART_CT_KCL.output_directory(test_mode) / "tutorial_06_heart"
    weights_dir = HEART_CT_KCL.weights_directory(test_mode)

    baselines_dir = repo_root / "tests" / "baselines"

    data_dir = HEART_CT_KCL.input_directory(test_mode)
    number_of_pca_components = HEART_CT_KCL.pca_components(test_mode)

    # Points kept per surface; 0 keeps every point.  The KCL meshes are the one
    # dataset with no downsampled test subset, so test mode reduces them here
    # instead, as tutorial_06_duke_heart_create_statistical_model.py does.
    model_points = HEART_CT_KCL.points_per_model(test_mode)

    # Distance-map weights finetuned by
    # tutorial_02_duke_heart_distancemap_finetune_icon.py.  Stock uniGradICON weights
    # are out of distribution for distance maps, so without these the
    # correspondences this model is built from barely move off the template,
    # and the modes come out far too tight.  Tutorial 7 fits with the same
    # checkpoint.
    icon_weights_path = (
        weights_dir
        / "icon_duke_heart_distancemap"
        / "icon_duke_heart_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    log_level = logging.INFO

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    reference_file = data_dir / "average_mesh.vtk"
    if not reference_file.exists():
        raise FileNotFoundError(
            f"KCL-Heart-Model reference mesh not found: {reference_file}\n"
            "See data/README.md for download instructions."
        )

    sample_dir = data_dir / "input_meshes"
    sample_files = sorted(sample_dir.glob("*.vtk"))
    if not sample_files:
        sample_files = sorted(data_dir.glob("*.vtk"))
        sample_files = [
            path for path in sample_files if path.name != reference_file.name
        ]
    # Tutorial 7 fits this model to the held-out case, so the model must not
    # have seen it.  The KCL meshes carry no DIR-Lab case, so this drops nothing
    # today; adding one cannot slip it in.
    sample_files = [
        path for path in sample_files if HEART_CT_KCL.hold_out_case not in path.name
    ]
    if test_mode:
        sample_files = sample_files[:3]
    if len(sample_files) < 3:
        raise FileNotFoundError(
            f"Need at least 3 sample meshes under {sample_dir} or {data_dir}.\n"
            "See data/README.md for download instructions."
        )

    contour_tools = ContourTools(log_level=log_level)

    def read_model_surface(path: Path) -> pv.DataSet:
        """Read a mesh, reduced to ``model_points`` when a budget is set."""
        mesh = cast(pv.DataSet, pv.read(str(path)))
        if not model_points:
            return mesh
        surface = contour_tools.extract_surface(mesh)
        return cast(
            pv.DataSet,
            contour_tools.remesh_and_smooth_surface(
                surface, 1.0 - model_points / surface.n_points, 0
            ),
        )

    reference_mesh = read_model_surface(reference_file)
    sample_meshes = [read_model_surface(path) for path in sample_files]

    # Workflow initialization

    workflow = WorkflowCreateStatisticalModel(
        sample_meshes=sample_meshes,
        reference_mesh=reference_mesh,
        number_of_pca_components=number_of_pca_components,
        # The distance maps step 3 registers are rasterized at this resolution,
        # and generating, dilating and affinely registering them is what the
        # step costs.  2 mm is an eighth of the voxels of the 1 mm default.
        reference_spatial_resolution=2.0 if test_mode else 1.0,
        icp_transform_type=HEART_CT_KCL.icp_transform_type,
        mask_dilation_mm=HEART_CT_KCL.mask_dilation_mm,
        distance_squared_max=HEART_CT_KCL.distancemap_squared_max,
        log_level=log_level,
    )

    # Build the correspondences with the same distance-map scaling and weights
    # Tutorial 7 fits with, so the model and the fit measure shape alike.
    if icon_weights_path.exists():
        workflow.set_icon_weights_path(str(icon_weights_path))
    else:
        workflow.log_warning(
            "Finetuned distance-map ICON weights not found at %s; building the "
            "model with the stock uniGradICON weights, which are out of "
            "distribution for distance maps and will understate the "
            "population's variance. Run "
            "tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py "
            "to create them.",
            icon_weights_path,
        )

    # Workflow execution
    result = workflow.process()

    # Result saving
    pca_model: dict[str, Any] = result["pca_model"]
    mean_surface: pv.PolyData = result["pca_mean_surface"]

    model_file = HEART_CT_KCL.pca_model_file(test_mode)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("w", encoding="utf-8") as f:
        json.dump(pca_model, f, indent=2)

    mean_surface_file = HEART_CT_KCL.pca_mean_surface_file(test_mode)
    mean_surface.save(str(mean_surface_file))

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    screenshots.append(
        tt.save_screenshot_mesh(
            mean_surface,
            "pca_mean_model.png",
            camera_position="iso",
            color="steelblue",
            opacity=0.9,
        )
    )

    components = pca_model.get("components", [])
    eigenvalues = pca_model.get("eigenvalues", [])
    mean_points = np.asarray(mean_surface.points)
    mode_count = min(2, number_of_pca_components, len(components), len(eigenvalues))

    xvfb_started = False
    try:
        pv.start_xvfb()
        xvfb_started = True
    except Exception:
        pass

    mode_surface_files: list[Path] = []

    try:
        for mode_idx in range(mode_count):
            sigma = float(np.sqrt(eigenvalues[mode_idx]))
            mode_offsets = np.asarray(components[mode_idx]).reshape(-1, 3)

            minus_mesh = mean_surface.copy()
            minus_mesh.points = mean_points - 2.0 * sigma * mode_offsets
            plus_mesh = mean_surface.copy()
            plus_mesh.points = mean_points + 2.0 * sigma * mode_offsets

            # Save the mode extremes so they can be inspected outside this
            # script; all share the mean surface's topology.
            for tag, mode_mesh in (("minus", minus_mesh), ("plus", plus_mesh)):
                mode_surface_file = (
                    output_dir / f"pca_mode_{mode_idx + 1:02d}_{tag}_2sigma.vtp"
                )
                mode_mesh.save(str(mode_surface_file))
                mode_surface_files.append(mode_surface_file)

            plotter = pv.Plotter(off_screen=True, window_size=[1200, 500], shape=(1, 3))
            plotter.subplot(0, 0)
            plotter.add_mesh(minus_mesh, color="royalblue", opacity=0.9)
            plotter.camera_position = "iso"
            plotter.subplot(0, 1)
            plotter.add_mesh(mean_surface, color="steelblue", opacity=0.9)
            plotter.camera_position = "iso"
            plotter.subplot(0, 2)
            plotter.add_mesh(plus_mesh, color="coral", opacity=0.9)
            plotter.camera_position = "iso"

            png_path = output_dir / f"pca_mode_{mode_idx + 1:02d}.png"
            plotter.screenshot(str(png_path))
            plotter.close()
            screenshots.append(png_path)
    finally:
        # Pair start_xvfb with cleanup, guarded like the startup above so
        # environments without Xvfb (e.g. Windows, pyvista >= 0.48) are unaffected.
        if xvfb_started:
            try:
                pv.stop_xvfb()
            except Exception:
                pass

    tutorial_results = {
        "pca_model": pca_model,
        "mean_surface": mean_surface,
        "model_file": model_file,
        "mean_surface_file": mean_surface_file,
        "mode_surface_files": mode_surface_files,
        "screenshots": screenshots,
    }
