"""
Tutorial 6 (Duke Heart): Create a PCA Statistical Shape Model

Purpose
-------
Build a PCA statistical shape model of the heart from the Duke-Heart-4DLabelmaps
population, starting from the surfaces Tutorial 4 (Duke Heart) wrote.  Only each
case's reference frame is read (``*_ref_*``), so the model spans patients rather
than cardiac phases.  An unbiased mean surface is built with
``WorkflowCreateMeanSurface``, and the population is decomposed into shape modes.
``ParametersDukeHeartLabelmaps.hold_out_case`` is left out of the population, so
that Tutorial 7 fitting the model to that case measures generalization rather
than reconstruction.  Tutorial 7 reuses the saved ``pca_model.json``.

The heart here is the whole heart minus its chamber cavities, the same structure
Tutorial 2 measures its distance maps to.

Data Required
-------------
``tutorials/output/tutorial_04_duke_heart_labelmap/
*_ref_heart_minus_interior_chambers.vtp``
(run ``tutorial_04_duke_heart_labelmap_to_vtk.py`` first)

Outputs (under ``tutorials/output/tutorial_06_duke_heart/``)
-----------------------------------------------------------
- ``reference_mean_surface.vtp`` - the unbiased atlas surface
- ``pca_model.json`` and ``pca_mean_surface.vtp`` - the shape model
- ``pca_mode_<k>_{minus,plus}_2sigma.vtp`` and ``pca_mode_<k>.png``

Runtime
-------
An ICP and a deformable registration per case per atlas iteration, then one more
of each per case for the model.  The atlas surface is cached on disk, so a
re-run only redoes the model.
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from monai_physio import (
    ContourTools,
    TestTools,
    WorkflowCreateMeanSurface,
    WorkflowCreateStatisticalModel,
)

# Only run if this script is not imported as a module
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_06_duke_heart_create_statistical_model"

    test_mode = TestTools.running_as_test()

    output_dir = DUKE_HEART.output_directory(test_mode) / "tutorial_06_duke_heart"
    weights_dir = DUKE_HEART.weights_directory(test_mode)

    baselines_dir = repo_root / "tests" / "baselines"

    input_dir = DUKE_HEART.input_directory(test_mode)
    number_of_pca_components = DUKE_HEART.pca_components(test_mode)

    # Atlas iterations used to build the reference surface; 1 is a single
    # template-biased pass.
    mean_surface_iterations = 1 if test_mode else 3

    # Points kept per surface; see the parameters module.
    model_points = DUKE_HEART.model_points

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

    contour_tools = ContourTools(log_level=log_level)

    # One reference-frame surface per case, less the held-out one: Tutorial 7
    # fits this model to that case, so the model must not have seen it.
    sample_files = [
        path
        for path in sorted(input_dir.glob("*_ref_heart_minus_interior_chambers.vtp"))
        if not path.name.startswith(DUKE_HEART.hold_out_case)
    ]
    if test_mode:
        sample_files = sample_files[:3]
    if len(sample_files) < 3:
        raise FileNotFoundError(
            f"Need at least 3 reference-frame heart surfaces under {input_dir}; "
            f"found {len(sample_files)}.\n"
            "Run tutorial_04_duke_heart_labelmap_to_vtk.py first."
        )

    sample_surfaces: list[pv.DataSet] = []
    for sample_file in sample_files:
        surface = cast(pv.PolyData, pv.read(str(sample_file)))
        sample_surfaces.append(
            contour_tools.remesh_and_smooth_surface(
                surface, 1.0 - model_points / surface.n_points, 0
            )
        )

    # The reference surface defines the topology every PCA input is expressed
    # in, so picking one case makes the model inherit that case's shape. Use the
    # unbiased mean of the population instead. Cached: it costs one deformable
    # registration per case per atlas iteration.
    reference_surface_file = output_dir / "reference_mean_surface.vtp"
    # Keyed on the settings the atlas was corresponded with, not on the file
    # merely being there: reusing an atlas built at one dilation, saturation
    # radius or checkpoint while the model below corresponds its samples at
    # another is the one way the two can disagree without saying so.
    mean_surface_settings = {
        "iterations": mean_surface_iterations,
        "mask_dilation_mm": DUKE_HEART.mask_dilation_mm,
        "distance_squared_max": DUKE_HEART.distancemap_squared_max,
        "icon_weights": (
            [str(icon_weights_path), icon_weights_path.stat().st_mtime_ns]
            if icon_weights_path.exists()
            else None
        ),
    }
    settings_file = output_dir / "reference_mean_surface_settings.json"
    cached_settings = (
        json.loads(settings_file.read_text(encoding="utf-8"))
        if reference_surface_file.exists() and settings_file.exists()
        else None
    )
    if cached_settings != mean_surface_settings:
        mean_workflow = WorkflowCreateMeanSurface(
            surfaces=sample_surfaces,
            template_surface=sample_surfaces[len(sample_surfaces) // 2],
            log_level=log_level,
        )
        mean_workflow.set_number_of_iterations(mean_surface_iterations)
        # Correspond the atlas with the same settings the model below uses, so
        # the template is not itself built from under-fitting registrations.
        mean_workflow.set_mask_dilation_mm(DUKE_HEART.mask_dilation_mm)
        mean_workflow.set_distance_squared_max(DUKE_HEART.distancemap_squared_max)
        if icon_weights_path.exists():
            mean_workflow.set_icon_weights_path(str(icon_weights_path))
        mean_result = mean_workflow.process()
        mean_result["mean_surface"].save(str(reference_surface_file))
        settings_file.write_text(
            json.dumps(mean_surface_settings, indent=2), encoding="utf-8"
        )
    reference_surface = pv.read(str(reference_surface_file))

    # Workflow initialization

    workflow = WorkflowCreateStatisticalModel(
        sample_meshes=sample_surfaces,
        reference_mesh=reference_surface,
        number_of_pca_components=number_of_pca_components,
        icp_transform_type=DUKE_HEART.icp_transform_type,
        mask_dilation_mm=DUKE_HEART.mask_dilation_mm,
        distance_squared_max=DUKE_HEART.distancemap_squared_max,
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
    model_file = DUKE_HEART.pca_model_file(test_mode)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("w", encoding="utf-8") as f:
        json.dump(pca_model, f, indent=2)

    mean_surface = result["pca_mean_surface"]
    mean_surface_file = DUKE_HEART.pca_mean_surface_file(test_mode)
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
    # PCA rank is capped by the sample count, so the model can hold fewer
    # components than requested.
    mode_count = min(number_of_pca_components, len(components), len(eigenvalues))

    mode_surface_files: list[Path] = []
    xvfb_started = False
    try:
        pv.start_xvfb()
        xvfb_started = True
    except Exception:
        pass

    try:
        for mode_idx in range(mode_count):
            sigma = float(np.sqrt(eigenvalues[mode_idx]))
            mode_offsets = np.asarray(components[mode_idx]).reshape(-1, 3)

            minus_mesh = mean_surface.copy()
            minus_mesh.points = mean_points - 2.0 * sigma * mode_offsets
            plus_mesh = mean_surface.copy()
            plus_mesh.points = mean_points + 2.0 * sigma * mode_offsets

            minus_file = output_dir / f"pca_mode_{mode_idx + 1:02d}_minus_2sigma.vtp"
            plus_file = output_dir / f"pca_mode_{mode_idx + 1:02d}_plus_2sigma.vtp"
            minus_mesh.save(str(minus_file))
            plus_mesh.save(str(plus_file))
            mode_surface_files.extend([minus_file, plus_file])

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
