"""
Tutorial 6 (Lung): Create a PCA Statistical Shape Model

Purpose
-------
Build a PCA statistical shape model of the lungs from the TCIA-4DLung
population, less ``ParametersTCIALung4DCT.hold_out_case``, which Tutorial 7 fits
it to, and ``mgn_hold_out_case``, which Tutorials 9-14 predict. Each case's g070 phase is segmented, an unbiased mean surface is built
with ``WorkflowCreateMeanSurface``, and the population is decomposed into
shape modes. Tutorials 7 and 8 reuse the saved ``pca_model.json``.

Data Required
-------------
Full data: ``data/TCIA-4DLung/*_HM10395/*_HM10395_g070.nii.gz``
TCIA-4DLung is not auto-downloaded - see ``data/TCIA-4DLung/README.md``.

Outputs
-------
Under ``tutorials/output/tutorial_06_create_statistical_model_tcia_lung_4d_ct/``:

- ``<case>_T70.vtp`` / ``<case>_T70_labelmap.nii.gz`` - per-case segmentations,
  cached and reused by Tutorial 8
- ``reference_mean_surface.vtp`` - the unbiased atlas surface
- ``pca_model.json`` and ``pca_mean_surface.vtp`` - the shape model
- ``pca_mode_<k>_{minus,plus}_2sigma.vtp`` and ``pca_mode_<k>.png``

Runtime
-------
One GPU segmentation per case, then ``mean_surface_iterations`` deformable
registrations per case to build the atlas. This is the slowest of Tutorials
1-7; every intermediate is cached on disk, so a re-run is cheap.
"""

# Imports
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pyvista as pv
from parameters_tcia_lung_4d_ct import TCIA_LUNG_4D_CT

from monai_physio import (
    ProcessImages,
    ProcessSurfaceMesh,
    ProcessTests,
    WorkflowConvertImageToVTK,
    WorkflowCreateMeanSurface,
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

    class_name = "tutorial_06_create_statistical_model_tcia_lung_4d_ct"

    test_mode = ProcessTests.running_as_test()

    output_dir = (
        TCIA_LUNG_4D_CT.output_directory(test_mode)
        / "tutorial_06_create_statistical_model_tcia_lung_4d_ct"
    )
    weights_dir = TCIA_LUNG_4D_CT.weights_directory(test_mode)

    baselines_dir = repo_root / "tests" / "baselines"

    number_of_pca_components = TCIA_LUNG_4D_CT.pca_components(test_mode)

    # Atlas iterations used to build the reference surface; 1 is a single
    # template-biased pass.
    mean_surface_iterations = 1 if test_mode else 3

    # Points kept per surface; 0 keeps every point.  The lung surfaces feed
    # both the atlas below and the model after it, so reducing them here cuts
    # the cost of each.
    model_points = TCIA_LUNG_4D_CT.points_per_model(test_mode)

    # Distance-map weights finetuned by
    # tutorial_02_finetune_icon_distancemap_tcia_lung_4d_ct.py.  Stock
    # uniGradICON weights are out of distribution for distance maps, so without
    # these the correspondences this model is built from barely move off the
    # template, and the modes come out far too tight.  Tutorial 7 fits with the
    # same checkpoint.
    icon_weights_path = (
        weights_dir
        / "icon_tcia_4dlung_distancemap"
        / "icon_tcia_4dlung_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    log_level = logging.INFO

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create lung surface files
    segmentation_method = TCIA_LUNG_4D_CT.segmenter(test_mode, log_level=log_level)
    workflow_method = WorkflowConvertImageToVTK(
        segmentation_method=segmentation_method, log_level=log_level
    )

    contour_tools = ProcessSurfaceMesh(log_level=log_level)
    image_tools = ProcessImages(log_level=log_level)

    # Tutorial 7 fits this model to the held-out study and Tutorials 10-14
    # predict the held-out case, so the model must not have seen either.
    held_out = {TCIA_LUNG_4D_CT.hold_out_case, TCIA_LUNG_4D_CT.mgn_hold_out_case}
    sample_image_files = [
        case_dir / f"{case_dir.name}_g070.nii.gz"
        for case_dir in TCIA_LUNG_4D_CT.case_directories(test_mode)
        if case_dir.name not in held_out
        and (case_dir / f"{case_dir.name}_g070.nii.gz").exists()
    ]
    sample_surfaces: list[pv.DataSet] = []
    for sample_image_file in sample_image_files:
        # ``.stem`` only strips ``.gz``, leaving a stray ``.nii`` in the name,
        # since these are ``.nii.gz`` (TCIA) rather than ``.mha`` (DIR-Lab).
        sample_image_stem = sample_image_file.name.removesuffix(".nii.gz")
        sample_surface_file = output_dir / f"{sample_image_stem}.vtp"
        # Cached for Tutorials 8 and 15, which read these instead of segmenting.
        if not sample_surface_file.exists():
            input_image = itk.imread(str(sample_image_file))
            if (
                TCIA_LUNG_4D_CT.mesh_element_size_mm != 0
                and input_image.GetSpacing()[0] != TCIA_LUNG_4D_CT.mesh_element_size_mm
            ):
                input_image = image_tools.make_isotropic_image(
                    input_image, TCIA_LUNG_4D_CT.mesh_element_size_mm
                )
            result = workflow_method.process(
                input_image=input_image,
                anatomy_groups=[TCIA_LUNG_4D_CT.anatomy_group],
                surface_reduction_rate=TCIA_LUNG_4D_CT.surface_reduction_rate,
                extract_label_surfaces=True,
                extract_group_surfaces=False,
            )
            WorkflowConvertImageToVTK.save_combined_surfaces(
                result["label_surfaces"], str(sample_surface_file)
            )
            sample_labelmap_file = output_dir / f"{sample_image_stem}_labelmap.nii.gz"
            itk.imwrite(result["labelmap"], str(sample_labelmap_file), compression=True)
        sample_surface = cast(pv.PolyData, pv.read(str(sample_surface_file)))
        if model_points:
            sample_surface = contour_tools.remesh_and_smooth_surface(
                sample_surface, 1.0 - model_points / sample_surface.n_points, 0
            )
        sample_surfaces.append(sample_surface)

    # The reference surface defines the topology every PCA input is expressed
    # in, so picking one case makes the model inherit that case's shape. Use the
    # unbiased mean of the population instead. Cached: it costs one deformable
    # registration per case per atlas iteration.
    reference_surface_file = output_dir / "reference_mean_surface.vtp"
    starting_case = TCIA_LUNG_4D_CT.mean_surface_starting_case(test_mode).name
    sample_case_ids = [path.parent.name for path in sample_image_files]
    # Keyed on the cases and settings the atlas was built from, not on the file
    # merely being there: an atlas from another population, or corresponded at
    # other settings than the model below, would disagree with it silently.
    mean_surface_settings = {
        "cases": sample_case_ids,
        "starting_case": starting_case,
        "model_points": model_points,
        "iterations": mean_surface_iterations,
        "mask_dilation_mm": TCIA_LUNG_4D_CT.mask_dilation_mm,
        "distance_squared_max": TCIA_LUNG_4D_CT.distancemap_squared_max,
        "icp_transform_type": TCIA_LUNG_4D_CT.icp_transform_type,
        "spatial_resolution": TCIA_LUNG_4D_CT.mesh_element_size_mm,
        "greedy_iterations": TCIA_LUNG_4D_CT.number_of_iterations_greedy,
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
            template_surface=sample_surfaces[sample_case_ids.index(starting_case)],
            log_level=log_level,
        )
        mean_workflow.set_number_of_iterations(mean_surface_iterations)
        mean_workflow.set_mask_dilation_mm(TCIA_LUNG_4D_CT.mask_dilation_mm)
        mean_workflow.set_distance_squared_max(TCIA_LUNG_4D_CT.distancemap_squared_max)
        mean_workflow.set_icp_transform_type(TCIA_LUNG_4D_CT.icp_transform_type)
        mean_workflow.set_distance_map_transform_type("Affine")
        mean_workflow.spatial_resolution = TCIA_LUNG_4D_CT.mesh_element_size_mm
        mean_workflow.set_greedy_iterations(TCIA_LUNG_4D_CT.number_of_iterations_greedy)
        mean_workflow.set_icon_iterations(TCIA_LUNG_4D_CT.number_of_iterations_icon)

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
        # The distance maps step 3 registers are rasterized at this resolution,
        # and generating, dilating and affinely registering them is what the
        # step costs.  2 mm is an eighth of the voxels of the 1 mm default.
        reference_spatial_resolution=3.0 if test_mode else 2.0,
        icp_transform_type=TCIA_LUNG_4D_CT.icp_transform_type,
        mask_dilation_mm=TCIA_LUNG_4D_CT.mask_dilation_mm,
        distance_squared_max=TCIA_LUNG_4D_CT.distancemap_squared_max,
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
            "tutorials/tutorial_02_finetune_icon_distancemap_tcia_lung_4d_ct.py "
            "to create them.",
            icon_weights_path,
        )

    # Workflow execution
    result = workflow.process()

    # Result saving
    pca_model: dict[str, Any] = result["pca_model"]
    model_file = TCIA_LUNG_4D_CT.pca_model_file(test_mode)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("w", encoding="utf-8") as f:
        json.dump(pca_model, f, indent=2)

    mean_surface = result["pca_mean_surface"]
    mean_surface_file = TCIA_LUNG_4D_CT.pca_mean_surface_file(test_mode)
    mean_surface.save(str(mean_surface_file))

    # Testing
    tt = ProcessTests(
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
    with contextlib.suppress(Exception):
        pv.start_xvfb()
        xvfb_started = True

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
            with contextlib.suppress(Exception):
                pv.stop_xvfb()

    tutorial_results = {
        "pca_model": pca_model,
        "mean_surface": mean_surface,
        "model_file": model_file,
        "mean_surface_file": mean_surface_file,
        "mode_surface_files": mode_surface_files,
        "screenshots": screenshots,
    }
