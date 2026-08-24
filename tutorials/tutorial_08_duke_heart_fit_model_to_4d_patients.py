"""
Tutorial 8 (Duke Heart): Fit the Heart SSM and Propagate It Through Cardiac Phases

Purpose
-------
Duke counterpart of ``tutorial_08_lung_fit_model_to_4d_patients.py``, run on the
Duke-Heart-4DLabelmaps cohort.  For each case it produces one
statistical-shape-model (SSM) surface per gated cardiac frame:

1. Fit the heart PCA model to the case's reference frame
   (``*_ref_labelmap.nii.gz``).  The whole heart minus its chamber cavities is
   contoured from that frame and the PCA model built by
   ``tutorial_06_duke_heart_create_statistical_model.py`` is fitted to it with
   PCA-based registration (``WorkflowFitStatisticalModelToPatient`` with
   ``use_pca_registration=True``).  This yields the case's PCA coefficients plus
   the fitted SSM surface, sharing the model's fixed topology.

2. Propagate that surface to every gated frame.  This data ships labelmaps
   rather than CT, so there is no intensity image to register: each frame's
   heart surface is contoured the same way and the fitted SSM surface is
   registered to it with ``RegisterModelsDistanceMaps`` (Greedy affine, then
   ICON on the distance maps), which warps the SSM while keeping its topology.
   The reference frame keeps the fitted surface itself rather than being
   registered to its own contour.

Every frame's distance maps are rasterized on one grid built around the
reference frame's heart, so the phases of a case are registered in a common
space even though their labelmaps carry different slice pitches.

Statistical models are typically dense tetrahedral volume meshes, written as
``.vtu``.  The heart PCA model from Tutorial 6 (Duke Heart) is built from
surfaces only, so every model here is a surface and every model file is a
``.vtp``.

Data Required
-------------
data: ``data/Duke-Heart-4DLabelmaps/pm????/*_labelmap.nii.gz``
PCA model: Tutorial 6 (Duke Heart) output
(``output/tutorial_06_duke_heart/pca_model.json``, ``pca_mean_surface.vtp``)
Surfaces: Tutorial 4 (Duke Heart) ``outputs = "full"`` output, optional -- any
frame it did not contour is contoured here instead.
ICON weights: Tutorial 2 (Duke Heart) output
(``network_weights/icon_duke_heart_distancemap/
icon_duke_heart_distancemap_model/checkpoints/network_weights_final.trch``),
optional -- the stock uniGradICON weights are used when it is absent.

Outputs (per case, under ``output/tutorial_08_duke_heart/<case>/``)
------------------------------------------------------------------
  * ``<case>_ssm_pca_coefficients.json``  - fitted PCA coefficient vector
  * ``<case>_ssm_pca_surface.vtp``        - PCA template before the final warp
  * ``<case>_ssm_surface.vtp``            - fitted reference-frame SSM surface
  * ``<frame_stem>_ssm_surface.vtp``      - SSM warped to that gated frame
  * ``<frame_stem>_heart_surface.vtp``    - contoured frame surface, cached
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from physiotwin4d import (
    ContourTools,
    RegisterModelsDistanceMaps,
    TestTools,
    WorkflowFitStatisticalModelToPatient,
)

# Structure name Tutorial 4 (Duke Heart) writes its whole-heart surfaces under.
WHOLE_HEART_NAME = "heart_minus_interior_chambers"
LABELMAP_SUFFIX = "_labelmap.nii.gz"

# Only run if this script is not imported as a module

# The registration backends spawn worker processes.  On Windows the spawn start
# method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the whole cohort in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    class_name = "tutorial_08_duke_heart_fit_model_to_4d_patients"

    output_dir = tutorials_dir / "output" / "tutorial_08_duke_heart"
    baselines_dir = repo_root / "tests" / "baselines"

    test_mode = TestTools.running_as_test()
    # The gated labelmaps, one directory per case.
    data_dir = DUKE_HEART.hold_out_directory(test_mode)
    # Tutorial 4's surfaces, read when its "full" pass wrote the frame.
    tutorial_04_dir = DUKE_HEART.input_directory(test_mode)

    # PCA model + mean surface produced by Tutorial 6 (Duke Heart).
    pca_model_file = DUKE_HEART.pca_json_file
    pca_mean_file = DUKE_HEART.pca_mean_file

    number_of_pca_components = DUKE_HEART.pca_components(test_mode)

    # Labels left out of the whole-heart structure, the same ones Tutorials 4,
    # 6 and 7 drop, so the frames and the model describe the same structure.
    interior_object_ids = DUKE_HEART.interior_object_ids

    # Contouring grid, shared with Tutorial 4 so every surface here carries the
    # same level of detail as the model's training surfaces.
    surface_spacing_mm = DUKE_HEART.surface_spacing_mm
    smoothing_iterations = DUKE_HEART.surface_smoothing_iterations

    # Pitch of the grid the phase distance maps are rasterized on.  Coarser than
    # the contouring pitch: it carries a distance field, not a boundary.
    registration_spacing_mm = 1.0

    # Distance-map weights finetuned by
    # tutorial_02_duke_heart_distancemap_finetune_icon.py, used both by the
    # labelmap-to-labelmap stage of the SSM fit and by the phase registrations.
    icon_weights_path = (
        tutorials_dir
        / "network_weights"
        / "icon_duke_heart_distancemap"
        / "icon_duke_heart_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    for required_file in (pca_model_file, pca_mean_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 6 output not found: {required_file}\n"
                "Run tutorials/tutorial_06_duke_heart_create_statistical_model.py "
                "first."
            )
    pca_mean_surface = cast(pv.DataSet, pv.read(str(pca_mean_file)))
    with pca_model_file.open(encoding="utf-8") as f:
        pca_model = json.load(f)

    # The Tutorial 2 distance-map weights are used when they exist; without them
    # the tutorial still runs, on the stock uniGradICON weights.
    use_finetuned_weights = icon_weights_path.exists()
    if use_finetuned_weights:
        logger.info(
            "Registering with finetuned distance-map ICON weights: %s",
            icon_weights_path,
        )
    else:
        logger.warning(
            "Finetuned distance-map ICON weights not found at %s; registering "
            "with the stock uniGradICON weights. Run "
            "tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py to "
            "create them.",
            icon_weights_path,
        )

    case_dirs = sorted(
        path for path in data_dir.glob("pm[0-9][0-9][0-9][0-9]") if path.is_dir()
    )
    if not case_dirs:
        raise FileNotFoundError(
            f"No pm???? case directories found under {data_dir}.\n"
            "See data/Duke-Heart-4DLabelmaps/README.md."
        )

    contour_tools = ContourTools(log_level=log_level)

    def heart_surface_for(labelmap_file: Path, case_output_dir: Path) -> pv.PolyData:
        """Return one frame's whole heart, minus its chamber cavities.

        Tutorial 4's ``"full"`` pass contours this surface for every gated
        frame, so its output is read when present.  Otherwise the surface is
        contoured here and cached beside this tutorial's other outputs, so a
        re-run pays for it once.
        """
        stem = labelmap_file.name[: -len(LABELMAP_SUFFIX)]
        tutorial_04_file = tutorial_04_dir / f"{stem}_{WHOLE_HEART_NAME}.vtp"
        if tutorial_04_file.exists():
            return cast(pv.PolyData, pv.read(str(tutorial_04_file)))

        surface_file = case_output_dir / f"{stem}_heart_surface.vtp"
        if not surface_file.exists():
            labelmap = itk.imread(str(labelmap_file))
            labels = itk.GetArrayViewFromImage(labelmap)
            heart_ids = [
                int(value)
                for value in np.unique(labels)
                if value != 0 and int(value) not in interior_object_ids
            ]
            heart_mask = itk.GetImageFromArray(
                np.isin(labels, heart_ids).astype(np.uint8)
            )
            heart_mask.CopyInformation(labelmap)
            heart_surface = contour_tools.extract_label_surfaces(
                heart_mask,
                isotropic_spacing_mm=surface_spacing_mm,
                smoothing_iterations=smoothing_iterations,
            )[1]
            heart_surface.save(str(surface_file))
        return cast(pv.PolyData, pv.read(str(surface_file)))

    tutorial_results: dict[str, Any] = {"cases": {}, "screenshots": []}

    for case_dir in case_dirs:
        case_id = case_dir.name
        frame_files = sorted(case_dir.glob(f"*{LABELMAP_SUFFIX}"))
        reference_files = [
            path for path in frame_files if path.name.endswith(f"_ref{LABELMAP_SUFFIX}")
        ]
        if not reference_files:
            logger.warning("Skipping %s: no *_ref_labelmap.nii.gz frame", case_id)
            continue
        reference_file = reference_files[0]

        logger.info("%s", "=" * 48)
        logger.info("Processing case %s: %d gated frames", case_id, len(frame_files))
        logger.info("%s", "=" * 48)

        case_output_dir = output_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: contour the reference frame's heart.
        reference_labelmap = itk.imread(str(reference_file))
        reference_surface = heart_surface_for(reference_file, case_output_dir)

        # Step 2: fit the statistical model to the reference frame.  This data
        # carries no intensity image, so the workflow rasterizes its own
        # reference grid from the patient surface.
        fit_workflow = WorkflowFitStatisticalModelToPatient(
            template_model=pca_mean_surface,
            patient_models=[reference_surface],
            patient_image=None,
            patient_labelmap=reference_labelmap,
            labelmap_interior_object_ids=interior_object_ids,
            log_level=log_level,
        )
        fit_workflow.set_use_pca_registration(
            use_pca_registration=True,
            pca_model=pca_model,
            number_of_pca_components=number_of_pca_components,
            use_surface=False,
        )
        fit_workflow.set_icp_transform_type(DUKE_HEART.icp_transform_type)
        fit_workflow.set_mask_dilation_mm(DUKE_HEART.mask_dilation_mm)
        fit_workflow.set_distancemap_squared_max(DUKE_HEART.distancemap_squared_max)
        if use_finetuned_weights:
            fit_workflow.set_labelmap_to_labelmap_icon_weights_path(
                str(icon_weights_path)
            )
        fit_result = fit_workflow.process()

        pca_coefficients_file = case_output_dir / f"{case_id}_ssm_pca_coefficients.json"
        pca_coefficients = fit_workflow.pca_coefficients
        assert pca_coefficients is not None
        with pca_coefficients_file.open(mode="w", encoding="utf-8") as f:
            json.dump(pca_coefficients.tolist(), f)

        pca_template_surface = fit_workflow.pca_template_model_surface
        assert pca_template_surface is not None
        pca_template_surface.save(
            str(case_output_dir / f"{case_id}_ssm_pca_surface.vtp")
        )

        # Typically the SSM is a dense tetrahedral volume mesh, saved as .vtu,
        # and its bounding surface is saved separately as .vtp.  The heart PCA
        # model from Tutorial 6 (Duke Heart) is built from surfaces only, so
        # here the model *is* a surface: "fitted_reference_model" and
        # "fitted_reference_mesh" are the same geometry, and only
        # the .vtp surface is written.
        fitted_reference_mesh = fit_result["fitted_reference_mesh"]
        fitted_reference_mesh_file = case_output_dir / f"{case_id}_ssm_surface.vtp"
        fitted_reference_mesh.save(str(fitted_reference_mesh_file))

        # Step 3: warp the fitted SSM surface onto every gated frame.  One grid
        # is built around the reference frame's heart and reused by every frame,
        # so the whole case is registered in a common space; its buffer holds
        # the frames the heart moves into.
        registration_grid = contour_tools.create_reference_image(
            mesh=reference_surface,
            spatial_resolution=registration_spacing_mm,
            buffer_factor=0.25,
            ptype=itk.F,
        )

        phase_outputs = []
        for frame_file in frame_files:
            stem = frame_file.name[: -len(LABELMAP_SUFFIX)]
            if frame_file == reference_file:
                # The fit already placed the SSM on this frame.
                logger.info("Case %s: reference frame %s", case_id, stem)
                phase_surface = fitted_reference_mesh
            else:
                logger.info("Case %s: warping to frame %s", case_id, stem)
                registrar = RegisterModelsDistanceMaps(
                    moving_model=fitted_reference_mesh,
                    fixed_model=heart_surface_for(frame_file, case_output_dir),
                    reference_image=registration_grid,
                    distance_squared_max=DUKE_HEART.distancemap_squared_max,
                    mask_dilation_mm=DUKE_HEART.mask_dilation_mm,
                    log_level=log_level,
                )
                if use_finetuned_weights:
                    registrar.set_icon_weights_path(str(icon_weights_path))
                phase_surface = registrar.register(transform_type="Deformable")[
                    "registered_model"
                ]

            surface_file = case_output_dir / f"{stem}_ssm_surface.vtp"
            phase_surface.save(str(surface_file))
            phase_outputs.append({"frame_stem": stem, "surface_file": surface_file})

        tutorial_results["cases"][case_id] = {
            "pca_coefficients_file": pca_coefficients_file,
            "fitted_reference_mesh_file": fitted_reference_mesh_file,
            "phase_outputs": phase_outputs,
        }

    if not tutorial_results["cases"]:
        raise RuntimeError(
            f"No case under {data_dir} carried a reference frame; nothing was fitted."
        )

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    last_case = list(tutorial_results["cases"].values())[-1]
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(last_case["fitted_reference_mesh_file"]))),
            "ssm_surface_reference.png",
            camera_position="iso",
            color="steelblue",
            opacity=0.9,
        ),
        tt.save_screenshot_mesh(
            cast(
                pv.DataSet,
                pv.read(str(last_case["phase_outputs"][0]["surface_file"])),
            ),
            "ssm_surface_first_phase.png",
            camera_position="iso",
            color="limegreen",
            opacity=0.9,
        ),
    ]
