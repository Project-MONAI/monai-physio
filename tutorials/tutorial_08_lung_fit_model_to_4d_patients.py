"""
Tutorial 8 (Lung): Fit the Lung SSM and Propagate It Through Respiratory Phases

Purpose
-------
Runs on the public DIR-Lab 4D CT data. For each DIR-Lab case it produces one
statistical-shape-model (SSM) surface per respiratory phase:

1. Fit the lung PCA model to the reference phase (``T70``). Lungs are segmented
   from the reference phase, a surface is extracted, and the PCA model built by
   ``tutorial_06_lung_create_statistical_model.py`` is fitted with PCA-based
   registration (``WorkflowFitStatisticalModelToPatient`` with
   ``use_pca_registration=True``). This yields the case's PCA coefficients plus
   the fitted SSM surface, sharing the model's fixed topology.

2. Propagate the fitted surface to every respiratory phase. Each phase is
   registered to the reference phase with ``RegisterImagesGreedy``
   (``WorkflowReconstructHighres4DCT``). The forward transform for each phase
   warps the fitted SSM surface (``TransformTools.transform_pvcontour``, with
   deformation magnitude attached), producing one ``*_T{PP}_ssm_surface.vtp``
   per phase.

Statistical models are typically dense tetrahedral volume meshes, written as
``.vtu``. The lung PCA model from Tutorial 6 is built from surfaces only, so
every model here is a surface and every model file is a ``.vtp``.

Data Required
-------------
data: ``data/DirLab-4DCT/Case*_T??.mha``
PCA model: Tutorial 6 output (``output/tutorial_06_lung/pca_model.json``,
``pca_mean_surface.vtp``)
ICON weights: Tutorial 2 output
(``network_weights/icon_dirlab_4dct_distancemap/
icon_dirlab_4dct_distancemap_model/checkpoints/network_weights_final.trch``) for
the distance-map stage of the SSM fit. Optional - the stock uniGradICON weights
are used when it is absent.

Outputs (per case, under ``output/tutorial_08_lung/<case>/``)
------------------------------------------------------------
  * ``*_ssm_pca_coefficients.json``                     - fitted PCA coefficient vector
  * ``*_ssm_pca_surface.vtp``                           - PCA template before final warp
  * ``*_ssm_surface.vtp``                               - fitted reference SSM surface
  * ``*_T{PP}_ssm_surface.vtp``                         - SSM warped to phase PP
  * ``*_T{PP}_warped_ref.mha``, ``*_T{PP}_*_tfm.hdf``   - registration artifacts
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import itk
import pyvista as pv
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from monai_physio import (
    ContourTools,
    RegisterImagesGreedy,
    SegmentNVSegmentCTMRI,
    TestTools,
    TransformTools,
    WorkflowConvertImageToVTK,
    WorkflowFitStatisticalModelToPatient,
    WorkflowReconstructHighres4DCT,
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

    class_name = "tutorial_08_lung_fit_model_to_4d_patients"

    test_mode = TestTools.running_as_test()

    data_dir = LUNG_CT_DIRLAB.input_directory(test_mode)

    output_dir = LUNG_CT_DIRLAB.output_directory(test_mode) / "tutorial_08_lung"
    weights_dir = LUNG_CT_DIRLAB.weights_directory(test_mode)

    baselines_dir = repo_root / "tests" / "baselines"

    # PCA model + mean surface produced by Tutorial 6 (lung).
    pca_model_file = LUNG_CT_DIRLAB.pca_model_file(test_mode)
    pca_mean_file = LUNG_CT_DIRLAB.pca_mean_surface_file(test_mode)
    # Tutorial 6 caches one segmentation per case beside its model.
    tutorial_06_dir = pca_model_file.parent

    # Distance-map weights finetuned on DIR-Lab by
    # tutorial_02_lung_distancemap_finetune_icon.py, used by the
    # labelmap-to-labelmap stage of the SSM fit.
    icon_distancemap_weights_path = (
        weights_dir
        / "icon_dirlab_4dct_distancemap"
        / "icon_dirlab_4dct_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    number_of_pca_components = LUNG_CT_DIRLAB.pca_components(test_mode)

    # Phase the SSM is fitted to; Tutorial 6 builds the lung PCA model from the
    # T70 surfaces, so the same phase is used here as the fitting reference.
    reference_phase = "T70"

    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    for required_file in (pca_model_file, pca_mean_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 6 output not found: {required_file}\n"
                "Run tutorials/tutorial_06_lung_create_statistical_model.py first."
            )
    pca_mean_surface = cast(pv.DataSet, pv.read(str(pca_mean_file)))
    with pca_model_file.open(encoding="utf-8") as f:
        pca_model = json.load(f)

    # The Tutorial 2 distance-map weights are used when they exist; without them
    # the tutorial still runs, on the stock uniGradICON weights.
    use_finetuned_distancemap_weights = icon_distancemap_weights_path.exists()
    if use_finetuned_distancemap_weights:
        logger.info(
            "Fitting the SSM with finetuned distance-map ICON weights: %s",
            icon_distancemap_weights_path,
        )
    else:
        logger.warning(
            "Finetuned distance-map ICON weights not found at %s; fitting the SSM "
            "with the stock uniGradICON weights. Run "
            "tutorials/tutorial_02_lung_distancemap_finetune_icon.py to create "
            "them.",
            icon_distancemap_weights_path,
        )

    reference_files = sorted(data_dir.glob(f"Case*_{reference_phase}.mha"))
    if not reference_files:
        raise FileNotFoundError(
            f"No DirLab {reference_phase} images found under {data_dir}.\n"
            "See data/DirLab-4DCT/README.md for download instructions."
        )

    segmentation_method = SegmentNVSegmentCTMRI(log_level=log_level)
    segmentation_workflow = WorkflowConvertImageToVTK(
        segmentation_method=segmentation_method, log_level=log_level
    )
    contour_tools = ContourTools(log_level=log_level)
    transform_tools = TransformTools(log_level=log_level)

    tutorial_results: dict[str, Any] = {"cases": {}, "screenshots": []}

    for reference_file in reference_files:
        case_id = reference_file.name.split("_")[0]
        logger.info("%s", "=" * 48)
        logger.info("Processing case %s", case_id)
        logger.info("%s", "=" * 48)

        case_output_dir = output_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        reference_image = itk.imread(str(reference_file))

        # Step 1: segment the lungs in the reference phase. Tutorial 6 already
        # segmented every Case*_T70 image, so its outputs are reused when present.
        lung_surface_file = tutorial_06_dir / f"{reference_file.stem}.vtp"
        lung_labelmap_file = tutorial_06_dir / f"{reference_file.stem}_labelmap.nii.gz"
        if not (lung_surface_file.exists() and lung_labelmap_file.exists()):
            lung_surface_file = case_output_dir / f"{reference_file.stem}.vtp"
            lung_labelmap_file = (
                case_output_dir / f"{reference_file.stem}_labelmap.nii.gz"
            )
            segmentation_result = segmentation_workflow.process(
                input_image=reference_image,
                anatomy_groups=["lung"],
                surface_reduction_rate=LUNG_CT_DIRLAB.surface_reduction_rate,
                extract_label_surfaces=True,
            )
            contour_tools.save_combined_surfaces(
                segmentation_result["label_surfaces"], str(lung_surface_file)
            )
            itk.imwrite(
                segmentation_result["labelmap"],
                str(lung_labelmap_file),
                compression=True,
            )
        lung_surface = cast(pv.PolyData, pv.read(str(lung_surface_file)))
        lung_labelmap = itk.imread(str(lung_labelmap_file))

        # Step 2: fit the statistical model to the reference phase
        fit_workflow = WorkflowFitStatisticalModelToPatient(
            template_model=pca_mean_surface,
            patient_models=[lung_surface],
            patient_image=reference_image,
            patient_labelmap=lung_labelmap,
            log_level=log_level,
        )
        fit_workflow.set_use_pca_registration(
            use_pca_registration=True,
            pca_model=pca_model,
            number_of_pca_components=number_of_pca_components,
            use_surface=False,
        )
        fit_workflow.set_icp_transform_type(LUNG_CT_DIRLAB.icp_transform_type)
        fit_workflow.set_mask_dilation_mm(LUNG_CT_DIRLAB.mask_dilation_mm)
        fit_workflow.set_distancemap_squared_max(LUNG_CT_DIRLAB.distancemap_squared_max)
        if use_finetuned_distancemap_weights:
            fit_workflow.set_labelmap_to_labelmap_icon_weights_path(
                str(icon_distancemap_weights_path)
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

        # Typically the SSM is a dense tetrahedral volume mesh, saved as .vtu, and
        # its bounding surface is saved separately as .vtp. The lung PCA model from
        # Tutorial 6 is built from surfaces only, so here the model *is* a surface:
        # "fitted_reference_model" and "fitted_reference_mesh" are
        # the same geometry, and only the .vtp surface is written.
        fitted_reference_mesh = fit_result["fitted_reference_mesh"]
        fitted_reference_mesh_file = case_output_dir / f"{case_id}_ssm_surface.vtp"
        fitted_reference_mesh.save(str(fitted_reference_mesh_file))

        # Step 3: register every respiratory phase to the reference phase
        phase_files = sorted(data_dir.glob(f"{case_id}_T??.mha"))
        phase_ids = [path.stem.split("_")[1] for path in phase_files]
        time_series = [itk.imread(str(path)) for path in phase_files]

        registration_method = RegisterImagesGreedy(log_level=log_level)

        reg_workflow = WorkflowReconstructHighres4DCT(
            time_series_images=time_series,
            reference_image=reference_image,
            reference_time_frame=phase_ids.index(reference_phase),
            register_reference_time_frame_to_reference_image=False,
            registration_method=registration_method,
            log_level=log_level,
        )
        reg_workflow.set_modality("ct")
        reg_result = reg_workflow.process()

        # Step 4: warp the fitted SSM surface to every respiratory phase
        phase_outputs = []
        for phase_index, phase_id in enumerate(phase_ids):
            logger.info("Case %s: warping to phase %s", case_id, phase_id)

            itk.imwrite(
                reg_result["reconstructed_images"][phase_index],
                str(case_output_dir / f"{case_id}_{phase_id}_warped_ref.mha"),
                compression=True,
            )

            fixed_to_moving_transform = reg_result["fixed_to_moving_transforms"][
                phase_index
            ]
            itk.transformwrite(
                fixed_to_moving_transform,
                str(case_output_dir / f"{case_id}_{phase_id}_forward_tfm.hdf"),
            )
            itk.transformwrite(
                reg_result["moving_to_fixed_transforms"][phase_index],
                str(case_output_dir / f"{case_id}_{phase_id}_inverse_tfm.hdf"),
            )

            surface = transform_tools.transform_pvcontour(
                fitted_reference_mesh,
                fixed_to_moving_transform,
                with_deformation_magnitude=True,
            )
            surface_file = case_output_dir / f"{case_id}_{phase_id}_ssm_surface.vtp"
            surface.save(str(surface_file))

            phase_outputs.append({"phase_id": phase_id, "surface_file": surface_file})

        tutorial_results["cases"][case_id] = {
            "pca_coefficients_file": pca_coefficients_file,
            "fitted_reference_mesh_file": fitted_reference_mesh_file,
            "phase_outputs": phase_outputs,
        }

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    last_case = tutorial_results["cases"][case_id]
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
