"""
Heart 3D to CSV: per-point cardiac displacement table for the held-out Duke case

Purpose
-------
Fits the heart PCA model to the held-out Duke case's reference frame, predicts
the cardiac cycle with the trained MeshGraphNet, and writes the motion as a
table instead of as per-stage surfaces or USD.

The fit is the one Tutorial 12 performs.  The prediction is
:meth:`WorkflowInferPhysicsNeMo.predict` called directly, which returns the
per-point displacement in mm from the fitted reference mesh without writing
any geometry.

Data Required
-------------
  * ``data/Duke-Heart-4DLabelmaps/<hold_out_case>/*_ref_labelmap.nii.gz``
  * ``network_weights/physicsnemo_mgn_duke_heart_motion/`` - Tutorial 9
    checkpoint with its ``pca_model.json`` and ``pca_mean_surface.vtp``
  * ``network_weights/icon_duke_heart_distancemap/`` - Tutorial 2 weights,
    optional

Outputs
-------
Under ``results/<case>/`` beside this script:

  * ``<case>_ssm_pca_coefficients.json`` - the case's shape parameters
  * ``<case>_ssm_surface.vtp``           - the model fitted to the reference frame
  * ``<case>_s{TTT}_pred.vtp``           - fitted surface displaced to each of the
    10 stages
  * ``<case>_displacements.csv``       - one row per mesh point; ``point_index``
    then ``s{TTT}_dx_mm, s{TTT}_dy_mm, s{TTT}_dz_mm`` for each of the 10 stages
"""

# Imports
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional, cast

import itk
import numpy as np
import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tutorials"))

from parameters_duke_heart_4d_labelmaps import DUKE_HEART_4D_LABELMAPS

from monai_physio import (
    MONAIPhysioBase,
    WorkflowConvertImageToVTK,
    WorkflowFitStatisticalModelToPatient,
    WorkflowInferPhysicsNeMo,
)

NUMBER_OF_STAGES = 10

# The registration backends and torch spawn worker processes. On Windows the
# spawn start method re-imports this script in each child; the
# __name__ == "__main__" guard keeps that from restarting the pipeline.
if __name__ == "__main__":
    case_id = DUKE_HEART_4D_LABELMAPS.hold_out_case
    test_mode = False
    log_level = logging.INFO
    logger = MONAIPhysioBase(class_name="heart_3dct_to_csv", log_level=log_level)

    model_dir = DUKE_HEART_4D_LABELMAPS.mgn_weights_directory(test_mode)
    epoch: Optional[int] = None
    pca_model_file = model_dir / "pca_model.json"
    pca_mean_file = model_dir / "pca_mean_surface.vtp"
    icon_weights_path = DUKE_HEART_4D_LABELMAPS.distancemap_icon_weights_file(test_mode)
    number_of_pca_components = DUKE_HEART_4D_LABELMAPS.pca_components(test_mode)

    output_dir = Path(__file__).resolve().parent / "results" / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for required_file in (
        pca_model_file,
        pca_mean_file,
        model_dir / "mgn_stage_model.pt",
    ):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required input not found: {required_file}\n"
                "Run tutorials/tutorial_09_train_physicsnemo_mgn_duke_heart_4d_"
                "labelmaps.py first, or download the pretrained network with "
                "monai-physio-download-data PhysicsNeMo-MGN-Duke-Heart-Motion."
            )

    labelmap_dir = DUKE_HEART_4D_LABELMAPS.hold_out_directory(test_mode) / case_id
    reference_files = sorted(labelmap_dir.glob("*_ref_labelmap.nii.gz"))
    if not reference_files:
        raise FileNotFoundError(
            f"No *_ref_labelmap.nii.gz frame in {labelmap_dir}.\n"
            "See data/Duke-Heart-4DLabelmaps/README.md."
        )
    reference_labelmap = itk.imread(str(reference_files[0]))

    pca_mean_surface = cast(pv.DataSet, pv.read(str(pca_mean_file)))
    with pca_model_file.open(encoding="utf-8") as f:
        pca_model = json.load(f)

    use_finetuned_weights = icon_weights_path.exists()
    if not use_finetuned_weights:
        logger.log_warning(
            "Finetuned distance-map ICON weights not found at %s; fitting the SSM "
            "with the stock uniGradICON weights.",
            icon_weights_path,
        )

    # Contour the reference frame's heart, minus the cavities and vessels the
    # shape model leaves out.
    heart_surface = WorkflowConvertImageToVTK(
        DUKE_HEART_4D_LABELMAPS.segmenter(test_mode, log_level), log_level=log_level
    ).process(
        labelmap_image=reference_labelmap,
        groups=DUKE_HEART_4D_LABELMAPS.surface_groups(reference_labelmap),
        isotropic_spacing_mm=DUKE_HEART_4D_LABELMAPS.surface_spacing_mm,
        smoothing_iterations=DUKE_HEART_4D_LABELMAPS.surface_smoothing_iterations,
    )["surfaces"][DUKE_HEART_4D_LABELMAPS.whole_heart_name]

    # Fit the shape model to the reference frame.
    fit_workflow = WorkflowFitStatisticalModelToPatient(
        template_model=pca_mean_surface,
        patient_models=[heart_surface],
        patient_image=None,
        patient_labelmap=reference_labelmap,
        labelmap_interior_object_ids=DUKE_HEART_4D_LABELMAPS.interior_object_ids,
        log_level=log_level,
    )
    fit_workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model=pca_model,
        number_of_pca_components=number_of_pca_components,
        use_surface=False,
    )
    fit_workflow.set_icp_transform_type(DUKE_HEART_4D_LABELMAPS.icp_transform_type)
    fit_workflow.set_mask_dilation_mm(DUKE_HEART_4D_LABELMAPS.mask_dilation_mm)
    fit_workflow.set_distancemap_squared_max(
        DUKE_HEART_4D_LABELMAPS.distancemap_squared_max
    )
    if use_finetuned_weights:
        fit_workflow.set_labelmap_to_labelmap_icon_weights_path(str(icon_weights_path))
    fit_result = fit_workflow.process()

    pca_coefficients = fit_workflow.pca_coefficients
    assert pca_coefficients is not None
    with (output_dir / f"{case_id}_ssm_pca_coefficients.json").open(
        mode="w", encoding="utf-8"
    ) as f:
        json.dump(pca_coefficients.tolist(), f)

    fitted_reference_mesh = fit_result["fitted_reference_mesh"]
    fitted_reference_mesh.save(str(output_dir / f"{case_id}_ssm_surface.vtp"))
    logger.log_info("Fitted the heart model to %s", reference_files[0].name)

    # Predict the displacement of every point at 10 evenly spaced stages of the
    # cardiac cycle: gates g000..g090 of the acquisition.
    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir, epoch=epoch, log_level=log_level
    )
    infer_workflow.check_held_out(case_id)
    n_points = fitted_reference_mesh.n_points
    if n_points != infer_workflow.template_mesh.n_points:
        raise ValueError(
            f"Fitted mesh has {n_points} points, expected "
            f"{infer_workflow.template_mesh.n_points} (template topology)."
        )

    stages = [index / NUMBER_OF_STAGES for index in range(NUMBER_OF_STAGES)]
    coefficients = np.asarray(pca_coefficients, dtype=np.float32)
    columns = {"point_index": np.arange(n_points)}
    for stage in stages:
        displacement = infer_workflow.predict(coefficients, stage)
        tag = f"s{round(stage * 100):03d}"
        stage_mesh = fitted_reference_mesh.copy(deep=True)
        stage_mesh.points = (
            np.asarray(fitted_reference_mesh.points, dtype=np.float32) + displacement
        )
        stage_mesh.save(str(output_dir / f"{case_id}_{tag}_pred.vtp"))
        for axis, name in enumerate(("dx", "dy", "dz")):
            columns[f"{tag}_{name}_mm"] = displacement[:, axis]

    csv_file = output_dir / f"{case_id}_displacements.csv"
    np.savetxt(
        csv_file,
        np.column_stack(list(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt=["%d"] + ["%.6f"] * (len(columns) - 1),
    )
    logger.log_info(
        "Displacement CSV: %s (%d points x %d stages)",
        csv_file,
        n_points,
        len(stages),
    )
