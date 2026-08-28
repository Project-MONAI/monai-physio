"""
Tutorial 18 (Duke Heart, PINN): Predict Myocardial Motion and the Stress It Implies

Purpose
-------
Score the mechanics-aware surrogate Tutorial 17 trained, and turn what it
predicts into something a mechanics-aware model can say and a data-only one
cannot: a stress field.

1. Predict the held-out case's motion with the physics-informed network, and
   again with the ``lambda_physics = 0`` ablation Tutorial 17 trained on
   identical data.  That ablation is the only honest comparison -- scoring
   against Tutorial 9 would confound the physics term with the change from a
   surface shape model to a volumetric one.
2. Score both with ``WorkflowEvaluateMovement`` and write one table comparing
   them per phase, including the minimum Jacobian, which is what says whether
   the predicted motion ever turns tissue inside out.
3. Derive the Cauchy stress each predicted deformation implies, from the same
   neo-Hookean law the training loss used, and reduce it to von Mises stress.
4. Export the animated result to USD with a stress colormap.

The success criterion worth holding this to is *not* that the physics-informed
model wins on RMSE.  It is that it is no worse while keeping every element's
Jacobian positive: a strain energy is a prior, and a prior that improved the
data fit would be suspicious.  Read ``mechanics_comparison.csv`` and report what
it says.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric::

    pip install "physiotwin4d[physicsnemo]"
    pip install torch-geometric

A CUDA GPU is required; a CPU-only run is not a supported
configuration.

Data Required
-------------
Tutorial 16 output: the fitted models, manifests and template
Tutorial 17 output: the trained networks
Labelmaps: ``data/Duke-Heart-4DLabelmaps/<case>/*_labelmap.nii.gz``

Outputs (under ``output/tutorial_18_duke_heart_physics_informed_motion/<case>/``)
--------------------------------------------------------------------------------
  * ``mechanics_comparison.csv``          - per-phase scores, both models
  * ``physics_informed/``, ``ablation/``  - each model's predictions and report
  * ``stress/<frame>_stress.vtu``         - predicted motion carrying stress
  * ``heart_physics_informed_motion.usd`` - animated, colored by von Mises stress

Cost
----
Two inference passes and two evaluations over the held-out case's gated frames,
then one stress evaluation per frame.  Far cheaper than Tutorials 16 and 17.
"""

# Imports
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv
from parameters_duke_heart_physics_informed import DUKE_HEART_PHYSICS_INFORMED

from physiotwin4d import (
    ConvertVTKToUSD,
    EvaluateMovementDukeHeart,
    TestTools,
    WorkflowEvaluateMovement,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)
from physiotwin4d.train_physicsnemo_physics_informed_motion import (
    NeoHookeanResidual,
    compute_deformation_gradient,
    tet_volumes,
)

# Nine-component point-data array ``ConvertVTKToUSD.compute_von_mises_stress``
# reduces to the scalar the USD is colored by.
STRESS_ARRAY = "stress"

# Range the stress colormap is stretched over, in kilopascals.  Fixed rather
# than per-frame, so a color means the same thing throughout the animation.
STRESS_RANGE_KPA = (0.0, 50.0)


def _stress_at_points(
    reference_mesh: pv.DataSet,
    predicted_mesh: pv.DataSet,
    tets: np.ndarray,
    law: NeoHookeanResidual,
) -> np.ndarray:
    """Return the ``(n_points, 9)`` Cauchy stress of a predicted deformation.

    The deformation gradient is constant within a tetrahedron, so stress is
    computed per element and then averaged onto the nodes weighted by element
    volume.  Nodes are what carries it because that is where a colormap reads
    it from.
    """
    import torch

    reference_points = torch.tensor(
        np.asarray(reference_mesh.points), dtype=torch.float64
    )
    displacement = torch.tensor(
        np.asarray(predicted_mesh.points) - np.asarray(reference_mesh.points),
        dtype=torch.float64,
    )
    element_tets = torch.tensor(tets, dtype=torch.int64)
    deformation_gradient = compute_deformation_gradient(
        reference_points, displacement, element_tets
    )
    element_stress = law.cauchy_stress(deformation_gradient).reshape(-1, 9).numpy()

    element_volumes, _ = tet_volumes(np.asarray(reference_mesh.points), tets)
    weights = np.repeat(element_volumes, 4)
    nodal_stress = np.zeros((reference_mesh.n_points, 9), dtype=np.float64)
    nodal_weight = np.zeros(reference_mesh.n_points, dtype=np.float64)
    flat_nodes = tets.ravel()
    np.add.at(
        nodal_stress,
        flat_nodes,
        np.repeat(element_stress, 4, axis=0) * weights[:, None],
    )
    np.add.at(nodal_weight, flat_nodes, weights)
    return nodal_stress / np.maximum(nodal_weight, 1.0e-12)[:, None]


def _render_stress(surface: pv.DataSet, plot_file: Path) -> Path:
    """Render a surface colored by von Mises stress and return the written path.

    ``TestTools.save_screenshot_mesh`` paints a mesh one flat color, so a scalar
    field needs its own plotter.
    """
    xvfb_started = False
    try:
        pv.start_xvfb()
        xvfb_started = True
    except Exception:
        pass
    try:
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        plotter.add_mesh(
            surface,
            scalars="von_mises_stress",
            cmap="rainbow",
            clim=STRESS_RANGE_KPA,
            scalar_bar_args={"title": "von Mises (kPa)"},
        )
        plotter.camera_position = "iso"
        plotter.screenshot(str(plot_file))
        plotter.close()
    finally:
        # Paired with the guarded start above, so environments without Xvfb
        # (Windows, pyvista >= 0.48) are unaffected.
        if xvfb_started:
            try:
                pv.stop_xvfb()
            except Exception:
                pass
    return plot_file


# Only run if this script is not imported as a module

# PhysicsNeMo and torch spawn worker processes. On Windows the spawn start
# method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the whole evaluation in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_18_duke_heart_physics_informed_motion_infer"

    test_mode = TestTools.running_as_test()
    parameters = DUKE_HEART_PHYSICS_INFORMED

    # The case held out of every fit in this chain.
    case_id = parameters.hold_out_case

    prep_dir = parameters.prep_directory(test_mode)
    case_dir = prep_dir / case_id
    output_dir = parameters.infer_directory(test_mode) / case_id
    baselines_dir = repo_root / "tests" / "baselines"
    labelmap_dir = parameters.hold_out_directory(test_mode) / case_id

    template_file = parameters.ssm_template_file(test_mode)
    manifest_file = prep_dir / "manifests" / f"{case_id}_manifest.json"

    # Checkpoint epoch to infer with; None uses the final weights.
    epoch: Optional[int] = None

    # Constitutive law; the same one the training loss priced motion against, so
    # the stress reported here is the stress the network was trained under.
    mu_kpa = parameters.mu_kpa
    lambda_lame_kpa = parameters.lambda_lame_kpa

    # Gaussian sigma, in mm, that spreads the predicted displacements into the
    # continuous field the labelmap is resampled through.
    smoothing_sigma_mm = 10.0
    # Isotropic pitch every metric is measured on.
    evaluation_spacing_mm = 1.0

    # Point-wise displacement reporting is off here, unlike Tutorial 11.  The
    # fitted reference is volumetric while the ground-truth per-frame fits are
    # its boundary, so there is no point-for-point pairing between them to
    # report against.  Dice, volume and surface RMSE are geometric and unaffected.
    report_displacement_data = False

    log_level = logging.INFO

    # Directory setup and data reading
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    # A GPU is assumed. Every registration, fit and network pass below runs on
    # one, so fail here rather than hours into the cohort at the first call.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device is visible. Tutorials 16 to 18 assume a GPU; a "
            "CPU-only run is not supported and would take days."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    fitted_reference_model_file = case_dir / f"{case_id}_ssm_model.vtu"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    for required_file in (
        template_file,
        manifest_file,
        fitted_reference_model_file,
        pca_file,
    ):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 16 output not found: {required_file}\n"
                "Run tutorials/"
                "tutorial_16_duke_heart_physics_informed_motion_prep.py first."
            )

    model_directories = {
        "physics_informed": parameters.physics_informed_weights_directory(test_mode),
    }
    ablation_directory = parameters.ablation_weights_directory(test_mode)
    if ablation_directory.exists():
        model_directories["ablation"] = ablation_directory
    else:
        logger.warning(
            "No ablation model at %s; scoring the physics-informed model alone. "
            "Set train_ablation_baseline and re-run Tutorial 17 to compare.",
            ablation_directory,
        )
    for name, directory in model_directories.items():
        if not directory.exists():
            raise FileNotFoundError(
                f"Tutorial 17 {name} model not found: {directory}\n"
                "Run tutorials/"
                "tutorial_17_duke_heart_physics_informed_motion_train.py first."
            )

    # Step 1: the cohort assembles what this case is scored against -- every
    # gated frame's labelmap and Tutorial 16's per-frame boundary fits.
    cohort = EvaluateMovementDukeHeart(log_level=log_level)
    ground_truth = cohort.assemble_ground_truth(
        case_id=case_id,
        frame_directory=labelmap_dir,
        fit_directory=case_dir,
    )

    # Step 2: predict and score with each model in turn.
    tutorial_results: dict[str, Any] = {"models": {}}
    for name, model_directory in model_directories.items():
        logger.info("%s", "=" * 48)
        logger.info("Scoring %s model from %s", name, model_directory)
        logger.info("%s", "=" * 48)

        infer_workflow = WorkflowInferPhysicsNeMo(
            model_directory=model_directory, epoch=epoch, log_level=log_level
        )
        evaluate_workflow = WorkflowEvaluateMovement(
            movement_workflow=WorkflowInferMovement(
                infer_workflow, log_level=log_level
            ),
            cohort=cohort,
            log_level=log_level,
        )
        tutorial_results["models"][name] = evaluate_workflow.process(
            case_id=case_id,
            shape_parameters=pca_file,
            fitted_reference_mesh=fitted_reference_model_file,
            ground_truth=ground_truth,
            output_directory=output_dir / name,
            smoothing_sigma_mm=smoothing_sigma_mm,
            evaluation_spacing_mm=evaluation_spacing_mm,
            report_displacement_data=report_displacement_data,
        )

    # Step 3: one table comparing the two, per phase and structure.
    comparison_file = output_dir / "mechanics_comparison.csv"
    rows_by_model = {
        name: result["rows"] for name, result in tutorial_results["models"].items()
    }
    fieldnames = ["model", *rows_by_model["physics_informed"][0].keys()]
    with comparison_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, rows in rows_by_model.items():
            for row in rows:
                writer.writerow({"model": name, **row})
    tutorial_results["comparison_file"] = comparison_file
    logger.info("Comparison: %s", comparison_file)

    # Step 4: the stress the physics-informed prediction implies.  The
    # deformation gradient is formed against the case's own fitted reference,
    # which is the configuration its displacements are measured from.
    template_mesh = cast(pv.UnstructuredGrid, pv.read(str(template_file)))
    tets = template_mesh.cells_dict[np.uint8(pv.CellType.TETRA)]
    reference_model = cast(pv.DataSet, pv.read(str(fitted_reference_model_file)))
    law = NeoHookeanResidual(mu_kpa, lambda_lame_kpa, log_level=log_level)

    stress_dir = output_dir / "stress"
    stress_dir.mkdir(parents=True, exist_ok=True)
    predicted_files = tutorial_results["models"]["physics_informed"][
        "predicted_surfaces"
    ]
    stress_meshes: list[pv.DataSet] = []
    stress_files: list[Path] = []
    for predicted_file in predicted_files:
        predicted_mesh = cast(pv.DataSet, pv.read(str(predicted_file)))
        if predicted_mesh.n_points != reference_model.n_points:
            raise ValueError(
                f"{predicted_file} carries {predicted_mesh.n_points} points but "
                f"the fitted reference carries {reference_model.n_points}; the "
                "deformation gradient needs them on the same nodes."
            )
        predicted_mesh.point_data[STRESS_ARRAY] = _stress_at_points(
            reference_model, predicted_mesh, tets, law
        )
        stress_file = stress_dir / f"{Path(predicted_file).stem}_stress.vtu"
        predicted_mesh.save(str(stress_file))
        stress_meshes.append(predicted_mesh)
        stress_files.append(stress_file)

    if law.inverted_element_count:
        logger.warning(
            "%d element evaluations inverted across the predicted phases; the "
            "predicted motion turns tissue inside out somewhere.",
            law.inverted_element_count,
        )
    tutorial_results["stress_files"] = stress_files

    # Step 5: export the animation, colored by von Mises stress.  The tutorial
    # supplies the stress tensor only; ConvertVTKToUSD derives the scalar from
    # it and writes both as USD primvars.
    usd_file = output_dir / "heart_physics_informed_motion.usd"
    converter = ConvertVTKToUSD(
        "heart_physics_informed_motion",
        stress_meshes,
        frames_per_second=float(len(stress_meshes)),
        log_level=log_level,
    )
    converter.compute_von_mises_stress(STRESS_ARRAY)
    converter.set_colormap("von_mises_stress", "rainbow", STRESS_RANGE_KPA)
    converter.convert(str(usd_file))
    tutorial_results["usd_file"] = usd_file
    logger.info("USD: %s", usd_file)

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )
    stress_surface = stress_meshes[-1].extract_surface(algorithm="dataset_surface")
    tutorial_results["screenshots"] = [
        _render_stress(stress_surface, output_dir / "von_mises_stress.png"),
        tt.save_screenshot_mesh(
            cast(pv.DataSet, stress_surface),
            "predicted_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
    ]
