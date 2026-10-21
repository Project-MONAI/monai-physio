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

Hardware Required
-----------------
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

import contextlib
import csv
import itertools
import logging
from pathlib import Path
from typing import Any, Optional, cast

import matplotlib
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from parameters_duke_heart_physics_informed import DUKE_HEART_PHYSICS_INFORMED

from monai_physio import (
    ConvertVTKToUSD,
    EvaluateMovementDukeHeart,
    InferPhysicsNeMoMGN,
    MONAIPhysioBase,
    ProcessTests,
    WorkflowEvaluateMovement,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)
from monai_physio.train_physicsnemo_physics_informed_motion import (
    NeoHookeanResidual,
    compute_deformation_gradient,
    tet_volumes,
)

# Nine-component point-data array ``ConvertVTKToUSD.compute_von_mises_stress``
# reduces to the scalar the USD is colored by.
STRESS_ARRAY = "stress"

# Range the stress colormap is stretched over, in kilopascals.  Fixed rather
# than per-frame, so a color means the same thing throughout the animation.
STRESS_RANGE_KPA = (1.0, 100.0)

# Colored on log10 of the stress so the low end is resolved and a few hot spots
# do not dominate the scale.
LOG_STRESS_ARRAY = "log10_von_mises_stress"
STRESS_COLORMAP = "heart_heat"
matplotlib.colormaps.register(
    LinearSegmentedColormap.from_list(
        STRESS_COLORMAP, ["#5a0a10", "#a01418", "#e0481c", "#ffd23c"]
    ),
    force=True,
)

# Fraction of the log range the colormap is shifted up by, which moves more of
# the surface toward the red end.
COLORMAP_SHIFT = 0.2
_log_low = float(np.log10(STRESS_RANGE_KPA[0]))
_log_high = float(np.log10(STRESS_RANGE_KPA[1]))
_log_shift = COLORMAP_SHIFT * (_log_high - _log_low)
LOG_STRESS_RANGE = (_log_low + _log_shift, _log_high + _log_shift)

# Passes of neighbor averaging over the tetrahedral mesh edges applied to the
# nodal stress tensor, which removes element-to-element noise.
STRESS_SMOOTHING_ITERATIONS = 20

# Passes of a median filter over each node's mesh neighbors applied to the
# point positions of the rendered surface, which removes isolated spikes.  It
# can fold elements, so it is applied to the surface only, never the tet mesh.
POSITION_MEDIAN_PASSES = 4


def _cell_edges(cells: np.ndarray) -> np.ndarray:
    """Return the ``(n_edges, 2)`` directed node pairs of simplex ``cells``."""
    edges = np.concatenate(
        [cells[:, [a, b]] for a, b in itertools.combinations(range(cells.shape[1]), 2)]
    )
    return np.concatenate([edges, edges[:, ::-1]])


def _median_filter_points(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return ``points`` with each node set to the per-axis median of itself
    and its edge neighbors in ``cells``."""
    points = np.asarray(points, dtype=np.float64)
    edges = _cell_edges(cells)
    edges = edges[np.argsort(edges[:, 0], kind="stable")]
    source = edges[:, 0]
    degree = np.bincount(source, minlength=len(points))
    starts = np.cumsum(degree) - degree
    slot = np.arange(len(source)) - starts[source]
    for _ in range(POSITION_MEDIAN_PASSES):
        window = np.full((len(points), degree.max() + 1, 3), np.nan)
        window[:, 0] = points
        window[source, slot + 1] = points[edges[:, 1]]
        points = np.nanmedian(window, axis=1)
    return points


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
    nodal_stress /= np.maximum(nodal_weight, 1.0e-12)[:, None]

    edges = _cell_edges(tets)
    degree = np.maximum(np.bincount(edges[:, 0], minlength=len(nodal_stress)), 1)
    for _ in range(STRESS_SMOOTHING_ITERATIONS):
        neighbor_sum = np.zeros_like(nodal_stress)
        np.add.at(neighbor_sum, edges[:, 0], nodal_stress[edges[:, 1]])
        nodal_stress = 0.5 * nodal_stress + 0.5 * neighbor_sum / degree[:, None]
    return nodal_stress


def _render_stress(surface: pv.DataSet, plot_file: Path) -> Path:
    """Render a surface colored by von Mises stress and return the written path.

    ``ProcessTests.save_screenshot_mesh`` paints a mesh one flat color, so a scalar
    field needs its own plotter.
    """
    xvfb_started = False
    with contextlib.suppress(Exception):
        pv.start_xvfb()
        xvfb_started = True
    try:
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        plotter.add_mesh(
            surface,
            scalars=LOG_STRESS_ARRAY,
            cmap=STRESS_COLORMAP,
            clim=LOG_STRESS_RANGE,
            scalar_bar_args={"title": "log10 von Mises (kPa)"},
        )
        plotter.camera_position = "iso"
        plotter.screenshot(str(plot_file))
        plotter.close()
    finally:
        # Paired with the guarded start above, so environments without Xvfb
        # (Windows, pyvista >= 0.48) are unaffected.
        if xvfb_started:
            with contextlib.suppress(Exception):
                pv.stop_xvfb()
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

    test_mode = ProcessTests.running_as_test()
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
    logger = MONAIPhysioBase(class_name=class_name, log_level=log_level)

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
        logger.log_warning(
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
        logger.log_info("%s", "=" * 48)
        logger.log_info("Scoring %s model from %s", name, model_directory)
        logger.log_info("%s", "=" * 48)

        infer_workflow = WorkflowInferPhysicsNeMo(
            model_directory=model_directory,
            inference_method=InferPhysicsNeMoMGN(
                model_tag="physics_informed_motion", log_level=log_level
            ),
            epoch=epoch,
            log_level=log_level,
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
    logger.log_info("Comparison: %s", comparison_file)

    # Step 4: the stress the physics-informed prediction implies.  The
    # deformation gradient is formed against the case's own fitted reference,
    # which is the configuration its displacements are measured from.
    template_mesh = cast(pv.UnstructuredGrid, pv.read(str(template_file)))
    tets = cast(np.ndarray, template_mesh.cells_dict[np.uint8(pv.CellType.TETRA)])
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
        logger.log_warning(
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
        output_tet_mesh=True,
    )
    converter.compute_von_mises_stress(STRESS_ARRAY)
    for mesh in stress_meshes:
        mesh.point_data[LOG_STRESS_ARRAY] = np.log10(
            np.maximum(mesh.point_data["von_mises_stress"], STRESS_RANGE_KPA[0])
        )
    converter.set_colormap(LOG_STRESS_ARRAY, STRESS_COLORMAP, LOG_STRESS_RANGE)
    converter.convert(str(usd_file))
    tutorial_results["usd_file"] = usd_file
    logger.log_info("USD: %s", usd_file)

    # Testing
    tt = ProcessTests(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )
    stress_surface = stress_meshes[-1].extract_surface(algorithm="dataset_surface")
    stress_surface.points = _median_filter_points(
        stress_surface.points, stress_surface.faces.reshape(-1, 4)[:, 1:]
    )
    tutorial_results["screenshots"] = [
        _render_stress(stress_surface, output_dir / "von_mises_stress.png"),
        tt.save_screenshot_mesh(
            cast(pv.DataSet, stress_surface),
            "predicted_surface.png",
            camera_position="iso",
            color="limegreen",
        ),
    ]
