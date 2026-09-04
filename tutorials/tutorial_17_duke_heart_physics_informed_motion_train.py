"""
Tutorial 17 (Duke Heart, PINN): Train a Mechanics-Aware Cardiac Motion Surrogate

Purpose
-------
Train the same MeshGraphNet Tutorial 9 trains, on the volumetric shape model
Tutorial 16 built, but score it against a neo-Hookean strain energy as well as
against measured displacement.

Why bother?  Tutorial 9's loss is L2 on displacement alone, so it has no opinion
about motion no myocardium could undergo: an element may inflate, thin past what
tissue allows, or invert outright, and the loss only notices to the extent that
the vertices land in the wrong place.  A strain energy prices exactly those
deformations.  The loss becomes::

    data + lambda_physics * (strain_energy + incompressibility)

with the strain energy of a compressible neo-Hookean solid,

    W = (mu / 2)(I1 - 3) - mu ln(J) + (lambda / 2) ln(J)^2

evaluated from the deformation gradient of each tetrahedron.  Spatial
derivatives come from PhysicsNeMo Sym's least-squares gradient reconstruction,
which is its method for unstructured meshes.

The residual is measured against each case's *own* fitted reference model, not
the population mean: the targets are displacements from that reference, so it is
the undeformed state.  Measuring against the mean would charge every subject a
strain energy for merely being shaped unlike the mean, confusing variation
between subjects with deformation within one.

By default a second model is trained on identical data with
``lambda_physics = 0``.  That ablation is the only comparison that isolates the
physics term -- measuring against Tutorial 9 instead would confound it with the
change from a surface shape model to a volumetric one -- and Tutorial 18 scores
the two against each other.

Extra Install Required
----------------------
PhysicsNeMo and PyTorch Geometric::

    pip install "monai-physio[physicsnemo]"
    pip install torch-geometric

A CUDA GPU is required; a CPU-only run is not a supported
configuration.

PhysicsNeMo Sym, which supplies ``PhysicsInformer``, ships inside
``nvidia-physicsnemo``; no separate install is needed.

Data Required
-------------
Tutorial 16 output: ``output/tutorial_16_duke_heart_physics_informed_motion/``
(``manifests/*_manifest.json``, ``pca_mean.vtu``, ``ssm_template.vtu``)

Outputs
-------
Under ``network_weights/physicsnemo_physics_informed_motion_duke_heart/``
(and ``..._ablation/`` for the comparison model):

  * ``physics_informed_motion_stage_model.pt``  - the trained network
  * ``training_losses.json``                    - per-epoch loss
  * ``training_validation_rmse.csv``            - intermittent validation RMSE

Under ``output/tutorial_16_duke_heart_physics_informed_motion/``:

  * ``training_losses.png``                     - both runs' loss curves

Cost
----
The template is far denser than Tutorial 9's surface one, so the mesh graph is
several times larger: expect to lower ``batch_size`` and to leave processor
gradient checkpointing on.  Training the ablation baseline doubles the run;
set ``ParametersDukeHeartPhysicsInformed.train_ablation_baseline`` to ``False``
to skip it.
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pv
from parameters_duke_heart_physics_informed import DUKE_HEART_PHYSICS_INFORMED

from monai_physio import TestTools, WorkflowTrainPhysicsNeMo
from monai_physio.train_physicsnemo_physics_informed_motion import (
    PhysicsInformedMotion,
    TrainPhysicsNeMoPhysicsInformedMotion,
)


def _plot_losses(loss_curves: dict[str, list[float]], plot_file: Path) -> Path:
    """Plot each run's per-epoch loss and return the written path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    for label, losses in loss_curves.items():
        if losses:
            axis.plot(range(1, len(losses) + 1), losses, label=label)
    axis.set_xlabel("epoch")
    axis.set_ylabel("training loss")
    axis.set_yscale("log")
    axis.set_title("Physics-informed vs data-only motion training")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plot_file), dpi=100)
    plt.close(figure)
    return plot_file


# Only run if this script is not imported as a module

# PhysicsNeMo and torch spawn worker processes. On Windows the spawn start
# method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart training in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_17_duke_heart_physics_informed_motion_train"

    test_mode = TestTools.running_as_test()
    parameters = DUKE_HEART_PHYSICS_INFORMED

    prep_dir = parameters.prep_directory(test_mode)
    manifests_dir = prep_dir / "manifests"
    template_file = parameters.ssm_template_file(test_mode)
    mean_volume_file = parameters.ssm_mean_volume_file(test_mode)
    model_output_dir = parameters.physics_informed_weights_directory(test_mode)
    ablation_output_dir = parameters.ablation_weights_directory(test_mode)
    baselines_dir = repo_root / "tests" / "baselines"

    # Constitutive law of the myocardium; see the parameters module.
    mu_kpa = parameters.mu_kpa
    lambda_lame_kpa = parameters.lambda_lame_kpa

    # Weight of the physics residual against the displacement loss.  The two are
    # not in the same units -- displacement is scored normalized, the residual in
    # kilopascals -- so treat this as a value to sweep, not one to trust.  The
    # two terms are logged separately for exactly that reason.
    lambda_physics = parameters.lambda_physics

    # Whether to also train the lambda_physics = 0 comparison model.
    train_ablation_baseline = parameters.train_ablation_baseline

    # Training hyperparameters.  The volumetric template carries several times
    # the nodes of the surface one Tutorial 9 trains on, so the mini-batch is
    # smaller and the processor is gradient-checkpointed: that recomputes
    # activations in the backward pass instead of storing them, which is what
    # keeps a graph this size inside a single card.
    epochs = parameters.epochs(test_mode)
    batch_size = 2  # mini-batch measured in (case, frame) graphs
    learning_rate = 1.0e-3
    processor_size = 3  # message-passing hops
    hidden_dim = 128
    num_layers = 2  # MLP layers inside each encoder / processor / decoder block
    processor_checkpoint_segments = 3

    # Explicit held-out split; every other case trains.  The held-out case is
    # the one Tutorials 2 and 16 also keep out, so it stays unseen end to end.
    test_cases = [parameters.hold_out_case]
    val_cases: list[str] = []

    log_level = logging.INFO

    # Directory setup and data reading
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(class_name)

    for required_file in (template_file, mean_volume_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Tutorial 16 output not found: {required_file}\n"
                "Run tutorials/"
                "tutorial_16_duke_heart_physics_informed_motion_prep.py first."
            )

    manifest_files = sorted(manifests_dir.glob("*_manifest.json"))
    if len(manifest_files) < 3:
        raise RuntimeError(
            f"Found only {len(manifest_files)} manifest(s) under {manifests_dir}; "
            "need at least 3 to hold one out and still train on a population. "
            "Run tutorials/"
            "tutorial_16_duke_heart_physics_informed_motion_prep.py first."
        )

    # Each case's fitted reference model is the undeformed configuration its
    # strain energy is measured against, so the trainer needs them by subject.
    manifests: dict[str, Path] = {}
    reference_meshes: dict[str, Path] = {}
    for manifest_file in manifest_files:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifests[manifest["subject_id"]] = manifest_file
        reference_meshes[manifest["subject_id"]] = Path(
            manifest["fitted_reference_mesh"]
        )

    unknown = [
        case_id for case_id in test_cases + val_cases if case_id not in manifests
    ]
    if unknown:
        raise ValueError(f"Split cases not found: {unknown}")

    test_manifests = [manifests[case_id] for case_id in test_cases]
    val_manifests = [manifests[case_id] for case_id in val_cases]
    train_manifests = [
        manifest_path
        for case_id, manifest_path in manifests.items()
        if case_id not in test_cases and case_id not in val_cases
    ]
    logger.info(
        "Case split - train: %d, val: %d, test: %d",
        len(train_manifests),
        len(val_manifests),
        len(test_manifests),
    )

    # The template's own cells are the physics elements: every subject and phase
    # inherits its topology, so one set of node ids is valid for all of them.
    template_mesh = cast(pv.UnstructuredGrid, pv.read(str(template_file)))
    tets = template_mesh.cells_dict[np.uint8(pv.CellType.TETRA)]
    logger.info(
        "Physics elements: %d tetrahedra over %d nodes",
        len(tets),
        template_mesh.n_points,
    )

    import torch

    # A GPU is assumed. Training this graph on a CPU is not a supported
    # configuration, so say so now rather than after the data is loaded.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device is visible. Tutorials 16 to 18 assume a GPU; a "
            "CPU-only run is not supported and would take days."
        )
    device = torch.device("cuda")

    def _train(
        weight_of_physics: float, output_directory: Path, label: str
    ) -> dict[str, Any]:
        """Train one model and return the workflow's result."""
        logger.info("%s", "=" * 48)
        logger.info("Training %s (lambda_physics=%.4g)", label, weight_of_physics)
        logger.info("%s", "=" * 48)

        training_method = TrainPhysicsNeMoPhysicsInformedMotion(log_level=log_level)
        training_method.set_epochs(epochs)
        training_method.set_batch_size(batch_size)
        training_method.set_learning_rate(learning_rate)
        training_method.set_processor_size(processor_size)
        training_method.set_hidden_dim(hidden_dim)
        training_method.set_num_layers(num_layers)
        training_method.set_num_processor_checkpoint_segments(
            processor_checkpoint_segments
        )
        if weight_of_physics > 0.0:
            training_method.set_elements(tets)
            training_method.set_reference_meshes(reference_meshes)
            training_method.set_mechanics(
                PhysicsInformedMotion(
                    tets=tets,
                    n_points=template_mesh.n_points,
                    mu_kpa=mu_kpa,
                    lambda_lame_kpa=lambda_lame_kpa,
                    device=device,
                    log_level=log_level,
                ),
                lambda_physics=weight_of_physics,
            )
        else:
            # No residual is built at all, so this run is the data-only
            # MeshGraphNet on exactly the same data.
            training_method.set_mechanics(None, lambda_physics=0.0)

        workflow = WorkflowTrainPhysicsNeMo(
            train_manifests=train_manifests,
            val_manifests=val_manifests,
            pca_mean_mesh=mean_volume_file,
            output_directory=output_directory,
            training_method=training_method,
            log_level=log_level,
        )
        result = workflow.process()
        inverted = training_method.inverted_element_count
        if inverted:
            logger.warning(
                "%s: %d element evaluations inverted during training; the "
                "clamp kept them finite, but the predicted motion turns tissue "
                "inside out somewhere.",
                label,
                inverted,
            )
        result["inverted_element_count"] = inverted
        return result

    tutorial_results: dict[str, Any] = {"runs": {}}
    loss_curves: dict[str, list[float]] = {}

    physics_result = _train(lambda_physics, model_output_dir, "physics-informed")
    tutorial_results["runs"]["physics_informed"] = physics_result
    loss_curves[f"physics-informed (lambda={lambda_physics:g})"] = physics_result[
        "training_loss"
    ]

    if train_ablation_baseline:
        ablation_result = _train(0.0, ablation_output_dir, "data-only ablation")
        tutorial_results["runs"]["ablation"] = ablation_result
        loss_curves["data-only ablation"] = ablation_result["training_loss"]

    # The two curves are not directly comparable -- the physics-informed one
    # sums a term the other does not have -- so this is a convergence check,
    # not a score.  Tutorial 18 is where the two models are actually compared.
    prep_dir.mkdir(parents=True, exist_ok=True)
    plot_file = _plot_losses(loss_curves, prep_dir / "training_losses.png")

    tutorial_results["model_directory"] = physics_result["output_directory"]
    tutorial_results["checkpoint"] = physics_result["checkpoint"]
    tutorial_results["plot_file"] = plot_file
    logger.info("Physics-informed model: %s", physics_result["output_directory"])

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=prep_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )
    tutorial_results["screenshots"] = [
        plot_file,
        tt.save_screenshot_mesh(
            cast(
                pv.DataSet, template_mesh.extract_surface(algorithm="dataset_surface")
            ),
            "trained_template_boundary.png",
            camera_position="iso",
            color="steelblue",
            opacity=0.9,
        ),
    ]
