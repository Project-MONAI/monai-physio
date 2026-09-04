"""Command-line interface for training a PhysicsNeMo cardiac mesh-stage model.

Selects the MeshGraphNet (``--network mgn``) or fully connected
(``--network mlp``) trainer, wires the per-subject manifest lists and tuning
options, and runs the workflow.
"""

import argparse
import sys
from pathlib import Path


def _apply_common(
    training_method: object, workflow: object, args: argparse.Namespace
) -> None:
    """Apply the shared tuning setters when supplied on the command line."""
    if args.epochs is not None:
        training_method.set_epochs(args.epochs)  # type: ignore[attr-defined]
    if args.batch_size is not None:
        training_method.set_batch_size(args.batch_size)  # type: ignore[attr-defined]
    if args.learning_rate is not None:
        training_method.set_learning_rate(args.learning_rate)  # type: ignore[attr-defined]
    if args.cache_size is not None:
        workflow.set_cache_size(args.cache_size)  # type: ignore[attr-defined]


def main() -> int:
    """CLI entry point for PhysicsNeMo training."""
    parser = argparse.ArgumentParser(
        description="Train a PhysicsNeMo cardiac mesh-stage model (MGN or MLP).",
    )
    parser.add_argument(
        "--network",
        choices=("mgn", "mlp"),
        required=True,
        help="Network architecture: mgn (MeshGraphNet) or mlp (FullyConnected).",
    )
    parser.add_argument(
        "--train-manifest",
        nargs="+",
        required=True,
        metavar="JSON",
        help="Per-subject training manifest files.",
    )
    parser.add_argument(
        "--val-manifest",
        nargs="*",
        default=[],
        metavar="JSON",
        help="Per-subject validation manifest files (intermittent RMSE).",
    )
    parser.add_argument(
        "--pca-mean-mesh",
        required=True,
        help="PCA template mesh (e.g. pca_mean.vtu) matching pca_model.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for checkpoints, metadata and logs.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional prior <tag>_stage_model.pt to resume from.",
    )

    # Shared tuning.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Mini-batch size in samples."
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--cache-size",
        type=int,
        default=None,
        help="RAM cache budget (decoded phase arrays); 0 = unbounded.",
    )

    # MGN-specific.
    parser.add_argument("--processor-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    # MLP-specific.
    parser.add_argument("--layer-size", type=int, default=None)
    # Shared architecture depth (both networks expose set_num_layers).
    parser.add_argument("--num-layers", type=int, default=None)

    args = parser.parse_args()

    train_manifests = [Path(p) for p in args.train_manifest]
    val_manifests = [Path(p) for p in args.val_manifest]
    pca_mean_mesh = Path(args.pca_mean_mesh)
    output_directory = Path(args.output)
    resume_from = Path(args.resume_from) if args.resume_from else None

    from ..train_physicsnemo_base import TrainPhysicsNeMoBase
    from ..workflow_train_physicsnemo import WorkflowTrainPhysicsNeMo

    training_method: TrainPhysicsNeMoBase
    if args.network == "mgn":
        from ..train_physicsnemo_mgn import TrainPhysicsNeMoMGN

        mgn_method = TrainPhysicsNeMoMGN()
        if args.processor_size is not None:
            mgn_method.set_processor_size(args.processor_size)
        if args.hidden_dim is not None:
            mgn_method.set_hidden_dim(args.hidden_dim)
        if args.num_layers is not None:
            mgn_method.set_num_layers(args.num_layers)
        training_method = mgn_method
    else:
        from ..train_physicsnemo_mlp import TrainPhysicsNeMoMLP

        mlp_method = TrainPhysicsNeMoMLP()
        if args.layer_size is not None:
            mlp_method.set_layer_size(args.layer_size)
        if args.num_layers is not None:
            mlp_method.set_num_layers(args.num_layers)
        training_method = mlp_method

    workflow = WorkflowTrainPhysicsNeMo(
        train_manifests=train_manifests,
        val_manifests=val_manifests,
        pca_mean_mesh=pca_mean_mesh,
        output_directory=output_directory,
        resume_from=resume_from,
        training_method=training_method,
    )
    _apply_common(training_method, workflow, args)
    result = workflow.process()

    print(f"Training complete. Checkpoint: {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
