"""Command-line interface for PhysicsNeMo cardiac mesh-stage inference.

Loads a trained model directory and predicts either from a per-subject manifest
(``--manifest``) or from a PCA shape-parameter file (``--shape-parameters``). The
network is auto-detected from the checkpoint files unless ``--network`` is given.
Manifest mode writes the raw target arrays; add ``--displacement`` to interpret
them as displacements and write deformed meshes instead. With
``--reference-image`` a deformation field and surface-normal image are
rasterized onto that image's grid.
"""

import argparse
import sys
from pathlib import Path


def _detect_network(model_dir: Path) -> str:
    """Return 'mgn' or 'mlp' based on the checkpoint present in ``model_dir``."""
    for tag in ("mgn", "mlp"):
        if (model_dir / f"{tag}_stage_model.pt").exists():
            return tag
    raise FileNotFoundError(
        f"No <tag>_stage_model.pt found in {model_dir}; pass --network explicitly."
    )


def main() -> int:
    """CLI entry point for PhysicsNeMo inference."""
    parser = argparse.ArgumentParser(
        description="Infer cardiac mesh stages with a trained PhysicsNeMo model.",
    )
    parser.add_argument("--model-dir", required=True, help="Trained model directory.")
    parser.add_argument(
        "--network",
        choices=("mgn", "mlp", "auto"),
        default="auto",
        help="Network architecture (auto-detected from the checkpoint by default).",
    )
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch.")
    parser.add_argument("--output", default=None, help="Output directory.")

    # Manifest-driven mode.
    parser.add_argument("--manifest", default=None, help="Per-subject manifest JSON.")
    parser.add_argument(
        "--stages",
        nargs="*",
        type=float,
        default=None,
        help="Arbitrary stages to predict (manifest mode; omit for phase eval).",
    )
    parser.add_argument(
        "--displacement",
        action="store_true",
        help="Treat the targets as displacements: write reference + prediction "
        "meshes instead of the raw target arrays.",
    )

    # Manifest-free single-subject mode.
    parser.add_argument(
        "--shape-parameters", default=None, help="PCA shape-parameter JSON file."
    )
    parser.add_argument(
        "--stage", type=float, default=None, help="Target stage (single-subject mode)."
    )
    parser.add_argument(
        "--fitted-reference-mesh",
        type=Path,
        default=None,
        help="The subject's fitted reference mesh, as written by "
        "physiotwin4d-fit-statistical-model-to-patient, whose points the "
        "displacements are added to (required in single-subject mode).",
    )
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Reference image; when given, write a deformation field + normal image.",
    )

    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    network = args.network if args.network != "auto" else _detect_network(model_dir)
    output = Path(args.output) if args.output else None

    from ..infer_physicsnemo_base import InferPhysicsNeMoBase
    from ..infer_physicsnemo_mgn import InferPhysicsNeMoMGN
    from ..infer_physicsnemo_mlp import InferPhysicsNeMoMLP
    from ..workflow_infer_movement import WorkflowInferMovement
    from ..workflow_infer_physicsnemo import WorkflowInferPhysicsNeMo

    inference_method: InferPhysicsNeMoBase = (
        InferPhysicsNeMoMGN() if network == "mgn" else InferPhysicsNeMoMLP()
    )
    workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir,
        inference_method=inference_method,
        epoch=args.epoch,
    )

    if args.manifest is not None:
        if args.displacement:
            result = WorkflowInferMovement(workflow).process(
                Path(args.manifest), stages=args.stages, output_directory=output
            )
            print(f"Predicted {len(result['predicted_surfaces'])} surface(s).")
            return 0
        result = workflow.process(
            Path(args.manifest), stages=args.stages, output_directory=output
        )
        print(f"Predicted {len(result['predicted_meshes'])} mesh(es).")
        return 0

    if args.shape_parameters is not None:
        if args.stage is None:
            parser.error("--stage is required with --shape-parameters.")
        # Both single-subject modes reconstruct geometry, so they need the
        # displacement interpretation of the model's targets.
        displacement = WorkflowInferMovement(workflow)
        if args.fitted_reference_mesh is None:
            parser.error(
                "--fitted-reference-mesh is required with --shape-parameters: the "
                "displacements are defined relative to the patient's fit, and a "
                "surface reconstructed from the shape parameters alone is not one."
            )
        if args.reference_image is not None:
            import itk

            reference_image = itk.imread(args.reference_image)
            result = displacement.create_deformation_field(
                Path(args.shape_parameters),
                args.stage,
                reference_image,
                fitted_reference_mesh=args.fitted_reference_mesh,
                output_directory=output,
            )
            print(
                f"Deformation field written to {result.get('deformation_field_file')}"
            )
            return 0

        result = displacement.predict_single(
            Path(args.shape_parameters),
            args.stage,
            fitted_reference_mesh=args.fitted_reference_mesh,
            output_directory=output,
        )
        print(f"Predicted surface written to {result['predicted_surface']}")
        return 0

    parser.error("Provide either --manifest or --shape-parameters.")


if __name__ == "__main__":
    sys.exit(main())
