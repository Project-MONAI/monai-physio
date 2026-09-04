"""
Command-line interface for Create Statistical Model workflow.

This script provides a CLI to build a PCA statistical shape model from a sample
of meshes aligned to a reference mesh, as in the Heart-Create_Statistical_Model
experiment notebooks. Outputs include pca_mean_surface.vtp, pca_mean.vtu (if
reference is volumetric), and pca_model.json.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    """Command-line interface for create statistical model workflow."""
    parser = argparse.ArgumentParser(
        description="Create a PCA statistical shape model from sample meshes aligned to a reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create model from a directory of sample meshes and a reference mesh
  %(prog)s \\
    --sample-meshes-dir ./meshes \\
    --reference-mesh average_mesh.vtk \\
    --output-dir ./pca_model

  # Specify sample meshes explicitly
  %(prog)s \\
    --sample-meshes 01.vtk 02.vtk 03.vtu \\
    --reference-mesh average_mesh.vtk \\
    --output-dir ./pca_model

  # Custom PCA components
  %(prog)s \\
    --sample-meshes-dir ./meshes \\
    --reference-mesh average_mesh.vtk \\
    --output-dir ./pca_model \\
    --number-of-pca-components 20
        """,
    )

    # Sample meshes: either a directory or a list of files
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample-meshes-dir",
        type=Path,
        metavar="DIR",
        help="Directory containing sample mesh files (.vtk, .vtu, .vtp)",
    )
    group.add_argument(
        "--sample-meshes",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="Paths to sample mesh files",
    )

    parser.add_argument(
        "--reference-mesh",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to reference mesh; its surface is used to align all samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory for pca_mean_surface.vtp, pca_mean.vtu, pca_model.json",
    )

    parser.add_argument(
        "--number-of-pca-components",
        type=int,
        default=7,
        help="Number of PCA components to retain (default: 7)",
    )

    args = parser.parse_args()

    # Resolve sample mesh paths
    if args.sample_meshes_dir is not None:
        smd = Path(args.sample_meshes_dir)
        sample_paths: list[Path] = []
        for ext in [".vtk", ".vtp", ".vtu"]:
            sample_paths.extend(smd.glob(f"*{ext}"))
        sample_paths = sorted(set(sample_paths))
        if not sample_paths:
            print(
                f"Error: No .vtk, .vtu, or .vtp files found in {args.sample_meshes_dir}"
            )
            return 1
    else:
        sample_paths = args.sample_meshes

    # Validate paths
    print("Validating input files...")
    if not args.reference_mesh.exists():
        print(f"Error: Reference mesh not found: {args.reference_mesh}")
        return 1
    for p in sample_paths:
        if not p.exists():
            print(f"Error: Sample mesh not found: {p}")
            return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Load meshes
    print("\nLoading meshes...")
    try:
        import pyvista as pv

        print(f"  Reference mesh: {args.reference_mesh}")
        reference_mesh_raw = pv.read(args.reference_mesh)
        assert isinstance(reference_mesh_raw, pv.DataSet), (
            "reference mesh must be a PyVista dataset"
        )
        reference_mesh = reference_mesh_raw
        print(f"  Sample meshes: {len(sample_paths)} files")
        sample_meshes = []
        for sample_path in sample_paths:
            sample_mesh_raw = pv.read(sample_path)
            assert isinstance(sample_mesh_raw, pv.DataSet), (
                f"sample mesh must be a PyVista dataset: {sample_path}"
            )
            sample_meshes.append(sample_mesh_raw)
    except (FileNotFoundError, OSError, RuntimeError) as e:
        print(f"Error loading meshes: {e}")
        traceback.print_exc()
        return 1

    # Run workflow
    print("\nInitializing create statistical model workflow...")
    try:
        from .. import WorkflowCreateStatisticalModel

        workflow = WorkflowCreateStatisticalModel(
            sample_meshes=sample_meshes,
            reference_mesh=reference_mesh,
            number_of_pca_components=args.number_of_pca_components,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error initializing workflow: {e}")
        traceback.print_exc()
        return 1

    try:
        print("\nRunning pipeline...")
        print("=" * 70)
        result = workflow.process()
        print("=" * 70)
        print("\nSaving outputs...")

        out_surface = args.output_dir / "pca_mean_surface.vtp"
        result["pca_mean_surface"].save(str(out_surface))
        print(f"  pca_mean_surface: {out_surface}")

        if result.get("pca_mean_mesh") is not None:
            out_mesh = args.output_dir / "pca_mean.vtu"
            result["pca_mean_mesh"].save(str(out_mesh))
            print(f"  pca_mean_mesh: {out_mesh}")

        out_json = args.output_dir / "pca_model.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result["pca_model"], f, indent=4)
        print(f"  pca_model: {out_json}")

        print("\nCreate statistical model completed successfully.")
        print(f"Outputs written to: {args.output_dir}")
        return 0

    except (RuntimeError, ValueError, OSError) as e:
        print(f"\nError during workflow: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
