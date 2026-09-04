"""
Command-line interface for Heart Model to Patient Registration workflow.

This script provides a CLI to register a generic heart model to patient-specific
imaging data and surface models using multi-stage registration (ICP, PCA, mask-based,
and optional image-based refinement).
"""

import argparse
import json
import os
import sys
import traceback


def main() -> int:
    """Command-line interface for heart model to patient registration."""
    parser = argparse.ArgumentParser(
        description="Register generic heart model to patient-specific data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic registration (no patient image: reference image created from patient models)
  %(prog)s \\
    --template-model heart_model.vtu \\
    --patient-models lv.vtp rv.vtp myo.vtp \\
    --output-dir ./results

  # With patient image and PCA shape fitting
  %(prog)s \\
    --template-model heart_model.vtu \\
    --patient-models lv.vtp rv.vtp myo.vtp \\
    --patient-image patient_ct.nii.gz \\
    --pca-json pca_model.json \\
    --number-of-pca-components 10 \\
    --output-dir ./results

  # Enable labelmap-to-image refinement (requires template labelmap and label IDs)
  %(prog)s \\
    --template-model heart_model.vtu \\
    --patient-models lv.vtp rv.vtp myo.vtp \\
    --patient-image patient_ct.nii.gz \\
    --labelmap-to-image \\
    --template-labelmap heart_labelmap.nii.gz \\
    --template-labelmap-muscle-ids 1 2 3 \\
    --template-labelmap-chamber-ids 4 5 6 \\
    --template-labelmap-background-ids 0 \\
    --output-dir ./results

  # With ICON refinement
  %(prog)s \\
    --template-model heart_model.vtu \\
    --patient-models lv.vtp rv.vtp \\
    --patient-image patient_ct.nii.gz \\
    --use-ICON-refinement \\
    --output-dir ./results
        """,
    )

    # Required arguments
    parser.add_argument(
        "--template-model",
        required=True,
        help="Path to template/generic heart model (.vtu, .vtk, .stl)",
    )
    parser.add_argument(
        "--patient-models",
        nargs="+",
        required=True,
        help="Paths to patient-specific surface models (e.g., lv.vtp rv.vtp myo.vtp)",
    )
    parser.add_argument(
        "--patient-image",
        help="Path to patient CT/MRI image (.nii.gz, .nrrd, .mha). If omitted, a reference image is created from the patient models.",
    )
    parser.add_argument(
        "--template-labelmap",
        help="Path to template labelmap image (.nii.gz, .nrrd, .mha). Required when --labelmap-to-image is set.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )

    # Template labelmap configuration
    parser.add_argument(
        "--template-labelmap-muscle-ids",
        nargs="+",
        type=int,
        default=[1],
        help="Label IDs for heart muscle in template labelmap (default: 1)",
    )
    parser.add_argument(
        "--template-labelmap-chamber-ids",
        nargs="+",
        type=int,
        default=[2],
        help="Label IDs for heart chambers in template labelmap (default: 2)",
    )
    parser.add_argument(
        "--template-labelmap-background-ids",
        nargs="+",
        type=int,
        default=[0],
        help="Label IDs for background in template labelmap (default: 0)",
    )

    # PCA registration options
    parser.add_argument(
        "--pca-json",
        help="Path to PCA JSON file for shape-based registration (optional)",
    )
    parser.add_argument(
        "--number-of-pca-components",
        type=int,
        default=0,
        help="Number of PCA components to use (default: 0, uses all if PCA enabled)",
    )

    # Registration configuration
    parser.add_argument(
        "--no-labelmap-to-labelmap",
        dest="use_labelmap_to_labelmap",
        action="store_false",
        default=True,
        help="Disable labelmap-to-labelmap deformable registration",
    )
    parser.add_argument(
        "--labelmap-to-image",
        dest="use_labelmap_to_image",
        action="store_true",
        default=False,
        help="Enable labelmap-to-image refinement (requires --template-labelmap and label IDs)",
    )
    parser.add_argument(
        "--use-ICON-refinement",
        action="store_true",
        default=False,
        help="Enable ICON registration refinement (default: disabled)",
    )

    # Output options
    parser.add_argument(
        "--output-prefix",
        default="registered",
        help="Prefix for output files (default: registered)",
    )

    args = parser.parse_args()

    # Validate input files
    print("Validating input files...")
    if not os.path.exists(args.template_model):
        print(f"Error: Template model not found: {args.template_model}")
        return 1

    for patient_model in args.patient_models:
        if not os.path.exists(patient_model):
            print(f"Error: Patient model not found: {patient_model}")
            return 1

    if args.patient_image is not None and not os.path.exists(args.patient_image):
        print(f"Error: Patient image not found: {args.patient_image}")
        return 1

    if args.use_labelmap_to_image:
        if args.template_labelmap is None:
            print(
                "Error: --template-labelmap is required when --labelmap-to-image is set."
            )
            return 1
        if not os.path.exists(args.template_labelmap):
            print(f"Error: Template labelmap not found: {args.template_labelmap}")
            return 1

    if args.pca_json and not os.path.exists(args.pca_json):
        print(f"Error: PCA JSON file not found: {args.pca_json}")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input data
    print("\nLoading input data...")
    try:
        import itk
        import pyvista as pv

        print(f"  Loading template model: {args.template_model}")
        template_model_raw = pv.read(args.template_model)
        assert isinstance(template_model_raw, pv.UnstructuredGrid), (
            f"Template model must be an UnstructuredGrid, got {type(template_model_raw)}"
        )
        template_model: pv.UnstructuredGrid = template_model_raw

        print("  Loading patient models:")
        patient_models: list[pv.DataSet] = []
        for patient_model_file in args.patient_models:
            print(f"    - {patient_model_file}")
            patient_model_raw = pv.read(patient_model_file)
            assert isinstance(patient_model_raw, pv.DataSet), (
                f"Patient model must be a PyVista dataset: {patient_model_file}"
            )
            patient_models.append(patient_model_raw)

        if args.patient_image is not None:
            print(f"  Loading patient image: {args.patient_image}")
            patient_image = itk.imread(args.patient_image)
        else:
            patient_image = None
            print(
                "  No patient image: reference image will be created from patient models"
            )

        template_labelmap = None
        if args.template_labelmap is not None:
            print(f"  Loading template labelmap: {args.template_labelmap}")
            template_labelmap = itk.imread(args.template_labelmap)

    except (FileNotFoundError, OSError, RuntimeError) as e:
        print(f"Error loading input data: {e}")
        traceback.print_exc()
        return 1

    # Initialize workflow
    print("\nInitializing heart model to patient registration workflow...")
    try:
        from .. import WorkflowFitStatisticalModelToPatient

        workflow = WorkflowFitStatisticalModelToPatient(
            template_model=template_model,
            patient_models=patient_models,
            patient_image=patient_image,
        )
        if args.pca_json is not None:
            with open(args.pca_json, encoding="utf-8") as f:
                pca_model = json.load(f)
            workflow.set_use_pca_registration(
                True,
                pca_model=pca_model,
                number_of_pca_components=args.number_of_pca_components,
            )

        workflow.set_use_labelmap_to_labelmap_registration(
            args.use_labelmap_to_labelmap
        )

        if args.use_labelmap_to_image:
            workflow.set_use_labelmap_to_image_registration(
                True,
                template_labelmap=template_labelmap,
                template_labelmap_organ_mesh_ids=args.template_labelmap_muscle_ids,
                template_labelmap_organ_extra_ids=args.template_labelmap_chamber_ids,
                template_labelmap_background_ids=args.template_labelmap_background_ids,
            )

    except (ValueError, RuntimeError, OSError) as e:
        print(f"Error initializing workflow: {e}")
        traceback.print_exc()
        return 1

    try:
        # Execute registration workflow
        print("\nStarting registration pipeline...")
        print("=" * 70)
        result = workflow.process(
            use_ICON_registration_refinement=args.use_ICON_refinement,
        )

        # Save results
        print("\n" + "=" * 70)
        print("Saving results...")

        # Save registered model
        registered_model = result["fitted_reference_model"]
        output_model_file = os.path.join(
            args.output_dir, f"{args.output_prefix}_model.vtu"
        )
        registered_model.save(output_model_file)
        print(f"  Registered model: {output_model_file}")

        # Save registered model surface
        registered_surface = result["fitted_reference_mesh"]
        output_surface_file = os.path.join(
            args.output_dir, f"{args.output_prefix}_model_surface.vtp"
        )
        registered_surface.save(output_surface_file)
        print(f"  Registered surface: {output_surface_file}")

        # Save registered labelmap if available
        if workflow.l2i_template_labelmap is not None:
            output_labelmap_file = os.path.join(
                args.output_dir, f"{args.output_prefix}_labelmap.nii.gz"
            )
            itk.imwrite(
                workflow.l2i_template_labelmap, output_labelmap_file, compression=True
            )
            print(f"  Registered labelmap: {output_labelmap_file}")
        elif workflow.l2l_template_labelmap is not None:
            output_labelmap_file = os.path.join(
                args.output_dir, f"{args.output_prefix}_labelmap.nii.gz"
            )
            itk.imwrite(
                workflow.l2l_template_labelmap, output_labelmap_file, compression=True
            )
            print(f"  Registered labelmap: {output_labelmap_file}")

        # Save intermediate results if available
        if workflow.icp_template_model_surface is not None:
            output_icp_file = os.path.join(
                args.output_dir, f"{args.output_prefix}_icp_surface.vtp"
            )
            workflow.icp_template_model_surface.save(output_icp_file)
            print(f"  ICP result: {output_icp_file}")

        if workflow.pca_template_model_surface is not None:
            output_pca_file = os.path.join(
                args.output_dir, f"{args.output_prefix}_pca_surface.vtp"
            )
            workflow.pca_template_model_surface.save(output_pca_file)
            print(f"  PCA result: {output_pca_file}")

        if workflow.l2l_template_model_surface is not None:
            output_l2l_file = os.path.join(
                args.output_dir, f"{args.output_prefix}_l2l_surface.vtp"
            )
            workflow.l2l_template_model_surface.save(output_l2l_file)
            print(f"  Labelmap-to-labelmap result: {output_l2l_file}")

        print("\n" + "=" * 70)
        print("Registration completed successfully!")
        print(f"\nAll output files saved to: {args.output_dir}")

        return 0

    except (RuntimeError, ValueError, OSError) as e:
        print(f"\nError during registration: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
