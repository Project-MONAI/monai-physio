"""Command-line interface for the image-to-VTK segmentation workflow.

Segments a 3D image using a chosen backend and writes per-anatomy-group VTP
surfaces annotated with anatomy labels and colors.
"""

import argparse
import os
import sys
import traceback

from ._method_factories import SEGMENTATION_METHODS, build_segmentation_method

anatomy_groups = (
    "heart",
    "lung",
    "major_vessels",
    "bone",
    "soft_tissue",
    "other",
    "contrast",
)


def main() -> int:
    """CLI entry point for image to VTK conversion."""
    parser = argparse.ArgumentParser(
        description="Segment a 3D image and export anatomy groups as VTK surfaces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Anatomy groups
--------------
  heart, lung, major_vessels, bone, soft_tissue, other, contrast
  (empty groups are skipped automatically)

Output files — combined mode (default)
---------------------------------------
  {prefix}_surfaces.vtp   all surfaces merged into one file

Output files — group mode (--output-mode group)
---------------------------------------------------
  {prefix}_{group}.vtp    one surface per anatomy group

Output files — label mode (--output-mode label)
---------------------------------------------------
  {prefix}_{label}.vtp    one surface per individual anatomical structure

Examples
--------
  # Segment with TotalSegmentator, combined output
  %(prog)s \\
    --input-image chest_ct.nii.gz \\
    --output-dir ./results

  # Simpleware heart-only, cardiac anatomy groups, combined output
  %(prog)s \\
    --input-image chest_ct.nii.gz \\
    --segmentation-method HeartSimpleware \\
    --anatomy-groups heart major_vessels \\
    --output-dir ./results \\
    --output-prefix patient01

  # Also save the ITK segmentation labelmap
  %(prog)s \\
    --input-image chest_ct.nii.gz \\
    --output-dir ./results \\
    --save-labelmap
        """,
    )

    # ── Required ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--input-image",
        required=True,
        help="Path to the input 3D image (.nii.gz, .nrrd, .mha, …).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output files (created if absent).",
    )

    # ── Segmentation ──────────────────────────────────────────────────────
    parser.add_argument(
        "--segmentation-method",
        default="ChestTotalSegmentator",
        choices=list(SEGMENTATION_METHODS),
        help=(
            "Segmentation backend.  ChestTotalSegmentator (default) | "
            "HeartSimpleware | HeartSimplewareTrimmedBranches "
            "(HeartSimpleware with pulmonary/great-vessel branches trimmed "
            "to the cardiac region)"
        ),
    )
    parser.add_argument(
        "--contrast",
        action="store_true",
        default=False,
        help="Enable contrast-enhanced blood segmentation (default: disabled).",
    )
    parser.add_argument(
        "--anatomy-groups",
        nargs="+",
        metavar="GROUP",
        choices=list(anatomy_groups),
        default=None,
        help=(
            "Anatomy groups to extract.  Default: all non-empty groups.  "
            "Choices: " + " ".join(anatomy_groups)
        ),
    )
    parser.add_argument(
        "--surface-reduction-rate",
        type=float,
        default=0.0,
        help=(
            "Fraction in [0, 1) of surface triangles to remove via "
            "decimate_pro (default: 0.0, no decimation)."
        ),
    )

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Filename prefix for output files (default: no prefix).",
    )
    parser.add_argument(
        "--output-mode",
        choices=("combined", "group", "label"),
        default="combined",
        help=(
            "combined (default): merge all surfaces into one VTP.  "
            "group: one VTP per anatomy group.  "
            "label: one VTP per individual anatomical structure."
        ),
    )
    parser.add_argument(
        "--save-labelmap",
        action="store_true",
        default=False,
        help="Also save the detailed per-structure segmentation labelmap as a NIfTI file.",
    )

    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────
    if not os.path.exists(args.input_image):
        print(f"Error: input image not found: {args.input_image}")
        return 1

    # ── Load image ─────────────────────────────────────────────────────────
    print(f"Loading input image: {args.input_image}")
    try:
        import itk

        input_image = itk.imread(args.input_image)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Error loading image: {exc}")
        traceback.print_exc()
        return 1

    # ── Run workflow ────────────────────────────────────────────────────────
    print(f"Segmentation method : {args.segmentation_method}")
    print(f"Contrast enhanced   : {args.contrast}")
    print(f"Anatomy groups      : {args.anatomy_groups or 'all'}")
    print("=" * 70)

    try:
        from .. import ContourTools, WorkflowConvertImageToVTK

        workflow = WorkflowConvertImageToVTK(
            segmentation_method=build_segmentation_method(
                args.segmentation_method, contrast=args.contrast
            ),
        )
        result = workflow.process(
            input_image=input_image,
            anatomy_groups=args.anatomy_groups,
            surface_reduction_rate=args.surface_reduction_rate,
            extract_label_surfaces=(args.output_mode == "label"),
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error during workflow: {exc}")
        traceback.print_exc()
        return 1

    surfaces = (
        result["label_surfaces"] if args.output_mode == "label" else result["surfaces"]
    )

    if not surfaces:
        print("No anatomy groups produced any output.  Check the input image.")
        return 1

    # ── Save results ────────────────────────────────────────────────────────
    print("=" * 70)
    print("Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.output_prefix

    try:
        if args.output_mode == "combined":
            stem = f"{prefix}_surfaces" if prefix else "surfaces"
            surface_file = ContourTools.save_combined_surfaces(
                surfaces, os.path.join(args.output_dir, f"{stem}.vtp")
            )
            print(f"  Combined surface -> {surface_file}")
        else:
            # One file per anatomy group or per individual label
            saved_surfaces = ContourTools.save_surfaces(
                surfaces, args.output_dir, prefix=prefix
            )
            for name, path in saved_surfaces.items():
                print(f"  Surface  [{name:20s}] -> {path}")

        if args.save_labelmap:
            labelmap = result["labelmap"]
            stem = f"{prefix}_labelmap" if prefix else "labelmap"
            labelmap_file = os.path.join(args.output_dir, f"{stem}.nii.gz")
            itk.imwrite(labelmap, labelmap_file)
            print(f"  Labelmap         -> {labelmap_file}")

    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error saving results: {exc}")
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("Conversion completed successfully.")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
