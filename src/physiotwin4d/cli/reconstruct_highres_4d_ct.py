"""
Command-line interface for high-resolution 4D CT reconstruction workflow.

This script provides a CLI to reconstruct high-resolution 4D CT time series from
lower-resolution time-series images and a single high-resolution reference image
using combined Greedy+ICON registration.
"""

import argparse
import glob
import os
import sys
import traceback

from ..register_images_greedy import RegisterImagesGreedy
from ..register_images_greedy_icon import RegisterImagesGreedyICON
from ..register_images_icon import RegisterImagesICON
from ._method_factories import build_registration_method


def main() -> int:
    """Command-line interface for high-resolution 4D CT reconstruction."""
    parser = argparse.ArgumentParser(
        description="Reconstruct high-resolution 4D CT from time series and reference image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction with default parameters
  %(prog)s \\
    --time-series-images frame_*.mha \\
    --fixed-image highres_reference.mha \\
    --output-dir ./results

  # Reconstruction with upsampling and custom reference frame
  %(prog)s \\
    --time-series-images frame_000.mha frame_001.mha frame_002.mha \\
    --fixed-image highres.mha \\
    --reference-frame 1 \\
    --output-dir ./results

  # Reconstruction with temporal smoothing
  %(prog)s \\
    --time-series-images frame_*.mha \\
    --fixed-image highres.mha \\
    --prior-weight 0.5 \\
    --register-reference \\
    --output-dir ./results

  # Reconstruction with custom registration parameters
  %(prog)s \\
    --time-series-images frame_*.mha \\
    --fixed-image highres.mha \\
    --registration-method Greedy_ICON \\
    --Greedy-iterations 30 15 7 3 \\
    --ICON-iterations 20 \\
    --output-dir ./results

  # Reconstruction with ICON only
  %(prog)s \\
    --time-series-images frame_*.mha \\
    --fixed-image highres.mha \\
    --registration-method ICON \\
    --ICON-iterations 50 \\
    --output-dir ./results

  # Reconstruction using a mean composite instead of the reference image
  %(prog)s \\
    --time-series-images frame_*.mha \\
    --fixed-image highres.mha \\
    --composite-mode mean \\
    --output-dir ./results
        """,
    )

    # Required arguments
    parser.add_argument(
        "--time-series-images",
        nargs="+",
        required=True,
        help="Paths to time-series images (supports wildcards, e.g., 'frame_*.mha')",
    )
    parser.add_argument(
        "--fixed-image",
        required=True,
        help="Path to high-resolution reference image (.mha, .nrrd, .nii.gz)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )

    # Registration configuration
    parser.add_argument(
        "--registration-method",
        choices=["Greedy", "ICON", "Greedy_ICON"],
        default="Greedy_ICON",
        help="Registration method to use (default: Greedy_ICON)",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Index of reference frame in time series (default: 0)",
    )
    parser.add_argument(
        "--register-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Register the reference time frame to the reference image "
            "(default: enabled; use --no-register-reference for an identity transform)"
        ),
    )
    # Registration iterations
    parser.add_argument(
        "--Greedy-iterations",
        nargs="+",
        type=int,
        help="Greedy multi-resolution iterations (e.g., 30 15 7 3). Default: [30, 15, 7, 3]",
    )
    parser.add_argument(
        "--ICON-iterations",
        type=int,
        default=None,
        help="ICON finetuning iterations. Default: None",
    )

    # Mask options
    parser.add_argument(
        "--fixed-mask",
        help="Path to fixed image mask (.mha, .nrrd, .nii.gz)",
    )
    parser.add_argument(
        "--moving-masks",
        nargs="+",
        help="Paths to moving image masks (one per time point)",
    )
    parser.add_argument(
        "--mask-dilation-mm",
        type=float,
        default=0.0,
        help="Mask dilation in millimeters (default: 0.0)",
    )

    # Modality
    parser.add_argument(
        "--modality",
        default="ct",
        help="Imaging modality for registration optimization (default: ct)",
    )

    # Reconstruction options
    parser.add_argument(
        "--composite-mode",
        choices=["reference", "mean", "max"],
        default="reference",
        help=(
            "Image warped back to each time point: 'reference' uses the "
            "fixed image as-is (default); 'mean'/'max' first build a "
            "pixel-by-pixel mean/max composite of the fixed image and all "
            "registered time-series images"
        ),
    )

    # Output options
    parser.add_argument(
        "--output-prefix",
        default="reconstructed",
        help="Prefix for output files (default: reconstructed)",
    )
    parser.add_argument(
        "--save-transforms",
        action="store_true",
        default=False,
        help="Save forward and inverse transforms (default: False)",
    )
    parser.add_argument(
        "--save-losses",
        action="store_true",
        default=False,
        help="Save registration loss values to text file (default: False)",
    )

    args = parser.parse_args()

    # Expand wildcards in time-series-images
    time_series_files = []
    for pattern in args.time_series_images:
        matches = glob.glob(pattern)
        if matches:
            time_series_files.extend(sorted(matches))
        elif os.path.exists(pattern):
            time_series_files.append(pattern)
        else:
            print(f"Warning: No files matched pattern: {pattern}")

    if not time_series_files:
        print("Error: No time-series images found")
        return 1

    # Validate input files
    print("Validating input files...")
    print(f"  Found {len(time_series_files)} time-series images")

    for ts_file in time_series_files:
        if not os.path.exists(ts_file):
            print(f"Error: Time-series image not found: {ts_file}")
            return 1

    if not os.path.exists(args.fixed_image):
        print(f"Error: Fixed image not found: {args.fixed_image}")
        return 1

    if args.fixed_mask and not os.path.exists(args.fixed_mask):
        print(f"Error: Fixed mask not found: {args.fixed_mask}")
        return 1

    if args.moving_masks:
        if len(args.moving_masks) != len(time_series_files):
            print(
                f"Error: Number of moving masks ({len(args.moving_masks)}) "
                f"must match number of time-series images ({len(time_series_files)})"
            )
            return 1
        for mask_file in args.moving_masks:
            if not os.path.exists(mask_file):
                print(f"Error: Moving mask not found: {mask_file}")
                return 1

    # Validate reference frame
    if args.reference_frame < 0 or args.reference_frame >= len(time_series_files):
        print(
            f"Error: Reference frame {args.reference_frame} out of range "
            f"[0, {len(time_series_files) - 1}]"
        )
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input data
    print("\nLoading input data...")
    try:
        import itk

        print(f"  Loading {len(time_series_files)} time-series images...")
        time_series_images = []
        for i, ts_file in enumerate(time_series_files):
            print(f"    [{i}/{len(time_series_files)}] {os.path.basename(ts_file)}")
            img = itk.imread(ts_file, pixel_type=itk.F)
            time_series_images.append(img)

        print(f"  Loading fixed image: {args.fixed_image}")
        fixed_image = itk.imread(args.fixed_image, pixel_type=itk.F)
        print(f"    Fixed image size: {itk.size(fixed_image)}")
        print(f"    Fixed image spacing: {itk.spacing(fixed_image)}")

        # Load masks if provided
        fixed_mask = None
        if args.fixed_mask:
            print(f"  Loading fixed mask: {args.fixed_mask}")
            fixed_mask = itk.imread(args.fixed_mask, pixel_type=itk.UC)

        moving_masks = None
        if args.moving_masks:
            print(f"  Loading {len(args.moving_masks)} moving masks...")
            moving_masks = []
            for mask_file in args.moving_masks:
                mask = itk.imread(mask_file, pixel_type=itk.UC)
                moving_masks.append(mask)

    except (FileNotFoundError, OSError, RuntimeError) as e:
        print(f"Error loading input data: {e}")
        traceback.print_exc()
        return 1

    # Initialize workflow
    print("\nInitializing high-resolution 4D CT reconstruction workflow...")
    try:
        from .. import WorkflowReconstructHighres4DCT

        registration_method = build_registration_method(args.registration_method)

        # Set number of iterations based on registration method and CLI arguments
        greedy_iterations = args.Greedy_iterations or [30, 15, 7, 3]
        if args.ICON_iterations is None or args.ICON_iterations > 0:
            icon_iterations = args.ICON_iterations
        else:
            icon_iterations = None
        if isinstance(registration_method, RegisterImagesGreedyICON):
            registration_method.greedy.set_number_of_iterations(greedy_iterations)
            registration_method.icon.set_number_of_iterations(icon_iterations)
        elif isinstance(registration_method, RegisterImagesGreedy):
            registration_method.set_number_of_iterations(greedy_iterations)
        elif isinstance(registration_method, RegisterImagesICON):
            registration_method.set_number_of_iterations(icon_iterations)

        workflow = WorkflowReconstructHighres4DCT(
            time_series_images=time_series_images,
            reference_image=fixed_image,
            reference_time_frame=args.reference_frame,
            register_reference_time_frame_to_reference_image=args.register_reference,
            registration_method=registration_method,
        )

        # Configure registration parameters
        workflow.set_modality(args.modality)
        workflow.set_mask_dilation(args.mask_dilation_mm)

        if fixed_mask is not None:
            workflow.set_fixed_mask(fixed_mask)

        if moving_masks is not None:
            workflow.set_moving_masks(moving_masks)

    except (ValueError, RuntimeError, OSError) as e:
        print(f"Error initializing workflow: {e}")
        traceback.print_exc()
        return 1

    try:
        # Execute reconstruction workflow
        print("\nStarting reconstruction pipeline...")
        print("=" * 70)
        workflow.set_upsample_to_fixed_resolution(True)
        workflow.set_composite_mode(args.composite_mode)
        result = workflow.process()

        # Save results
        print("\n" + "=" * 70)
        print("Saving results...")

        # Save reconstructed images
        reconstructed_images = result["reconstructed_images"]
        print(f"  Saving {len(reconstructed_images)} reconstructed images...")
        for i, img in enumerate(reconstructed_images):
            output_file = os.path.join(
                args.output_dir, f"{args.output_prefix}_{i:03d}.mha"
            )
            itk.imwrite(img, output_file, compression=True)
            if i == 0:
                print(f"    {output_file}")
            elif i == len(reconstructed_images) - 1:
                print(f"    ... {output_file}")

        # Save transforms if requested
        if args.save_transforms:
            print("  Saving transforms...")
            forward_transforms = result["forward_transforms"]
            inverse_transforms = result["inverse_transforms"]

            for i, (fwd_tfm, inv_tfm) in enumerate(
                zip(forward_transforms, inverse_transforms)
            ):
                fwd_file = os.path.join(
                    args.output_dir, f"{args.output_prefix}_forward_{i:03d}.hdf5"
                )
                inv_file = os.path.join(
                    args.output_dir, f"{args.output_prefix}_inverse_{i:03d}.hdf5"
                )
                itk.transformwrite(fwd_tfm, fwd_file, compression=True)
                itk.transformwrite(inv_tfm, inv_file, compression=True)

                if i == 0:
                    print(f"    {fwd_file}")
                    print(f"    {inv_file}")
                elif i == len(forward_transforms) - 1:
                    print(f"    ... {fwd_file}")
                    print(f"    ... {inv_file}")

        # Save losses if requested
        if args.save_losses:
            print("  Saving registration losses...")
            losses = result["losses"]
            loss_file = os.path.join(
                args.output_dir, f"{args.output_prefix}_losses.txt"
            )
            with open(loss_file, "w") as f:
                f.write("# Frame, Loss\n")
                for i, loss in enumerate(losses):
                    f.write(f"{i}, {loss:.6f}\n")
            print(f"    {loss_file}")

            # Print loss statistics
            avg_loss = sum(losses) / len(losses)
            min_loss = min(losses)
            max_loss = max(losses)
            print("\n  Registration loss statistics:")
            print(f"    Average: {avg_loss:.6f}")
            print(f"    Min: {min_loss:.6f}")
            print(f"    Max: {max_loss:.6f}")

        print("\n" + "=" * 70)
        print("Reconstruction completed successfully!")
        print(f"\nAll output files saved to: {args.output_dir}")
        print(f"  - {len(reconstructed_images)} reconstructed images")
        if args.save_transforms:
            print(f"  - {len(forward_transforms) * 2} transform files")
        if args.save_losses:
            print("  - 1 loss statistics file")

        return 0

    except (RuntimeError, ValueError, OSError) as e:
        print(f"\nError during reconstruction: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
