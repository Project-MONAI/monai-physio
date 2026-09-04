"""
Command-line interface for the Image-to-USD workflow.

This script provides a CLI to process 4D CT images through the complete workflow,
generating dynamic USD models suitable for visualization in NVIDIA Omniverse.
"""

import argparse
import os
import sys

import itk

from ..convert_image_4d_to_3d import ConvertImage4DTo3D
from ..register_images_greedy import RegisterImagesGreedy
from ..register_images_icon import RegisterImagesICON
from ._method_factories import build_registration_method, build_segmentation_method


def main() -> int:
    """Command-line interface for the Image-to-USD workflow."""
    parser = argparse.ArgumentParser(
        description="Process 4D CT images to dynamic USD models for Omniverse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single 4D NRRD file
  %(prog)s input_4d.nrrd --contrast --output-dir ./results

  # Process multiple 3D NRRD files as time series
  %(prog)s frame_*.nrrd --output-dir ./results --project-name cardiac

  # Specify reference image and registration iterations
  %(prog)s input.nrrd --reference-image ref.mha --registration-iterations 50

  # Use ANTs registration instead of ICON
  %(prog)s input.nrrd --contrast --registration-method ANTS

  # Use the cardiac-only Simpleware segmentation backend
  %(prog)s input.nrrd --segmentation-method HeartSimpleware

  # Set animated USD playback to 30 frames per second
  %(prog)s input.nrrd --fps 30
        """,
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help=(
            "Input image source(s): a single 4D file (NRRD/NIfTI/MHA/...), "
            "a directory containing a DICOM series (3D or 4D), or a list of "
            "3D files representing a time series."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--project-name",
        default="cardiac_model",
        help="Project name for USD organization (default: cardiac_model)",
    )
    parser.add_argument(
        "--contrast", action="store_true", help="Indicate if study is contrast enhanced"
    )
    parser.add_argument(
        "--reference-image",
        help="Path to reference image file (default: uses 70%% time point)",
    )
    parser.add_argument(
        "--registration-iterations",
        type=int,
        default=1,
        help="Number of registration iterations (default: 1)",
    )
    parser.add_argument(
        "--segmentation-method",
        choices=[
            "ChestTotalSegmentator",
            "HeartSimpleware",
            "HeartSimplewareTrimmedBranches",
        ],
        default="ChestTotalSegmentator",
        help=(
            "Segmentation backend to use: ChestTotalSegmentator (default), "
            "HeartSimpleware, or HeartSimplewareTrimmedBranches "
            "(HeartSimpleware with pulmonary/great-vessel branches trimmed "
            "to the cardiac region)."
        ),
    )
    parser.add_argument(
        "--registration-method",
        choices=["Greedy", "ICON"],
        default="ICON",
        help="Registration method to use: Greedy or ICON (default: ICON)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        dest="frames_per_second",
        help="Frames per second for animated USD time series (default: 24)",
    )

    args = parser.parse_args()

    # Validate input files
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return 1

    # Initialize processor
    print("Initializing Image-to-USD processor...")
    try:
        from .. import WorkflowConvertImageToUSD

        segmentation_method = build_segmentation_method(
            args.segmentation_method, contrast=args.contrast
        )
        registration_method = build_registration_method(args.registration_method)
        if (
            args.registration_iterations is not None
            and args.registration_iterations > 0
        ):
            if isinstance(registration_method, RegisterImagesGreedy):
                registration_method.set_number_of_iterations(
                    [
                        args.registration_iterations,
                        args.registration_iterations // 2,
                        0,
                    ]
                )
            elif isinstance(registration_method, RegisterImagesICON):
                registration_method.set_number_of_iterations(
                    args.registration_iterations
                )

        if len(args.input_files) == 1:
            convert_image_4d_to_3d = ConvertImage4DTo3D()
            convert_image_4d_to_3d.load_image_4d(args.input_files[0])
            time_series_images = convert_image_4d_to_3d.get_3d_images()
        else:
            time_series_images = [
                itk.imread(str(input_file)) for input_file in args.input_files
            ]
        if args.reference_image is not None:
            reference_image = itk.imread(str(args.reference_image))
        else:
            reference_image = time_series_images[int(len(time_series_images) * 0.7)]
        processor = WorkflowConvertImageToUSD(
            time_series_images=time_series_images,
            reference_image=reference_image,
            usd_project_name=args.project_name,
            output_directory=args.output_dir,
            segmentation_method=segmentation_method,
            registration_method=registration_method,
            frames_per_second=args.frames_per_second,
        )
    except Exception as e:
        print(f"Error initializing workflow: {e}")
        return 1

    try:
        # Execute complete workflow
        print("\nStarting Image-to-USD processing pipeline...")
        print("=" * 60)
        processor.process()

        print("\n" + "=" * 60)
        print("Processing completed successfully!")
        print(f"\nOutput files created in: {args.output_dir}")
        print(f"  - {args.project_name}.dynamic_painted.usd")
        print(f"  - {args.project_name}.static_painted.usd")
        print(f"  - {args.project_name}.all_painted.usd")
        print("\nYou can now open these files in NVIDIA Omniverse.")

        return 0

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
