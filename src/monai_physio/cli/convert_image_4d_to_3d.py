"""Command-line interface for splitting a 3D/4D image into a 3D time series.

Reads a 3D or 4D medical image and writes one ``.mha`` file per temporal frame
to the chosen output directory.  Supported inputs:

* A single 4D file readable by ITK (NRRD, NIfTI, MHA, …); each temporal frame
  is written separately.
* A single 3D file readable by ITK; it is written as a one-frame series.
* A directory containing a DICOM series (3D or 4D / gated); slices are grouped
  by temporal phase and each phase is written as a 3D frame.
"""

import argparse
import os
import sys
import traceback


def main() -> int:
    """CLI entry point for 4D-to-3D image conversion."""
    parser = argparse.ArgumentParser(
        description=(
            "Split a 4D medical image into a 3D time series using ITK readers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files
------------
  {output_dir}/{basename}_000.mha
  {output_dir}/{basename}_001.mha
  ...

Examples
--------
  # Split a 4D NRRD file into 3D MHA frames
  %(prog)s \\
    --input-image heart_4d.nrrd \\
    --output-dir ./frames
        """,
    )

    parser.add_argument(
        "--input-image",
        required=True,
        help=(
            "Path to a 3D or 4D image file (ITK-readable, e.g. NRRD/NIfTI/MHA) "
            "or a directory containing a DICOM series (3D or 4D)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output .mha files (created if absent).",
    )
    parser.add_argument(
        "--basename",
        default="slice",
        help="Filename stem for each output frame (default: slice).",
    )
    parser.add_argument(
        "--suffix",
        default="mha",
        help="Suffix for each output frame (default: mha).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_image):
        print(f"Error: input image not found: {args.input_image}")
        return 1
    try:
        from .. import ConvertImage4DTo3D

        converter = ConvertImage4DTo3D()
        print(f"Loading 4D image: {args.input_image}")
        converter.load_image_4d(args.input_image)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error loading input: {exc}")
        traceback.print_exc()
        return 1

    num_time_points = converter.get_number_of_3d_images()
    if num_time_points <= 0:
        print("No time points were extracted from the input.")
        return 1

    print(f"Extracted {num_time_points} time point(s).")
    print(f"Writing to: {args.output_dir}")

    try:
        converter.save_3d_images(
            args.output_dir,
            args.basename,
            args.suffix,
        )
    except (OSError, RuntimeError) as exc:
        print(f"Error saving images: {exc}")
        traceback.print_exc()
        return 1

    print(
        f"Done. Files: {args.basename}_000.{args.suffix} … "
        f"{args.basename}_{num_time_points - 1:03d}.{args.suffix}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
