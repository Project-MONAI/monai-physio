"""Command-line interface for downloading MONAI Physio example data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..data_download_tools import DataDownloadTools

SLICER_HEART_CT = "Slicer-Heart-CT"
KCL_HEART_MODEL = "KCL-Heart-Model"
CHOP_VALVE4D = "CHOP-Valve4D"
CHEST_CT = "Chest-CT"


def main(argv: Optional[list[str]] = None) -> int:
    """Download a supported MONAI Physio example dataset."""
    parser = argparse.ArgumentParser(
        description="Download MONAI Physio example data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s {SLICER_HEART_CT} --directory data/Slicer-Heart-CT
  %(prog)s {KCL_HEART_MODEL} --directory data/KCL-Heart-Model
  %(prog)s {CHOP_VALVE4D} --directory data/CHOP-Valve4D
  %(prog)s {CHEST_CT} --directory data/Chest-CT
        """,
    )
    parser.add_argument(
        "data_name",
        nargs="?",
        choices=[SLICER_HEART_CT, KCL_HEART_MODEL, CHOP_VALVE4D, CHEST_CT],
        default=None,
        help="Dataset to download",
    )
    parser.add_argument(
        "--directory",
        default=None,
        help="Directory where data will be stored (default: data/<data_name>)",
    )

    args = parser.parse_args(argv)
    if args.data_name is None:
        parser.print_help()
        return 1

    directory = args.directory or f"data/{args.data_name}"
    output_dir = Path(directory)

    if args.data_name == SLICER_HEART_CT:
        data_file = DataDownloadTools.DownloadSlicerHeartCTData(output_dir)
        print(f"Downloaded {SLICER_HEART_CT} to: {data_file}")
        return 0

    if args.data_name == KCL_HEART_MODEL:
        data_dir = DataDownloadTools.DownloadKCLHeartModelData(output_dir)
        print(f"Downloaded {KCL_HEART_MODEL} to: {data_dir}")
        return 0

    if args.data_name == CHOP_VALVE4D:
        data_dir = DataDownloadTools.DownloadCHOPValve4DData(output_dir)
        print(f"Downloaded {CHOP_VALVE4D} to: {data_dir}")
        return 0

    if args.data_name == CHEST_CT:
        data_file = DataDownloadTools.DownloadChestCTData(output_dir)
        print(f"Downloaded {CHEST_CT} to: {data_file}")
        return 0

    parser.error(f"Unsupported dataset: {args.data_name}")


if __name__ == "__main__":
    sys.exit(main())
