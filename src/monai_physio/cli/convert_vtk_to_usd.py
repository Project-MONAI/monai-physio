"""
Command-line interface for VTK to USD conversion workflow.

Converts one or more VTK files to USD with optional splitting by connectivity
or cell type, and applies a chosen appearance: solid color, anatomic material,
or colormap from a primvar (with auto or specified intensity range).
"""

import argparse
import os
import sys
from pathlib import Path

import pyvista as pv

from ..usd_anatomy_tools import DEFAULT_RENDER_PARAMS

# Anatomy types accepted by --anatomy-type, sourced from the renderer's
# registered defaults so that new groups/organs registered in
# DEFAULT_RENDER_PARAMS automatically appear here without code changes.
ANATOMY_TYPES = list(DEFAULT_RENDER_PARAMS.keys())


def main() -> int:
    """Command-line interface for VTK to USD conversion."""
    parser = argparse.ArgumentParser(
        description="Convert VTK file(s) to USD with optional splitting and appearance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file, solid gray
  %(prog)s mesh.vtk -o output.usd --appearance solid

  # Single file, red
  %(prog)s mesh.vtk -o output.usd --appearance solid --color 1 0 0

  # Time series, split by connectivity, colormap from stress (auto range)
  %(prog)s frame_*.vtk -o out.usd --by-connectivity --appearance colormap --primvar vtk_point_stress_c0

  # Time series, colormap with specified intensity range
  %(prog)s frame_*.vtk -o out.usd --appearance colormap --primvar stress --intensity-range 0 500

  # Single file, anatomic heart material
  %(prog)s heart.vtp -o heart.usd --appearance anatomy --anatomy-type heart

  # Split by cell type (triangle vs quad), solid color
  %(prog)s mesh.vtk -o out.usd --by-cell-type --appearance solid
        """,
    )

    parser.add_argument(
        "vtk_files",
        nargs="+",
        help="One or more VTK files (.vtk, .vtp, .vtu). Multiple files form an "
        "ordered time series unless --static-merge is set.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output_usd",
        help="Output USD file path",
    )
    parser.add_argument(
        "--by-connectivity",
        action="store_true",
        dest="separate_by_connectivity",
        help="Split mesh into separate objects by connected components (default)",
    )
    parser.add_argument(
        "--by-cell-type",
        action="store_true",
        dest="separate_by_cell_type",
        help="Split mesh by cell type (triangle, quad, etc.). Cannot use with --by-connectivity",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Do not split; output a single mesh (clears --by-connectivity and --by-cell-type)",
    )
    parser.add_argument(
        "--static-merge",
        action="store_true",
        help="Treat multiple input files as separate static objects in one scene "
        "(no time samples) instead of an ordered time series.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        dest="frames_per_second",
        help="Frames per second for time series (default: 60)",
    )
    parser.add_argument(
        "--no-extract-surface",
        action="store_false",
        dest="extract_surface",
        help="Do not extract surface for .vtu files",
    )

    # Appearance
    parser.add_argument(
        "--appearance",
        choices=["solid", "anatomy", "colormap"],
        default="solid",
        help="Appearance to apply to all meshes: solid color, anatomic material, or colormap from primvar (default: solid)",
    )
    parser.add_argument(
        "--color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        help="Solid color as R G B in [0,1] or [0,255] (default: 0.8 0.8 0.8). Used when --appearance solid.",
    )
    parser.add_argument(
        "--anatomy-type",
        choices=ANATOMY_TYPES,
        default=None,
        help=(
            "Anatomy material applied to every mesh when --appearance anatomy. "
            "Omit to pick a material per object from its name (default)."
        ),
    )
    parser.add_argument(
        "--primvar",
        dest="colormap_primvar",
        default=None,
        help="Primvar name for colormap (e.g. vtk_point_stress_c0). If omitted, one is auto-picked when --appearance colormap.",
    )
    parser.add_argument(
        "--cmap",
        dest="colormap_name",
        default="viridis",
        help="Matplotlib colormap name when --appearance colormap (default: viridis)",
    )
    parser.add_argument(
        "--intensity-range",
        nargs=2,
        type=float,
        metavar=("VMIN", "VMAX"),
        default=None,
        dest="colormap_intensity_range",
        help="Colormap range (vmin vmax). If omitted, range is computed from data. Use with --appearance colormap.",
    )

    args = parser.parse_args()

    # Resolve split mode
    if args.no_split:
        separate_by_connectivity = False
        separate_by_cell_type = False
    else:
        separate_by_connectivity = args.separate_by_connectivity
        separate_by_cell_type = args.separate_by_cell_type
        if not separate_by_connectivity and not separate_by_cell_type:
            separate_by_connectivity = True  # default

    if separate_by_connectivity and separate_by_cell_type:
        print("Error: --by-connectivity and --by-cell-type cannot both be set.")
        return 1

    # Solid color
    solid_color = (0.8, 0.8, 0.8)
    if args.color:
        try:
            if isinstance(args.color, (list, tuple)):
                components = [float(v) for v in args.color]
                if len(components) != 3:
                    raise ValueError(
                        "Color must have exactly three components (R G B)."
                    )
                # Interpret either as normalized [0, 1] or byte [0, 255] values, but do not mix scales.
                if all(0.0 <= v <= 1.0 for v in components):
                    solid_color = (components[0], components[1], components[2])
                elif all(0.0 <= v <= 255.0 for v in components):
                    solid_color = (
                        components[0] / 255.0,
                        components[1] / 255.0,
                        components[2] / 255.0,
                    )
                else:
                    raise ValueError(
                        "Color values must all be in [0, 1] or all in [0, 255]."
                    )
            else:
                raise ValueError("Color must be specified as a list of 3 float values")
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Colormap intensity range
    intensity_range = None
    if args.colormap_intensity_range is not None:
        intensity_range = tuple(args.colormap_intensity_range)

    # Validate input files
    for p in args.vtk_files:
        if not os.path.exists(p):
            print(f"Error: Input file not found: {p}")
            return 1

    output_path = Path(args.output_usd)
    if output_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        print(
            "Error: Output file must have a .usd, .usda, or .usdc extension, "
            f"got: {output_path}"
        )
        return 1

    try:
        from .. import WorkflowConvertVTKToUSD

        input_meshes = [pv.read(str(p)) for p in args.vtk_files]

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=input_meshes,
            usd_project_name=output_path.name,
            output_directory=output_path.parent,
            separate_by_connectivity=separate_by_connectivity,
            separate_by_cell_type=separate_by_cell_type,
            frames_per_second=args.frames_per_second,
            extract_surface=args.extract_surface,
            static_merge=args.static_merge,
            appearance=args.appearance,
            solid_color=solid_color,
            anatomy_type=args.anatomy_type,
            colormap_primvar=args.colormap_primvar,
            colormap_name=args.colormap_name,
            colormap_intensity_range=intensity_range,
        )
    except (ValueError, OSError, TypeError) as e:
        print(f"Error: {e}")
        return 1

    try:
        result = workflow.process()
        print("\nConversion completed successfully.")
        print(f"Output: {result['usd_file']}")
        return 0
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
