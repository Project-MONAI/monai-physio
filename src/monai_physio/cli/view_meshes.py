"""Command-line interface for the interactive mesh web viewer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

_USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}


def main() -> int:
    """Run the Trame mesh viewer for USD or VTP input."""
    parser = argparse.ArgumentParser(
        description=(
            "Preview one static or animated USD file, or overlay one or more "
            "VTP surfaces, in a web browser."
        ),
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="One USD file or one or more VTP surface files to preview.",
    )
    parser.add_argument(
        "--prim-path",
        default="/World",
        help="Root prim to display (default: /World).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to listen on (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="TCP port for the viewer (default: 8080).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Playback FPS override (default: use the USD stage's "
            "timeCodesPerSecond)."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a local browser (useful in Docker).",
    )
    args = parser.parse_args()

    from monai_physio.mesh_web_viewer import MeshWebViewer

    missing_files = [path for path in args.input_files if not path.is_file()]
    if missing_files:
        parser.error(f"Viewer input file not found: {missing_files[0]}")
    suffixes = {path.suffix.lower() for path in args.input_files}
    is_usd = len(args.input_files) == 1 and suffixes <= _USD_SUFFIXES
    is_vtp = suffixes == {".vtp"}
    if not is_usd and not is_vtp:
        parser.error("input must be one USD file or one or more VTP files")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0.0):
        parser.error("--fps must be finite and greater than zero")
    if is_vtp and args.fps is not None:
        parser.error("--fps is only supported for USD input")

    viewer = MeshWebViewer(
        args.input_files,
        prim_path=args.prim_path,
        playback_fps=args.fps,
    )
    viewer.start(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
