"""
Convert MP4 files into web-friendly animated GIFs.

Produces GIFs with a consistent width (384 px by default) and playback rate
(10 fps by default) so that a collection of them renders uniformly in a
documentation image gallery.  Uses the two-pass ffmpeg palette workflow
(``palettegen`` then ``paletteuse``) for good color fidelity at small size.

The ffmpeg binary bundled with ``imageio-ffmpeg`` is used when available,
otherwise ``ffmpeg`` from PATH.

Usage:
    python utils/mp4_to_gif.py [input.mp4 ...] [--width 384] [--fps 10]

If no inputs are given, every ``.mp4`` in ``docs/assets`` is converted.
Each GIF is written next to its source MP4 unless ``--output-dir`` is given.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# Default source of documentation videos, relative to the project root
ASSETS_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"


def _ffmpeg_exe() -> str:
    """Return the path to an ffmpeg executable."""
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        return "ffmpeg"


def convert(mp4_path: Path, gif_path: Path, width: int, fps: int) -> None:
    """Convert one MP4 to a GIF of the given width and frame rate."""
    ffmpeg = _ffmpeg_exe()
    # "-2" keeps the height even and preserves the aspect ratio.
    filters = f"fps={fps},scale={width}:-2:flags=lanczos"

    with tempfile.TemporaryDirectory() as tmp_dir:
        palette = Path(tmp_dir) / "palette.png"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4_path),
                "-vf",
                f"{filters},palettegen=stats_mode=diff",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4_path),
                "-i",
                str(palette),
                "-lavfi",
                (
                    f"{filters}[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=5"
                    ":diff_mode=rectangle"
                ),
                "-loop",
                "0",
                str(gif_path),
            ],
            check=True,
        )


def main() -> int:
    """Parse arguments and convert the requested MP4 files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="MP4 files to convert (default: all .mp4 in docs/assets)",
    )
    parser.add_argument("--width", type=int, default=384, help="GIF width in pixels")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the GIFs (default: beside each source MP4)",
    )
    args = parser.parse_args()

    inputs = args.inputs or sorted(ASSETS_DIR.glob("*.mp4"))
    if not inputs:
        print("No MP4 files found.", file=sys.stderr)
        return 1

    for mp4_path in inputs:
        out_dir = args.output_dir or mp4_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        gif_path = out_dir / f"{mp4_path.stem}.gif"
        convert(mp4_path, gif_path, args.width, args.fps)
        size_kb = gif_path.stat().st_size / 1024
        print(f"{mp4_path.name} -> {gif_path} ({size_kb:.0f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
