"""Browse, screenshot or record the saved 3D aortic flow fields.

By default this serves the interactive page at http://127.0.0.1:8702.
``--screenshot PATH`` saves one frame; ``--movie PATH`` records the reported
beat off screen into an MP4, with a slice 10 mm above the root, velocity arrows
and color limits fixed over that beat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from monai_physio.visualize_flow_fields import VisualizeFlowFields, create_app

ROOT = Path(__file__).resolve().parent
TITLE = "Rigid aorta: 3D blood flow"
CAPTION = (
    "Rigid sample-69 anatomy; P2/P1 Warp FEM; RCR outlets\n"
    "Demonstration: time and mesh convergence not established"
)
NOTES = [
    (
        "Rigid wall. LV-derived inlet flow with a blunt profile. Three-element "
        "Windkessel (RCR) outlets with backflow stabilization."
    ),
    (
        "Speed is zero at the no-slip wall; use the slice and velocity vectors "
        "to see interior flow. Automatic color ranges change with time; choose "
        "Fixed range to compare times."
    ),
    (
        "The arch branches are short stumps, and the LCCA stump has very few "
        "mesh nodes, so flow near the branch outlets is not resolved."
    ),
]


def main() -> None:
    """Serve the viewer, save one frame or record a movie."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preparation", type=Path, default=ROOT / "output" / "prepared"
    )
    parser.add_argument("--results", type=Path, default=ROOT / "output" / "flow")
    parser.add_argument("--port", type=int, default=8702)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--movie", type=Path)
    parser.add_argument(
        "--field", choices=sorted(VisualizeFlowFields.FIELDS), default="pressure"
    )
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    anatomy = args.preparation / "source.vtp"
    viewer = VisualizeFlowFields(args.results, anatomy, caption=CAPTION)
    try:
        viewer.set_field(args.field)
        if args.index is not None:
            viewer.set_time(args.index)
        if args.movie:
            ports = json.loads((args.preparation / "ports.json").read_text())
            low, high = viewer.grid.bounds[2:4]
            root = ports["root_inlet"]["center_m"][1]
            viewer.set_slice("y", (root + 0.010 - low) / (high - low))
            viewer.set_vectors(True)
            viewer.rotate_camera(30, 25)
            run = json.loads((args.results / "run.json").read_text())
            start = run["config"]["report_from_s"]
            report = viewer.record_movie(args.movie.resolve(), start)
            print(json.dumps(report, indent=2))
        elif args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            viewer.screenshot(args.screenshot)
            print(json.dumps(viewer.snapshot(), indent=2))
        else:
            create_app(viewer, TITLE, NOTES).start(
                port=args.port, open_browser=not args.no_browser
            )
    finally:
        viewer.plotter.close()


if __name__ == "__main__":
    main()
