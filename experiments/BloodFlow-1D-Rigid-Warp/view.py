"""Browse the 1D trunk fields over the fixed source anatomy.

The coloured line shows section-mean values from run.py, not a 3D field.
Run with ``--screenshot PATH`` to save one frame, or ``--movie PATH.mp4`` to
record the last beat, instead of serving the page.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, ClassVar, Optional, cast

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent
MMHG = 133.322
LAST_BEAT_START_S = 2.01  # Third LV ejection onset, source sample 67.


class TrunkViewer:
    """Render saved trunk velocity or pressure over translucent anatomy."""

    LABELS: ClassVar[dict[str, str]] = {
        "velocity": "Section-mean velocity (m/s)",
        "pressure": "Pressure (mmHg)",
    }

    def __init__(self, preparation: Path, results: Path) -> None:
        """Load the saved fields and build an off-screen scene."""
        fields = np.load(results / "fields.npz")
        self.times: NDArray[np.float64] = fields["times_s"]
        self.flux: NDArray[np.float64] = fields["inlet_q_m3_s"]
        self.values = {
            "velocity": fields["velocity_m_s"],
            "pressure": fields["pressure_pa"] / MMHG,
        }
        self.limits = {
            k: (float(v.min()), float(v.max())) for k, v in self.values.items()
        }
        curve = np.load(preparation / "geometry.npz")["centerline_m"]
        self.line = pv.lines_from_points(curve)
        self.source = cast(pv.PolyData, pv.read(preparation / "source.vtp"))
        self.index = int(np.argmax(self.flux))
        self.field = "velocity"
        self.plotter: Any = cast(
            Any, pv.Plotter(off_screen=True, window_size=[1100, 800])
        )
        self.plotter.set_background("#101c2b")
        self.anatomy_actor = self.plotter.add_mesh(
            self.source, color="#cbd6e2", opacity=0.18, smooth_shading=True
        )
        self.line["display"] = self.values[self.field][self.index]
        self.line_actor = self.plotter.add_mesh(
            self.line,
            scalars="display",
            cmap="turbo",
            line_width=11,
            render_lines_as_tubes=True,
            lighting=False,
            show_scalar_bar=False,
        )
        self.scalar_bar = self.plotter.add_scalar_bar(
            self.LABELS[self.field],
            mapper=self.line_actor.mapper,
            color="white",
            vertical=True,
            position_x=0.83,
            position_y=0.2,
            width=0.11,
            height=0.55,
            fmt="%.3g",
        )
        self.plotter.add_text(
            "Rigid sample-69 anatomy; the line shows 1D section means\n"
            "RCR outlet; rigid wall; no branch flow",
            position="lower_left",
            font_size=10,
            color="#d2dfec",
        )
        self.plotter.add_axes()
        self.reset_camera()
        self.update()

    def update(self) -> None:
        """Show the current field and time with fixed all-time color limits."""
        self.line["display"] = self.values[self.field][self.index]
        self.line_actor.mapper.scalar_range = self.limits[self.field]
        self.scalar_bar.SetTitle(self.LABELS[self.field])
        self.plotter.add_text(
            f"t = {self.times[self.index]:.3f} s\n"
            f"Q = {self.flux[self.index] * 1e6:.1f} mL/s",
            position="upper_left",
            font_size=12,
            color="white",
            name="time",
        )
        self.plotter.render()

    def set_time(self, index: int) -> None:
        """Show a saved time sample."""
        self.index = int(np.clip(index, 0, len(self.times) - 1))
        self.update()

    def set_field(self, field: str) -> None:
        """Show velocity or pressure."""
        if field not in self.LABELS:
            raise ValueError(f"Unknown field: {field}")
        self.field = field
        self.update()

    def set_opacity(self, opacity: float) -> None:
        """Change the anatomy opacity."""
        self.anatomy_actor.prop.opacity = float(np.clip(opacity, 0, 1))
        self.plotter.render()

    def reset_camera(self) -> None:
        """Frame the whole source anatomy from the front."""
        self.plotter.view_xy()
        self.plotter.reset_camera(bounds=self.source.bounds)
        self.plotter.camera.Zoom(0.78)
        self.plotter.render()

    def screenshot(self, path: Path) -> NDArray[np.uint8]:
        """Save the rendered view and return its pixels."""
        self.plotter.render()
        return cast(NDArray[np.uint8], np.asarray(self.plotter.screenshot(str(path))))

    def waveform_svg(self) -> str:
        """Inlet flow over time with a cursor at the displayed sample."""
        x = 35 + self.times / self.times[-1] * 530
        y = 135 - self.flux / self.flux.max() * 105
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x[::3], y[::3]))
        cursor = x[self.index]
        return (
            '<svg viewBox="0 0 600 175"><rect width="600" height="175" '
            'fill="#142337" rx="8"/><path d="M35 20 V135 H565" fill="none" '
            f'stroke="#70849a"/><polyline points="{points}" fill="none" '
            f'stroke="#6fd3d6" stroke-width="2"/><line x1="{cursor:.1f}" '
            f'x2="{cursor:.1f}" y1="20" y2="135" stroke="#ffce72" stroke-width="2"/>'
            '<g fill="#d9e5f1" font-family="sans-serif" font-size="12">'
            f'<text x="35" y="158">0</text><text x="545" y="158">'
            f'{self.times[-1]:.0f} s</text><text x="43" y="20">Q peak '
            f"{self.flux.max() * 1e6:.0f} mL/s</text></g></svg>"
        )


def create_app(viewer: TrunkViewer) -> Any:
    """Build the local Trame page with time, field and opacity controls."""
    from trame.app import get_server  # type: ignore[import-untyped]
    from trame.ui.vuetify3 import SinglePageLayout  # type: ignore[import-untyped]
    from trame.widgets import html, vuetify3  # type: ignore[import-untyped]
    from trame_pyvista import plotter_ui

    server = get_server("aorta-1d", client_type="vue3")
    state, controller = server.state, server.controller
    state.waveform_svg = viewer.waveform_svg()
    task: Optional[asyncio.Task[None]] = None

    def refresh() -> None:
        state.waveform_svg = viewer.waveform_svg()
        controller.view_update()

    def time_changed(time_index: int, **_: Any) -> None:
        viewer.set_time(int(time_index))
        refresh()

    def field_changed(field: str, **_: Any) -> None:
        viewer.set_field(field)
        refresh()

    def opacity_changed(opacity: float, **_: Any) -> None:
        viewer.set_opacity(float(opacity))
        refresh()

    state.change("time_index")(time_changed)
    state.change("field")(field_changed)
    state.change("opacity")(opacity_changed)

    async def animate() -> None:
        origin = time.perf_counter() - float(viewer.times[viewer.index])
        while state.playing:
            elapsed = (time.perf_counter() - origin) % viewer.times[-1]
            with state:
                state.time_index = int(np.searchsorted(viewer.times, elapsed))
            await asyncio.sleep(1 / 30)

    def toggle() -> None:
        nonlocal task
        state.playing = not state.playing
        if state.playing:
            task = asyncio.create_task(animate())
        elif task is not None:
            task.cancel()
            task = None

    def reset() -> None:
        viewer.reset_camera()
        controller.view_update()

    controller.toggle_play, controller.reset_view = toggle, reset
    with SinglePageLayout(server) as layout:
        layout.title.set_text("Rigid aorta | 1D blood flow")
        with (
            layout.content,
            vuetify3.VContainer(fluid=True, classes="pa-0 fill-height"),
            vuetify3.VRow(no_gutters=True, classes="fill-height"),
        ):
            with vuetify3.VCol(cols=9, classes="fill-height"):
                view = cast(Any, plotter_ui)(
                    viewer.plotter, server=server, mode="server", add_menu=False
                )
                controller.view_update = view.update
            with vuetify3.VCol(cols=3, classes="pa-5 bg-grey-darken-4"):
                vuetify3.VSelect(
                    label="Displayed field",
                    v_model=("field", viewer.field),
                    items=(
                        "field_items",
                        [{"title": t, "value": k} for k, t in viewer.LABELS.items()],
                    ),
                    density="compact",
                )
                vuetify3.VSlider(
                    label="Time",
                    v_model=("time_index", viewer.index),
                    min=0,
                    max=len(viewer.times) - 1,
                    step=1,
                    hide_details=True,
                )
                vuetify3.VBtn(
                    "{{ playing ? 'Pause' : 'Play' }}",
                    click=controller.toggle_play,
                    color="teal-lighten-2",
                    classes="mb-4",
                )
                html.Div(v_html="waveform_svg", classes="mb-4")
                vuetify3.VSlider(
                    label="Anatomy opacity",
                    v_model=("opacity", 0.18),
                    min=0,
                    max=1,
                    step=0.01,
                    hide_details=True,
                )
                vuetify3.VBtn(
                    "Reset camera",
                    click=controller.reset_view,
                    variant="outlined",
                    classes="my-4",
                )
                html.P(
                    "Rigid-tube inertia and Poiseuille drag along the trunk, "
                    "added to a three-element Windkessel outlet pressure. "
                    "Colour limits are fixed over all times.",
                    classes="text-caption",
                )
    state.playing = False
    return server


def record_movie(
    viewer: TrunkViewer, path: Path, frames_per_second: float = 30.0
) -> dict[str, Any]:
    """Write the last beat as an MP4 with color limits fixed over that beat."""
    import imageio.v2 as imageio

    indices = np.flatnonzero(viewer.times >= LAST_BEAT_START_S)[::2]
    beat = viewer.values[viewer.field][indices]
    viewer.limits[viewer.field] = (float(beat.min()), float(beat.max()))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: Any = imageio.get_writer(
        path, fps=frames_per_second, codec="libx264", macro_block_size=4
    )
    with writer:
        for index in indices:
            viewer.set_time(int(index))
            writer.append_data(np.asarray(viewer.plotter.screenshot()))
    return {
        "movie": str(path),
        "field": viewer.field,
        "frames": len(indices),
        "duration_s": len(indices) / frames_per_second,
        "color_limits": viewer.limits[viewer.field],
    }


def main() -> None:
    """Serve the viewer or save one frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preparation", type=Path, default=ROOT / "output" / "prepared"
    )
    parser.add_argument("--results", type=Path, default=ROOT / "output" / "flow")
    parser.add_argument("--port", type=int, default=8701)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--movie", type=Path, help="Record the last beat as MP4")
    parser.add_argument(
        "--field", choices=sorted(TrunkViewer.LABELS), default="velocity"
    )
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    viewer = TrunkViewer(args.preparation, args.results)
    viewer.set_field(args.field)
    if args.index is not None:
        viewer.set_time(args.index)
    try:
        if args.movie:
            print(json.dumps(record_movie(viewer, args.movie.resolve()), indent=2))
        elif args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            viewer.screenshot(args.screenshot)
            print(json.dumps({"time_s": float(viewer.times[viewer.index])}))
        else:
            create_app(viewer).start(port=args.port, open_browser=not args.no_browser)
    finally:
        viewer.plotter.close()


if __name__ == "__main__":
    main()
