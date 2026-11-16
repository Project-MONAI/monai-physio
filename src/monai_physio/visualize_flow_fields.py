"""Interactive browser views and off-screen movies of saved P2/P1 flow fields.

A results folder written by a :class:`WarpNavierStokes` run holds
``mesh.npz`` (``points_m``, ``tetrahedra``, ``edges``), ``time.npz``
(``frame_times_s``, ``frame_inlet_flow_m3_s``), ``velocity_m_s.npy`` (frames x
canonical P2 nodes x 3) and ``pressure_pa.npy`` (frames x vertices). A moving
wall adds ``displacement_m.npy`` (frames x vertices x 3), which moves the mesh
and is offered as a third field. Fields are shown on genuine ten-node
quadratic tetrahedra over optional fixed source anatomy.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, ClassVar, Optional, cast

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from .monai_physio_base import MONAIPhysioBase

MMHG = 133.322
VTK_TET_EDGES = [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]]


class VisualizeFlowFields(MONAIPhysioBase):
    """Render saved velocity and pressure frames with time and slice controls.

    Automatic speed ranges stop at the 99.5th percentile of the shown values,
    so a few fast nodes at small outlets do not hide the interior flow.
    """

    SPEED_PERCENTILE = 99.5

    FIELDS: ClassVar[dict[str, tuple[str, str]]] = {
        "velocity": ("Speed (m/s)", "turbo"),
        "pressure": ("Pressure (mmHg)", "plasma"),
    }
    COLOR_MODES: ClassVar[dict[str, str]] = {
        "frame": "Current frame (surface + slice)",
        "slice": "Current slice (surface colors clipped)",
        "global": "Fixed range",
    }

    def __init__(
        self,
        results: Path,
        anatomy: Optional[Path] = None,
        *,
        caption: str = "",
        off_screen: bool = True,
    ) -> None:
        """Load the saved frames and build the scene.

        Args:
            results: Folder written by the flow run.
            anatomy: Optional surface file shown as translucent context.
            caption: Text shown in the lower left corner.
            off_screen: Render without a desktop window.
        """
        super().__init__(class_name=self.__class__.__name__)
        mesh = np.load(results / "mesh.npz")
        points, tets, self.edges = mesh["points_m"], mesh["tetrahedra"], mesh["edges"]
        self.nodes = np.concatenate([points, points[self.edges].mean(axis=1)])
        keys = self.edges[:, 0] * len(points) + self.edges[:, 1]
        local = np.sort(tets[:, VTK_TET_EDGES], axis=2)
        edge_ids = np.searchsorted(keys, local[:, :, 0] * len(points) + local[:, :, 1])
        self.grid: Any = pv.UnstructuredGrid(
            np.c_[np.full(len(tets), 10), tets, len(points) + edge_ids].ravel(),
            np.full(len(tets), pv.CellType.QUADRATIC_TETRA, dtype=np.uint8),
            self.nodes.copy(),
        )
        self.velocity = np.load(results / "velocity_m_s.npy", mmap_mode="r")
        self.pressure = np.load(results / "pressure_pa.npy", mmap_mode="r")
        moving = results / "displacement_m.npy"
        self.displacement = np.load(moving, mmap_mode="r") if moving.is_file() else None
        clock = np.load(results / "time.npz")
        self.times: NDArray[np.float64] = clock["frame_times_s"]
        self.flux: NDArray[np.float64] = clock["frame_inlet_flow_m3_s"]
        self.fields = dict(self.FIELDS)
        if self.displacement is not None:
            self.fields["displacement"] = ("Wall displacement (mm)", "viridis")
        frames = np.arange(len(self.times))
        self.limits = {name: self.field_limits(name, frames) for name in self.fields}
        self.index = int(np.argmax(self.flux))
        self.field = "pressure"
        self.color_mode = "frame"
        self.vector_scale = 0.05
        self.opacity = 0.15
        self.slice_axis = "z"
        self.slice_position = 0.5
        self.vectors_visible = False
        self.playing = False
        self.plotter: Any = cast(
            Any, pv.Plotter(off_screen=off_screen, window_size=[1100, 800])
        )
        self.plotter.set_background("#101c2b")
        self.anatomy: Optional[pv.PolyData] = None
        self.anatomy_actor: Any = None
        if anatomy is not None:
            self.anatomy = cast(pv.PolyData, pv.read(anatomy))
            self.anatomy_actor = self.plotter.add_mesh(
                self.anatomy,
                color="#cbd6e2",
                opacity=self.opacity,
                smooth_shading=True,
                name="anatomy",
            )
        self.grid["velocity_m_s"] = np.zeros((len(self.nodes), 3))
        self.grid["display"] = np.zeros(len(self.nodes))
        self.surface: Any = self.grid.extract_surface(
            nonlinear_subdivision=2, algorithm="dataset_surface"
        )
        self.field_actor = self.plotter.add_mesh(
            self.surface,
            scalars="display",
            opacity=0.5,
            show_scalar_bar=False,
            name="lumen",
        )
        self.scalar_bar = self.plotter.add_scalar_bar(
            self.fields[self.field][0],
            mapper=self.field_actor.mapper,
            color="white",
            vertical=True,
            position_x=0.84,
            position_y=0.22,
            width=0.1,
            height=0.5,
            title_font_size=13,
            label_font_size=12,
            n_labels=5,
            fmt="%.3g",
        )
        self.plotter.add_axes()
        self.plotter.add_text(
            caption,
            position="lower_left",
            font_size=10,
            color="#d2dfec",
            name="caption",
        )
        self.reset_camera()
        self.update()

    def field_limits(
        self, field: str, frames: NDArray[np.int64]
    ) -> tuple[float, float]:
        """Fixed color limits of a field over the given frames.

        Speed and wall displacement start at 0; pressure spans its range.
        """
        if field == "pressure":
            pressure = np.asarray(self.pressure[frames]) / MMHG
            return float(pressure.min()), float(pressure.max())
        if field == "velocity":
            speed = np.linalg.norm(self.velocity[frames], axis=2)
            return 0.0, float(np.percentile(speed, self.SPEED_PERCENTILE))
        assert self.displacement is not None
        magnitude = np.linalg.norm(self.displacement[frames], axis=2)
        return 0.0, 1e3 * float(magnitude.max())

    def update(self) -> None:
        """Show the current frame on the lumen surface and the slice."""
        velocity = np.array(self.velocity[self.index])
        corner = np.asarray(self.pressure[self.index]) / MMHG
        if self.displacement is not None:
            moved = np.asarray(self.displacement[self.index])
            self.grid.points = self.nodes + np.concatenate(
                [moved, moved[self.edges].mean(axis=1)]
            )
            if self.field == "displacement":
                corner = 1e3 * np.linalg.norm(moved, axis=1)
        self.grid["velocity_m_s"] = velocity
        self.grid["display"] = (
            np.linalg.norm(velocity, axis=1)
            if self.field == "velocity"
            else np.r_[corner, corner[self.edges].mean(axis=1)]
        )
        self.surface = self.grid.extract_surface(
            nonlinear_subdivision=2, algorithm="dataset_surface"
        )
        if self.field == "velocity":
            # Interpolate vectors before taking the norm on subdivided cells.
            self.surface["display"] = np.linalg.norm(
                self.surface["velocity_m_s"], axis=1
            )
        self.field_actor.mapper.dataset = self.surface
        self.field_actor.mapper.lookup_table.cmap = self.fields[self.field][1]
        self.scalar_bar.SetTitle(self.fields[self.field][0])
        self.update_slice()
        self.plotter.add_text(
            f"t = {self.times[self.index]:.3f} s\n"
            f"{self.fields[self.field][0]}; {self.COLOR_MODES[self.color_mode]}\n"
            f"Q = {self.flux[self.index] * 1e6:.1f} mL/s",
            position="upper_left",
            font_size=11,
            color="white",
            name="current-time",
        )
        self.plotter.render()

    def update_slice(self) -> None:
        """Probe the quadratic volume on the current plane, with velocity glyphs."""
        axis = "xyz".index(self.slice_axis)
        bounds = np.asarray(self.grid.bounds).reshape(3, 2)
        origin = bounds.mean(axis=1)
        origin[axis] = bounds[axis, 0] + self.slice_position * np.ptp(bounds[axis])
        cut = self.grid.slice(normal=self.slice_axis, origin=origin)
        self.slice: Any = cut.sample(self.grid, tolerance=1e-10)
        if self.slice.n_points and self.field == "velocity":
            self.slice["display"] = np.linalg.norm(self.slice["velocity_m_s"], axis=1)
        self.plotter.remove_actor("slice", render=False)
        self.plotter.remove_actor("vectors", render=False)
        self.slice_actor: Any = None
        self.vector_actor: Any = None
        self.glyphs: Any = pv.PolyData()
        self.update_color_range()
        if not self.slice.n_points:
            return
        self.slice_actor = self.plotter.add_mesh(
            self.slice,
            scalars="display",
            clim=self.display_limits,
            cmap=self.fields[self.field][1],
            lighting=False,
            show_scalar_bar=False,
            reset_camera=False,
            name="slice",
        )
        stride = max(1, self.slice.n_points // 180)
        samples = pv.PolyData(np.asarray(self.slice.points)[::stride].copy())
        samples["velocity_m_s"] = np.asarray(self.slice["velocity_m_s"])[::stride]
        samples["speed"] = np.linalg.norm(samples["velocity_m_s"], axis=1)
        self.glyphs = samples.glyph(
            orient="velocity_m_s", scale="speed", factor=self.vector_scale
        )
        self.vector_actor = self.plotter.add_mesh(
            self.glyphs,
            color="#ffffff",
            show_scalar_bar=False,
            reset_camera=False,
            name="vectors",
        )
        self.vector_actor.SetVisibility(self.vectors_visible)

    def update_color_range(self) -> None:
        """Share one color range between the lumen surface and the slice."""
        if self.color_mode == "global":
            low, high = self.limits[self.field]
        else:
            values = [np.asarray(self.slice["display"])] if self.slice.n_points else []
            if self.color_mode == "frame" or not values:
                values.append(np.asarray(self.surface["display"]))
            shown = np.concatenate(values)
            low = float(shown.min())
            high = float(
                np.percentile(shown, self.SPEED_PERCENTILE)
                if self.field == "velocity"
                else shown.max()
            )
            if low == high:
                low, high = low - 1e-6, high + 1e-6
        self.display_limits = (float(low), float(high))
        self.field_actor.mapper.scalar_range = self.display_limits

    def set_color_mode(self, mode: str) -> None:
        """Select current-frame, slice-focused or fixed all-frame color limits."""
        if mode not in self.COLOR_MODES:
            raise ValueError(f"Unknown color mode: {mode}")
        self.color_mode = mode
        self.update()

    def set_time(self, index: int) -> None:
        """Show a saved frame."""
        self.index = int(np.clip(index, 0, len(self.times) - 1))
        self.update()

    def set_field(self, field: str) -> None:
        """Show speed, pressure or (for a moving wall) wall displacement."""
        if field not in self.fields:
            raise ValueError(f"Unknown field: {field}")
        self.field = field
        self.update()

    def set_slice(self, axis: str, position: float) -> None:
        """Move the slice to a fraction of the lumen bounds along an axis."""
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Unknown slice axis: {axis}")
        self.slice_axis = axis
        self.slice_position = float(np.clip(position, 0.01, 0.99))
        self.update_slice()
        self.plotter.render()

    def set_vectors(self, visible: bool) -> None:
        """Show or hide velocity arrows on the slice."""
        self.vectors_visible = bool(visible)
        if self.vector_actor is not None:
            self.vector_actor.SetVisibility(self.vectors_visible)
        self.plotter.render()

    def set_opacity(self, opacity: float) -> None:
        """Change the opacity of the anatomy overlay."""
        self.opacity = float(np.clip(opacity, 0, 1))
        if self.anatomy_actor is not None:
            self.anatomy_actor.prop.opacity = self.opacity
        self.plotter.render()

    def advance(self, elapsed: float) -> None:
        """Show the frame at ``elapsed`` seconds of looping playback."""
        if self.playing:
            moment = elapsed % self.times[-1]
            self.set_time(int(np.searchsorted(self.times, moment, side="right")) - 1)

    def rotate_camera(self, azimuth: float, elevation: float) -> None:
        """Rotate the camera about the focal point (degrees)."""
        self.plotter.camera.Azimuth(azimuth)
        self.plotter.camera.Elevation(elevation)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def zoom_camera(self, factor: float) -> None:
        """Zoom the camera."""
        self.plotter.camera.Zoom(factor)
        self.plotter.render()

    def reset_camera(self) -> None:
        """Frame the anatomy (or the lumen) from the front with Y up."""
        self.plotter.view_xy()
        self.plotter.camera.up = (0, 1, 0)
        bounds = (self.anatomy if self.anatomy is not None else self.grid).bounds
        self.plotter.reset_camera(bounds=bounds)
        self.plotter.camera.Zoom(0.85)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def screenshot(self, path: Path) -> NDArray[np.uint8]:
        """Save the rendered view and return its pixels."""
        self.plotter.render()
        return cast(NDArray[np.uint8], np.asarray(self.plotter.screenshot(str(path))))

    def snapshot(self) -> dict[str, Any]:
        """Describe the visible state."""
        return {
            "index": self.index,
            "time_s": float(self.times[self.index]),
            "field": self.field,
            "label": self.fields[self.field][0],
            "color_mode": self.color_mode,
            "display_limits": self.display_limits,
            "slice_axis": self.slice_axis,
            "slice_position": self.slice_position,
            "slice_points": self.slice.n_points,
            "vectors_visible": self.vectors_visible,
            "camera": [
                list(self.plotter.camera.GetPosition()),
                list(self.plotter.camera.GetFocalPoint()),
                list(self.plotter.camera.GetViewUp()),
            ],
        }

    def record_movie(
        self, path: Path, start_s: float = 0.0, frames_per_second: float = 10.0
    ) -> dict[str, Any]:
        """Write the saved frames from ``start_s`` on as an MP4 of this view.

        The color limits of the shown field are fixed over those frames.
        """
        import imageio.v2 as imageio

        frames = np.flatnonzero(self.times >= start_s - 1e-9)
        self.limits[self.field] = self.field_limits(self.field, frames)
        self.color_mode = "global"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer: Any = imageio.get_writer(
            path, fps=frames_per_second, codec="libx264", macro_block_size=4
        )
        with writer:
            for index in frames:
                self.set_time(int(index))
                writer.append_data(np.asarray(self.plotter.screenshot()))
        return {
            "movie": str(path),
            "field": self.field,
            "frames": len(frames),
            "times_s": [float(self.times[frames[0]]), float(self.times[frames[-1]])],
            "color_limits": self.limits[self.field],
        }


def create_app(viewer: VisualizeFlowFields, title: str, notes: list[str]) -> Any:
    """Build the local Trame browser interface around a viewer.

    Args:
        viewer: Scene to control.
        title: Toolbar title.
        notes: Short paragraphs describing the model and its limitations.

    Returns:
        The Trame server; call ``start(port=...)`` to serve it.
    """
    from trame.app import get_server  # type: ignore[import-untyped]
    from trame.ui.vuetify3 import SinglePageLayout  # type: ignore[import-untyped]
    from trame.widgets import html, vuetify3  # type: ignore[import-untyped]
    from trame_pyvista.ui import plotter_ui

    server = get_server("flow-fields", client_type="vue3")
    state, controller = server.state, server.controller
    state.time_label = f"{viewer.times[viewer.index]:.3f} s"
    task: Optional[asyncio.Task[None]] = None

    def refresh() -> None:
        state.time_label = f"{viewer.times[viewer.index]:.3f} s"
        controller.view_update()

    def bind(name: str, action: Any) -> None:
        def changed(**values: Any) -> None:
            action(values[name])
            refresh()

        state.change(name)(changed)

    bind("time_index", lambda v: viewer.set_time(int(v)))
    bind("field", viewer.set_field)
    bind("color_mode", viewer.set_color_mode)
    bind("opacity", lambda v: viewer.set_opacity(float(v)))
    bind("slice_axis", lambda v: viewer.set_slice(v, viewer.slice_position))
    bind("slice_position", lambda v: viewer.set_slice(viewer.slice_axis, float(v)))
    bind("vectors", viewer.set_vectors)

    async def animate() -> None:
        origin = time.perf_counter() - float(viewer.times[viewer.index])
        while state.playing:
            viewer.advance(time.perf_counter() - origin)
            with state:
                state.time_index = viewer.index
            refresh()
            await asyncio.sleep(0.1)

    def toggle() -> None:
        nonlocal task
        state.playing = not state.playing
        viewer.playing = state.playing
        if state.playing:
            task = asyncio.create_task(animate())
        elif task is not None:
            task.cancel()
            task = None

    def reset() -> None:
        viewer.reset_camera()
        refresh()

    controller.toggle_play, controller.reset_view = toggle, reset
    with SinglePageLayout(server) as layout:
        layout.title.set_text(title)
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
                        [
                            {"title": label, "value": name}
                            for name, (label, _) in viewer.fields.items()
                        ],
                    ),
                    density="compact",
                )
                html.P("Time: {{ time_label }}", id="visible-time")
                vuetify3.VSelect(
                    label="Color range",
                    v_model=("color_mode", viewer.color_mode),
                    items=(
                        "color_mode_items",
                        [
                            {"title": text, "value": mode}
                            for mode, text in viewer.COLOR_MODES.items()
                        ],
                    ),
                    density="compact",
                )
                vuetify3.VSlider(
                    id="time-slider",
                    label="Time",
                    v_model=("time_index", viewer.index),
                    min=0,
                    max=len(viewer.times) - 1,
                    step=1,
                    hide_details=True,
                )
                vuetify3.VBtn(
                    "{{ playing ? 'Pause' : 'Play' }}",
                    id="play-toggle",
                    click=controller.toggle_play,
                    color="teal-lighten-2",
                    classes="mb-4",
                )
                vuetify3.VSlider(
                    label="Anatomy opacity",
                    v_model=("opacity", viewer.opacity),
                    min=0,
                    max=1,
                    step=0.01,
                    hide_details=True,
                )
                vuetify3.VSelect(
                    label="Slice axis",
                    v_model=("slice_axis", viewer.slice_axis),
                    items=(
                        "axis_items",
                        [{"title": a.upper(), "value": a} for a in "xyz"],
                    ),
                    density="compact",
                )
                vuetify3.VSlider(
                    label="Slice position",
                    v_model=("slice_position", viewer.slice_position),
                    min=0.01,
                    max=0.99,
                    step=0.01,
                    hide_details=True,
                )
                vuetify3.VSwitch(
                    label="Velocity vectors",
                    v_model=("vectors", viewer.vectors_visible),
                    color="teal-lighten-2",
                    hide_details=True,
                )
                vuetify3.VBtn(
                    "Reset camera",
                    click=controller.reset_view,
                    variant="outlined",
                    classes="my-4",
                )
                html.P(
                    "Drag to rotate; scroll to zoom. Arrow length = velocity "
                    f"x {viewer.vector_scale} s.",
                    classes="text-caption mb-3",
                )
                for note in notes:
                    html.P(note, classes="text-caption mb-3")
    state.playing = False
    return server
