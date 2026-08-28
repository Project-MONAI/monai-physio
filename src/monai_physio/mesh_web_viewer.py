"""Interactive Trame mesh viewer for OpenUSD and VTP inputs."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional, Union

import numpy as np
import pyvista as pv
from pxr import Usd, UsdGeom

from .monai_physio_base import MONAIPhysioBase
from .usd_tools import USDTools


class MeshWebViewer(MONAIPhysioBase):
    """Serve USD or VTP mesh data as a Trame application.

    USD input supports frame scrubbing and looping playback. Multiple VTP
    surfaces are rendered as distinctly colored overlays with visibility
    controls. Both modes support orbit, pan, and zoom through PyVista/VTK.
    USD preview materials are reduced to their diffuse colors and opacity.
    """

    _USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
    _SURFACE_COLORS = (
        "#ff6b6b",
        "#51cf66",
        "#4dabf7",
        "#ffd43b",
        "#cc5de8",
        "#22b8cf",
    )

    def __init__(
        self,
        input_files: Union[
            str,
            Path,
            Sequence[Union[str, Path]],
        ],
        prim_path: str = "/World",
        log_level: Union[int, str] = logging.INFO,
        playback_fps: Optional[float] = None,
    ) -> None:
        """Initialize a viewer for one USD stage or one or more VTP surfaces.

        Args:
            input_files: One USD path or one or more VTP surface paths.
            prim_path: Root prim displayed when the input is USD.
            log_level: Logging level for viewer messages.
            playback_fps: Optional playback-rate override. By default, use the
                USD stage's ``timeCodesPerSecond`` metadata. VTP input does not
                support playback.

        Raises:
            FileNotFoundError: If an input file does not exist.
            ValueError: If inputs mix formats, contain an unsupported format,
                the USD stage or root prim cannot be opened, or playback is
                invalid for the selected input.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        if isinstance(input_files, (str, Path)):
            paths = (Path(input_files).resolve(),)
        else:
            paths = tuple(Path(path).resolve() for path in input_files)
        if not paths:
            raise ValueError("At least one input file is required")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Viewer input file not found: {path}")

        suffixes = {path.suffix.lower() for path in paths}
        is_usd = len(paths) == 1 and suffixes <= self._USD_SUFFIXES
        is_vtp = suffixes == {".vtp"}
        if not is_usd and not is_vtp:
            raise ValueError("Input must be one USD file or one or more VTP files")
        if is_vtp and playback_fps is not None:
            raise ValueError("playback_fps is only supported for USD input")

        self.input_files = paths
        self.source_kind = "usd" if is_usd else "vtp"
        self.usd_file: Optional[Path] = paths[0] if is_usd else None
        self.prim_path = prim_path
        if playback_fps is not None and (
            not math.isfinite(playback_fps) or playback_fps <= 0.0
        ):
            raise ValueError("playback_fps must be finite and greater than zero")

        self._usd_tools = USDTools(log_level=log_level)
        self._stage: Optional[Any] = None
        self._surface_meshes: tuple[pv.PolyData, ...] = ()
        self._surface_labels: tuple[str, ...] = ()
        if is_usd:
            assert self.usd_file is not None
            self._stage = Usd.Stage.Open(str(self.usd_file))
            if self._stage is None:
                raise ValueError(f"Could not open USD file: {self.usd_file}")
            root_prim = self._stage.GetPrimAtPath(self.prim_path)
            if not root_prim.IsValid():
                raise ValueError(f"USD prim path not found: {self.prim_path}")
            self._time_codes = self._collect_time_codes(root_prim)
            self._stage_frames_per_second = float(self._stage.GetTimeCodesPerSecond())
            if self._stage_frames_per_second <= 0.0:
                self._stage_frames_per_second = 24.0
            self._frames_per_second = (
                float(playback_fps)
                if playback_fps is not None
                else self._stage_frames_per_second
            )
        else:
            self._surface_meshes = tuple(self._read_vtp(path) for path in paths)
            self._surface_labels = self._make_surface_labels(paths)
            self._time_codes = (0.0,)
            self._stage_frames_per_second = 0.0
            self._frames_per_second = 0.0

        self._server: Optional[Any] = None
        self._state: Optional[Any] = None
        self._plotter: Optional[pv.Plotter] = None
        self._view: Optional[Any] = None
        self._display_mesh: Optional[pv.PolyData] = None
        self._point_frames: Optional[tuple[np.ndarray, ...]] = None
        self._surface_actors: list[Any] = []

    @property
    def time_codes(self) -> tuple[float, ...]:
        """Return the ordered USD time codes available for playback."""
        return self._time_codes

    @property
    def frames_per_second(self) -> float:
        """Return the effective playback rate in frames per second."""
        return self._frames_per_second

    @property
    def stage_frames_per_second(self) -> float:
        """Return the authored USD playback rate, or zero for VTP input."""
        return self._stage_frames_per_second

    @property
    def surface_meshes(self) -> tuple[pv.PolyData, ...]:
        """Return the loaded VTP surfaces, or an empty tuple for USD input."""
        return self._surface_meshes

    @property
    def surface_labels(self) -> tuple[str, ...]:
        """Return display labels for loaded VTP surfaces."""
        return self._surface_labels

    def mesh_at_index(self, frame_index: int) -> pv.PolyData:
        """Evaluate and return scene geometry at a playback frame.

        Parameters
        ----------
        frame_index : int
            Zero-based index into :attr:`time_codes`.

        Returns
        -------
        pyvista.PolyData
            Merged USD scene geometry or merged VTP surface geometry. USD
            output includes per-point preview colors.

        Raises
        ------
        IndexError
            If ``frame_index`` is outside the available frames.
        """
        if frame_index < 0 or frame_index >= len(self._time_codes):
            raise IndexError(f"Frame index out of range: {frame_index}")
        if self.source_kind == "vtp":
            return pv.merge(self._surface_meshes)
        assert self.usd_file is not None
        return self._usd_tools.load_usd_as_vtk(
            self.usd_file,
            prim_path=self.prim_path,
            time_code=self._time_codes[frame_index],
        )

    def start(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        open_browser: bool = True,
    ) -> None:
        """Build and start the Trame server.

        Parameters
        ----------
        host : str
            Interface on which the web server listens.
        port : int
            TCP port exposed by the server.
        open_browser : bool
            Open the viewer in the default local browser.

        Raises
        ------
        ImportError
            If the optional ``viewer`` dependencies are missing.
        """
        self._build_application(host)
        assert self._server is not None
        names = ", ".join(path.name for path in self.input_files)
        self.log_info("Serving %s at http://%s:%d", names, host, port)
        self._server.start(port=port, open_browser=open_browser)

    @staticmethod
    def _read_vtp(path: Path) -> pv.PolyData:
        """Read and validate one VTP surface."""
        mesh = pv.read(path)
        if not isinstance(mesh, pv.PolyData):
            raise ValueError(f"VTP input is not polygonal surface data: {path}")
        if mesh.n_points == 0:
            raise ValueError(f"VTP input contains no points: {path}")
        return mesh

    @staticmethod
    def _make_surface_labels(paths: tuple[Path, ...]) -> tuple[str, ...]:
        """Create concise labels by removing a shared filename prefix."""
        stems = [path.stem for path in paths]
        prefix = os.path.commonprefix(stems) if len(stems) > 1 else ""
        while prefix and not prefix.endswith(("_", "-", " ")):
            prefix = prefix[:-1]
        labels = []
        for stem in stems:
            short_name = stem[len(prefix) :] if prefix else stem
            labels.append(short_name.replace("_", " ").replace("-", " ").title())
        return tuple(labels)

    @staticmethod
    def _collect_time_codes(root_prim: Any) -> tuple[float, ...]:
        """Collect mesh point samples beneath a root prim."""
        time_codes: set[float] = set()
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                time_codes.update(
                    float(value) for value in mesh.GetPointsAttr().GetTimeSamples()
                )
        if not time_codes:
            return (0.0,)
        return tuple(sorted(time_codes))

    def _preload_point_frames(
        self,
        initial_mesh: pv.PolyData,
    ) -> Optional[tuple[np.ndarray, ...]]:
        """Preload animated points when the scene has fixed mesh topology."""
        if self._stage is None:
            return None
        root_prim = self._stage.GetPrimAtPath(self.prim_path)
        mesh_prims = [
            prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Mesh)
        ]
        if not mesh_prims:
            return None

        for prim in mesh_prims:
            mesh = UsdGeom.Mesh(prim)
            if (
                mesh.GetFaceVertexCountsAttr().ValueMightBeTimeVarying()
                or mesh.GetFaceVertexIndicesAttr().ValueMightBeTimeVarying()
            ):
                self.log_warning(
                    "Scene topology varies over time; using per-frame USD reloads"
                )
                return None

        started = time.perf_counter()
        point_frames: list[np.ndarray] = []
        expected_chunk_sizes: Optional[tuple[int, ...]] = None
        for time_code in self._time_codes:
            usd_time = Usd.TimeCode(time_code)
            xform_cache = UsdGeom.XformCache(usd_time)
            point_chunks: list[np.ndarray] = []
            for prim in mesh_prims:
                mesh = UsdGeom.Mesh(prim)
                points_value = mesh.GetPointsAttr().Get(usd_time)
                face_counts = mesh.GetFaceVertexCountsAttr().Get(usd_time)
                face_indices = mesh.GetFaceVertexIndicesAttr().Get(usd_time)
                if (
                    points_value is None
                    or len(points_value) == 0
                    or face_counts is None
                    or face_indices is None
                ):
                    continue

                world_matrix = xform_cache.GetLocalToWorldTransform(prim)
                matrix = np.array(
                    [[float(world_matrix[i][j]) for j in range(4)] for i in range(4)],
                    dtype=np.float64,
                )
                local_points = np.asarray(points_value, dtype=np.float64)
                homogeneous = np.empty(
                    (local_points.shape[0], 4),
                    dtype=np.float64,
                )
                homogeneous[:, :3] = local_points
                homogeneous[:, 3] = 1.0
                point_chunks.append((homogeneous @ matrix)[:, :3].astype(np.float32))

            chunk_sizes = tuple(len(points) for points in point_chunks)
            if expected_chunk_sizes is None:
                expected_chunk_sizes = chunk_sizes
            if chunk_sizes != expected_chunk_sizes or not point_chunks:
                self.log_warning(
                    "Scene point counts vary over time; using per-frame USD reloads"
                )
                return None
            point_frames.append(np.concatenate(point_chunks, axis=0))

        if any(points.shape != initial_mesh.points.shape for points in point_frames):
            self.log_warning(
                "Preloaded point layout does not match the display mesh; "
                "using per-frame USD reloads"
            )
            return None
        if not np.allclose(point_frames[0], initial_mesh.points):
            self.log_warning(
                "Preloaded point order does not match the display mesh; "
                "using per-frame USD reloads"
            )
            return None

        frames = tuple(point_frames)
        cache_mib = sum(points.nbytes for points in frames) / (1024.0**2)
        self.log_info(
            "Preloaded %d point frames (%.1f MiB) in %.2f seconds",
            len(frames),
            cache_mib,
            time.perf_counter() - started,
        )
        return frames

    def _build_application(self, host: str) -> None:
        """Create the Trame server, VTK view, and controls."""
        if self._server is not None:
            return

        try:
            from trame.app import get_server
            from trame.ui.vuetify3 import SinglePageLayout
            from trame.widgets import vtk as vtk_widgets
            from trame.widgets import vuetify3 as v3
        except ImportError as error:
            raise ImportError(
                "The web viewer requires optional dependencies. Install "
                "them with: pip install 'monai-physio[viewer]'"
            ) from error

        existing_args = os.environ.get("TRAME_ARGS", "").strip()
        host_arg = f"--host {host}"
        os.environ["TRAME_ARGS"] = f"{existing_args} {host_arg}".strip()

        server = get_server("monai-physio-mesh-viewer", client_type="vue3")
        state = server.state
        controller = server.controller
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("#263238")
        if self.source_kind == "usd":
            initial_mesh = self.mesh_at_index(0)
            self._display_mesh = initial_mesh
            self._point_frames = self._preload_point_frames(initial_mesh)
            plotter.add_mesh(
                initial_mesh,
                name="usd-scene",
                scalars="openusd_rgba",
                rgb=True,
                smooth_shading=False,
            )
        else:
            self._add_vtp_surfaces(plotter)
        plotter.view_isometric()
        plotter.reset_camera()

        self._server = server
        self._state = state
        self._plotter = plotter

        if self.source_kind == "usd":
            state.frame_index = 0
            state.frame_count = len(self._time_codes)
            state.frame_label = self._frame_label(0)
            state.playing = False
            state.change("frame_index")(self._on_frame_index_changed)
        else:
            for index, _label in enumerate(self._surface_labels):
                state_key = self._surface_state_key(index)
                setattr(state, state_key, True)
                state.change(state_key)(
                    self._make_visibility_callback(index, state_key)
                )

        with SinglePageLayout(server) as layout:
            layout.title.set_text(self._viewer_title())
            with layout.toolbar:
                if self.source_kind == "usd":
                    v3.VBtn(
                        "{{ playing ? 'Pause' : 'Play' }}",
                        click=self._toggle_playback,
                        disabled=("frame_count < 2",),
                        variant="text",
                    )
                    v3.VSlider(
                        v_model=("frame_index", 0),
                        min=0,
                        max=("frame_count - 1",),
                        step=1,
                        disabled=("frame_count < 2",),
                        hide_details=True,
                        density="compact",
                        style="max-width: 420px",
                    )
                    v3.VLabel("{{ frame_label }}", classes="ml-4")
                else:
                    for index, label in enumerate(self._surface_labels):
                        v3.VCheckbox(
                            label=label,
                            v_model=(self._surface_state_key(index), True),
                            color=self._surface_color(index),
                            hide_details=True,
                            density="compact",
                            classes="mr-4",
                        )
                v3.VSpacer()
                with v3.VBtn(icon=True, click=controller.view_reset_camera):
                    v3.VIcon("mdi-crop-free")
            with layout.content:
                with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                    view = vtk_widgets.VtkRemoteView(
                        plotter.ren_win,
                        interactive_ratio=1,
                    )
                    controller.view_update = view.update
                    controller.view_reset_camera = view.reset_camera
                    controller.on_server_ready.add(view.update)
                    self._view = view

    def _add_vtp_surfaces(self, plotter: pv.Plotter) -> None:
        """Add loaded VTP surfaces with stable overlay styling."""
        self._surface_actors = []
        for index, (mesh, label) in enumerate(
            zip(self._surface_meshes, self._surface_labels)
        ):
            actor = plotter.add_mesh(
                mesh,
                name=f"vtp-surface-{index}",
                label=label,
                color=self._surface_color(index),
                opacity=0.4 if index == 0 else 0.7,
                smooth_shading=False,
            )
            self._surface_actors.append(actor)

    def _surface_color(self, index: int) -> str:
        """Return a repeatable color for one VTP surface."""
        return self._SURFACE_COLORS[index % len(self._SURFACE_COLORS)]

    @staticmethod
    def _surface_state_key(index: int) -> str:
        """Return the Trame state key for one surface visibility toggle."""
        return f"surface_visible_{index}"

    def _make_visibility_callback(
        self,
        index: int,
        state_key: str,
    ) -> Callable[..., None]:
        """Create a state callback bound to one surface actor."""

        def set_visibility(**kwargs: Any) -> None:
            visible = bool(kwargs.get(state_key, True))
            self._surface_actors[index].SetVisibility(visible)
            assert self._plotter is not None
            self._plotter.render()
            if self._view is not None:
                self._view.update()

        return set_visibility

    def _viewer_title(self) -> str:
        """Return the page title for the selected input mode."""
        if self.source_kind == "usd":
            assert self.usd_file is not None
            return self.usd_file.name
        return f"{len(self._surface_meshes)} VTP surfaces"

    def _frame_label(self, frame_index: int) -> str:
        """Format a frame label for the toolbar."""
        time_code = self._time_codes[frame_index]
        return f"Frame {frame_index + 1}/{len(self._time_codes)} (time {time_code:g})"

    def _on_frame_index_changed(
        self,
        frame_index: int,
        **_kwargs: Any,
    ) -> None:
        """Update scene geometry when the timeline slider changes."""
        index = max(0, min(int(frame_index), len(self._time_codes) - 1))
        assert self._plotter is not None
        assert self._state is not None
        if self._display_mesh is not None and self._point_frames is not None:
            self._display_mesh.points[:] = self._point_frames[index]
            self._display_mesh.GetPoints().Modified()
            self._display_mesh.Modified()
            self._plotter.render()
        else:
            camera_position = self._plotter.camera_position
            self._plotter.add_mesh(
                self.mesh_at_index(index),
                name="usd-scene",
                scalars="openusd_rgba",
                rgb=True,
                smooth_shading=False,
                reset_camera=False,
            )
            self._plotter.camera_position = camera_position
        self._state.frame_label = self._frame_label(index)
        if self._view is not None:
            self._view.update()

    def _toggle_playback(self, **_kwargs: Any) -> None:
        """Start or pause looping playback."""
        assert self._state is not None
        self._state.playing = not bool(self._state.playing)
        if self._state.playing:
            from trame.app import asynchronous

            asynchronous.create_task(self._play())

    async def _play(self) -> None:
        """Advance frames at the effective rate until playback is paused."""
        assert self._state is not None
        interval = 1.0 / self._frames_per_second
        loop = asyncio.get_running_loop()
        started = loop.time()
        first_index = int(self._state.frame_index)
        next_step = 1
        while bool(self._state.playing):
            deadline = started + next_step * interval
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            if not bool(self._state.playing):
                break
            elapsed_steps = max(
                next_step,
                int((loop.time() - started) / interval),
            )
            with self._state:
                self._state.frame_index = (first_index + elapsed_steps) % len(
                    self._time_codes
                )
            next_step = elapsed_steps + 1
