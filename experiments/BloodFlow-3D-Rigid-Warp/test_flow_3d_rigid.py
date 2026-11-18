"""Checks of the 3D rigid-flow experiment outputs and its viewer.

Output checks use ``output/prepared`` and ``output/flow`` (or the folders in
AORTA_3D_PREPARED and AORTA_3D_RESULTS) and skip when they are missing. The
solver itself is tested on analytic pipes in tests/test_warp_navier_stokes.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from monai_physio.visualize_flow_fields import MMHG, VisualizeFlowFields
from monai_physio.warp_navier_stokes import (
    WarpNavierStokes,
    Windkessel,
    find_cudss_library,
)

ROOT = Path(__file__).resolve().parent
PREPARED = Path(os.environ.get("AORTA_3D_PREPARED", ROOT / "output" / "prepared"))
RESULTS = Path(os.environ.get("AORTA_3D_RESULTS", ROOT / "output" / "flow"))
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def require(path: Path) -> None:
    """Skip when a saved output is missing."""
    if not path.exists():
        pytest.skip(f"Run prepare.py and run.py first: {path} is missing")


def test_prepared_lumen() -> None:
    """Positive tetrahedra, a closed outward boundary and five planar ports."""
    require(PREPARED / "geometry.npz")
    data = np.load(PREPARED / "geometry.npz")
    points, tets = data["points_m"], data["tetrahedra"]
    faces, regions = data["boundary_faces"], data["boundary_regions"]
    q = points[tets]
    six = np.einsum(
        "ij,ij->i", q[:, 1] - q[:, 0], np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0])
    )
    assert np.all(six > 0)
    corners = points[faces]
    area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flux_volume = np.einsum("ij,ij->", corners.mean(axis=1), area) / 6
    assert abs(flux_volume / (six.sum() / 6) - 1) < 1e-10
    assert pv.PolyData(points, np.c_[np.full(len(faces), 3), faces]).n_open_edges == 0
    ports = json.loads((PREPARED / "ports.json").read_text(encoding="utf-8"))
    assert set(np.unique(regions)) == {0} | {p["region_id"] for p in ports.values()}
    for name, port in ports.items():
        selected = corners[regions == port["region_id"]].reshape(-1, 3)
        offset = (selected - selected.mean(axis=0)) @ np.asarray(port["normal"])
        # Netgen smooths the voxelized lumen, so cap vertices stay within ~1 mm.
        assert np.abs(offset).max() < 0.0015, name
        assert abs(port["mesh_area_m2"] / port["source_area_m2"] - 1) < 0.1, name
        # Half the triangles of a disc port_cells_across elements wide: enough
        # free velocity nodes that the opening's flow is not carried by one.
        cells = CONFIG["port_cells_across"]
        assert port["mesh_faces"] >= 0.5 * np.pi / 4 * cells**2, name
    report = json.loads((PREPARED / "preparation.json").read_text(encoding="utf-8"))
    distal = ports["distal_crop_outlet"]["center_m"][1]
    assert distal >= report["distal_crop_y_m"] + CONFIG["distal_margin_m"] - 1e-9


def test_saved_flow() -> None:
    """Conserved flow, replayable Windkessels, no-slip walls and plausible pressures."""
    require(RESULTS / "steps.npz")
    steps = np.load(RESULTS / "steps.npz")
    names = list(steps["port_names"])
    flows = steps["port_flow_m3_s"]
    inflow = steps["inlet_flow_m3_s"]
    report = json.loads((RESULTS / "run.json").read_text(encoding="utf-8"))
    assert steps["residual"].max() <= report["gmres_tolerance"]
    assert np.abs(flows.sum(axis=1)).max() <= 1e-6 * inflow.max()
    root = names.index("root_inlet")
    np.testing.assert_allclose(-flows[:, root], inflow, atol=1e-10 * inflow.max())
    settings = report["config"]["windkessel"]
    outlets = [name for name in names if name != "root_inlet"]
    for column, name in enumerate(outlets):
        windkessel = Windkessel.from_total(
            settings["flow_fractions"][name],
            settings["total_resistance_pa_s_m3"],
            settings["total_compliance_m3_pa"],
            settings["proximal_fraction"],
            settings["initial_pressure_pa"],
        )
        replay = [
            windkessel.advance(q, report["dt_s"]) for q in flows[:, names.index(name)]
        ]
        np.testing.assert_allclose(steps["outlet_pressure_pa"][:, column], replay)
        # Every outlet carries about its configured share of the flow; an
        # opening tiled too coarsely carries little or none.
        share = report["last_beat"]["outlets"][name]["flow_share"]
        assert abs(share / settings["flow_fractions"][name] - 1) < 0.1, name
    mesh = np.load(RESULTS / "mesh.npz")
    wall = mesh["boundary_faces"][mesh["boundary_regions"] == 0]
    velocity = np.load(RESULTS / "velocity_m_s.npy", mmap_mode="r")
    assert np.isfinite(velocity[-1]).all()
    assert np.count_nonzero(velocity[-1][np.unique(wall)]) == 0
    beat = report["last_beat"]
    assert 60 < beat["root_diastolic_mmhg"] < beat["root_systolic_mmhg"] < 160


def test_viewer_controls(tmp_path: Path) -> None:
    """Quadratic cells, the controls and a fixed-range movie render."""
    require(RESULTS / "mesh.npz")
    viewer = VisualizeFlowFields(RESULTS, PREPARED / "source.vtp")
    try:
        assert np.all(viewer.grid.celltypes == pv.CellType.QUADRATIC_TETRA)
        cells = viewer.grid.cells.reshape(-1, 11)[:10, 1:]
        viewer.set_field("velocity")
        centers = viewer.grid.points[cells[:, :4]].mean(axis=1)
        probes = pv.PolyData(centers).sample(viewer.grid, tolerance=1e-10)
        values = viewer.grid["velocity_m_s"][cells]
        expected = -0.125 * values[:, :4].sum(1) + 0.25 * values[:, 4:].sum(1)
        np.testing.assert_allclose(probes["velocity_m_s"], expected, atol=1e-8)
        speed = viewer.screenshot(tmp_path / "speed.png")
        viewer.set_field("pressure")
        pressure = viewer.screenshot(tmp_path / "pressure.png")
        assert not np.array_equal(speed, pressure)
        values = np.r_[viewer.surface["display"], viewer.slice["display"]]
        assert viewer.display_limits == (float(values.min()), float(values.max()))
        viewer.set_color_mode("global")
        assert viewer.display_limits == viewer.limits["pressure"]
        viewer.set_slice("y", 0.6)
        viewer.set_vectors(True)
        assert viewer.slice.n_points > 0 and viewer.glyphs.n_points > 0
        vectors = viewer.screenshot(tmp_path / "vectors.png")
        assert not np.array_equal(pressure, vectors)
        camera = viewer.snapshot()["camera"]
        viewer.rotate_camera(35, 15)
        assert viewer.snapshot()["camera"] != camera
        viewer.reset_camera()
        assert viewer.snapshot()["camera"] == camera
        viewer.playing = True
        viewer.advance(viewer.times[3] + 1e-9)
        assert viewer.index == 3
        movie = viewer.record_movie(tmp_path / "movie.mp4", float(viewer.times[-2]))
        assert movie["frames"] == 2 and (tmp_path / "movie.mp4").stat().st_size
        shown = np.asarray(viewer.pressure[-2:]) / MMHG
        assert movie["color_limits"] == pytest.approx((shown.min(), shown.max()))
    finally:
        viewer.plotter.close()


@pytest.mark.requires_gpu
def test_float32_steps() -> None:
    """A few float32 steps on the prepared lumen converge to finite fields."""
    require(PREPARED / "geometry.npz")
    try:
        find_cudss_library()
    except RuntimeError as error:
        pytest.skip(str(error))
    geometry = np.load(PREPARED / "geometry.npz")
    ports = json.loads((PREPARED / "ports.json").read_text(encoding="utf-8"))
    settings = CONFIG["windkessel"]
    solver = WarpNavierStokes(
        geometry["points_m"],
        geometry["tetrahedra"],
        geometry["boundary_faces"],
        geometry["boundary_regions"],
        inlet_region=ports["root_inlet"]["region_id"],
        outlets={
            port["region_id"]: Windkessel.from_total(
                settings["flow_fractions"][name],
                settings["total_resistance_pa_s_m3"],
                settings["total_compliance_m3_pa"],
                settings["proximal_fraction"],
                settings["initial_pressure_pa"],
            )
            for name, port in ports.items()
            if port["role"] == "outlet"
        },
        dt=CONFIG["dt_s"],
        precision="float32",
    )
    try:
        for flow in (50e-6, 100e-6, 150e-6):
            result = solver.step(flow)
            assert result["residual"] <= solver.tolerance
            assert abs(sum(result["flows_m3_s"].values())) < 1e-4 * flow
        velocity, pressure = solver.velocity(), solver.pressure()
    finally:
        solver.close()
    assert velocity.dtype == pressure.dtype == np.float64
    assert np.isfinite(velocity).all() and np.isfinite(pressure).all()
