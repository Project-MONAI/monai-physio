"""Checks of the compliant-aorta FSI outputs and their viewer.

Output checks use ``output/prepared`` and ``output/fsi`` (or the folders in
AORTA_FSI_PREPARED and AORTA_FSI_RESULTS) and skip when they are missing. The
wall and the coupling are tested on analytic tubes in
tests/test_warp_fluid_structure.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from monai_physio.visualize_flow_fields import VisualizeFlowFields
from monai_physio.warp_navier_stokes import Windkessel

ROOT = Path(__file__).resolve().parent
PREPARED = Path(os.environ.get("AORTA_FSI_PREPARED", ROOT / "output" / "prepared"))
RESULTS = Path(os.environ.get("AORTA_FSI_RESULTS", ROOT / "output" / "fsi"))


def require(path: Path) -> None:
    """Skip when a saved output is missing."""
    if not path.exists():
        pytest.skip(f"Run prepare.py and run.py first: {path} is missing")


def six_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return six times the signed volume of every tetrahedron."""
    q = points[tets]
    return np.asarray(
        np.einsum(
            "ij,ij->i",
            q[:, 1] - q[:, 0],
            np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]),
        )
    )


def test_prepared_meshes() -> None:
    """Positive meshes, a closed lumen, a conforming wall and clamped port rims."""
    require(PREPARED / "mesh.npz")
    mesh = np.load(PREPARED / "mesh.npz")
    fluid, solid = mesh["fluid_points_m"], mesh["solid_points_m"]
    assert np.all(six_volumes(fluid, mesh["fluid_tetrahedra"]) > 0)
    assert np.all(six_volumes(solid, mesh["solid_tetrahedra"]) > 0)
    faces, regions = mesh["fluid_boundary_faces"], mesh["fluid_boundary_regions"]
    corners = fluid[faces]
    area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flux_volume = np.einsum("ij,ij->", corners.mean(axis=1), area) / 6
    volume = six_volumes(fluid, mesh["fluid_tetrahedra"]).sum() / 6
    assert abs(flux_volume / volume - 1) < 1e-10
    ports = json.loads((PREPARED / "ports.json").read_text(encoding="utf-8"))
    assert set(np.unique(regions)) == {0} | {p["region_id"] for p in ports.values()}
    distance, nearest = cKDTree(solid).query(fluid[np.unique(faces[regions == 0])])
    assert np.max(distance) == 0
    clamped = mesh["solid_clamped"]
    for name, port in ports.items():
        cap = np.unique(faces[regions == port["region_id"]])
        gap, rim = cKDTree(solid).query(fluid[cap])
        assert np.all(clamped[np.asarray(rim)[np.asarray(gap) == 0]]), name
        assert port["rim_vertices"] > 0
        assert 0 < port["collar_depth_m"] < 0.003, name
    assert clamped[nearest].mean() < 0.1


def test_saved_fsi() -> None:
    """Finite fields, the requested inflow, replayable Windkessels, a moving wall.

    A full run must also give a 30-50 mmHg root pulse and converge the
    coupling in at least 95% of the reported steps.
    """
    require(RESULTS / "steps.npz")
    steps = np.load(RESULTS / "steps.npz")
    report = json.loads((RESULTS / "run.json").read_text(encoding="utf-8"))
    names = list(steps["port_names"])
    flows = steps["port_flow_m3_s"]
    inflow = steps["inlet_flow_m3_s"]
    assert steps["residual"].max() <= report["gmres_tolerance"]
    np.testing.assert_allclose(
        -flows[:, names.index("root_inlet")], inflow, atol=1e-10 * inflow.max()
    )
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
    for name in ("velocity_m_s", "pressure_pa", "displacement_m"):
        assert np.isfinite(np.load(RESULTS / f"{name}.npy", mmap_mode="r")).all()
    summary = report["summary"]
    assert summary["peak_wall_displacement_mm"] < 5
    if report["steps"] * report["dt_s"] > 1.5:  # A full run, not a smoke test.
        assert 60 < summary["root_diastolic_mmhg"] < summary["root_systolic_mmhg"] < 160
        pulse = summary["root_systolic_mmhg"] - summary["root_diastolic_mmhg"]
        assert 30 < pulse < 50
        assert 0.2 < summary["peak_wall_displacement_mm"]
        reported = int(np.sum(steps["time_s"] >= report["report_from_s"]))
        assert summary["nonconverged_steps"] <= summary["capped_steps"]
        assert summary["capped_steps"] <= 0.05 * reported


def test_viewer_shows_wall_displacement(tmp_path: Path) -> None:
    """The viewer moves the lumen and colors it by the wall displacement from 0."""
    require(RESULTS / "displacement_m.npy")
    viewer = VisualizeFlowFields(RESULTS)
    try:
        index = len(viewer.times) - 1
        viewer.set_time(index)
        viewer.set_field("displacement")
        moved = np.asarray(viewer.displacement[index])  # type: ignore[index]
        count = len(moved)
        np.testing.assert_allclose(
            viewer.grid.points[:count], viewer.nodes[:count] + moved
        )
        np.testing.assert_allclose(
            viewer.grid["display"][:count], 1e3 * np.linalg.norm(moved, axis=1)
        )
        shown = viewer.screenshot(tmp_path / "displacement.png")
        movie = viewer.record_movie(tmp_path / "movie.mp4", float(viewer.times[-2]))
        beat = np.linalg.norm(viewer.displacement[-2:], axis=2)  # type: ignore[index]
        assert movie["color_limits"] == (0.0, pytest.approx(1e3 * beat.max()))
        viewer.set_field("pressure")
        assert not np.array_equal(shown, viewer.screenshot(tmp_path / "pressure.png"))
    finally:
        viewer.plotter.close()
