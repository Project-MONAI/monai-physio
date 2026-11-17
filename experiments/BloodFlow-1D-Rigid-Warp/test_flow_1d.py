"""Checks of the 1D trunk experiment: the CUDA closure and the saved outputs.

Output checks use ``output/prepared`` and ``output/flow`` (or the folders in
AORTA_1D_PREPARED and AORTA_1D_RESULTS) and skip when they are missing.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
PREPARED = Path(os.environ.get("AORTA_1D_PREPARED", ROOT / "output" / "prepared"))
RESULTS = Path(os.environ.get("AORTA_1D_RESULTS", ROOT / "output" / "flow"))


def load(name: str) -> ModuleType:
    """Import a sibling script without running its command line."""
    spec = importlib.util.spec_from_file_location(
        f"aorta_1d_{name}", ROOT / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(path: Path) -> None:
    """Skip when a saved output is missing."""
    if not path.exists():
        pytest.skip(f"Run prepare.py and run.py first: {path} is missing")


@pytest.mark.requires_gpu
def test_closure_matches_pipe_laws() -> None:
    """A uniform pipe gives exact Poiseuille drag plus rho L / A inertia."""
    run = load("run")
    radius, length, density, viscosity = 0.01, 0.3, 1060.0, 0.004
    arc = np.linspace(0, length, 65)
    area = np.full(len(arc), np.pi * radius**2)
    flux, derivative = np.array([2e-5, 5e-5]), np.array([0.0, 2e-4])
    velocity, drop = run.evaluate(arc, area, flux, derivative, density, viscosity)
    expected = (
        density * length / area[0] * derivative
        + 8 * viscosity * length / (np.pi * radius**4) * flux
    )
    np.testing.assert_allclose(drop[:, 0], expected, rtol=1e-12)
    np.testing.assert_array_equal(drop[:, -1], 0.0)
    np.testing.assert_allclose(velocity, flux[:, None] / area, rtol=1e-14)


def test_prepared_trunk() -> None:
    """Sections are planar, normal to the curve and span the shared 3D cuts."""
    require(PREPARED / "geometry.npz")
    data = np.load(PREPARED / "geometry.npz")
    report = json.loads((PREPARED / "preparation.json").read_text(encoding="utf-8"))
    curve, area = data["centerline_m"], data["area_m2"]
    np.testing.assert_allclose(curve[0], report["root_plane"]["center_m"], atol=1e-9)
    np.testing.assert_allclose(curve[-1], report["distal_plane"]["center_m"], atol=1e-9)
    margin = report["config"]["distal_margin_m"]
    assert curve[-1, 1] >= report["distal_crop_y_m"] + margin - 1e-9
    np.testing.assert_allclose(
        np.diff(data["arc_length_m"]), np.linalg.norm(np.diff(curve, axis=0), axis=1)
    )
    contour, offsets = data["contour_points_m"], data["contour_offsets"]
    for i, tangent in enumerate(data["tangent"]):
        ring = contour[offsets[i] : offsets[i + 1]] - curve[i]
        assert np.abs(ring @ tangent).max() < 1e-8
        centered = ring - ring.mean(axis=0)
        polygon = abs(np.cross(centered, np.roll(centered, -1, 0)).sum(0) @ tangent)
        assert abs(polygon / 2 / area[i] - 1) < 1e-10
    assert np.all(area > 0) and np.abs(np.diff(area) / area[:-1]).max() < 0.1


def test_prepared_waveform() -> None:
    """The LV flux covers 100 intervals of 0.03 s and flows only when gated."""
    require(PREPARED / "waveform.npz")
    waveform = np.load(PREPARED / "waveform.npz")
    start, end = waveform["times_s"], waveform["interval_end_times_s"]
    np.testing.assert_allclose(end - start, 0.03, atol=1e-14)
    np.testing.assert_allclose(end[-1], 3.0, atol=1e-12)
    volume, flux = waveform["lv_volume_m3"], waveform["inlet_interval_flux_m3_s"]
    loss = volume - np.roll(volume, -1)
    np.testing.assert_array_equal(waveform["ejection_gate"], loss > 0)
    np.testing.assert_array_equal(flux > 0, waveform["ejection_gate"])
    np.testing.assert_allclose(flux * (end - start), np.maximum(loss, 0), rtol=1e-12)


def test_saved_fields() -> None:
    """Saved fields conserve flow and volume and add the Windkessel pressure."""
    require(RESULTS / "fields.npz")
    fields = np.load(RESULTS / "fields.npz")
    interval = np.load(PREPARED / "waveform.npz")["inlet_interval_flux_m3_s"]
    q, area, arc = fields["inlet_q_m3_s"], fields["area_m2"], fields["arc_length_m"]
    dt = float(np.diff(fields["times_s"])[0])
    assert np.all(q >= 0)
    assert abs(q.sum() * dt / (interval.sum() * 0.03) - 1) < 1e-6
    np.testing.assert_allclose(fields["velocity_m_s"] * area, q[:, None] + 0 * area)
    inverse, segment = 1 / area, np.diff(arc) / 2
    inertia = 1060 * (inverse[:-1] + inverse[1:]) * segment
    drag = 8 * np.pi * 0.004 * (inverse[:-1] ** 2 + inverse[1:] ** 2) * segment

    def to_outlet(values: np.ndarray) -> np.ndarray:
        return np.concatenate([np.cumsum(values[::-1])[::-1], [0.0]])

    qdot = fields["inlet_qdot_m3_s2"]
    drop = np.outer(qdot, to_outlet(inertia)) + np.outer(q, to_outlet(drag))
    pressure = fields["pressure_pa"] - fields["outlet_pressure_pa"][:, None]
    np.testing.assert_allclose(pressure, drop, atol=1e-9 * np.abs(drop).max())
    report = json.loads((RESULTS / "run.json").read_text(encoding="utf-8"))
    beat = report["last_beat"]
    assert 60 < beat["root_diastolic_mmhg"] < beat["root_systolic_mmhg"] < 160


def test_viewer_renders(tmp_path: Path) -> None:
    """The viewer shows different pixels for each field and time."""
    require(RESULTS / "fields.npz")
    viewer = load("view").TrunkViewer(PREPARED, RESULTS)
    try:
        velocity = viewer.screenshot(tmp_path / "velocity.png")
        np.testing.assert_array_equal(
            viewer.line["display"], viewer.values["velocity"][viewer.index]
        )
        viewer.set_field("pressure")
        pressure = viewer.screenshot(tmp_path / "pressure.png")
        viewer.set_time(0)
        earlier = viewer.screenshot(tmp_path / "earlier.png")
        assert not np.array_equal(velocity, pressure)
        assert not np.array_equal(pressure, earlier)
        assert "svg" in viewer.waveform_svg()
    finally:
        viewer.plotter.close()
