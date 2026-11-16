"""Tests of the aortic inlet waveform and lumen preparation."""

import os
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from monai_physio.simulate_aorta_flow_warp import AortaSource, SimulateAortaFlowWarp

SOURCE = Path(
    os.environ.get(
        "AORTA_SOURCE_USD",
        Path(__file__).resolve().parents[1]
        / "tutorials/output/tutorial_13_combine_heart_and_lung_motion_chest_3d_ct"
        / "heart_and_lung_motion.usd",
    )
)


def test_inlet_flow_preserves_volume_and_sign() -> None:
    """Gaussian smoothing keeps Q >= 0, every beat volume and a consistent dQ/dt."""
    flux = np.zeros(100)
    for start in (0, 34, 67):
        flux[start : start + 12] = np.repeat([20e-6, 120e-6, 180e-6, 60e-6], 3)
    times = (np.arange(30000) + 0.5) * 1e-4
    flow, derivative = SimulateAortaFlowWarp.inlet_flow(times, flux, 0.03)
    assert np.all(flow >= 0)
    assert flow.max() < flux.max()
    for start, low, high in ((0, -0.2, 0.82), (34, 0.82, 1.81), (67, 1.81, 2.8)):
        window = np.mod(times - low, 3.0) < high - low
        beat = flow[window].sum() * 1e-4
        assert abs(beat / (flux[start : start + 12].sum() * 0.03) - 1) <= 1e-6
    numerical = np.gradient(flow, times)
    scale = np.abs(derivative).max()
    np.testing.assert_allclose(derivative[1:-1], numerical[1:-1], atol=1e-5 * scale)


def test_infer_inlet_from_pulsing_sphere() -> None:
    """Interval flux equals the positive volume loss of a three-beat sphere."""
    sphere = pv.Sphere(radius=0.02, theta_resolution=24, phi_resolution=24)
    seconds = np.arange(100) * 0.03
    lv = np.stack(
        [np.asarray(sphere.points) * (1 + 0.1 * np.cos(2 * np.pi * t)) for t in seconds]
    )
    faces = sphere.faces.reshape(-1, 4)[:, 1:]
    source = AortaSource(
        Path("synthetic"), "", np.arange(100.0), seconds, lv, faces, lv, faces, {}, {}
    )
    inlet = SimulateAortaFlowWarp().infer_inlet(source)
    volume = np.asarray(inlet["lv_volume_m3"])
    loss = np.maximum(volume - np.roll(volume, -1), 0)
    flux = np.asarray(inlet["inlet_interval_flux_m3_s"])
    np.testing.assert_allclose(flux * 0.03, loss)
    assert len(inlet["ejection_start_samples"]) == 3
    assert abs(sum(inlet["stroke_volumes_m3"]) / loss.sum() - 1) < 1e-12


def test_section_centroid_on_tilted_cylinder() -> None:
    """An oblique section of a cylinder has the analytic centroid and area."""
    surface = pv.Cylinder(radius=0.01, height=0.1, resolution=200, capping=False)
    surface = surface.triangulate().clean()
    points = np.asarray(surface.points)
    faces = surface.faces.reshape(-1, 4)[:, 1:]
    normal = np.array([1.0, 0.2, 0.0]) / np.linalg.norm([1.0, 0.2, 0.0])
    origin = np.array([0.01, 0.0, 0.0])
    ring, _, _ = SimulateAortaFlowWarp.section(points, faces, origin, normal, origin)
    centroid = SimulateAortaFlowWarp.polygon_centroid(ring, normal)
    np.testing.assert_allclose(centroid, origin, atol=1e-6)
    area = abs(np.cross(ring - origin, np.roll(ring, -1, 0) - origin).sum(0) @ normal)
    assert abs(area / 2 / (np.pi * 0.01**2 / normal[0]) - 1) < 1e-3


def test_aorta_boundary_from_source() -> None:
    """The Tutorial 13 source yields a closed five-port lumen and three beats."""
    if not SOURCE.is_file():
        pytest.skip(f"Tutorial 13 source USD not found: {SOURCE}")
    tools = SimulateAortaFlowWarp()
    source = tools.load_source(SOURCE)
    boundary = tools.prepare_boundaries(source, distal_margin_m=0.010)
    assert set(np.unique(boundary.regions)) == {0, 1, 2, 3, 4, 5}
    assert boundary.surface().n_open_edges == 0
    assert tools.signed_surface_volume(boundary.points, boundary.faces) > 0
    distal = boundary.ports["distal_crop_outlet"]["plane_origin_m"][1]
    assert distal >= source.aorta[69][:, 1].min() + 0.0099
    inlet = tools.infer_inlet(source)
    assert inlet["ejection_start_samples"] == [0, 34, 67]
    assert all(30e-6 < v < 40e-6 for v in inlet["stroke_volumes_m3"])
