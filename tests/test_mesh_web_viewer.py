"""Tests for USD and VTP evaluation used by the Trame mesh viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from monai_physio.mesh_web_viewer import MeshWebViewer
from monai_physio.usd_tools import USDTools


def _create_triangle_stage(path: Path, animated: bool) -> None:
    """Create a tiny static or animated triangle stage."""
    stage = Usd.Stage.CreateNew(str(path))
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Triangle")
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    points = mesh.CreatePointsAttr()
    frame_zero = [
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 0.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
    ]
    if animated:
        frame_two = [
            Gf.Vec3f(0.0, 0.0, 1.0),
            Gf.Vec3f(1.0, 0.0, 1.0),
            Gf.Vec3f(0.0, 1.0, 1.0),
        ]
        points.Set(frame_zero, 0.0)
        points.Set(frame_two, 2.0)
        stage.SetTimeCodesPerSecond(12.0)
    else:
        points.Set(frame_zero)
    stage.Save()


def _create_triangle_surface(path: Path, z_offset: float) -> None:
    """Create a tiny VTP triangle surface."""
    points = np.array(
        [
            [0.0, 0.0, z_offset],
            [1.0, 0.0, z_offset],
            [0.0, 1.0, z_offset],
        ]
    )
    pv.PolyData(points, faces=[3, 0, 1, 2]).save(path)


def test_viewer_discovers_and_evaluates_animated_frames(tmp_path: Path) -> None:
    """Viewer exposes mesh point samples and the stage playback rate."""
    usd_path = tmp_path / "animated.usd"
    _create_triangle_stage(usd_path, animated=True)

    viewer = MeshWebViewer(usd_path)

    assert viewer.time_codes == (0.0, 2.0)
    assert viewer.frames_per_second == 12.0
    assert viewer.stage_frames_per_second == 12.0
    assert np.allclose(viewer.mesh_at_index(0).points[:, 2], 0.0)
    assert np.allclose(viewer.mesh_at_index(1).points[:, 2], 1.0)
    with pytest.raises(IndexError, match="Frame index out of range"):
        viewer.mesh_at_index(2)


def test_viewer_overrides_playback_rate_without_changing_stage(tmp_path: Path) -> None:
    """Viewer playback can differ from the rate authored in the USD."""
    usd_path = tmp_path / "animated.usd"
    _create_triangle_stage(usd_path, animated=True)

    viewer = MeshWebViewer(usd_path, playback_fps=3.0)

    assert viewer.frames_per_second == 3.0
    assert viewer.stage_frames_per_second == 12.0


@pytest.mark.parametrize("playback_fps", [0.0, -1.0, float("inf")])
def test_viewer_rejects_invalid_playback_rate(
    tmp_path: Path,
    playback_fps: float,
) -> None:
    """Playback overrides must be finite and positive."""
    usd_path = tmp_path / "animated.usd"
    _create_triangle_stage(usd_path, animated=True)

    with pytest.raises(ValueError, match="finite and greater than zero"):
        MeshWebViewer(usd_path, playback_fps=playback_fps)


def test_viewer_preloads_animated_points(tmp_path: Path) -> None:
    """Fixed-topology animation is cached without rebuilding the mesh."""
    usd_path = tmp_path / "animated.usd"
    _create_triangle_stage(usd_path, animated=True)
    viewer = MeshWebViewer(usd_path)

    frames = viewer._preload_point_frames(viewer.mesh_at_index(0))

    assert frames is not None
    assert len(frames) == 2
    assert np.allclose(frames[0][:, 2], 0.0)
    assert np.allclose(frames[1][:, 2], 1.0)


def test_viewer_uses_one_default_frame_for_static_stage(tmp_path: Path) -> None:
    """Static stages remain inspectable through a single timeline frame."""
    usd_path = tmp_path / "static.usd"
    _create_triangle_stage(usd_path, animated=False)

    viewer = MeshWebViewer(usd_path)

    assert viewer.time_codes == (0.0,)
    assert viewer.mesh_at_index(0).n_points == 3


def test_viewer_loads_multiple_vtp_surfaces(tmp_path: Path) -> None:
    """VTP inputs stay separate for colored overlay rendering."""
    patient_path = tmp_path / "case_patient_surface.vtp"
    fitted_path = tmp_path / "case_fitted_surface.vtp"
    _create_triangle_surface(patient_path, z_offset=0.0)
    _create_triangle_surface(fitted_path, z_offset=1.0)

    viewer = MeshWebViewer([patient_path, fitted_path])

    assert viewer.source_kind == "vtp"
    assert viewer.surface_labels == ("Patient Surface", "Fitted Surface")
    assert len(viewer.surface_meshes) == 2
    assert viewer.mesh_at_index(0).n_points == 6
    assert viewer.frames_per_second == 0.0
    assert viewer.stage_frames_per_second == 0.0


def test_viewer_rejects_mixed_usd_and_vtp_inputs(tmp_path: Path) -> None:
    """One viewer invocation cannot mix animated USD and VTP modes."""
    usd_path = tmp_path / "scene.usd"
    vtp_path = tmp_path / "surface.vtp"
    _create_triangle_stage(usd_path, animated=False)
    _create_triangle_surface(vtp_path, z_offset=0.0)

    with pytest.raises(ValueError, match="one USD file or one or more VTP"):
        MeshWebViewer([usd_path, vtp_path])


def test_viewer_rejects_playback_rate_for_vtp_input(tmp_path: Path) -> None:
    """Static VTP overlays do not accept an animation playback rate."""
    vtp_path = tmp_path / "surface.vtp"
    _create_triangle_surface(vtp_path, z_offset=0.0)

    with pytest.raises(ValueError, match="only supported for USD"):
        MeshWebViewer(vtp_path, playback_fps=3.0)


def test_vtk_loader_uses_bound_omnisurface_diffuse_color(tmp_path: Path) -> None:
    """The lightweight preview approximates a bound OmniSurface color."""
    usd_path = tmp_path / "material.usd"
    _create_triangle_stage(usd_path, animated=False)
    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None
    material = UsdShade.Material.Define(stage, "/World/Looks/Tissue")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Tissue/Shader")
    shader.CreateInput(
        "diffuse_reflection_color",
        Sdf.ValueTypeNames.Color3f,
    ).Set(Gf.Vec3f(0.2, 0.4, 0.6))
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.ConnectableAPI(),
        "out",
    )
    mesh_prim = stage.GetPrimAtPath("/World/Triangle")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)
    stage.Save()

    loaded = USDTools().load_usd_as_vtk(usd_path, time_code=0.0)

    expected = np.array([51, 102, 153], dtype=np.uint8)
    assert np.all(loaded.point_data["openusd_rgb"] == expected)
    assert np.all(loaded.point_data["openusd_rgba"][:, :3] == expected)
    assert np.all(loaded.point_data["openusd_rgba"][:, 3] == 255)


def test_vtk_loader_preserves_bound_preview_surface_opacity(tmp_path: Path) -> None:
    """Animated USD previews retain material alpha for overlapping surfaces."""
    usd_path = tmp_path / "transparent_material.usd"
    _create_triangle_stage(usd_path, animated=True)
    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None
    material = UsdShade.Material.Define(stage, "/World/Looks/Prediction")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Prediction/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.42, 0.42)
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.65)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )
    mesh_prim = stage.GetPrimAtPath("/World/Triangle")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)
    stage.Save()

    loaded = USDTools().load_usd_as_vtk(usd_path, time_code=0.0)

    assert np.all(loaded.point_data["openusd_rgba"][:, 3] == 165)
