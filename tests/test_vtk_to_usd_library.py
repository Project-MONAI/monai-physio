"""
Tests for VTK-to-USD conversion through ConvertVTKToUSD.

The low-level monai_physio.vtk_to_usd package is exercised only through
ConvertVTKToUSD here. It remains a public advanced API, but repository tests
should validate the supported application entry point.

Note: Tests marked requires_data need manually downloaded data:
- KCL-Heart-Model: data/KCL-Heart-Model/
- CHOP-Valve4D: data/CHOP-Valve4D/
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from pxr import Gf, Usd, UsdGeom, UsdShade

from monai_physio import ConvertVTKToUSD
from monai_physio.test_tools import TestTools
from monai_physio.usd_tools import USDTools


def get_data_dir() -> Path:
    """Get the data directory path."""
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent
    return project_root / "data"


def check_kcl_heart_data() -> bool:
    """Check if KCL Heart Model data is available."""
    data_dir = get_data_dir() / "KCL-Heart-Model"
    vtk_file = data_dir / "average_mesh.vtk"
    return vtk_file.exists()


def get_or_create_average_surface(test_directories: dict[str, Path]) -> Path:
    """
    Get or create average_surface.vtp from average_mesh.vtk.

    Args:
        test_directories: Dictionary with 'output' key pointing to test output
            directory.

    Returns:
        Path to the average_surface.vtp file.
    """
    output_dir = test_directories["output"] / "vtk_to_usd_library"
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_file = output_dir / "average_surface.vtp"
    if surface_file.exists():
        return surface_file

    data_dir = get_data_dir() / "KCL-Heart-Model"
    vtk_file = data_dir / "average_mesh.vtk"
    vtk_mesh = pv.read(str(vtk_file))
    surface = vtk_mesh.extract_surface(algorithm="dataset_surface")
    surface.save(str(surface_file))
    return surface_file


@pytest.fixture(scope="session")
def kcl_average_surface(test_directories: dict[str, Path]) -> Path:
    """
    Fixture providing the KCL average heart surface.

    Returns:
        Path to average_surface.vtp file.
    """
    if not check_kcl_heart_data():
        pytest.skip("KCL-Heart-Model data not available")

    return get_or_create_average_surface(test_directories)


class TestFromFilesValidation:
    """Synthetic tests for ConvertVTKToUSD.from_files()."""

    def test_time_codes_length_mismatch_raises(self, tmp_path: Path) -> None:
        """from_files() must reject time_codes whose length != len(vtk_files)."""
        sphere = pv.Sphere(theta_resolution=4, phi_resolution=4)
        f0 = tmp_path / "f0.vtp"
        f1 = tmp_path / "f1.vtp"
        sphere.save(str(f0))
        sphere.save(str(f1))
        with pytest.raises(ValueError, match="time_codes length"):
            ConvertVTKToUSD.from_files("X", [f0, f1], time_codes=[0.0])

    def test_time_codes_non_monotone_raises(self, tmp_path: Path) -> None:
        """from_files() must reject time_codes that decrease between frames."""
        sphere = pv.Sphere(theta_resolution=4, phi_resolution=4)
        f0 = tmp_path / "f0.vtp"
        f1 = tmp_path / "f1.vtp"
        sphere.save(str(f0))
        sphere.save(str(f1))
        with pytest.raises(ValueError, match="non-decreasing order"):
            ConvertVTKToUSD.from_files("X", [f0, f1], time_codes=[2.0, 1.0])

    def test_time_codes_equal_consecutive_is_valid(self, tmp_path: Path) -> None:
        """Equal consecutive time codes are non-decreasing and must not raise."""
        sphere = pv.Sphere(theta_resolution=4, phi_resolution=4)
        f0 = tmp_path / "f0.vtp"
        f1 = tmp_path / "f1.vtp"
        sphere.save(str(f0))
        sphere.save(str(f1))
        stage = ConvertVTKToUSD.from_files(
            "X", [f0, f1], time_codes=[1.0, 1.0]
        ).convert(str(tmp_path / "out.usd"))
        assert stage.GetPrimAtPath("/World/X/Mesh").IsValid()

    def test_from_files_single_file_writes_static_mesh(self, tmp_path: Path) -> None:
        """A single-file converter writes a static mesh with no time range."""
        plane = pv.Plane(i_resolution=2, j_resolution=2)
        f0 = tmp_path / "p0.vtp"
        plane.save(str(f0))

        stage = ConvertVTKToUSD.from_files("P", [f0]).convert(str(tmp_path / "p.usd"))

        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/P/Mesh"))
        assert len(mesh.GetPointsAttr().Get()) == plane.n_points
        assert not stage.HasAuthoredTimeCodeRange()

    def test_openusd_screenshot_uses_vtk_loader(self, tmp_path: Path) -> None:
        """Render a tiny OpenUSD mesh through TestTools without USD imaging plugins."""
        usd_path = tmp_path / "triangle.usd"
        stage = Usd.Stage.CreateNew(str(usd_path))
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)
        mesh = UsdGeom.Mesh.Define(stage, "/World/Triangle")
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        stage.Save()

        loaded = USDTools().load_usd_as_vtk(usd_path)
        assert loaded.n_points == 3
        assert "openusd_rgb" in loaded.point_data
        assert np.all(loaded.point_data["openusd_rgb"] == np.array([255, 0, 0]))

        # GitHub's hosted windows-latest runners lack a real OpenGL stack,
        # so VTK's render path access-violates inside Plotter.screenshot
        # with no Mesa software-render fallback available. Linux runners
        # render off-screen successfully via xvfb + libosmesa6. Skip only
        # on Windows; the load_usd_as_vtk assertions above still run.
        if sys.platform == "win32":
            pytest.skip(
                "Skipping VTK render on Windows hosted runners: no usable "
                "OpenGL context"
            )

        # mypy runs on Windows in this repo (no python_platform override), so
        # it statically infers the `sys.platform == "win32"` branch above as
        # always-taken and flags this line as unreachable; on the Linux CI
        # runner (or any non-Windows dev machine) it is reachable.
        tt = TestTools(  # type: ignore[unreachable]
            class_name="openusd", results_dir=tmp_path
        )
        screenshot = tt.save_screenshot_openusd(usd_path, "triangle.png")

        assert screenshot.exists()
        assert screenshot.stat().st_size > 0

    def test_from_files_static_merge_writes_separate_meshes(
        self, tmp_path: Path
    ) -> None:
        """static_merge=True treats files as static objects, not time samples."""
        plane = pv.Plane(i_resolution=2, j_resolution=2)
        f0 = tmp_path / "p0.vtp"
        f1 = tmp_path / "p1.vtp"
        plane.save(str(f0))
        plane.save(str(f1))

        stage = ConvertVTKToUSD.from_files("P", [f0, f1], static_merge=True).convert(
            str(tmp_path / "static.usd")
        )

        assert stage.GetPrimAtPath("/World/P/P_0").IsValid()
        assert stage.GetPrimAtPath("/World/P/P_1").IsValid()
        assert not stage.HasAuthoredTimeCodeRange()


class TestSyntheticConversion:
    """Synthetic ConvertVTKToUSD tests that do not require downloaded data."""

    def test_inspect_file_reports_public_summary(self, tmp_path: Path) -> None:
        """inspect_file() reports geometry, bounds, arrays, and cell types."""
        mesh = pv.Plane(i_resolution=2, j_resolution=2)
        mesh.point_data["pressure"] = np.linspace(
            0.0, 1.0, mesh.n_points, dtype=np.float32
        )
        vtk_file = tmp_path / "inspect_plane.vtp"
        mesh.save(str(vtk_file))

        summary = ConvertVTKToUSD.inspect_file(vtk_file)

        assert summary["is_empty"] is False
        assert summary["points"] == mesh.n_points
        assert summary["faces"] > 0
        assert len(summary["bounds_min"]) == 3
        assert len(summary["bounds_max"]) == 3
        assert len(summary["bounds_size"]) == 3
        assert summary["cell_types"]
        assert any(array["name"] == "pressure" for array in summary["arrays"])

    def test_inspect_file_reports_empty_mesh(self, tmp_path: Path) -> None:
        """inspect_file() reports empty meshes without raising."""
        vtk_file = tmp_path / "empty.vtp"
        pv.PolyData().save(str(vtk_file))

        summary = ConvertVTKToUSD.inspect_file(vtk_file)

        assert summary["is_empty"] is True
        assert summary["points"] == 0
        assert summary["faces"] == 0
        assert summary["bounds_min"] == (0.0, 0.0, 0.0)
        assert summary["bounds_max"] == (0.0, 0.0, 0.0)
        assert summary["bounds_size"] == (0.0, 0.0, 0.0)
        assert summary["arrays"] == []
        assert summary["cell_types"] == []

    def test_file_primvar_preservation(self, tmp_path: Path) -> None:
        """Point arrays in a VTP file are preserved as USD primvars."""
        mesh = pv.Plane(i_resolution=2, j_resolution=2)
        mesh.point_data["pressure"] = np.linspace(
            0.0, 1.0, mesh.n_points, dtype=np.float32
        )
        vtk_file = tmp_path / "plane.vtp"
        mesh.save(str(vtk_file))

        stage = ConvertVTKToUSD.from_files(
            data_basename="Plane",
            vtk_files=[vtk_file],
        ).convert(str(tmp_path / "plane.usd"))

        mesh_prim = stage.GetPrimAtPath("/World/Plane/Mesh")
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        primvar_names = [p.GetPrimvarName() for p in primvars_api.GetPrimvars()]
        assert "vtk_point_pressure" in primvar_names

    def test_time_series_conversion(self, tmp_path: Path) -> None:
        """Multiple VTP files write point time samples and stage time metadata."""
        files = []
        for i in range(3):
            mesh = pv.Plane(i_resolution=2, j_resolution=2)
            mesh.points[:, 2] += float(i)
            path = tmp_path / f"p{i}.vtp"
            mesh.save(str(path))
            files.append(path)

        time_codes = [0.0, 1.0, 2.0]
        stage = ConvertVTKToUSD.from_files(
            data_basename="Plane",
            vtk_files=files,
            time_codes=time_codes,
        ).convert(str(tmp_path / "time.usd"))

        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Plane/Mesh"))
        assert stage.GetStartTimeCode() == 0.0
        assert stage.GetEndTimeCode() == 2.0
        assert mesh.GetPointsAttr().GetTimeSamples() == time_codes


class TestVTKToUSDConversion:
    """Test ConvertVTKToUSD on optional real VTK data."""

    def test_single_file_conversion(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test converting a single VTK file to USD."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_usd = output_dir / "heart_surface.usd"
        stage = ConvertVTKToUSD.from_files(
            data_basename="HeartSurface",
            vtk_files=[kcl_average_surface],
        ).convert(str(output_usd))

        assert output_usd.exists()
        mesh_prim = stage.GetPrimAtPath("/World/HeartSurface/Mesh")
        assert mesh_prim.IsValid()
        assert mesh_prim.IsA(UsdGeom.Mesh)
        assert len(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()) > 0

    def test_conversion_with_material(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test conversion with a custom solid color material."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_usd = output_dir / "heart_with_material.usd"
        stage = ConvertVTKToUSD.from_files(
            data_basename="HeartSurface",
            vtk_files=[kcl_average_surface],
            solid_color=(0.9, 0.3, 0.3),
        ).convert(str(output_usd))

        material_prim = stage.GetPrimAtPath("/World/Looks/Mesh_material")
        assert material_prim.IsValid()
        assert material_prim.IsA(UsdShade.Material)

        mesh_prim = stage.GetPrimAtPath("/World/HeartSurface/Mesh")
        binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
        bound_material = binding_api.ComputeBoundMaterial()[0]
        assert bound_material.GetPrim().IsValid()

        shader_prim = stage.GetPrimAtPath("/World/Looks/Mesh_material/PreviewSurface")
        shader = UsdShade.Shader(shader_prim)
        diffuse_value = shader.GetInput("diffuseColor").Get()
        assert tuple(diffuse_value) == pytest.approx((0.9, 0.3, 0.3), abs=1e-5)

    def test_conversion_settings(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test that ConvertVTKToUSD applies correct default stage metadata."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        stage = ConvertVTKToUSD.from_files(
            data_basename="Mesh",
            vtk_files=[kcl_average_surface],
        ).convert(str(output_dir / "heart_custom_settings.usd"))

        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y


@pytest.mark.slow
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_end_to_end_conversion(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test complete conversion workflow with all features."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_usd = output_dir / "heart_complete.usd"
        stage = ConvertVTKToUSD.from_files(
            data_basename="CardiacModel",
            vtk_files=[kcl_average_surface],
            solid_color=(0.85, 0.2, 0.2),
            frames_per_second=24.0,
        ).convert(str(output_usd))

        mesh_prim = stage.GetPrimAtPath("/World/CardiacModel/Mesh")
        assert output_usd.exists()
        assert mesh_prim.IsValid()
        assert len(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()) > 0
        assert stage.GetPrimAtPath("/World/Looks/Mesh_material").IsValid()
        assert UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvars()


class TestUnitScaling:
    """Verify that VTK mm coordinates are converted to USD meter coordinates."""

    def test_mm_to_m_point_scaling(self, tmp_path: Path) -> None:
        """Points written to USD must be 0.001x their original mm values."""
        mesh = pv.Sphere(radius=100.0)
        output_usd = tmp_path / "sphere.usd"

        stage = ConvertVTKToUSD(
            data_basename="Sphere",
            input_polydata=[mesh],
        ).convert(str(output_usd))

        mesh_prim = stage.GetPrimAtPath("/World/Sphere/Mesh")
        usd_mesh = UsdGeom.Mesh(mesh_prim)
        usd_points = usd_mesh.GetPointsAttr().Get()
        assert usd_points is not None and len(usd_points) > 0

        coords = np.array(usd_points)
        max_coord = float(np.abs(coords).max())
        assert max_coord < 0.15
        assert max_coord > 0.05

    def test_normals_remain_unit_length(self, tmp_path: Path) -> None:
        """Normal vectors must not be scaled."""
        mesh = pv.Sphere(radius=100.0)
        mesh.compute_normals(inplace=True)
        output_usd = tmp_path / "sphere_normals.usd"

        stage = ConvertVTKToUSD(
            data_basename="Sphere",
            input_polydata=[mesh],
        ).convert(str(output_usd))

        mesh_prim = stage.GetPrimAtPath("/World/Sphere/Mesh")
        normals_attr = UsdGeom.Mesh(mesh_prim).GetNormalsAttr()
        if normals_attr is None or normals_attr.Get() is None:
            pytest.skip("No normals on this mesh")

        normals = np.array(normals_attr.Get())
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)

    def test_stage_meters_per_unit(self, tmp_path: Path) -> None:
        """Stage metersPerUnit metadata must be 1.0."""
        mesh = pv.Sphere(radius=100.0)
        output_usd = tmp_path / "sphere_meta.usd"

        stage = ConvertVTKToUSD(
            data_basename="Sphere",
            input_polydata=[mesh],
        ).convert(str(output_usd))

        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
