"""
Test for VTK to USD conversion.

This test depends on test_contour_tools and uses the extracted contours
to test USD conversion functionality.
"""

from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest
import pyvista as pv
from pxr import UsdGeom

from monai_physio import ConvertVTKToUSD
from monai_physio.contour_tools import ContourTools


def _make_poly(label_ids: list[int] | None = None) -> pv.PolyData:
    """Return a small synthetic PolyData (9 quad cells, 16 points).

    Args:
        label_ids: If given, attached as ``cell_data['boundary_labels']`` (int32).
            Must have exactly 9 elements to match the plane's cell count.
    """
    mesh = pv.Plane(i_resolution=3, j_resolution=3)
    if label_ids is not None:
        mesh.cell_data["boundary_labels"] = np.array(label_ids, dtype=np.int32)
    return mesh


@pytest.mark.slow
class TestConvertVTKToUSD:
    """Test suite for VTK to USD PolyMesh conversion."""

    @pytest.fixture(scope="class")
    def contour_meshes(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> list[Any]:
        """Extract or load contour meshes for USD conversion testing."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"

        # Check if contour files exist
        heart_contour_000 = contour_output_dir / "heart_contours_slice000.vtp"
        heart_contour_001 = contour_output_dir / "heart_contours_slice001.vtp"

        if not heart_contour_000.exists() or not heart_contour_001.exists():
            # Extract contours if they don't exist
            print("\nContour files not found, extracting them...")
            contour_output_dir.mkdir(parents=True, exist_ok=True)

            meshes = []
            for i, result in enumerate(test_labelmaps):
                heart_mask = result["heart"]
                contours = contour_tools.extract_contours(heart_mask)
                meshes.append(contours)

                # Save contours
                output_file = contour_output_dir / f"heart_contours_slice{i:03d}.vtp"
                contours.save(str(output_file))

            return meshes
        # Load existing contours
        print("\nLoading existing contour files...")
        loaded_meshes: list[Any] = [
            pv.read(str(contour_output_dir / "heart_contours_slice000.vtp")),
            pv.read(str(contour_output_dir / "heart_contours_slice001.vtp")),
        ]
        return loaded_meshes

    def test_converter_initialization(self) -> None:
        """Test that ConvertVTKToUSD initializes correctly."""
        converter = ConvertVTKToUSD(
            data_basename="TestModel", input_polydata=[], mask_ids=None
        )

        assert converter is not None, "Converter not initialized"
        assert converter.data_basename == "TestModel", "Data basename not set correctly"

        print("\nConverter initialized successfully")

    def test_supports_mesh_type(self, contour_meshes: list[Any]) -> None:
        """Test that converter correctly identifies supported mesh types."""
        mesh = contour_meshes[0]

        converter = ConvertVTKToUSD(
            data_basename="TestModel", input_polydata=[mesh], mask_ids=None
        )

        # PolyData should be supported
        assert converter.supports_mesh_type(mesh), "PolyData should be supported"

        print("\nMesh type support check passed")

    def test_convert_single_time_point(
        self, contour_meshes: list[Any], test_directories: dict[str, Path]
    ) -> None:
        """Test converting a single time point to USD."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Use first time point only
        mesh = contour_meshes[0]

        print("\nConverting single time point to USD...")
        print(f"  Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

        converter = ConvertVTKToUSD(
            data_basename="HeartSingle", input_polydata=[mesh], mask_ids=None
        )

        output_file = usd_output_dir / "heart_single_time.usd"
        stage = converter.convert(str(output_file))

        # Verify USD stage was created
        assert stage is not None, "USD stage not created"
        assert output_file.exists(), f"USD file not created: {output_file}"

        # Verify stage contents (actual path includes /World prefix)
        prim = stage.GetPrimAtPath("/World/HeartSingle")
        assert prim.IsValid(), "Root prim not found at /World/HeartSingle"

        print("Single time point converted to USD")
        print(f"  Output: {output_file}")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")

    def test_convert_multiple_time_points(
        self, contour_meshes: list[Any], test_directories: dict[str, Path]
    ) -> None:
        """Test converting multiple time points to USD."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        print("\nConverting multiple time points to USD...")
        print(f"  Time points: {len(contour_meshes)}")

        converter = ConvertVTKToUSD(
            data_basename="HeartMulti", input_polydata=contour_meshes, mask_ids=None
        )

        output_file = usd_output_dir / "heart_multi_time.usd"
        stage = converter.convert(str(output_file))

        # Verify USD stage
        assert stage is not None, "USD stage not created"
        assert output_file.exists(), f"USD file not created: {output_file}"

        # Verify time samples (actual path includes /World prefix)
        prim = stage.GetPrimAtPath("/World/HeartMulti")
        assert prim.IsValid(), "Root prim not found at /World/HeartMulti"

        # Check that mesh exists
        mesh_path = "/World/HeartMulti/Mesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        assert mesh_prim.IsValid(), f"Mesh not found at {mesh_path}"

        print("Multiple time points converted to USD")
        print(f"  Output: {output_file}")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")

    def test_convert_with_deformation(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> None:
        """Test converting meshes with deformation magnitude."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Extract contours
        heart_mask = test_labelmaps[0]["heart"]
        contours = contour_tools.extract_contours(heart_mask)

        # Add deformation magnitude (simulate with random values)
        import numpy as np

        deformation = np.random.uniform(0, 5, contours.n_points)
        contours["DeformationMagnitude"] = deformation

        print("\nConverting mesh with deformation magnitude...")

        converter = ConvertVTKToUSD(
            data_basename="HeartDeformation", input_polydata=[contours], mask_ids=None
        )

        output_file = usd_output_dir / "heart_with_deformation.usd"
        stage = converter.convert(str(output_file))

        assert stage is not None, "USD stage not created"
        assert output_file.exists(), "USD file not created"

        # Check that mesh was created (actual path includes /World prefix)
        mesh_path = "/World/HeartDeformation/Mesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        assert mesh_prim.IsValid(), f"Mesh not found at {mesh_path}"

        print("Mesh with deformation converted to USD")
        print(f"  Output: {output_file}")

    def test_convert_with_colormap(
        self, contour_meshes: list[Any], test_directories: dict[str, Path]
    ) -> None:
        """Test converting meshes with colormap visualization."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Add scalar field to mesh for colormapping
        import numpy as np

        mesh = contour_meshes[0]
        scalar_values = np.random.uniform(0, 100, mesh.n_points)
        mesh["pressure"] = scalar_values

        print("\nConverting mesh with colormap...")

        converter = ConvertVTKToUSD(
            data_basename="HeartColormap", input_polydata=[mesh], mask_ids=None
        )

        # Set colormap
        converter.set_colormap(color_by_array="pressure", colormap="plasma")

        output_file = usd_output_dir / "heart_with_colormap.usd"
        stage = converter.convert(str(output_file))

        assert stage is not None, "USD stage not created"
        assert output_file.exists(), "USD file not created"

        # Verify mesh was created (actual path includes /World prefix)
        mesh_path = "/World/HeartColormap/Mesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        assert mesh_prim.IsValid(), f"Mesh not found at {mesh_path}"

        print("Mesh with colormap converted to USD")
        print("  Colormap: plasma")
        print(f"  Output: {output_file}")

    def test_convert_unstructured_grid_to_surface(
        self, test_directories: dict[str, Path]
    ) -> None:
        """Test converting UnstructuredGrid to surface mesh."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Create a simple UnstructuredGrid (cube)
        import numpy as np
        import pyvista as pv

        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        ).astype(np.float32)
        cells = [8, 0, 1, 2, 3, 4, 5, 6, 7]
        cell_types = [12]  # VTK_HEXAHEDRON

        ugrid = pv.UnstructuredGrid(cells, cell_types, points)

        print("\nConverting UnstructuredGrid to USD...")
        print(f"  Grid: {ugrid.n_points} points, {ugrid.n_cells} cells")

        converter = ConvertVTKToUSD(
            data_basename="CubeSurface",
            input_polydata=[ugrid],
            mask_ids=None,
            convert_to_surface=True,
        )

        output_file = usd_output_dir / "cube_surface.usd"
        stage = converter.convert(str(output_file))

        assert stage is not None, "USD stage not created"
        assert output_file.exists(), "USD file not created"

        print("UnstructuredGrid converted to surface USD")
        print(f"  Output: {output_file}")

    def test_usd_file_structure(
        self, contour_meshes: list[Any], test_directories: dict[str, Path]
    ) -> None:
        """Test the structure of generated USD file."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        converter = ConvertVTKToUSD(
            data_basename="HeartStructure",
            input_polydata=[contour_meshes[0]],
            mask_ids=None,
        )

        output_file = usd_output_dir / "heart_structure_test.usd"
        stage = converter.convert(str(output_file))

        print("\nVerifying USD file structure...")

        # Check root prim (actual path includes /World prefix)
        root_prim = stage.GetPrimAtPath("/World/HeartStructure")
        assert root_prim.IsValid(), "Root prim not found at /World/HeartStructure"
        assert UsdGeom.Xform(root_prim), "Root should be an Xform"

        # Check mesh structure
        mesh_prim = stage.GetPrimAtPath("/World/HeartStructure/Mesh")
        assert mesh_prim.IsValid(), "Mesh prim not found at /World/HeartStructure/Mesh"

        print("USD file structure verified")
        print(f"  Root: {root_prim.GetPath()}")
        print(f"  Mesh: {mesh_prim.GetPath()}")

    def test_time_varying_topology(
        self, contour_meshes: list[Any], test_directories: dict[str, Path]
    ) -> None:
        """Test handling of time-varying topology."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Modify one mesh to have different topology
        mesh1 = contour_meshes[0].copy()
        mesh2 = contour_meshes[1].copy()

        # Decimate second mesh to change topology
        mesh2 = mesh2.decimate(0.5)

        print("\nConverting meshes with varying topology...")
        print(f"  Mesh 1: {mesh1.n_points} points, {mesh1.n_cells} cells")
        print(f"  Mesh 2: {mesh2.n_points} points, {mesh2.n_cells} cells")

        converter = ConvertVTKToUSD(
            data_basename="HeartVarying", input_polydata=[mesh1, mesh2], mask_ids=None
        )

        output_file = usd_output_dir / "heart_varying_topology.usd"
        stage = converter.convert(str(output_file))

        assert stage is not None, "USD stage not created"
        assert output_file.exists(), "USD file not created"

        # Each frame carries its own topology, so neither one is indexed by the
        # other's faces.
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/HeartVarying/Mesh"))
        assert mesh.GetPrim().IsValid(), "Mesh prim not found"
        indices_attr = mesh.GetFaceVertexIndicesAttr()
        assert indices_attr.GetTimeSamples() == [0.0, 1.0]
        for time_code in (0.0, 1.0):
            points = mesh.GetPointsAttr().Get(time_code)
            indices = indices_attr.Get(time_code)
            assert max(indices) < len(points), (
                f"Face indices at t={time_code} address points that frame "
                f"does not have ({max(indices)} >= {len(points)})"
            )

        print("Time-varying topology handled")
        print(f"  Output: {output_file}")

    def test_batch_conversion(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> None:
        """Test converting multiple anatomy structures in batch."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Extract contours from multiple anatomies
        anatomy_groups = ["lung", "heart"]
        meshes_dict = {}

        for group in anatomy_groups:
            mask = test_labelmaps[0][group]
            mask_arr = itk.array_from_image(mask)

            import numpy as np

            if np.sum(mask_arr > 0) > 100:
                contours = contour_tools.extract_contours(mask)
                meshes_dict[group] = contours

        if len(meshes_dict) >= 2:
            print(f"\nConverting {len(meshes_dict)} anatomy structures...")

            # Convert each anatomy separately
            for anatomy, mesh in meshes_dict.items():
                converter = ConvertVTKToUSD(
                    data_basename=f"{anatomy.capitalize()}",
                    input_polydata=[mesh],
                    mask_ids=None,
                )

                output_file = usd_output_dir / f"{anatomy}_anatomy.usd"
                stage = converter.convert(str(output_file))

                assert stage is not None, f"USD stage not created for {anatomy}"
                assert output_file.exists(), f"USD file not created for {anatomy}"

                print(f"  {anatomy}: {output_file}")

            print("Batch conversion complete")
        else:
            pytest.skip("Not enough anatomies with sufficient voxels")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


class TestSyntheticConversion:
    """Synthetic (no-disk-data) tests for ConvertVTKToUSD.

    Covers:
    - Gap C: single-frame prim carries explicit time sample after create_time_varying_mesh change
    - Gap D: mask_ids / _convert_with_labels - per-label prims, time-code filtering
    - Gap E: static-merge prim naming uses data_basename
    """

    # ------------------------------------------------------------------
    # Gap C - single-part prim must carry explicit time sample
    # ------------------------------------------------------------------

    def test_single_frame_prim_has_time_sample(self, tmp_path: Path) -> None:
        """Single-frame _convert_unified() must author one time sample, not a static prim."""
        mesh = _make_poly()
        converter = ConvertVTKToUSD(data_basename="P", input_polydata=[mesh])
        stage = converter.convert(str(tmp_path / "out.usd"))

        prim = stage.GetPrimAtPath("/World/P/Mesh")
        assert prim.IsValid(), "Mesh prim not found"
        samples = UsdGeom.Mesh(prim).GetPointsAttr().GetTimeSamples()
        assert len(samples) == 1, f"Expected 1 time sample, got {len(samples)}"
        assert samples[0] == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # Gap E - static-merge prim naming uses data_basename
    # ------------------------------------------------------------------

    def test_static_merge_prim_names_use_data_basename(self, tmp_path: Path) -> None:
        """Static-merge prims must be named {data_basename}_{i}, not Mesh_{i}."""
        mesh_a, mesh_b = _make_poly(), _make_poly()
        converter = ConvertVTKToUSD(
            data_basename="Organ", input_polydata=[mesh_a, mesh_b]
        )
        converter._is_static_merge = True
        stage = converter.convert(str(tmp_path / "out.usd"))

        assert stage.GetPrimAtPath("/World/Organ/Organ_0").IsValid()
        assert stage.GetPrimAtPath("/World/Organ/Organ_1").IsValid()
        assert not stage.GetPrimAtPath("/World/Organ/Mesh_0").IsValid(), (
            "Old naming still present"
        )
        assert not stage.GetPrimAtPath("/World/Organ/Mesh_1").IsValid(), (
            "Old naming still present"
        )
        # Static prims carry no time samples
        for prim_path in ("/World/Organ/Organ_0", "/World/Organ/Organ_1"):
            samples = (
                UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
                .GetPointsAttr()
                .GetTimeSamples()
            )
            assert samples == [], (
                f"{prim_path} should have no time samples but got {samples}"
            )

    # ------------------------------------------------------------------
    # Frames that share no triangulation must each author their own topology
    # ------------------------------------------------------------------

    def test_varying_topology_time_samples_faces(self, tmp_path: Path) -> None:
        """Independently built frames must not be indexed by frame 0's faces."""
        frames = [
            pv.Sphere(theta_resolution=8 + index, phi_resolution=8)
            for index in range(3)
        ]
        converter = ConvertVTKToUSD(data_basename="P", input_polydata=frames)
        stage = converter.convert(str(tmp_path / "out.usd"))

        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/P/Mesh"))
        counts_attr = mesh.GetFaceVertexCountsAttr()
        indices_attr = mesh.GetFaceVertexIndicesAttr()
        assert counts_attr.GetTimeSamples() == [0.0, 1.0, 2.0]
        assert indices_attr.GetTimeSamples() == [0.0, 1.0, 2.0]

        for time_code in (0.0, 1.0, 2.0):
            points = mesh.GetPointsAttr().Get(time_code)
            indices = indices_attr.Get(time_code)
            counts = counts_attr.Get(time_code)
            assert max(indices) < len(points), (
                f"Face indices at t={time_code} address points that frame "
                f"does not have ({max(indices)} >= {len(points)})"
            )
            assert sum(counts) == len(indices)

    def test_constant_topology_leaves_faces_untimed(self, tmp_path: Path) -> None:
        """A deformed series keeps one topology, so viewers can interpolate."""
        base = _make_poly()
        moved = base.copy()
        moved.points[:, 2] += 1.0
        converter = ConvertVTKToUSD(data_basename="P", input_polydata=[base, moved])
        stage = converter.convert(str(tmp_path / "out.usd"))

        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/P/Mesh"))
        assert mesh.GetFaceVertexCountsAttr().GetTimeSamples() == []
        assert mesh.GetFaceVertexIndicesAttr().GetTimeSamples() == []
        assert mesh.GetPointsAttr().GetTimeSamples() == [0.0, 1.0]

    def test_mask_ids_split_on_segmentation_label_ids(self, tmp_path: Path) -> None:
        """A merged surface file splits on the array save_combined_surfaces writes."""
        mesh = _make_poly()
        mesh.cell_data["SegmentationLabelIds"] = np.array(
            [141] * 4 + [142] * 5, dtype=np.int32
        )
        converter = ConvertVTKToUSD(
            data_basename="P",
            input_polydata=[mesh],
            mask_ids={141: "atrium_left", 142: "ventricle_left"},
        )
        stage = converter.convert(str(tmp_path / "out.usd"))

        assert stage.GetPrimAtPath("/World/P/Anatomy/atrium_left").IsValid()
        assert stage.GetPrimAtPath("/World/P/Anatomy/ventricle_left").IsValid()

    def test_static_merge_object_names_name_prims(self, tmp_path: Path) -> None:
        """object_names replaces the positional {data_basename}_{i} naming."""
        mesh_a, mesh_b = _make_poly(), _make_poly()
        converter = ConvertVTKToUSD(
            data_basename="Organ",
            input_polydata=[mesh_a, mesh_b],
            static_merge=True,
            object_names=["myocardium", "ventricle_left"],
        )
        stage = converter.convert(str(tmp_path / "out.usd"))

        assert stage.GetPrimAtPath("/World/Organ/myocardium").IsValid()
        assert stage.GetPrimAtPath("/World/Organ/ventricle_left").IsValid()
        assert not stage.GetPrimAtPath("/World/Organ/Organ_0").IsValid(), (
            "Positional naming still present"
        )

    def test_object_names_length_mismatch_raises(self) -> None:
        """A short object_names list would silently mis-name prims."""
        with pytest.raises(ValueError, match="object_names length"):
            ConvertVTKToUSD(
                data_basename="Organ",
                input_polydata=[_make_poly(), _make_poly()],
                static_merge=True,
                object_names=["myocardium"],
            )

    # ------------------------------------------------------------------
    # Gap D - mask_ids / _convert_with_labels
    # ------------------------------------------------------------------

    def test_mask_ids_basic_produces_per_label_prims(self, tmp_path: Path) -> None:
        """mask_ids must produce one USD prim per label grouped under /Anatomy
        when no segmenter is supplied; no unified /Mesh prim."""
        label_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2]
        mesh = _make_poly(label_ids=label_ids)
        converter = ConvertVTKToUSD(
            data_basename="Heart",
            input_polydata=[mesh],
            mask_ids={1: "ventricle", 2: "atrium"},
        )
        stage = converter.convert(str(tmp_path / "out.usd"))

        ventricle = stage.GetPrimAtPath("/World/Heart/Anatomy/ventricle")
        atrium = stage.GetPrimAtPath("/World/Heart/Anatomy/atrium")
        assert ventricle.IsValid(), "ventricle prim not found"
        assert atrium.IsValid(), "atrium prim not found"
        assert ventricle.IsA(UsdGeom.Mesh)
        assert atrium.IsA(UsdGeom.Mesh)
        assert list(UsdGeom.Mesh(ventricle).GetPointsAttr().GetTimeSamples()) == [0.0]
        assert list(UsdGeom.Mesh(atrium).GetPointsAttr().GetTimeSamples()) == [0.0]
        # Unified prim must NOT exist when mask_ids is active
        assert not stage.GetPrimAtPath("/World/Heart/Mesh").IsValid()
        # Materials live under the same /Anatomy group as the meshes
        assert stage.GetPrimAtPath("/World/Looks/Anatomy/ventricle_material").IsValid()
        assert stage.GetPrimAtPath("/World/Looks/Anatomy/atrium_material").IsValid()

    def test_structured_grid_extracts_surface(self, tmp_path: Path) -> None:
        """StructuredGrid input is surface-extracted when convert_to_surface is true."""
        x, y, z = np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], indexing="ij")
        grid = pv.StructuredGrid(x, y, z)
        converter = ConvertVTKToUSD(
            data_basename="Grid",
            input_polydata=[grid],
            convert_to_surface=True,
        )
        stage = converter.convert(str(tmp_path / "grid.usd"))

        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Grid/Mesh"))
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        assert face_counts is not None
        assert len(face_counts) > 0

    def test_mask_ids_missing_label_filters_time_codes(self, tmp_path: Path) -> None:
        """Time codes for a label must be filtered to frames where it actually appears.

        3-frame setup:
          Frame 0 (t=0.0): labels 1 and 2 both present
          Frame 1 (t=1.0): only label 1
          Frame 2 (t=2.0): labels 1 and 2 both present

        Both labels have > 1 frame, so create_time_varying_mesh() is called for each.
        The atrium (label 2) must carry time samples [0.0, 2.0], not [0.0, 1.0, 2.0].
        """
        mesh0 = _make_poly(label_ids=[1, 1, 1, 1, 1, 2, 2, 2, 2])
        mesh1 = _make_poly(label_ids=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        mesh2 = _make_poly(label_ids=[1, 1, 1, 1, 1, 2, 2, 2, 2])
        converter = ConvertVTKToUSD(
            data_basename="Heart",
            input_polydata=[mesh0, mesh1, mesh2],
            mask_ids={1: "ventricle", 2: "atrium"},
        )
        converter._time_codes = [0.0, 1.0, 2.0]
        stage = converter.convert(str(tmp_path / "out.usd"))

        ventricle = stage.GetPrimAtPath("/World/Heart/Anatomy/ventricle")
        atrium = stage.GetPrimAtPath("/World/Heart/Anatomy/atrium")
        assert ventricle.IsValid()
        assert atrium.IsValid()

        v_samples = UsdGeom.Mesh(ventricle).GetPointsAttr().GetTimeSamples()
        a_samples = UsdGeom.Mesh(atrium).GetPointsAttr().GetTimeSamples()
        assert list(v_samples) == [0.0, 1.0, 2.0], f"ventricle samples: {v_samples}"
        # atrium absent from frame 1 - time code 1.0 must NOT appear
        assert list(a_samples) == [0.0, 2.0], (
            f"atrium should only appear at t=0 and t=2, got {a_samples}"
        )

    def test_mask_ids_missing_boundary_labels_falls_back(self, tmp_path: Path) -> None:
        """Mesh without boundary_labels array falls back to a 'default' prim."""
        mesh = _make_poly()  # no boundary_labels
        converter = ConvertVTKToUSD(
            data_basename="FB",
            input_polydata=[mesh],
            mask_ids={1: "ventricle"},
        )
        stage = converter.convert(str(tmp_path / "out.usd"))

        assert stage.GetPrimAtPath("/World/FB/Anatomy/default").IsValid(), (
            "'default' fallback prim missing"
        )
        assert not stage.GetPrimAtPath("/World/FB/Anatomy/ventricle").IsValid()

    def test_mask_ids_groups_by_segmenter_type(self, tmp_path: Path) -> None:
        """When a segmenter is supplied, labels are grouped under their
        anatomy type (heart/lung/...) for both meshes and materials."""
        from monai_physio.segment_chest_total_segmentator import (
            SegmentChestTotalSegmentator,
        )

        # Label 51 is "heart" (heart group), 10 is "lung_upper_lobe_left" (lung group).
        label_ids = [51, 51, 51, 51, 51, 10, 10, 10, 10]
        mesh = _make_poly(label_ids=label_ids)
        seg = SegmentChestTotalSegmentator()
        converter = ConvertVTKToUSD(
            data_basename="Chest",
            input_polydata=[mesh],
            mask_ids={51: "heart", 10: "lung_upper_lobe_left"},
            segmenter=seg,
        )
        stage = converter.convert(str(tmp_path / "out.usd"))

        # Meshes grouped under per-type Xforms
        heart_mesh = stage.GetPrimAtPath("/World/Chest/heart/heart")
        lung_mesh = stage.GetPrimAtPath("/World/Chest/lung/lung_upper_lobe_left")
        assert heart_mesh.IsValid(), "heart mesh should land under /heart group"
        assert lung_mesh.IsValid(), "lung mesh should land under /lung group"
        assert heart_mesh.IsA(UsdGeom.Mesh)
        assert lung_mesh.IsA(UsdGeom.Mesh)

        # Materials mirror the mesh hierarchy
        assert stage.GetPrimAtPath("/World/Looks/heart/heart_material").IsValid()
        assert stage.GetPrimAtPath(
            "/World/Looks/lung/lung_upper_lobe_left_material"
        ).IsValid()
        # Flat fallback group must NOT be defined when segmenter classifies all labels
        assert not stage.GetPrimAtPath("/World/Chest/Anatomy").IsValid()
