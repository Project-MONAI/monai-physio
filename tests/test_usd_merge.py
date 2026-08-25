"""
Test USD file merging with material and time-series preservation.

This test validates that both merge_usd_files() and merge_usd_files_flattened()
properly preserve materials, material bindings, and time-varying animation data.
"""

from pathlib import Path
from typing import Any

import pytest
from pxr import Usd, UsdGeom, UsdShade

from physiotwin4d import USDTools


def analyze_usd_file(filepath: str) -> dict[str, Any]:
    """
    Analyze a USD file for materials and time samples.

    Parameters
    ----------
    filepath : str
        Path to USD file

    Returns
    -------
    dict
        Statistics about the USD file including material and mesh counts
    """
    stage = Usd.Stage.Open(filepath)

    # Count materials
    materials = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            materials.append(prim.GetPath())

    # Count meshes and check for material bindings
    meshes_with_materials = 0
    meshes_without_materials = 0
    time_varying_meshes = 0
    max_time_samples = 0

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            # Check material binding
            bindingAPI = UsdShade.MaterialBindingAPI(prim)
            material = bindingAPI.ComputeBoundMaterial()
            if bool(material):
                meshes_with_materials += 1
            else:
                meshes_without_materials += 1

            # Check for time-varying attributes
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr()
            time_samples = points_attr.GetTimeSamples()
            if time_samples:
                time_varying_meshes += 1
                max_time_samples = max(max_time_samples, len(time_samples))

    return {
        "materials": len(materials),
        "meshes_with_materials": meshes_with_materials,
        "meshes_without_materials": meshes_without_materials,
        "time_varying_meshes": time_varying_meshes,
        "max_time_samples": max_time_samples,
    }


class TestUSDMerge:
    """Test suite for USD file merging."""

    @pytest.fixture(scope="class")
    @staticmethod
    def test_data_files() -> dict[str, str]:
        """Locate test USD files with materials and time-varying data."""
        dynamic_file = Path(
            "experiments/Heart-GatedCT_To_USD/results/Slicer_CardiacGatedCT.dynamic_anatomy_painted.usd"
        )
        static_file = Path(
            "experiments/Heart-GatedCT_To_USD/results/Slicer_CardiacGatedCT.static_anatomy_painted.usd"
        )

        if not dynamic_file.exists() or not static_file.exists():
            pytest.skip("Test data not available. Run Heart-GatedCT experiments first.")

        return {"dynamic": str(dynamic_file), "static": str(static_file)}

    @pytest.fixture(scope="class")
    @staticmethod
    def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Create temporary output directory for test results."""
        output_dir = tmp_path_factory.mktemp("usd_merge_tests")
        return output_dir

    @pytest.fixture(scope="class")
    @staticmethod
    def input_stats(test_data_files: dict[str, str]) -> dict[str, dict[str, Any]]:
        """Analyze input USD files."""
        dynamic_stats = analyze_usd_file(test_data_files["dynamic"])
        static_stats = analyze_usd_file(test_data_files["static"])
        return {"dynamic": dynamic_stats, "static": static_stats}

    def test_merge_usd_files_copy_method(
        self,
        test_data_files: dict[str, str],
        input_stats: dict[str, dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Test merge_usd_files() manual copy method."""
        # Merge files using manual copy method
        usd_tools = USDTools()
        merged_file = output_dir / "test_merged_copy.usd"

        usd_tools.merge_usd_files(
            str(merged_file), [test_data_files["dynamic"], test_data_files["static"]]
        )

        # Verify file was created
        assert merged_file.exists(), "Merged USD file was not created"

        # Analyze merged file
        merged_stats = analyze_usd_file(str(merged_file))

        # Verify material preservation
        expected_materials = (
            input_stats["dynamic"]["materials"] + input_stats["static"]["materials"]
        )
        assert merged_stats["materials"] == expected_materials, (
            f"Materials not preserved: expected {expected_materials}, got {merged_stats['materials']}"
        )

        # Verify mesh material bindings
        expected_meshes = (
            input_stats["dynamic"]["meshes_with_materials"]
            + input_stats["static"]["meshes_with_materials"]
        )
        assert merged_stats["meshes_with_materials"] == expected_meshes, (
            f"Material bindings not preserved: expected {expected_meshes}, got {merged_stats['meshes_with_materials']}"
        )

        # Verify no meshes lost materials
        assert merged_stats["meshes_without_materials"] == 0, (
            f"Some meshes lost material bindings: {merged_stats['meshes_without_materials']} meshes without materials"
        )

        # Verify time-varying mesh preservation
        expected_time_varying = (
            input_stats["dynamic"]["time_varying_meshes"]
            + input_stats["static"]["time_varying_meshes"]
        )
        assert merged_stats["time_varying_meshes"] == expected_time_varying, (
            f"Time-varying meshes not preserved: expected {expected_time_varying}, got {merged_stats['time_varying_meshes']}"
        )

        # Verify time sample count
        expected_max_samples = max(
            input_stats["dynamic"]["max_time_samples"],
            input_stats["static"]["max_time_samples"],
        )
        assert merged_stats["max_time_samples"] == expected_max_samples, (
            f"Time samples not preserved: expected {expected_max_samples}, got {merged_stats['max_time_samples']}"
        )

    def test_merge_usd_files_flattened_method(
        self,
        test_data_files: dict[str, str],
        input_stats: dict[str, dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Test merge_usd_files_flattened() composition method."""
        # Merge files using flattened method
        usd_tools = USDTools()
        merged_file = output_dir / "test_merged_flattened.usd"

        usd_tools.merge_usd_files_flattened(
            str(merged_file), [test_data_files["dynamic"], test_data_files["static"]]
        )

        # Verify file was created
        assert merged_file.exists(), "Merged USD file was not created"

        # Analyze merged file
        merged_stats = analyze_usd_file(str(merged_file))

        # Verify material preservation
        expected_materials = (
            input_stats["dynamic"]["materials"] + input_stats["static"]["materials"]
        )
        assert merged_stats["materials"] == expected_materials, (
            f"Materials not preserved: expected {expected_materials}, got {merged_stats['materials']}"
        )

        # Verify mesh material bindings
        expected_meshes = (
            input_stats["dynamic"]["meshes_with_materials"]
            + input_stats["static"]["meshes_with_materials"]
        )
        assert merged_stats["meshes_with_materials"] == expected_meshes, (
            f"Material bindings not preserved: expected {expected_meshes}, got {merged_stats['meshes_with_materials']}"
        )

        # Verify no meshes lost materials
        assert merged_stats["meshes_without_materials"] == 0, (
            f"Some meshes lost material bindings: {merged_stats['meshes_without_materials']} meshes without materials"
        )

        # Verify time-varying mesh preservation
        expected_time_varying = (
            input_stats["dynamic"]["time_varying_meshes"]
            + input_stats["static"]["time_varying_meshes"]
        )
        assert merged_stats["time_varying_meshes"] == expected_time_varying, (
            f"Time-varying meshes not preserved: expected {expected_time_varying}, got {merged_stats['time_varying_meshes']}"
        )

        # Verify time sample count
        expected_max_samples = max(
            input_stats["dynamic"]["max_time_samples"],
            input_stats["static"]["max_time_samples"],
        )
        assert merged_stats["max_time_samples"] == expected_max_samples, (
            f"Time samples not preserved: expected {expected_max_samples}, got {merged_stats['max_time_samples']}"
        )

    def test_both_methods_produce_equivalent_results(
        self, test_data_files: dict[str, str], output_dir: Path
    ) -> None:
        """Verify both merge methods produce equivalent results."""
        usd_tools = USDTools()

        # Merge with both methods
        copy_file = output_dir / "test_copy.usd"
        flattened_file = output_dir / "test_flattened.usd"

        usd_tools.merge_usd_files(
            str(copy_file), [test_data_files["dynamic"], test_data_files["static"]]
        )

        usd_tools.merge_usd_files_flattened(
            str(flattened_file), [test_data_files["dynamic"], test_data_files["static"]]
        )

        # Analyze both outputs
        copy_stats = analyze_usd_file(str(copy_file))
        flattened_stats = analyze_usd_file(str(flattened_file))

        # Verify equivalence
        assert copy_stats["materials"] == flattened_stats["materials"], (
            "Methods produce different material counts"
        )
        assert (
            copy_stats["meshes_with_materials"]
            == flattened_stats["meshes_with_materials"]
        ), "Methods produce different mesh material binding counts"
        assert (
            copy_stats["time_varying_meshes"] == flattened_stats["time_varying_meshes"]
        ), "Methods produce different time-varying mesh counts"
        assert copy_stats["max_time_samples"] == flattened_stats["max_time_samples"], (
            "Methods produce different time sample counts"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
