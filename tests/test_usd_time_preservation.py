"""
Test USD time-varying data preservation in merged files.

This test validates that merge operations preserve correct time codes,
TimeCodesPerSecond metadata, and actual point position data at all time steps.
"""

from pathlib import Path
from typing import Any, Optional

import pytest
from pxr import Usd, UsdGeom

from monai_physio import USDTools


def get_time_metadata(filepath: str) -> dict[str, Any]:
    """
    Extract time metadata from a USD file.

    Parameters
    ----------
    filepath : str
        Path to USD file

    Returns
    -------
    dict
        Time metadata including start/end times and time codes per second
    """
    stage = Usd.Stage.Open(filepath)

    return {
        "start_time": stage.GetStartTimeCode(),
        "end_time": stage.GetEndTimeCode(),
        "time_codes_per_second": stage.GetTimeCodesPerSecond(),
        "frames_per_second": stage.GetFramesPerSecond(),
    }


def get_mesh_time_samples(
    filepath: str, mesh_name: str = "inferior_vena_cava"
) -> Optional[dict[str, Any]]:
    """
    Get time sample data for a specific mesh in a USD file.

    Parameters
    ----------
    filepath : str
        Path to USD file
    mesh_name : str
        Name of mesh to analyze

    Returns
    -------
    dict or None
        Time sample information including codes and point positions,
        or None if the mesh is not found
    """
    stage = Usd.Stage.Open(filepath)

    # Find the specified mesh
    for prim in stage.Traverse():
        if prim.GetName() == mesh_name and prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr()
            time_samples = points_attr.GetTimeSamples()

            # Get point positions at key time codes
            positions = {}
            for t in [0.0, 1.0, 2.0, 10.0, 20.0]:
                if t <= max(time_samples) if time_samples else 0.0:
                    points = points_attr.Get(t)
                    if points and len(points) > 0:
                        positions[t] = points[0]  # First point

            return {
                "num_samples": len(time_samples),
                "time_codes": list(time_samples),
                "positions": positions,
                "prim_path": str(prim.GetPath()),
            }

    return None


class TestUSDTimePreservation:
    """Test suite for USD time-varying data preservation."""

    @pytest.fixture(scope="class")
    @staticmethod
    def test_data_files() -> dict[str, str]:
        """Locate test USD files with time-varying data."""
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
        output_dir = tmp_path_factory.mktemp("usd_time_tests")
        return output_dir

    @pytest.fixture(scope="class")
    @staticmethod
    def source_metadata(test_data_files: dict[str, str]) -> dict[str, Any]:
        """Get time metadata from source file."""
        return get_time_metadata(test_data_files["dynamic"])

    @pytest.fixture(scope="class")
    @staticmethod
    def source_time_samples(
        test_data_files: dict[str, str],
    ) -> Optional[dict[str, Any]]:
        """Get time sample data from source file."""
        return get_mesh_time_samples(test_data_files["dynamic"])

    def test_merge_copy_preserves_time_metadata(
        self,
        test_data_files: dict[str, str],
        source_metadata: dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Test that merge_usd_files() preserves time metadata."""
        usd_tools = USDTools()
        merged_file = output_dir / "test_time_copy.usd"

        usd_tools.merge_usd_files(
            str(merged_file), [test_data_files["dynamic"], test_data_files["static"]]
        )

        merged_metadata = get_time_metadata(str(merged_file))

        # Verify all time metadata matches
        assert merged_metadata["start_time"] == source_metadata["start_time"], (
            f"Start time not preserved: {merged_metadata['start_time']} != {source_metadata['start_time']}"
        )

        assert merged_metadata["end_time"] == source_metadata["end_time"], (
            f"End time not preserved: {merged_metadata['end_time']} != {source_metadata['end_time']}"
        )

        assert (
            merged_metadata["time_codes_per_second"]
            == source_metadata["time_codes_per_second"]
        ), (
            f"TimeCodesPerSecond not preserved: {merged_metadata['time_codes_per_second']} != {source_metadata['time_codes_per_second']}"
        )

    def test_merge_flattened_preserves_time_metadata(
        self,
        test_data_files: dict[str, str],
        source_metadata: dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Test that merge_usd_files_flattened() preserves time metadata."""
        usd_tools = USDTools()
        merged_file = output_dir / "test_time_flattened.usd"

        usd_tools.merge_usd_files_flattened(
            str(merged_file), [test_data_files["dynamic"], test_data_files["static"]]
        )

        merged_metadata = get_time_metadata(str(merged_file))

        # Verify all time metadata matches
        assert merged_metadata["start_time"] == source_metadata["start_time"], (
            f"Start time not preserved: {merged_metadata['start_time']} != {source_metadata['start_time']}"
        )

        assert merged_metadata["end_time"] == source_metadata["end_time"], (
            f"End time not preserved: {merged_metadata['end_time']} != {source_metadata['end_time']}"
        )

        assert (
            merged_metadata["time_codes_per_second"]
            == source_metadata["time_codes_per_second"]
        ), (
            f"TimeCodesPerSecond not preserved: {merged_metadata['time_codes_per_second']} != {source_metadata['time_codes_per_second']}"
        )

    def test_merge_copy_preserves_time_samples(
        self,
        test_data_files: dict[str, str],
        source_time_samples: Optional[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Test that merge_usd_files() preserves actual time sample data."""
        if source_time_samples is None:
            pytest.skip("Test mesh not found in source data")

        usd_tools = USDTools()
        merged_file = output_dir / "test_samples_copy.usd"

        usd_tools.merge_usd_files(
            str(merged_file), [test_data_files["dynamic"], test_data_files["static"]]
        )

        merged_samples = get_mesh_time_samples(str(merged_file))
        assert merged_samples is not None, "Test mesh not found in merged file"

        # Verify time sample count
        assert merged_samples["num_samples"] == source_time_samples["num_samples"], (
            f"Time sample count mismatch: {merged_samples['num_samples']} != {source_time_samples['num_samples']}"
        )

        # Verify time codes
        assert merged_samples["time_codes"] == source_time_samples["time_codes"], (
            f"Time codes mismatch: {merged_samples['time_codes']} != {source_time_samples['time_codes']}"
        )

        # Verify point positions at key time codes
        for time_code, source_pos in source_time_samples["positions"].items():
            merged_pos = merged_samples["positions"].get(time_code)
            assert merged_pos is not None, f"Position missing at time {time_code}"

            # Compare positions with small tolerance for floating point
            for i in range(3):
                assert abs(merged_pos[i] - source_pos[i]) < 1e-5, (
                    f"Position mismatch at time {time_code}: {merged_pos} != {source_pos}"
                )

    def test_merge_flattened_preserves_time_samples(
        self,
        test_data_files: dict[str, str],
        source_time_samples: Optional[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Test that merge_usd_files_flattened() preserves actual time sample data."""
        if source_time_samples is None:
            pytest.skip("Test mesh not found in source data")

        usd_tools = USDTools()
        merged_file = output_dir / "test_samples_flattened.usd"

        usd_tools.merge_usd_files_flattened(
            str(merged_file), [test_data_files["dynamic"], test_data_files["static"]]
        )

        merged_samples = get_mesh_time_samples(str(merged_file))
        assert merged_samples is not None, "Test mesh not found in merged file"

        # Verify time sample count
        assert merged_samples["num_samples"] == source_time_samples["num_samples"], (
            f"Time sample count mismatch: {merged_samples['num_samples']} != {source_time_samples['num_samples']}"
        )

        # Verify time codes
        assert merged_samples["time_codes"] == source_time_samples["time_codes"], (
            f"Time codes mismatch: {merged_samples['time_codes']} != {source_time_samples['time_codes']}"
        )

        # Verify point positions at key time codes
        for time_code, source_pos in source_time_samples["positions"].items():
            merged_pos = merged_samples["positions"].get(time_code)
            assert merged_pos is not None, f"Position missing at time {time_code}"

            # Compare positions with small tolerance for floating point
            for i in range(3):
                assert abs(merged_pos[i] - source_pos[i]) < 1e-5, (
                    f"Position mismatch at time {time_code}: {merged_pos} != {source_pos}"
                )

    def test_animation_range_matches_actual_motion(
        self,
        test_data_files: dict[str, str],
        source_time_samples: Optional[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """
        Test that the full animation range is accessible.

        This is a regression test for the bug where TimeCodesPerSecond=24
        caused only a fraction of the animation to be visible.
        """
        if source_time_samples is None:
            pytest.skip("Test mesh not found in source data")

        usd_tools = USDTools()
        merged_file = output_dir / "test_animation_range.usd"

        usd_tools.merge_usd_files_flattened(
            str(merged_file), [test_data_files["dynamic"], test_data_files["static"]]
        )

        # Get metadata and samples
        metadata = get_time_metadata(str(merged_file))
        samples = get_mesh_time_samples(str(merged_file))

        # Calculate the time duration that the animation represents
        # This should equal the number of time codes (e.g., 21 codes = 21 seconds at TCPS=1.0)
        assert samples is not None, "Test mesh not found in merged file"
        num_time_codes = len(samples["time_codes"])
        expected_duration = metadata["end_time"] - metadata["start_time"]

        # With TCPS=1.0, the duration in seconds should equal the number of time codes
        # With TCPS=24.0 (the bug), only 21/24 = 0.875 seconds would be used
        tcps = metadata["time_codes_per_second"]
        actual_duration_in_playback = expected_duration / tcps

        # The playback duration should be close to the number of time codes
        # (allowing some tolerance for different TCPS values)
        assert abs(actual_duration_in_playback - (num_time_codes - 1)) < 2.0, (
            f"Animation duration mismatch: {actual_duration_in_playback} seconds playback "
            f"for {num_time_codes} time codes (TCPS={tcps})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
