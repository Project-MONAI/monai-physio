"""
Test for converting a 4D image to a 3D time series using ITK readers.

This test depends on test_download_heart_data and replicates the functionality
from cell 3 of the notebook Heart-GatedCT_To_USD/0-download_and_convert_4d_to_3d.ipynb.
"""

from pathlib import Path

import pytest

from monai_physio.convert_image_4d_to_3d import ConvertImage4DTo3D


class TestConvertImage4DTo3D:
    """Test suite for converting a 4D image to a 3D time series."""

    def test_load_image_4d_and_save_3d_images(
        self,
        download_test_data: Path,
        test_directories: dict[str, Path],
    ) -> None:
        """Test loading a 4D image."""
        input_4d_file = download_test_data

        conv = ConvertImage4DTo3D()
        assert conv is not None, "Converter is not initialized"

        conv.load_image_4d(str(input_4d_file))

        num_time_points = conv.get_number_of_3d_images()
        assert num_time_points > 0, "No time points found in 4D image"

        print(f"\nLoaded 4D image with {conv.get_number_of_3d_images()} time points")

        output_dir = test_directories["output"] / "convert_image_4d_to_3d"
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in output_dir.glob("test_slice_*.mha"):
            stale_file.unlink()

        conv.save_3d_images(output_dir, "test_slice")

        test_slice_files = list(output_dir.glob("test_slice_*.mha"))
        assert len(test_slice_files) > 0, "No test slice files were created"
        assert len(test_slice_files) == num_time_points, (
            f"Expected {num_time_points} files, found {len(test_slice_files)}"
        )

        print(f"\nSaved {len(test_slice_files)} 3D images")

        for test_file in test_slice_files:
            test_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
