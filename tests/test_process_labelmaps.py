"""
Tests for ProcessLabelmaps functionality.

Covers thresholding a multi-label labelmap into a binary registration mask,
physically isotropic dilation that respects per-axis spacing, and forcing
selected labels to background via ``exclude_labels``.
"""

from __future__ import annotations

import itk
import numpy as np
import pytest

from monai_physio.process_labelmaps import ProcessLabelmaps


class TestProcessLabelmaps:
    """Test suite for ProcessLabelmaps.convert_labelmap_to_mask."""

    @pytest.fixture
    def labelmap_tools(self) -> ProcessLabelmaps:
        """Create ProcessLabelmaps instance."""
        return ProcessLabelmaps()

    def test_threshold_without_dilation(self, labelmap_tools: ProcessLabelmaps) -> None:
        """Every non-zero label becomes foreground; no dilation grows it."""
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 3  # non-zero label id
        labelmap = itk.image_from_array(arr)
        labelmap.SetSpacing([1.0, 1.0, 1.0])

        mask = labelmap_tools.convert_labelmap_to_mask(labelmap, dilation_in_mm=0.0)
        mask_arr = itk.array_from_image(mask)

        assert set(np.unique(mask_arr).tolist()) == {0, 1}
        assert int(mask_arr.sum()) == 1
        assert mask_arr[2, 2, 2] == 1

    def test_dilation_grows_mask(self, labelmap_tools: ProcessLabelmaps) -> None:
        """Positive dilation_in_mm grows the mask but keeps the seed voxel."""
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 3
        labelmap = itk.image_from_array(arr)
        # Unit isotropic spacing so dilation_in_mm == voxel radius.
        labelmap.SetSpacing([1.0, 1.0, 1.0])

        dilated = labelmap_tools.convert_labelmap_to_mask(labelmap, dilation_in_mm=1.0)
        dilated_arr = itk.array_from_image(dilated)

        assert int(dilated_arr.sum()) > 1
        assert dilated_arr[2, 2, 2] == 1

    def test_dilation_respects_anisotropic_spacing(
        self, labelmap_tools: ProcessLabelmaps
    ) -> None:
        """A 5 mm radius covers more voxels along the finely spaced axis."""
        arr = np.zeros((11, 11, 11), dtype=np.uint8)
        arr[5, 5, 5] = 1
        labelmap = itk.image_from_array(arr)
        # numpy axes are (Z, Y, X); ITK spacing is (X, Y, Z). Make X coarse
        # (5 mm/voxel -> 1-voxel radius) and Z fine (1 mm/voxel -> 5-voxel
        # radius) so the per-axis radius differs.
        labelmap.SetSpacing([5.0, 1.0, 1.0])

        dilated = itk.array_from_image(
            labelmap_tools.convert_labelmap_to_mask(labelmap, dilation_in_mm=5.0)
        )

        # Z axis (numpy axis 0) reaches 5 voxels out; X axis (numpy axis 2)
        # only 1 voxel out.
        assert dilated[0, 5, 5] == 1
        assert dilated[10, 5, 5] == 1
        assert dilated[5, 5, 0] == 0
        assert dilated[5, 5, 10] == 0

    def test_exclude_labels_removes_voxels(
        self, labelmap_tools: ProcessLabelmaps
    ) -> None:
        """Excluded labels become background before thresholding."""
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[1, 1, 1] = 2  # kept
        arr[3, 3, 3] = 7  # excluded
        labelmap = itk.image_from_array(arr)
        labelmap.SetSpacing([1.0, 1.0, 1.0])

        mask_arr = itk.array_from_image(
            labelmap_tools.convert_labelmap_to_mask(
                labelmap, dilation_in_mm=0.0, exclude_labels=[7]
            )
        )

        assert mask_arr[1, 1, 1] == 1
        assert mask_arr[3, 3, 3] == 0
        assert int(mask_arr.sum()) == 1

    def test_preserves_image_information(
        self, labelmap_tools: ProcessLabelmaps
    ) -> None:
        """Origin, spacing, and direction are copied from the labelmap."""
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[2, 2, 2] = 1
        labelmap = itk.image_from_array(arr)
        labelmap.SetSpacing([0.5, 1.0, 2.0])
        labelmap.SetOrigin([10.0, -5.0, 3.0])

        mask = labelmap_tools.convert_labelmap_to_mask(labelmap, dilation_in_mm=0.0)

        assert list(mask.GetSpacing()) == [0.5, 1.0, 2.0]
        assert list(mask.GetOrigin()) == [10.0, -5.0, 3.0]
        assert np.allclose(
            itk.array_from_matrix(mask.GetDirection()),
            itk.array_from_matrix(labelmap.GetDirection()),
        )


def _mask(arr: np.ndarray, spacing: tuple[float, float, float]) -> itk.Image:
    """Return *arr* as an image with *spacing*, a non-zero origin and direction."""
    image = itk.image_from_array(arr)
    image.SetSpacing(list(spacing))
    image.SetOrigin([10.0, -5.0, 3.0])
    image.SetDirection(itk.matrix_from_array(np.diag([1.0, -1.0, 1.0])))
    return image


class TestMaskCleanup:
    """Test suite for ProcessLabelmaps.remove_islands and fill_holes."""

    @pytest.fixture
    def labelmap_tools(self) -> ProcessLabelmaps:
        """Create ProcessLabelmaps instance."""
        return ProcessLabelmaps()

    @staticmethod
    def _islands() -> np.ndarray:
        """A 27-voxel cube, a 2-voxel island, and a voxel touching it at a corner."""
        arr = np.zeros((10, 10, 10), dtype=np.uint16)
        arr[1:4, 1:4, 1:4] = 7  # non-binary value, still foreground
        arr[4, 4, 4] = 7  # corner contact only: a separate component
        arr[8, 8, 7:9] = 7
        return arr

    def test_remove_islands_keeps_largest_component(
        self, labelmap_tools: ProcessLabelmaps
    ) -> None:
        """By default only the largest face-connected component survives."""
        kept = itk.array_from_image(
            labelmap_tools.remove_islands(_mask(self._islands(), (1.0, 1.0, 1.0)))
        )

        assert set(np.unique(kept).tolist()) == {0, 1}
        assert int(kept.sum()) == 27
        assert kept[2, 2, 2] == 1
        assert kept[4, 4, 4] == 0
        assert kept[8, 8, 7] == 0

    def test_remove_islands_volume_threshold_uses_spacing(
        self, labelmap_tools: ProcessLabelmaps
    ) -> None:
        """The threshold is a physical volume: 2 voxels of 6 mm^3 clear 10 mm^3."""
        mask = _mask(self._islands(), (1.0, 2.0, 3.0))

        kept = itk.array_from_image(
            labelmap_tools.remove_islands(mask, min_island_volume_mm3=10.0)
        )

        assert int(kept.sum()) == 27 + 2
        assert kept[4, 4, 4] == 0
        assert kept[8, 8, 7] == 1

    def test_fill_holes_fills_only_enclosed_cavities(
        self, labelmap_tools: ProcessLabelmaps
    ) -> None:
        """An enclosed cavity is filled; a channel open to the border is not."""
        arr = np.zeros((11, 11, 11), dtype=np.uint8)
        arr[1:10, 1:10, 1:10] = 1
        arr[2:4, 2:4, 2:4] = 0  # enclosed cavity
        arr[6:8, 6:8, 6:8] = 0  # cavity with a channel to the border along x
        arr[7, 7, 8:11] = 0
        filled = itk.array_from_image(
            labelmap_tools.fill_holes(_mask(arr, (1.0, 1.0, 1.0)))
        )

        assert np.all(filled[2:4, 2:4, 2:4] == 1)
        assert np.all(filled[6:8, 6:8, 6:8] == 0)
        assert np.all(filled[7, 7, 8:11] == 0)

    def test_fill_holes_fills_a_closed_cavity(
        self, labelmap_tools: ProcessLabelmaps
    ) -> None:
        """Every voxel of a fully enclosed cavity becomes foreground."""
        arr = np.zeros((7, 7, 7), dtype=np.uint8)
        arr[1:6, 1:6, 1:6] = 1
        arr[2:5, 2:5, 2:5] = 0
        filled = itk.array_from_image(
            labelmap_tools.fill_holes(_mask(arr, (1.0, 1.0, 1.0)))
        )

        assert int(filled.sum()) == 125

    @pytest.mark.parametrize("method", ["remove_islands", "fill_holes"])
    def test_cleanup_preserves_image_information(
        self, labelmap_tools: ProcessLabelmaps, method: str
    ) -> None:
        """Origin, spacing, and direction survive; the result is a uint8 mask."""
        mask = _mask(self._islands(), (0.5, 1.0, 2.0))

        cleaned = getattr(labelmap_tools, method)(mask)

        assert itk.array_from_image(cleaned).dtype == np.uint8
        assert list(cleaned.GetSpacing()) == [0.5, 1.0, 2.0]
        assert list(cleaned.GetOrigin()) == [10.0, -5.0, 3.0]
        assert np.allclose(
            itk.array_from_matrix(cleaned.GetDirection()),
            itk.array_from_matrix(mask.GetDirection()),
        )

    @pytest.mark.parametrize("method", ["remove_islands", "fill_holes"])
    def test_cleanup_of_empty_mask_is_empty(
        self, labelmap_tools: ProcessLabelmaps, method: str
    ) -> None:
        """An empty mask stays empty."""
        empty = _mask(np.zeros((4, 4, 4), dtype=np.uint8), (1.0, 1.0, 1.0))

        assert (
            int(itk.array_from_image(getattr(labelmap_tools, method)(empty)).sum()) == 0
        )
