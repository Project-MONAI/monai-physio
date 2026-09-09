"""
Tests for LabelmapTools functionality.

Covers thresholding a multi-label labelmap into a binary registration mask,
physically isotropic dilation that respects per-axis spacing, and forcing
selected labels to background via ``exclude_labels``.
"""

from __future__ import annotations

import itk
import numpy as np
import pytest

from monai_physio.labelmap_tools import LabelmapTools


class TestLabelmapTools:
    """Test suite for LabelmapTools.convert_labelmap_to_mask."""

    @pytest.fixture
    def labelmap_tools(self) -> LabelmapTools:
        """Create LabelmapTools instance."""
        return LabelmapTools()

    def test_threshold_without_dilation(self, labelmap_tools: LabelmapTools) -> None:
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

    def test_dilation_grows_mask(self, labelmap_tools: LabelmapTools) -> None:
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
        self, labelmap_tools: LabelmapTools
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

    def test_exclude_labels_removes_voxels(self, labelmap_tools: LabelmapTools) -> None:
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

    def test_preserves_image_information(self, labelmap_tools: LabelmapTools) -> None:
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
