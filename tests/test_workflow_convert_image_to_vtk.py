"""Tests for the image-to-VTK workflow's instance-based segmentation_method API."""

from __future__ import annotations

from typing import Any

import itk
import numpy as np
import pytest

from monai_physio.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from monai_physio.segment_heart_simpleware import SegmentHeartSimpleware
from monai_physio.workflow_convert_image_to_vtk import WorkflowConvertImageToVTK


def test_default_segmentation_method_is_chest_total_segmentator() -> None:
    """Omitting segmentation_method defaults to SegmentChestTotalSegmentator,
    matching this workflow's historical string default."""
    workflow = WorkflowConvertImageToVTK()
    assert isinstance(workflow._segmenter, SegmentChestTotalSegmentator)


def test_segmentation_method_rejects_wrong_type() -> None:
    """A non-SegmentAnatomyBase segmentation_method raises TypeError."""
    invalid_method: Any = "ChestTotalSegmentator"
    with pytest.raises(TypeError, match="segmentation_method must be"):
        WorkflowConvertImageToVTK(segmentation_method=invalid_method)


def test_caller_supplied_instance_is_used_as_is() -> None:
    """A caller-supplied segmenter instance is stored unmodified."""
    segmenter = SegmentHeartSimpleware()
    workflow = WorkflowConvertImageToVTK(segmentation_method=segmenter)
    assert workflow._segmenter is segmenter


def test_extract_label_surface_isolates_single_label() -> None:
    """_extract_label_surface must isolate one label id from a multi-value labelmap.

    Regression coverage for the extract_label_surfaces=True path added to
    process(): each label's surface should come from only that label's
    voxels, not the whole labelmap, and an absent label id should yield
    None (mirrors the empty-mask skip used for whole-group surfaces).
    """
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 1
    arr[6:9, 6:9, 6:9] = 2
    labelmap = itk.image_from_array(arr)

    workflow = WorkflowConvertImageToVTK()

    surface_1 = workflow._extract_label_surface(labelmap, arr, 1)
    surface_2 = workflow._extract_label_surface(labelmap, arr, 2)
    surface_absent = workflow._extract_label_surface(labelmap, arr, 99)

    assert surface_1 is not None and surface_1.n_points > 0
    assert surface_2 is not None and surface_2.n_points > 0
    assert surface_absent is None
