"""Tests for the image-to-VTK workflow's instance-based segmentation_method API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest
import pyvista as pv

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


def _heart_labelmap() -> itk.Image:
    """Myocardium (5) wrapping a left ventricle (1), next to an aorta (7)."""
    arr = np.zeros((20, 24, 32), dtype=np.uint8)
    arr[4:16, 4:20, 4:18] = 5
    arr[7:13, 8:16, 8:14] = 1
    arr[6:14, 8:16, 18:28] = 7
    return itk.GetImageFromArray(arr)


@pytest.fixture(scope="module")
def heart_workflow() -> WorkflowConvertImageToVTK:
    return WorkflowConvertImageToVTK(segmentation_method=SegmentHeartSimpleware())


@pytest.mark.requires_gpu
def test_labelmap_gives_annotated_group_and_label_surfaces(
    heart_workflow: WorkflowConvertImageToVTK,
) -> None:
    result = heart_workflow.process(
        labelmap_image=_heart_labelmap(), extract_label_surfaces=True
    )

    assert sorted(result["surfaces"]) == ["heart", "major_vessels"]
    assert sorted(result["label_surfaces"]) == ["aorta", "left_ventricle", "myocardium"]
    heart = result["surfaces"]["heart"]
    assert heart.field_data["AnatomyGroup"].tolist() == ["heart"]
    assert sorted(heart.field_data["SegmentationLabelIds"].tolist()) == [1, 5]
    myocardium = result["label_surfaces"]["myocardium"]
    assert myocardium.field_data["SegmentationLabelIds"].tolist() == [5]
    assert myocardium.cell_data["Color"].shape == (myocardium.n_cells, 4)
    # Myocardium has its own color; the ventricle falls back to its group's.
    assert not np.array_equal(
        myocardium.field_data["AnatomyColor"], heart.field_data["AnatomyColor"]
    )
    assert np.array_equal(
        result["label_surfaces"]["left_ventricle"].field_data["AnatomyColor"],
        heart.field_data["AnatomyColor"],
    )


@pytest.mark.requires_gpu
def test_custom_groups_and_tetrahedra(
    heart_workflow: WorkflowConvertImageToVTK,
) -> None:
    """A group of chosen labels gets one surface and its tetrahedral mesh."""
    result = heart_workflow.process(
        labelmap_image=_heart_labelmap(),
        groups={"heart_minus_interior_chambers": [5]},
        tetrahedra_element_size_mm=1.0,
    )

    assert list(result["surfaces"]) == ["heart_minus_interior_chambers"]
    assert list(result["tetrahedra"]) == ["heart_minus_interior_chambers"]
    mesh = result["tetrahedra"]["heart_minus_interior_chambers"]
    assert mesh.n_cells > 0
    assert mesh.field_data["SegmentationLabelIds"].tolist() == [5]
    assert result["label_surfaces"] == {}


def test_process_needs_exactly_one_input(
    heart_workflow: WorkflowConvertImageToVTK,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        heart_workflow.process()
    with pytest.raises(ValueError, match="exactly one"):
        heart_workflow.process(
            input_image=_heart_labelmap(), labelmap_image=_heart_labelmap()
        )


def test_groups_must_not_share_a_label(
    heart_workflow: WorkflowConvertImageToVTK,
) -> None:
    with pytest.raises(ValueError, match="must not share"):
        heart_workflow.process(
            labelmap_image=_heart_labelmap(), groups={"a": [1, 5], "b": [5]}
        )


@pytest.mark.requires_gpu
def test_save_surfaces_and_combined(
    heart_workflow: WorkflowConvertImageToVTK, tmp_path: Path
) -> None:
    result = heart_workflow.process(
        labelmap_image=_heart_labelmap(), extract_label_surfaces=True
    )

    saved = WorkflowConvertImageToVTK.save_surfaces(
        result["surfaces"], str(tmp_path), prefix="p"
    )
    combined_file = WorkflowConvertImageToVTK.save_combined_surfaces(
        result["label_surfaces"], str(tmp_path / "sub" / "p_surfaces.vtp")
    )

    assert sorted(Path(path).name for path in saved.values()) == [
        "p_heart.vtp",
        "p_major_vessels.vtp",
    ]
    combined = pv.read(combined_file)
    assert sorted(np.unique(combined.cell_data["SegmentationLabelIds"])) == [1, 5, 7]
