"""Tests for the image-to-USD workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pytest
from pxr import Usd, UsdGeom

from monai_physio.register_images_base import RegisterImagesBase
from monai_physio.register_images_greedy import RegisterImagesGreedy
from monai_physio.register_images_icon import RegisterImagesICON
from monai_physio.segment_chest_total_segmentator_with_contrast import (
    SegmentChestTotalSegmentatorWithContrast,
)
from monai_physio.workflow_convert_image_to_usd import WorkflowConvertImageToUSD


def _small_image() -> itk.Image:
    """A tiny synthetic image, shape (X, Y, Z) = (3, 3, 3), LPS world frame."""
    return itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))


def test_default_segmentation_and_registration_methods(tmp_path: Path) -> None:
    """Omitting segmentation_method/registration_method defaults to
    SegmentChestTotalSegmentatorWithContrast (contrast_threshold=500) and
    RegisterImagesGreedy, matching this workflow's documented defaults."""
    reference_image = _small_image()
    workflow = WorkflowConvertImageToUSD(
        time_series_images=[reference_image],
        reference_image=reference_image,
        usd_project_name="patient",
        output_directory=str(tmp_path),
        log_level=logging.CRITICAL,
    )

    assert isinstance(workflow.segmenter, SegmentChestTotalSegmentatorWithContrast)
    assert workflow.segmenter.contrast_threshold == 500
    assert isinstance(workflow.registrar, RegisterImagesGreedy)


def test_segmentation_method_rejects_wrong_type(tmp_path: Path) -> None:
    """A non-SegmentAnatomyBase segmentation_method raises TypeError."""
    reference_image = _small_image()
    with pytest.raises(TypeError, match="segmentation_method must be"):
        WorkflowConvertImageToUSD(
            time_series_images=[reference_image],
            reference_image=reference_image,
            usd_project_name="patient",
            output_directory=str(tmp_path),
            segmentation_method="ChestTotalSegmentator",  # type: ignore[arg-type]
            log_level=logging.CRITICAL,
        )


def test_registration_method_rejects_wrong_type(tmp_path: Path) -> None:
    """A non-RegisterImagesBase registration_method raises TypeError."""
    reference_image = _small_image()
    with pytest.raises(TypeError, match="registration_method must be"):
        WorkflowConvertImageToUSD(
            time_series_images=[reference_image],
            reference_image=reference_image,
            usd_project_name="patient",
            output_directory=str(tmp_path),
            registration_method="ICON",  # type: ignore[arg-type]
            log_level=logging.CRITICAL,
        )


def test_caller_supplied_instances_are_used_as_is(tmp_path: Path) -> None:
    """A caller-supplied segmenter/registrar instance is stored unmodified
    (beyond the documented shared setters): the workflow must not apply its
    default-only contrast_threshold=500 tuning to a caller-supplied segmenter."""
    reference_image = _small_image()
    segmenter = SegmentChestTotalSegmentatorWithContrast()
    segmenter.contrast_threshold = 800
    original_contrast_threshold = segmenter.contrast_threshold
    registrar: RegisterImagesBase = RegisterImagesICON()

    workflow = WorkflowConvertImageToUSD(
        time_series_images=[reference_image],
        reference_image=reference_image,
        usd_project_name="patient",
        output_directory=str(tmp_path),
        segmentation_method=segmenter,
        registration_method=registrar,
        log_level=logging.CRITICAL,
    )

    assert workflow.segmenter is segmenter
    assert workflow.registrar is registrar
    assert workflow.segmenter.contrast_threshold == original_contrast_threshold


@pytest.mark.requires_gpu
@pytest.mark.slow
def test_workflow_convert_image_to_usd_default_operation(
    test_images: list[Any],
    tmp_path: Path,
) -> None:
    """Convert one real Slicer-Heart frame to USD.

    Input frame shape is the downloaded/resampled slicer_heart_small 3D image
    with axes (X, Y, Z) in LPS world frame.
    """
    reference_image = test_images[0]
    workflow = WorkflowConvertImageToUSD(
        time_series_images=[reference_image],
        reference_image=reference_image,
        usd_project_name="slicer_heart_small",
        output_directory=str(tmp_path),
        log_level=logging.CRITICAL,
    )

    assert isinstance(workflow.segmenter, SegmentChestTotalSegmentatorWithContrast)
    assert workflow.segmenter.contrast_threshold == 500
    assert isinstance(workflow.registrar, RegisterImagesGreedy)
    workflow.registrar.set_number_of_iterations([2])

    result_filenames = workflow.process()

    assert result_filenames == {"all": "slicer_heart_small.all_painted.usd"}
    assert workflow.reference_segmentation_results is not None
    assert "all" in workflow.reference_contours
    assert len(workflow.transformed_contours["all"]) == 1
    assert len(workflow.registration_results) == 1
    assert "all" in workflow.registration_results[0]
    assert workflow.registration_results[0]["all"]["forward_transform"] is not None
    assert workflow.registration_results[0]["all"]["inverse_transform"] is not None

    reference_labelmap = cast(
        itk.Image,
        workflow.reference_segmentation_results["labelmap"],
    )
    assert itk.size(reference_labelmap) == itk.size(reference_image)

    # No slice_NNN_labelmap.mha here: the workflow segments each moving frame
    # only when dynamic_labelmap_ids is non-empty, and the default operation
    # this test covers leaves it empty.
    expected_outputs = [
        "reference_labelmap.mha",
        "slicer_heart_small.all.usd",
        "slicer_heart_small.all_painted.usd",
    ]
    for output_name in expected_outputs:
        output_path = tmp_path / output_name
        assert output_path.exists(), f"Missing workflow output: {output_path}"
        assert output_path.stat().st_size > 0, f"Empty workflow output: {output_path}"

    stage = Usd.Stage.Open(str(tmp_path / "slicer_heart_small.all_painted.usd"))
    assert stage is not None
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    assert stage.GetPrimAtPath("/World").IsValid()
    assert stage.GetPrimAtPath("/World/slicer_heart_small").IsValid()
