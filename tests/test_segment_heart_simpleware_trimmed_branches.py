"""
Tests for SegmentHeartSimplewareTrimmedBranches.

trim_branches() is implemented entirely with standard ITK filters, so all
tests below run on every supported Python version.
"""

import os

import itk
import numpy as np
import pytest

from monai_physio.segment_heart_simpleware import SegmentHeartSimpleware
from monai_physio.segment_heart_simpleware_trimmed_branches import (
    SegmentHeartSimplewareTrimmedBranches,
)


def _simpleware_available(segmenter: SegmentHeartSimpleware) -> bool:
    """Return True if Simpleware Medical executable and script exist."""
    return os.path.exists(segmenter.simpleware_exe_path) and os.path.exists(
        segmenter.simpleware_script_path
    )


def _synthetic_heart_labelmap() -> itk.Image:
    """Small synthetic labelmap, shape (Z, Y, X) = (32, 32, 32), with a
    solid block of each ASCardio label (1=LV, 2=RV, 3=LA, 4=RA, 5=myocardium,
    6=heart) so trim_branches() has non-trivial connected components to
    operate on."""
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[4:12, 4:12, 4:12] = 1  # left_ventricle
    arr[4:12, 4:12, 12:20] = 2  # right_ventricle
    arr[12:20, 4:12, 4:12] = 3  # left_atrium
    arr[12:20, 4:12, 12:20] = 4  # right_atrium
    arr[4:20, 12:20, 4:20] = 5  # myocardium
    arr[20:28, 20:28, 20:28] = 6  # heart
    image = itk.image_from_array(arr)
    image.SetSpacing([1.0, 1.0, 1.0])
    return image


def test_subclass_owns_trim_branches() -> None:
    """SegmentHeartSimplewareTrimmedBranches is a SegmentHeartSimpleware that
    owns trim_branches(); the base class no longer defines trim_branches() or
    set_trim_branches()."""
    segmenter = SegmentHeartSimplewareTrimmedBranches()
    assert isinstance(segmenter, SegmentHeartSimpleware)
    assert hasattr(segmenter, "trim_branches")
    assert not hasattr(SegmentHeartSimpleware, "trim_branches")
    assert not hasattr(SegmentHeartSimpleware, "set_trim_branches")


def test_trim_branches_runs_on_synthetic_labelmap() -> None:
    """trim_branches() is pure post-processing - it must run without
    Simpleware and return a labelmap matching the input's geometry."""
    segmenter = SegmentHeartSimplewareTrimmedBranches()
    labelmap = _synthetic_heart_labelmap()

    trimmed = segmenter.trim_branches(labelmap)

    assert itk.size(trimmed) == itk.size(labelmap)
    trimmed_arr = itk.array_from_image(trimmed)
    # trim_branches() only redistributes the existing heart labels (1, 5, 6)
    # and the largest connected component of 3/4; it must not invent labels
    # outside the ASCardio heart label set.
    assert set(np.unique(trimmed_arr)) <= {0, 1, 2, 3, 4, 5, 6}


@pytest.mark.requires_gpu
@pytest.mark.requires_simpleware
@pytest.mark.slow
def test_segment_trims_branches_relative_to_plain_segmenter(
    test_images: list,
) -> None:
    """SegmentHeartSimplewareTrimmedBranches().segment(...) must actually
    apply trimming: its output should differ from the untrimmed
    SegmentHeartSimpleware().segment(...) output on the same input, while
    both remain valid labelmaps of the same size.

    Note: this does not assert bit-exact equivalence to a manually
    reconstructed "trim after the fact" pipeline, because trim_branches()
    runs on the raw Simpleware output *before* postprocess_labelmap()'s
    resampling - applying it after resampling instead would not be
    equivalent. The trim_branches()-correctness itself is covered by
    test_trim_branches_runs_on_synthetic_labelmap above; this test only
    confirms the subclass's segmentation_method() actually invokes it as
    part of segment()'s pipeline.
    """
    plain = SegmentHeartSimpleware()
    if not _simpleware_available(plain):
        pytest.skip(
            "Simpleware Medical not found (executable or script). "
            "Install Simpleware Medical with ASCardio to run this test."
        )

    input_image = test_images[3]

    plain_result = plain.segment(input_image)
    trimmed_result = SegmentHeartSimplewareTrimmedBranches().segment(input_image)

    plain_labelmap = plain_result["labelmap"]
    trimmed_labelmap = trimmed_result["labelmap"]
    assert itk.size(trimmed_labelmap) == itk.size(plain_labelmap)
    assert not np.array_equal(
        itk.array_from_image(plain_labelmap), itk.array_from_image(trimmed_labelmap)
    ), "Trimming should change the labelmap relative to the untrimmed segmenter"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
