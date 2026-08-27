"""
Test for chest CT segmentation using TotalSegmentator.

This test depends on test_convert_image_4d_to_3d and tests segmentation
functionality on two time points from the converted 3D data.
"""

from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest

from physiotwin4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiotwin4d.segment_chest_total_segmentator_with_contrast import (
    SegmentChestTotalSegmentatorWithContrast,
)


@pytest.mark.requires_gpu
@pytest.mark.slow
class TestSegmentChestTotalSegmentator:
    """Test suite for TotalSegmentator chest CT segmentation."""

    def test_segmenter_initialization(
        self, segmenter_total_segmentator: SegmentChestTotalSegmentator
    ) -> None:
        """Test that SegmentChestTotalSegmentator initializes correctly."""
        assert segmenter_total_segmentator is not None, "Segmenter not initialized"
        assert segmenter_total_segmentator.target_spacing == 1.0, (
            "Target spacing not set correctly"
        )

        # Check that anatomical structure ID mappings are defined via the
        # shared taxonomy.
        taxonomy = segmenter_total_segmentator.taxonomy
        assert len(taxonomy.labels_in_group("heart")) > 0, "Heart mask IDs not defined"
        assert len(taxonomy.labels_in_group("major_vessels")) > 0, (
            "Major vessels mask IDs not defined"
        )
        assert len(taxonomy.labels_in_group("lung")) > 0, "Lung mask IDs not defined"
        assert len(taxonomy.labels_in_group("bone")) > 0, "Bone mask IDs not defined"
        assert len(taxonomy.labels_in_group("soft_tissue")) > 0, (
            "Soft tissue mask IDs not defined"
        )

        print("\nSegmenter initialized with correct parameters")
        print(f"  Heart structures: {len(taxonomy.labels_in_group('heart'))}")
        print(f"  Major vessels: {len(taxonomy.labels_in_group('major_vessels'))}")
        print(f"  Lung structures: {len(taxonomy.labels_in_group('lung'))}")
        print(f"  Bone structures: {len(taxonomy.labels_in_group('bone'))}")
        print(
            f"  Soft tissue structures: {len(taxonomy.labels_in_group('soft_tissue'))}"
        )

    def test_segment_single_image(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test segmentation on a single time point."""
        output_dir = test_directories["output"]

        # Test on first time point only
        input_image = test_images[0]

        print("\nSegmenting time point 0...")
        print(f"  Input image size: {itk.size(input_image)}")

        # Run segmentation
        result = segmenter_total_segmentator.segment(input_image)

        # Verify result is a dictionary with expected keys
        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = [
            "labelmap",
            "lung",
            "heart",
            "major_vessels",
            "bone",
            "soft_tissue",
            "other",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in result"
            assert result[key] is not None, f"Result['{key}'] is None"

        # Verify labelmap properties
        labelmap = result["labelmap"]
        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"

        # Check that labels are present
        labelmap_arr = itk.array_from_image(labelmap)
        unique_labels = np.unique(labelmap_arr)
        assert len(unique_labels) > 1, "Labelmap should contain multiple labels"

        print("Segmentation complete for time point 0")
        print(f"  Labelmap size: {itk.size(labelmap)}")
        print(f"  Unique labels: {len(unique_labels)}")

        # Save results
        seg_output_dir = output_dir / "segmentation_total_segmentator"
        seg_output_dir.mkdir(exist_ok=True)

        itk.imwrite(
            labelmap, str(seg_output_dir / "slice_000_labelmap.mha"), compression=True
        )
        print(f"  Saved labelmap to: {seg_output_dir / 'slice_000_labelmap.mha'}")

    def test_segment_multiple_images(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test segmentation on two time points."""
        output_dir = test_directories["output"]
        seg_output_dir = output_dir / "segmentation_total_segmentator"
        seg_output_dir.mkdir(exist_ok=True)

        results = []
        for i, input_image in enumerate(test_images[0:2]):
            print(f"\nSegmenting time point {i}...")

            result = segmenter_total_segmentator.segment(input_image)
            results.append(result)

            # Save labelmap for each time point
            labelmap = result["labelmap"]
            output_file = seg_output_dir / f"slice_{i:03d}_labelmap.mha"
            itk.imwrite(labelmap, str(output_file), compression=True)

            print(f"Time point {i} complete")
            print(f"  Saved to: {output_file}")

        assert len(results) == 2, "Expected 2 segmentation results"
        print(f"\nSuccessfully segmented {len(results)} time points")

    def test_anatomy_group_labelmaps(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
    ) -> None:
        """Test that anatomy group labelmaps are created correctly."""
        input_image = test_images[0]

        # Run segmentation
        result = segmenter_total_segmentator.segment(input_image)

        # Check each anatomy group labelmap
        anatomy_groups = [
            "lung",
            "heart",
            "major_vessels",
            "bone",
            "soft_tissue",
            "other",
        ]
        taxonomy = segmenter_total_segmentator.taxonomy

        for group in anatomy_groups:
            group_labelmap = result[group]
            assert group_labelmap is not None, f"{group} labelmap is None"

            group_labelmap_arr = itk.array_from_image(group_labelmap)
            unique_values = set(np.unique(group_labelmap_arr).tolist())
            assert 0 in unique_values, f"{group} labelmap should contain background"

            # "other" collects whatever ids no group claimed, so it has no
            # fixed id set to check against.
            if group != "other":
                allowed_values = {0} | set(taxonomy.labels_in_group(group).keys())
                assert unique_values <= allowed_values, (
                    f"{group} labelmap contains unexpected label ids: "
                    f"{unique_values - allowed_values}"
                )

            # Check that labelmap has same size as input
            assert itk.size(group_labelmap) == itk.size(input_image), (
                f"{group} labelmap size mismatch"
            )

        print("\nAll anatomy group labelmaps created correctly")
        for group in anatomy_groups:
            group_labelmap_arr = itk.array_from_image(result[group])
            num_voxels = np.sum(group_labelmap_arr > 0)
            print(f"  {group}: {num_voxels} voxels")

    def test_contrast_detection(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        segmenter_total_segmentator_with_contrast: (
            SegmentChestTotalSegmentatorWithContrast
        ),
        test_images: list[Any],
    ) -> None:
        """Test contrast detection functionality."""
        input_image = test_images[0]

        # Plain segmenter has no "contrast" group
        result_no_contrast = segmenter_total_segmentator.segment(input_image)
        assert "contrast" not in result_no_contrast

        # Contrast-enhanced segmenter adds a "contrast" group
        result_with_contrast = segmenter_total_segmentator_with_contrast.segment(
            input_image
        )
        contrast_mask_yes = result_with_contrast["contrast"]
        assert contrast_mask_yes is not None, "Contrast mask (with flag) is None"

        print("\nContrast detection tested")

        contrast_arr_yes = itk.array_from_image(contrast_mask_yes)
        print(f"  Contrast voxels: {np.sum(contrast_arr_yes > 0)}")

    def test_preprocessing(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
    ) -> None:
        """Test preprocessing functionality."""
        input_image = test_images[0]

        # Get original properties
        original_spacing = itk.spacing(input_image)

        # Preprocessing is done internally by segment(), not exposed as public method
        # Just verify that segment() works (which includes preprocessing)
        result = segmenter_total_segmentator.segment(input_image)

        # Check that segmentation was successful (which means preprocessing worked)
        assert result is not None, "Segmentation result is None"
        assert "labelmap" in result, "Labelmap not in result"

        print("\nPreprocessing tested (via successful segmentation)")
        print(f"  Original image spacing: {original_spacing}")

    def test_postprocessing(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
    ) -> None:
        """Test postprocessing functionality."""
        input_image = test_images[0]

        # Run full segmentation to get labelmap
        result = segmenter_total_segmentator.segment(input_image)
        labelmap = result["labelmap"]

        # Postprocessing is part of segment(), verify output is properly sized
        assert itk.size(labelmap) == itk.size(input_image), (
            "Postprocessing failed: size mismatch"
        )

        # Check that labelmap has been resampled to original spacing
        original_spacing = itk.spacing(input_image)
        labelmap_spacing = itk.spacing(labelmap)

        # Spacing should match (within floating point tolerance)
        for i in range(3):
            assert abs(labelmap_spacing[i] - original_spacing[i]) < 0.01, (
                f"Spacing mismatch at dimension {i}"
            )

        print("\nPostprocessing tested")
        print(f"  Original spacing: {original_spacing}")
        print(f"  Labelmap spacing: {labelmap_spacing}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


def test_academic_license_request_is_honoured_when_a_license_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a license present, the licensed tasks stay requested."""
    # staticmethod(), because setattr on the class would otherwise install a
    # plain function and the call would pass self into it.
    monkeypatch.setattr(
        SegmentChestTotalSegmentator,
        "_academic_license_is_valid",
        staticmethod(lambda: True),
    )
    segmenter = SegmentChestTotalSegmentator()

    segmenter.set_has_academic_license(True)

    assert segmenter.has_academic_license is True


def test_academic_license_request_falls_back_when_no_license_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a license, asking for the licensed tasks must not abort the run.

    ``heartchambers_highres`` and ``tissue_4_types`` are not openly available.
    Requesting them unlicensed makes ``totalsegmentator`` call ``sys.exit(1)``
    from inside the segmentation, which surfaces as a bare ``SystemExit``
    partway through whatever workflow was running.  The request is dropped
    here instead, so the heart is segmented as one structure.
    """
    monkeypatch.setattr(
        SegmentChestTotalSegmentator,
        "_academic_license_is_valid",
        staticmethod(lambda: False),
    )
    segmenter = SegmentChestTotalSegmentator()

    segmenter.set_has_academic_license(True)

    assert segmenter.has_academic_license is False


def test_declining_the_academic_license_never_checks_for_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a request for the licensed tasks should cost a license lookup."""

    def _fail() -> bool:
        raise AssertionError("license checked when it was not requested")

    monkeypatch.setattr(
        SegmentChestTotalSegmentator, "_academic_license_is_valid", staticmethod(_fail)
    )
    segmenter = SegmentChestTotalSegmentator()

    segmenter.set_has_academic_license(False)

    assert segmenter.has_academic_license is False


def test_license_check_tracks_totalsegmentators_own_offline_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check must agree with the gate it is predicting, weakness included.

    ``show_license_info`` exits unless ``has_valid_license_offline`` returns
    "yes", and that only tests for a configured 18-character key.  A key of the
    right length but no longer entitled therefore reads as installed here, and
    TotalSegmentator fails later, while downloading the licensed weights.
    Asking the backend instead would report an offline runner as unlicensed and
    silently change the anatomy it produces, so the weaker check is the
    deliberate choice and this pins it.
    """
    import totalsegmentator.libs as ts_libs

    monkeypatch.setattr(
        ts_libs,
        "has_valid_license_offline",
        lambda: ("yes", "SUCCESS: License is valid."),
    )
    assert SegmentChestTotalSegmentator._academic_license_is_valid() is True

    monkeypatch.setattr(
        ts_libs,
        "has_valid_license_offline",
        lambda: ("invalid_license", "ERROR: Invalid license number (too-short)."),
    )
    assert SegmentChestTotalSegmentator._academic_license_is_valid() is False

    monkeypatch.setattr(
        ts_libs,
        "has_valid_license_offline",
        lambda: ("missing_license", "ERROR: A license number has not been set."),
    )
    assert SegmentChestTotalSegmentator._academic_license_is_valid() is False
