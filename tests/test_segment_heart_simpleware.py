"""
Tests for heart CT segmentation using SegmentHeartSimpleware (Simpleware Medical ASCardio).

Uses the same input data as experiments/Heart-Simpleware_Segmentation:
  data/CHOP-Valve4D/CT/RVOT28-Dias.nii.gz

Requires Synopsys Simpleware Medical with ASCardio and the test data to run
full segmentation tests. Initialization and path tests run without Simpleware.
"""

import os
from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest

from physiotwin4d.segment_heart_simpleware import SegmentHeartSimpleware


def _simpleware_available(segmenter: SegmentHeartSimpleware) -> bool:
    """Return True if Simpleware Medical executable and script exist."""
    return os.path.exists(segmenter.simpleware_exe_path) and os.path.exists(
        segmenter.simpleware_script_path
    )


@pytest.mark.requires_gpu
@pytest.mark.requires_simpleware
@pytest.mark.slow
class TestSegmentHeartSimpleware:
    """Test suite for SegmentHeartSimpleware (Simpleware Medical ASCardio)."""

    def test_segmenter_initialization(
        self, segmenter_simpleware: SegmentHeartSimpleware
    ) -> None:
        """Test that SegmentHeartSimpleware initializes correctly."""
        seg = segmenter_simpleware
        assert seg is not None, "Segmenter not initialized"
        assert seg.target_spacing == 1.0, (
            "Target spacing should be 1.0 mm for Simpleware"
        )

        taxonomy = seg.taxonomy
        assert len(taxonomy.labels_in_group("heart")) > 0, "Heart mask IDs not defined"
        assert len(taxonomy.labels_in_group("major_vessels")) > 0, (
            "Major vessels mask IDs not defined"
        )
        # ASCardio segments neither lung, bone, soft tissue nor contrast, and
        # SegmentAnatomyBase seeds no defaults, so labels_in_group returns an
        # empty dict for each of them.
        assert taxonomy.labels_in_group("lung") == {}, "ASCardio does not segment lungs"
        assert taxonomy.labels_in_group("bone") == {}, "ASCardio does not segment bone"
        assert taxonomy.labels_in_group("soft_tissue") == {}, (
            "ASCardio does not segment soft tissue"
        )
        assert taxonomy.labels_in_group("contrast") == {}, (
            "ASCardio does not detect contrast"
        )

        assert seg.simpleware_exe_path is not None, "Simpleware executable path not set"
        assert seg.simpleware_script_path is not None, "Simpleware script path not set"
        assert "SimplewareScript_heart_segmentation" in seg.simpleware_script_path

        print("\nSegmenter initialized with correct parameters")
        print(f"  Target spacing: {seg.target_spacing} mm")
        print(f"  Heart structures: {len(taxonomy.labels_in_group('heart'))}")
        print(f"  Major vessels: {len(taxonomy.labels_in_group('major_vessels'))}")

    def test_set_simpleware_executable_path(
        self, segmenter_simpleware: SegmentHeartSimpleware
    ) -> None:
        """Test setting custom Simpleware executable path."""
        seg = segmenter_simpleware
        original = seg.simpleware_exe_path
        custom = "D:/Custom/ConsoleSimplewareMedical.exe"
        seg.set_simpleware_executable_path(custom)
        assert seg.simpleware_exe_path == custom
        seg.set_simpleware_executable_path(original)
        assert seg.simpleware_exe_path == original
        print("\nset_simpleware_executable_path works correctly")

    def test_segment_single_image(
        self,
        segmenter_simpleware: SegmentHeartSimpleware,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test segmentation on a cardiac CT time point."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip(
                "Simpleware Medical not found (executable or script). "
                "Install Simpleware Medical with ASCardio to run this test."
            )

        output_dir = test_directories["output"]
        input_image = test_images[3]

        print("\nSegmenting cardiac CT...")
        print(f"  Image size: {itk.size(input_image)}")

        result = segmenter_simpleware.segment(input_image)

        assert isinstance(result, dict), "Result should be a dictionary"
        # The Simpleware segmenter only registers the groups it actually
        # populates: heart and major_vessels from the subclass, plus the
        # "other" group that collects every unclaimed id. lung, bone,
        # soft_tissue and contrast are NOT in the result, because ASCardio
        # does not produce them and SegmentAnatomyBase seeds no placeholder
        # for them; callers that need those groups must check membership.
        expected_keys = [
            "labelmap",
            "heart",
            "major_vessels",
            "other",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in result"
            assert result[key] is not None, f"Result['{key}'] is None"
        for absent in ("lung", "bone", "soft_tissue", "contrast"):
            assert absent not in result, (
                f"ASCardio does not produce {absent}; key must be absent"
            )

        labelmap = result["labelmap"]
        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"

        labelmap_arr = itk.array_from_image(labelmap)
        unique_labels = np.unique(labelmap_arr)
        assert len(unique_labels) > 1, "Labelmap should contain multiple labels"

        print("Segmentation complete")
        print(f"  Unique labels: {len(unique_labels)}")

        seg_output_dir = output_dir / "segmentation_simpleware"
        seg_output_dir.mkdir(exist_ok=True)
        itk.imwrite(
            labelmap,
            str(seg_output_dir / "heart_labelmap_simpleware.nii.gz"),
            compression=True,
        )
        print(f"  Saved to: {seg_output_dir / 'heart_labelmap_simpleware.nii.gz'}")

    def test_anatomy_group_labelmaps(
        self,
        segmenter_simpleware: SegmentHeartSimpleware,
        test_images: list[Any],
    ) -> None:
        """Test that anatomy group labelmaps are created (heart, vessels, etc.).

        A group entry is a labelmap carrying that group's own label ids, not a
        binary mask, so the ids are checked against the taxonomy rather than
        the value count.
        """
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip("Simpleware Medical not found. Install to run this test.")

        input_image = test_images[3]
        result = segmenter_simpleware.segment(input_image)

        # Only assert on groups Simpleware/ASCardio actually populates.
        anatomy_groups = [
            "heart",
            "major_vessels",
            "other",
        ]
        taxonomy = segmenter_simpleware.taxonomy

        for group in anatomy_groups:
            assert group in result, f"{group} labelmap should be present"
            group_labelmap = result[group]
            assert group_labelmap is not None, f"{group} labelmap is None"

            group_labelmap_arr = itk.array_from_image(group_labelmap)
            unique_values = set(np.unique(group_labelmap_arr).tolist())
            assert 0 in unique_values, f"{group} labelmap should contain background"

            # "other" is checked too: _finalize_other_group claims every id
            # no other group took, so it has just as fixed an id set.
            allowed_values = {0} | set(taxonomy.labels_in_group(group).keys())
            assert unique_values <= allowed_values, (
                f"{group} labelmap contains unexpected label ids: "
                f"{unique_values - allowed_values}"
            )

            assert itk.size(group_labelmap) == itk.size(input_image), (
                f"{group} labelmap size mismatch"
            )

        heart_arr = itk.array_from_image(result["heart"])
        vessels_arr = itk.array_from_image(result["major_vessels"])
        print("ANATOMY GROUP LABELMAPS")
        print(f"  heart: {np.sum(heart_arr > 0)} voxels")
        print(f"  major_vessels: {np.sum(vessels_arr > 0)} voxels")

    def test_contrast_group_is_absent(
        self,
        segmenter_simpleware: SegmentHeartSimpleware,
        test_images: list[Any],
    ) -> None:
        """ASCardio does not detect contrast, so no contrast group is reported.

        SegmentAnatomyBase used to seed a contrast placeholder that this
        segmenter inherited; it seeds nothing now, so callers must check for
        the key instead of indexing it.
        """
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip("Simpleware Medical not found. Install to run this test.")

        input_image = test_images[3]
        result = segmenter_simpleware.segment(input_image)
        assert "contrast" not in result

    def test_postprocessing(
        self,
        segmenter_simpleware: SegmentHeartSimpleware,
        test_images: list[Any],
    ) -> None:
        """Test that output labelmap matches input size and spacing."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip("Simpleware Medical not found. Install to run this test.")

        input_image = test_images[3]
        result = segmenter_simpleware.segment(input_image)
        labelmap = result["labelmap"]

        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"
        original_spacing = itk.spacing(input_image)
        labelmap_spacing = itk.spacing(labelmap)
        for i in range(3):
            assert abs(labelmap_spacing[i] - original_spacing[i]) < 0.01, (
                f"Spacing mismatch at dimension {i}"
            )
        print("\nPostprocessing: labelmap size and spacing match input")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
