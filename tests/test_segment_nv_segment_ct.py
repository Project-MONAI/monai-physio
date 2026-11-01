"""
Tests for CT segmentation using NVIDIA NV-Segment-CT.

The taxonomy and configuration tests run by default: constructing
SegmentNVSegmentCT downloads nothing. The end-to-end test is gated behind
--run-gpu and --run-slow because it pulls the model weights and needs a CUDA
device.
"""

import logging
from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest

from monai_physio.segment_nv_segment_ct import SegmentNVSegmentCT

#: Anatomy groups this segmenter must populate, besides the inherited "other".
EXPECTED_GROUPS = (
    "heart",
    "major_vessels",
    "lung",
    "bone",
    "soft_tissue",
)


class RecordCollector(logging.Handler):
    """Collect emitted records.

    The shared MONAI Physio logger sets ``propagate = False``, so pytest's
    ``caplog`` (which handles records at the root logger) never sees them.
    Attaching this handler to the logger itself is the way to assert on
    MONAI Physio log output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestSegmentNVSegmentCTConfiguration:
    """Configuration and taxonomy tests that need neither GPU nor network."""

    def test_segmenter_initialization(
        self, segmenter_nv_segment_ct: SegmentNVSegmentCT
    ) -> None:
        """Test that SegmentNVSegmentCT initializes correctly."""
        segmenter = segmenter_nv_segment_ct
        assert segmenter.target_spacing == 1.5, "Target spacing not set correctly"
        assert segmenter.labelmap_dtype is np.uint8, "Class index space fits in uint8"

        taxonomy = segmenter.taxonomy
        for group in EXPECTED_GROUPS:
            assert len(taxonomy.labels_in_group(group)) > 0, (
                f"{group} mask IDs not defined"
            )

        print("\nSegmenter initialized with correct parameters")
        for group in EXPECTED_GROUPS:
            print(f"  {group}: {len(taxonomy.labels_in_group(group))} structures")

    def test_native_label_ids(
        self, segmenter_nv_segment_ct: SegmentNVSegmentCT
    ) -> None:
        """Labelmap ids must be the model's published indices, unmapped."""
        labels = segmenter_nv_segment_ct.taxonomy.all_labels()

        assert labels[6] == "aorta"
        assert labels[115] == "heart"
        assert labels[132] == "airway"

        # Ids omitted from the model's label_dict.json are never emitted, so
        # they must fall through to the "other" group rather than claim an
        # organ name.
        for unused_id in (2, 16, 18, 20, 21, 129, 131):
            assert (
                segmenter_nv_segment_ct.taxonomy.group_for_id(unused_id) == "other"
            ), f"Unused id {unused_id} should be unclaimed"

    def test_other_group_covers_full_index_space(
        self, segmenter_nv_segment_ct: SegmentNVSegmentCT
    ) -> None:
        """_finalize_other_group must sweep the model's full index space."""
        labels = segmenter_nv_segment_ct.taxonomy.all_labels()
        assert set(labels.keys()) == set(range(1, 133)), (
            "Taxonomy should cover the model's full [1, 133) index space"
        )

    def test_license_warning_logged_on_first_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The research-only license must be surfaced before weights load."""
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", lambda **kwargs: "/nonexistent"
        )

        segmenter = SegmentNVSegmentCT()
        collector = RecordCollector()
        segmenter.logger.addHandler(collector)
        try:
            segmenter._ensure_model()
            segmenter._ensure_model()
        finally:
            segmenter.logger.removeHandler(collector)

        warnings = [r for r in collector.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "License warning should fire once per instance"
        assert segmenter.license_warning in warnings[0].getMessage()


@pytest.mark.requires_gpu
@pytest.mark.slow
class TestSegmentNVSegmentCT:
    """End-to-end test; downloads the model weights on first run."""

    def test_segment_single_image(
        self,
        segmenter_nv_segment_ct: SegmentNVSegmentCT,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test segmentation on a single time point."""
        output_dir = test_directories["output"]
        input_image = test_images[0]

        print("\nSegmenting time point 0...")
        print(f"  Input image size: {itk.size(input_image)}")

        result = segmenter_nv_segment_ct.segment(input_image)

        assert isinstance(result, dict), "Result should be a dictionary"
        for key in ["labelmap", *EXPECTED_GROUPS, "other"]:
            assert key in result, f"Missing key '{key}' in result"
            assert result[key] is not None, f"Result['{key}'] is None"

        labelmap = result["labelmap"]
        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"

        labelmap_arr = itk.array_from_image(labelmap)
        assert labelmap_arr.dtype == np.uint8, "Labelmap should be uint8"

        unique_labels = np.unique(labelmap_arr)
        assert len(unique_labels) > 1, "Labelmap should contain multiple labels"

        allowed = {0} | set(segmenter_nv_segment_ct.taxonomy.all_labels().keys())
        assert set(unique_labels.tolist()) <= allowed, (
            "Labelmap contains ids outside the model's index space"
        )

        heart_ids = set(segmenter_nv_segment_ct.taxonomy.labels_in_group("heart"))
        assert heart_ids & set(unique_labels.tolist()), (
            "Cardiac CT should yield at least one heart label"
        )

        print("Segmentation complete for time point 0")
        print(f"  Unique labels: {len(unique_labels)}")

        seg_output_dir = output_dir / "segmentation_nv_segment_ct"
        seg_output_dir.mkdir(exist_ok=True)
        itk.imwrite(
            labelmap, str(seg_output_dir / "slice_000_labelmap.mha"), compression=True
        )
        print(f"  Saved labelmap to: {seg_output_dir / 'slice_000_labelmap.mha'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
