"""
Tests for CT/MRI segmentation using NVIDIA NV-Segment-CTMR.

The taxonomy and configuration tests run by default: constructing
SegmentNVSegmentCTMRI downloads nothing. The end-to-end test is gated behind
--run-gpu and --run-slow because it pulls ~872 MB of model weights and needs a
CUDA device.
"""

import logging
from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest

from monai_physio.segment_nv_segment_ct_mri import SegmentNVSegmentCTMRI

#: Anatomy groups this segmenter must populate, besides the inherited "other".
EXPECTED_GROUPS = (
    "heart",
    "major_vessels",
    "lung",
    "bone",
    "soft_tissue",
    "brain_parcellation",
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


class TestSegmentNVSegmentCTMRIConfiguration:
    """Configuration and taxonomy tests that need neither GPU nor network."""

    def test_segmenter_initialization(
        self, segmenter_nv_segment_ct_mri: SegmentNVSegmentCTMRI
    ) -> None:
        """Test that SegmentNVSegmentCTMRI initializes correctly."""
        segmenter = segmenter_nv_segment_ct_mri
        assert segmenter.target_spacing == 1.5, "Target spacing not set correctly"
        assert segmenter.labelmap_dtype is np.uint16, (
            "Class index space exceeds 255 and requires uint16"
        )
        assert segmenter.modality == "CT_BODY", "Default modality not set correctly"

        taxonomy = segmenter.taxonomy
        for group in EXPECTED_GROUPS:
            assert len(taxonomy.labels_in_group(group)) > 0, (
                f"{group} mask IDs not defined"
            )

        print("\nSegmenter initialized with correct parameters")
        for group in EXPECTED_GROUPS:
            print(f"  {group}: {len(taxonomy.labels_in_group(group))} structures")

    def test_native_label_ids(
        self, segmenter_nv_segment_ct_mri: SegmentNVSegmentCTMRI
    ) -> None:
        """Labelmap ids must be the model's published indices, unmapped."""
        labels = segmenter_nv_segment_ct_mri.taxonomy.all_labels()

        assert labels[6] == "aorta"
        assert labels[115] == "heart"
        assert labels[345] == "ttg_transverse_temporal_gyrus_left"

        # Ids the model marks "(deprecated)" are never emitted, so they must
        # fall through to the "other" group rather than claim an organ name.
        for deprecated_id in (16, 131, 155, 162):
            assert (
                segmenter_nv_segment_ct_mri.taxonomy.group_for_id(deprecated_id)
                == "other"
            ), f"Deprecated id {deprecated_id} should be unclaimed"

    def test_other_group_covers_full_index_space(
        self, segmenter_nv_segment_ct_mri: SegmentNVSegmentCTMRI
    ) -> None:
        """_finalize_other_group must sweep ids above the uint8 ceiling."""
        labels = segmenter_nv_segment_ct_mri.taxonomy.all_labels()
        assert set(labels.keys()) == set(range(1, 346)), (
            "Taxonomy should cover the model's full [1, 346) index space"
        )

    def test_set_modality(
        self, segmenter_nv_segment_ct_mri: SegmentNVSegmentCTMRI
    ) -> None:
        """set_modality accepts the supported modalities and rejects others."""
        segmenter = SegmentNVSegmentCTMRI()

        segmenter.set_modality("MRI_BODY")
        assert segmenter.modality == "MRI_BODY"

        with pytest.raises(ValueError):
            segmenter.set_modality("PET")

        # The session-scoped fixture must be left untouched.
        assert segmenter_nv_segment_ct_mri.modality == "CT_BODY"

    def test_license_warning_logged_on_first_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The non-commercial license must be surfaced before weights load."""
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", lambda **kwargs: "/nonexistent"
        )

        segmenter = SegmentNVSegmentCTMRI()
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
class TestSegmentNVSegmentCTMRI:
    """End-to-end test; downloads ~872 MB of weights on first run."""

    def test_segment_single_image(
        self,
        segmenter_nv_segment_ct_mri: SegmentNVSegmentCTMRI,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test segmentation on a single time point."""
        output_dir = test_directories["output"]
        input_image = test_images[0]

        print("\nSegmenting time point 0...")
        print(f"  Input image size: {itk.size(input_image)}")

        result = segmenter_nv_segment_ct_mri.segment(input_image)

        assert isinstance(result, dict), "Result should be a dictionary"
        for key in ["labelmap", *EXPECTED_GROUPS, "other"]:
            assert key in result, f"Missing key '{key}' in result"
            assert result[key] is not None, f"Result['{key}'] is None"

        labelmap = result["labelmap"]
        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"

        labelmap_arr = itk.array_from_image(labelmap)
        assert labelmap_arr.dtype == np.uint16, "Labelmap should be uint16"

        unique_labels = np.unique(labelmap_arr)
        assert len(unique_labels) > 1, "Labelmap should contain multiple labels"
        assert 255 not in unique_labels, (
            "255 is the model's unpredicted-voxel sentinel and must be cleared "
            "for CT_BODY"
        )

        allowed = {0} | set(segmenter_nv_segment_ct_mri.taxonomy.all_labels().keys())
        assert set(unique_labels.tolist()) <= allowed, (
            "Labelmap contains ids outside the model's index space"
        )

        print("Segmentation complete for time point 0")
        print(f"  Unique labels: {len(unique_labels)}")

        seg_output_dir = output_dir / "segmentation_nv_segment_ct_mri"
        seg_output_dir.mkdir(exist_ok=True)
        itk.imwrite(
            labelmap, str(seg_output_dir / "slice_000_labelmap.mha"), compression=True
        )
        print(f"  Saved labelmap to: {seg_output_dir / 'slice_000_labelmap.mha'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
