"""The cohort strategies a movement evaluation is parameterized by.

Each cohort owns three things the evaluation workflow deliberately does not
know: which structures to score, how a stage is read off a filename, and how
that cohort's ground truth is assembled.  The first two are checked here against
the real taxonomies and real filename shapes, which needs no image and no
network.  Assembling ground truth reads whole gated series, so it is exercised
by the tutorial tests instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physiotwin4d import (
    EvaluateMovementBase,
    EvaluateMovementDukeHeart,
    EvaluateMovementLung,
)


def test_lung_scores_the_five_lobes() -> None:
    """The lobe ids resolve to the segmenter's own names, not restated ones."""
    names = EvaluateMovementLung().label_names()

    assert list(names) == [28, 29, 30, 31, 32]
    assert names[28] == "lung_upper_lobe_left"
    assert all("lobe" in name for name in names.values())


def test_heart_renames_the_structure_the_model_represents() -> None:
    """The taxonomy's "heart" is the muscle, not the sum of the chambers."""
    names = EvaluateMovementDukeHeart().label_names()

    assert list(names) == [1, 2, 3, 4, 5, 6]
    assert names[6] == "heart muscle"
    assert names[1] == "left_ventricle"


def test_a_cohort_that_names_no_segmenter_cannot_name_its_structures() -> None:
    """The base is a seam, not a usable cohort."""
    with pytest.raises(NotImplementedError, match="segmenter_class"):
        EvaluateMovementBase().label_names()


def test_lung_reads_the_respiratory_phase_tag() -> None:
    """``T{PP}`` of a ``.mha`` frame or a ``.vtp`` fit, as a fraction."""
    lung = EvaluateMovementLung()

    assert lung.stage_from_filename(Path("Case1Pack_T00.mha")) == 0.0
    assert lung.stage_from_filename(Path("Case1Pack_T70.mha")) == 0.7
    assert lung.stage_from_filename(Path("Case1Pack_T50_ssm_surface.vtp")) == 0.5
    with pytest.raises(ValueError, match="respiratory phase"):
        lung.stage_from_filename(Path("Case1Pack.mha"))


def test_heart_reads_the_cardiac_gate_through_a_double_suffix() -> None:
    """``.nii.gz`` is two suffixes, so the gate has to be read off the full name."""
    heart = EvaluateMovementDukeHeart()
    frame = Path("pm0002_dupr_111_4700_g020_s1.500_n0338_8_labelmap.nii.gz")

    assert heart.stage_from_filename(frame) == 0.2
    # A gate past end-systole exceeds 1.0; three digits, still divided by 100.
    assert heart.stage_from_filename(Path("pm0002_g120_x_labelmap.nii.gz")) == 1.2
    with pytest.raises(ValueError, match="cardiac gate"):
        heart.stage_from_filename(Path("pm0002_labelmap.nii.gz"))


def test_the_cohorts_carry_the_settings_their_anatomy_needs() -> None:
    """Dice is dropped for lobes and kept for chambers; the pitches differ."""
    lung, heart = EvaluateMovementLung(), EvaluateMovementDukeHeart()

    assert lung.report_dice is False
    assert heart.report_dice is True
    assert (lung.evaluation_spacing_mm, heart.evaluation_spacing_mm) == (2.0, 1.0)


def test_the_lung_refuses_to_segment_without_somewhere_to_cache_it(
    tmp_path: Path,
) -> None:
    """A segmentation pass per phase is too expensive to repeat every run."""
    with pytest.raises(ValueError, match="cache_directory"):
        EvaluateMovementLung().assemble_ground_truth(
            "Case1Pack", tmp_path, tmp_path, cache_directory=None
        )
