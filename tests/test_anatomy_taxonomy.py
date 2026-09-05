"""Unit tests for :class:`monai_physio.AnatomyTaxonomy`.

These tests exercise the pure-data taxonomy in isolation - no ITK, no pxr,
no GPU. They are fast and run unconditionally in the default test suite.
"""

from __future__ import annotations

import pytest

from monai_physio import AnatomyGroup, AnatomyTaxonomy


def test_add_organ_creates_group_lazily() -> None:
    tax = AnatomyTaxonomy()
    assert tax.group_names() == []

    tax.add_organ("heart", 51, "heart")
    assert tax.group_names() == ["heart"]
    assert tax.labels_in_group("heart") == {51: "heart"}


def test_group_names_preserve_insertion_order() -> None:
    tax = AnatomyTaxonomy()
    tax.add_organ("lung", 10, "lung_upper_lobe_left")
    tax.add_organ("heart", 51, "heart")
    tax.add_organ("bone", 91, "skull")
    assert tax.group_names() == ["lung", "heart", "bone"]


def test_labels_in_group_unknown_returns_empty_dict() -> None:
    tax = AnatomyTaxonomy()
    tax.add_organ("heart", 51, "heart")
    assert tax.labels_in_group("does_not_exist") == {}


def test_labels_in_group_returns_copy_not_alias() -> None:
    """Caller mutation of the returned dict must not corrupt the taxonomy."""
    tax = AnatomyTaxonomy()
    tax.add_organ("heart", 51, "heart")
    snapshot = tax.labels_in_group("heart")
    snapshot[999] = "bogus"
    assert tax.labels_in_group("heart") == {51: "heart"}


def test_all_labels_merges_every_group() -> None:
    tax = AnatomyTaxonomy()
    tax.add_organ("heart", 51, "heart")
    tax.add_organ("heart", 61, "atrial_appendage_left")
    tax.add_organ("lung", 10, "lung_upper_lobe_left")
    assert tax.all_labels() == {
        51: "heart",
        61: "atrial_appendage_left",
        10: "lung_upper_lobe_left",
    }


def test_group_for_label_finds_by_organ_name() -> None:
    tax = AnatomyTaxonomy()
    tax.add_organ("heart", 51, "heart")
    tax.add_organ("heart", 61, "atrial_appendage_left")
    tax.add_organ("lung", 10, "lung_upper_lobe_left")
    assert tax.group_for_label("atrial_appendage_left") == "heart"
    assert tax.group_for_label("lung_upper_lobe_left") == "lung"


def test_group_for_label_unknown_falls_back_to_other() -> None:
    tax = AnatomyTaxonomy()
    tax.add_organ("heart", 51, "heart")
    assert tax.group_for_label("not_in_any_group") == "other"


def test_group_for_id_finds_by_id() -> None:
    tax = AnatomyTaxonomy()
    tax.add_organ("heart", 51, "heart")
    tax.add_organ("lung", 10, "lung_upper_lobe_left")
    assert tax.group_for_id(51) == "heart"
    assert tax.group_for_id(10) == "lung"
    assert tax.group_for_id(9999) == "other"


def test_fill_other_group_only_claims_unassigned_ids() -> None:
    tax = AnatomyTaxonomy()
    tax.add_organ("heart", 51, "heart")
    tax.add_organ("lung", 10, "lung_upper_lobe_left")
    tax.fill_other_group(id_range=range(1, 12))

    # Existing assignments untouched.
    assert tax.group_for_id(51) == "heart"
    assert tax.group_for_id(10) == "lung"

    # Unclaimed ids in [1, 12) all landed under 'other' with synthetic names.
    other_labels = tax.labels_in_group("other")
    expected_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11}
    assert set(other_labels.keys()) == expected_ids
    for label_id in expected_ids:
        assert other_labels[label_id] == f"other_{label_id}"


def test_fill_other_group_idempotent_for_already_claimed_other() -> None:
    """Calling fill_other_group twice must not duplicate or overwrite."""
    tax = AnatomyTaxonomy()
    tax.add_organ("heart", 51, "heart")
    tax.fill_other_group(id_range=range(50, 53))
    snapshot = dict(tax.labels_in_group("other"))
    tax.fill_other_group(id_range=range(50, 53))
    assert tax.labels_in_group("other") == snapshot


def test_anatomy_group_dataclass_default_organs() -> None:
    group = AnatomyGroup(name="heart")
    assert group.name == "heart"
    assert group.organs == {}


def test_segment_anatomy_base_default_taxonomy_seeded() -> None:
    # Import lazily to avoid pulling itk in test collection if it's unused
    # by sibling tests in this module.
    from monai_physio import SegmentAnatomyBase

    seg = SegmentAnatomyBase()
    assert len(seg.taxonomy.all_labels()) == 0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
