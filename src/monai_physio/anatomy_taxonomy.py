"""Pure-data anatomy taxonomy shared by segmenters and USD renderers.

This module defines :class:`AnatomyTaxonomy` and :class:`AnatomyGroup`, a
minimal data type that maps anatomical groups (``heart``, ``lung``, ``bone``,
...) to the organ labels they contain.

The taxonomy is the single source of truth for the label hierarchy. Both
:class:`monai_physio.SegmentAnatomyBase` (which populates one via its
subclasses) and :class:`monai_physio.USDAnatomyTools` (which consumes one
when applying materials) depend on this class. The two consumers do not
depend on each other, which lets either side be used without the other.

The class is pure stdlib (`dataclasses`, `typing`); it does not import
``itk``, ``pxr``, or any other heavy dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AnatomyGroup:
    """One named anatomy group together with the organ labels it contains.

    Attributes:
        name: Group name (e.g. ``"heart"``, ``"lung"``).
        organs: Maps integer label id to organ name within this group.
    """

    name: str
    organs: dict[int, str] = field(default_factory=dict)


class AnatomyTaxonomy:
    """Mapping of anatomical groups to the organs each group contains.

    Groups are added in insertion order, which determines the order returned
    by :meth:`group_names` and the iteration order of :meth:`all_labels`.

    Example:
        >>> tax = AnatomyTaxonomy()
        >>> tax.add_organ("heart", 51, "heart")
        >>> tax.add_organ("heart", 61, "atrial_appendage_left")
        >>> tax.add_organ("lung", 10, "lung_upper_lobe_left")
        >>> tax.group_for_label("atrial_appendage_left")
        'heart'
        >>> tax.labels_in_group("lung")
        {10: 'lung_upper_lobe_left'}
    """

    OTHER_GROUP = "other"
    """Sentinel group name used by :meth:`fill_other_group` for unclaimed ids."""

    def __init__(self) -> None:
        self._groups: dict[str, AnatomyGroup] = {}

    def add_group(self, name: str) -> AnatomyGroup:
        """Ensure a group exists and return it.

        Args:
            name: Group name.

        Returns:
            The (new or existing) :class:`AnatomyGroup`.
        """
        if name not in self._groups:
            self._groups[name] = AnatomyGroup(name=name)
        return self._groups[name]

    def add_organ(self, group: str, label_id: int, organ_name: str) -> None:
        """Add one organ label to the named group.

        Creates the group if it does not yet exist. Reassigning the same
        label id within the same group is silently allowed (last write wins).
        If the same id is already registered in a *different* group, a warning
        is logged and the new assignment is dropped, so the first registration
        wins and :meth:`group_for_id` remains deterministic.

        Args:
            group: Target group name.
            label_id: Integer label id (e.g. TotalSegmentator class index).
            organ_name: Human-readable organ name.
        """
        for existing in self._groups.values():
            if existing.name == group:
                continue
            if label_id in existing.organs:
                logger.warning(
                    "label_id %d already registered in group %r as %r; "
                    "ignoring duplicate add to group %r as %r",
                    label_id,
                    existing.name,
                    existing.organs[label_id],
                    group,
                    organ_name,
                )
                return
        self.add_group(group).organs[label_id] = organ_name

    def group_names(self) -> list[str]:
        """Return group names in the order they were first added."""
        return list(self._groups.keys())

    def labels_in_group(self, group: str) -> dict[int, str]:
        """Return ``{label_id: organ_name}`` for *group*; empty dict if absent."""
        anatomy_group = self._groups.get(group)
        return dict(anatomy_group.organs) if anatomy_group is not None else {}

    def all_labels(self) -> dict[int, str]:
        """Return the union of every group's organs as a single id→name dict."""
        merged: dict[int, str] = {}
        for anatomy_group in self._groups.values():
            merged.update(anatomy_group.organs)
        return merged

    def group_for_label(self, label_name: str) -> str:
        """Return the group containing *label_name*.

        Falls back to :data:`OTHER_GROUP` if no group contains the name; this
        keeps :class:`monai_physio.ConvertVTKToUSD` happy when it encounters
        labels the segmenter did not classify.
        """
        for anatomy_group in self._groups.values():
            if label_name in anatomy_group.organs.values():
                return anatomy_group.name
        return self.OTHER_GROUP

    def group_for_id(self, label_id: int) -> str:
        """Return the group containing *label_id*; :data:`OTHER_GROUP` if absent."""
        for anatomy_group in self._groups.values():
            if label_id in anatomy_group.organs:
                return anatomy_group.name
        return self.OTHER_GROUP

    def fill_other_group(
        self,
        id_range: range = range(1, 256),
        name_template: str = "other_{id}",
    ) -> None:
        """Populate the ``other`` group with any ids not already claimed.

        Called by :class:`monai_physio.SegmentAnatomyBase` subclasses at the
        end of ``__init__`` to mark every id in the segmenter's class index
        space that no specific group claimed.

        Args:
            id_range: Inclusive-lower / exclusive-upper id space to scan.
            name_template: ``str.format`` template for synthetic organ names;
                receives the unclaimed id as ``{id}``.
        """
        claimed_ids: set[int] = set()
        for anatomy_group in self._groups.values():
            if anatomy_group.name == self.OTHER_GROUP:
                continue
            claimed_ids.update(anatomy_group.organs.keys())
        for label_id in id_range:
            if label_id not in claimed_ids:
                self.add_organ(
                    self.OTHER_GROUP, label_id, name_template.format(id=label_id)
                )
