"""Migrate code from ``ProcessContours`` to the split mesh tools.

``ProcessContours`` was split: its tetrahedral methods (``extract_tetrahedra``,
``trim_tetrahedra_to_surface``, ``repair_inverted_tetrahedra``) moved to
:class:`ProcessVolumetricMesh`, ``apply_anatomy_color`` to
:class:`ProcessUSDAnatomy`, the savers (``save_surfaces``,
``save_combined_surfaces``) to :class:`WorkflowConvertImageToVTK`, and
everything else to :class:`ProcessSurfaceMesh`.  This rewrites, in ``.py`` and
``.ipynb`` files:

- ``monai_physio.process_contours`` → ``monai_physio.process_surface_mesh``;
- the whole word ``ProcessContours`` → ``ProcessSurfaceMesh``;
- on a surface-tools receiver - ``ProcessSurfaceMesh`` itself, or a name the
  file assigns ``ProcessSurfaceMesh(...)`` to, such as ``contour_tools`` or
  ``self.contour_tools`` - ``<receiver>.extract_tetrahedra(`` (and the other
  two tetrahedral methods) → ``ProcessVolumetricMesh().extract_tetrahedra(``,
  ``<receiver>.apply_anatomy_color(`` →
  ``ProcessUSDAnatomy(stage=None).apply_anatomy_color(``, and
  ``<receiver>.save_surfaces(`` / ``save_combined_surfaces(`` →
  ``WorkflowConvertImageToVTK.save_surfaces(`` and so on;

adding ``from monai_physio import <class>`` for each class a rewrite needs,
after the first single-line ``from monai_physio... import`` of the file.  A
call on any other receiver is left alone, so check calls on objects passed in
from elsewhere by hand.

These have no mechanical rewrite, so lines using them are flagged instead:

- ``extract_contours`` and ``extract_watertight_surface``: use
  ``ProcessSurfaceMesh.extract_label_surfaces``, then
  ``ProcessSurfaceMesh.combine_surfaces`` for one multi-label mesh, or ``[1]``
  for a binary mask's surface;
- ``boundary_labels``: read per-cell ``SegmentationLabelIds`` instead;
- ``segmentation_masks``: no longer returned by
  ``WorkflowConvertImageToVTK.process``;
- a path passed to ``ProcessUSD``'s stage methods: open the stage with
  ``ProcessUSD.open_stage`` and write it with ``ProcessUSD.save_stage``.

Calls already made on the right class are left alone.  Every changed line is
printed, and so is every flagged line and any import that could not be placed;
fix those by hand.

Usage::

    py utils/migrate_process_contours.py --dry-run PATH [PATH ...]
    py utils/migrate_process_contours.py PATH [PATH ...]

Each ``PATH`` is a file or a directory searched recursively.
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

TETRA_METHODS = (
    "extract_tetrahedra",
    "trim_tetrahedra_to_surface",
    "repair_inverted_tetrahedra",
)
SAVE_METHODS = ("save_surfaces", "save_combined_surfaces")
#: Class → (methods that moved to it off ProcessContours, and the expression
#: their calls are now made on).
MOVED_METHODS = {
    "ProcessVolumetricMesh": (TETRA_METHODS, "ProcessVolumetricMesh()"),
    "ProcessUSDAnatomy": (("apply_anatomy_color",), "ProcessUSDAnatomy(stage=None)"),
    "WorkflowConvertImageToVTK": (SAVE_METHODS, "WorkflowConvertImageToVTK"),
}
USD_STAGE_METHODS = (
    "list_mesh_primvars",
    "apply_colormap_from_primvar",
    "set_solid_display_color",
    "list_mesh_paths_under",
    "repair_mesh_primvar_element_sizes",
)
FLAGGED = re.compile(
    r"\b(extract_contours|extract_watertight_surface|boundary_labels"
    r"|segmentation_masks)\b"
    rf"|\.({'|'.join(USD_STAGE_METHODS)})\(\s*(str\(|[\"'])"
)


def _surface_receivers(text: str) -> set[str]:
    """Return the names *text* uses for surface tools, by their last component."""
    assigned = re.findall(
        r"([A-Za-z_][\w.]*)\s*(?::[^=\n]+)?=\s*ProcessSurfaceMesh\(", text
    )
    return {"ProcessSurfaceMesh"} | {name.split(".")[-1] for name in assigned}


def _call_pattern(receivers: set[str], methods: tuple[str, ...]) -> re.Pattern[str]:
    """Match ``<receiver>.<method>(``, the receiver optionally called or dotted."""
    names = "|".join(re.escape(name) for name in sorted(receivers))
    return re.compile(
        rf"(?<![\w.])(?:[A-Za-z_]\w*\.)*(?:{names})(?:\([^()\n]*\))?"
        rf"\.({'|'.join(methods)})\("
    )


def _add_import(text: str, name: str) -> tuple[str, bool]:
    """Return *text* importing *name*, and whether no place for it was found."""
    if re.search(rf"import[^\n]*\b{name}\b", text):
        return text, False
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if re.match(r"\s*from monai_physio[\w.]* import [^(\n]+$", line):
            indent = line[: len(line) - len(line.lstrip())]
            lines.insert(index + 1, f"{indent}from monai_physio import {name}")
            return "\n".join(lines), False
    return text, True


def migrate(text: str) -> tuple[str, list[str]]:
    """Return *text* migrated, and the imports that must be added by hand."""
    text = text.replace(
        "monai_physio.process_contours", "monai_physio.process_surface_mesh"
    )
    text = text.replace(
        "from .process_contours import", "from .process_surface_mesh import"
    )
    text = re.sub(r"\bProcessContours\b", "ProcessSurfaceMesh", text)
    receivers = _surface_receivers(text)
    missing: list[str] = []
    for owner, (methods, receiver) in MOVED_METHODS.items():
        text, calls = _call_pattern(receivers, methods).subn(rf"{receiver}.\1(", text)
        # References that are not calls, such as ":meth:" roles in docstrings.
        text = re.sub(
            rf"\bProcessSurfaceMesh\.({'|'.join(methods)})\b", rf"{owner}.\1", text
        )
        if calls:
            text, not_placed = _add_import(text, owner)
            if not_placed:
                missing.append(owner)
    return text, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix in (".py", ".ipynb"))
        else:
            files.append(path)

    changed = 0
    flagged = 0
    for path in files:
        original = path.read_text(encoding="utf-8")
        migrated, missing_imports = migrate(original)
        hits = [
            (number, line.strip())
            for number, line in enumerate(migrated.split("\n"), start=1)
            if FLAGGED.search(line)
        ]
        if migrated == original and not hits:
            continue
        print(f"== {path}")
        if migrated != original:
            changed += 1
            for line in difflib.unified_diff(
                original.split("\n"), migrated.split("\n"), lineterm="", n=0
            ):
                if line[:1] in "-+" and line[:3] not in ("---", "+++"):
                    print(f"  {line[0]} {line[1:].strip()}")
        for name in missing_imports:
            print(f"  ! add `from monai_physio import {name}` to {path} by hand")
        for number, line in hits:
            flagged += 1
            print(f"  ? line {number}, convert by hand: {line}")
        if not args.dry_run and migrated != original:
            path.write_text(migrated, encoding="utf-8")
    print(
        f"{changed} file(s) {'would change' if args.dry_run else 'changed'}, "
        f"{flagged} line(s) to convert by hand"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
