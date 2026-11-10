"""Convert ``ProcessVolumetricMesh.extract_tetrahedra`` calls to ``extract_label_tetrahedra_vtk``.

``extract_tetrahedra`` meshed one binary mask and returned one grid;
``extract_label_tetrahedra_vtk`` meshes every label of a labelmap and returns
``{label id: grid}``.  This rewrites every ``extract_tetrahedra(...)`` call in
``.py`` and ``.ipynb`` files to ``extract_label_tetrahedra_vtk(...)[1]``, which
is the same mesh for a 0/1 mask (every mask monai_physio builds).  Names that
only start with ``extract_tetrahedra``, such as ``extract_tetrahedra_netgen``,
are left alone.  Every changed line is printed, so check that none of them was
an unrelated name of your own.

Two cases need a hand edit:

- A call passing ``anatomy_names`` (reported, along with any call of three
  or more arguments): the argument is gone; color the mesh with
  ``ProcessUSDAnatomy.apply_anatomy_color`` instead.
- A mask whose foreground is not 1, or one that may be empty: index the result
  by that value, or use ``.get(1)``, since an empty mask now gives ``{}``.

Code still using ``ProcessContours`` should run
``utils/migrate_process_contours.py`` first, then this script.

Usage::

    py utils/migrate_extract_label_tetrahedra.py --dry-run PATH [PATH ...]
    py utils/migrate_extract_label_tetrahedra.py PATH [PATH ...]

Each ``PATH`` is a file or a directory searched recursively.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

OLD_CALL = re.compile(r"\bextract_tetrahedra\s*\(")
NEW_NAME = "extract_label_tetrahedra_vtk"


def _scan(text: str, start: int) -> tuple[int, int]:
    """Return the ``)`` closing the ``(`` before *start*, and its top-level commas."""
    depth = 1
    commas = 0
    quote = ""
    index = start
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index, commas
        elif char == "," and depth == 1:
            commas += 1
        index += 1
    raise ValueError("unbalanced parentheses after extract_tetrahedra(")


def migrate(text: str) -> tuple[str, list[str]]:
    """Return *text* with every call converted, and the calls to check by hand."""
    pieces: list[str] = []
    manual: list[str] = []
    position = 0
    for match in OLD_CALL.finditer(text):
        if match.start() < position:
            continue
        close, commas = _scan(text, match.end())
        arguments = text[match.end() : close]
        pieces.append(text[position : match.start()])
        pieces.append(f"{NEW_NAME}({arguments})[1]")
        position = close + 1
        if (
            "anatomy_names" in arguments
            or commas - arguments.rstrip().endswith(",") >= 2
        ):
            manual.append(" ".join(f"{NEW_NAME}({arguments})".split()))
    pieces.append(text[position:])
    return "".join(pieces), manual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()
    # Changed lines can hold characters a Windows console cannot encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix in (".py", ".ipynb"))
        else:
            files.append(path)

    changed = 0
    for path in files:
        original = path.read_text(encoding="utf-8")
        migrated, manual = migrate(original)
        if migrated == original:
            continue
        changed += 1
        print(f"== {path}")
        for line in difflib.unified_diff(
            original.split("\n"), migrated.split("\n"), lineterm="", n=0
        ):
            if line[:1] in "-+" and line[:3] not in ("---", "+++"):
                print(f"  {line[0]} {line[1:].strip()}")
        for call in manual:
            print(f"  ! drop anatomy_names by hand: {call}")
        if not args.dry_run:
            path.write_text(migrated, encoding="utf-8")
    print(f"{changed} file(s) {'would change' if args.dry_run else 'changed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
