"""Rename the `*_tools` module and class family to `process_*` / `Process*`.

Renames each old module (and, where present, its dedicated test file and
Sphinx API page) via ``git mv``, then rewrites, tree-wide, only the
*anchored* forms of each old module/class name - import statements, dotted
``monai_physio.<module>`` references, Sphinx cross-reference roles, and
``<module>.py`` filename mentions in prose - plus the whole-word PascalCase
class names (including their ``Test<Class>`` pytest mirrors). It deliberately
does not touch bare lowercase identifiers such as ``self.transform_tools``
instance attributes or the ``contour_tools``/``transform_tools`` pytest
fixtures, since those are a separate, intentionally unchanged, kind of name.

Usage::

    py utils/rename_tools_to_process.py --dry-run   # preview the diff
    py utils/rename_tools_to_process.py              # apply

Safe to run from any directory inside the target checkout; the file list
comes from ``git ls-files``, so it never touches build artifacts, caches, or
untracked files, and it refuses to run against a dirty working tree unless
``--allow-dirty`` is passed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: (old_module, new_module, old_class_or_None, new_class_or_None)
RENAMES: list[tuple[str, str, str | None, str | None]] = [
    ("image_tools", "process_images", "ImageTools", "ProcessImages"),
    ("labelmap_tools", "process_labelmaps", "LabelmapTools", "ProcessLabelmaps"),
    ("transform_tools", "process_transforms", "TransformTools", "ProcessTransforms"),
    ("landmark_tools", "process_landmarks", "LandmarkTools", "ProcessLandmarks"),
    ("contour_tools", "process_contours", "ContourTools", "ProcessContours"),
    ("test_tools", "process_tests", "TestTools", "ProcessTests"),
    ("data_download_tools", "download_data", "DataDownloadTools", "DownloadData"),
    (
        "physicsnemo_tools",
        "process_physicsnemo",
        "PhysicsNemoTools",
        "ProcessPhysicsNemo",
    ),
    (
        "usd_anatomy_tools",
        "process_usd_anatomy",
        "USDAnatomyTools",
        "ProcessUSDAnatomy",
    ),
    ("usd_tools", "process_usd", "USDTools", "ProcessUSD"),
]

#: Extra ``git mv`` pairs (relative to repo root) that don't follow the
#: ``src/monai_physio/<old>.py`` / ``tests/test_<old>.py`` naming pattern.
EXTRA_FILE_RENAMES: list[tuple[str, str]] = [
    ("tests/test_physicsnemo_tools.py", "tests/test_process_physicsnemo.py"),
    ("docs/api/utilities/data_download.rst", "docs/api/utilities/download_data.rst"),
    ("docs/api/usd/anatomy_tools.rst", "docs/api/usd/process_usd_anatomy.rst"),
    ("docs/api/usd/tools.rst", "docs/api/usd/process_usd.rst"),
    ("docs/api/utilities/test_tools.rst", "docs/api/utilities/process_tests.rst"),
]

_ALLOWED_SUFFIXES = {".py", ".rst", ".md", ".txt"}


def _git_mv(root: Path, old: str, new: str, dry_run: bool) -> None:
    old_path = root / old
    if not old_path.exists():
        return
    if dry_run:
        print(f"would rename: {old} -> {new}")
        return
    subprocess.run(["git", "mv", old, new], cwd=root, check=True)
    print(f"renamed: {old} -> {new}")


def _rename_files(root: Path, dry_run: bool) -> None:
    for old_mod, new_mod, _old_cls, _new_cls in RENAMES:
        _git_mv(
            root,
            f"src/monai_physio/{old_mod}.py",
            f"src/monai_physio/{new_mod}.py",
            dry_run,
        )
        _git_mv(
            root,
            f"tests/test_{old_mod}.py",
            f"tests/test_{new_mod}.py",
            dry_run,
        )
        _git_mv(
            root,
            f"docs/api/utilities/{old_mod}.rst",
            f"docs/api/utilities/{new_mod}.rst",
            dry_run,
        )
    for old, new in EXTRA_FILE_RENAMES:
        _git_mv(root, old, new, dry_run)


def _tracked_files(root: Path, file_list: list[str] | None) -> list[Path]:
    """Return every git-tracked file under *root* worth scanning.

    ``git ls-files`` already respects ``.gitignore``, so build artifacts and
    caches (``docs/_build/``, ``.mypy_cache/``, ``graphify-out/``, ...) are
    never touched without a manual exclusion list. *file_list*, when given
    (one ``git ls-files``-style relative path per line), is used instead of
    shelling out to git - some environments don't have git on the Python
    process's ``PATH`` even though the shell running this script does.
    """
    if file_list is None:
        result = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        )
        file_list = result.stdout.splitlines()
    files = []
    for line in file_list:
        rel = line.strip()
        if not rel:
            continue
        path = Path(rel)
        if path.suffix not in _ALLOWED_SUFFIXES:
            continue
        # Migration docs deliberately document the *old* names in their
        # Before/After snippets and must stay accurate.
        if re.fullmatch(r"docs/developer/migration(_next|_[\w.]+)?\.md", rel):
            continue
        files.append(root / path)
    return files


def _build_patterns() -> list[tuple[re.Pattern[str], str]]:
    patterns: list[tuple[re.Pattern[str], str]] = []
    for old_mod, new_mod, old_cls, new_cls in RENAMES:
        old_mod_re = re.escape(old_mod)
        new_mod_esc = new_mod.replace("\\", "\\\\")

        # Dotted references: `monai_physio.<old_mod>`, `import
        # monai_physio.<old_mod>`, autodoc directives, Sphinx cross-reference
        # roles (`:func:`monai_physio.<old_mod>.foo``), etc.
        patterns.append(
            (re.compile(rf"\bmonai_physio\.{old_mod_re}\b"), f"monai_physio.{new_mod}")
        )
        # Relative imports, only meaningful inside src/monai_physio/*.py or its
        # subpackages: `from .<old_mod> import ...` / `from ..<old_mod> import ...`.
        patterns.append(
            (re.compile(rf"from (\.\.?){old_mod_re}\b"), rf"from \g<1>{new_mod_esc}")
        )
        # Bare-name imports: `from monai_physio import <old_mod>` /
        # `from monai_physio import <old_mod> as alias`, and the relative form
        # `from . import <old_mod> as alias` / `from .. import <old_mod>`.
        patterns.append(
            (
                re.compile(rf"(from monai_physio import ){old_mod_re}\b"),
                rf"\g<1>{new_mod_esc}",
            )
        )
        patterns.append(
            (
                re.compile(rf"(from \.\.? import ){old_mod_re}\b"),
                rf"\g<1>{new_mod_esc}",
            )
        )
        # Filename mentions in prose/docs: `<old_mod>.py`.
        patterns.append((re.compile(rf"\b{old_mod_re}\.py\b"), f"{new_mod}.py"))

        if old_cls and new_cls:
            new_cls_esc = new_cls.replace("\\", "\\\\")
            # The pytest mirror class first, so `TestContourTools` becomes
            # `TestProcessContours` rather than leaving a `TestContourTools`/
            # `ProcessContours` mismatch (the bare-class pattern below cannot
            # match inside `TestContourTools`: no word boundary precedes it).
            patterns.append((re.compile(rf"\bTest{old_cls}\b"), f"Test{new_cls_esc}"))
            patterns.append((re.compile(rf"\b{old_cls}\b"), new_cls_esc))

    # Doc-page filename mentions that don't mirror the module 1:1.
    patterns.append(
        (
            re.compile(r"\bdocs/api/utilities/data_download\.rst\b"),
            "docs/api/utilities/download_data.rst",
        )
    )
    patterns.append(
        (
            re.compile(r"\bdocs/api/usd/anatomy_tools\.rst\b"),
            "docs/api/usd/process_usd_anatomy.rst",
        )
    )
    patterns.append(
        (re.compile(r"\bdocs/api/usd/tools\.rst\b"), "docs/api/usd/process_usd.rst")
    )
    patterns.append(
        (
            re.compile(r"\bdocs/api/utilities/test_tools\.rst\b"),
            "docs/api/utilities/process_tests.rst",
        )
    )
    return patterns


def _rewrite(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> str:
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Repository root (default: cwd)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a diff-style report, change nothing",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Skip the clean-working-tree check",
    )
    parser.add_argument(
        "--files-from",
        type=Path,
        default=None,
        help=(
            "Read the tracked-file list (one 'git ls-files'-style relative "
            "path per line) from this file instead of shelling out to git."
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not args.allow_dirty:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        if status.stdout.strip():
            print(
                "Working tree is not clean; commit/stash first or pass --allow-dirty.",
                file=sys.stderr,
            )
            return 1

    _rename_files(root, args.dry_run)

    file_list = (
        args.files_from.read_text(encoding="utf-8").splitlines()
        if args.files_from
        else None
    )
    patterns = _build_patterns()
    changed_files = 0
    for path in _tracked_files(root, file_list):
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = _rewrite(original, patterns)
        if updated == original:
            continue
        changed_files += 1
        rel = path.relative_to(root)
        if args.dry_run:
            print(f"would change: {rel}")
        else:
            path.write_text(updated, encoding="utf-8")
            print(f"changed: {rel}")

    print(f"\n{changed_files} file(s) {'would be ' if args.dry_run else ''}changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
