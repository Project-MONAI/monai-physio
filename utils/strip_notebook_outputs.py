"""Strip outputs and execution counts from Jupyter notebooks.

Running a notebook rewrites its outputs, execution counts and per-run metadata,
so a notebook that was only run, not edited, shows up as changed.  This clears,
in every code cell, ``outputs`` and ``execution_count``, and the run timestamps
some front ends store in cell metadata (``execution``, ``ExecuteTime``); and it
drops the notebook-level ``widgets`` state.  Sources and all other metadata are
kept, and the JSON is written back in the layout Jupyter uses, so an unchanged
notebook is left byte for byte as it was.

Run it on the given notebooks, or with none on every notebook git tracks::

    python utils/strip_notebook_outputs.py [NOTEBOOK ...]

It exits with status 1 when it changed a file, so that pre-commit, which runs
it on the staged notebooks, stops the commit until the stripped files are
staged.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_RUN_CELL_METADATA = ("execution", "ExecuteTime")
_RUN_NOTEBOOK_METADATA = ("widgets",)


def strip_notebook(notebook: dict[str, Any]) -> None:
    """Clear the outputs and run state of ``notebook`` in place."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        metadata = cell.get("metadata", {})
        for key in _RUN_CELL_METADATA:
            metadata.pop(key, None)
    for key in _RUN_NOTEBOOK_METADATA:
        notebook.get("metadata", {}).pop(key, None)


def strip_file(path: Path) -> bool:
    """Strip one notebook file, returning whether it changed."""
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    notebook = json.loads(original.decode("utf-8"))
    strip_notebook(notebook)
    text = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    stripped = text.encode("utf-8").replace(b"\n", newline)
    if stripped == original:
        return False
    path.write_bytes(stripped)
    return True


def tracked_notebooks() -> list[Path]:
    """Return every ``.ipynb`` file git tracks in this repository."""
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    )
    names = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        capture_output=True,
        check=True,
        cwd=root,
        text=True,
    ).stdout.splitlines()
    return [root / name for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebooks to strip. Default: every notebook git tracks.",
    )
    args = parser.parse_args()
    changed = [
        path for path in args.notebooks or tracked_notebooks() if strip_file(path)
    ]
    for path in changed:
        print(f"Stripped outputs from {path}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
