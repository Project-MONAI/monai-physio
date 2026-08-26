"""
setup_feature_worktree.py — Automate creation of a Git feature worktree on Windows.

Workflow:
  1. Validate prerequisites (git, py.exe).
  2. Resolve the repository root and derive default paths.
  3. Sanitize the feature name into a safe branch name and folder name.
  4. Create a new Git branch + worktree.
  5. Create a virtual environment inside the worktree using py.exe.
  6. Install uv into that venv via pip.
  7. Install project dependencies via uv (auto-detected or explicit mode).
  8. Print a summary with the activation command.

Usage:
  py utils/setup_feature_worktree.py my-feature
  py utils/setup_feature_worktree.py my-feature --base-branch main
  py utils/setup_feature_worktree.py my-feature --worktree-root C:/worktrees
  py utils/setup_feature_worktree.py my-feature --dependency-mode editable

Dependency modes: 'requirements' reads requirements.txt, 'pyproject' does a bare
'-e .', and 'editable' does a full '-e .[all]' (staging setuptools/wheel and CUDA
torch first, since the cuda13 and physicsnemo extras need them preinstalled).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Shell command helpers
# ---------------------------------------------------------------------------


def run(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    capture: bool = False,
    description: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run a command, raising on failure with a helpful message.

    Args:
        cmd: Command and arguments as a list of strings.
        cwd: Working directory for the command.
        capture: If True, capture stdout/stderr instead of letting them print.
        description: Human-readable label used in error messages.

    Returns:
        The CompletedProcess result.

    Raises:
        SystemExit: If the command returns a non-zero exit code.
    """
    label = description or " ".join(cmd[:3])
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            cwd=cwd,
            capture_output=capture,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        print(f'\n[ERROR] "{label}" failed (exit {exc.returncode}).')
        if stderr:
            print(f"        {stderr}")
        sys.exit(exc.returncode)
    return result


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------


def require_tool(name: str) -> str:
    """Ensure a tool is on PATH. Returns its resolved path.

    Args:
        name: Executable name (e.g. 'git', 'py').

    Raises:
        SystemExit: If the tool cannot be found.
    """
    path = shutil.which(name)
    if path is None:
        print(
            f'[ERROR] "{name}" was not found on PATH. Please install it and try again.'
        )
        sys.exit(1)
    return path


def require_no_active_venv() -> None:
    """Abort if a virtual environment is active in the invoking shell or interpreter.

    The script creates a fresh per-worktree venv and runs ``uv pip install`` inside
    it. uv honors the parent shell's ``VIRTUAL_ENV`` env var, so an active outer
    venv silently hijacks the install and the new worktree's venv is left empty.
    Refuse to run in that case and tell the user how to recover.

    Raises:
        SystemExit: If ``VIRTUAL_ENV`` is set or this script is being run by a
            venv's Python rather than the base interpreter.
    """
    active_venv = os.environ.get("VIRTUAL_ENV")
    running_in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    if active_venv or running_in_venv:
        location = active_venv or sys.prefix
        print("[ERROR] A Python virtual environment is currently active:")
        print(f"        {location}")
        print()
        print("        Deactivate it before running this script, then re-run.")
        print("        PowerShell : deactivate")
        print("        cmd.exe    : venv\\Scripts\\deactivate.bat")
        sys.exit(1)


def check_prerequisites() -> str:
    """Check that git and py.exe are available on PATH and no venv is active.

    Returns:
        The resolved path to py.exe. git is only checked, not returned — git is
        always invoked as a bare "git" so it resolves through PATH.
    """
    print("[*] Checking prerequisites...")
    require_no_active_venv()
    git_path = require_tool("git")
    py_path = require_tool("py")
    print(f"    git : {git_path}")
    print(f"    py  : {py_path}")
    return py_path


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def get_repo_root() -> Path:
    """Return the absolute path to the repository root.

    Raises:
        SystemExit: If the current directory is not inside a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        print(
            "[ERROR] Not inside a Git repository. Run this script from within a repo."
        )
        sys.exit(1)
    return Path(result.stdout.strip())


def get_current_branch() -> str:
    """Return the name of the currently checked-out branch.

    Falls back to the commit SHA if HEAD is detached.
    """
    result = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    # Detached HEAD — use the short SHA instead
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def branch_exists(branch_name: str) -> bool:
    """Return True if a local branch with this name already exists."""
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        text=True,
        capture_output=True,
    )
    return bool(result.stdout.strip())


# ---------------------------------------------------------------------------
# Name sanitisation
# ---------------------------------------------------------------------------


def sanitize_name(raw: str) -> tuple[str, str]:
    """Convert a raw feature name into a safe branch name and folder name.

    Rules applied:
      - Lowercase everything.
      - Replace whitespace and underscores with hyphens.
      - Strip any character that is not alphanumeric, a hyphen, or a dot.
      - Collapse consecutive hyphens.
      - Strip leading/trailing hyphens and dots (invalid in git branch names).

    Args:
        raw: The user-supplied feature name.

    Returns:
        (branch_name, folder_name) — both derived from the same sanitised slug.
        The branch_name is prefixed with 'feature/' per convention.

    Raises:
        SystemExit: If the sanitised name is empty.
    """
    slug = raw.lower()
    slug = re.sub(r"[\s_]+", "-", slug)  # spaces/underscores → hyphens
    slug = re.sub(r"[^a-z0-9\-.]", "", slug)  # remove unsafe chars
    slug = re.sub(r"-{2,}", "-", slug)  # collapse runs of hyphens
    slug = slug.strip("-.")

    if not slug:
        print(
            f'[ERROR] Feature name "{raw}" produced an empty slug after sanitisation.'
        )
        sys.exit(1)

    branch_name = f"feature/{slug}"
    folder_name = slug
    return branch_name, folder_name


# ---------------------------------------------------------------------------
# Worktree creation
# ---------------------------------------------------------------------------


def create_worktree(
    worktree_path: Path,
    branch_name: str,
    base_branch: Optional[str],
) -> None:
    """Create a new git branch and worktree.

    Args:
        worktree_path: Absolute path where the worktree will be created.
        branch_name: New branch to create inside the worktree.
        base_branch: Branch or commit to base the new branch on.
                     If None, the new branch is created from the current HEAD.

    Raises:
        SystemExit: If the worktree path already exists or the branch already exists.
    """
    if worktree_path.exists():
        print(f"[ERROR] Worktree path already exists: {worktree_path}")
        print("        Delete it or choose a different feature name / --worktree-root.")
        sys.exit(1)

    if branch_exists(branch_name):
        print(f'[ERROR] Branch "{branch_name}" already exists locally.')
        print(f"        Delete it with: git branch -D {branch_name}")
        sys.exit(1)

    print(f'[*] Creating branch "{branch_name}" and worktree at:')
    print(f"    {worktree_path}")

    cmd = ["git", "worktree", "add", str(worktree_path), "-b", branch_name]
    if base_branch:
        cmd.append(base_branch)
    # When base_branch is omitted, git creates the branch from HEAD automatically.

    run(cmd, description=f"git worktree add ({branch_name})")
    print("    Done.")


# ---------------------------------------------------------------------------
# Venv + uv setup
# ---------------------------------------------------------------------------


def create_venv(worktree_path: Path, py_path: str) -> Path:
    """Create a virtual environment inside the worktree.

    Args:
        worktree_path: Root of the worktree.
        py_path: Absolute path to py.exe.

    Returns:
        Path to the venv directory.
    """
    venv_dir = worktree_path / "venv"
    print(f"[*] Creating virtual environment at: {venv_dir}")
    run(
        [py_path, "-m", "venv", str(venv_dir)],
        cwd=worktree_path,
        description="py -m venv",
    )
    print("    Done.")
    return venv_dir


def install_uv(venv_dir: Path) -> tuple[Path, Path]:
    """Install uv into the venv using pip.

    Args:
        venv_dir: Path to the venv directory.

    Returns:
        Tuple of (uv_exe, venv_python) — both inside the venv.
    """
    venv_python = venv_dir / "Scripts" / "python.exe"
    uv_exe = venv_dir / "Scripts" / "uv.exe"

    if not venv_python.exists():
        print(f"[ERROR] Expected venv Python not found: {venv_python}")
        sys.exit(1)

    print("[*] Installing uv into the virtual environment...")
    run(
        [str(venv_python), "-m", "pip", "install", "--quiet", "uv"],
        description="pip install uv",
    )

    if not uv_exe.exists():
        print(f"[ERROR] uv.exe not found after installation: {uv_exe}")
        sys.exit(1)

    print("    Done.")
    return uv_exe, venv_python


# ---------------------------------------------------------------------------
# Dependency detection and installation
# ---------------------------------------------------------------------------


def detect_dependency_mode(worktree_path: Path) -> str:
    """Auto-detect the best dependency installation mode for the project.

    Detection order:
      1. requirements.txt  → 'requirements'
      2. pyproject.toml    → 'pyproject'
      3. fallback          → 'editable'

    Args:
        worktree_path: Root of the worktree.

    Returns:
        One of 'requirements', 'pyproject', 'editable'.
    """
    if (worktree_path / "requirements.txt").exists():
        return "requirements"
    if (worktree_path / "pyproject.toml").exists():
        return "pyproject"
    return "editable"


def install_dependencies(
    uv_exe: Path,
    venv_python: Path,
    worktree_path: Path,
    mode: str,
) -> None:
    """Install project dependencies using uv.

    Every uv invocation passes ``--python venv_python`` explicitly. uv does not
    infer the target environment from the uv.exe that runs it: it looks at
    VIRTUAL_ENV, CONDA_PREFIX, then a directory named ``.venv``. The worktree venv
    is named ``venv`` and ``require_no_active_venv`` guarantees VIRTUAL_ENV is
    unset, so without ``--python`` uv aborts with "No virtual environment found".

    Args:
        uv_exe: Absolute path to the uv executable inside the venv.
        venv_python: Absolute path to python.exe inside the venv.
        worktree_path: Root of the worktree (used as the working directory).
        mode: One of 'requirements', 'pyproject', 'editable', 'auto'.

    Raises:
        SystemExit: If the mode is unrecognised or required files are missing.
    """
    if mode == "auto":
        mode = detect_dependency_mode(worktree_path)
        print(f"[*] Auto-detected dependency mode: {mode}")

    print(f"[*] Installing dependencies (mode: {mode})...")

    uv_pip_install = [str(uv_exe), "pip", "install", "--python", str(venv_python)]

    if mode == "requirements":
        req_file = worktree_path / "requirements.txt"
        if not req_file.exists():
            print(f"[ERROR] requirements.txt not found at: {req_file}")
            sys.exit(1)
        run(
            uv_pip_install + ["-r", str(req_file)],
            cwd=worktree_path,
            description="uv pip install -r requirements.txt",
        )

    elif mode == "pyproject":
        # pyproject.toml is present but we cannot assume a uv lockfile exists or
        # that the project uses uv's project workflow (uv sync). The most robust
        # fallback that works with any PEP 517 build backend is an editable install,
        # which reads [project.dependencies] from pyproject.toml and installs them.
        # If the caller wants the project's optional extras as well, they should
        # use --dependency-mode editable.
        pyproject_file = worktree_path / "pyproject.toml"
        if not pyproject_file.exists():
            print(f"[ERROR] pyproject.toml not found at: {pyproject_file}")
            sys.exit(1)
        print(
            "    (pyproject mode: using editable install to read [project.dependencies])"
        )
        run(
            uv_pip_install + ["-e", "."],
            cwd=worktree_path,
            description="uv pip install -e . (pyproject)",
        )

    elif mode == "editable":
        # ".[all]" pulls in [cuda13] and [physicsnemo], which per the comment above
        # the "all" extra in pyproject.toml require torch and setuptools to already
        # be installed, plus --no-build-isolation when torch-scatter has no matching
        # wheel. A brand-new venv has neither, so stage the prerequisites first.
        # This mirrors .github/workflows/nightly-health.yml.
        print("    Installing build prerequisites (setuptools, wheel)...")
        run(
            uv_pip_install + ["setuptools", "wheel"],
            cwd=worktree_path,
            description="uv pip install setuptools wheel",
        )
        print("    Installing torch from the CUDA 13.0 index.")
        print("    This downloads several GB and can take many minutes.")
        run(
            uv_pip_install
            + [
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu130",
            ],
            cwd=worktree_path,
            description="uv pip install torch (cu130)",
        )
        print("    Installing the project with all extras...")
        run(
            uv_pip_install
            + ["-e", ".[all]", "--no-build-isolation-package", "torch-scatter"],
            cwd=worktree_path,
            description="uv pip install -e .[all]",
        )

    else:
        print(f'[ERROR] Unknown dependency mode: "{mode}".')
        print("        Valid values: auto, requirements, pyproject, editable")
        sys.exit(1)

    print("    Done.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    branch_name: str,
    worktree_path: Path,
    venv_dir: Path,
) -> None:
    """Print a formatted summary of the created worktree environment."""
    venv_python = venv_dir / "Scripts" / "python.exe"

    separator = "=" * 60
    print(f"\n{separator}")
    print("  Worktree setup complete!")
    print(separator)
    print(f"  Branch      : {branch_name}")
    print(f"  Worktree    : {worktree_path}")
    print(f"  Venv Python : {venv_python}")
    print()
    print("  To activate the venv in PowerShell, run:")
    print(f'    cd "{worktree_path}"')
    print("    .\\venv\\Scripts\\Activate.ps1")
    print()
    print("  Or directly invoke the Python interpreter:")
    print(f'    "{venv_python}" your_script.py')
    print(separator)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="setup_feature_worktree.py",
        description="Create a Git feature worktree with a Python venv for Windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py utils/setup_feature_worktree.py my-cool-feature
  py utils/setup_feature_worktree.py my-cool-feature --base-branch main
  py utils/setup_feature_worktree.py my-cool-feature --worktree-root C:/worktrees
  py utils/setup_feature_worktree.py my-cool-feature --dependency-mode requirements
""",
    )

    parser.add_argument(
        "feature_name",
        help="Feature name (will be sanitised into a branch name and folder name).",
    )
    parser.add_argument(
        "--worktree-root",
        metavar="DIR",
        default=None,
        help=(
            "Parent directory in which to create the worktree folder. "
            "Default: <repo-parent>/<repo-name>-worktrees/"
        ),
    )
    parser.add_argument(
        "--base-branch",
        metavar="BRANCH",
        default=None,
        help=(
            "Branch or commit to base the new branch on. "
            "Default: the currently checked-out branch / HEAD."
        ),
    )
    parser.add_argument(
        "--dependency-mode",
        metavar="MODE",
        default="auto",
        choices=["auto", "requirements", "pyproject", "editable"],
        help=(
            "How to install dependencies. "
            "auto (default): detect from project files. "
            "requirements: use requirements.txt. "
            "pyproject: use pyproject.toml (via a bare editable install). "
            'editable: editable install with all extras ("-e .[all]", '
            "staging setuptools/wheel and CUDA torch first; downloads several GB)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the worktree setup script."""
    args = parse_args()

    # --- Prerequisites ---
    py_path = check_prerequisites()

    # --- Resolve repository root ---
    repo_root = get_repo_root()
    repo_name = repo_root.name
    print(f"[*] Repository root : {repo_root}")

    # --- Derive default worktree root ---
    if args.worktree_root is not None:
        worktree_root = Path(args.worktree_root)
    else:
        # Place worktrees alongside the repo: <repo-parent>/<repo-name>-worktrees/
        worktree_root = repo_root.parent / f"{repo_name}-worktrees"

    # --- Sanitise feature name ---
    branch_name, folder_name = sanitize_name(args.feature_name)
    worktree_path = worktree_root / folder_name

    print(f"[*] Branch name     : {branch_name}")
    print(f"[*] Worktree path   : {worktree_path}")

    # --- Determine base branch ---
    base_branch: Optional[str] = args.base_branch
    if base_branch is None:
        base_branch_display = get_current_branch()
        print(f"[*] Base branch     : {base_branch_display} (current HEAD)")
    else:
        print(f"[*] Base branch     : {base_branch} (explicit)")

    # Ensure the worktree root directory exists before creating the worktree.
    worktree_root.mkdir(parents=True, exist_ok=True)

    # --- Create worktree ---
    create_worktree(worktree_path, branch_name, base_branch)

    # --- Create venv ---
    venv_dir = create_venv(worktree_path, py_path)

    # --- Install uv ---
    uv_exe, venv_python = install_uv(venv_dir)

    # --- Install dependencies ---
    install_dependencies(uv_exe, venv_python, worktree_path, args.dependency_mode)

    # --- Summary ---
    print_summary(branch_name, worktree_path, venv_dir)


if __name__ == "__main__":
    main()
