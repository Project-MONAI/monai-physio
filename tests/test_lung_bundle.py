"""Tests for the lung workshop bundle utility."""

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_bundle_module() -> ModuleType:
    script = Path(__file__).parents[1] / "utils" / "lung_bundle.py"
    spec = importlib.util.spec_from_file_location("lung_bundle", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lung_bundle = _load_bundle_module()


def test_build_and_verify_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A built archive preserves and verifies its declared payload."""
    repository_root = tmp_path / "repository"
    source_file = repository_root / "data" / "sample.bin"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"monai-physio")
    monkeypatch.setitem(
        lung_bundle.PROFILE_PATTERNS, "test-profile", ("data/sample.bin",)
    )

    output_directory = tmp_path / "release"
    manifest_path = lung_bundle.build_bundles(
        repository_root, output_directory, ["test-profile"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest["profiles"]["test-profile"]
    assert profile["installed_size"] == len(b"monai-physio")

    installed_root = tmp_path / "installed"
    installed_root.mkdir()
    lung_bundle._extract_archive(output_directory / profile["archive"], installed_root)
    lung_bundle._restore_profile_mtimes(installed_root, manifest, "test-profile")
    lung_bundle.verify_profiles(installed_root, manifest, ["test-profile"])
    assert (installed_root / "data" / "sample.bin").read_bytes() == b"monai-physio"


def test_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    """Archive extraction rejects a member outside the repository root."""
    archive_path = tmp_path / "unsafe.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("unsafe", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(payload, arcname="../outside")

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    with pytest.raises(ValueError, match="Unsafe archive member"):
        lung_bundle._extract_archive(archive_path, repository_root)


def test_collect_profile_reports_missing_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile fails before packaging when required input is missing."""
    monkeypatch.setitem(
        lung_bundle.PROFILE_PATTERNS, "test-profile", ("missing/*.bin",)
    )
    with pytest.raises(FileNotFoundError, match="missing/\\*.bin"):
        lung_bundle.collect_profile_files(tmp_path, "test-profile")


def test_git_head_is_unknown_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bundle metadata remains usable in runtime images without Git."""

    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(lung_bundle.subprocess, "run", missing_git)

    assert lung_bundle._git_head(tmp_path) == "unknown"
