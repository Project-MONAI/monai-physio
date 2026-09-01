"""Tests for the lung workshop bundle installer."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

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


def _create_release(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    release_directory = tmp_path / "release"
    release_directory.mkdir()
    payload = tmp_path / "sample.bin"
    payload.write_bytes(b"monai-physio")

    archive_path = release_directory / "test-profile.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(payload, arcname="data/sample.bin")

    manifest = {
        "format_version": 1,
        "profiles": {
            "test-profile": {
                "archive": archive_path.name,
                "archive_size": archive_path.stat().st_size,
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            }
        },
    }
    (release_directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return release_directory, manifest


def _mock_huggingface_download(
    monkeypatch: pytest.MonkeyPatch, release_directory: Path
) -> None:
    huggingface_hub = ModuleType("huggingface_hub")

    def fake_hf_hub_download(**kwargs: str) -> str:
        assert kwargs["repo_type"] == "dataset"
        return str(release_directory / kwargs["filename"])

    huggingface_hub.__dict__["hf_hub_download"] = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)


def test_install_verifies_and_extracts_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installer verifies and extracts a downloaded archive."""
    release_directory, _ = _create_release(tmp_path)
    _mock_huggingface_download(monkeypatch, release_directory)

    repository_root = tmp_path / "repository"
    installed_manifest = lung_bundle.install_bundles(
        repository_root,
        repo_id="example/workshop",
        revision="pinned-revision",
        profiles=["test-profile"],
    )

    installed_file = repository_root / "data" / "sample.bin"
    assert installed_file.read_bytes() == b"monai-physio"
    assert installed_manifest.is_file()


def test_install_rejects_archive_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installer rejects an archive that does not match its manifest."""
    release_directory, manifest = _create_release(tmp_path)
    manifest["profiles"]["test-profile"]["archive_sha256"] = "0" * 64
    (release_directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _mock_huggingface_download(monkeypatch, release_directory)

    with pytest.raises(RuntimeError, match="Archive checksum mismatch"):
        lung_bundle.install_bundles(
            tmp_path / "repository",
            repo_id="example/workshop",
            revision="pinned-revision",
            profiles=["test-profile"],
        )


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


def test_extract_allows_internal_cache_symlink(tmp_path: Path) -> None:
    """Extraction preserves a Hugging Face cache link that stays in the root."""
    archive_path = tmp_path / "cache.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        blob = tarfile.TarInfo("cache/models/example/blobs/model")
        blob.size = len(b"weights")
        archive.addfile(blob, io.BytesIO(b"weights"))
        link = tarfile.TarInfo("cache/models/example/snapshots/revision/model.pt")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../blobs/model"
        archive.addfile(link)

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    lung_bundle._extract_archive(archive_path, repository_root)

    assert (
        repository_root / "cache/models/example/snapshots/revision/model.pt"
    ).read_bytes() == b"weights"
