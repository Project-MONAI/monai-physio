"""Download and install the MONAI Physio lung workshop bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, cast

BUNDLE_FORMAT_VERSION = 1
DEFAULT_REPO_ID = "maximilianofir/physioMotionWorkshop"
DEFAULT_REVISION = "a6127dd1d2e27c5b59ed3a81c5b4e7490b4bd1bf"
PROFILE_NAMES = ("course", "offline-segmentation")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load a supported bundle manifest."""
    manifest = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported bundle manifest format: {manifest.get('format_version')!r}"
        )
    return manifest


def _safe_tar_filter(
    member: tarfile.TarInfo, destination: str
) -> Optional[tarfile.TarInfo]:  # noqa: UP045
    """Apply Python's restrictive data-archive extraction policy."""
    try:
        return tarfile.data_filter(member, destination)
    except tarfile.FilterError as error:
        raise ValueError(f"Unsafe archive member: {member.name}") from error


def _extract_archive(archive_path: Path, repository_root: Path) -> None:
    """Safely extract one verified data archive into the repository."""
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(repository_root, filter=_safe_tar_filter)


def install_bundles(
    repository_root: Path,
    repo_id: str,
    revision: str,
    profiles: Sequence[str],
) -> Path:
    """Download, verify, and safely extract selected bundle profiles."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required; run this command in the tutorial image"
        ) from error

    manifest_download = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="manifest.json",
            repo_type="dataset",
            revision=revision,
        )
    )
    manifest = _load_manifest(manifest_download)

    repository_root.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        try:
            profile_manifest = manifest["profiles"][profile]
        except KeyError as error:
            raise ValueError(f"Profile is absent from manifest: {profile}") from error

        archive_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=profile_manifest["archive"],
                repo_type="dataset",
                revision=revision,
            )
        )
        print(f"Checking archive: {archive_path.name}")
        if archive_path.stat().st_size != profile_manifest["archive_size"]:
            raise RuntimeError(f"Archive has the wrong size: {archive_path}")
        if _sha256(archive_path) != profile_manifest["archive_sha256"]:
            raise RuntimeError(f"Archive checksum mismatch: {archive_path}")

        print(f"Extracting {profile} into {repository_root}")
        _extract_archive(archive_path, repository_root)

    installed_manifest = (
        repository_root / ".cache" / "monai-physio" / "bundles" / "manifest.json"
    )
    installed_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_download, installed_manifest)
    print(f"Installed profiles: {', '.join(profiles)}")
    return installed_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--profile",
        action="append",
        choices=PROFILE_NAMES,
        help="Profile to install; repeat for multiple profiles (default: course)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP045
    """Install selected lung workshop bundle profiles."""
    args = _parser().parse_args(argv)
    profiles = list(dict.fromkeys(args.profile or ["course"]))
    install_bundles(
        args.repository_root.resolve(),
        repo_id=args.repo_id,
        revision=args.revision,
        profiles=profiles,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
