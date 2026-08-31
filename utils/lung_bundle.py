#!/usr/bin/env python3
"""Build, download, and verify MONAI Physio lung workshop bundles."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

BUNDLE_FORMAT_VERSION = 1
DEFAULT_REPO_ID = "maximilianofir/physioMotionWorkshop"
DEFAULT_REVISION = "a6127dd1d2e27c5b59ed3a81c5b4e7490b4bd1bf"
NV_SEGMENT_REVISION = "4fb8b4a6b2532be9f1c449a3726fe5440ab4213a"

COURSE_PATTERNS = (
    "data/DirLab-4DCT/Case1Pack_T??.mha",
    "tutorials/output/tutorial_01_lung/lung_model.all_painted.usd",
    "tutorials/output/tutorial_04_lung/" "patient_nvsegmentctmri_lung_labelmap.mha",
    "tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_lung.vtp",
    "tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_surfaces.vtp",
    "tutorials/output/tutorial_04_lung/" "patient_nvsegmentctmri_lung_*_lobe_*.vtp",
    "tutorials/output/tutorial_06_lung/pca_model.json",
    "tutorials/output/tutorial_06_lung/pca_mean_surface.vtp",
    "tutorials/output/tutorial_06_lung/pca_mode_01_minus_2sigma.vtp",
    "tutorials/output/tutorial_06_lung/pca_mode_01_plus_2sigma.vtp",
    "tutorials/output/tutorial_07_lung/**/*",
    "tutorials/output/tutorial_08_lung/Case1Pack/Case1Pack_ssm_surface.vtp",
    "tutorials/output/tutorial_08_lung/Case1Pack/"
    "Case1Pack_ssm_pca_coefficients.json",
    "tutorials/output/tutorial_08_lung/Case1Pack/Case1Pack_T??_ssm_surface.vtp",
    "tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_ssm_surface.vtp",
    "tutorials/output/tutorial_08_lung/Case2Pack/"
    "Case2Pack_ssm_pca_coefficients.json",
    "tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_T??_ssm_surface.vtp",
    "tutorials/output/tutorial_08_lung/Case3Pack/Case3Pack_ssm_surface.vtp",
    "tutorials/output/tutorial_08_lung/Case3Pack/"
    "Case3Pack_ssm_pca_coefficients.json",
    "tutorials/output/tutorial_08_lung/Case3Pack/Case3Pack_T??_ssm_surface.vtp",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/" "mgn_stage_model.pt",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/"
    "mgn_stage_model_metadata.json",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/" "pca_mean_surface.vtp",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/" "pca_mean_template.vtp",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/pca_model.json",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/" "shared_edge_features.pt",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/" "shared_edge_index.pt",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/" "training_losses.json",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/"
    "training_validation_rmse.csv",
    "tutorials/network_weights/physicsnemo_mgn_lung_motion/"
    "training_validation_rmse.json",
    "tutorials/output/tutorial_11_lung/Case1Pack/ground_truth/**/*",
)

OFFLINE_SEGMENTATION_PATTERNS = (
    ".cache/monai-physio/huggingface/hub/"
    "models--nvidia--NV-Segment-CTMR/snapshots/"
    f"{NV_SEGMENT_REVISION}/*.py",
    ".cache/monai-physio/huggingface/hub/"
    "models--nvidia--NV-Segment-CTMR/snapshots/"
    f"{NV_SEGMENT_REVISION}/config.json",
    ".cache/monai-physio/huggingface/hub/"
    "models--nvidia--NV-Segment-CTMR/snapshots/"
    f"{NV_SEGMENT_REVISION}/metadata.json",
    ".cache/monai-physio/huggingface/hub/"
    "models--nvidia--NV-Segment-CTMR/snapshots/"
    f"{NV_SEGMENT_REVISION}/scripts/*.py",
    ".cache/monai-physio/huggingface/hub/"
    "models--nvidia--NV-Segment-CTMR/snapshots/"
    f"{NV_SEGMENT_REVISION}/vista3d_pretrained_model/config.json",
    ".cache/monai-physio/huggingface/hub/"
    "models--nvidia--NV-Segment-CTMR/snapshots/"
    f"{NV_SEGMENT_REVISION}/vista3d_pretrained_model/model.pt",
    ".cache/monai-physio/totalsegmentator/**/*",
)

PROFILE_PATTERNS = {
    "course": COURSE_PATTERNS,
    "offline-segmentation": OFFLINE_SEGMENTATION_PATTERNS,
}

IGNORED_PARTS = {"__MACOSX", "__pycache__"}
IGNORED_NAMES = {".DS_Store"}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repository_root: Path) -> str:
    """Return the repository HEAD commit, or ``unknown`` outside Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _is_ignored(path: Path) -> bool:
    """Return whether a cache artifact is packaging noise."""
    return (
        any(part in IGNORED_PARTS for part in path.parts)
        or path.name in IGNORED_NAMES
        or path.name.startswith("._")
    )


def collect_profile_files(repository_root: Path, profile: str) -> list[Path]:
    """Resolve the required files for ``profile`` under ``repository_root``."""
    try:
        patterns = PROFILE_PATTERNS[profile]
    except KeyError as error:
        raise ValueError(f"Unknown bundle profile: {profile}") from error

    files: set[Path] = set()
    missing_patterns = []
    for pattern in patterns:
        matches = [
            path
            for path in repository_root.glob(pattern)
            if (path.is_file() or path.is_symlink()) and not _is_ignored(path)
        ]
        if not matches:
            missing_patterns.append(pattern)
        files.update(matches)

    symlink_targets = {
        path.resolve()
        for path in files
        if path.is_symlink() and path.resolve().is_file()
    }
    files.update(
        path
        for path in symlink_targets
        if path == repository_root or repository_root in path.parents
    )

    if missing_patterns:
        missing = "\n  - ".join(missing_patterns)
        raise FileNotFoundError(
            f"Profile {profile!r} is incomplete; no files match:\n  - {missing}"
        )
    return sorted(files, key=lambda path: path.as_posix())


def _manifest_entry(path: Path, repository_root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(repository_root).as_posix()
    if path.is_symlink():
        return {
            "path": relative_path,
            "type": "symlink",
            "target": os.readlink(path),
        }
    return {
        "path": relative_path,
        "type": "file",
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": _sha256(path),
    }


def _normalize_tar_info(tar_info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Normalize ownership while retaining modes and modification times."""
    tar_info.uid = 0
    tar_info.gid = 0
    tar_info.uname = ""
    tar_info.gname = ""
    return tar_info


def build_profile_archive(
    repository_root: Path, output_directory: Path, profile: str
) -> dict[str, Any]:
    """Build one deterministic gzip-compressed tar archive and its metadata."""
    files = collect_profile_files(repository_root, profile)
    archive_name = f"physiomotion-workshop-{profile}.tar.gz"
    archive_path = output_directory / archive_name
    print(f"Building {profile}: {len(files)} files -> {archive_path}")

    with archive_path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            compresslevel=6,
            mtime=0,
        ) as gzip_file:
            with tarfile.open(
                fileobj=gzip_file,
                mode="w",
                format=tarfile.PAX_FORMAT,
                dereference=False,
            ) as archive:
                for path in files:
                    archive.add(
                        path,
                        arcname=path.relative_to(repository_root).as_posix(),
                        recursive=False,
                        filter=_normalize_tar_info,
                    )

    entries = [_manifest_entry(path, repository_root) for path in files]
    return {
        "archive": archive_name,
        "archive_size": archive_path.stat().st_size,
        "archive_sha256": _sha256(archive_path),
        "installed_size": sum(int(entry.get("size", 0)) for entry in entries),
        "files": entries,
    }


def build_bundles(
    repository_root: Path, output_directory: Path, profiles: Sequence[str]
) -> Path:
    """Build selected profiles and write their shared release manifest."""
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": _git_head(repository_root),
        "repo_id": DEFAULT_REPO_ID,
        "profiles": {},
    }
    for profile in profiles:
        manifest["profiles"][profile] = build_profile_archive(
            repository_root, output_directory, profile
        )

    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme_path = output_directory / "README.md"
    readme_path.write_text(
        "---\n"
        "license: other\n"
        "pretty_name: PhysioMotion Workshop Deployment Bundles\n"
        "---\n\n"
        "# PhysioMotion Workshop Deployment Bundles\n\n"
        "Private deployment artifacts for the MONAI Physio lung workshop. "
        "Use `utils/lung_bundle.py download`; do not redistribute the "
        "contained medical data or third-party model caches without confirming "
        "their source terms.\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_path}")
    return manifest_path


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported bundle manifest format: " f"{manifest.get('format_version')!r}"
        )
    return manifest


def _safe_member_destination(repository_root: Path, member_name: str) -> Path:
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    destination = repository_root.joinpath(*member_path.parts)
    resolved_root = repository_root.resolve()
    resolved_parent = destination.parent.resolve()
    if (
        resolved_parent != resolved_root
        and resolved_root not in resolved_parent.parents
    ):
        raise ValueError(f"Archive member escapes repository root: {member_name}")
    return destination


def _extract_archive(archive_path: Path, repository_root: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            destination = _safe_member_destination(repository_root, member.name)
            if member.ischr() or member.isblk() or member.isfifo():
                raise ValueError(f"Unsupported special archive member: {member.name}")
            if member.issym() or member.islnk():
                link_target = (destination.parent / member.linkname).resolve()
                resolved_root = repository_root.resolve()
                if (
                    link_target != resolved_root
                    and resolved_root not in link_target.parents
                ):
                    raise ValueError(f"Archive link escapes repository: {member.name}")
        try:
            archive.extractall(  # noqa: S202
                repository_root, filter="fully_trusted"
            )
        except TypeError:
            # Python 3.10 lacks the extraction filter argument. All members
            # have already passed the path, link, and special-file checks.
            archive.extractall(repository_root)  # noqa: S202


def _restore_profile_mtimes(
    repository_root: Path, manifest: dict[str, Any], profile: str
) -> None:
    """Restore exact file timestamps used by checkpoint-cache provenance."""
    for entry in manifest["profiles"][profile]["files"]:
        if entry["type"] != "file":
            continue
        path = repository_root / entry["path"]
        mtime_ns = int(entry["mtime_ns"])
        os.utime(path, ns=(mtime_ns, mtime_ns))


def verify_profiles(
    repository_root: Path,
    manifest: dict[str, Any],
    profiles: Sequence[str],
) -> None:
    """Verify installed files for selected profiles against the manifest."""
    for profile in profiles:
        try:
            entries = manifest["profiles"][profile]["files"]
        except KeyError as error:
            raise ValueError(f"Profile is absent from manifest: {profile}") from error
        print(f"Verifying {profile}: {len(entries)} files")
        for entry in entries:
            path = repository_root / entry["path"]
            if entry["type"] == "symlink":
                if not path.is_symlink() or os.readlink(path) != entry["target"]:
                    raise RuntimeError(f"Invalid installed symlink: {path}")
                continue
            if not path.is_file():
                raise RuntimeError(f"Missing installed file: {path}")
            if path.stat().st_size != entry["size"]:
                raise RuntimeError(f"Installed file has the wrong size: {path}")
            if path.stat().st_mtime_ns != entry["mtime_ns"]:
                raise RuntimeError(f"Installed file has the wrong timestamp: {path}")
            if _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"Installed file checksum mismatch: {path}")


def download_bundles(
    repository_root: Path,
    repo_id: str,
    revision: str,
    profiles: Sequence[str],
) -> Path:
    """Download, verify, extract, and verify selected bundle profiles."""
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
        _restore_profile_mtimes(repository_root, manifest, profile)

    verify_profiles(repository_root, manifest, profiles)
    installed_manifest = (
        repository_root / ".cache" / "monai-physio" / "bundles" / "manifest.json"
    )
    installed_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_download, installed_manifest)
    print(f"Installed profiles: {', '.join(profiles)}")
    return installed_manifest


def _profiles_from_args(values: Optional[list[str]]) -> list[str]:
    profiles = values or ["course"]
    invalid = sorted(set(profiles) - set(PROFILE_PATTERNS))
    if invalid:
        raise ValueError(f"Unknown bundle profiles: {', '.join(invalid)}")
    return list(dict.fromkeys(profiles))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build release archives")
    build_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    build_parser.add_argument("--output-directory", type=Path, required=True)
    build_parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILE_PATTERNS),
        help="Profile to build; repeat for multiple profiles (default: course)",
    )

    download_parser = subparsers.add_parser(
        "download", help="Download and install release archives"
    )
    download_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    download_parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    download_parser.add_argument("--revision", default=DEFAULT_REVISION)
    download_parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILE_PATTERNS),
        help="Profile to install; repeat for multiple profiles (default: course)",
    )

    verify_parser = subparsers.add_parser("verify", help="Verify installed files")
    verify_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILE_PATTERNS),
        help="Profile to verify; repeat for multiple profiles (default: course)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the bundle utility command line."""
    args = _parser().parse_args(argv)
    profiles = _profiles_from_args(args.profile)
    repository_root = args.repository_root.resolve()

    if args.command == "build":
        build_bundles(repository_root, args.output_directory.resolve(), profiles)
        return 0
    if args.command == "download":
        download_bundles(
            repository_root,
            repo_id=args.repo_id,
            revision=args.revision,
            profiles=profiles,
        )
        return 0
    if args.command == "verify":
        verify_profiles(repository_root, _load_manifest(args.manifest), profiles)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
