"""Tests for the dataset download CLI wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pytest

from monai_physio.cli import download_data
from monai_physio.download_data import DownloadData


def test_download_data_cli_with_no_args_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No positional argument prints usage/help instead of downloading anything."""
    result = download_data.main([])

    assert result == 1
    assert "usage:" in capsys.readouterr().out


def test_download_data_cli_uses_requested_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The --directory option controls where Slicer-Heart-CT is stored."""
    calls: list[Path] = []

    def fake_download(dirname: Union[str, Path]) -> Path:
        calls.append(Path(dirname))
        return Path(dirname) / DownloadData.SLICER_HEART_CT_FILENAME

    monkeypatch.setattr(DownloadData, "DownloadSlicerHeartCTData", fake_download)

    result = download_data.main(["Slicer-Heart-CT", "--directory", str(tmp_path)])

    assert result == 0
    assert calls == [tmp_path]


def test_download_data_cli_routes_kcl_heart_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """KCL-Heart-Model routes to its own downloader and default directory."""
    calls: list[Path] = []

    def fake_download(dirname: Union[str, Path]) -> Path:
        calls.append(Path(dirname))
        return Path(dirname)

    monkeypatch.setattr(DownloadData, "DownloadKCLHeartModelData", fake_download)

    result = download_data.main(["KCL-Heart-Model"])

    assert result == 0
    assert calls == [Path("data/KCL-Heart-Model")]
    assert "Downloaded KCL-Heart-Model" in capsys.readouterr().out


def test_download_data_cli_routes_chop_valve4d(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CHOP-Valve4D routes to its own downloader and default directory."""
    calls: list[Path] = []

    def fake_download(dirname: Union[str, Path]) -> Path:
        calls.append(Path(dirname))
        return Path(dirname)

    monkeypatch.setattr(DownloadData, "DownloadCHOPValve4DData", fake_download)

    result = download_data.main(["CHOP-Valve4D"])

    assert result == 0
    assert calls == [Path("data/CHOP-Valve4D")]
    assert "Downloaded CHOP-Valve4D" in capsys.readouterr().out


def test_download_data_cli_routes_chest_ct(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Chest-CT routes to its own downloader and default directory."""
    calls: list[Path] = []

    def fake_download(dirname: Union[str, Path]) -> Path:
        calls.append(Path(dirname))
        return Path(dirname) / DownloadData.CHEST_CT_FILENAME

    monkeypatch.setattr(DownloadData, "DownloadChestCTData", fake_download)

    result = download_data.main(["Chest-CT"])

    assert result == 0
    assert calls == [Path("data/Chest-CT")]
    assert "Downloaded Chest-CT" in capsys.readouterr().out


def test_download_data_cli_routes_tcia_4d_lung(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """TCIA-4DLung routes to its own downloader and default directory."""
    calls: list[Path] = []

    def fake_download(dirname: Union[str, Path]) -> Path:
        calls.append(Path(dirname))
        return Path(dirname)

    monkeypatch.setattr(DownloadData, "DownloadTCIA4DLungData", fake_download)

    result = download_data.main(["TCIA-4DLung"])

    assert result == 0
    assert calls == [Path("data/TCIA-4DLung")]
    assert "Downloaded TCIA-4DLung" in capsys.readouterr().out


def test_download_data_cli_routes_duke_heart_4d_labelmaps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Duke-Heart-4DLabelmaps routes to its own downloader and default directory."""
    calls: list[Path] = []

    def fake_download(dirname: Union[str, Path]) -> Path:
        calls.append(Path(dirname))
        return Path(dirname)

    monkeypatch.setattr(DownloadData, "DownloadDukeHeart4DLabelmapsData", fake_download)

    result = download_data.main(["Duke-Heart-4DLabelmaps"])

    assert result == 0
    assert calls == [Path("data/Duke-Heart-4DLabelmaps")]
    assert "Downloaded Duke-Heart-4DLabelmaps" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("data_name", "method_name"),
    [
        ("PhysicsNeMo-MGN-Lung-Motion", "DownloadPhysicsNeMoMGNLungMotionData"),
        (
            "PhysicsNeMo-MGN-Duke-Heart-Motion",
            "DownloadPhysicsNeMoMGNDukeHeartMotionData",
        ),
        (
            "PhysicsNeMo-PhysicsInformed-Motion-Duke-Heart",
            "DownloadPhysicsNeMoPhysicsInformedMotionDukeHeartData",
        ),
        (
            "PhysicsNeMo-PhysicsInformed-Motion-Duke-Heart-Ablation",
            "DownloadPhysicsNeMoPhysicsInformedMotionDukeHeartAblationData",
        ),
    ],
)
def test_download_data_cli_routes_network_weights(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    data_name: str,
    method_name: str,
) -> None:
    """Each pretrained network routes to its downloader and the weights root."""
    calls: list[Path] = []

    def fake_download(dirname: Union[str, Path]) -> Path:
        calls.append(Path(dirname))
        return Path(dirname)

    monkeypatch.setattr(DownloadData, method_name, fake_download)

    result = download_data.main([data_name])

    assert result == 0
    assert calls == [Path("tutorials/network_weights")]
    assert f"Downloaded {data_name}" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("download_name", "verify_name", "weights_dirname", "checkpoint"),
    [
        (
            "DownloadPhysicsNeMoMGNDukeHeartMotionData",
            "VerifyPhysicsNeMoMGNDukeHeartMotionData",
            "physicsnemo_mgn_duke_heart_motion",
            "mgn_stage_model.pt",
        ),
        (
            "DownloadPhysicsNeMoPhysicsInformedMotionDukeHeartData",
            "VerifyPhysicsNeMoPhysicsInformedMotionDukeHeartData",
            "physicsnemo_physics_informed_motion_duke_heart",
            "physics_informed_motion_stage_model.pt",
        ),
        (
            "DownloadPhysicsNeMoPhysicsInformedMotionDukeHeartAblationData",
            "VerifyPhysicsNeMoPhysicsInformedMotionDukeHeartAblationData",
            "physicsnemo_physics_informed_motion_duke_heart_ablation",
            "physics_informed_motion_stage_model.pt",
        ),
    ],
)
def test_download_network_weights_skips_existing_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    download_name: str,
    verify_name: str,
    weights_dirname: str,
    checkpoint: str,
) -> None:
    """A present checkpoint verifies and is never re-downloaded over."""
    verify = getattr(DownloadData, verify_name)
    assert not verify(tmp_path)

    (tmp_path / weights_dirname).mkdir()
    (tmp_path / weights_dirname / checkpoint).touch()
    assert verify(tmp_path)

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("existing checkpoint was re-downloaded")

    monkeypatch.setattr(DownloadData, "_DownloadAndExtractZip", fail_download)
    assert getattr(DownloadData, download_name)(tmp_path) == tmp_path / weights_dirname
