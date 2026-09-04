"""Tests for the dataset download CLI wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pytest

from monai_physio.cli import download_data
from monai_physio.data_download_tools import DataDownloadTools


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
        return Path(dirname) / DataDownloadTools.SLICER_HEART_CT_FILENAME

    monkeypatch.setattr(DataDownloadTools, "DownloadSlicerHeartCTData", fake_download)

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

    monkeypatch.setattr(DataDownloadTools, "DownloadKCLHeartModelData", fake_download)

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

    monkeypatch.setattr(DataDownloadTools, "DownloadCHOPValve4DData", fake_download)

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
        return Path(dirname) / DataDownloadTools.CHEST_CT_FILENAME

    monkeypatch.setattr(DataDownloadTools, "DownloadChestCTData", fake_download)

    result = download_data.main(["Chest-CT"])

    assert result == 0
    assert calls == [Path("data/Chest-CT")]
    assert "Downloaded Chest-CT" in capsys.readouterr().out
