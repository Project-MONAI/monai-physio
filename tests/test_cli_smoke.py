"""CLI help smoke tests."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any, Optional

import pytest


CLI_MODULES = [
    "monai_physio.cli.convert_image_to_vtk",
    "monai_physio.cli.convert_image_4d_to_3d",
    "monai_physio.cli.convert_image_to_usd",
    "monai_physio.cli.convert_vtk_to_usd",
    "monai_physio.cli.create_statistical_model",
    "monai_physio.cli.download_data",
    "monai_physio.cli.fit_statistical_model_to_patient",
    "monai_physio.cli.reconstruct_highres_4d_ct",
    "monai_physio.cli.view_meshes",
    "monai_physio.cli.visualize_pca_modes",
]


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_cli_help(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each CLI module exits successfully for --help."""
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "usage:" in captured.out.lower()


def test_convert_image_to_usd_help_includes_fps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Image-to-USD CLI exposes playback FPS for animated USD output."""
    module = importlib.import_module("monai_physio.cli.convert_image_to_usd")
    monkeypatch.setattr(sys, "argv", ["convert_image_to_usd", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "--fps" in captured.out


def test_convert_image_to_usd_cli_passes_fps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Image-to-USD CLI forwards --fps as frames_per_second."""
    import monai_physio

    module = importlib.import_module("monai_physio.cli.convert_image_to_usd")
    input_file = tmp_path / "input.mha"
    input_file.write_text("placeholder")
    captured_kwargs: dict[str, Any] = {}
    fake_image = object()

    class FakeConvertImage4DTo3D:
        def load_image_4d(self, input_filename: str) -> None:
            assert input_filename == str(input_file)

        def get_3d_images(self) -> list[object]:
            return [fake_image]

    class FakeWorkflowConvertImageToUSD:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def process(self) -> str:
            return "output.usd"

    monkeypatch.setattr(
        module,
        "ConvertImage4DTo3D",
        FakeConvertImage4DTo3D,
    )
    monkeypatch.setattr(
        monai_physio,
        "WorkflowConvertImageToUSD",
        FakeWorkflowConvertImageToUSD,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_image_to_usd",
            str(input_file),
            "--output-dir",
            str(tmp_path),
            "--fps",
            "30",
        ],
    )

    assert module.main() == 0
    assert captured_kwargs["time_series_images"] == [fake_image]
    assert captured_kwargs["reference_image"] is fake_image
    assert captured_kwargs["frames_per_second"] == 30.0


def test_view_meshes_cli_forwards_server_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mesh viewer CLI forwards the scene and web-server options."""
    module = importlib.import_module("monai_physio.cli.view_meshes")
    usd_file = tmp_path / "animated.usd"
    usd_file.write_text("placeholder")
    captured: dict[str, Any] = {}

    class FakeMeshWebViewer:
        def __init__(
            self,
            input_files: list[Path],
            prim_path: str,
            playback_fps: Optional[float],
        ) -> None:
            captured["input_files"] = input_files
            captured["prim_path"] = prim_path
            captured["playback_fps"] = playback_fps

        def start(self, host: str, port: int, open_browser: bool) -> None:
            captured["host"] = host
            captured["port"] = port
            captured["open_browser"] = open_browser

    import monai_physio.mesh_web_viewer

    monkeypatch.setattr(
        monai_physio.mesh_web_viewer,
        "MeshWebViewer",
        FakeMeshWebViewer,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "view_meshes",
            str(usd_file),
            "--prim-path",
            "/Scene",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--fps",
            "3",
            "--no-browser",
        ],
    )

    assert module.main() == 0
    assert captured == {
        "input_files": [usd_file],
        "prim_path": "/Scene",
        "playback_fps": 3.0,
        "host": "0.0.0.0",
        "port": 9000,
        "open_browser": False,
    }


def test_view_meshes_cli_forwards_multiple_vtp_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Viewer CLI forwards all VTP surfaces in command-line order."""
    module = importlib.import_module("monai_physio.cli.view_meshes")
    patient_file = tmp_path / "patient.vtp"
    fitted_file = tmp_path / "fitted.vtp"
    patient_file.write_text("placeholder")
    fitted_file.write_text("placeholder")
    captured: dict[str, Any] = {}

    class FakeMeshWebViewer:
        def __init__(
            self,
            input_files: list[Path],
            prim_path: str,
            playback_fps: Optional[float],
        ) -> None:
            captured["input_files"] = input_files
            captured["prim_path"] = prim_path
            captured["playback_fps"] = playback_fps

        def start(self, host: str, port: int, open_browser: bool) -> None:
            captured["host"] = host
            captured["port"] = port
            captured["open_browser"] = open_browser

    import monai_physio.mesh_web_viewer

    monkeypatch.setattr(
        monai_physio.mesh_web_viewer,
        "MeshWebViewer",
        FakeMeshWebViewer,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "view_meshes",
            str(patient_file),
            str(fitted_file),
            "--no-browser",
        ],
    )

    assert module.main() == 0
    assert captured == {
        "input_files": [patient_file, fitted_file],
        "prim_path": "/World",
        "playback_fps": None,
        "host": "127.0.0.1",
        "port": 8080,
        "open_browser": False,
    }
