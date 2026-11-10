"""CLI help smoke tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
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


@pytest.mark.parametrize("percentile", [None, "50"])
def test_reconstruction_cli_percentile(
    percentile: Optional[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI forwards default and custom percentiles to the workflow."""
    import itk
    import numpy as np

    from monai_physio import WorkflowReconstructHighres4DCT

    module = importlib.import_module("monai_physio.cli.reconstruct_highres_4d_ct")
    input_file = tmp_path / "input.mha"
    input_file.touch()
    image = itk.image_from_array(np.zeros((3, 3, 3), np.float32))
    monkeypatch.setattr(itk, "imread", lambda *args, **kwargs: image)

    def process(workflow: WorkflowReconstructHighres4DCT) -> dict[str, Any]:
        assert workflow.composite_mode == "percentile"
        assert workflow.composite_percentile == float(percentile or 95)
        return {"reconstructed_images": []}

    monkeypatch.setattr(WorkflowReconstructHighres4DCT, "process", process)
    args = [
        "reconstruct_highres_4d_ct",
        "--time-series-images",
        str(input_file),
        "--fixed-image",
        str(input_file),
        "--output-dir",
        str(tmp_path),
        "--registration-method",
        "Greedy",
        "--composite-mode",
        "percentile",
    ]
    if percentile is not None:
        args.extend(["--composite-percentile", percentile])
    monkeypatch.setattr(sys, "argv", args)
    assert module.main() == 0


@pytest.mark.parametrize("percentile", ["-1", "101", "nan", "inf"])
def test_reconstruction_cli_rejects_invalid_percentile(
    percentile: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid percentile arguments fail before loading any files."""
    module = importlib.import_module("monai_physio.cli.reconstruct_highres_4d_ct")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconstruct_highres_4d_ct",
            "--time-series-images",
            "frame.mha",
            "--fixed-image",
            "fixed.mha",
            "--output-dir",
            "results",
            "--composite-mode",
            "percentile",
            f"--composite-percentile={percentile}",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        module.main()
    assert exc_info.value.code == 2
    assert (
        "--composite-percentile must be finite and in [0, 100]"
        in capsys.readouterr().err
    )


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
