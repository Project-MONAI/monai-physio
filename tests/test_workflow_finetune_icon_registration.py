"""Unit tests for WorkflowFinetuneICONRegistration.

Exercises constructor validation, ``prepare_dataset`` / ``prepare_config``
file generation, mask derivation, and the ``process`` subprocess
launch.  Real uniGradICON training is not exercised here; the subprocess is
monkey-patched.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import itk
import numpy as np
import pytest
import yaml

from monai_physio.workflow_finetune_icon_registration import (
    WorkflowFinetuneICONRegistration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_image(path: Path, value: int = 1) -> None:
    """Write a 3x3x3 ``uint8`` ITK image with a single foreground voxel at center."""
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = value
    img = itk.image_from_array(arr)
    itk.imwrite(img, str(path), compression=True)


def _write_fake_checkpoint(workflow: WorkflowFinetuneICONRegistration) -> None:
    """Stand in for the checkpoint the monkey-patched subprocess never writes."""
    weights_path = workflow.expected_weights_path()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.touch()


@pytest.fixture
def two_subject_dataset(tmp_path: Path) -> dict[str, Any]:
    """Two patients, two frames each, with matching labelmaps and masks on disk."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "ft_out"

    subject_image_files: list[list[str]] = []
    subject_labelmap_files: list[list[Optional[str]]] = []
    subject_mask_files: list[list[Optional[str]]] = []
    for patient_id in ("pm0001", "pm0002"):
        pdir = data_dir / patient_id
        pdir.mkdir()
        images: list[str] = []
        segs: list[Optional[str]] = []
        masks: list[Optional[str]] = []
        for frame in ("g000", "g050"):
            image_path = pdir / f"{patient_id}_{frame}.nii.gz"
            label_path = pdir / f"{patient_id}_{frame}_labelmap.nii.gz"
            mask_path = pdir / f"{patient_id}_{frame}_mask.nii.gz"
            _make_image(image_path)
            _make_image(label_path)
            _make_image(mask_path)
            images.append(str(image_path))
            segs.append(str(label_path))
            masks.append(str(mask_path))
        subject_image_files.append(images)
        subject_labelmap_files.append(segs)
        subject_mask_files.append(masks)

    return {
        "output_dir": output_dir,
        "finetune_name": "test_exp",
        "subject_ids": ["pm0001", "pm0002"],
        "subject_image_files": subject_image_files,
        "subject_labelmap_files": subject_labelmap_files,
        "subject_mask_files": subject_mask_files,
    }


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_init_requires_output_dir_and_name(tmp_path: Path) -> None:
    """output_dir and finetune_name are required positional args."""
    with pytest.raises(TypeError):
        WorkflowFinetuneICONRegistration(  # type: ignore[call-arg]
            subject_image_files=[["a.nii.gz"]],
        )


def test_init_rejects_empty_image_files(tmp_path: Path) -> None:
    """Empty subject list raises immediately."""
    with pytest.raises(ValueError, match="must not be empty"):
        WorkflowFinetuneICONRegistration(
            subject_image_files=[],
            output_dir=tmp_path,
            finetune_name="x",
        )


def test_init_rejects_mismatched_companion_lengths(tmp_path: Path) -> None:
    """Mask/seg/landmark lists must match subject_image_files shape exactly."""
    with pytest.raises(ValueError, match="subject_mask_files\\[0\\] length"):
        WorkflowFinetuneICONRegistration(
            subject_image_files=[["a.nii.gz", "b.nii.gz"]],
            output_dir=tmp_path,
            finetune_name="x",
            subject_mask_files=[["m.nii.gz"]],
        )


def test_init_rejects_duplicate_subject_ids(tmp_path: Path) -> None:
    """Duplicate subject IDs collapse paired groups, so reject them up front."""
    with pytest.raises(ValueError, match="unique"):
        WorkflowFinetuneICONRegistration(
            subject_image_files=[["a"], ["b"]],
            output_dir=tmp_path,
            finetune_name="x",
            subject_ids=["same", "same"],
        )


def test_init_rejects_mismatched_subject_ids_length(tmp_path: Path) -> None:
    """subject_ids must have one entry per subject."""
    with pytest.raises(ValueError, match="subject_ids length"):
        WorkflowFinetuneICONRegistration(
            subject_image_files=[["a"]],
            output_dir=tmp_path,
            finetune_name="x",
            subject_ids=["a", "b"],
        )


def test_use_labelmaps_and_use_masks_flags(tmp_path: Path) -> None:
    """The two helper flags reflect supplied companions independently."""
    base: dict[str, Any] = {
        "subject_image_files": [["a"]],
        "output_dir": tmp_path,
        "finetune_name": "x",
    }
    none_wf = WorkflowFinetuneICONRegistration(**base)
    assert not none_wf.use_labelmaps
    assert not none_wf.use_masks

    seg_only = WorkflowFinetuneICONRegistration(
        **base, subject_labelmap_files=[["seg.nii.gz"]]
    )
    assert seg_only.use_labelmaps
    assert not seg_only.use_masks  # masks are never derived from segs

    mask_only = WorkflowFinetuneICONRegistration(
        **base, subject_mask_files=[["mask.nii.gz"]]
    )
    assert not mask_only.use_labelmaps
    assert mask_only.use_masks


# ---------------------------------------------------------------------------
# prepare_dataset
# ---------------------------------------------------------------------------


def test_prepare_dataset_uses_real_subject_ids(
    two_subject_dataset: dict[str, Any],
) -> None:
    """Subject IDs round-trip from the caller into every dataset entry."""
    workflow = WorkflowFinetuneICONRegistration(
        log_level=logging.CRITICAL, **two_subject_dataset
    )
    dataset_json_path = workflow.prepare_dataset()

    payload = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    entries = payload["data"]
    assert len(entries) == 4
    ids = {entry["subject_id"] for entry in entries}
    assert ids == {"pm0001", "pm0002"}
    for entry in entries:
        assert set(entry).issuperset({"image", "segmentation", "mask", "subject_id"})
        # Paths are forward-slashed for uniGradICON.
        assert "\\" not in entry["image"]
        assert "\\" not in entry["segmentation"]
        assert "\\" not in entry["mask"]


def test_prepare_dataset_skips_frames_with_missing_segmentation(
    tmp_path: Path,
) -> None:
    """A frame with no seg available is dropped when use_label is required."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    img_a = data_dir / "img_a.nii.gz"
    img_b = data_dir / "img_b.nii.gz"
    seg_a = data_dir / "seg_a.nii.gz"
    _make_image(img_a)
    _make_image(img_b)
    _make_image(seg_a)

    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=[[str(img_a), str(img_b)]],
        output_dir=tmp_path / "out",
        finetune_name="exp",
        subject_labelmap_files=[[str(seg_a), None]],
        log_level=logging.CRITICAL,
    )
    dataset_json_path = workflow.prepare_dataset()

    entries = json.loads(dataset_json_path.read_text(encoding="utf-8"))["data"]
    assert len(entries) == 1
    assert entries[0]["image"].endswith("img_a.nii.gz")


def test_prepare_dataset_uses_supplied_mask(tmp_path: Path) -> None:
    """Masks come straight from subject_mask_files; none are written to disk."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image = data_dir / "image.nii.gz"
    seg = data_dir / "seg.nii.gz"
    mask = data_dir / "explicit_mask.nii.gz"
    _make_image(image)
    _make_image(seg)
    _make_image(mask)

    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=[[str(image)]],
        output_dir=tmp_path / "out",
        finetune_name="exp",
        subject_labelmap_files=[[str(seg)]],
        subject_mask_files=[[str(mask)]],
        log_level=logging.CRITICAL,
    )
    dataset_json_path = workflow.prepare_dataset()
    entry = json.loads(dataset_json_path.read_text(encoding="utf-8"))["data"][0]

    assert entry["mask"].endswith("explicit_mask.nii.gz")
    assert entry["segmentation"].endswith("seg.nii.gz")
    assert sorted(p.name for p in data_dir.iterdir()) == [
        "explicit_mask.nii.gz",
        "image.nii.gz",
        "seg.nii.gz",
    ]


def test_prepare_dataset_mask_only_no_segmentations(tmp_path: Path) -> None:
    """Mask-only input: entries have ``mask`` but no ``segmentation`` field."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image = data_dir / "image.nii.gz"
    mask = data_dir / "mask.nii.gz"
    _make_image(image)
    _make_image(mask)

    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=[[str(image)]],
        output_dir=tmp_path / "out",
        finetune_name="exp",
        subject_mask_files=[[str(mask)]],
        log_level=logging.CRITICAL,
    )
    entry = json.loads(workflow.prepare_dataset().read_text(encoding="utf-8"))["data"][
        0
    ]
    assert "mask" in entry
    assert "segmentation" not in entry


def test_prepare_dataset_skips_frames_with_missing_mask(tmp_path: Path) -> None:
    """A frame with no mask is dropped when loss-function masking is required."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    img_a = data_dir / "img_a.nii.gz"
    img_b = data_dir / "img_b.nii.gz"
    mask_a = data_dir / "mask_a.nii.gz"
    _make_image(img_a)
    _make_image(img_b)
    _make_image(mask_a)

    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=[[str(img_a), str(img_b)]],
        output_dir=tmp_path / "out",
        finetune_name="exp",
        subject_mask_files=[[str(mask_a), None]],
        log_level=logging.CRITICAL,
    )
    entries = json.loads(workflow.prepare_dataset().read_text(encoding="utf-8"))["data"]
    assert len(entries) == 1
    assert entries[0]["image"].endswith("img_a.nii.gz")


def test_prepare_dataset_raises_on_missing_image_file(tmp_path: Path) -> None:
    """Image existence is a hard requirement; missing image aborts the build."""
    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=[[str(tmp_path / "does_not_exist.nii.gz")]],
        output_dir=tmp_path / "out",
        finetune_name="exp",
        log_level=logging.CRITICAL,
    )
    with pytest.raises(FileNotFoundError, match="Image not found"):
        workflow.prepare_dataset()


# ---------------------------------------------------------------------------
# prepare_config
# ---------------------------------------------------------------------------


def test_prepare_config_emits_uniGradICON_yaml(
    two_subject_dataset: dict[str, Any],
) -> None:
    """YAML config matches uniGradICON's expected structure when seg is present."""
    workflow = WorkflowFinetuneICONRegistration(
        log_level=logging.CRITICAL,
        epochs=10,
        batch_size=2,
        learning_rate=1e-4,
        input_shape=(64, 64, 64),
        gpus=[1],
        **two_subject_dataset,
    )
    dataset_json = workflow.prepare_dataset()
    config_yaml = workflow.prepare_config(dataset_json)

    config = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    assert config["experiment"]["model_weights"] == "unigradicon"
    assert config["experiment"]["name"].endswith("test_exp_model")
    training = config["training"]
    assert training["epochs"] == 10
    assert training["batch_size"] == 2
    assert training["learning_rate"] == 1e-4
    assert training["input_shape"] == [64, 64, 64]
    assert training["gpus"] == [1]
    # loss_function_masking is driven by data availability; use_label is always False.
    assert training["use_label"] is False
    assert training["loss_function_masking"] is True
    assert training["roi_masking"] is False

    dataset_cfg = config["datasets"][0]
    assert dataset_cfg["type"] == "paired"
    assert dataset_cfg["is_ct"] is True
    assert dataset_cfg["json_file"].endswith("test_exp_dataset.json")
    assert "\\" not in dataset_cfg["json_file"]


def test_prepare_config_flags_off_when_no_companions(tmp_path: Path) -> None:
    """Without seg or mask, ``use_label`` and ``loss_function_masking`` are False."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image = data_dir / "image.nii.gz"
    _make_image(image)

    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=[[str(image)]],
        output_dir=tmp_path / "out",
        finetune_name="exp",
        log_level=logging.CRITICAL,
    )
    dataset_json = workflow.prepare_dataset()
    config = yaml.safe_load(
        workflow.prepare_config(dataset_json).read_text(encoding="utf-8")
    )
    assert config["training"]["use_label"] is False
    assert config["training"]["loss_function_masking"] is False


def test_prepare_config_requires_dataset_json(tmp_path: Path) -> None:
    """Calling prepare_config without first preparing the dataset is an error."""
    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=[["a"]],
        output_dir=tmp_path,
        finetune_name="x",
        log_level=logging.CRITICAL,
    )
    with pytest.raises(ValueError, match="prepare_dataset"):
        workflow.prepare_config()


# ---------------------------------------------------------------------------
# expected_weights_path
# ---------------------------------------------------------------------------


def test_expected_weights_path_layout(tmp_path: Path) -> None:
    """Weights land at ``output_dir/<name>/<name>_model/checkpoints/...``."""
    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=[["a"]],
        output_dir=tmp_path,
        finetune_name="exp",
        log_level=logging.CRITICAL,
    )
    expected = workflow.expected_weights_path()
    assert expected == (
        tmp_path / "exp" / "exp_model" / "checkpoints" / "network_weights_final.trch"
    )

    # The filename must track uniGradICON's own checkpoint prefix, so an
    # upstream rename breaks this test rather than silently sending the
    # tutorials to a path that only ever holds stock weights.  The finetuning
    # submodule exists only on the feat-add-finetuning branch, so skip where
    # it is absent.
    finetune = pytest.importorskip("unigradicon.finetuning.finetune")
    assert expected.name == f"{finetune.NETWORK_WEIGHTS_PREFIX}_final.trch"


# ---------------------------------------------------------------------------
# process (subprocess is monkey-patched)
# ---------------------------------------------------------------------------


def test_process_invokes_unigradicon_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    two_subject_dataset: dict[str, Any],
) -> None:
    """process launches the uniGradICON finetune module with the YAML path."""
    captured: dict[str, Any] = {}

    unigradicon_src = two_subject_dataset["output_dir"].parent / "fake_unigradicon_src"
    workflow = WorkflowFinetuneICONRegistration(
        log_level=logging.CRITICAL,
        unigradicon_src_path=unigradicon_src,
        **two_subject_dataset,
    )

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["env"] = env
        _write_fake_checkpoint(workflow)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    weights = workflow.process()

    assert captured["check"] is True
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1:4] == ["-m", "unigradicon.finetuning.finetune", "--config"]
    yaml_arg = Path(captured["cmd"][4])
    assert yaml_arg.exists()
    assert yaml_arg.name == "test_exp_config.yaml"

    # Environment overrides.
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert str(unigradicon_src) in captured["env"]["PYTHONPATH"]

    assert weights == workflow.expected_weights_path()


def test_process_without_unigradicon_src(
    monkeypatch: pytest.MonkeyPatch,
    two_subject_dataset: dict[str, Any],
) -> None:
    """When unigradicon_src_path is None, PYTHONPATH is not prefixed."""

    workflow = WorkflowFinetuneICONRegistration(
        log_level=logging.CRITICAL,
        **two_subject_dataset,
    )

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        # No leading entry referencing a "fake" src tree.
        assert "fake_unigradicon_src" not in env.get("PYTHONPATH", "")
        _write_fake_checkpoint(workflow)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    workflow.process()


def test_process_raises_when_checkpoint_missing(
    monkeypatch: pytest.MonkeyPatch,
    two_subject_dataset: dict[str, Any],
) -> None:
    """A successful subprocess that wrote no checkpoint is an error.

    uniGradICON treats an unknown weights path as a download destination, so an
    unnoticed missing checkpoint silently degrades to stock weights.
    """

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    workflow = WorkflowFinetuneICONRegistration(
        log_level=logging.CRITICAL,
        **two_subject_dataset,
    )
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        workflow.process()
