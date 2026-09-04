"""Synthetic training run covering what a checkpoint needs beside it.

A long training run is evaluated from its intermittent checkpoints while it is
still going, so everything inference reads from the model directory -- the
template mesh, the shared graph tensors, the PCA assets and the metadata -- has
to be on disk by the time the first of those checkpoints is written, not only
after the last epoch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import pyvista as pv

pytest.importorskip("torch")
pytest.importorskip("physicsnemo")
pytest.importorskip("torch_geometric")

from monai_physio import (  # noqa: E402
    DistributedContext,
    TrainPhysicsNeMoMGN,
    WorkflowInferPhysicsNeMo,
    WorkflowTrainPhysicsNeMo,
)
from monai_physio.physicsnemo_tools import uncompiled_state_dict  # noqa: E402

_TARGET_ARRAY = "displacement"
_STAGES = (0.0, 1.0)


class _IndexDataset:
    """Stands in for PhaseSampleDataset, one identifiable row per sample.

    Each sample is a single node feature row carrying its own index, so a
    batch's first column is the list of samples that went into it.
    """

    def __init__(self, n_samples: int) -> None:
        self._n_samples = n_samples

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full((1, 2), index, dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )


class _FakeDDP:
    """The one attribute of DistributedDataParallel checkpointing cares about.

    Constructing the real thing needs an initialized process group, which a
    unit test has no reason to stand up just to check a prefix.
    """

    def __init__(self, module: Any) -> None:
        self.module = module


def _sphere() -> pv.PolyData:
    """Small sphere shared by the template, reference and phase meshes."""
    return pv.Sphere(radius=10.0, theta_resolution=8, phi_resolution=8)


def _write_subject(subject_id: str, directory: Path, offset: float) -> Path:
    """Write one subject's reference mesh, phase targets and manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    reference = _sphere()
    reference.save(str(directory / "reference.vtp"))
    (directory / "coefficients.json").write_text(
        json.dumps([offset, -offset]), encoding="utf-8"
    )

    phases = []
    for index, stage in enumerate(_STAGES):
        phase_mesh = _sphere()
        phase_mesh.point_data[_TARGET_ARRAY] = np.full(
            (phase_mesh.n_points, 3), offset + stage, dtype=np.float32
        )
        phase_file = directory / f"phase_{index}.vtp"
        phase_mesh.save(str(phase_file))
        phases.append({"mesh": str(phase_file), "stage": stage})

    manifest_path = directory / f"{subject_id}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "subject_id": subject_id,
                "fitted_reference_mesh": str(directory / "reference.vtp"),
                "pca_coefficients": str(directory / "coefficients.json"),
                "target_array": _TARGET_ARRAY,
                "phases": phases,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_template(directory: Path) -> Path:
    """Write the PCA template mesh and the model JSON that sits beside it."""
    directory.mkdir(parents=True, exist_ok=True)
    template = _sphere()
    template_file = directory / "pca_mean_surface.vtp"
    template.save(str(template_file))
    (directory / "pca_model.json").write_text(
        json.dumps(
            {
                "mean": np.asarray(template.points, dtype=np.float64).ravel().tolist(),
                "components": np.zeros((2, template.n_points * 3)).tolist(),
            }
        ),
        encoding="utf-8",
    )
    return template_file


class _RecordingMGN(TrainPhysicsNeMoMGN):
    """MeshGraphNet method that lists the model directory at each checkpoint."""

    def __init__(self, model_directory: Path) -> None:
        super().__init__()
        self.model_directory = model_directory
        self.snapshots: list[set[str]] = []

    def build_checkpoint(self, model: Any, stats: dict) -> dict[str, Any]:
        self.snapshots.append({path.name for path in self.model_directory.iterdir()})
        return cast("dict[str, Any]", super().build_checkpoint(model, stats))


def _train(tmp_path: Path) -> tuple[Path, _RecordingMGN]:
    """Run two epochs over two synthetic subjects; return the model directory."""
    template_file = _write_template(tmp_path / "template")
    manifests = [
        _write_subject(
            f"subject_{index:02d}", tmp_path / f"subject_{index:02d}", offset
        )
        for index, offset in enumerate((0.5, -0.5))
    ]
    model_directory = tmp_path / "weights"

    method = _RecordingMGN(model_directory)
    method.set_epochs(2)
    method.set_batch_size(1)
    method.set_processor_size(1)
    method.set_hidden_dim(8)
    method.set_num_layers(1)
    # Every epoch, so the first intermittent checkpoint lands in epoch one.
    method.rmse_log_interval = 1

    workflow = WorkflowTrainPhysicsNeMo(
        train_manifests=manifests,
        val_manifests=[],
        pca_mean_mesh=template_file,
        output_directory=model_directory,
        training_method=method,
    )
    workflow.process()
    return model_directory, method


def test_first_checkpoint_has_its_companions(tmp_path: Path) -> None:
    """Inference's inputs are on disk before the first checkpoint is written."""
    _, method = _train(tmp_path)

    assert len(method.snapshots) > 1, "expected intermittent checkpoints, not just one"
    assert {
        "pca_mean_template.vtp",
        "pca_mean_surface.vtp",
        "pca_model.json",
        "shared_edge_index.pt",
        "shared_edge_features.pt",
        "mgn_stage_model_metadata.json",
    } <= method.snapshots[0]


def test_an_intermittent_checkpoint_can_be_inferred_from(tmp_path: Path) -> None:
    """The model directory loads at an epoch, not only at the final weights."""
    model_directory, _ = _train(tmp_path)

    infer = WorkflowInferPhysicsNeMo(model_directory=model_directory, epoch=1)
    targets = infer.predict(np.array([0.5, -0.5], dtype=np.float32), stage=0.5)

    assert targets.shape == (infer.template_mesh.n_points, 3)
    assert np.all(np.isfinite(targets))


def test_ranks_split_the_samples_without_overlapping() -> None:
    """Every sample lands on exactly one rank, and the ranks take equal steps.

    A rank that yielded one batch more than its peers would hang them all at
    the gradient all-reduce of the step they never take, so the equal-length
    assertion below is the one that keeps a distributed run from deadlocking.
    """
    import torch

    method = TrainPhysicsNeMoMGN()
    method.set_batch_size(2)
    dataset = cast(Any, _IndexDataset(11))
    world_size = 3

    per_rank = []
    for rank in range(world_size):
        context = DistributedContext(
            device=torch.device("cpu"),
            rank=rank,
            local_rank=rank,
            world_size=world_size,
        )
        batches = list(
            method._iter_batches(
                dataset, np.random.default_rng(0), shuffle=True, context=context
            )
        )
        per_rank.append([int(value) for batch in batches for value in batch[0][:, 0]])

    assert all(len(indices) == len(per_rank[0]) for indices in per_rank)
    # 11 samples over 3 ranks at batch_size 2 is one whole batch each; the
    # remainder is dropped rather than handed to whichever rank happens to
    # have it.
    assert len(per_rank[0]) == 2
    seen = [index for indices in per_rank for index in indices]
    assert len(seen) == len(set(seen))


def test_one_rank_iterates_the_whole_dataset() -> None:
    """Without a distributed context, nothing is sharded and nothing is dropped."""
    method = TrainPhysicsNeMoMGN()
    method.set_batch_size(2)
    dataset = cast(Any, _IndexDataset(11))

    batches = list(
        method._iter_batches(dataset, np.random.default_rng(0), shuffle=True)
    )
    seen = sorted(int(value) for batch in batches for value in batch[0][:, 0])

    assert seen == list(range(11))


def test_a_ddp_wrapped_model_checkpoints_without_its_prefix() -> None:
    """``module.`` never reaches a checkpoint, so inference loads it unchanged."""
    import torch

    inner = torch.nn.Linear(2, 2)
    wrapped = _FakeDDP(inner)

    state = uncompiled_state_dict(wrapped)

    assert set(state) == set(inner.state_dict())
    assert not any(key.startswith("module.") for key in state)
