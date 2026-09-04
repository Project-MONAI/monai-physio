"""Synthetic tests for the PhysicsNeMo manifest and lazy sample dataset.

These cover the target contract — the manifest names a point-data array and the
dataset returns it verbatim — with no torch or PhysicsNeMo involvement, so they
run in the default fast suite.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from monai_physio import physicsnemo_tools as pnt

_TARGET_ARRAY = "displacement"
_STAGES = (0.0, 0.5)


def _sphere() -> pv.PolyData:
    """Small sphere shared by the reference and phase meshes."""
    return pv.Sphere(radius=10.0, theta_resolution=8, phi_resolution=8)


def _write_subject(tmp_path: Path, targets: list[np.ndarray]) -> Path:
    """Write a reference mesh, phase target meshes and a manifest; return it."""
    reference = _sphere()
    reference.save(str(tmp_path / "reference.vtp"))

    (tmp_path / "coefficients.json").write_text(
        json.dumps([0.5, -0.25]), encoding="utf-8"
    )

    phases = []
    for index, (stage, values) in enumerate(zip(_STAGES, targets)):
        phase_mesh = _sphere()
        phase_mesh.point_data[_TARGET_ARRAY] = values
        phase_file = tmp_path / f"phase_{index}.vtp"
        phase_mesh.save(str(phase_file))
        phases.append({"mesh": str(phase_file), "stage": stage})

    manifest = {
        "subject_id": "subject_01",
        "fitted_reference_mesh": str(tmp_path / "reference.vtp"),
        "pca_coefficients": str(tmp_path / "coefficients.json"),
        "target_array": _TARGET_ARRAY,
        "phases": phases,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _targets(n_points: int, n_target: int, offset: float) -> np.ndarray:
    """Deterministic, non-constant target values."""
    values = np.arange(n_points * n_target, dtype=np.float32).reshape(
        n_points, n_target
    )
    return values / 100.0 + offset


def test_parse_manifest_rejects_a_manifest_without_a_fitted_reference_mesh(
    tmp_path: Path,
) -> None:
    """The fitted mesh is what the displacements are defined against.

    Shape parameters alone do not reconstruct it, so a manifest that omits it
    has to fail rather than fall back to anything.
    """
    n_points = _sphere().n_points
    manifest_path = _write_subject(tmp_path, [_targets(n_points, 3, 0.0)])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["fitted_reference_mesh"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="fitted_reference_mesh"):
        pnt.parse_manifest(manifest_path)


def _without_physicsnemo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``physicsnemo`` import raise, as an install without the extra."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "physicsnemo" or name.startswith("physicsnemo."):
            raise ImportError("No module named 'physicsnemo'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.parametrize(
    "variable", ["WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"]
)
def test_multi_process_launch_without_physicsnemo_is_refused(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    """Eight ranks with no rank assigner is eight processes overwriting each other.

    The refusal has to happen here, before the training workflow creates its
    output directory, because by then the damage is already on disk.
    """
    for name in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "8")
    _without_physicsnemo(monkeypatch)

    with pytest.raises(ImportError, match="1 of 8"):
        pnt.distributed_context()


def test_a_single_process_without_physicsnemo_still_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MLP path never needed the extra, so one process is not an error."""
    for name in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    _without_physicsnemo(monkeypatch)

    context = pnt.distributed_context()

    assert context.world_size == 1
    assert context.rank == 0
    assert context.is_main
    assert not context.is_distributed


def test_parse_manifest_round_trips_the_new_schema(tmp_path: Path) -> None:
    n_points = _sphere().n_points
    manifest_path = _write_subject(
        tmp_path, [_targets(n_points, 3, 0.0), _targets(n_points, 3, 1.0)]
    )

    manifest = pnt.parse_manifest(manifest_path)

    assert manifest.subject_id == "subject_01"
    assert manifest.target_array == _TARGET_ARRAY
    assert manifest.fitted_reference_mesh.name == "reference.vtp"
    assert [phase.stage for phase in manifest.phases] == list(_STAGES)
    assert [phase.mesh.name for phase in manifest.phases] == [
        "phase_0.vtp",
        "phase_1.vtp",
    ]


def test_parse_manifest_requires_the_target_array(tmp_path: Path) -> None:
    n_points = _sphere().n_points
    manifest_path = _write_subject(tmp_path, [_targets(n_points, 3, 0.0)] * 2)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    del data["target_array"]
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="target_array"):
        pnt.parse_manifest(manifest_path)


@pytest.mark.parametrize("n_target", [1, 3, 6])
def test_load_target_array_returns_two_dimensional_targets(
    tmp_path: Path, n_target: int
) -> None:
    mesh = _sphere()
    values = _targets(mesh.n_points, n_target, 0.0)
    # A width-1 target is stored as a flat scalar array, as VTK writes it.
    mesh.point_data[_TARGET_ARRAY] = values[:, 0] if n_target == 1 else values
    mesh_file = tmp_path / "target.vtp"
    mesh.save(str(mesh_file))

    loaded = pnt.load_target_array(mesh_file, _TARGET_ARRAY)

    assert loaded.shape == (mesh.n_points, n_target)
    assert np.allclose(loaded, values)


def test_load_target_array_reports_missing_arrays(tmp_path: Path) -> None:
    mesh_file = tmp_path / "no_target.vtp"
    _sphere().save(str(mesh_file))

    with pytest.raises(KeyError, match=_TARGET_ARRAY):
        pnt.load_target_array(mesh_file, _TARGET_ARRAY)


def test_dataset_returns_the_stored_targets_scaled(tmp_path: Path) -> None:
    """The dataset must return stored targets, never derive them from geometry."""
    n_points = _sphere().n_points
    stored = [_targets(n_points, 3, 0.0), _targets(n_points, 3, 1.0)]
    manifest_path = _write_subject(tmp_path, stored)
    manifest = pnt.parse_manifest(manifest_path)

    target_scale = 2.0
    coords_norm = np.zeros((n_points, 3), dtype=np.float32)
    samples = [
        pnt._Sample(
            subject_id=manifest.subject_id,
            pca_norm=np.array([0.5, -0.25], dtype=np.float32),
            target_mesh=phase.mesh,
            stage=phase.stage,
        )
        for phase in manifest.phases
    ]
    dataset = pnt.PhaseSampleDataset(
        samples, coords_norm, manifest.target_array, target_scale
    )

    assert len(dataset) == len(stored)
    assert dataset.n_points == n_points
    assert dataset.n_target == 3
    assert dataset.n_features == 3 + 2 + 1

    for index, expected in enumerate(stored):
        node_feats, target = dataset[index]
        assert node_feats.shape == (n_points, dataset.n_features)
        assert np.allclose(node_feats[:, -1], _STAGES[index])
        assert np.allclose(target, expected / target_scale)


def test_mesh_to_edge_index_preserves_volumetric_point_ids(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    volume = pv.UnstructuredGrid(pv.Box().triangulate().delaunay_3d())

    edge_index = pnt.mesh_to_edge_index(volume)

    assert isinstance(edge_index, torch.Tensor)
    assert edge_index.shape[0] == 2
    assert int(edge_index.max()) < volume.n_points
    # Undirected: every edge appears in both directions.
    edges = {(int(a), int(b)) for a, b in edge_index.t().tolist()}
    assert all((b, a) in edges for a, b in edges)
