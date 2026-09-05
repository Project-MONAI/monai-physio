"""Shared helpers for the PhysicsNeMo train / infer workflows.

This module holds the pieces common to the MeshGraphNet (MGN) and fully
connected (MLP) PhysicsNeMo workflows so the workflow classes stay focused on
orchestration.  It provides:

- :class:`SubjectManifest` / :func:`parse_manifest` - the per-subject JSON
  manifest that lists a fitted reference mesh, a PCA shape-parameter file, the
  name of the point-data array holding the training targets, and the phase
  meshes that carry that array with their stages.
- :func:`load_target_array` - read one phase's ``(n_points, n_target)`` target
  values out of a mesh's point data.
- :func:`build_node_features` - the shared per-vertex feature layout
  ``[mean_coords_norm, pca_norm (tiled), stage]`` used by both networks.
- :func:`mesh_to_edge_index` / :func:`compute_edge_features` - MGN mesh-graph
  construction from the shared template mesh (surface or volumetric).
- :func:`uncompiled_state_dict` / :func:`strip_compile_prefix` - checkpoint I/O
  that is robust to ``torch.compile`` and ``DistributedDataParallel`` wrapping.
- :class:`DistributedContext` / :func:`distributed_context` - the rank, device
  and world size of the current process, from PhysicsNeMo's
  ``DistributedManager``.  A run started without a launcher gets world size 1,
  so single-process callers need no distributed-specific code.
- :class:`PhaseSampleDataset` - a lazy ``(subject, phase)`` sample provider with
  a bounded in-RAM cache so the training set need not fit in memory.

Targets are whatever the caller stored in the manifest's ``target_array``: the
stack never computes them.  A displacement model is one application of that -
the caller writes ``phase.points - reference.points`` into the array - but any
per-point vector of any width works the same way.

``torch`` and ``torch_geometric`` are base dependencies, but every function
that needs them still imports them locally to keep ``import monai_physio``
lightweight.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyvista as pv

if TYPE_CHECKING:  # imported lazily at runtime; typed here for mypy only
    import torch
    from physicsnemo.models.meshgraphnet import MeshGraphNet


# --------------------------------------------------------------------------- #
# MeshGraphNet import guard                                                    #
# --------------------------------------------------------------------------- #
def import_meshgraphnet() -> type[MeshGraphNet]:
    """Import PhysicsNeMo's ``MeshGraphNet``, reporting install faults clearly.

    PhysicsNeMo builds MeshGraphNet on ``torch_scatter``, a compiled extension
    whose prebuilt wheels are published only for specific
    ``(torch, CUDA, Python, platform)`` combinations. A wheel built against a
    different torch than the installed one loads as an opaque
    ``OSError: Could not load this library: ..._scatter_cuda.pyd``, which names
    neither torch_scatter nor the version mismatch that caused it. Translate
    that into a message that does.

    Returns:
        The ``physicsnemo.models.meshgraphnet.MeshGraphNet`` class.

    Raises:
        ImportError: PhysicsNeMo or PyTorch Geometric could not be imported, or
            the installed ``torch_scatter`` binary does not match the installed
            torch.
    """
    try:
        import torch_geometric  # noqa: F401 - needed by the graph seams
        from physicsnemo.models.meshgraphnet import MeshGraphNet
    except OSError as exc:
        if "scatter" not in str(exc).lower():
            raise ImportError(
                f"Failed to import MeshGraphNet's dependencies: {exc}"
            ) from exc
        import torch

        raise ImportError(
            "torch_scatter failed to load its compiled extension. It was "
            "built against a different torch than the installed torch "
            f"{torch.__version__}. Reinstall a torch_scatter matching this "
            "torch, or install monai-physio[cuda12], whose torch pin stays "
            "inside the range with prebuilt torch_scatter wheels. See the "
            "installation guide for the platform-by-platform wheel matrix."
        ) from exc
    except ModuleNotFoundError as exc:
        raise ImportError(
            "MeshGraphNet requires PhysicsNeMo and PyTorch Geometric, which "
            "are base dependencies of monai-physio. Reinstall with: "
            "pip install --force-reinstall monai-physio"
        ) from exc
    return cast("type[MeshGraphNet]", MeshGraphNet)


# --------------------------------------------------------------------------- #
# Per-subject manifest                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class PhaseEntry:
    """One phase mesh carrying the target array, and its normalized stage."""

    mesh: Path
    stage: float


@dataclass
class SubjectManifest:
    """A single subject's training/inference inputs.

    Attributes:
        subject_id: Identifier used for output naming.
        fitted_reference_mesh: The subject's SSM surface as fitted to that
            subject by
            :class:`monai_physio.WorkflowFitStatisticalModelToPatient` (``.vtp``
            surface or ``.vtu`` volume) --- shape parameters *and* a deformable
            registration, never shape parameters alone. It supplies the point
            positions the targets are defined at; the stack never derives
            targets from it.
        pca_coefficients: JSON file holding the subject's PCA shape-parameter
            vector (a flat list of floats).
        target_array: Name of the point-data array holding the target values in
            every phase mesh.
        phases: One :class:`PhaseEntry` per phase (at least one).
    """

    subject_id: str
    fitted_reference_mesh: Path
    pca_coefficients: Path
    target_array: str
    phases: list[PhaseEntry]


def parse_manifest(manifest_path: Path) -> SubjectManifest:
    """Parse a per-subject JSON manifest.

    Paths inside the manifest are resolved relative to the manifest's own
    directory unless already absolute.  Every phase must declare a ``stage``.

    Args:
        manifest_path: Path to the subject manifest JSON file.

    Returns:
        The parsed :class:`SubjectManifest`.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If required fields are missing, a phase lacks ``stage``, or
            no phases are listed.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent

    def _resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (base / p)

    for key in (
        "subject_id",
        "fitted_reference_mesh",
        "pca_coefficients",
        "target_array",
        "phases",
    ):
        if key not in data:
            raise ValueError(f"Manifest {manifest_path} is missing '{key}'.")

    raw_phases = data["phases"]
    if not raw_phases:
        raise ValueError(f"Manifest {manifest_path} lists no phases.")

    phases: list[PhaseEntry] = []
    for entry in raw_phases:
        if "mesh" not in entry or "stage" not in entry:
            raise ValueError(
                f"Manifest {manifest_path} has a phase missing 'mesh' or "
                "'stage' (stage must be supplied by the caller)."
            )
        phases.append(
            PhaseEntry(mesh=_resolve(entry["mesh"]), stage=float(entry["stage"]))
        )

    return SubjectManifest(
        subject_id=str(data["subject_id"]),
        fitted_reference_mesh=_resolve(data["fitted_reference_mesh"]),
        pca_coefficients=_resolve(data["pca_coefficients"]),
        target_array=str(data["target_array"]),
        phases=phases,
    )


def load_pca_coefficients(path: Path) -> np.ndarray:
    """Load a PCA shape-parameter vector saved as a JSON list of floats."""
    return np.asarray(
        json.loads(Path(path).read_text(encoding="utf-8")), dtype=np.float32
    )


def load_target_array(path: Path, array_name: str) -> np.ndarray:
    """Read one mesh's target values out of its point data.

    Args:
        path: Mesh holding the targets (``.vtp`` surface or ``.vtu`` volume).
        array_name: Point-data array name declared by the manifest.

    Returns:
        ``(n_points, n_target)`` float32 targets; a scalar array is returned as
        ``(n_points, 1)``.

    Raises:
        KeyError: If ``array_name`` is not among the mesh's point-data arrays.
    """
    mesh = pv.read(str(path))
    if array_name not in mesh.point_data:
        raise KeyError(
            f"{path} has no point-data array '{array_name}'; available arrays: "
            f"{sorted(mesh.point_data.keys())}"
        )
    values = np.asarray(mesh.point_data[array_name], dtype=np.float32)
    return values.reshape(len(values), -1)


# --------------------------------------------------------------------------- #
# Feature construction (shared by MGN and MLP)                                 #
# --------------------------------------------------------------------------- #
def build_node_features(
    mean_coords_norm: np.ndarray, pca_norm: np.ndarray, stage: float
) -> np.ndarray:
    """Assemble per-vertex node features ``[coords_norm, pca_norm, stage]``.

    Args:
        mean_coords_norm: ``(n_points, 3)`` normalized mean-shape coordinates
            (identical for every subject/phase).
        pca_norm: ``(n_pca,)`` normalized PCA shape parameters for the subject.
        stage: Normalized cardiac stage (RR-interval fraction) for the phase.

    Returns:
        ``(n_points, 3 + n_pca + 1)`` float32 feature array.
    """
    n = len(mean_coords_norm)
    pca_tile = np.tile(pca_norm, (n, 1))
    stage_col = np.full((n, 1), stage, dtype=np.float32)
    return np.hstack([mean_coords_norm, pca_tile, stage_col]).astype(np.float32)


# --------------------------------------------------------------------------- #
# MGN mesh-graph construction                                                  #
# --------------------------------------------------------------------------- #
def mesh_to_edge_index(mesh: pv.DataSet) -> "torch.Tensor":
    """Build an undirected ``edge_index`` from a surface or volumetric mesh.

    Args:
        mesh: Template mesh whose cells encode the topology. ``pv.PolyData`` is
            read straight from its triangulated faces; any other dataset (a
            volumetric ``pv.UnstructuredGrid``, for example) goes through
            ``extract_all_edges``.

    Returns:
        ``(2, n_edges)`` long tensor of undirected edges indexing the mesh's own
        points.

    Raises:
        ValueError: If edge extraction renumbers the points, which would break
            the correspondence between node features and graph nodes.
    """
    import torch
    import torch_geometric.utils as pyg_utils

    if isinstance(mesh, pv.PolyData):
        # A surface may carry quads/other polygons; triangulate (fan) so every
        # face is a triangle. vtkTriangleFilter reuses the existing vertices, so
        # the point ordering (and thus edge-index correspondence) is preserved.
        tri = mesh.triangulate()
        faces = tri.faces.reshape(-1, 4)[:, 1:]  # (F, 3) - strip leading count
        src = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        dst = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    else:
        # extract_all_edges keeps every input point in the output, so the line
        # connectivity indexes the original point ids.  The check below is what
        # holds that; the guarantee is not stated in the pyvista contract.
        edges = mesh.extract_all_edges(clear_data=True)
        if edges.n_points != mesh.n_points:
            raise ValueError(
                f"Edge extraction returned {edges.n_points} points for a mesh "
                f"with {mesh.n_points}; the point ids were renumbered."
            )
        lines = edges.lines.reshape(-1, 3)[:, 1:]  # (E, 2) - strip leading count
        src, dst = lines[:, 0], lines[:, 1]

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return cast("torch.Tensor", pyg_utils.to_undirected(edge_index))


def compute_edge_features(
    coords: np.ndarray, edge_index: "torch.Tensor"
) -> "torch.Tensor":
    """Build ``(n_edges, 4)`` edge features ``[rel_x, rel_y, rel_z, distance]``."""
    import torch

    ei = edge_index.numpy()
    disp = coords[ei[1]] - coords[ei[0]]
    dist = np.linalg.norm(disp, axis=1, keepdims=True)
    return torch.tensor(np.hstack([disp, dist]), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Distributed context                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class DistributedContext:
    """Rank, device and world size of the current process.

    ``world_size == 1`` covers both a run started without a launcher and a
    single-GPU run under one, so callers branch on that rather than on how the
    process was started.
    """

    device: "torch.device"
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        """True on the one rank that owns writing to disk and logging."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """True when gradients have to be synchronized across ranks."""
        return self.world_size > 1

    def barrier(self) -> None:
        """Block until every rank arrives; a no-op outside a distributed run."""
        if not self.is_distributed:
            return
        import torch

        torch.distributed.barrier()


def _launched_world_size() -> int:
    """How many processes the launcher says are in this job, 1 if none says.

    Read straight from the environment rather than from PhysicsNeMo, because
    this is what has to be known *before* PhysicsNeMo is known to be there.
    The three variables are the ones ``DistributedManager`` itself consults,
    for ``torchrun``, SLURM and OpenMPI respectively.
    """
    sizes = [
        int(os.environ[name])
        for name in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE")
        if os.environ.get(name, "").isdigit()
    ]
    return max(sizes, default=1)


def distributed_context() -> DistributedContext:
    """Initialize PhysicsNeMo's ``DistributedManager`` and read it back.

    ``DistributedManager.initialize`` picks up ``torchrun``, SLURM or OpenMPI
    environments and falls back to a single process when it finds none, so this
    is safe to call from any entry point.  Initialization is done once per
    process; calling this again returns the same context.

    Raises:
        ImportError: If the process was launched as one of several and
            PhysicsNeMo is not installed. Raised before anything is imported or
            written, so the caller has not yet created an output directory.
    """
    try:
        from physicsnemo.distributed import DistributedManager
    except ImportError as exc:
        # PhysicsNeMo is what reads the launcher's environment, so without it
        # every rank of a multi-process launch would call itself rank 0 of a
        # world of 1: each would train on the whole dataset and each would
        # write over the others' checkpoints, silently and at full cost.
        launched = _launched_world_size()
        if launched > 1:
            raise ImportError(
                f"This process is 1 of {launched} in a distributed launch, but "
                "PhysicsNeMo is not installed, and it is what assigns the "
                "ranks. Without it every process would call itself rank 0 and "
                "overwrite the others' output. Reinstall with: pip install "
                "--force-reinstall monai-physio, or run in a single process."
            ) from exc

        import torch

        # The MLP path does not otherwise need PhysicsNeMo, so a missing
        # PhysicsNeMo means a single process rather than an error.
        return DistributedContext(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            rank=0,
            local_rank=0,
            world_size=1,
        )

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    manager = DistributedManager()
    return DistributedContext(
        device=manager.device,
        rank=int(manager.rank),
        local_rank=int(manager.local_rank),
        world_size=int(manager.world_size),
    )


# --------------------------------------------------------------------------- #
# Checkpoint I/O                                                               #
# --------------------------------------------------------------------------- #
def unwrap_model(model: Any) -> Any:
    """Return the bare module inside any ``torch.compile`` / DDP wrappers.

    Both wrappers can be applied, in either order, so peel until neither
    attribute is left rather than checking for one of them.
    """
    while True:
        inner = getattr(model, "_orig_mod", None) or getattr(model, "module", None)
        if inner is None:
            return model
        model = inner


def uncompiled_state_dict(model: Any) -> dict[str, Any]:
    """Return a model's state dict, unwrapping ``torch.compile`` and DDP."""
    return cast(dict[str, Any], unwrap_model(model).state_dict())


def strip_compile_prefix(state: dict) -> dict:
    """Strip the ``_orig_mod.`` prefix that ``torch.compile`` adds to keys."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state):
        return {
            k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()
        }
    return state


# --------------------------------------------------------------------------- #
# Lazy dataset with bounded RAM cache                                          #
# --------------------------------------------------------------------------- #
@dataclass
class _Sample:
    """Everything needed to materialize one ``(subject, phase)`` sample."""

    subject_id: str
    pca_norm: np.ndarray  # (n_pca,) normalized shape parameters
    target_mesh: Path
    stage: float


class PhaseSampleDataset:
    """Lazy provider of ``(node_features, normalized_target)`` samples.

    One item is one ``(subject, phase)`` pair.  Node features are rebuilt on
    access from the shared normalized template coordinates plus the subject's
    normalized PCA parameters and the phase stage (cheap).  Only the phase
    target arrays are read from disk, and those are held in a bounded LRU cache
    so an arbitrarily large training set streams from disk while a small set
    stays resident.  Targets are returned as stored - the dataset never derives
    them from geometry.

    Args:
        samples: Flat list of :class:`_Sample` (built by the workflow).
        mean_coords_norm: ``(n_points, 3)`` normalized template coordinates.
        target_array: Point-data array name holding the targets.
        target_scale: Target normalization factor (targets are divided by it so
            they land in ``~[-1, 1]``).
        cache_max_samples: Maximum decoded target arrays to cache. ``0`` means
            unbounded (all-in-RAM, fastest); a small value forces disk streaming.
    """

    def __init__(
        self,
        samples: list[_Sample],
        mean_coords_norm: np.ndarray,
        target_array: str,
        target_scale: float,
        cache_max_samples: int = 0,
    ) -> None:
        self._samples = samples
        self._mean_coords_norm = mean_coords_norm.astype(np.float32)
        self._target_array = target_array
        self._target_scale = float(target_scale)
        self._cache_max_samples = int(cache_max_samples)
        self._cache: "OrderedDict[Path, np.ndarray]" = OrderedDict()
        self._n_points = int(mean_coords_norm.shape[0])
        self._n_features = int(3 + samples[0].pca_norm.shape[0] + 1) if samples else 0
        self._n_target = (
            int(self._target_values(samples[0].target_mesh).shape[1]) if samples else 0
        )

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def n_points(self) -> int:
        """Vertices per sample (shared across all subjects)."""
        return self._n_points

    @property
    def n_features(self) -> int:
        """Node feature dimension ``3 + n_pca + 1``."""
        return self._n_features

    @property
    def n_target(self) -> int:
        """Target width (columns of the stored target array)."""
        return self._n_target

    @property
    def subject_ids(self) -> list[str]:
        """Subject id of every sample, in dataset order.

        A physics residual is measured against the subject's own reference
        geometry rather than the shared template, so a training method that
        adds one needs to know which subject a batch row came from.
        """
        return [sample.subject_id for sample in self._samples]

    def _target_values(self, path: Path) -> np.ndarray:
        """Read (and cache) a phase's ``(n_points, n_target)`` target array."""
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        values = load_target_array(path, self._target_array)
        if values.shape[0] != self._n_points:
            raise ValueError(
                f"{path} has {values.shape[0]} points, expected {self._n_points}."
            )

        self._cache[path] = values
        if self._cache_max_samples != 0:
            while len(self._cache) > self._cache_max_samples:
                self._cache.popitem(last=False)
        return values

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(node_features, normalized_target)`` for one sample."""
        sample = self._samples[index]
        node_feats = build_node_features(
            self._mean_coords_norm, sample.pca_norm, sample.stage
        )
        target = self._target_values(sample.target_mesh) / self._target_scale
        return node_feats, target.astype(np.float32)
