"""MeshGraphNet training method for PhysicsNeMo mesh-stage models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .train_physicsnemo_base import TrainPhysicsNeMoBase

if TYPE_CHECKING:  # typed for mypy; imported lazily at runtime
    import torch


class TrainPhysicsNeMoMGN(TrainPhysicsNeMoBase):
    """Train a PhysicsNeMo :class:`MeshGraphNet` on mesh stages.

    The mesh-graph topology and edge features are extracted once from the shared
    PCA template mesh (surface or volume) and reused for every
    ``(subject, phase)`` sample; PyTorch Geometric batches join disconnected
    sub-graphs.
    """

    model_tag = "mgn"
    architecture_name = "physicsnemo.models.meshgraphnet.MeshGraphNet"
    _shuffle_points_within_batch = False

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the MeshGraphNet training method.

        Args:
            log_level: Logging level. Default: ``logging.INFO``.
        """
        super().__init__(log_level=log_level)
        self.batch_size = 4  # graphs per step
        self.processor_size: int = 3
        self.hidden_dim: int = 128
        self.num_layers: int = 2
        self.num_processor_checkpoint_segments: int = 0
        # Runtime MGN state (set in setup_inputs).
        self._device: Optional["torch.device"] = None
        self._shared_graph: Any = None
        self._shared_edge_index: Any = None
        self._shared_edge_feats: Any = None
        self._batched_graph_cache: dict[int, tuple[Any, Any]] = {}

    def set_processor_size(self, processor_size: int) -> None:
        """Set the number of message-passing hops."""
        if processor_size < 1:
            raise ValueError(f"processor_size must be >= 1, got {processor_size}")
        self.processor_size = processor_size

    def set_hidden_dim(self, hidden_dim: int) -> None:
        """Set the processor/encoder/decoder hidden dimension."""
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.hidden_dim = hidden_dim

    def set_num_layers(self, num_layers: int) -> None:
        """Set the MLP layer count inside each encoder/processor/decoder block."""
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers

    def set_num_processor_checkpoint_segments(self, num_segments: int) -> None:
        """Set the gradient-checkpointing segment count for the processor.

        Gradient checkpointing recomputes processor activations during the
        backward pass instead of storing them, trading compute for GPU memory.
        A large mesh graph makes that trade worthwhile: a 179k-point, 1.07M-edge
        template peaks near 43 GiB at ``batch_size`` 4 without it.

        Args:
            num_segments: Number of checkpointed segments; ``0`` (the default)
                disables checkpointing and stores every activation.
        """
        if num_segments < 0:
            raise ValueError(f"num_segments must be >= 0, got {num_segments}")
        self.num_processor_checkpoint_segments = num_segments

    def build_model(self, in_features: int, out_features: int) -> "torch.nn.Module":
        MeshGraphNet = pnt.import_meshgraphnet()

        model = MeshGraphNet(
            input_dim_nodes=in_features,
            input_dim_edges=4,  # rel_x, rel_y, rel_z, distance
            output_dim=out_features,
            processor_size=self.processor_size,
            hidden_dim_processor=self.hidden_dim,
            hidden_dim_node_encoder=self.hidden_dim,
            num_layers_node_encoder=self.num_layers,
            hidden_dim_node_decoder=self.hidden_dim,
            num_layers_node_decoder=self.num_layers,
            hidden_dim_edge_encoder=self.hidden_dim,
            num_layers_edge_encoder=self.num_layers,
            num_layers_edge_processor=self.num_layers,
            num_layers_node_processor=self.num_layers,
            aggregation="mean",
            num_processor_checkpoint_segments=self.num_processor_checkpoint_segments,
        )
        return cast("torch.nn.Module", model)

    def setup_inputs(
        self,
        device: "torch.device",
        template_mesh: pv.DataSet,
        template_coords: np.ndarray,
    ) -> None:
        from torch_geometric.data import Data

        self._device = device
        self._shared_edge_index = pnt.mesh_to_edge_index(template_mesh)
        self._shared_edge_feats = pnt.compute_edge_features(
            template_coords, self._shared_edge_index
        )
        self._shared_graph = Data(
            edge_index=self._shared_edge_index,
            num_nodes=len(template_coords),
        )
        self._batched_graph_cache = {}

    def forward(
        self, model: "torch.nn.Module", node_feats: "torch.Tensor", batch_len: int
    ) -> "torch.Tensor":
        graph, edge_feats = self._batched_graph(batch_len)
        return cast("torch.Tensor", model(node_feats, edge_feats, graph))

    def checkpoint_fields(self) -> dict:
        return {
            "processor_size": self.processor_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_processor_checkpoint_segments": self.num_processor_checkpoint_segments,
            "input_dim_edges": 4,
        }

    def save_artifacts(self, output_dir: Path) -> None:
        import torch

        torch.save(self._shared_edge_index, output_dir / "shared_edge_index.pt")
        torch.save(self._shared_edge_feats, output_dir / "shared_edge_features.pt")

    def _batched_graph(self, batch_len: int) -> tuple[Any, Any]:
        """Return (and cache) a batched graph + tiled edge features for ``batch_len``."""
        cached = self._batched_graph_cache.get(batch_len)
        if cached is not None:
            return cached
        from torch_geometric.data import Batch

        assert self._device is not None
        graph = Batch.from_data_list([self._shared_graph] * batch_len).to(self._device)
        edge_feats = self._shared_edge_feats.repeat(batch_len, 1).to(self._device)
        self._batched_graph_cache[batch_len] = (graph, edge_feats)
        return graph, edge_feats
