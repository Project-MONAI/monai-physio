"""MeshGraphNet inference method for PhysicsNeMo mesh-stage models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from .infer_physicsnemo_base import InferPhysicsNeMoBase

if TYPE_CHECKING:  # typed for mypy; imported lazily at runtime
    import torch


class InferPhysicsNeMoMGN(InferPhysicsNeMoBase):
    """Predict mesh stages with a trained PhysicsNeMo MeshGraphNet.

    The shared graph topology and edge features are loaded from the tensors the
    training workflow saved next to the checkpoint.
    """

    model_tag = "mgn"

    def build_model(self, meta: dict) -> "torch.nn.Module":
        try:
            import torch_geometric  # noqa: F401 - needed by the graph seams

            from physicsnemo.models.meshgraphnet import MeshGraphNet
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The MGN inferencer requires PhysicsNeMo and PyTorch Geometric. "
                'Install with: pip install "monai-physio[physicsnemo]" && '
                "pip install torch-geometric"
            ) from exc

        num_layers = int(meta.get("num_layers", 2))
        hidden_dim = int(meta["hidden_dim"])
        model = MeshGraphNet(
            input_dim_nodes=int(meta["in_features"]),
            input_dim_edges=int(meta.get("input_dim_edges", 4)),
            output_dim=int(meta["n_target"]),
            processor_size=int(meta["processor_size"]),
            hidden_dim_processor=hidden_dim,
            hidden_dim_node_encoder=hidden_dim,
            num_layers_node_encoder=num_layers,
            hidden_dim_node_decoder=hidden_dim,
            num_layers_node_decoder=num_layers,
            hidden_dim_edge_encoder=hidden_dim,
            num_layers_edge_encoder=num_layers,
            num_layers_edge_processor=num_layers,
            num_layers_node_processor=num_layers,
            aggregation="mean",
            num_processor_checkpoint_segments=int(
                meta.get("num_processor_checkpoint_segments", 0)
            ),
        )
        return cast("torch.nn.Module", model)

    def load_artifacts(
        self, model_directory: Path, n_points: int, device: "torch.device"
    ) -> None:
        import torch
        from torch_geometric.data import Data

        edge_index = torch.load(
            str(model_directory / "shared_edge_index.pt"),
            map_location="cpu",
            weights_only=True,
        )
        edge_feats = torch.load(
            str(model_directory / "shared_edge_features.pt"),
            map_location="cpu",
            weights_only=True,
        )
        self._shared_edge_index = edge_index
        self._shared_edge_feats = edge_feats.to(device)
        self._shared_graph = Data(edge_index=edge_index, num_nodes=n_points).to(device)

    def predict(self, node_feats: np.ndarray) -> np.ndarray:
        import torch

        nf = torch.from_numpy(node_feats.astype(np.float32)).to(self._device)
        with torch.no_grad():
            pred = self._model(nf, self._shared_edge_feats, self._shared_graph)
        return np.asarray(pred.cpu().numpy(), dtype=np.float32)
