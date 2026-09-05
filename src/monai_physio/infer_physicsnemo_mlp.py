"""Fully connected (MLP) inference method for PhysicsNeMo mesh-stage models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from .infer_physicsnemo_base import InferPhysicsNeMoBase

if TYPE_CHECKING:  # typed for mypy; imported lazily at runtime
    import torch


class InferPhysicsNeMoMLP(InferPhysicsNeMoBase):
    """Predict mesh stages with a trained PhysicsNeMo FullyConnected model."""

    model_tag = "mlp"
    _INFER_CHUNK = 262144

    def build_model(self, meta: dict) -> "torch.nn.Module":
        try:
            from physicsnemo.models.mlp import FullyConnected
        except ImportError as exc:  # pragma: no cover - broken environment
            raise ImportError(
                "The MLP inferencer requires PhysicsNeMo, a base dependency of "
                "monai-physio. Reinstall with: "
                "pip install --force-reinstall monai-physio"
            ) from exc

        model = FullyConnected(
            in_features=int(meta["in_features"]),
            layer_size=int(meta["layer_size"]),
            out_features=int(meta["n_target"]),
            num_layers=int(meta["num_layers"]),
            activation_fn="silu",
            skip_connections=True,
        )
        return cast("torch.nn.Module", model)

    def load_artifacts(
        self, model_directory: Path, n_points: int, device: "torch.device"
    ) -> None:
        # The MLP has no shared graph artifacts.
        return None

    def predict(self, node_feats: np.ndarray) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(node_feats), self._INFER_CHUNK):
                block = node_feats[start : start + self._INFER_CHUNK].astype(np.float32)
                tensor = torch.from_numpy(block).to(self._device)
                chunks.append(self._model(tensor).cpu().numpy())
        return np.vstack(chunks).astype(np.float32)
