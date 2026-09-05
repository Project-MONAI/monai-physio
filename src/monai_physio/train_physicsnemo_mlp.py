"""Fully connected (MLP) training method for PhysicsNeMo mesh-stage models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pyvista as pv

from .train_physicsnemo_base import TrainPhysicsNeMoBase

if TYPE_CHECKING:  # typed for mypy; imported lazily at runtime
    import torch


class TrainPhysicsNeMoMLP(TrainPhysicsNeMoBase):
    """Train a PhysicsNeMo :class:`FullyConnected` (MLP) on mesh stages.

    Each mesh point is an independent training row; batches group several
    ``(subject, phase)`` samples and shuffle points within the batch to retain
    gradient mixing while still streaming from disk.
    """

    model_tag = "mlp"
    architecture_name = "physicsnemo.models.mlp.FullyConnected"
    _shuffle_points_within_batch = True

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the fully connected training method.

        Args:
            log_level: Logging level. Default: ``logging.INFO``.
        """
        super().__init__(log_level=log_level)
        self.batch_size = 32  # samples per step (points shuffled within the batch)
        self.layer_size: int = 512
        self.num_layers: int = 6

    def set_layer_size(self, layer_size: int) -> None:
        """Set the hidden layer width."""
        if layer_size < 1:
            raise ValueError(f"layer_size must be >= 1, got {layer_size}")
        self.layer_size = layer_size

    def set_num_layers(self, num_layers: int) -> None:
        """Set the number of fully connected layers."""
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers

    def build_model(self, in_features: int, out_features: int) -> "torch.nn.Module":
        try:
            from physicsnemo.models.mlp import FullyConnected
        except ImportError as exc:  # pragma: no cover - broken environment
            raise ImportError(
                "The MLP trainer requires PhysicsNeMo, a base dependency of "
                "monai-physio. Reinstall with: "
                "pip install --force-reinstall monai-physio"
            ) from exc

        model = FullyConnected(
            in_features=in_features,
            layer_size=self.layer_size,
            out_features=out_features,
            num_layers=self.num_layers,
            activation_fn="silu",
            skip_connections=True,
        )
        return cast("torch.nn.Module", model)

    def setup_inputs(
        self,
        device: "torch.device",
        template_mesh: pv.DataSet,
        template_coords: np.ndarray,
    ) -> None:
        # The MLP needs no shared graph inputs.
        return None

    def forward(
        self, model: "torch.nn.Module", node_feats: "torch.Tensor", batch_len: int
    ) -> "torch.Tensor":
        return cast("torch.Tensor", model(node_feats))

    def checkpoint_fields(self) -> dict:
        return {"layer_size": self.layer_size, "num_layers": self.num_layers}

    def save_artifacts(self, output_dir: Path) -> None:
        # No MGN-only graph artifacts for the MLP.
        return None
