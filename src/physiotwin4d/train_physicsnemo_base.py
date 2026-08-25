"""Training methods for PhysicsNeMo mesh-stage models.

A training method owns the network, the optimization loop and the checkpoint
payload. The surrounding data work — manifests, normalization statistics,
dataset construction and artifact saving — lives in the workflows that drive
these methods (:mod:`physiotwin4d.workflow_train_physicsnemo`).

:class:`TrainPhysicsNeMoBase` holds every step common to the supported
networks; the concrete :class:`physiotwin4d.TrainPhysicsNeMoMGN` (MeshGraphNet)
and :class:`physiotwin4d.TrainPhysicsNeMoMLP` (fully connected) subclasses
supply only the network-specific seams. Both learn the same task: given
per-vertex features ``[mean_shape_x, mean_shape_y, mean_shape_z, pca_c1..cN,
stage]`` predict the per-vertex target stored in the manifest, whose width sets
the network's output size.

MLP note: the natural batch unit is a group of whole ``(subject, phase)``
samples, so the MLP batches several samples per step and shuffles points
*within* the batch to retain gradient mixing (``batch_size`` therefore counts
samples, not points). The MGN keeps per-sample vertex order intact because it
indexes the shared mesh graph.

PhysicsNeMo (and, for the MGN, PyTorch Geometric) are optional dependencies;
they are imported lazily so ``import physiotwin4d`` works without them.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .physicsnemo_tools import DistributedContext, PhaseSampleDataset
from .physiotwin4d_base import PhysioTwin4DBase

if TYPE_CHECKING:  # typed for mypy; imported lazily at runtime
    import torch


class TrainPhysicsNeMoBase(PhysioTwin4DBase):
    """Base class for a PhysicsNeMo mesh-stage training method.

    Not instantiated directly — use :class:`physiotwin4d.TrainPhysicsNeMoMGN` or
    :class:`physiotwin4d.TrainPhysicsNeMoMLP`. Subclasses implement the network
    seams (:meth:`build_model`, :meth:`setup_inputs`, :meth:`forward`,
    :meth:`checkpoint_fields`, :meth:`save_artifacts`) and set the class
    attributes ``model_tag``, ``architecture_name`` and
    ``_shuffle_points_within_batch``.
    """

    # Network identity / behavior — overridden by subclasses.
    model_tag: str = "base"
    architecture_name: str = "base"
    _shuffle_points_within_batch: bool = False

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the training method with its default hyper-parameters.

        Args:
            log_level: Logging level. Default: ``logging.INFO``.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.epochs: int = 1500
        self.batch_size: int = 4
        self.learning_rate: float = 1.0e-3
        self.rmse_log_interval: int = 100
        self.loss_log_interval: int = 10
        self.seed: int = 42

    # ─────────────────────────── Tuning setters ────────────────────────────
    def set_epochs(self, epochs: int) -> None:
        """Set the number of training epochs."""
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        self.epochs = epochs

    def set_batch_size(self, batch_size: int) -> None:
        """Set the mini-batch size, measured in ``(subject, phase)`` samples."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the Adam learning rate."""
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        self.learning_rate = learning_rate

    # ─────────────────────────── Network seams ─────────────────────────────
    def build_model(self, in_features: int, out_features: int) -> "torch.nn.Module":
        """Construct the (uncompiled) network. Implemented by subclasses."""
        raise NotImplementedError

    def setup_inputs(
        self,
        device: "torch.device",
        template_mesh: pv.DataSet,
        template_coords: np.ndarray,
    ) -> None:
        """Prepare any shared per-forward inputs (MGN graph tensors)."""
        raise NotImplementedError

    def forward(
        self, model: "torch.nn.Module", node_feats: "torch.Tensor", batch_len: int
    ) -> "torch.Tensor":
        """Run the network for a flattened ``(batch_len * n_points, F)`` batch."""
        raise NotImplementedError

    def checkpoint_fields(self) -> dict:
        """Return architecture-specific fields to store in the checkpoint."""
        raise NotImplementedError

    def save_artifacts(self, output_dir: Path) -> None:
        """Save any architecture-specific artifacts (MGN graph tensors).

        Called by :meth:`train` as soon as :meth:`setup_inputs` has run, so the
        artifacts are in place for inference from the first intermittent
        checkpoint rather than only after the last epoch.
        """
        raise NotImplementedError

    # ─────────────────────────── Training loop ─────────────────────────────
    def train(
        self,
        train_dataset: PhaseSampleDataset,
        val_dataset: PhaseSampleDataset,
        stats: dict,
        context: DistributedContext,
        epochs: int,
        output_dir: Path,
        template_mesh: pv.DataSet,
        template_coords: np.ndarray,
        resume_from: Optional[Path] = None,
    ) -> tuple["torch.nn.Module", list[float], list[dict]]:
        """Train the network, returning the model and the loss / RMSE logs.

        Every rank runs this. Each steps over its own disjoint slice of the
        samples and the gradients are averaged across ranks, so the effective
        batch is ``batch_size * world_size``. Only rank 0 writes to disk.

        Args:
            train_dataset: Lazy training samples built by the workflow.
            val_dataset: Lazy validation samples; may be empty.
            stats: Normalization statistics computed by the workflow.
            context: Rank, device and world size of this process.
            epochs: Number of epochs to run. The workflow may clamp
                :attr:`epochs` (for example in test mode), so the effective
                count is passed in rather than read from the method.
            output_dir: Directory for the intermittent epoch checkpoints.
            template_mesh: Shared PCA template mesh, source of the mesh graph.
            template_coords: Shared template node coordinates.
            resume_from: Optional checkpoint whose weights are loaded before
                training starts.

        Returns:
            Tuple of ``(model, training_losses, val_rmse_log)``.
        """
        import torch

        device = context.device
        # One seed for every rank: _iter_batches shards a single shared
        # permutation, so the ranks have to draw the identical one for their
        # slices to tile the dataset exactly once.
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        in_features = train_dataset.n_features

        model = self.build_model(in_features, train_dataset.n_target).to(device)
        if resume_from is not None:
            ckpt = torch.load(str(resume_from), map_location=device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(pnt.strip_compile_prefix(state))
            self._log_main(context, "Loaded model weights from %s", resume_from)

        self.setup_inputs(device, template_mesh, template_coords)
        # Written now rather than after the last epoch: inference against an
        # intermittent checkpoint needs them, and that is the point of writing
        # those checkpoints while a long run is still going.
        if context.is_main:
            self.save_artifacts(output_dir)

        # Wrapped before compiling, not after: Dynamo only splits the graph at
        # the gradient-bucket boundaries, and so only overlaps communication
        # with compute, when it can see the DistributedDataParallel module.
        if context.is_distributed:
            # Both networks use every parameter on every step, so searching for
            # unused ones would only cost a graph traversal per iteration.
            model = cast(
                "torch.nn.Module",
                torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[context.local_rank] if device.type == "cuda" else None,
                    output_device=device if device.type == "cuda" else None,
                    find_unused_parameters=False,
                ),
            )
            self._log_main(
                context, "DistributedDataParallel over %d ranks.", context.world_size
            )

        if sys.platform != "win32":
            try:
                model = cast("torch.nn.Module", torch.compile(model))
                self._log_main(context, "torch.compile enabled.")
            except Exception as exc:  # pragma: no cover - platform dependent
                self._log_main(context, "torch.compile skipped (%s).", exc)
        else:
            self._log_main(context, "torch.compile skipped on Windows.")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        target_scale = stats["target_scale"]

        losses: list[float] = []
        rmse_log: list[dict] = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_rows = 0
            for node_feats, targets, batch_len in self._iter_batches(
                train_dataset, rng, shuffle=True, context=context
            ):
                nf = torch.from_numpy(node_feats).to(device)
                tgt = torch.from_numpy(targets).to(device)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast(device):
                    pred = self.forward(model, nf, batch_len)
                    loss = loss_fn(pred, tgt)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(nf)
                n_rows += len(nf)
            # Every rank saw a disjoint slice, so the epoch mean is only the
            # mean over the whole epoch once the two sums are pooled.
            epoch_loss, n_rows = self._reduce_sums(context, epoch_loss, n_rows)
            losses.append(epoch_loss / max(n_rows, 1))

            if (epoch + 1) % self.loss_log_interval == 0 or epoch + 1 == epochs:
                self._log_main(
                    context, "  epoch %05d/%d  loss=%.6f", epoch + 1, epochs, losses[-1]
                )

            scored_epoch = (
                epoch + 1
            ) % self.rmse_log_interval == 0 or epoch + 1 == epochs
            # Scored on rank 0 alone, over the whole unsharded dataset and
            # against the bare module: the RMSE describes the model, not how
            # the epoch happened to be split across ranks.
            if scored_epoch and context.is_main:
                bare = pnt.unwrap_model(model)
                train_rmse = self._evaluate_rmse(
                    bare, train_dataset, target_scale, device
                )
                # None, not NaN: rmse_log is serialized with json.dumps, which
                # emits a bare NaN token that strict JSON parsers reject.
                val_rmse = (
                    self._evaluate_rmse(bare, val_dataset, target_scale, device)
                    if len(val_dataset) > 0
                    else None
                )
                rmse_log.append(
                    {
                        "epoch": epoch + 1,
                        "train_rmse": train_rmse,
                        "val_rmse": val_rmse,
                    }
                )
                ckpt_path = (
                    output_dir
                    / f"{self.model_tag}_stage_model_epoch_{epoch + 1:05d}.pt"
                )
                torch.save(self.build_checkpoint(model, stats), ckpt_path)
                self._log_main(
                    context,
                    "  intermittent test epoch %05d/%d  train RMSE=%.4f  "
                    "val RMSE=%s  checkpoint=%s",
                    epoch + 1,
                    epochs,
                    train_rmse,
                    "n/a" if val_rmse is None else f"{val_rmse:.4f}",
                    ckpt_path.name,
                )
        model.eval()
        return model, losses, rmse_log

    def build_checkpoint(self, model: "torch.nn.Module", stats: dict) -> dict[str, Any]:
        """Assemble a self-describing checkpoint (weights + normalization stats).

        Both the periodic epoch checkpoints and the final model share this
        payload so training can resume from — and inference can load — any saved
        checkpoint, not just the final one.
        """
        checkpoint: dict[str, Any] = {
            "model_state_dict": pnt.uncompiled_state_dict(model),
            "architecture": self.architecture_name,
            "in_features": 3 + int(stats["pca_mean"].shape[0]) + 1,
            "n_pca": int(stats["pca_mean"].shape[0]),
            "coordinate_mean": stats["coordinate_mean"].tolist(),
            "coordinate_scale": stats["coordinate_scale"].tolist(),
            "pca_mean": stats["pca_mean"].tolist(),
            "pca_scale": stats["pca_scale"].tolist(),
            "target_scale": stats["target_scale"],
            "n_target": int(stats["n_target"]),
        }
        checkpoint.update(self.checkpoint_fields())
        return checkpoint

    # ─────────────────────────── Internal steps ────────────────────────────
    def _log_main(self, context: DistributedContext, msg: str, *args: Any) -> None:
        """Log from rank 0 only, so an N-rank run prints one copy of each line."""
        if context.is_main:
            self.log_info(msg, *args)

    @staticmethod
    def _reduce_sums(
        context: DistributedContext, loss_sum: float, row_count: int
    ) -> tuple[float, int]:
        """Sum a rank's loss and row totals across every rank."""
        if not context.is_distributed:
            return loss_sum, row_count
        import torch

        totals = torch.tensor(
            [loss_sum, float(row_count)], dtype=torch.float64, device=context.device
        )
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        return float(totals[0].item()), int(totals[1].item())

    def _iter_batches(
        self,
        dataset: PhaseSampleDataset,
        rng: Any,
        shuffle: bool,
        context: Optional[DistributedContext] = None,
    ) -> Any:
        """Yield ``(node_feats, targets, batch_len)`` flattened mini-batches.

        With a distributed ``context``, each rank takes a strided slice of the
        one shared permutation, and the slice is then truncated to a whole
        number of batches, equal on every rank: a rank that yielded one batch
        more than its peers would hang them all at the gradient all-reduce of
        the step they never take. Between them the ranks therefore cover the
        retained samples exactly once and no two ranks see the same one, but
        the remainder the truncation drops is not covered at all -- an epoch is
        the truncated subset, not the whole dataset.
        Omitting ``context`` iterates the whole dataset, which is what the
        RMSE evaluation wants.
        """
        n = len(dataset)
        order = rng.permutation(n) if shuffle else np.arange(n)
        if context is not None and context.is_distributed:
            order = order[context.rank :: context.world_size]
            n_per_rank = (n // context.world_size // self.batch_size) * self.batch_size
            if n_per_rank == 0:
                raise ValueError(
                    f"{n} samples over {context.world_size} ranks at batch_size "
                    f"{self.batch_size} leaves at least one rank with no full "
                    "batch. Lower batch_size or the rank count."
                )
            order = order[:n_per_rank]
        n = len(order)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            pairs = [dataset[int(i)] for i in idx]
            node_feats = np.vstack([p[0] for p in pairs])
            targets = np.vstack([p[1] for p in pairs])
            if shuffle and self._shuffle_points_within_batch:
                perm = rng.permutation(len(node_feats))
                node_feats = node_feats[perm]
                targets = targets[perm]
            yield node_feats, targets, len(idx)

    def _autocast(self, device: "torch.device") -> Any:
        """BF16 autocast on CUDA; a no-op context elsewhere."""
        import contextlib

        import torch

        if device.type == "cuda":
            return torch.amp.autocast(device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _evaluate_rmse(
        self,
        model: "torch.nn.Module",
        dataset: PhaseSampleDataset,
        target_scale: float,
        device: "torch.device",
    ) -> float:
        """Per-point RMSE over a dataset, in the units of the stored targets."""
        import torch

        rng = np.random.default_rng(0)
        model.eval()
        total_sq = 0.0
        n_points = 0
        with torch.no_grad():
            for node_feats, targets, batch_len in self._iter_batches(
                dataset, rng, shuffle=False
            ):
                nf = torch.from_numpy(node_feats).to(device)
                pred = self.forward(model, nf, batch_len).cpu().numpy()
                err = (pred - targets) * target_scale
                total_sq += float(np.sum(err**2))
                n_points += err.shape[0]
        model.train()
        return float(np.sqrt(total_sq / max(n_points, 1)))
