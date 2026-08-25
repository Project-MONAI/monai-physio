"""Workflow for predicting mesh-stage targets with a trained PhysicsNeMo model.

The workflow owns everything around the network: the checkpoint and its
normalization statistics, the shared PCA template mesh, manifests, and the
writing of predicted meshes. Scoring those predictions belongs to
:class:`physiotwin4d.WorkflowEvaluateMovement`. The network itself is supplied
as an inference method (:class:`physiotwin4d.InferPhysicsNeMoMGN` or
:class:`physiotwin4d.InferPhysicsNeMoMLP`).

Predictions are the targets the model was trained on, whatever those are — the
manifest's ``target_array`` values at each template point. For the common case
where those targets are displacements from the subject's reference mesh, wrap
this workflow in :class:`physiotwin4d.WorkflowInferMovement` to get
reconstructed surfaces and deformation fields.

PhysicsNeMo (and, for the MGN, PyTorch Geometric) are optional dependencies,
imported lazily so ``import physiotwin4d`` works without them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .infer_physicsnemo_base import InferPhysicsNeMoBase
from .infer_physicsnemo_mgn import InferPhysicsNeMoMGN
from .physiotwin4d_base import PhysioTwin4DBase


class WorkflowInferPhysicsNeMo(PhysioTwin4DBase):
    """Predict per-point targets for a subject at requested stages.

    The network is supplied as an inference method — pass a
    :class:`physiotwin4d.InferPhysicsNeMoMGN` or
    :class:`physiotwin4d.InferPhysicsNeMoMLP` instance as ``inference_method``;
    a default MeshGraphNet method is used when none is given.
    """

    def __init__(
        self,
        model_directory: Path,
        inference_method: Optional[InferPhysicsNeMoBase] = None,
        epoch: Optional[int] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """Load a trained model and its normalization statistics.

        Args:
            model_directory: Directory written by
                :class:`physiotwin4d.WorkflowTrainPhysicsNeMo` (holds
                ``<tag>_stage_model.pt``, ``pca_mean_template.vtp`` or ``.vtu``
                and, for the MGN, the shared graph tensors).
            inference_method: Inference method carrying the network. Defaults to
                a new :class:`physiotwin4d.InferPhysicsNeMoMGN`.
            epoch: Optional intermittent-checkpoint epoch to load
                (``<tag>_stage_model_epoch_#####.pt``). When ``None`` the final
                weights stored in the main checkpoint are used.
            log_level: Logging level. Default: ``logging.INFO``.

        Raises:
            FileNotFoundError: If the model checkpoint or the template mesh is
                missing.
            TypeError: If ``inference_method`` is neither None nor an
                InferPhysicsNeMoBase instance.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        import torch

        if inference_method is not None and not isinstance(
            inference_method, InferPhysicsNeMoBase
        ):
            raise TypeError(
                "inference_method must be an InferPhysicsNeMoBase instance, got "
                f"{type(inference_method).__name__}"
            )
        self.inference_method = (
            inference_method
            if inference_method is not None
            else InferPhysicsNeMoMGN(log_level=log_level)
        )

        self.model_directory = Path(model_directory)
        tag = self.inference_method.model_tag
        if epoch is not None:
            checkpoint_file = (
                self.model_directory / f"{tag}_stage_model_epoch_{epoch:05d}.pt"
            )
        else:
            checkpoint_file = self.model_directory / f"{tag}_stage_model.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
        self.epoch = epoch
        self.checkpoint_file = checkpoint_file

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_info("Loading %s model from %s", tag.upper(), checkpoint_file)
        meta = torch.load(str(checkpoint_file), map_location="cpu", weights_only=True)
        self._meta = meta

        # Normalization statistics and target description.
        self.coordinate_mean = np.array(meta["coordinate_mean"], dtype=np.float32)
        self.coordinate_scale = np.array(meta["coordinate_scale"], dtype=np.float32)
        self.pca_mean = np.array(meta["pca_mean"], dtype=np.float32)
        self.pca_scale = np.array(meta["pca_scale"], dtype=np.float32)
        self.target_scale = float(meta["target_scale"])
        self.n_target = int(meta["n_target"])
        self.target_array = str(meta.get("target_array", "target"))

        # Shared template mesh (node coordinates + output topology).
        self._template_mesh = self._load_template_mesh()
        self._template_coords = np.asarray(self._template_mesh.points, dtype=np.float32)
        self._mean_coords_norm = (
            self._template_coords - self.coordinate_mean
        ) / self.coordinate_scale

        # Build the network, load its weights and attach it to the method.
        model = self.inference_method.build_model(meta).to(self._device)
        self.inference_method.load_artifacts(
            self.model_directory, len(self._template_coords), self._device
        )
        state = self._load_weights(epoch)
        model.load_state_dict(pnt.strip_compile_prefix(state))
        model.eval()
        self.inference_method.set_model(model, self._device)

    # ─────────────────────────── Shared assets ─────────────────────────────
    @property
    def template_mesh(self) -> pv.DataSet:
        """The shared PCA template mesh defining node order and topology."""
        return self._template_mesh

    def _load_template_mesh(self) -> pv.DataSet:
        """Read the template mesh the training workflow copied into the model dir."""
        for suffix in (".vtp", ".vtu"):
            candidate = self.model_directory / f"pca_mean_template{suffix}"
            if candidate.exists():
                return cast(pv.DataSet, pv.read(str(candidate)))
        raise FileNotFoundError(
            f"pca_mean_template.vtp/.vtu not found in {self.model_directory}"
        )

    def _load_weights(self, epoch: Optional[int]) -> dict:
        """Return the state dict for the requested epoch (or final weights)."""
        import torch

        if epoch is None:
            return dict(self._meta["model_state_dict"])
        tag = self.inference_method.model_tag
        epoch_file = self.model_directory / f"{tag}_stage_model_epoch_{epoch:05d}.pt"
        if not epoch_file.exists():
            raise FileNotFoundError(f"Epoch checkpoint not found: {epoch_file}")
        ckpt = torch.load(str(epoch_file), map_location="cpu", weights_only=True)
        # Self-describing checkpoints wrap the weights under "model_state_dict";
        # bare/legacy epoch checkpoints are the state dict itself.
        return cast(dict, ckpt.get("model_state_dict", ckpt))

    # ─────────────────────────── Core predictor ────────────────────────────
    def predict(self, pca_coeffs: np.ndarray, stage: float) -> np.ndarray:
        """Predict ``(n_points, n_target)`` targets for a subject at a stage."""
        pca_norm = (pca_coeffs - self.pca_mean) / self.pca_scale
        node_feats = pnt.build_node_features(self._mean_coords_norm, pca_norm, stage)
        return self.inference_method.predict(node_feats) * self.target_scale

    def predicted_mesh(self, targets: np.ndarray) -> pv.DataSet:
        """Return a template copy carrying ``targets`` as its target array."""
        mesh = self._template_mesh.copy(deep=True)
        mesh.point_data[self.target_array] = targets
        return mesh

    # ─────────────────────────── Public API ────────────────────────────────
    def process(
        self,
        subject_manifest: Path,
        stages: Optional[list[float]] = None,
        output_directory: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Predict a subject's targets from a manifest.

        Every phase in the manifest is predicted, or the arbitrary ``stages``
        given instead.

        Args:
            subject_manifest: Path to the subject manifest JSON.
            stages: Optional list of stages to predict.
            output_directory: Output directory; defaults to
                ``<model_directory>/<subject_id>``.

        Returns:
            Dict with ``subject_id`` and ``predicted_meshes`` (paths).
        """
        manifest = pnt.parse_manifest(subject_manifest)
        pca_coeffs = pnt.load_pca_coefficients(manifest.pca_coefficients)

        out_dir = (
            Path(output_directory)
            if output_directory is not None
            else self.model_directory / manifest.subject_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = self.inference_method.model_tag
        self.log_section("INFER %s [%s]", tag.upper(), manifest.subject_id)

        suffix = ".vtp" if isinstance(self._template_mesh, pv.PolyData) else ".vtu"
        sid = manifest.subject_id
        meshes: list[Path] = []

        requested = stages if stages is not None else [p.stage for p in manifest.phases]
        for stage in requested:
            predicted = self.predict(pca_coeffs, stage)
            path = out_dir / f"{sid}_pred_s{int(stage * 100):03d}{suffix}"
            self.predicted_mesh(predicted).save(str(path))
            meshes.append(path)

            self.log_info("stage %.3f -> %s", stage, path.name)

        return {"subject_id": sid, "predicted_meshes": meshes}
