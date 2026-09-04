"""Workflow for training PhysicsNeMo mesh-stage models.

The workflow owns the data side of training: per-subject manifests,
normalization statistics, lazy dataset construction, output-directory
resolution and the saving of checkpoints, metadata, logs and PCA assets. The
network itself — model construction, the optimization loop and the checkpoint
payload — lives in the training method it drives
(:class:`monai_physio.TrainPhysicsNeMoMGN` or
:class:`monai_physio.TrainPhysicsNeMoMLP`).

Design highlights:

- **Data is a list of per-subject manifest files** (see
  :func:`monai_physio.physicsnemo_tools.parse_manifest`). The caller chooses the
  train / validation / held-out-test split externally; the workflow receives the
  training manifests and validation manifest(s) and the training method reports
  validation RMSE intermittently as training proceeds.
- **Targets come from the manifest**, read verbatim from the phase meshes'
  ``target_array`` point data. Their width sets the network's output size, so a
  displacement model is just the case where the caller stored three columns of
  ``phase.points - reference.points``.
- **The dataset streams lazily** through
  :class:`monai_physio.physicsnemo_tools.PhaseSampleDataset` with a bounded RAM
  cache, so the training set need not fit in memory.
- **Coordinates are always the PCA template mesh** (shared across subjects), a
  surface or a volume; the subject is described by its PCA parameters and the
  stage.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .physicsnemo_tools import PhaseSampleDataset, SubjectManifest, _Sample
from .monai_physio_base import MONAIPhysioBase
from .train_physicsnemo_base import TrainPhysicsNeMoBase
from .train_physicsnemo_mgn import TrainPhysicsNeMoMGN


class WorkflowTrainPhysicsNeMo(MONAIPhysioBase):
    """Train a PhysicsNeMo mesh-stage model from per-subject manifests.

    The network is supplied as a training method — pass a configured
    :class:`monai_physio.TrainPhysicsNeMoMGN` or
    :class:`monai_physio.TrainPhysicsNeMoMLP` instance as ``training_method``;
    a default MeshGraphNet method is used when none is given.
    """

    def __init__(
        self,
        train_manifests: list[Path],
        val_manifests: list[Path],
        pca_mean_mesh: Path,
        output_directory: Path,
        use_template_surface: bool = False,
        resume_from: Optional[Path] = None,
        training_method: Optional[TrainPhysicsNeMoBase] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """Initialize the training workflow.

        Args:
            train_manifests: Per-subject manifest files for the training set.
            val_manifests: Per-subject manifest files for the validation set
                (used for intermittent RMSE reporting during training). May be
                empty to skip validation.
            pca_mean_mesh: PCA template mesh whose point count matches
                ``pca_model.json`` (typically ``pca_mean.vtu``). Its points
                define the shared node coordinates and — for the MGN — the
                mesh-graph topology, so a volumetric template trains on volume
                points and a surface template on surface points. The sibling
                ``pca_model.json`` (if present) is copied into
                ``output_directory`` for inference.
            output_directory: Directory for checkpoints, metadata and logs.
            use_template_surface: Train on the template's extracted surface
                instead of its own points. Set this when the PCA model is
                volumetric but the manifests reference surface meshes.
            resume_from: Optional ``*_stage_model.pt`` to resume from; its
                normalization statistics are inherited so the loaded weights stay
                valid, and a fresh numbered output directory is used.
            training_method: Training method instance carrying the network and
                its hyper-parameters. Defaults to a new
                :class:`monai_physio.TrainPhysicsNeMoMGN`.
            log_level: Logging level. Default: ``logging.INFO``.

        Raises:
            ValueError: If ``train_manifests`` is empty.
            FileNotFoundError: If ``pca_mean_mesh`` does not exist.
            TypeError: If ``training_method`` is neither None nor a
                TrainPhysicsNeMoBase instance.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        if not train_manifests:
            raise ValueError("train_manifests cannot be empty.")
        pca_mean_mesh = Path(pca_mean_mesh)
        if not pca_mean_mesh.exists():
            raise FileNotFoundError(f"pca_mean_mesh not found: {pca_mean_mesh}")
        if training_method is not None and not isinstance(
            training_method, TrainPhysicsNeMoBase
        ):
            raise TypeError(
                "training_method must be a TrainPhysicsNeMoBase instance, got "
                f"{type(training_method).__name__}"
            )

        self.train_manifest_paths = [Path(p) for p in train_manifests]
        self.val_manifest_paths = [Path(p) for p in val_manifests]
        self.pca_mean_mesh = pca_mean_mesh
        self.output_directory = Path(output_directory)
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.training_method = (
            training_method
            if training_method is not None
            else TrainPhysicsNeMoMGN(log_level=log_level)
        )

        # PCA assets shared by every subject.
        self.use_template_surface = use_template_surface
        self._template_mesh: pv.DataSet = pv.read(str(pca_mean_mesh))
        if use_template_surface:
            self._template_mesh = self._template_mesh.extract_surface(
                algorithm="dataset_surface"
            )
        self._template_coords = np.asarray(self._template_mesh.points, dtype=np.float32)
        self._pca_model_path: Optional[Path] = None
        candidate = pca_mean_mesh.parent / "pca_model.json"
        if candidate.exists():
            self._pca_model_path = candidate

        # Dataset streaming budget (decoded phase arrays); 0 = unbounded.
        self.cache_max_samples: int = 0

        # Results (populated by process()).
        self.checkpoint_file: Optional[Path] = None
        self.metadata_file: Optional[Path] = None
        self.training_loss: Optional[list[float]] = None
        self.val_rmse_log: Optional[list[dict]] = None

    def set_cache_size(self, cache_max_samples: int) -> None:
        """Set the RAM cache budget (decoded phase arrays); ``0`` = unbounded."""
        if cache_max_samples < 0:
            raise ValueError(f"cache_max_samples must be >= 0, got {cache_max_samples}")
        self.cache_max_samples = cache_max_samples

    # ─────────────────────────── Main workflow ─────────────────────────────
    def process(self) -> dict[str, Any]:
        """Train the model and write checkpoints, metadata and logs.

        Under a distributed launcher every rank calls this and they train one
        model together; rank 0 alone writes to ``output_directory``, so on the
        other ranks ``checkpoint`` and ``metadata`` come back unset.

        Returns:
            Dict with ``output_directory``, ``checkpoint``, ``metadata``,
            ``training_loss`` and ``val_rmse_log``.
        """
        model_tag = self.training_method.model_tag
        self.log_section("STARTING PHYSICSNEMO %s TRAINING WORKFLOW", model_tag.upper())

        epochs = self.training_method.epochs

        # Picks up torchrun, SLURM or OpenMPI, and reports one rank of one when
        # the process was started without any of them.
        context = pnt.distributed_context()

        output_dir = self._resolve_output_dir(context)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        context.barrier()
        self.log_info("Output directory: %s", output_dir)
        self.log_info(
            "Device: %s  rank %d/%d", context.device, context.rank, context.world_size
        )

        subjects = self._load_subjects()
        resume_ckpt = self._load_resume_checkpoint()
        stats = self._compute_normalization(subjects, resume_ckpt)

        train_dataset, val_dataset = self._build_datasets(subjects, stats)
        self.log_info(
            "Training samples: %d, validation samples: %d, in_features=%d, "
            "n_target=%d, target_scale=%.4f",
            len(train_dataset),
            len(val_dataset),
            train_dataset.n_features,
            train_dataset.n_target,
            stats["target_scale"],
        )

        # Everything inference needs except the weights, written before the
        # first epoch so a run in progress can be evaluated from one of its
        # intermittent checkpoints.  The barrier is what stops another rank
        # reading a half-written shared_edge_index.pt.
        if context.is_main:
            self._save_shared_assets(subjects, stats, output_dir, epochs)
        context.barrier()

        model, losses, rmse_log = self.training_method.train(
            train_dataset,
            val_dataset,
            stats,
            context,
            epochs,
            output_dir,
            self._template_mesh,
            self._template_coords,
            resume_from=self.resume_from,
        )
        if context.is_main:
            self._save_model(model, subjects, stats, losses, rmse_log, output_dir)
        context.barrier()

        self.log_section("PHYSICSNEMO %s TRAINING COMPLETE", model_tag.upper())
        return {
            "output_directory": output_dir,
            "checkpoint": self.checkpoint_file,
            "metadata": self.metadata_file,
            "training_loss": losses,
            "val_rmse_log": rmse_log,
        }

    # ─────────────────────────── Internal steps ────────────────────────────
    def _resolve_output_dir(self, context: pnt.DistributedContext) -> Path:
        """Return the output directory, using a fresh sibling when resuming.

        The sibling search races when several ranks run it at once, so rank 0
        picks the directory and hands the answer to the others.
        """
        base = self.output_directory
        if self.resume_from is None or not base.exists():
            return base
        resolved: list[Any] = [None]
        if context.is_main:
            n = 1
            while (base.parent / f"{base.name}_{n}").exists():
                n += 1
            resolved[0] = base.parent / f"{base.name}_{n}"
        if context.is_distributed:
            import torch

            torch.distributed.broadcast_object_list(resolved, src=0)
        return cast(Path, resolved[0])

    def _load_subjects(self) -> dict[str, dict]:
        """Parse every manifest and load PCA coefficients + target-array names."""
        n_points = len(self._template_coords)
        subjects: dict[str, dict] = {}

        def _load(paths: list[Path], split: str) -> None:
            for manifest_path in paths:
                manifest: SubjectManifest = pnt.parse_manifest(manifest_path)
                if manifest.subject_id in subjects:
                    raise ValueError(
                        f"Duplicate subject_id '{manifest.subject_id}': already "
                        f"loaded in the '{subjects[manifest.subject_id]['split']}' "
                        f"split, seen again in the '{split}' split. Each subject "
                        "must appear in exactly one manifest."
                    )
                fitted_reference_mesh = pv.read(str(manifest.fitted_reference_mesh))
                if fitted_reference_mesh.n_points != n_points:
                    raise ValueError(
                        f"{manifest.fitted_reference_mesh} has {fitted_reference_mesh.n_points} "
                        f"points, expected {n_points} (template topology)."
                    )
                subjects[manifest.subject_id] = {
                    "split": split,
                    "pca_coeffs": pnt.load_pca_coefficients(manifest.pca_coefficients),
                    "target_array": manifest.target_array,
                    "phases": manifest.phases,
                }

        _load(self.train_manifest_paths, "train")
        _load(self.val_manifest_paths, "val")
        n_train = sum(1 for s in subjects.values() if s["split"] == "train")
        if n_train == 0:
            raise ValueError("No training subjects were loaded.")
        target_arrays = {s["target_array"] for s in subjects.values()}
        if len(target_arrays) > 1:
            raise ValueError(
                "All manifests must declare the same target_array; got "
                f"{sorted(target_arrays)}."
            )
        return subjects

    def _load_resume_checkpoint(self) -> Optional[dict]:
        """Load prior-run normalization statistics when resuming."""
        if self.resume_from is None:
            return None
        import torch

        self.log_info("Resuming from %s", self.resume_from)
        return cast(
            dict,
            torch.load(str(self.resume_from), map_location="cpu", weights_only=True),
        )

    def _compute_normalization(
        self, subjects: dict[str, dict], resume_ckpt: Optional[dict]
    ) -> dict:
        """Compute (or inherit) coordinate, PCA and target statistics."""
        # Inherit the exact stats when the checkpoint carries them (final models
        # and, since this change, periodic epoch checkpoints). Bare/legacy epoch
        # checkpoints hold only weights: recompute from the data, which is
        # identical for an unchanged subject set (the normal resume case).
        if resume_ckpt is not None and "coordinate_mean" in resume_ckpt:
            return {
                "coordinate_mean": np.array(resume_ckpt["coordinate_mean"], np.float32),
                "coordinate_scale": np.array(
                    resume_ckpt["coordinate_scale"], np.float32
                ),
                "pca_mean": np.array(resume_ckpt["pca_mean"], np.float32),
                "pca_scale": np.array(resume_ckpt["pca_scale"], np.float32),
                "target_scale": float(resume_ckpt["target_scale"]),
                "n_target": int(resume_ckpt["n_target"]),
            }
        if resume_ckpt is not None:
            self.log_warning(
                "Resume checkpoint has no normalization stats (bare weights-only "
                "checkpoint); recomputing them from the current data."
            )

        coord = self._template_coords
        coordinate_mean = coord.mean(axis=0)
        coordinate_scale = np.where(coord.std(axis=0) == 0.0, 1.0, coord.std(axis=0))

        train_pca = np.vstack(
            [s["pca_coeffs"] for s in subjects.values() if s["split"] == "train"]
        )
        pca_mean = train_pca.mean(axis=0)
        pca_scale = np.where(train_pca.std(axis=0) == 0.0, 1.0, train_pca.std(axis=0))

        target_scale, n_target = self._compute_target_scale(subjects)
        return {
            "coordinate_mean": coordinate_mean.astype(np.float32),
            "coordinate_scale": coordinate_scale.astype(np.float32),
            "pca_mean": pca_mean.astype(np.float32),
            "pca_scale": pca_scale.astype(np.float32),
            "target_scale": target_scale,
            "n_target": n_target,
        }

    def _compute_target_scale(self, subjects: dict[str, dict]) -> tuple[float, int]:
        """One streaming pass over the training targets for their max abs value.

        Returns:
            ``(target_scale, n_target)``; ``target_scale`` falls back to ``1.0``
            when every target is zero.

        Raises:
            ValueError: If a phase's target array disagrees with the template
                point count or with the target width seen so far.
        """
        n_points = len(self._template_coords)
        max_abs = 0.0
        n_target: Optional[int] = None
        for data in subjects.values():
            if data["split"] != "train":
                continue
            for phase in data["phases"]:
                values = pnt.load_target_array(phase.mesh, data["target_array"])
                if values.shape[0] != n_points:
                    raise ValueError(
                        f"{phase.mesh} has {values.shape[0]} points, "
                        f"expected {n_points}."
                    )
                if n_target is None:
                    n_target = int(values.shape[1])
                elif values.shape[1] != n_target:
                    raise ValueError(
                        f"{phase.mesh} has target width {values.shape[1]}, "
                        f"expected {n_target}."
                    )
                max_abs = max(max_abs, float(np.max(np.abs(values))))
        if n_target is None:
            raise ValueError("No training phases were found.")
        return (max_abs if max_abs > 0.0 else 1.0), n_target

    def _build_datasets(
        self, subjects: dict[str, dict], stats: dict
    ) -> tuple[PhaseSampleDataset, PhaseSampleDataset]:
        """Build lazy train and validation datasets sharing the template coords."""
        mean_coords_norm = (self._template_coords - stats["coordinate_mean"]) / stats[
            "coordinate_scale"
        ]
        target_array = next(iter(subjects.values()))["target_array"]

        def _samples(split: str) -> list[_Sample]:
            out: list[_Sample] = []
            for sid, data in sorted(subjects.items()):
                if data["split"] != split:
                    continue
                pca_norm = (data["pca_coeffs"] - stats["pca_mean"]) / stats["pca_scale"]
                for phase in data["phases"]:
                    out.append(
                        _Sample(
                            subject_id=sid,
                            pca_norm=pca_norm.astype(np.float32),
                            target_mesh=phase.mesh,
                            stage=phase.stage,
                        )
                    )
            return out

        train_dataset = PhaseSampleDataset(
            _samples("train"),
            mean_coords_norm,
            target_array,
            stats["target_scale"],
            self.cache_max_samples,
        )
        val_dataset = PhaseSampleDataset(
            _samples("val"),
            mean_coords_norm,
            target_array,
            stats["target_scale"],
            self.cache_max_samples,
        )
        return train_dataset, val_dataset

    def _save_shared_assets(
        self,
        subjects: dict[str, dict],
        stats: dict,
        output_dir: Path,
        epochs: int,
    ) -> None:
        """Write the metadata and PCA assets inference needs beside the weights.

        None of this depends on the trained weights, so it is written before
        training starts:
        :class:`monai_physio.WorkflowInferPhysicsNeMo` reads the template mesh
        and — through the inference method — the shared graph tensors from the
        model directory, and cannot load an intermittent epoch checkpoint until
        they are there.  The training method writes its own artifacts once its
        inputs are set up, at the top of its training loop.
        """
        method = self.training_method
        in_features = 3 + int(stats["pca_mean"].shape[0]) + 1
        metadata_file = output_dir / f"{method.model_tag}_stage_model_metadata.json"

        n_pca = int(stats["pca_mean"].shape[0])
        n_target = int(stats["n_target"])
        target_array = next(iter(subjects.values()))["target_array"]
        input_feature_names = (
            ["mean_shape_x", "mean_shape_y", "mean_shape_z"]
            + [f"pca_c{i + 1}" for i in range(n_pca)]
            + ["stage"]
        )
        metadata = {
            "architecture": method.architecture_name,
            "input_features": input_feature_names,
            "output_features": [f"{target_array}_{i}" for i in range(n_target)],
            "in_features": in_features,
            "n_mesh_points": int(self._template_coords.shape[0]),
            "epochs": epochs,
            "learning_rate": method.learning_rate,
            "batch_size_samples": method.batch_size,
            "coordinate_mean": stats["coordinate_mean"].tolist(),
            "coordinate_scale": stats["coordinate_scale"].tolist(),
            "pca_mean": stats["pca_mean"].tolist(),
            "pca_scale": stats["pca_scale"].tolist(),
            "target_array": target_array,
            "n_target": n_target,
            "target_scale": stats["target_scale"],
            "resumed_from": str(self.resume_from) if self.resume_from else None,
        }
        metadata.update(method.checkpoint_fields())
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Copy PCA assets so the model directory is self-contained for inference.
        shutil.copy2(self.pca_mean_mesh, output_dir / self.pca_mean_mesh.name)
        suffix = ".vtp" if isinstance(self._template_mesh, pv.PolyData) else ".vtu"
        self._template_mesh.save(str(output_dir / f"pca_mean_template{suffix}"))
        if self._pca_model_path is not None:
            shutil.copy2(self._pca_model_path, output_dir / "pca_model.json")

        self.metadata_file = metadata_file

    def _save_model(
        self,
        model: Any,
        subjects: dict[str, dict],
        stats: dict,
        losses: list[float],
        rmse_log: list[dict],
        output_dir: Path,
    ) -> None:
        """Persist the final checkpoint and the training logs."""
        import torch

        tag = self.training_method.model_tag
        checkpoint_file = output_dir / f"{tag}_stage_model.pt"

        train_ids = sorted(s for s, d in subjects.items() if d["split"] == "train")
        val_ids = sorted(s for s, d in subjects.items() if d["split"] == "val")

        checkpoint = self.training_method.build_checkpoint(model, stats)
        checkpoint["target_array"] = next(iter(subjects.values()))["target_array"]
        checkpoint["train_subject_ids"] = train_ids
        checkpoint["val_subject_ids"] = val_ids
        checkpoint["resumed_from"] = str(self.resume_from) if self.resume_from else None
        torch.save(checkpoint, checkpoint_file)

        (output_dir / "training_losses.json").write_text(
            json.dumps(losses, indent=2), encoding="utf-8"
        )
        (output_dir / "training_validation_rmse.json").write_text(
            json.dumps(rmse_log, indent=2), encoding="utf-8"
        )
        with (output_dir / "training_validation_rmse.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=["epoch", "train_rmse", "val_rmse"])
            writer.writeheader()
            writer.writerows(rmse_log)

        self.checkpoint_file = checkpoint_file
        self.training_loss = losses
        self.val_rmse_log = rmse_log
        self.log_info("Model saved to %s", checkpoint_file)
