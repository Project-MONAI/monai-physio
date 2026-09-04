"""Finetune uniGradICON registration.

This module provides :class:`WorkflowFinetuneICONRegistration`, which builds a
paired dataset JSON and YAML config from per-subject lists of image files (with
optional labelmaps and landmark CSVs) and launches
``unigradicon.finetuning.finetune`` as a subprocess.

Conventions:
    - Finetuning is file-based: it reads images/labelmaps/landmarks from disk
      because ``unigradicon.finetuning.finetune`` is launched as a subprocess
      that consumes JSON paths.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .monai_physio_base import MONAIPhysioBase


class WorkflowFinetuneICONRegistration(MONAIPhysioBase):
    """Finetune uniGradICON on paired 3D images.

    Build a paired dataset JSON and YAML config from per-subject lists of
    image, labelmap, and landmark files, then launch
    ``unigradicon.finetuning.finetune`` as a subprocess.  Each subject's
    time-point images form one paired group (they share a ``subject_id``).

    Attributes:
        subject_image_files (list[list[str]]): Per-subject lists of image
            paths.  Images within one inner list share a subject_id during
            finetuning.
        output_dir (Path): Directory where the dataset JSON, YAML config, and
            the uniGradICON ``checkpoints/`` tree are written.
        finetune_name (str): Sub-directory name for the experiment outputs.
        subject_ids (Optional[list[str]]): One ID per subject (e.g. patient
            identifiers).  Written into the dataset JSON's ``subject_id``
            field; falls back to synthetic ``subject_NNNN`` when ``None``.
        subject_labelmap_files (Optional[list[list[Optional[str]]]]):
            Per-subject multi-label labelmap paths aligned with
            ``subject_image_files``.  ``None`` (or per-image ``None``) means no
            labelmap for that image.  If supplied for at least one image,
            paired-with-seg training is enabled.
        subject_mask_files (Optional[list[list[Optional[str]]]]):
            Per-subject binary mask paths aligned with ``subject_image_files``,
            used for loss-function masking.  ``None`` disables masking.
        subject_landmark_files (Optional[list[list[Optional[str]]]]):
            Per-subject landmark CSV paths (``Name,X,Y,Z`` format) aligned with
            ``subject_image_files``.  Recorded in the dataset JSON for
            traceability; not consumed by uniGradICON finetuning itself.

    Example:
        >>> workflow = WorkflowFinetuneICONRegistration(
        ...     subject_image_files=[
        ...         ['pm0001/g000.nii.gz', 'pm0001/g050.nii.gz'],
        ...         ['pm0002/g000.nii.gz', 'pm0002/g050.nii.gz'],
        ...     ],
        ...     output_dir=Path('d:/MONAI-Physio/icon_finetuned'),
        ...     finetune_name='duke_4d_gated_icon_ft',
        ...     subject_labelmap_files=[
        ...         ['pm0001/g000_labelmap.nii.gz', 'pm0001/g050_labelmap.nii.gz'],
        ...         ['pm0002/g000_labelmap.nii.gz', 'pm0002/g050_labelmap.nii.gz'],
        ...     ],
        ... )
        >>> weights_path = workflow.process()
    """

    def __init__(
        self,
        subject_image_files: list[list[str]],
        output_dir: Path,
        finetune_name: str,
        subject_ids: Optional[list[str]] = None,
        subject_labelmap_files: Optional[list[list[Optional[str]]]] = None,
        subject_mask_files: Optional[list[list[Optional[str]]]] = None,
        subject_landmark_files: Optional[list[list[Optional[str]]]] = None,
        epochs: int = 500,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        input_shape: tuple[int, int, int] = (175, 175, 175),
        similarity: str = "lncc",
        lambda_value: float = 1.5,
        dice_loss_weight: float = 0.5,
        lncc_sigma: int = 1,
        ct_window: tuple[float, float] = (-1000.0, 1000.0),
        is_ct: bool = True,
        gpus: Optional[list[int]] = None,
        eval_period: int = 10,
        save_period: int = 50,
        unigradicon_src_path: Optional[Path] = None,
        log_level: Union[int, str] = logging.INFO,
    ) -> None:
        """Initialize the ICON finetuning workflow.

        Args:
            subject_image_files: Per-subject lists of image file paths.  Each
                inner list groups frames belonging to one subject; all of those
                frames share a ``subject_id`` for paired training.
            output_dir: Directory for the dataset JSON, YAML config, and the
                uniGradICON checkpoint tree.
            finetune_name: Sub-directory name for the experiment outputs
                (used as the uniGradICON ``experiment.name`` stem).
            subject_ids: One ID per subject, in the same order as
                ``subject_image_files``.  Written verbatim into the dataset
                JSON's ``subject_id`` field so paired training groups frames
                that share an ID.  ``None`` falls back to synthetic IDs of the
                form ``subject_0000``, ``subject_0001``, ...  Must be unique.
            subject_labelmap_files: Per-subject multi-label segmentation
                (labelmap) paths matching ``subject_image_files``.  ``None``
                disables paired-with-seg training.
                Individual ``None`` entries inside the inner lists skip just
                those frames when paired-with-seg training is enabled.
            subject_mask_files: Per-subject binary mask paths matching
                ``subject_image_files``, used for ICON loss-function masking.
                ``None`` disables loss-function masking.  Per-image ``None``
                entries skip just those frames.
            subject_landmark_files: Per-subject landmark CSV paths matching
                ``subject_image_files``.  Stored in the dataset JSON for
                traceability; not consumed by uniGradICON finetuning.
            epochs: uniGradICON ``training.epochs``.
            batch_size: uniGradICON ``training.batch_size``.
            learning_rate: uniGradICON ``training.learning_rate``.
            input_shape: uniGradICON ``training.input_shape`` (voxels, X/Y/Z).
            similarity: uniGradICON ``training.similarity`` metric (e.g. ``lncc``).
            lambda_value: uniGradICON ``training.lambda`` regularization weight.
            dice_loss_weight: uniGradICON ``training.dice_loss_weight``.
            lncc_sigma: uniGradICON ``training.lncc_sigma``.
            ct_window: uniGradICON dataset ``ct_window`` ``[low, high]`` in HU.
            is_ct: Whether the dataset is CT (passes through to dataset config).
            gpus: GPU device indices for training.  Defaults to ``[0]``.
            eval_period: uniGradICON ``training.eval_period``.
            save_period: uniGradICON ``training.save_period``.
            unigradicon_src_path: Optional path to a local uniGradICON source
                tree to prepend to ``PYTHONPATH`` when running finetuning.
                Useful for using a checked-out copy instead of the installed
                package.
            log_level: Logging level (``logging.DEBUG``, ``logging.INFO``, ...).

        Raises:
            ValueError: If ``subject_image_files`` is empty.
            ValueError: If ``subject_labelmap_files``,
                ``subject_mask_files``, or ``subject_landmark_files`` is
                provided with a shape that does not match
                ``subject_image_files``.
        """
        super().__init__(
            class_name="WorkflowFinetuneICONRegistration", log_level=log_level
        )

        if not subject_image_files:
            raise ValueError("subject_image_files must not be empty")

        if subject_ids is not None:
            if len(subject_ids) != len(subject_image_files):
                raise ValueError(
                    f"subject_ids length ({len(subject_ids)}) must match "
                    f"subject_image_files length ({len(subject_image_files)})"
                )
            if len(set(subject_ids)) != len(subject_ids):
                raise ValueError(f"subject_ids must be unique, got {subject_ids}")

        self._validate_companion_shape(
            subject_image_files,
            subject_labelmap_files,
            "subject_labelmap_files",
        )
        self._validate_companion_shape(
            subject_image_files, subject_mask_files, "subject_mask_files"
        )
        self._validate_companion_shape(
            subject_image_files, subject_landmark_files, "subject_landmark_files"
        )

        self.subject_image_files = subject_image_files
        self.subject_ids = subject_ids
        self.subject_labelmap_files = subject_labelmap_files
        self.subject_mask_files = subject_mask_files
        self.subject_landmark_files = subject_landmark_files

        self.use_labelmaps: bool = subject_labelmap_files is not None
        self.use_masks: bool = subject_mask_files is not None

        self.output_dir = Path(output_dir).resolve()
        self.finetune_name = finetune_name
        self.experiment_dir = self.output_dir / finetune_name

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_shape = tuple(input_shape)
        self.similarity = similarity
        self.lambda_value = lambda_value
        self.dice_loss_weight = dice_loss_weight
        self.lncc_sigma = lncc_sigma
        self.ct_window = tuple(ct_window)
        self.is_ct = is_ct
        self.gpus = list(gpus) if gpus is not None else [0]
        self.eval_period = eval_period
        self.save_period = save_period
        self.unigradicon_src_path = (
            Path(unigradicon_src_path) if unigradicon_src_path is not None else None
        )

        self._use_labelmaps: bool = self.use_labelmaps
        self._use_masks: bool = self.use_masks

        self._dataset_json_path: Optional[Path] = None
        self._config_yaml_path: Optional[Path] = None

    @staticmethod
    def _validate_companion_shape(
        image_files: list[list[str]],
        companion: Optional[list[list[Optional[str]]]],
        name: str,
    ) -> None:
        """Confirm a companion list has the same nested shape as ``image_files``."""
        if companion is None:
            return
        if len(companion) != len(image_files):
            raise ValueError(
                f"{name} length ({len(companion)}) must match "
                f"subject_image_files length ({len(image_files)})"
            )
        for i, (images, items) in enumerate(zip(image_files, companion, strict=True)):
            if len(items) != len(images):
                raise ValueError(
                    f"{name}[{i}] length ({len(items)}) must match "
                    f"subject_image_files[{i}] length ({len(images)})"
                )

    @staticmethod
    def _posix(path: Union[str, Path]) -> str:
        """Return a forward-slashed string path (uniGradICON expects POSIX paths)."""
        return str(path).replace("\\", "/")

    def prepare_dataset(
        self,
        use_labelmaps: Optional[bool] = None,
        use_masks: Optional[bool] = None,
    ) -> Path:
        """Write the uniGradICON dataset JSON from the configured file lists.

        Builds one entry per image with ``image``, optional ``segmentation``,
        optional ``mask``, optional ``landmarks`` (path only), and a
        ``subject_id`` derived from the inner-list index.

        Masks come from ``subject_mask_files`` only; none are derived.  Frames
        are skipped (with a log warning) when a required companion
        (segmentation for paired-with-seg training, or mask for loss-function
        masking) is missing.

        Returns:
            Path to the dataset JSON written under :attr:`experiment_dir`.

        Raises:
            FileNotFoundError: If an image listed in ``subject_image_files``
                does not exist on disk.
        """
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        if use_labelmaps is None:
            use_labelmaps = self.use_labelmaps
        if use_masks is None:
            use_masks = self.use_masks

        self._use_labelmaps = use_labelmaps
        self._use_masks = use_masks

        dataset_entries: list[dict[str, str]] = []
        for subject_index, image_files in enumerate(self.subject_image_files):
            subject_id = (
                self.subject_ids[subject_index]
                if self.subject_ids is not None
                else f"subject_{subject_index:04d}"
            )
            seg_list: list[Optional[str]]
            if not use_labelmaps:
                seg_list = [None] * len(image_files)
            else:
                seg_list = (
                    self.subject_labelmap_files[subject_index]
                    if self.subject_labelmap_files is not None
                    else [None] * len(image_files)
                )
            mask_list: list[Optional[str]]
            if not use_masks:
                mask_list = [None] * len(image_files)
            else:
                mask_list = (
                    self.subject_mask_files[subject_index]
                    if self.subject_mask_files is not None
                    else [None] * len(image_files)
                )
            landmark_list = (
                self.subject_landmark_files[subject_index]
                if self.subject_landmark_files is not None
                else [None] * len(image_files)
            )

            for image_file, seg_file, mask_file, landmark_file in zip(
                image_files, seg_list, mask_list, landmark_list, strict=True
            ):
                image_path = Path(image_file)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                entry: dict[str, str] = {
                    "image": self._posix(image_path),
                    "subject_id": subject_id,
                }

                if use_labelmaps:
                    if seg_file is None or not Path(seg_file).exists():
                        self.log_warning(
                            "Skipping %s: segmentation missing for paired-with-seg "
                            "training (seg=%s)",
                            image_path,
                            seg_file,
                        )
                        continue
                    entry["segmentation"] = self._posix(seg_file)

                if use_masks:
                    if mask_file is None or not Path(mask_file).exists():
                        self.log_warning(
                            "Skipping %s: mask missing for loss-function masking "
                            "(mask=%s)",
                            image_path,
                            mask_file,
                        )
                        continue
                    entry["mask"] = self._posix(mask_file)

                if landmark_file is not None:
                    entry["landmarks"] = self._posix(landmark_file)

                dataset_entries.append(entry)

        dataset_json_path = self.experiment_dir / f"{self.finetune_name}_dataset.json"
        with dataset_json_path.open("w") as fh:
            json.dump({"data": dataset_entries}, fh, indent=2)

        self.log_info(
            "Wrote dataset JSON %s with %d entries",
            dataset_json_path,
            len(dataset_entries),
        )
        self._dataset_json_path = dataset_json_path
        return dataset_json_path

    def prepare_config(self, dataset_json_path: Optional[Path] = None) -> Path:
        """Write the uniGradICON finetuning YAML config.

        Args:
            dataset_json_path: Path to the dataset JSON to reference.  Defaults
                to the JSON last produced by :meth:`prepare_dataset`.

        Returns:
            Path to the YAML config written under :attr:`experiment_dir`.

        Raises:
            ValueError: If no dataset JSON path is available.
        """
        if dataset_json_path is None:
            dataset_json_path = self._dataset_json_path
        if dataset_json_path is None:
            raise ValueError(
                "dataset_json_path not provided and prepare_dataset() has not "
                "been called yet"
            )

        experiment_name = self.experiment_dir / f"{self.finetune_name}_model"

        config: dict[str, Any] = {
            "experiment": {
                "name": self._posix(experiment_name),
                "model_weights": "unigradicon",
            },
            "training": {
                "batch_size": self.batch_size,
                "gpus": self.gpus,
                "epochs": self.epochs,
                "eval_period": self.eval_period,
                "save_period": self.save_period,
                "learning_rate": self.learning_rate,
                "input_shape": list(self.input_shape),
                "similarity": self.similarity,
                "lambda": self.lambda_value,
                "dice_loss_weight": self.dice_loss_weight,
                "lncc_sigma": self.lncc_sigma,
                "loss_function_masking": self._use_masks,
                "use_label": False,
                "roi_masking": False,
            },
            "datasets": [
                {
                    "name": self.finetune_name,
                    "weight": 1.0,
                    "type": "paired",
                    "json_file": self._posix(dataset_json_path),
                    "is_ct": self.is_ct,
                    "ct_window": list(self.ct_window),
                    "shuffle": True,
                    "use_cache": True,
                }
            ],
        }

        config_yaml_path = self.experiment_dir / f"{self.finetune_name}_config.yaml"
        with config_yaml_path.open("w") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
        self.log_info("Wrote config YAML %s", config_yaml_path)
        self._config_yaml_path = config_yaml_path
        return config_yaml_path

    def expected_weights_path(self) -> Path:
        """Return the path uniGradICON writes its final checkpoint to.

        ``unigradicon.finetuning.finetune`` writes
        ``<experiment.name>/checkpoints/network_weights_final.trch`` at the end
        of training -- its ``NETWORK_WEIGHTS_PREFIX`` plus the ``"final"`` epoch
        label.  Also the return value of :meth:`process`.

        The filename is hard-coded rather than imported: training runs in a
        subprocess so that this process never imports ``unigradicon`` (its
        ``finetuning`` submodule exists only on the ``feat-add-finetuning``
        branch, so an import here would raise on a stock install).  An upstream
        rename is caught by ``test_expected_weights_path_layout``, which
        compares this filename against ``NETWORK_WEIGHTS_PREFIX`` wherever the
        submodule is installed, and by the ``FileNotFoundError`` :meth:`process`
        raises when the checkpoint is not where this says it should be.
        """
        return (
            self.experiment_dir
            / f"{self.finetune_name}_model"
            / "checkpoints"
            / "network_weights_final.trch"
        )

    def process(self) -> Path:
        """Build configs and launch ``unigradicon.finetuning.finetune``.

        Equivalent to running
        ``prepare_dataset()`` → ``prepare_config()`` → subprocess launch.  Any
        existing dataset JSON or YAML in :attr:`experiment_dir` is overwritten.

        Returns:
            Path to the final checkpoint (``network_weights_final.trch``).  The
            file is written by the subprocess and exists only after a successful
            run.

        Raises:
            subprocess.CalledProcessError: If the finetuning subprocess exits
                with a non-zero status.
            FileNotFoundError: If the subprocess succeeded but the checkpoint is
                not where :meth:`expected_weights_path` says it should be.
        """
        self.log_section("FINETUNING UNIGRADICON", width=70)

        dataset_json_path = self.prepare_dataset()
        config_yaml_path = self.prepare_config(dataset_json_path)

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        if self.unigradicon_src_path is not None:
            env["PYTHONPATH"] = (
                str(self.unigradicon_src_path) + os.pathsep + env.get("PYTHONPATH", "")
            )

        cmd = [
            sys.executable,
            "-m",
            "unigradicon.finetuning.finetune",
            "--config",
            str(config_yaml_path),
        ]
        self.log_info("Launching finetuning subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

        # A missing checkpoint here is silent otherwise: uniGradICON treats an
        # unknown weights path as a download destination, so a stale filename,
        # or a run directory renamed with a "-N" suffix by the ``footsteps``
        # package uniGradICON uses to lay out its runs, would yield
        # stock-weight registrations that look finetuned.
        weights_path = self.expected_weights_path()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Finetuning finished but no checkpoint at {weights_path}.  "
                "Check for a suffixed run directory (uniGradICON appends "
                "'-1', '-2', ... when the experiment directory already exists)."
            )
        self.log_info("Finetuning complete. Weights at %s", weights_path)
        return weights_path
