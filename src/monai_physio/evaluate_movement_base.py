"""What one cohort of subjects is scored against, and how to assemble it.

:class:`EvaluateMovementBase` holds the parts of a movement evaluation that
depend on *which cohort* is being scored -- the structures to report, how a
stage is read off a filename, and how that cohort's ground truth is gathered --
so that :class:`monai_physio.WorkflowEvaluateMovement` can hold the parts that
do not. The workflow composes a cohort the way
:class:`monai_physio.WorkflowInferPhysicsNeMo` composes an inference method.

Keeping the cohort out of the workflow is what stops the workflow growing
anatomy branches. It scores whatever structures it is handed, by whatever
partition the shape model supports, and never learns which cohort produced them.

Concrete cohorts: :class:`monai_physio.EvaluateMovementLung` and
:class:`monai_physio.EvaluateMovementDukeHeart`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import itk

from .monai_physio_base import MONAIPhysioBase
from .segment_anatomy_base import SegmentAnatomyBase


@dataclass(frozen=True)
class MovementGroundTruth:
    """One case's acquired frames and the fits its prediction is scored against.

    Attributes:
        labelmaps: Acquired labelmap per stage, keyed by the normalized stage in
            ``[0, 1]``. Every volume and overlap figure is measured against
            these.
        reference_labelmap: Labelmap of the frame every predicted deformation
            starts from, the anatomy carried into every other stage.
        reference_stage: Which stage that frame is.
        meshes: The case's fitted surface per stage, keyed the same way and
            sharing the fitted reference mesh's topology and point ordering. The
            acquired labelmaps carry no point correspondence, so these are the
            only source of a true per-point displacement.
    """

    labelmaps: dict[float, itk.Image]
    reference_labelmap: itk.Image
    reference_stage: float
    meshes: dict[float, Path]


class EvaluateMovementBase(MONAIPhysioBase):
    """A cohort of subjects and the ground truth its movement is scored against.

    Not instantiated directly -- use :class:`monai_physio.EvaluateMovementLung`
    or :class:`monai_physio.EvaluateMovementDukeHeart`. Subclasses set the class
    attributes below and implement :meth:`stage_from_filename` and
    :meth:`assemble_ground_truth`; :meth:`display_name` is an optional hook.

    Attributes:
        segmenter_class: Segmenter whose taxonomy names this cohort's
            structures. Instantiated for its taxonomy alone, which downloads and
            loads nothing.
        label_ids: The structures to score, in report order.
        report_dice: Whether Dice belongs in this cohort's report. Off for a
            structure whose motion is small against its own size: Dice is an
            overlap fraction, so a lung lobe scores over 0.96 undeformed.
        evaluation_spacing_mm: Isotropic pitch every metric is measured on.
        smoothing_sigma_mm: Gaussian sigma, in millimeters, that turns the
            network's surface-shell deformation into a continuous field.

    Args:
        log_level: Logging level. Default: ``logging.INFO``.
    """

    segmenter_class: Optional[type[SegmentAnatomyBase]] = None
    label_ids: tuple[int, ...] = ()
    report_dice: bool = True
    evaluation_spacing_mm: float = 1.0
    smoothing_sigma_mm: float = 10.0

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    # ─────────────────────────── Public API ────────────────────────────────
    def label_names(self) -> dict[int, str]:
        """The structures this cohort scores, ``{label id: report name}``.

        Read from :attr:`segmenter_class`'s taxonomy and passed through
        :meth:`display_name`, so a cohort never restates names the segmenter
        already knows.

        Raises:
            NotImplementedError: On a cohort that names no segmenter.
        """
        if self.segmenter_class is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} names no segmenter_class, so it "
                "cannot look its structures up; set that attribute or override "
                "label_names."
            )
        # Constructed for the taxonomy alone: no weights are fetched and no
        # model is loaded until a segmentation is actually asked for.
        taxonomy = self.segmenter_class(log_level=logging.WARNING).taxonomy
        all_labels = taxonomy.all_labels()
        return {
            label_id: self.display_name(label_id, all_labels[label_id])
            for label_id in self.label_ids
        }

    def display_name(self, label_id: int, taxonomy_name: str) -> str:
        """The report name of one structure. The base reports the taxonomy's own.

        Overridden by a cohort whose taxonomy name would mislead a reader of the
        report.
        """
        return taxonomy_name

    def stage_from_filename(self, path: Path) -> float:
        """The normalized stage in ``[0, 1]`` this frame was acquired at.

        Raises:
            NotImplementedError: Implemented by subclasses.
        """
        raise NotImplementedError

    def assemble_ground_truth(
        self,
        case_id: str,
        frame_directory: Path,
        fit_directory: Path,
        cache_directory: Optional[Path] = None,
    ) -> MovementGroundTruth:
        """Gather one case's acquired frames and its per-stage fits.

        Args:
            case_id: The case being scored.
            frame_directory: Where that case's acquired frames live.
            fit_directory: Where its per-stage ``*_ssm_surface.vtp`` fits live --
                Tutorial 8's case directory, or one leave-one-out fold's own
                fits, which is what makes a fold's score honest.
            cache_directory: Where a cohort that has to *derive* its labelmaps
                writes them and reads them back. A cohort whose labelmaps ship
                with the data ignores it.

        Raises:
            NotImplementedError: Implemented by subclasses.
        """
        raise NotImplementedError
