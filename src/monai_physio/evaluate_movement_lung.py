"""The DIR-Lab gated-CT lung cohort, as a movement-evaluation subject.

This cohort ships CT, not labelmaps, so its ground truth has to be *derived*:
every gated frame is segmented on its own, which means the lobes a phase is
scored against came from that phase's image rather than from a registration or a
shape-model fit. Segmentation dominates the runtime, so each labelmap is cached
and reused on a re-run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import itk

from .evaluate_movement_base import EvaluateMovementBase, MovementGroundTruth
from .segment_nv_segment_ct_mri import SegmentNVSegmentCTMRI


class EvaluateMovementLung(EvaluateMovementBase):
    """The five lung lobes, scored against per-phase segmentations of gated CT.

    Args:
        reference_phase: The phase the shape model was fitted to, and therefore
            the one whose anatomy the predicted deformations carry into every
            other phase. Default: ``"T70"``.
        log_level: Logging level. Default: ``logging.INFO``.
    """

    segmenter_class = SegmentNVSegmentCTMRI
    # The five lobes of ``SegmentNVSegmentCTMRI``. Its "lung" group also carries
    # whole-lung, tumor and airway labels, which are not lobes.
    label_ids = (28, 29, 30, 31, 32)
    # A lobe barely changes shape over a breath compared to how big it is, so
    # Dice says more about the lobe than about the motion. Volume difference and
    # surface RMSE are what resolve it here.
    report_dice = False
    # Coarser than the CT, whose in-plane pitch is finer than the accuracy being
    # reported, and fine enough that a lobe boundary is not quantized away.
    evaluation_spacing_mm = 2.0

    def __init__(
        self, reference_phase: str = "T70", log_level: int | str = logging.INFO
    ) -> None:
        super().__init__(log_level=log_level)
        self.reference_phase = reference_phase

    def stage_from_filename(self, path: Path) -> float:
        """Read the normalized respiratory stage from a ``T{PP}`` filename stem."""
        for part in path.stem.split("_"):
            if part.startswith("T") and part[1:].isdigit():
                return int(part[1:]) / 100.0
        raise ValueError(f"Cannot parse respiratory phase from filename: {path}")

    def assemble_ground_truth(
        self,
        case_id: str,
        frame_directory: Path,
        fit_directory: Path,
        cache_directory: Optional[Path] = None,
    ) -> MovementGroundTruth:
        """Segment every gated frame of one case, then read its per-phase fits.

        ``cache_directory`` is required: a segmentation pass per phase is the
        expensive part of an evaluation, so it is written once and reused. A
        caller that has already produced those labelmaps under the same names --
        a leave-one-out study building a shared cache, say -- gets them read
        straight back and pays nothing.

        Raises:
            ValueError: If ``cache_directory`` is not given.
            FileNotFoundError: If the case has no gated frames, no per-phase
                fits, or no frame at :attr:`reference_phase`.
        """
        if cache_directory is None:
            raise ValueError(
                "EvaluateMovementLung needs a cache_directory: it segments every "
                "gated frame, which is too expensive to repeat on every run."
            )
        frame_files = sorted(frame_directory.glob(f"{case_id}_T??.mha"))
        if not frame_files:
            raise FileNotFoundError(
                f"No {case_id}_T??.mha frames found under {frame_directory}.\n"
                "See data/DirLab-4DCT/README.md for download instructions."
            )
        cache_directory.mkdir(parents=True, exist_ok=True)

        segmenter = self.segmenter_class(log_level=self.log_level)
        labelmaps: dict[float, itk.Image] = {}
        for frame_file in frame_files:
            labelmap_file = cache_directory / f"{frame_file.stem}_labelmap.nii.gz"
            if not labelmap_file.exists():
                self.log_info("Segmenting ground-truth frame %s", frame_file.name)
                segmentation = segmenter.segment(itk.imread(str(frame_file)))
                itk.imwrite(
                    segmentation["labelmap"], str(labelmap_file), compression=True
                )
            labelmaps[self.stage_from_filename(frame_file)] = itk.imread(
                str(labelmap_file)
            )

        reference_file = (
            cache_directory / f"{case_id}_{self.reference_phase}_labelmap.nii.gz"
        )
        if not reference_file.exists():
            raise FileNotFoundError(
                f"Reference phase {self.reference_phase} is not among "
                f"{frame_directory}'s frames, so {reference_file} was never "
                "segmented."
            )

        surface_files = sorted(fit_directory.glob(f"{case_id}_T??_ssm_surface.vtp"))
        if not surface_files:
            raise FileNotFoundError(
                f"No per-phase SSM surfaces found in {fit_directory}.\n"
                "Run tutorials/tutorial_08_lung_fit_model_to_4d_patients.py first."
            )

        return MovementGroundTruth(
            labelmaps=labelmaps,
            reference_labelmap=itk.imread(str(reference_file)),
            reference_stage=self.stage_from_filename(reference_file),
            meshes={
                self.stage_from_filename(surface_file): surface_file
                for surface_file in surface_files
            },
        )
