"""The Duke gated-labelmap heart cohort, as a movement-evaluation subject.

Unlike the lung, this cohort ships one labelmap per gated frame, each already
carrying the four chambers, the myocardium and the whole heart, so nothing has
to be segmented and there is nothing to cache.

The shape model this cohort scores is one structure -- the whole heart minus its
chamber cavities -- so the chambers exist only in these acquired labelmaps.
Scoring through them is what makes per-chamber figures possible at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import itk

from .evaluate_movement_base import EvaluateMovementBase, MovementGroundTruth
from .segment_heart_simpleware_trimmed_branches import (
    SegmentHeartSimplewareTrimmedBranches,
)


class EvaluateMovementDukeHeart(EvaluateMovementBase):
    """The heart chambers, scored against the labelmaps the cohort ships.

    Args:
        log_level: Logging level. Default: ``logging.INFO``.
    """

    segmenter_class = SegmentHeartSimplewareTrimmedBranches
    # The four chambers, plus the myocardium and the whole heart for context: 5
    # and 6 are what the shape model itself represents, 1-4 are the cavities it
    # does not. The great vessels and coronaries (7-10) are left out; they come
    # and go between frames and are not part of the model.
    label_ids = (1, 2, 3, 4, 5, 6)
    report_dice = True
    # Coarser than these labelmaps, whose in-plane pitch is finer than the
    # accuracy being reported, and still below the thinnest wall of the heart.
    evaluation_spacing_mm = 1.0

    labelmap_suffix = "_labelmap.nii.gz"
    # Tutorial 8's per-frame SSM fits, the same surfaces Tutorial 10 scores.
    phase_surface_pattern = "*_g[0-9][0-9][0-9]_*_ssm_surface.vtp"

    def display_name(self, label_id: int, taxonomy_name: str) -> str:
        """Report the taxonomy's "heart" as "heart muscle".

        The taxonomy's ``heart`` is the whole heart *minus* its chamber
        cavities -- the muscle the shape model represents -- and a report that
        listed it beside four chambers under the bare name "heart" would read as
        if it were their sum.
        """
        return "heart muscle" if taxonomy_name == "heart" else taxonomy_name

    def stage_from_filename(self, path: Path) -> float:
        """Read the normalized cardiac stage from a ``g{PPP}`` filename.

        Matched against the full name rather than the stem, since ``.nii.gz`` is
        a double suffix that :attr:`Path.stem` only half removes.
        """
        for part in path.name.split("_"):
            if part.startswith("g") and part[1:].isdigit():
                return int(part[1:]) / 100.0
        raise ValueError(f"Cannot parse cardiac gate from filename: {path}")

    def assemble_ground_truth(
        self,
        case_id: str,
        frame_directory: Path,
        fit_directory: Path,
        cache_directory: Optional[Path] = None,
    ) -> MovementGroundTruth:
        """Read one case's gated labelmaps and its per-frame fits.

        ``cache_directory`` is ignored: these labelmaps are acquired, not
        derived, so there is nothing to cache.

        Raises:
            FileNotFoundError: If the case has no gated labelmaps, no frame
                marked ``*_ref``, or no per-frame fits.
        """
        frame_files = sorted(frame_directory.glob(f"*{self.labelmap_suffix}"))
        if not frame_files:
            raise FileNotFoundError(
                f"No gated labelmaps found in {frame_directory}.\n"
                "See data/Duke-Heart-4DLabelmaps/README.md."
            )
        reference_files = [
            path
            for path in frame_files
            if path.name.endswith(f"_ref{self.labelmap_suffix}")
        ]
        if not reference_files:
            raise FileNotFoundError(
                f"No *_ref{self.labelmap_suffix} frame in {frame_directory}; "
                "Tutorial 8 fitted the SSM to that frame, so it is the one the "
                "deformations start from."
            )

        surface_files = sorted(fit_directory.glob(self.phase_surface_pattern))
        if not surface_files:
            raise FileNotFoundError(
                f"No per-frame SSM surfaces found in {fit_directory}.\n"
                "Run tutorials/tutorial_08_duke_heart_fit_model_to_4d_patients.py "
                "first."
            )

        self.log_info("Case %s: %d gated frames", case_id, len(frame_files))
        return MovementGroundTruth(
            labelmaps={
                self.stage_from_filename(frame_file): itk.imread(str(frame_file))
                for frame_file in frame_files
            },
            reference_labelmap=itk.imread(str(reference_files[0])),
            reference_stage=self.stage_from_filename(reference_files[0]),
            meshes={
                self.stage_from_filename(surface_file): surface_file
                for surface_file in surface_files
            },
        )
