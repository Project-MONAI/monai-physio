"""Shared parameters for the Duke-Heart-4DLabelmaps tutorials.

Mirrors :mod:`parameters_heart_ct_kcl` for the ``duke_heart`` tutorials, which
read segmented labelmaps rather than CT and therefore need their own label ids
and their own held-out case.  The registration values match the KCL heart ones
on purpose: Tutorial 2 finetunes uniGradICON on distance maps whose appearance
is fixed by ``mask_dilation_mm`` and ``distancemap_squared_max``, and Tutorial 7
infers with those weights, so the two must agree.

The directories and the shape-model files the tutorials read and write live here
too, so that Tutorial 6 writes the model where Tutorial 7 looks for it.  Every
path is derived from this file's location, so they hold wherever the clone is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from physiotwin4d import SegmentAnatomyBase, SegmentHeartSimplewareTrimmedBranches

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TUTORIAL_OUTPUT = _REPO_ROOT / "tutorials" / "output"
_TUTORIAL_WEIGHTS = _REPO_ROOT / "tutorials" / "network_weights"


@dataclass(frozen=True)
class ParametersDukeHeartLabelmaps:
    """Settings shared by the Duke heart tutorials.

    Attributes:
        icp_transform_type: Alignment applied to a surface before the shape
            model corresponds or fits it, one of ``"Rigid"``, ``"Similarity"``
            or ``"Affine"``.  Model building and model fitting must use the
            same value: whatever the ICP absorbs is variation the eigenmodes
            never see, so a mismatch makes the fit ask the modes to explain
            shape that has already been removed.
        mask_dilation_mm: Dilation of the binary registration masks, in
            millimeters.
        distancemap_squared_max: Saturation radius of every heart distance map,
            in squared millimeters.  Fixes their intensity distribution, so the
            finetuning tutorial and every tutorial that registers these distance
            maps must use this one value.
        surface_spacing_mm: Isotropic pitch of the grid every heart surface is
            contoured on.  Finer than the slice pitch of this data, which is
            what lets a boundary land between two slices instead of terracing at
            one of them.
        surface_smoothing_iterations: Taubin smoothing iterations applied to
            those surfaces.
        mesh_element_size_mm: Edge length of the isotropic voxels a labelmap is
            resampled to before it is meshed into tetrahedra, which is the
            resulting element size.  Below the thinnest wall of the heart, so
            that the myocardium survives the coarsening.
        model_points: Points kept per surface when building the shape model.
            The contours carry thirty times this, which a model built from a few
            dozen patients cannot support and every PCA mode would have to
            carry as one vector of ``3 * model_points``.
        number_of_pca_components: PCA components retained when building the
            heart statistical model, and used when fitting it to a patient.
        number_of_pca_components_test: Same, under ``TestTools.running_as_test``.
        number_of_iterations_greedy: Greedy coarse-to-fine iteration schedule.
        number_of_iterations_greedy_test: Same, under
            ``TestTools.running_as_test``.
        segmenter_class: Segmenter that produced these labelmaps, so the
            tutorials name their labels the way it does.
        anatomy_group: Anatomy group name that segmenter registers for the heart.
        interior_object_ids: Labels left out of the whole-heart structure, and
            therefore never measured to by a distance map.  The four chambers
            (1-4) cover the cavities rather than their walls, so measuring to
            them measures the inside of the heart; the great vessels and
            coronaries (7-10) vary too much in extent between patients to be
            part of a shape model.  What remains is the myocardium (5) and the
            heart wall (6).
        input_dir: Surfaces Tutorial 6 builds the model from, which are what
            Tutorial 4 wrote; ``input_dir_test`` is its counterpart under
            ``TestTools.running_as_test``.
        hold_out_dir: Labelmaps the held-out case is read from by Tutorials 2
            and 7, and ``hold_out_dir_test`` its counterpart under
            ``TestTools.running_as_test``.
        pca_json_file: Shape model Tutorial 6 writes and Tutorials 7, 8 and 9
            read.
        pca_mean_file: Mean surface of that model, written and read the same way.
        mgn_weights_dir: MeshGraphNet Tutorial 9 trains and Tutorial 10 infers
            with, alongside the normalization statistics and PCA assets that
            make that directory self-contained.  It sits beside the finetuned
            ICON checkpoints rather than under ``output/``, because it is a
            trained network rather than a per-run result.
        hold_out_case: Case held out of every fit: Tutorial 6 builds the shape
            model without it and Tutorial 7 fits that model to it, so the fit
            measures generalization rather than reconstruction.  Tutorial 2
            scores its registrations on the same case.
    """

    icp_transform_type: str = "Similarity"

    mask_dilation_mm: float = 10.0
    distancemap_squared_max: float = (1.25 * 10.0) ** 2

    surface_spacing_mm: float = 0.5
    surface_smoothing_iterations: int = 20
    mesh_element_size_mm: float = 1.5

    model_points: int = 20000
    number_of_pca_components: int = 10
    number_of_pca_components_test: int = 5

    number_of_iterations_greedy: list[int] = field(
        default_factory=lambda: [30, 15, 7, 3]
    )
    number_of_iterations_greedy_test: list[int] = field(default_factory=lambda: [1, 0])

    segmenter_class: type[SegmentAnatomyBase] = SegmentHeartSimplewareTrimmedBranches
    anatomy_group: str = "heart"
    interior_object_ids: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 7, 8, 9, 10]
    )

    hold_out_case: str = "pm0027"

    input_dir: Path = _TUTORIAL_OUTPUT / "tutorial_04_duke_heart_labelmap"
    input_dir_test: Path = _TUTORIAL_OUTPUT / "tutorial_04_duke_heart_labelmap"
    hold_out_dir: Path = _REPO_ROOT / "data" / "Duke-Heart-4DLabelmaps"
    hold_out_dir_test: Path = _REPO_ROOT / "data" / "test" / "Duke-Heart-4DLabelmaps"
    pca_json_file: Path = _TUTORIAL_OUTPUT / "tutorial_06_duke_heart" / "pca_model.json"
    pca_mean_file: Path = (
        _TUTORIAL_OUTPUT / "tutorial_06_duke_heart" / "pca_mean_surface.vtp"
    )
    mgn_weights_dir: Path = _TUTORIAL_WEIGHTS / "physicsnemo_mgn_duke_heart_motion"

    def input_directory(self, test_mode: bool) -> Path:
        """Return the model population directory for this run mode.

        Tutorial 4 writes to one directory whichever mode it ran in, so both
        modes read the same one.
        """
        return self.input_dir_test if test_mode else self.input_dir

    def hold_out_directory(self, test_mode: bool) -> Path:
        """Return the held-out case's directory for this run mode."""
        return self.hold_out_dir_test if test_mode else self.hold_out_dir

    def pca_components(self, test_mode: bool) -> int:
        """Return the PCA component count for this run mode."""
        return (
            self.number_of_pca_components_test
            if test_mode
            else self.number_of_pca_components
        )

    def greedy_iterations(self, test_mode: bool) -> list[int]:
        """Return the Greedy iteration schedule for this run mode."""
        return list(
            self.number_of_iterations_greedy_test
            if test_mode
            else self.number_of_iterations_greedy
        )


#: The single instance every Duke heart tutorial imports.
DUKE_HEART = ParametersDukeHeartLabelmaps()
