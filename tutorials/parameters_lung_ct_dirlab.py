"""Shared parameters for the DIR-Lab 4D CT lung tutorials.

Every lung tutorial reads its settings from :data:`LUNG_CT_DIRLAB` so the
distance maps one tutorial finetunes ICON on are rasterized exactly the way the
tutorials that later register them rasterize theirs.  A saturation radius or a
dilation that drifts between two of these scripts silently trains on one image
distribution and infers on another.

The directories and the shape-model files the tutorials read and write live here
too, so that Tutorial 6 writes the model where Tutorials 7 and 8 look for it.
Every path is derived from this file's location, so they hold wherever the clone
is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from physiotwin4d import SegmentAnatomyBase, SegmentNVSegmentCTMRI

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_ROOT = _REPO_ROOT / "tutorials" / "output" / "tutorial_06_lung"


@dataclass(frozen=True)
class ParametersLungCTDirLab:
    """Settings shared by the DIR-Lab lung tutorials.

    Attributes:
        icp_transform_type: Alignment applied to a surface before the shape
            model corresponds or fits it, one of ``"Rigid"``, ``"Similarity"``
            or ``"Affine"``.  Model building and model fitting must use the
            same value: whatever the ICP absorbs is variation the eigenmodes
            never see, so a mismatch makes the fit ask the modes to explain
            shape that has already been removed.
        mask_dilation_mm: Dilation of the binary registration masks, in
            millimeters.  Also sets how far outside the lung surface the
            registration is allowed to look.
        distancemap_squared_max: Saturation radius of every lung distance map,
            in squared millimeters.  Fixes their intensity distribution, so the
            finetuning tutorial and every tutorial that registers lung distance
            maps must use this one value.
        surface_reduction_rate: Fraction of triangles removed from every
            extracted lung surface.  ``0.0`` keeps them at full resolution.
        mesh_reduction_rate: Fraction of the voxel resolution removed along
            each axis before a labelmap is meshed into tetrahedra, so the
            tetrahedron count falls by roughly ``(1 - rate) ** 3``.
        number_of_pca_components: PCA components retained when building the
            lung statistical model, and used when fitting it to a patient.
        number_of_pca_components_test: Same, under ``TestTools.running_as_test``.
        number_of_iterations_greedy: Greedy coarse-to-fine iteration schedule.
        number_of_iterations_greedy_test: Same, under
            ``TestTools.running_as_test``.
        segmenter_class: Segmenter every lung tutorial instantiates, so the
            surfaces they compare share a definition of "lung".
        anatomy_group: Anatomy group name that segmenter registers for lungs.
        input_dir: Population Tutorial 6 builds the model from, and
            ``input_dir_test`` its counterpart under
            ``TestTools.running_as_test``.
        hold_out_dir: Dataset the held-out study is read from by Tutorial 7, and
            ``hold_out_dir_test`` its counterpart under
            ``TestTools.running_as_test``.
        pca_json_file: Shape model Tutorial 6 writes and Tutorials 7 and 8 read.
        pca_mean_file: Mean surface of that model, written and read the same way.
        hold_out_case: Image fitted by Tutorial 7 and therefore kept out of the
            population Tutorial 6 builds the model from, so that the fit
            measures generalization rather than reconstruction.  It is a Chest-CT
            study while the model is built from DIR-Lab phases, so today the
            exclusion never fires; Tutorial 6 applies it anyway, so adding the
            study to that population cannot slip it in.  Tutorial 2 holds out a
            DIR-Lab case of its own, which measures registration rather than
            shape.
        mgn_weights_dir: Directory the lung-motion MeshGraphNet is trained into
            by Tutorial 9 and loaded from by Tutorial 10, beside the ICON
            weights the registration tutorials finetune.  Tutorial 9 writes to a
            numbered sibling of it when resuming from a checkpoint, in which
            case Tutorial 10 has to be pointed at that sibling.
        mgn_hold_out_case: DIR-Lab case kept out of the Tutorial 9 training and
            predicted by Tutorial 10, so that the prediction measures
            generalization.  Distinct from ``hold_out_case``, which is the
            Chest-CT study the shape model is fitted to: this one is a 4D case
            Tutorial 8 has fitted every phase of.  It is also the case Tutorial
            2 holds out of its ICON finetuning, so the surfaces it is scored on
            came from a registration network that never saw it either.

    There is no interior-structure list here, the counterpart of the heart's
    chamber ids: the lung labels are the lobes, and every one of them is on the
    surface a distance map is measured to.
    """

    icp_transform_type: str = "Affine"

    mask_dilation_mm: float = 40.0
    distancemap_squared_max: float = (1.25 * 40.0) ** 2

    surface_reduction_rate: float = 0.0
    mesh_reduction_rate: float = 0.0

    number_of_pca_components: int = 6
    number_of_pca_components_test: int = 5

    number_of_iterations_greedy: list[int] = field(
        default_factory=lambda: [30, 15, 7, 3]
    )
    number_of_iterations_greedy_test: list[int] = field(default_factory=lambda: [1, 0])

    segmenter_class: type[SegmentAnatomyBase] = SegmentNVSegmentCTMRI
    anatomy_group: str = "lung"

    hold_out_case: str = "Chest-CT.mha"
    mgn_hold_out_case: str = "Case1Pack"

    input_dir: Path = _REPO_ROOT / "data" / "DirLab-4DCT"
    input_dir_test: Path = _REPO_ROOT / "data" / "test" / "DirLab-4DCT"
    hold_out_dir: Path = _REPO_ROOT / "data" / "Chest-CT"
    hold_out_dir_test: Path = _REPO_ROOT / "data" / "test" / "Chest-CT"
    pca_json_file: Path = _OUTPUT_ROOT / "pca_model.json"
    pca_mean_file: Path = _OUTPUT_ROOT / "pca_mean_surface.vtp"
    mgn_weights_dir: Path = (
        _REPO_ROOT / "tutorials" / "network_weights" / "physicsnemo_mgn_lung_motion"
    )

    def input_directory(self, test_mode: bool) -> Path:
        """Return the model population directory for this run mode."""
        return self.input_dir_test if test_mode else self.input_dir

    def hold_out_directory(self, test_mode: bool) -> Path:
        """Return the held-out study's directory for this run mode."""
        return self.hold_out_dir_test if test_mode else self.hold_out_dir

    def pca_components(self, test_mode: bool) -> int:
        """Return the PCA component count for this run mode."""
        return (
            self.number_of_pca_components_test
            if test_mode
            else (self.number_of_pca_components)
        )

    def greedy_iterations(self, test_mode: bool) -> list[int]:
        """Return the Greedy iteration schedule for this run mode."""
        return list(
            self.number_of_iterations_greedy_test
            if test_mode
            else self.number_of_iterations_greedy
        )


#: The single instance every lung tutorial imports.
LUNG_CT_DIRLAB = ParametersLungCTDirLab()
