"""Shared parameters for the heart CT tutorials.

Mirrors :mod:`parameters_lung_ct_dirlab` for the heart use cases, and carries
different values: the heart is registered with a much tighter mask than the
lungs, so its distance maps saturate over a correspondingly shorter radius.
That is why the heart has its own distance-map finetuning tutorial rather than
reusing the lung one's weights -- the two organs' distance maps do not share an
intensity distribution.

The shape-model files the tutorials read and write live here too, so that
Tutorial 6 writes the model where Tutorial 7 looks for it.  Every path hangs off
one of the three roots :class:`parameters_base.ParametersBase` resolves, so
pointing an environment variable at another disk moves them all together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from parameters_base import ParametersBase
from monai_physio import SegmentAnatomyBase, SegmentChestTotalSegmentator


@dataclass(frozen=True)
class ParametersHeartCTKCL(ParametersBase):
    """Settings shared by the heart tutorials.

    Attributes:
        icp_transform_type: Alignment applied to a surface before the shape
            model corresponds or fits it, one of ``"Rigid"``, ``"Similarity"``
            or ``"Affine"``.  Model building and model fitting must use the
            same value: whatever the ICP absorbs is variation the eigenmodes
            never see, so a mismatch makes the fit ask the modes to explain
            shape that has already been removed.
        mask_dilation_mm: Dilation of the binary registration masks, in
            millimeters.  Tighter than the lungs': the heart is a compact organ
            whose neighbours are not part of the model.
        distancemap_squared_max: Saturation radius of every heart distance map,
            in squared millimeters.  Fixes their intensity distribution, so the
            finetuning tutorial and every tutorial that registers heart distance
            maps must use this one value.
        surface_reduction_rate: Fraction of triangles removed from every
            extracted heart surface.  ``0.0`` keeps them at full resolution.
        mesh_element_size_mm: Edge length of the isotropic voxels a labelmap
            is resampled to before it is meshed into tetrahedra, which is the
            resulting element size.  Below the thinnest wall of the heart, so
            that the myocardium survives the coarsening.
        model_points: Points kept per surface when building the shape model.
            ``0`` keeps every point, which is what a full run does.
        model_points_test: Same, under ``TestTools.running_as_test``, where the
            KCL meshes are read at full resolution because that dataset has no
            downsampled test subset.
        number_of_pca_components: PCA components retained when building the
            heart statistical model, and used when fitting it to a patient.
        number_of_pca_components_test: Same, under ``TestTools.running_as_test``.
        number_of_iterations_greedy: Greedy coarse-to-fine iteration schedule.
        number_of_iterations_greedy_test: Same, under
            ``TestTools.running_as_test``.
        segmenter_class: Segmenter every heart tutorial instantiates, so the
            surfaces they compare share a definition of "heart".
        anatomy_group: Anatomy group name that segmenter registers for the heart.
        interior_object_ids_totalsegmentator: Chamber labels in a
            TotalSegmentator labelmap.  The chambers are interior to the
            myocardium, so a distance map must not measure to them.
        input_dir: Population Tutorial 6 builds the model from, and
            ``input_dir_test`` its counterpart under
            ``TestTools.running_as_test``.
        hold_out_dir: Dataset the held-out case is read from by Tutorial 7, and
            ``hold_out_dir_test`` its counterpart under
            ``TestTools.running_as_test``.  A different dataset from
            ``input_dir``: the model is built from KCL meshes and fitted to a
            DIR-Lab patient.
        pca_json_file: Shape model Tutorial 6 writes and Tutorial 7 reads.
        pca_mean_file: Mean surface of that model, written and read the same way.
        hold_out_case: DIR-Lab case fitted by Tutorial 7 and therefore kept out
            of the population Tutorial 6 builds the model from, so that the fit
            measures generalization rather than reconstruction.  The KCL model
            meshes carry no DIR-Lab case, so today the exclusion never fires;
            Tutorial 6 applies it anyway, so adding one cannot slip it in.
            The Duke heart tutorials name their own in
            ``parameters_duke_heart_labelmaps.py``.
    """

    icp_transform_type: str = "Affine"

    mask_dilation_mm: float = 10.0
    distancemap_squared_max: float = (1.25 * 10.0) ** 2

    surface_reduction_rate: float = 0.5
    mesh_element_size_mm: float = 1.5

    model_points: int = 0
    model_points_test: int = 20000

    number_of_pca_components: int = 10
    number_of_pca_components_test: int = 5

    number_of_iterations_greedy: list[int] = field(
        default_factory=lambda: [30, 15, 7, 3]
    )
    number_of_iterations_greedy_test: list[int] = field(default_factory=lambda: [1, 0])

    segmenter_class: type[SegmentAnatomyBase] = SegmentChestTotalSegmentator
    anatomy_group: str = "heart"
    interior_object_ids_totalsegmentator: list[int] = field(
        default_factory=lambda: [141, 142, 143, 144]
    )

    hold_out_case: str = "Case1Pack"

    def input_directory(self, test_mode: bool) -> Path:
        """Return the population Tutorial 6 builds the model from."""
        return self.data_directory(test_mode) / "KCL-Heart-Model"

    def hold_out_directory(self, test_mode: bool) -> Path:
        """Return the dataset Tutorial 7 reads the held-out case from.

        A different dataset from ``input_directory``: the model is built from
        KCL meshes and fitted to a DIR-Lab patient.
        """
        return self.data_directory(test_mode) / "DirLab-4DCT"

    def pca_model_file(self, test_mode: bool) -> Path:
        """Return the shape model Tutorial 6 writes and Tutorial 7 reads."""
        return self.output_directory(test_mode) / "tutorial_06_heart" / "pca_model.json"

    def pca_mean_surface_file(self, test_mode: bool) -> Path:
        """Return that model's mean surface, written and read the same way."""
        return (
            self.output_directory(test_mode)
            / "tutorial_06_heart"
            / "pca_mean_surface.vtp"
        )

    def pca_components(self, test_mode: bool) -> int:
        """Return the PCA component count for this run mode."""
        return (
            self.number_of_pca_components_test
            if test_mode
            else (self.number_of_pca_components)
        )

    def points_per_model(self, test_mode: bool) -> int:
        """Return the per-surface point budget for this run mode."""
        return self.model_points_test if test_mode else self.model_points

    def greedy_iterations(self, test_mode: bool) -> list[int]:
        """Return the Greedy iteration schedule for this run mode."""
        return list(
            self.number_of_iterations_greedy_test
            if test_mode
            else self.number_of_iterations_greedy
        )


#: The single instance every heart tutorial imports.
HEART_CT_KCL = ParametersHeartCTKCL()
