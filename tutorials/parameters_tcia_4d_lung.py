"""Shared parameters for the TCIA 4D-Lung tutorials.

TCIA-4DLung lays out one subdirectory per case
(``TCIA-4DLung/<case>_HM10395/<case>_HM10395_g0??.nii.gz``, ten respiratory
phases ``g000``..``g090`` per case), unlike DIR-Lab's flat
``DirLab-4DCT/<case>_T??.mha`` layout.  :meth:`ParametersTCIA4DLung.input_directory`
resolves the single reference case (``100_HM10395``) that Tutorials 1, 3 and 4
read; :meth:`ParametersTCIA4DLung.cases_directory` resolves the root every
other case's directory lives under, for tutorials that build a population
(Tutorials 6, 8, 9, 15) or hold one case out of it (``mgn_hold_out_case``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from parameters_base import ParametersBase

from monai_physio import SegmentAnatomyBase, SegmentChestTotalSegmentator


@dataclass(frozen=True)
class ParametersTCIA4DLung(ParametersBase):
    """Settings shared by the TCIA 4D-Lung tutorials.

    Attributes:
        surface_reduction_rate: Fraction of triangles removed from every
            extracted lung surface. ``0.0`` keeps them at full resolution.
        mesh_element_size_mm: Edge length, in millimeters, of the tetrahedral
            mesh elements built from the lung mask.
        number_of_iterations_icon: Finetuning steps for ICON registration.
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
        model_points: Points kept per surface when building the shape model.
            ``0`` keeps every point, which is what a full run does.
        model_points_test: Same, under ``ProcessTests.running_as_test``.
        number_of_pca_components: PCA components retained when building the
            lung statistical model, and used when fitting it to a patient.
        number_of_pca_components_test: Same, under ``ProcessTests.running_as_test``.
        number_of_iterations_greedy_test: Greedy coarse-to-fine iteration
            schedule, under ``ProcessTests.running_as_test``.
        segmenter_class: Segmenter every lung tutorial instantiates, so the
            surfaces they compare share a definition of "lung".
        anatomy_group: Anatomy group name that segmenter registers for lungs.
        hold_out_case: Image Tutorial 7 fits and Tutorial 6 excludes from the
            population it builds the model from.  Unrelated to TCIA-4DLung: it
            is a Chest-CT study, kept out of every case population by name
            alone.
        mgn_hold_out_case: TCIA-4DLung case directory kept out of the
            Tutorial 9 training population and predicted by Tutorial 10, so
            that the prediction measures generalization.  ``100_HM10395`` has
            no lettered re-scan variant, unlike every other case.
    """

    surface_reduction_rate: float = 0.25
    mesh_element_size_mm: float = 3.0

    number_of_iterations_icon: int = 20
    number_of_iterations_greedy: list[int] = field(
        default_factory=lambda: [100, 100, 10, 5]  # with CC
        # default_factory=lambda: [100, 100, 200, 50]  # with mean squares
    )
    number_of_iterations_greedy_test: list[int] = field(default_factory=lambda: [1, 0])
    greedy_metric: str = "CC"

    icp_transform_type: str = "Affine"

    mask_dilation_mm: float = 20.0
    distancemap_squared_max: float = (1.25 * 20.0) ** 2

    model_points: int = 80000
    model_points_test: int = 20000

    number_of_pca_components: int = 6
    number_of_pca_components_test: int = 5

    segmenter_class: type[SegmentAnatomyBase] = SegmentChestTotalSegmentator
    anatomy_group: str = "lung"

    hold_out_case: str = "Chest-CT.mha"
    mgn_hold_out_case: str = "100_HM10395"

    def input_directory(self, test_mode: bool) -> Path:
        """Return the TCIA-4DLung reference case directory."""
        return self.data_directory(test_mode) / "TCIA-4DLung" / "100_HM10395"

    def cases_directory(self, test_mode: bool) -> Path:
        """Return the root every TCIA-4DLung case directory lives under."""
        return self.data_directory(test_mode) / "TCIA-4DLung"

    def hold_out_directory(self, test_mode: bool) -> Path:
        """Return the dataset Tutorial 7 reads the held-out study from."""
        return self.data_directory(test_mode) / "Chest-CT"

    def pca_model_file(self, test_mode: bool) -> Path:
        """Return the shape model Tutorial 6 writes and 7 and 8 read."""
        return self.output_directory(test_mode) / "tutorial_06_lung" / "pca_model.json"

    def pca_mean_surface_file(self, test_mode: bool) -> Path:
        """Return that model's mean surface, written and read the same way."""
        return (
            self.output_directory(test_mode)
            / "tutorial_06_lung"
            / "pca_mean_surface.vtp"
        )

    def mgn_weights_directory(self, test_mode: bool) -> Path:
        """Return the lung-motion MeshGraphNet directory for this run mode."""
        return self.weights_directory(test_mode) / "physicsnemo_mgn_lung_motion"

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


#: The single instance every TCIA 4D-Lung tutorial imports.
TCIA_4D_LUNG = ParametersTCIA4DLung()
