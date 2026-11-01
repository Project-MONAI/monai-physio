"""Shared parameters for the Slicer-Heart-CT tutorials.

Slicer-Heart-CT is one cardiac-gated 4D CT series, stored as one 3D volume per
phase (``data/Slicer-Heart-CT/slice_???.mha``; ``data/test/slicer_heart_small``
under ``ProcessTests.running_as_test``).  The tutorials that use it only
process data and run inference -- segmentation, registration, meshing and USD
export -- so this carries no shape-model or training settings.  Its segmenter,
registrar and meshing values are the ones :mod:`parameters_kcl_heart_4d_meshes`
carries, so a heart segmented here is segmented the way Tutorial 7 (KCL heart)
segments its patient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from parameters_base import ParametersBase

from monai_physio import (
    RegisterImagesBase,
    RegisterImagesGreedy,
    SegmentAnatomyBase,
    SegmentChestTotalSegmentatorWithContrast,
)


@dataclass(frozen=True)
class ParametersSlicerHeart4DCT(ParametersBase):
    """Settings shared by the Slicer-Heart-CT tutorials.

    Attributes:
        surface_reduction_rate: Fraction of triangles removed from every
            extracted heart surface.  ``0.0`` keeps them at full resolution.
        surface_distance_sigma_mm: Blur applied to the label distance maps
            before surfaces are contoured.  Twice the 0.6 mm voxel spacing
            removes the fine ripple the segmentation leaves while keeping the
            small pulmonary vessels; 1.8 mm starts to break them off.
        mesh_element_size_mm: Edge length of the isotropic voxels a labelmap
            is resampled to before it is meshed into tetrahedra, which is the
            resulting element size.
        segmenter_class: Segmenter every Slicer-Heart-CT tutorial instantiates.
        registration_class: Registration method every Slicer-Heart-CT tutorial
            instantiates.
        anatomy_group: Anatomy group name that segmenter registers for the heart.
    """

    surface_reduction_rate: float = 0.2
    surface_distance_sigma_mm: float = 1.2
    mesh_element_size_mm: float = 1.0

    segmenter_class: type[SegmentAnatomyBase] = SegmentChestTotalSegmentatorWithContrast
    registration_class: type[RegisterImagesBase] = RegisterImagesGreedy
    anatomy_group: str = "heart"

    def input_directory(self, test_mode: bool) -> Path:
        """Return the directory holding the phase volumes for this run mode."""
        if test_mode:
            return self.data_directory(test_mode) / "slicer_heart_small"
        return self.data_directory(test_mode) / "Slicer-Heart-CT"

    def segmenter(
        self, test_mode: bool, log_level: int | str = logging.INFO
    ) -> SegmentAnatomyBase:
        """Return the shared heart segmenter, in fast mode only for tests.

        Fast mode segments on a coarser grid, which leaves voxel steps in
        the extracted surfaces, so full runs use the full-resolution model.

        Args:
            test_mode: Whether to enable the segmenter's fast mode.
        """
        segmenter = self.segmenter_class(log_level=log_level)
        segmenter.set_fast_mode(test_mode)
        return segmenter

    def registrar(
        self, test_mode: bool, log_level: int | str = logging.INFO
    ) -> RegisterImagesBase:
        """Return the shared heart registrar, tuned for this run mode."""
        registrar = cast(
            RegisterImagesGreedy, self.registration_class(log_level=log_level)
        )
        registrar.set_number_of_iterations([1, 0] if test_mode else [100, 100, 10, 5])
        registrar.set_metric("CC")
        return registrar


#: The single instance every Slicer-Heart-CT tutorial imports.
SLICER_HEART_4D_CT = ParametersSlicerHeart4DCT()
