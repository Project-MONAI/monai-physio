"""Workflow computing an unbiased mean surface from a population of surfaces.

Surfaces from different subjects have different point counts, so their points
cannot be averaged directly. Correspondence is established by warping a single
template through each sample and averaging the *template's* points:

1. Align each sample to the template with :class:`RegisterModelsICP` (rigid by
   default, so size differences remain part of the averaged shape).
2. Register each aligned sample to the template with
   :class:`RegisterModelsDistanceMaps` (Greedy affine + ICON on distance maps).
3. Warp the template by each sample's forward transform, giving one surface per
   sample that carries the sample's shape on the template's topology.
4. Average those point sets.

A single pass leaves the mean in the shape space of whichever surface was chosen
as the template. Repeating the four steps with the previous mean as the new
template washes that bias out — the standard groupwise / atlas iteration, as in
``antsMultivariateTemplateConstruction``. Iteration stops once the mean moves
less than the convergence tolerance.

The same correspondence mechanism drives
:class:`physiotwin4d.WorkflowCreateStatisticalModel`, which additionally solves
for the PCA modes; use this workflow when only the mean shape is needed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv

from .contour_tools import ContourTools
from .physiotwin4d_base import PhysioTwin4DBase
from .register_models_distance_maps import RegisterModelsDistanceMaps
from .register_models_icp import RegisterModelsICP


class WorkflowCreateMeanSurface(PhysioTwin4DBase):
    """Compute the mean shape of N surfaces with differing point counts.

    Example:
        >>> workflow = WorkflowCreateMeanSurface(surfaces=[s0, s1, s2])
        >>> workflow.set_number_of_iterations(3)
        >>> result = workflow.process()
        >>> mean_surface = result["mean_surface"]
    """

    def __init__(
        self,
        surfaces: list[pv.DataSet],
        template_surface: Optional[pv.DataSet] = None,
        spatial_resolution: float = 1.0,
        buffer_factor: float = 0.25,
        log_level: int | str = logging.INFO,
    ) -> None:
        """Initialize the mean-surface workflow.

        Args:
            surfaces: Population of surfaces to average. Point counts and
                topologies may differ; only the template's topology survives.
            template_surface: Surface providing the topology of the mean and the
                starting point of the atlas iteration. Defaults to the middle
                entry of ``surfaces``. With enough iterations the result no
                longer depends on this choice.
            spatial_resolution: Voxel size of the distance-map grid used for
                registration. Default: 1.0
            buffer_factor: Padding around the template bounding box for that
                grid. Default: 0.25
            log_level: Logging level. Default: ``logging.INFO``.

        Raises:
            ValueError: If fewer than two surfaces are supplied.
        """
        super().__init__(class_name="WorkflowCreateMeanSurface", log_level=log_level)

        if len(surfaces) < 2:
            raise ValueError(f"At least 2 surfaces are required, got {len(surfaces)}")

        self.contour_tools = ContourTools(log_level=log_level)
        self.surfaces = [self.contour_tools.extract_surface(s) for s in surfaces]
        self.template_surface = (
            self.contour_tools.extract_surface(template_surface)
            if template_surface is not None
            else self.surfaces[len(self.surfaces) // 2]
        )
        self.spatial_resolution = spatial_resolution
        self.buffer_factor = buffer_factor

        # Atlas iteration controls.
        self.number_of_iterations: int = 3
        self.convergence_tolerance: float = 0.1  # mm of RMS point motion
        # Rigid alignment keeps size differences in the averaged shape; use
        # "Affine" to average only the residual, size-and-pose-normalized shape.
        self.alignment_transform_type: str = "Rigid"
        self.registration_transform_type: str = "Deformable"

        # Correspondence tuning, mirroring WorkflowFitStatisticalModelToPatient
        # so that a mean built here and a fit against it see distance maps on
        # the same scale.
        self.mask_dilation_mm: float = 20.0
        self.distance_squared_max: Optional[float] = None
        self.icon_weights_path: Optional[str] = None

        # Results (populated by process()).
        self.mean_surface: Optional[pv.PolyData] = None
        self.corresponded_surfaces: list[pv.PolyData] = []
        self.forward_transforms: list[Any] = []
        self.iteration_rms_mm: list[float] = []

    # ─────────────────────────── Tuning setters ────────────────────────────
    def set_number_of_iterations(self, number_of_iterations: int) -> None:
        """Set the atlas iteration count; ``1`` is a single template-biased pass."""
        if number_of_iterations < 1:
            raise ValueError(
                f"number_of_iterations must be >= 1, got {number_of_iterations}"
            )
        self.number_of_iterations = number_of_iterations

    def set_convergence_tolerance(self, convergence_tolerance: float) -> None:
        """Set the RMS point motion (mm) below which iteration stops."""
        if convergence_tolerance < 0.0:
            raise ValueError(
                f"convergence_tolerance must be >= 0, got {convergence_tolerance}"
            )
        self.convergence_tolerance = convergence_tolerance

    def set_alignment_transform_type(self, transform_type: str) -> None:
        """Set the ICP alignment type: ``'Rigid'`` (default) or ``'Affine'``."""
        if transform_type not in ("Rigid", "Affine"):
            raise ValueError(
                f"Invalid alignment transform '{transform_type}'. "
                "Must be 'Rigid' or 'Affine'."
            )
        self.alignment_transform_type = transform_type

    def set_mask_dilation_mm(self, mask_dilation_mm: float) -> None:
        """Set the dilation (mm) of the binary masks the Greedy stage registers in."""
        self.mask_dilation_mm = mask_dilation_mm

    def set_distance_squared_max(self, distance_squared_max: float) -> None:
        """Set the squared millimetres the distance maps are normalized against.

        The maps saturate at its square root, and a sample further than that
        from the template has no gradient pulling it in, so the correspondence
        stops short and the mean creeps toward the template.

        Args:
            distance_squared_max: Saturation radius in squared millimeters.

        Raises:
            ValueError: If it is not positive. The maps are normalized against
                its square root, so zero or less saturates every voxel alike
                and leaves the registration nothing to descend.
        """
        if distance_squared_max <= 0.0:
            raise ValueError(
                f"distance_squared_max must be positive, got {distance_squared_max}."
            )
        self.distance_squared_max = distance_squared_max

    def set_icon_weights_path(self, weights_path: str) -> None:
        """Use a finetuned uniGradICON checkpoint for the deformable stage.

        Stock weights are out of distribution for distance maps; see
        ``RegisterModelsDistanceMaps.set_icon_weights_path``.

        Args:
            weights_path: Path to an existing uniGradICON checkpoint.
        """
        self.icon_weights_path = weights_path

    def _distance_squared_max(self) -> float:
        """Return the configured saturation radius, or one sized to the mask."""
        if self.distance_squared_max is not None:
            return self.distance_squared_max
        return (1.25 * self.mask_dilation_mm) ** 2

    def set_registration_transform_type(self, transform_type: str) -> None:
        """Set the distance-map registration type used for correspondence.

        ``'Deformable'`` (default) runs Greedy affine + ICON; ``'Affine'`` and
        ``'Rigid'`` stop after the Greedy stage.
        """
        if transform_type not in ("Rigid", "Affine", "Deformable"):
            raise ValueError(
                f"Invalid registration transform '{transform_type}'. "
                "Must be 'Rigid', 'Affine' or 'Deformable'."
            )
        self.registration_transform_type = transform_type

    # ─────────────────────────── Main workflow ─────────────────────────────
    def process(self) -> dict[str, Any]:
        """Run the atlas iteration and return the mean surface.

        Returns:
            Dict with ``mean_surface`` (template topology, mean shape),
            ``corresponded_surfaces`` and ``forward_transforms`` from the final
            iteration, the per-iteration ``iteration_rms_mm``, and
            ``number_of_iterations_run``.
        """
        self.log_section("STARTING CREATE MEAN SURFACE WORKFLOW", width=70)
        self.log_info(
            "Averaging %d surfaces (%s alignment, %s correspondence, %d point(s) "
            "in template)",
            len(self.surfaces),
            self.alignment_transform_type,
            self.registration_transform_type,
            self.template_surface.n_points,
        )

        template = self.template_surface
        self.iteration_rms_mm = []
        iterations_run = 0

        for iteration in range(self.number_of_iterations):
            self.log_info(
                "Atlas iteration %d/%d", iteration + 1, self.number_of_iterations
            )
            corresponded, forward_transforms, aligned = self._correspond(template)

            mean_points = np.mean(
                [np.asarray(surface.points) for surface in corresponded], axis=0
            )
            mean_points = self._normalize_scale(mean_points, template, aligned)
            rms = float(
                np.sqrt(
                    np.mean(
                        np.sum((mean_points - np.asarray(template.points)) ** 2, axis=1)
                    )
                )
            )
            self.iteration_rms_mm.append(rms)
            iterations_run = iteration + 1

            new_template = template.copy(deep=True)
            new_template.points = mean_points
            template = new_template

            self.corresponded_surfaces = corresponded
            self.forward_transforms = forward_transforms
            self.log_info("  mean moved %.4f mm (RMS) from the previous template", rms)

            if rms < self.convergence_tolerance:
                self.log_info(
                    "  converged (< %.4f mm); stopping", self.convergence_tolerance
                )
                break

        self.mean_surface = template
        self.log_section("CREATE MEAN SURFACE WORKFLOW COMPLETE", width=70)
        return {
            "mean_surface": self.mean_surface,
            "corresponded_surfaces": self.corresponded_surfaces,
            "forward_transforms": self.forward_transforms,
            "iteration_rms_mm": self.iteration_rms_mm,
            "number_of_iterations_run": iterations_run,
        }

    # ─────────────────────────── Internal steps ────────────────────────────
    @staticmethod
    def _shape_scale(surface: pv.PolyData) -> float:
        """Size proxy that does not depend on how finely the surface is sampled."""
        return float(np.sqrt(max(float(surface.area), 1.0e-12)))

    def _normalize_scale(
        self,
        mean_points: np.ndarray,
        template: pv.PolyData,
        aligned: list[pv.PolyData],
    ) -> np.ndarray:
        """Rescale the mean about its centroid to the population's mean size.

        Registration never recovers a sample's size exactly, and the residual
        error is one-sided: each iteration averages slightly under-deformed
        templates, so an unconstrained atlas creeps toward — and eventually
        below — the smallest input. Setting the size explicitly from the aligned
        inputs, which are measured rather than registered, removes that drift.
        With ``Affine`` alignment the inputs already carry the template's size,
        so the correction is a no-op and the mean stays size-normalized.
        """
        scaled = template.copy(deep=True)
        scaled.points = mean_points
        current_scale = self._shape_scale(scaled)
        target_scale = float(
            np.mean([self._shape_scale(surface) for surface in aligned])
        )
        if current_scale <= 0.0:
            return mean_points

        factor = target_scale / current_scale
        self.log_info("  scale correction: %.4f", factor)
        centroid = mean_points.mean(axis=0)
        return cast(np.ndarray, centroid + (mean_points - centroid) * factor)

    def _correspond(
        self, template: pv.PolyData
    ) -> tuple[list[pv.PolyData], list[Any], list[pv.PolyData]]:
        """Warp ``template`` onto every input surface.

        Returns one surface per input, each carrying that input's shape on the
        template's topology, the forward transform that produced it, and the
        ICP-aligned inputs those transforms were computed against.
        """
        aligned: list[pv.PolyData] = []
        for index, surface in enumerate(self.surfaces):
            self.log_info("  aligning surface %d/%d", index + 1, len(self.surfaces))
            icp_result = RegisterModelsICP(
                fixed_model=template, log_level=self.log_level
            ).register(
                moving_model=surface,
                transform_type=self.alignment_transform_type,
                max_iterations=2000,
            )
            aligned.append(icp_result["registered_model"])

        # The distance-map grid must contain the template *and* every aligned
        # sample: a sample clipped by the grid registers as if it were smaller,
        # which would bias the mean toward the template's size.
        bounding_cloud = pv.PolyData(
            np.vstack(
                [np.asarray(template.points)]
                + [np.asarray(surface.points) for surface in aligned]
            )
        )
        reference_image = self.contour_tools.create_reference_image(
            mesh=bounding_cloud,
            spatial_resolution=self.spatial_resolution,
            buffer_factor=self.buffer_factor,
            ptype=itk.UC,
        )

        corresponded: list[pv.PolyData] = []
        forward_transforms: list[Any] = []
        for index, aligned_surface in enumerate(aligned):
            self.log_info("  registering surface %d/%d", index + 1, len(aligned))

            registrar = RegisterModelsDistanceMaps(
                moving_model=aligned_surface,
                fixed_model=template,
                reference_image=reference_image,
                distance_squared_max=self._distance_squared_max(),
                mask_dilation_mm=self.mask_dilation_mm,
                log_level=self.log_level,
            )
            if self.icon_weights_path is not None:
                registrar.set_icon_weights_path(self.icon_weights_path)
            result = registrar.register(transform_type=self.registration_transform_type)

            # The forward (image-convention) transform maps template points into
            # the sample's shape, so warping the template by it yields template
            # topology with sample shape.
            corresponded.append(
                self.contour_tools.transform_contours(
                    template,
                    tfm=result["forward_transform"],
                    with_deformation_magnitude=False,
                )
            )
            forward_transforms.append(result["forward_transform"])

        return corresponded, forward_transforms, aligned
