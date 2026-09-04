"""Accuracy of an inferred moving anatomy, per anatomical structure.

:class:`WorkflowEvaluateMovement` scores a
:class:`monai_physio.WorkflowInferMovement` against geometry extracted from a
gated image sequence. For every gated time point it carries the reference
frame's labelmap into that time point with the network's own deformation and
compares the result, structure by structure, to the labelmap of the frame that
was actually acquired: volume difference, Dice, and surface RMSE per lung lobe
or per heart chamber. Given the true surface of each frame it also reports the
distance between where the network puts each shape-model point and where the
model was fitted, per structure.

Going through labelmaps rather than through the model's own surface is what lets
one workflow serve both anatomies. The lung shape model carries its five lobes
as per-cell labels, but the heart model is a single structure -- the whole heart
minus its chamber cavities -- so its chambers exist only in the acquired
labelmaps. Warping those labelmaps scores every structure the acquisition
contains, whether or not the shape model represents it separately. The
point-by-point metric follows the model where it can: the lung's triangles carry
their lobe, so its points are scored under the lobe the model names, while the
heart's unlabelled surface is partitioned by nearest structure instead and its
chambers are scored on the pieces of wall that bound them.

Everything is measured on one isotropic evaluation grid built around the
reference anatomy, so a case whose gated frames carry different slice pitches is
still scored on a single, stated voxel volume.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .contour_tools import ContourTools
from .evaluate_movement_base import EvaluateMovementBase, MovementGroundTruth
from .monai_physio_base import MONAIPhysioBase
from .report_evaluate_movement import ReportEvaluateMovement
from .workflow_infer_movement import WorkflowInferMovement


class WorkflowEvaluateMovement(MONAIPhysioBase):
    """Score inferred motion per anatomical structure against acquired frames.

    Args:
        movement_workflow: The displacement decoder whose predictions are
            scored.
        cohort: The cohort being scored ---
            :class:`monai_physio.EvaluateMovementLung`,
            :class:`monai_physio.EvaluateMovementDukeHeart` or another
            :class:`monai_physio.EvaluateMovementBase`. It supplies the
            structures to score and this cohort's defaults, and its
            :meth:`~monai_physio.EvaluateMovementBase.assemble_ground_truth`
            produces what :meth:`process` scores against. The workflow never
            learns *which* cohort it holds, which is what keeps it free of
            anatomy branches.
        label_names: Structures to score, ``{label_id: name}``, for a caller
            with no cohort. Ids the reference frame does not contain are dropped
            with a warning; ids a single acquired frame does not contain are
            skipped for that frame alone, since a structure can leave the field
            of view.
        log_level: Logging level. Default: ``logging.INFO``.
    """

    # Columns of the per-point displacement CSV, in the order they are written.
    _DISPLACEMENT_COLUMNS = (
        "subject_id",
        "stage",
        "point_id",
        "fitted_reference_x_mm",
        "fitted_reference_y_mm",
        "fitted_reference_z_mm",
        "predicted_dx_mm",
        "predicted_dy_mm",
        "predicted_dz_mm",
        "true_dx_mm",
        "true_dy_mm",
        "true_dz_mm",
        "error_mm",
    )

    # Per-cell structure ids a shape model may carry, written by
    # ``ContourTools.save_combined_surfaces``.
    _MESH_LABEL_ARRAY = "SegmentationLabelIds"

    def __init__(
        self,
        movement_workflow: WorkflowInferMovement,
        cohort: Optional[EvaluateMovementBase] = None,
        label_names: Optional[dict[int, str]] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        if cohort is None and label_names is None:
            raise ValueError(
                "WorkflowEvaluateMovement needs a cohort, or the label_names to "
                "score without one."
            )
        self.movement_workflow = movement_workflow
        self.cohort = cohort
        self.label_names = (
            dict(label_names)
            if label_names is not None
            else cast(EvaluateMovementBase, cohort).label_names()
        )
        self.contour_tools = ContourTools(log_level=log_level)
        self.displacement_data_file: Optional[Path] = None
        self.report_tools = ReportEvaluateMovement(log_level=log_level)

    # ─────────────────────────── Public API ────────────────────────────────
    def process(
        self,
        case_id: str,
        shape_parameters: Path,
        fitted_reference_mesh: Path,
        ground_truth: MovementGroundTruth,
        output_directory: Path,
        smoothing_sigma_mm: Optional[float] = None,
        evaluation_spacing_mm: Optional[float] = None,
        report_dice: Optional[bool] = None,
        report_displacement_data: bool = False,
        include_predicted_displacements: bool = False,
        include_true_displacements: bool = False,
        include_displacement_error: bool = False,
    ) -> dict[str, Any]:
        """Score every gated time point of one case.

        Args:
            case_id: Name of the case being scored, recorded in every output.
            shape_parameters: JSON file with the case's PCA coefficient vector.
            fitted_reference_mesh: The case's fitted reference-frame SSM surface,
                as produced by
                :class:`monai_physio.WorkflowFitStatisticalModelToPatient`. The
                predicted displacements are added to its points, and its extent
                defines the evaluation grid.
            ground_truth: What this case is scored against --- the acquired
                labelmap per stage, the reference frame's labelmap, and the
                per-stage fits the point-by-point error is measured against ---
                as assembled by
                :meth:`monai_physio.EvaluateMovementBase.assemble_ground_truth`.
            output_directory: Directory the report, the CSV and the per-stage
                geometry are written to.
            smoothing_sigma_mm: Gaussian sigma, in millimeters, that turns the
                network's surface-shell deformation into a continuous field.
                ``None`` takes the cohort's own value.
            evaluation_spacing_mm: Isotropic pitch every metric is measured on.
                It sets both the voxel volume the Dice and volume figures are
                quantized to and the resolution of the deformation fields, whose
                memory grows with its cube. ``None`` takes the cohort's own
                value.
            report_dice: Report the Dice overlap. Turn it off for a structure
                whose motion is small against its own size: Dice is an overlap
                fraction, so a lung lobe scores over 0.96 undeformed and the
                column says more about the organ's bulk than about the motion.
                The volume and surface figures still resolve it. ``None`` takes
                the cohort's own value.
            report_displacement_data: Write ``displacement_per_point.csv``, one
                row per mesh point per stage carrying that point's predicted and
                true displacement and the error between them.
            include_predicted_displacements: Carry
                ``predicted_displacement_mm`` as point data on every stage's
                predicted surface.
            include_true_displacements: Carry ``true_displacement_mm`` as point
                data on every stage's predicted surface.
            include_displacement_error: Carry ``displacement_error_mm``, the
                distance between the predicted and the true position, as point
                data on every stage's predicted surface, and report that error's
                RMS, 95th percentile and maximum --- per structure on every
                metric row and in the report, and pooled over every point and
                stage in the report and the returned dict. It is the one metric
                here measured point by point, so a displacement predicted in the
                wrong direction cannot cancel against another. A shape-model
                point is attributed to the structure its own per-cell
                ``SegmentationLabelIds`` names, or, on a model carrying no such
                ids, to the structure whose reference-frame surface is nearest
                --- the same nearest-label partition the labelmap metrics are
                already scored under.

        Returns:
            Dict with ``rows`` (every metric row), ``csv_file``,
            ``report_file``, ``volume_plot_file``, ``predicted_surfaces``,
            ``warped_labelmaps``, ``displacement_data_file``,
            ``displacement_statistics`` (one entry per stage, empty unless
            ``include_displacement_error``) and that error pooled over every
            point and stage as ``displacement_rms_mm``,
            ``displacement_95th_mm`` and ``displacement_max_mm``.

        Raises:
            ValueError: If ``ground_truth.labelmaps`` is empty, any stage it
                carries a labelmap for has no entry in ``ground_truth.meshes``
                --- checked whether or not an option needing the true
                displacement was requested --- none of the requested labels are
                in the reference frame, or no label survives scoring because
                the acquired frames contain none of them.
        """
        if not ground_truth.labelmaps:
            raise ValueError("No ground-truth labelmaps to evaluate against.")

        reference_labelmap = ground_truth.reference_labelmap
        ground_truth_labelmaps = ground_truth.labelmaps
        ground_truth_meshes = ground_truth.meshes
        stages = sorted(ground_truth_labelmaps)
        missing = [stage for stage in stages if stage not in ground_truth_meshes]
        if missing:
            raise ValueError(
                "The ground truth needs a fitted surface for every stage it "
                "carries a labelmap for; missing "
                f"{', '.join(f'{stage:.3f}' for stage in missing)}."
            )

        # A setting the caller left unset is the cohort's to decide.
        smoothing_sigma_mm = self._setting(smoothing_sigma_mm, "smoothing_sigma_mm")
        evaluation_spacing_mm = self._setting(
            evaluation_spacing_mm, "evaluation_spacing_mm"
        )
        report_dice = self._setting(report_dice, "report_dice")

        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.log_section("EVALUATE MOVEMENT [%s]: %d stages", case_id, len(stages))

        fitted_reference_surface = cast(pv.DataSet, pv.read(str(fitted_reference_mesh)))
        grid = self.contour_tools.create_reference_image(
            mesh=fitted_reference_surface,
            spatial_resolution=evaluation_spacing_mm,
            buffer_factor=0.25,
            ptype=itk.template(reference_labelmap)[1][0],
        )
        self.log_info(
            "Evaluation grid: %s voxels at %.2f mm",
            list(itk.size(grid)),
            evaluation_spacing_mm,
        )
        reference_on_grid = self._resample_labelmap(reference_labelmap, grid)
        scored_labels = self._labels_present(reference_on_grid)
        provenance = self._provenance(case_id, shape_parameters)

        # One deformation per stage, from the network's own predictions. Each
        # stage's warped image is the reference labelmap carried into that
        # stage, which is exactly what the metrics below compare.
        series = self.movement_workflow.process_time_series(
            shape_parameters=shape_parameters,
            stages=stages,
            output_directory=out_dir,
            fitted_reference_mesh=fitted_reference_mesh,
            reference_image=reference_on_grid,
            warp_interpolation="nearest",
            warp_background_value=0.0,
            smoothing_sigma_mm=smoothing_sigma_mm,
        )

        # The per-point displacement error is the mesh-space counterpart of the
        # labelmap metrics, and exists only when the true surfaces were given.
        displacement_errors, displacement_statistics = self._score_displacements(
            case_id,
            stages,
            series,
            fitted_reference_surface,
            ground_truth_meshes,
            out_dir,
            report_displacement_data=report_displacement_data,
            include_predicted_displacements=include_predicted_displacements,
            include_true_displacements=include_true_displacements,
            include_displacement_error=include_displacement_error,
        )
        pooled = self.pool_displacement_error(
            displacement_errors if include_displacement_error else []
        )
        # Which mesh points belong to which structure, decided once against the
        # reference frame so a structure owns the same points at every stage.
        # Only the scored structures take part, so every shape-model point goes
        # to one of them rather than being lost to a structure -- a coronary,
        # say -- that no row reports.
        label_points = (
            self._label_point_indices(
                fitted_reference_surface, scored_labels, reference_on_grid
            )
            if include_displacement_error
            else {}
        )

        rows: list[dict[str, Any]] = []
        for index, stage in enumerate(stages):
            truth = self._resample_labelmap(ground_truth_labelmaps[stage], grid)
            truth_surfaces = self._label_surfaces(truth)
            predicted = itk.imread(str(series["warped_images"][index]))
            rows.extend(
                self._score(
                    case_id,
                    stage,
                    truth,
                    truth_surfaces,
                    predicted,
                    self._label_surfaces(predicted),
                    scored_labels,
                    provenance,
                    report_dice,
                    displacement_errors[index] if include_displacement_error else None,
                    label_points,
                )
            )

        if not rows:
            raise ValueError(
                "Nothing was scored: every label present in the reference frame is "
                "absent from all of the acquired frames."
            )

        written = self.report_tools.report(
            rows,
            provenance,
            stages,
            smoothing_sigma_mm,
            evaluation_spacing_mm,
            displacement_statistics,
            pooled,
            out_dir,
        )
        if displacement_statistics:
            self.log_info(
                "Displacement error over every point and stage: rms=%.3f mm  "
                "95th=%.3f mm  max=%.3f mm",
                pooled["rms_mm"],
                pooled["p95_mm"],
                pooled["max_mm"],
            )
        return {
            "rows": rows,
            "csv_file": written["csv_file"],
            "report_file": written["report_file"],
            "volume_plot_file": written["volume_plot_file"],
            "displacement_statistics": displacement_statistics,
            "displacement_rms_mm": pooled["rms_mm"],
            "displacement_95th_mm": pooled["p95_mm"],
            "displacement_max_mm": pooled["max_mm"],
            "predicted_surfaces": series["predicted_surfaces"],
            "warped_labelmaps": series["warped_images"],
            "displacement_data_file": self.displacement_data_file,
        }

    # ──────────────────────────── Metrics ──────────────────────────────────
    @staticmethod
    def dice(truth: np.ndarray, predicted: np.ndarray, label: int) -> float:
        """Dice overlap of one label. ``nan`` when neither volume contains it."""
        truth_mask = truth == label
        predicted_mask = predicted == label
        denominator = np.count_nonzero(truth_mask) + np.count_nonzero(predicted_mask)
        if denominator == 0:
            return float("nan")
        return float(2.0 * np.count_nonzero(truth_mask & predicted_mask) / denominator)

    @staticmethod
    def volume_mm3(labels: np.ndarray, label: int, voxel_volume_mm3: float) -> float:
        """Volume of one label, in cubic millimeters."""
        return float(np.count_nonzero(labels == label) * voxel_volume_mm3)

    @staticmethod
    def surface_rmse_mm(truth: pv.PolyData, predicted: pv.PolyData) -> float:
        """Symmetric point-to-surface RMSE, in millimeters.

        Both directions are pooled before the root-mean-square. A one-sided RMSE
        misses a prediction that covers the truth everywhere but also bulges
        somewhere the truth does not reach.
        """
        forward = predicted.copy().compute_implicit_distance(truth)
        reverse = truth.copy().compute_implicit_distance(predicted)
        distances = np.concatenate(
            [
                np.asarray(forward["implicit_distance"], dtype=np.float64),
                np.asarray(reverse["implicit_distance"], dtype=np.float64),
            ]
        )
        return float(np.sqrt(np.mean(distances**2)))

    # ──────────────────────────── Internals ────────────────────────────────
    def _setting(self, given: Any, name: str) -> Any:
        """The caller's value, or the cohort's when the caller gave none.

        With no cohort the generic default on
        :class:`monai_physio.EvaluateMovementBase` stands in, so scoring
        something no cohort describes still works.
        """
        if given is not None:
            return given
        return getattr(self.cohort or EvaluateMovementBase, name)

    def _score_displacements(
        self,
        case_id: str,
        stages: list[float],
        series: dict[str, Any],
        fitted_reference_surface: pv.DataSet,
        ground_truth_meshes: Optional[dict[float, Path]],
        out_dir: Path,
        report_displacement_data: bool,
        include_predicted_displacements: bool,
        include_true_displacements: bool,
        include_displacement_error: bool,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        """Score every stage's prediction against its true surface, point by point.

        The inference workflow hands back the stage meshes it wrote, so this
        annotates and re-saves those very objects rather than reading them back.
        Each stage's Euclidean error is computed **once** and then reduced four
        ways: the mesh array, the per-point CSV, the per-stage row and the
        pooled figures.

        Returns:
            The per-stage error arrays and their per-stage statistics rows, both
            empty when no true surfaces were given or when
            ``include_displacement_error`` is off --- the caller reports the
            pooled figures only when it was asked for the error, so scoring one
            it will not report would be measured and thrown away.
        """
        # Cleared before any early return: the attribute outlives the call, so
        # leaving it set would report the previous case's CSV as this one's.
        self.displacement_data_file = None
        if ground_truth_meshes is None:
            return [], []

        # Reading a stage's true surface only pays for itself if something asked
        # for it: the true displacement, the per-point error, or the CSV that
        # carries both.
        needs_true_points = (
            include_true_displacements
            or include_displacement_error
            or report_displacement_data
        )
        # An option that adds no point data leaves each stage mesh exactly as the
        # inference workflow already wrote it, so re-saving would rewrite an
        # identical file.
        annotated = (
            include_predicted_displacements
            or include_true_displacements
            or include_displacement_error
        )
        if not needs_true_points and not include_predicted_displacements:
            return [], []

        fitted_reference_points = np.asarray(
            fitted_reference_surface.points, dtype=np.float32
        )
        # Header first, then one append per stage, so a long series never holds
        # every point of every stage in memory at once.
        displacement_file: Optional[Path] = None
        if report_displacement_data:
            displacement_file = out_dir / "displacement_per_point.csv"
            with displacement_file.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(self._DISPLACEMENT_COLUMNS)

        errors: list[np.ndarray] = []
        statistics: list[dict[str, Any]] = []
        for index, stage in enumerate(stages):
            stage_mesh = series["stage_meshes"][index]
            predicted_points = np.asarray(stage_mesh.points, dtype=np.float32)

            if include_predicted_displacements:
                stage_mesh.point_data["predicted_displacement_mm"] = (
                    predicted_points - fitted_reference_points
                ).astype(np.float32)

            if needs_true_points:
                true_points = np.asarray(
                    pv.read(str(ground_truth_meshes[stage])).points, dtype=np.float32
                )
                differences = predicted_points - true_points
                stage_errors = np.linalg.norm(differences, axis=1).astype(np.float32)

                if include_true_displacements:
                    stage_mesh.point_data["true_displacement_mm"] = (
                        true_points - fitted_reference_points
                    ).astype(np.float32)
                if include_displacement_error:
                    stage_mesh.point_data["displacement_error_mm"] = stage_errors

                if displacement_file is not None:
                    with displacement_file.open(
                        "a", newline="", encoding="utf-8"
                    ) as fh:
                        csv.writer(fh).writerows(
                            self._displacement_rows(
                                case_id,
                                stage,
                                fitted_reference_points,
                                predicted_points,
                                true_points,
                                stage_errors,
                            )
                        )

                if include_displacement_error:
                    errors.append(stage_errors)
                    statistics.append(
                        self.displacement_error_row(
                            case_id, stage, differences, stage_errors
                        )
                    )
                    self.log_info(
                        "stage %.3f: mean=%.3f mm  95th=%.3f mm  max=%.3f mm",
                        stage,
                        statistics[-1]["mean_error_mm"],
                        statistics[-1]["p95_error_mm"],
                        statistics[-1]["max_error_mm"],
                    )

            if annotated:
                stage_mesh.save(str(series["predicted_surfaces"][index]))

        self.displacement_data_file = displacement_file
        return errors, statistics

    @staticmethod
    def pool_displacement_error(
        stage_errors: list[np.ndarray],
    ) -> dict[str, float]:
        """RMS, 95th percentile and maximum over every point of every stage.

        The errors themselves are pooled rather than their per-stage summaries:
        a percentile of percentiles is not the percentile of the whole. Every
        value is ``nan`` when no stage was scored point by point.
        """
        if not stage_errors:
            return {
                "rms_mm": float("nan"),
                "p95_mm": float("nan"),
                "max_mm": float("nan"),
            }
        errors = np.concatenate(stage_errors)
        return {
            "rms_mm": float(np.sqrt(np.mean(errors.astype(np.float64) ** 2))),
            "p95_mm": float(np.percentile(errors, 95.0)),
            "max_mm": float(errors.max()),
        }

    @staticmethod
    def displacement_error_row(
        case_id: str, stage: float, differences: np.ndarray, errors: np.ndarray
    ) -> dict:
        """One stage's displacement error, summarized over the mesh points.

        Args:
            case_id: Case being scored.
            stage: Normalized stage.
            differences: ``(n_points, 3)`` predicted minus true position.
            errors: ``(n_points,)`` Euclidean norm of those differences,
                computed once by the caller rather than again here.
        """
        return {
            "subject_id": case_id,
            "stage": stage,
            "n_points": int(len(errors)),
            "mean_error_mm": float(errors.mean()),
            "median_error_mm": float(np.median(errors)),
            "max_error_mm": float(errors.max()),
            "p95_error_mm": float(np.percentile(errors, 95.0)),
            "rms_error_mm": float(np.sqrt(np.mean(errors**2))),
            "std_error_mm": float(errors.std()),
            "mean_abs_error_x_mm": float(np.abs(differences[:, 0]).mean()),
            "mean_abs_error_y_mm": float(np.abs(differences[:, 1]).mean()),
            "mean_abs_error_z_mm": float(np.abs(differences[:, 2]).mean()),
        }

    @staticmethod
    def _displacement_rows(
        case_id: str,
        stage: float,
        fitted_reference_points: np.ndarray,
        predicted_points: np.ndarray,
        true_points: np.ndarray,
        errors: np.ndarray,
    ) -> list[list[Any]]:
        """One row per mesh point, in ``_DISPLACEMENT_COLUMNS`` order."""
        block = np.column_stack(
            [
                fitted_reference_points,
                predicted_points - fitted_reference_points,
                true_points - fitted_reference_points,
                errors,
            ]
        )
        return [
            [case_id, stage, point_id, *values]
            for point_id, values in enumerate(block.tolist())
        ]

    @staticmethod
    def _resample_labelmap(labelmap: itk.Image, grid: itk.Image) -> itk.Image:
        """Resample a labelmap onto ``grid``, preserving its discrete values."""
        return itk.resample_image_filter(
            labelmap,
            use_reference_image=True,
            reference_image=grid,
            interpolator=itk.NearestNeighborInterpolateImageFunction.New(labelmap),
            default_pixel_value=0,
        )

    def _labels_present(self, reference_labelmap: itk.Image) -> dict[int, str]:
        """Drop the requested labels the reference frame does not contain."""
        present = set(np.unique(itk.GetArrayViewFromImage(reference_labelmap)).tolist())
        scored = {
            label: name for label, name in self.label_names.items() if label in present
        }
        missing = sorted(set(self.label_names) - set(scored))
        if missing:
            self.log_warning(
                "Reference frame has no voxels for label(s) %s; not scored.", missing
            )
        if not scored:
            raise ValueError(
                "None of the requested labels are present in the reference frame."
            )
        self.log_info(
            "Scoring %d structure(s): %s",
            len(scored),
            ", ".join(scored[label] for label in sorted(scored)),
        )
        return scored

    def _label_surfaces(self, labelmap: itk.Image) -> dict[int, pv.PolyData]:
        """Contour every label of one labelmap on the evaluation grid's pitch."""
        return self.contour_tools.extract_label_surfaces(labelmap)

    def _label_point_indices(
        self,
        mesh: pv.DataSet,
        scored_labels: dict[int, str],
        reference_labelmap: itk.Image,
    ) -> dict[int, np.ndarray]:
        """Shape-model point indices per structure.

        A mesh that names the structure of every cell is taken at its word: the
        lung model tags each triangle with its lobe, and no inference beats
        that. A mesh that does not --- the heart is one surface, its chambers
        being cavities the model excludes --- is partitioned by nearest label
        surface instead, the same nearest-label rule
        :meth:`ContourTools.extract_label_surfaces` already scores the labelmap
        metrics under. A chamber is then scored on the piece of wall bounding
        it. Either way a point belongs to at most one structure, and the
        partition is decided once, at the reference frame.
        """
        labels = sorted(scored_labels)
        indices = self._mesh_label_points(mesh, labels)
        if indices is None:
            self.log_info(
                "Shape model carries no %s; attributing its points to the "
                "nearest structure surface instead.",
                self._MESH_LABEL_ARRAY,
            )
            indices = self._nearest_label_points(
                mesh,
                {
                    label: surface
                    for label, surface in self._label_surfaces(
                        reference_labelmap
                    ).items()
                    if label in scored_labels
                },
            )
        empty = [label for label in labels if indices.get(label, np.empty(0)).size == 0]
        if empty:
            self.log_warning(
                "No shape-model point belongs to label(s) %s; their displacement "
                "error is not measured.",
                empty,
            )
        self.log_info(
            "Shape-model points per structure: %s",
            ", ".join(
                f"{label}={indices.get(label, np.empty(0)).size}" for label in labels
            ),
        )
        return indices

    def _mesh_label_points(
        self, mesh: pv.DataSet, labels: list[int]
    ) -> Optional[dict[int, np.ndarray]]:
        """Point indices per structure from the mesh's own per-cell ids.

        ``None`` when the mesh does not carry them, or carries none of the ids
        being scored, which is the caller's signal to fall back. A point shared
        by cells of two structures --- a lobe fissure --- goes to the
        lowest-numbered of them, so no point is counted twice.
        """
        if self._MESH_LABEL_ARRAY not in mesh.cell_data:
            return None
        cell_labels = np.asarray(mesh.cell_data[self._MESH_LABEL_ARRAY]).ravel()
        if not set(cell_labels.tolist()) & set(labels):
            return None

        owner = np.full(mesh.n_points, -1, dtype=np.int64)
        contested = 0
        for label in labels:
            cells = np.flatnonzero(cell_labels == label).astype(int)
            if cells.size == 0:
                continue
            points = np.asarray(
                mesh.extract_cells(cells).point_data["vtkOriginalPointIds"],
                dtype=np.int64,
            )
            taken = owner[points]
            contested += int(np.count_nonzero(taken != -1))
            owner[points[taken == -1]] = label
        if contested:
            self.log_info(
                "%d shape-model point(s) sit on a boundary between structures; "
                "each went to the lowest-numbered structure touching it.",
                contested,
            )
        unclaimed = int(np.count_nonzero(owner == -1))
        if unclaimed:
            self.log_info(
                "%d shape-model point(s) carry no scored structure id and are "
                "left out of the displacement error.",
                unclaimed,
            )
        return {label: np.flatnonzero(owner == label) for label in labels}

    @staticmethod
    def _nearest_label_points(
        mesh: pv.DataSet, label_surfaces: dict[int, pv.PolyData]
    ) -> dict[int, np.ndarray]:
        """Point indices per structure, by nearest label surface.

        Costs one implicit distance per structure over the mesh points, paid
        once per case against :meth:`surface_rmse_mm`'s two per structure per
        stage.
        """
        labels = sorted(label_surfaces)
        distances = np.stack(
            [
                np.abs(
                    np.asarray(
                        mesh.copy().compute_implicit_distance(label_surfaces[label])[
                            "implicit_distance"
                        ],
                        dtype=np.float64,
                    )
                )
                for label in labels
            ]
        )
        nearest = np.argmin(distances, axis=0)
        return {
            label: np.flatnonzero(nearest == position)
            for position, label in enumerate(labels)
        }

    def _provenance(self, case_id: str, shape_parameters: Path) -> dict[str, Any]:
        """Case name, shape parameters and network weights, with their dates."""
        inference = self.movement_workflow.inference_workflow
        checkpoint = Path(inference.checkpoint_file)
        info = checkpoint.stat()
        coefficients = pnt.load_pca_coefficients(shape_parameters)
        provenance: dict[str, Any] = {
            "case_id": case_id,
            "shape_parameters_file": str(shape_parameters),
            "network_weights_file": str(checkpoint),
            # st_birthtime is the real creation time on Windows and macOS; on
            # Linux it is absent, where st_ctime is inode-change time rather
            # than creation, so fall back to the modification time.
            "network_weights_created": self._timestamp(
                getattr(info, "st_birthtime", info.st_mtime)
            ),
            "network_weights_modified": self._timestamp(info.st_mtime),
            "network_epoch": "final" if inference.epoch is None else inference.epoch,
        }
        for index, coefficient in enumerate(coefficients, start=1):
            provenance[f"pca_c{index:02d}"] = float(coefficient)
        return provenance

    @staticmethod
    def _timestamp(seconds: float) -> str:
        """Format a filesystem timestamp as an ISO-8601 UTC string."""
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(
            timespec="seconds"
        )

    def _score(
        self,
        case_id: str,
        stage: float,
        truth: itk.Image,
        truth_surfaces: dict[int, pv.PolyData],
        predicted: itk.Image,
        predicted_surfaces: dict[int, pv.PolyData],
        scored_labels: dict[int, str],
        provenance: dict[str, Any],
        report_dice: bool = True,
        displacement_errors: Optional[np.ndarray] = None,
        label_points: Optional[dict[int, np.ndarray]] = None,
    ) -> list[dict[str, Any]]:
        """One metric row per scored label of one stage.

        ``displacement_errors`` is every shape-model point's error at this
        stage, reduced per structure over the point indices ``label_points``
        gives it.
        """
        truth_array = itk.GetArrayViewFromImage(truth)
        predicted_array = itk.GetArrayViewFromImage(predicted)
        voxel_volume_mm3 = float(np.prod(np.asarray(truth.GetSpacing())))

        rows: list[dict[str, Any]] = []
        for label in sorted(scored_labels):
            truth_volume = self.volume_mm3(truth_array, label, voxel_volume_mm3)
            if truth_volume == 0.0:
                self.log_info(
                    "stage %.3f: %s absent from the acquired frame; skipped.",
                    stage,
                    scored_labels[label],
                )
                continue
            predicted_volume = self.volume_mm3(predicted_array, label, voxel_volume_mm3)
            rmse = (
                self.surface_rmse_mm(truth_surfaces[label], predicted_surfaces[label])
                if label in truth_surfaces and label in predicted_surfaces
                else float("nan")
            )
            row: dict[str, Any] = {
                "case_id": case_id,
                "stage": stage,
                "label_id": label,
                "label_name": scored_labels[label],
            }
            if report_dice:
                row["dice"] = self.dice(truth_array, predicted_array, label)
            row.update(
                {
                    "volume_truth_mm3": truth_volume,
                    "volume_predicted_mm3": predicted_volume,
                    "volume_difference_mm3": predicted_volume - truth_volume,
                    "volume_difference_percent": (
                        100.0 * (predicted_volume - truth_volume) / truth_volume
                    ),
                    "surface_rmse_mm": rmse,
                }
            )
            if displacement_errors is not None:
                points = (label_points or {}).get(label, np.array([], dtype=np.int64))
                displacement = self.pool_displacement_error(
                    [displacement_errors[points]] if points.size else []
                )
                row.update(
                    {
                        "displacement_rms_mm": displacement["rms_mm"],
                        "displacement_95th_mm": displacement["p95_mm"],
                        "displacement_max_mm": displacement["max_mm"],
                    }
                )
            row.update(
                {key: value for key, value in provenance.items() if key != "case_id"}
            )
            rows.append(row)
            self.log_info(
                "stage %.3f %-24s %sdV=%+.2f%%  rmse=%.3f mm%s",
                stage,
                scored_labels[label],
                f"dice={row['dice']:.4f}  " if report_dice else "",
                row["volume_difference_percent"],
                rmse,
                (
                    f"  d95={row['displacement_95th_mm']:.3f} mm"
                    if displacement_errors is not None
                    else ""
                ),
            )
        return rows
