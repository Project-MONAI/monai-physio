"""Movement interpretation of PhysicsNeMo mesh-stage predictions.

:class:`WorkflowInferMovement` wraps a
:class:`monai_physio.WorkflowInferPhysicsNeMo` whose targets are per-point
displacements from the subject's fitted reference mesh, and turns those raw
predictions into geometry: deformed meshes (``fitted reference + displacement``)
and rasterized deformation / surface-normal fields. Scoring those predictions is
:class:`monai_physio.WorkflowEvaluateMovement`'s job, so no error statistic is
computed here.

The generic workflow stays target-agnostic; everything that assumes "the target
is a 3-vector displacement in mm" lives here. Composition, not inheritance, so
one decoder serves both the MeshGraphNet and the MLP inference methods.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional, cast

import itk
import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .monai_physio_base import MONAIPhysioBase
from .transform_tools import TransformTools
from .workflow_convert_vtk_to_usd import WorkflowConvertVTKToUSD
from .workflow_infer_physicsnemo import WorkflowInferPhysicsNeMo


class WorkflowInferMovement(MONAIPhysioBase):
    """Reconstruct geometry from displacement predictions.

    The displacements are added to the subject's fitted reference mesh — the
    manifest's ``fitted_reference_mesh``, or the ``fitted_reference_mesh``
    argument of the single-subject methods — which keeps the result in that
    mesh's world frame. That mesh is what
    :class:`monai_physio.WorkflowFitStatisticalModelToPatient` produces: PCA
    shape parameters *and* a deformable registration to the patient. A surface
    reconstructed from the shape parameters alone is not a substitute and is not
    accepted.

    Args:
        inference_workflow: A loaded :class:`WorkflowInferPhysicsNeMo` whose
            model predicts three-component displacements.
        log_level: Logging level. Default: ``logging.INFO``.

    Raises:
        ValueError: If the wrapped model does not predict exactly three
            components, which cannot be a displacement.
    """

    def __init__(
        self,
        inference_workflow: WorkflowInferPhysicsNeMo,
        log_level: int | str = logging.INFO,
    ) -> None:
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        if inference_workflow.n_target != 3:
            raise ValueError(
                "WorkflowInferMovement needs a 3-component target, but the "
                f"model predicts {inference_workflow.n_target} components."
            )
        self.inference_workflow = inference_workflow
        self.model_directory = inference_workflow.model_directory

    def _fitted_reference_points(self, fitted_reference_mesh: pv.DataSet) -> np.ndarray:
        """Return the points the displacements are added to.

        Raises:
            ValueError: If the supplied mesh has the wrong point count.
        """
        points = np.asarray(fitted_reference_mesh.points, dtype=np.float32)
        n_expected = self.inference_workflow.template_mesh.n_points
        if points.shape[0] != n_expected:
            raise ValueError(
                f"Fitted reference mesh has {points.shape[0]} points, expected "
                f"{n_expected} (template topology)."
            )
        return points

    # ─────────────────────────── Public API ────────────────────────────────
    def process(
        self,
        subject_manifest: Path,
        stages: Optional[list[float]] = None,
        output_directory: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Predict a subject's deformed meshes from a manifest.

        Every phase in the manifest is predicted, or the arbitrary ``stages``
        given instead. The displacements are added to the manifest's
        ``fitted_reference_mesh`` points. Scoring the result against a ground
        truth belongs to :class:`monai_physio.WorkflowEvaluateMovement`.

        Args:
            subject_manifest: Path to the subject manifest JSON.
            stages: Optional list of stages to predict.
            output_directory: Output directory; defaults to
                ``<model_directory>/<subject_id>``.

        Returns:
            Dict with ``subject_id`` and ``predicted_surfaces`` (paths).
        """
        workflow = self.inference_workflow
        manifest = pnt.parse_manifest(subject_manifest)
        pca_coeffs = pnt.load_pca_coefficients(manifest.pca_coefficients)
        fitted_reference_mesh = cast(
            pv.DataSet, pv.read(str(manifest.fitted_reference_mesh))
        )
        fitted_reference_points = self._fitted_reference_points(fitted_reference_mesh)

        out_dir = (
            Path(output_directory)
            if output_directory is not None
            else self.model_directory / manifest.subject_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        sid = manifest.subject_id
        self.log_section("INFER MOVEMENT [%s]", sid)

        suffix = ".vtp" if isinstance(fitted_reference_mesh, pv.PolyData) else ".vtu"
        requested = stages if stages is not None else [p.stage for p in manifest.phases]
        surfaces: list[Path] = []

        for stage in requested:
            pred_points = fitted_reference_points + workflow.predict(pca_coeffs, stage)
            pred_mesh = fitted_reference_mesh.copy(deep=True)
            pred_mesh.points = pred_points
            path = out_dir / f"{sid}_s{int(stage * 100):03d}_pred{suffix}"
            pred_mesh.save(str(path))
            surfaces.append(path)
            self.log_info("stage %.3f -> %s", stage, path.name)

        return {"subject_id": sid, "predicted_surfaces": surfaces}

    def predict_single(
        self,
        shape_parameters: Path,
        stage: float,
        fitted_reference_mesh: Path,
        output_directory: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Predict one subject at one stage without a manifest.

        The prediction stays in the fitted reference mesh's world frame, since
        that is where the displacements are applied.

        Args:
            shape_parameters: JSON file with the subject PCA coefficient vector.
            stage: Target stage to predict.
            fitted_reference_mesh: The subject's fitted reference mesh, as
                produced by
                :class:`monai_physio.WorkflowFitStatisticalModelToPatient`.
            output_directory: Output directory; defaults to
                ``<model_directory>/single_prediction``.

        Returns:
            Dict with ``predicted_surface`` (path) and ``predicted_points``.
        """
        workflow = self.inference_workflow
        coeffs = pnt.load_pca_coefficients(shape_parameters)
        fitted_mesh = cast(pv.DataSet, pv.read(str(fitted_reference_mesh)))
        fitted_reference_points = self._fitted_reference_points(fitted_mesh)
        pred_points = fitted_reference_points + workflow.predict(coeffs, stage)

        out_dir = (
            Path(output_directory)
            if output_directory is not None
            else self.model_directory / "single_prediction"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_mesh = fitted_mesh.copy(deep=True)
        pred_mesh.points = pred_points
        suffix = ".vtp" if isinstance(fitted_mesh, pv.PolyData) else ".vtu"
        stem = Path(shape_parameters).stem
        path = out_dir / f"{stem}_pred_s{int(stage * 100):03d}{suffix}"
        pred_mesh.save(str(path))
        self.log_info("single prediction stage %.3f -> %s", stage, path.name)

        result: dict[str, Any] = {
            "predicted_surface": path,
            "predicted_points": pred_points,
        }
        return result

    def process_time_series(
        self,
        shape_parameters: Path,
        stages: Sequence[float],
        output_directory: Path,
        fitted_reference_mesh: Path,
        reference_image: Optional[itk.Image] = None,
        warp_interpolation: str = "linear",
        warp_background_value: float = 0.0,
        smoothing_sigma_mm: float = 10.0,
        usd_project_name: Optional[str] = None,
        anatomy_type: Optional[str] = None,
        separate_by_connectivity: bool = False,
    ) -> dict[str, Any]:
        """Predict one subject across a whole time series and write its geometry.

        One prediction per entry of ``stages``, each written as a mesh. When
        ``reference_image`` is supplied, each stage also gets a deformation
        field, which is Gaussian-smoothed into a continuous
        :class:`itk.DisplacementFieldTransform` and used to carry
        ``reference_image`` into that stage's frame. The smoothing spreads a
        surface-shell field into the volume, so the warped image is an
        interpolation of the surface motion, not an independent registration.

        Args:
            shape_parameters: JSON file with the subject PCA coefficient vector.
            stages: Stages to predict, in the order they are to be animated.
            output_directory: Directory every artifact is written to.
            fitted_reference_mesh: The subject's fitted reference mesh, as
                produced by
                :class:`monai_physio.WorkflowFitStatisticalModelToPatient`. The
                displacements are added to its points and the result stays in
                its world frame.
            reference_image: Image carried through each stage's deformation, and
                the grid the deformation field is rasterized on. Omit to write
                meshes only.
            warp_interpolation: Interpolation used to resample
                ``reference_image``: ``"linear"`` for intensity images,
                ``"nearest"`` for labelmaps and masks.
            warp_background_value: Value written where a stage's grid samples
                outside ``reference_image``. ``0.0`` suits labelmaps; CT needs
                ``-1000.0``, which is air in Hounsfield units.
            smoothing_sigma_mm: Gaussian sigma, in millimeters, that turns the
                sparse surface-shell field into a continuous deformation.
            usd_project_name: When given, the stage meshes are also written as
                one animated USD under this name, one time sample per stage.
            anatomy_type: Anatomy whose materials color that USD.
            separate_by_connectivity: Whether that USD splits each frame into
                separate objects by connectivity.

        Returns:
            Dict with ``stages``, ``predicted_surfaces``, ``warped_images``,
            ``transforms``, ``usd_file`` and ``stage_meshes``. Entries that were
            not requested are empty lists or ``None``.

            ``stage_meshes`` are the very objects written to
            ``predicted_surfaces``, handed back so a caller that scores them ---
            :class:`monai_physio.WorkflowEvaluateMovement` --- can annotate and
            re-save them without re-reading. They are retained for the USD
            writer regardless, so returning them costs nothing.

        Raises:
            ValueError: If ``stages`` is empty.
        """
        if not stages:
            raise ValueError("process_time_series needs at least one stage.")

        workflow = self.inference_workflow
        coeffs = pnt.load_pca_coefficients(shape_parameters)
        fitted_mesh = cast(pv.DataSet, pv.read(str(fitted_reference_mesh)))
        fitted_reference_points = self._fitted_reference_points(fitted_mesh)

        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(shape_parameters).stem
        suffix = ".vtp" if isinstance(fitted_mesh, pv.PolyData) else ".vtu"
        self.log_section("INFER MOVEMENT TIME SERIES [%s]", stem)

        transform_tools = TransformTools(log_level=self.log_level)
        stage_meshes: list[pv.DataSet] = []
        surfaces: list[Path] = []
        warped_images: list[Path] = []
        transforms: list[itk.Transform] = []

        for index, stage in enumerate(stages):
            tag = f"s{int(stage * 100):03d}"
            pred_points = fitted_reference_points + workflow.predict(coeffs, stage)
            pred_mesh = fitted_mesh.copy(deep=True)
            pred_mesh.points = pred_points
            # The displacement the network itself predicted. Scoring it against
            # a ground truth is the evaluation workflow's job, not this one's.
            pred_mesh.point_data["predicted_displacement_mm"] = (
                pred_points - fitted_reference_points
            ).astype(np.float32)
            surface_file = out_dir / f"{stem}_{tag}_pred{suffix}"
            pred_mesh.save(str(surface_file))
            stage_meshes.append(pred_mesh)
            surfaces.append(surface_file)

            if reference_image is not None:
                field = self.create_deformation_field(
                    shape_parameters=shape_parameters,
                    stage=stage,
                    reference_image=reference_image,
                    fitted_reference_mesh=fitted_reference_mesh,
                    direction="inverse",
                )
                transform = transform_tools.smooth_deformation_field_transform(
                    field["deformation_field"],
                    sigma=smoothing_sigma_mm,
                    weight_image=field["weight_image"],
                )
                transforms.append(transform)
                warped = transform_tools.transform_image(
                    reference_image,
                    transform,
                    reference_image=reference_image,
                    interpolation_method=warp_interpolation,
                    background_value=warp_background_value,
                )
                warped_file = out_dir / f"{stem}_{tag}_warped.mha"
                itk.imwrite(warped, str(warped_file), compression=True)
                warped_images.append(warped_file)

            self.log_info("stage %.3f -> %s", stage, surface_file.name)

        usd_file: Optional[Path] = None
        if usd_project_name is not None:
            usd_workflow = WorkflowConvertVTKToUSD(
                input_meshes=stage_meshes,
                usd_project_name=usd_project_name,
                output_directory=out_dir,
                appearance="anatomy" if anatomy_type is not None else "solid",
                anatomy_type=anatomy_type,
                separate_by_connectivity=separate_by_connectivity,
                frames_per_second=float(len(stage_meshes)),
                log_level=self.log_level,
            )
            usd_file = Path(usd_workflow.process()["usd_file"])

        return {
            "stages": list(stages),
            "predicted_surfaces": surfaces,
            "warped_images": warped_images,
            "transforms": transforms,
            "usd_file": usd_file,
            "stage_meshes": stage_meshes,
        }

    def create_deformation_field(
        self,
        shape_parameters: Path,
        stage: float,
        reference_image: itk.Image,
        fitted_reference_mesh: Path,
        output_directory: Optional[Path] = None,
        direction: Literal["forward", "inverse"] = "forward",
    ) -> dict[str, Any]:
        """Rasterize the inferred deformation onto a reference image grid.

        Each mesh vertex is binned by its **reference (undeformed) position**
        into ``reference_image``'s voxel grid. Each voxel of the deformation
        field holds the mean network displacement ``(dx, dy, dz)`` of the
        vertices that fall in it; each voxel of the normal image holds the mean
        (renormalized) reference-surface normal of those vertices. Empty voxels
        are zero.

        That is the ``"forward"`` field, which maps reference positions to stage
        positions and is what transforming a *mesh* needs. Resampling an
        *image*, though, maps each output point through the transform to find
        where to sample the input, so carrying the reference image into the
        stage frame needs the opposite mapping. ``direction="inverse"`` builds
        it exactly rather than by negating the forward field: each vertex is
        binned by its **deformed** position ``reference + displacement`` and
        contributes ``-displacement``.

        The binning positions come from ``fitted_reference_mesh``, so a patient
        scan whose statistical-model fit applied a pose transform not captured
        by the shape coefficients is binned where it actually aligns with
        ``reference_image``. The network displacements themselves depend only on
        the coefficients and stage, not on the binning positions.

        Args:
            shape_parameters: JSON file with the subject PCA coefficient vector.
            stage: Target stage for the deformation.
            reference_image: The frame's image; defines the output grid geometry
                (size, spacing, origin, direction).
            fitted_reference_mesh: The subject's fitted reference mesh, as
                produced by
                :class:`monai_physio.WorkflowFitStatisticalModelToPatient`. Its
                points supply the binning positions and normals, and must share
                the template topology (same point count and ordering).
            output_directory: If given, the three images are written there as
                compressed ``.mha`` files.
            direction: ``"forward"`` for the reference-to-stage field that
                deforms meshes, ``"inverse"`` for the stage-to-reference field
                that resamples images into the stage frame.

        Returns:
            Dict with ``deformation_field`` and ``normal_image`` (ITK vector
            images), ``weight_image`` (the vertex count per voxel, which
            distinguishes an empty voxel from one whose displacement happens to
            be zero and is what
            :meth:`TransformTools.smooth_deformation_field_transform` normalizes
            by), ``deformed_surface`` (the stage mesh as ``pv.DataSet``) and,
            when written, their paths.
        """
        workflow = self.inference_workflow
        coeffs = pnt.load_pca_coefficients(shape_parameters)
        fitted_mesh = cast(pv.DataSet, pv.read(str(fitted_reference_mesh)))
        fitted_reference_points = self._fitted_reference_points(fitted_mesh)
        disps = workflow.predict(coeffs, stage)

        # Reference (undeformed) surface normals. Extraction drops the interior
        # points of a volumetric template, so the normals come back on a subset
        # of the points in a different order; ``vtkOriginalPointIds`` scatters
        # them back into ``fitted_reference_points`` order. Interior points keep a zero
        # normal, which contributes nothing to a voxel's mean.
        normals_mesh = fitted_mesh.copy(deep=True)
        normals_mesh.points = fitted_reference_points
        surface = normals_mesh.extract_surface(
            pass_pointid=True, algorithm="dataset_surface"
        ).compute_normals(
            point_normals=True, cell_normals=False, auto_orient_normals=True
        )
        surface_normals = np.asarray(surface.point_data["Normals"], dtype=np.float64)
        original_ids = np.asarray(
            surface.point_data["vtkOriginalPointIds"], dtype=np.intp
        )
        normals = np.zeros((fitted_reference_points.shape[0], 3), dtype=np.float64)
        normals[original_ids] = surface_normals

        size = itk.size(reference_image)  # x, y, z
        sx, sy, sz = int(size[0]), int(size[1]), int(size[2])
        disp_sum = np.zeros((sz, sy, sx, 3), dtype=np.float64)
        normal_sum = np.zeros((sz, sy, sx, 3), dtype=np.float64)
        count = np.zeros((sz, sy, sx), dtype=np.float64)

        if direction == "inverse":
            bin_points = fitted_reference_points + disps
            bin_disps = -disps
        else:
            bin_points = fitted_reference_points
            bin_disps = disps

        for i in range(bin_points.shape[0]):
            point = [float(c) for c in bin_points[i]]
            idx = reference_image.TransformPhysicalPointToIndex(point)
            ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
            if 0 <= ix < sx and 0 <= iy < sy and 0 <= iz < sz:
                disp_sum[iz, iy, ix] += bin_disps[i]
                normal_sum[iz, iy, ix] += normals[i]
                count[iz, iy, ix] += 1.0

        occupied = count > 0
        disp_field = np.zeros_like(disp_sum, dtype=np.float32)
        normal_field = np.zeros_like(normal_sum, dtype=np.float32)
        disp_field[occupied] = (disp_sum[occupied] / count[occupied, None]).astype(
            np.float32
        )
        mean_normal = normal_sum[occupied] / count[occupied, None]
        norm = np.linalg.norm(mean_normal, axis=1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        normal_field[occupied] = (mean_normal / norm).astype(np.float32)

        deformation_image = self._vector_image_like(disp_field, reference_image)
        normal_image = self._vector_image_like(normal_field, reference_image)
        weight_image = self._scalar_image_like(
            count.astype(np.float32), reference_image
        )
        self.log_info(
            "Deformation field: %d/%d voxels populated by %d vertices",
            int(occupied.sum()),
            sx * sy * sz,
            bin_points.shape[0],
        )

        # Deformed (stage) mesh: reference positions displaced by the network,
        # keeping the template topology.
        deformed_surface = fitted_mesh.copy(deep=True)
        deformed_surface.points = (fitted_reference_points + disps).astype(np.float32)

        result: dict[str, Any] = {
            "deformation_field": deformation_image,
            "normal_image": normal_image,
            "weight_image": weight_image,
            "deformed_surface": deformed_surface,
        }
        if output_directory is not None:
            out_dir = Path(output_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = ".vtp" if isinstance(fitted_mesh, pv.PolyData) else ".vtu"
            field_path = out_dir / "deformation_field.mha"
            normal_path = out_dir / "surface_normal_field.mha"
            weight_path = out_dir / "deformation_weight.mha"
            surface_path = out_dir / f"deformed_surface{suffix}"
            itk.imwrite(deformation_image, str(field_path), compression=True)
            itk.imwrite(normal_image, str(normal_path), compression=True)
            itk.imwrite(weight_image, str(weight_path), compression=True)
            deformed_surface.save(str(surface_path))
            result["deformation_field_file"] = field_path
            result["normal_image_file"] = normal_path
            result["weight_image_file"] = weight_path
            result["deformed_surface_file"] = surface_path
        return result

    @staticmethod
    def _vector_image_like(array: np.ndarray, reference_image: itk.Image) -> itk.Image:
        """Wrap a ``(z, y, x, 3)`` array as an ITK vector image on ``reference``'s grid."""
        image = itk.image_from_array(np.ascontiguousarray(array), is_vector=True)
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())
        return image

    @staticmethod
    def _scalar_image_like(array: np.ndarray, reference_image: itk.Image) -> itk.Image:
        """Wrap a ``(z, y, x)`` array as an ITK scalar image on ``reference``'s grid."""
        image = itk.image_from_array(np.ascontiguousarray(array))
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())
        return image
