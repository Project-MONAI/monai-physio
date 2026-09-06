"""Create a PCA statistical shape model from a sample of meshes.

This module provides the WorkflowCreateStatisticalModel class that implements
the pipeline from the Heart-Create_Statistical_Model experiment notebooks

Optionally forms PCA from/for surfaces or from full meshes.

Returns a dictionary of surfaces, meshes, and PCA model structure (no file I/O).
"""

import logging
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

from .contour_tools import ContourTools
from .monai_physio_base import MONAIPhysioBase
from .register_models_distance_maps import RegisterModelsDistanceMaps
from .register_models_icp import RegisterModelsICP
from .transform_tools import TransformTools


class WorkflowCreateStatisticalModel(MONAIPhysioBase):
    """Create a PCA statistical shape model from a sample of meshes aligned to a reference.

    Pipeline:

    1. Extract surfaces from sample and reference meshes, or keep as meshes
    2. ICP alignment: align each sample surface to the reference (template) surface. Always
       extract surfaces for ICP alignment.
    3. Deformable registration: establish dense correspondence via Greedy affine + ICON deformable registration.  Uses
       either full meshes or surfaces.
    4. Correspondence: warp reference model by each transform to get aligned shapes,
       optionally snapped onto the measured ICP-aligned surfaces
    5. PCA: compute mean and modes from corresponded shapes


    Attributes:
        sample_meshes (list): List of sample mesh DataSets (.vtk/.vtu/.vtp geometry)
        reference_mesh (pv.DataSet): Reference mesh; its surface is used for alignment
        number_of_pca_components (int): Number of PCA components to retain
        reference_spatial_resolution (float): Resolution for reference image from mesh
        reference_buffer_factor (float): Buffer around mesh for reference image
        icp_transform_type (str): Alignment applied before correspondence
        mask_dilation_mm (float): Dilation of the deformable stage's masks
        distance_squared_max (float): Squared mm the distance maps saturate at
        project_to_measured_surfaces (bool): Snap corresponded points onto the
            measured surfaces before the PCA
        projection_max_distance_mm (Optional[float]): Residual above which a
            point is left unprojected
        pca_input_residual_rms (list[float]): Per sample, the RMS distance from
            the corresponded shape to the measured surface, before projection
    """

    def __init__(
        self,
        sample_meshes: list[pv.DataSet],
        reference_mesh: pv.DataSet,
        number_of_pca_components: int = 7,
        reference_spatial_resolution: float = 1.0,
        reference_buffer_factor: float = 0.25,
        solve_for_surface_pca: bool = True,
        icp_transform_type: str = "Affine",
        mask_dilation_mm: float = 20.0,
        distance_squared_max: Optional[float] = None,
        project_to_measured_surfaces: bool = True,
        projection_max_distance_mm: Optional[float] = None,
        log_level: int | str = logging.INFO,
    ):
        """Initialize the create-statistical-model workflow.

        Args:
            sample_meshes: List of sample mesh DataSets (PyVista PolyData or UnstructuredGrid).
            reference_mesh: Reference mesh; its surface is used to align all samples.
            number_of_pca_components: Number of PCA components. Default 7.
            reference_spatial_resolution: Isotropic resolution (mm) for reference image. Default 1.0.
            reference_buffer_factor: Buffer factor around mesh for reference image. Default 0.25.
            solve_for_surface_pca: Whether to reduce the reference mesh to a surface. Default True.
            icp_transform_type: Alignment applied before correspondence, one of
                ``"Rigid"``, ``"Similarity"`` or ``"Affine"``. Default
                ``"Affine"``, which normalizes size and gross proportion away,
                so the modes describe only the residual shape. The workflow
                that fits this model must align the same way.
            mask_dilation_mm: Dilation (mm) of the binary registration masks used
                by the deformable stage. Default 20.0.
            distance_squared_max: Squared millimetres the distance maps are
                normalized against, so the saturation radius is its square root.
                Default None, which sizes it to the mask as the fitting workflow
                does: ``(1.25 * mask_dilation_mm) ** 2``.
            project_to_measured_surfaces: Snap each corresponded point onto the
                subject's measured ICP-aligned surface before the PCA. Default
                True. Ignored when solve_for_surface_pca is False.
            projection_max_distance_mm: Only project points whose residual is at
                or below this distance. Default None, which projects every point.
            log_level: Logging level.
        """
        super().__init__(
            class_name="WorkflowCreateStatisticalModel", log_level=log_level
        )
        self.sample_meshes = list(sample_meshes)
        self.reference_mesh = reference_mesh
        self.number_of_pca_components = number_of_pca_components
        self.reference_spatial_resolution = reference_spatial_resolution
        self.reference_buffer_factor = reference_buffer_factor
        self.solve_for_surface_pca = solve_for_surface_pca
        # Through the setters, so the constructor cannot accept a value the
        # setter would reject.
        self.set_icp_transform_type(icp_transform_type)
        self.mask_dilation_mm = mask_dilation_mm
        if distance_squared_max is None:
            self.distance_squared_max = (1.25 * mask_dilation_mm) ** 2
        elif distance_squared_max <= 0.0:
            # The distance maps are normalized against its square root, so zero
            # or less saturates every voxel alike and leaves the registration
            # nothing to descend.
            raise ValueError(
                f"distance_squared_max must be positive, got {distance_squared_max}."
            )
        else:
            self.distance_squared_max = distance_squared_max
        self.project_to_measured_surfaces = project_to_measured_surfaces
        self.projection_max_distance_mm = projection_max_distance_mm
        self.icon_weights_path: Optional[str] = None

        self.contour_tools = ContourTools()
        self.transform_tools = TransformTools()

        # Set by pipeline
        self.reference_model: Optional[pv.DataSet] = None
        self.sample_models: list[pv.DataSet] = []
        self.sample_ids: list[str] = []
        self.aligned_models: list[pv.DataSet] = []
        self.fixed_to_moving_transforms: list = []
        self.pca_input_models: list[pv.DataSet] = []
        self.pca_input_residual_rms: list[float] = []
        self.pca_fitted: Optional[PCA] = None
        self.pca_mean_surface: Optional[pv.PolyData] = None
        self.pca_mean_mesh: Optional[pv.DataSet] = None

    def set_number_of_pca_components(self, n: int) -> None:
        """Set number of PCA components to retain."""
        self.number_of_pca_components = n

    def set_icp_transform_type(self, transform_type: str) -> None:
        """Set the alignment applied to each sample before correspondence.

        ``"Affine"`` (default) normalizes size, anisotropic scale and shear
        away, leaving only residual shape for the PCA; ``"Rigid"`` keeps them
        as modes.  Whatever is chosen, the workflow that fits the model has to
        align the same way, or the model is asked to explain variation its ICP
        has already absorbed.

        Args:
            transform_type: One of ``"Rigid"``, ``"Similarity"``, ``"Affine"``.

        Raises:
            ValueError: If transform_type is not one of those.
        """
        if transform_type not in ("Rigid", "Similarity", "Affine"):
            raise ValueError(
                f"Invalid ICP transform '{transform_type}'. "
                "Must be 'Rigid', 'Similarity' or 'Affine'."
            )
        self.icp_transform_type = transform_type

    def set_icon_weights_path(self, weights_path: str) -> None:
        """Use a finetuned uniGradICON checkpoint for the deformable stage.

        Stock weights are out of distribution for distance maps, so without this
        the correspondences the model is built from barely move off the
        template; see ``RegisterModelsDistanceMaps.set_icon_weights_path``.

        Args:
            weights_path: Path to an existing uniGradICON checkpoint.
        """
        self.icon_weights_path = weights_path

    def _step1_extract_surfaces(self) -> None:
        """Extract reference surface and all sample surfaces (notebook 1)."""
        self.log_section("Step 1: Extract reference and sample surfaces", width=70)
        if not self.sample_meshes:
            raise ValueError("sample_meshes must not be empty")
        if self.solve_for_surface_pca:
            self.reference_model = self.contour_tools.extract_surface(
                self.reference_mesh
            )
        else:
            self.reference_model = self.reference_mesh
        self.log_info(
            "Reference surface: %d points",
            self.reference_model.n_points,
        )
        self.sample_models = []
        self.sample_ids = []
        for i, mesh in enumerate(self.sample_meshes):
            model: pv.DataSet
            if self.solve_for_surface_pca:
                model = self.contour_tools.extract_surface(mesh)
            else:
                model = mesh
            self.sample_models.append(model)
            self.sample_ids.append(str(i))
        self.log_info("Extracted %d sample models", len(self.sample_models))

    def _step2_icp_align(self) -> None:
        """ICP (affine) align each sample surface to reference (notebook 2)."""
        self.log_section("Step 2: ICP alignment to reference surface", width=70)
        assert self.reference_model is not None and self.sample_models
        self.aligned_models = []
        self.fixed_to_moving_transforms = []

        reference_surface = self.contour_tools.extract_surface(self.reference_model)
        for i, (sid, moving) in enumerate(zip(self.sample_ids, self.sample_models)):
            self.log_info(
                "ICP aligning %s (%d/%d)", sid, i + 1, len(self.sample_models)
            )
            # Always extract surfaces for ICP alignment
            moving_surface = self.contour_tools.extract_surface(moving)
            registrar = RegisterModelsICP(fixed_model=reference_surface)
            result = registrar.register(
                moving_model=moving_surface,
                transform_type=self.icp_transform_type,
                max_iterations=2000,
            )
            if self.solve_for_surface_pca:
                aligned_model = result["registered_model"]
            else:
                aligned_model = self.contour_tools.transform_contours(
                    cast(pv.PolyData, moving),
                    tfm=result["moving_to_fixed_transform"],
                    with_deformation_magnitude=False,
                )
            self.aligned_models.append(aligned_model)
            self.fixed_to_moving_transforms.append(result["moving_to_fixed_transform"])
        self.log_info("ICP alignment complete for %d samples", len(self.aligned_models))

    def _step3_deformable_correspondence(self) -> None:
        """Deformable registration of each aligned sample to reference (notebook 3)."""
        self.log_section("Step 3: Deformable registration (correspondence)", width=70)
        assert self.reference_model is not None and self.aligned_models
        # The distance-map grid must contain the template *and* every aligned
        # sample: a sample clipped by the grid registers as if it were smaller,
        # which biases every PCA input toward the template's size.
        bounding_cloud = pv.PolyData(
            np.vstack(
                [np.asarray(self.reference_model.points)]
                + [np.asarray(model.points) for model in self.aligned_models]
            )
        )
        reference_image = self.contour_tools.create_reference_image(
            mesh=bounding_cloud,
            spatial_resolution=self.reference_spatial_resolution,
            buffer_factor=self.reference_buffer_factor,
            ptype=itk.UC,
        )
        self.fixed_to_moving_transforms = []

        for i, (sid, moving) in enumerate(zip(self.sample_ids, self.aligned_models)):
            self.log_info(
                "Deformable registration %s (%d/%d)",
                sid,
                i + 1,
                len(self.aligned_models),
            )
            registrar = RegisterModelsDistanceMaps(
                moving_model=cast(pv.PolyData, moving),
                fixed_model=cast(pv.PolyData, self.reference_model),
                reference_image=reference_image,
                distance_squared_max=self.distance_squared_max,
                mask_dilation_mm=self.mask_dilation_mm,
            )
            if self.icon_weights_path is not None:
                registrar.set_icon_weights_path(self.icon_weights_path)
            result = registrar.register(
                transform_type="Deformable",
            )

            # Only the fixed_to_moving transform is kept.  Each one owns dense
            # full-grid displacement fields, and holding the moving_to_fixed
            # one as well doubled the cost of a population for a value
            # nothing read.
            self.fixed_to_moving_transforms.append(result["fixed_to_moving_transform"])

        # aligned_models stays the ICP-aligned input: step 4 measures the
        # corresponded shapes against it.
        self.log_info(
            "Deformable registration complete for %d samples",
            len(self.fixed_to_moving_transforms),
        )

    def _step4_build_pca_inputs(self) -> None:
        """Build corresponded shapes in reference space (notebook 4).

        For each case, reference_model is warped by the fixed_to_moving_transform
        from step 3, so that we get reference topology in ICP-aligned space with
        residual deformation per subject to be used as PCA input.

        The warped template only approximates its subject: the deformable
        registration always stops short of it, and because that shortfall is
        one-sided it shrinks every subject toward the template, and the PCA's
        variance with it.  The shortfall is measured here against the measured
        ICP-aligned surface, and removed when ``project_to_measured_surfaces``
        is set, so that the modes are scaled by the population's real spread.

        For a volume template only the boundary nodes are measured, since an
        interior node's distance to the bounding surface says nothing about how
        well the registration landed.
        """
        self.log_section("Step 4: Build PCA inputs (corresponded shapes)", width=70)
        assert self.reference_model is not None and self.fixed_to_moving_transforms
        project = self.project_to_measured_surfaces
        if project and not self.solve_for_surface_pca:
            # The PCA inputs are volume meshes here, so snapping their interior
            # nodes onto the bounding surface would collapse the mesh.
            self.log_warning(
                "Ignoring project_to_measured_surfaces: it is only meaningful "
                "for surface PCA."
            )
            project = False

        self.pca_input_models = []
        self.pca_input_residual_rms = []
        for sid, fwd_tfm, aligned in zip(
            self.sample_ids, self.fixed_to_moving_transforms, self.aligned_models
        ):
            pca_input_model = self.contour_tools.transform_contours(
                cast(pv.PolyData, self.reference_model),
                tfm=fwd_tfm,
                with_deformation_magnitude=False,
            )
            measured_surface = self.contour_tools.extract_surface(aligned)
            points = np.asarray(pca_input_model.points)
            measured_ids: Any = slice(None)
            if not self.solve_for_surface_pca:
                # A volume template's interior nodes sit a wall thickness away
                # from the bounding surface by construction, so scoring them
                # against it would report that thickness rather than the
                # registration's shortfall.  Only the boundary is measured.
                measured_ids = np.asarray(
                    pca_input_model.extract_surface(
                        algorithm="dataset_surface"
                    ).point_data["vtkOriginalPointIds"]
                )
            measured_points = points[measured_ids]
            _, closest = cast(
                "tuple[np.ndarray, np.ndarray]",
                measured_surface.find_closest_cell(
                    measured_points, return_closest_point=True
                ),
            )
            residuals = np.linalg.norm(closest - measured_points, axis=1)
            self.pca_input_residual_rms.append(float(np.sqrt(np.mean(residuals**2))))
            if project:
                if self.projection_max_distance_mm is None:
                    points[:] = closest
                else:
                    # Where the registration landed far from the subject its
                    # nearest point is not the corresponding one, and snapping
                    # would fold several template points onto one feature.
                    accepted = residuals <= self.projection_max_distance_mm
                    points[accepted] = closest[accepted]
                pca_input_model.points = points
            self.pca_input_models.append(pca_input_model)
            self.log_info(
                "  %s: %.3f mm RMS from the measured surface (max %.3f mm)",
                sid,
                self.pca_input_residual_rms[-1],
                float(residuals.max()),
            )

        # Compare the registration's shortfall against the spread it has to
        # measure: a residual near the spread means the modes are mostly noise.
        rows = np.array([np.asarray(m.points).ravel() for m in self.pca_input_models])
        deviations = rows - rows.mean(axis=0)
        population_rms = float(
            np.sqrt(np.mean(np.sum(deviations**2, axis=1) / (rows.shape[1] // 3)))
        )
        self.log_info(
            "Built %d corresponded surfaces for PCA: residual %.3f mm RMS, "
            "population spread %.3f mm RMS",
            len(self.pca_input_models),
            float(np.mean(self.pca_input_residual_rms)),
            population_rms,
        )

    def _step5_compute_pca(self) -> None:
        """Compute PCA and mean surface (notebook 5)."""
        self.log_section("Step 5: Compute PCA model", width=70)
        assert self.reference_model is not None and self.pca_input_models
        template = self.reference_model
        n_points = template.n_points

        data_rows: list[np.ndarray] = []
        for i, model in enumerate(self.pca_input_models):
            if model.n_points != n_points:
                raise ValueError(
                    f"Sample {self.sample_ids[i]} has {model.n_points} points, "
                    f"expected {n_points}. Topology must match."
                )
            data_rows.append(model.points.flatten())
        data_matrix = np.array(data_rows)

        if data_matrix.shape[0] - 1 < 2:
            raise ValueError(
                f"At least 2 samples are required for PCA. Got {data_matrix.shape[0]} samples."
            )
        n_comp = min(self.number_of_pca_components, data_matrix.shape[0] - 1)
        if n_comp < self.number_of_pca_components:
            self.log_warning(
                "Reducing PCA components from %d to %d (n_samples=%d)",
                self.number_of_pca_components,
                n_comp,
                data_matrix.shape[0],
            )
        self.pca_fitted = PCA(n_components=n_comp)
        self.pca_fitted.fit(data_matrix)

        pca_mean_model = template.copy()
        pca_mean_model.points = self.pca_fitted.mean_.reshape(-1, 3)
        self.log_info(
            "PCA complete: %d components, variance explained %.4f",
            len(self.pca_fitted.explained_variance_ratio_),
            self.pca_fitted.explained_variance_ratio_.sum(),
        )

        reference_image = self.contour_tools.create_reference_image(
            mesh=pca_mean_model,
            spatial_resolution=self.reference_spatial_resolution,
            buffer_factor=self.reference_buffer_factor,
            ptype=itk.UC,
        )
        mean_deformation_array = pca_mean_model.points - template.points
        mean_deformation_field = self.contour_tools.create_deformation_field(
            points=template.points,
            point_displacements=mean_deformation_array,
            reference_image=reference_image,
            blur_sigma=2.5,
            ptype=itk.D,
        )
        mean_deformation_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
        mean_deformation_transform.SetDisplacementField(mean_deformation_field)
        if self.solve_for_surface_pca:
            self.pca_mean_mesh = self.contour_tools.transform_contours(
                cast(pv.PolyData, self.reference_model),
                tfm=mean_deformation_transform,
                with_deformation_magnitude=False,
            )
            self.pca_mean_surface = cast(pv.PolyData, pca_mean_model)
        else:
            self.pca_mean_mesh = pca_mean_model
            self.pca_mean_surface = pca_mean_model.extract_surface(
                algorithm="dataset_surface"
            )

    def _build_result(self) -> dict[str, Any]:
        """Build result dictionary: surfaces, meshes, and PCA model structure."""
        assert self.pca_mean_surface is not None and self.pca_fitted is not None
        result: dict[str, Any] = {
            "pca_mean_surface": self.pca_mean_surface,
            "pca_mean_mesh": self.pca_mean_mesh,
            "pca_model": {
                "explained_variance_ratio": self.pca_fitted.explained_variance_ratio_.tolist(),
                "eigenvalues": self.pca_fitted.explained_variance_.tolist(),
                "components": [c.tolist() for c in self.pca_fitted.components_],
            },
            "pca_fitted": self.pca_fitted,
        }
        return result

    def process(self) -> dict[str, Any]:
        """Run the full pipeline and return a dictionary of results (no file I/O).

        Returns:
            dict with keys:
                - pca_mean_surface: pv.PolyData mean shape surface
                - pca_mean_mesh: pv.UnstructuredGrid reference volume mesh, or None if reference was surface-only
                - pca_model: dict with "explained_variance_ratio", "eigenvalues", "components" (same structure as pca_model.json)
                - pca_fitted: fitted sklearn PCA object
        """
        self.log_section("STARTING CREATE STATISTICAL MODEL WORKFLOW", width=70)
        self._step1_extract_surfaces()
        self._step2_icp_align()
        self._step3_deformable_correspondence()
        self._step4_build_pca_inputs()
        self._step5_compute_pca()
        result = self._build_result()
        self.log_section("CREATE STATISTICAL MODEL WORKFLOW COMPLETE", width=70)
        return result
