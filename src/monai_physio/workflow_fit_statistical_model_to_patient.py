"""Model-to-image and model-to-model registration for anatomical models.

This module provides the WorkflowFitStatisticalModelToPatient class for registering generic
anatomical models to patient-specific imaging data and surface models.
The workflow includes:
1. Rough alignment using ICP (RegisterModelsICP)
1.5. Optional PCA-based registration (RegisterModelsPCA) if PCA data provided
2. Labelmap-based deformable registration (RegisterModelsDistanceMaps)
3. Optional final labelmap-to-image refinement using Icon

The registration is particularly useful for cardiac modeling where a generic heart model
needs to be fitted to patient-specific imaging data.

Key Features:
    - Automatic labelmap generation if not provided by user
    - Modular design using RegisterModelsICP, RegisterModelsPCA, and
        RegisterModelsDistanceMaps
    - Multi-stage registration pipeline:
        ICP → (optional PCA) → labelmap-to-labelmap → labelmap-to-image
    - Optional PCA-based shape fitting
    - Support for multi-label anatomical structures
    - Optional Icon-based final refinement
"""

import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv

from .contour_tools import ContourTools
from .image_tools import ImageTools
from .labelmap_tools import LabelmapTools
from .monai_physio_base import MONAIPhysioBase
from .register_images_greedy import RegisterImagesGreedy
from .register_images_icon import RegisterImagesICON
from .register_models_distance_maps import RegisterModelsDistanceMaps
from .register_models_icp import RegisterModelsICP
from .register_models_pca import RegisterModelsPCA
from .segment_anatomy_base import SegmentAnatomyBase
from .segment_heart_simpleware_trimmed_branches import (
    SegmentHeartSimplewareTrimmedBranches,
)
from .transform_tools import TransformTools
from .workflow_convert_image_to_vtk import WorkflowConvertImageToVTK


class WorkflowFitStatisticalModelToPatient(MONAIPhysioBase):
    """Register anatomical models using multi-stage ICP, labelmap-based, and image-based
        registration.

    This class provides a flexible workflow for registering generic anatomical models
    (e.g., cardiac models) to patient-specific surface models and images. The
    registration pipeline combines:
    - Initial model alignment using RegisterModelsICP (centroid + configurable ICP)
    - Labelmap-based deformable registration using RegisterModelsDistanceMaps (Greedy/ICON)
    - Optional final labelmap-to-image refinement using Icon registration

    **Registration Pipeline:**
        1. **ICP Alignment**: Rough alignment using RegisterModelsICP, rigid,
            similarity or affine per :attr:`icp_transform_type`
        2. **PCA Registration**: Performs PCA-based shape fitting using
            RegisterModelsPCA
        3. **Labelmap-to-Labelmap**: Deformable registration using RegisterModelsDistanceMaps
        4. **Labelmap-to-Image**: Final refinement

    Attributes:
        template_model (pv.DataSet): Generic anatomical model to be registered
        template_model_surface (pv.PolyData): Surface extracted from
            template_model_surface
        template_labelmap (itk.Image): Multi-label labelmap for template model
        template_mask (itk.Image): Binary mask for template model registration region
        patient_models (list of pv.DataSet): Patient-specific models
        patient_model_surface (pv.PolyData): Primary patient model surface (first in
            list)
        combined_patient_model (pv.PolyData): Merged patient models before surface
            extraction; used when use_surface=False.
        patient_image (itk.Image): Reference image providing coordinate frame
        patient_labelmap (itk.Image): Multi-label labelmap for patient model
        patient_mask (itk.Image): Binary mask for patient registration region
        mask_dilation_mm (float): Dilation for binary mask generation
        icp_transform_type (str): Alignment the Stage 1 ICP solves for, one of
            "Rigid", "Similarity" or "Affine". It has to match the value the
            statistical model was built with (set via set_icp_transform_type)
        distancemap_squared_max (Optional[float]): Saturation radius of the
            labelmap-to-labelmap distance maps, in squared millimeters. None
            means derive it from mask_dilation_mm as (1.25 * mask_dilation_mm)**2
        transform_tools (TransformTools): Transform utilities
        registrar_ICON (RegisterImagesICON): ICON registration instance
        registrar_Greedy (RegisterImagesGreedy): Greedy registration instance
        use_pca_registration (bool): Whether PCA registration is enabled (set via set_use_pca_registration)
        pca_model (dict): PCA model dict when PCA enabled; same structure as WorkflowCreateStatisticalModel output
        number_of_pca_components (int): Number of PCA components when PCA enabled
        labelmap_interior_object_ids (list): List of labelmap IDs corresponding to interior objects that should
            not be used when computing a distance map.
        icp_forward_point_transform : ICP transforms
        icp_inverse_point_transform : ICP inverse transforms
        icp_template_model_surface: template model surface after ICP alignment
        icp_template_model: template model (UnstructuredGrid) after ICP alignment
        pca_coefficients: PCA shape coefficients (if PCA used)
        pca_template_model (pv.DataSet): template model after PCA registration
            (if PCA used)
        pca_template_model_surface: template model surface after PCA registration (if
            PCA used)
        l2l_forward_transform: Labelmap-to-labelmap forward transform
        l2l_inverse_transform: Labelmap-to-labelmap inverse transform
        l2l_template_model_surface: template model surface after labelmap-to-labelmap
            registration
        l2i_forward_transform: Labelmap-to-image forward transform
        l2i_inverse_transform: Labelmap-to-image inverse transform
        l2i_template_model_surface: template model surface after labelmap-to-image
            registration
        l2i_template_labelmap: template labelmap after labelmap-to-image registration
        fitted_reference_model: Final registered model
        fitted_reference_mesh: Final registered model surface

    Example:
        >>> # Initialize with minimal parameters (no labelmap; no patient image -> reference created from patient models)
        >>> registrar = WorkflowFitStatisticalModelToPatient(
        ...     template_model=heart_model,
        ...     patient_models=[lv_model, mc_model, rv_model],
        ... )
        >>> registrar.set_mask_dilation_mm(20)
        >>> # To enable PCA registration, call before process():
        >>> # registrar.set_use_pca_registration(True, pca_model=pca_model_dict, number_of_pca_components=10)
        >>> # To enable labelmap-to-image refinement:
        >>> # registrar.set_use_labelmap_to_image_registration(True, template_labelmap, organ_mesh_ids, organ_extra_ids, background_ids)
        >>> result = registrar.process()
    """

    def __init__(
        self,
        template_model: pv.DataSet,
        patient_models: list[pv.DataSet] | None = None,
        patient_image: Optional[itk.Image] = None,
        patient_labelmap: Optional[itk.Image] = None,
        template_labelmap: Optional[itk.Image] = None,
        labelmap_interior_object_ids: Optional[list] = None,
        segmentation_method: Optional[SegmentAnatomyBase] = None,
        log_level: int | str = logging.INFO,
    ):
        """Initialize the model-to-image-and-model registration pipeline.

        Args:
            template_model: Generic anatomical model to be registered
            patient_models: List of patient-specific models extracted from imaging
                data. Typically 3 models for cardiac applications: LV, myocardium, RV.
            patient_image: Optional patient image providing the target coordinate frame.
                If None, a reference image is created from the patient model surface
                via create_reference_image (contour_tools).
            labelmap_interior_object_ids: Optional list of labelmap IDs that should
                not be used when computing the distance map since they are interior (surrounded
                by other objects).
            segmentation_method: Segmentation backend instance used by
                WorkflowConvertImageToVTK when patient_models is None and
                patient_image is provided. Defaults to a new
                :class:`SegmentHeartSimplewareTrimmedBranches` (matches
                KCL-Heart-Model template extent) when None.
                Ignored when patient_models is supplied.
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING).
                Default: logging.INFO

        Raises:
            TypeError: If segmentation_method is neither None nor a
                SegmentAnatomyBase instance.
        """
        # Initialize base class with logging
        super().__init__(
            class_name="WorkflowFitStatisticalModelToPatient", log_level=log_level
        )

        if segmentation_method is not None and not isinstance(
            segmentation_method, SegmentAnatomyBase
        ):
            raise TypeError(
                "segmentation_method must be a SegmentAnatomyBase instance or None"
            )

        self.template_model = template_model
        self.template_model_surface = template_model.extract_surface(
            algorithm="dataset_surface"
        )
        self.template_labelmap: Optional[itk.Image] = template_labelmap
        self.template_labelmap_organ_mesh_ids: Optional[list[int]] = None
        self.template_labelmap_organ_extra_ids: Optional[list[int]] = None
        self.template_labelmap_background_ids: Optional[list[int]] = None

        self.labelmap_interior_object_ids: Optional[list] = labelmap_interior_object_ids

        if patient_models is None and patient_image is not None:
            if segmentation_method is None:
                segmentation_method = SegmentHeartSimplewareTrimmedBranches(
                    log_level=log_level
                )
            convert_image_to_vtk = WorkflowConvertImageToVTK(
                segmentation_method=segmentation_method,
                log_level=log_level,
            )
            patient_models_data = convert_image_to_vtk.process(
                input_image=patient_image,
                anatomy_groups=["heart"],
            )
            patient_models = [patient_models_data["surfaces"]["heart"]]
        elif patient_models is None:
            raise ValueError("Either patient_models or patient_image must be provided.")
        self.patient_models = patient_models
        patient_models_surfaces = [
            model.extract_surface(algorithm="dataset_surface")
            for model in patient_models
        ]
        self.combined_patient_model = pv.merge(patient_models_surfaces)
        self.patient_model_surface = self.combined_patient_model.extract_surface(
            algorithm="dataset_surface"
        )
        self.patient_labelmap = patient_labelmap

        # Utilities (needed for create_reference_image when patient_image is None)
        self.transform_tools = TransformTools()
        self.contour_tools = ContourTools()
        self.labelmap_tools = LabelmapTools()

        if patient_image is not None:
            self.patient_image = patient_image
            spacing = np.asarray(patient_image.GetSpacing(), dtype=np.float64)
            isotropic_spacing = bool(np.allclose(spacing, spacing[0]))
            if not isotropic_spacing:
                self.patient_image = ImageTools().make_isotropic_image(
                    self.patient_image
                )
        else:
            self.patient_image = self.contour_tools.create_reference_image(
                mesh=self.patient_model_surface,
                spatial_resolution=1.0,
                buffer_factor=0.25,
                ptype=itk.F,
            )

        self.registrar_Greedy = RegisterImagesGreedy()
        self.registrar_Greedy.set_number_of_iterations([5, 2, 5])
        # Icon registration for final labelmap-to-image step
        self.registrar_ICON = RegisterImagesICON()
        self.registrar_ICON.set_modality("ct")
        self.registrar_ICON.set_mass_preservation(False)
        self.registrar_ICON.set_multi_modality(True)
        self.registrar_ICON.set_number_of_iterations(50)

        # Labelmap/mask configuration (auto-generated)
        self.template_mask: Optional[itk.Image] = None
        self.patient_mask: Optional[itk.Image] = None

        # Alignment applied before the PCA fit.  It must match the alignment
        # the model was built with; see WorkflowCreateStatisticalModel.
        self.icp_transform_type: str = "Affine"

        # Parameters for labelmap and mask generation
        self.mask_dilation_mm: float = 10.0  # For binary registration mask generation
        self.distancemap_squared_max: Optional[float] = None

        # Optional finetuned ICON checkpoint for the labelmap-to-labelmap stage
        self.l2l_icon_weights_path: Optional[str] = None

        # Stage 1: ICP alignment results
        self.icp_registrar: Optional[RegisterModelsICP] = None
        self.icp_inverse_point_transform: Optional[itk.Transform] = None
        self.icp_forward_point_transform: Optional[itk.Transform] = None
        self.icp_template_model: Optional[pv.DataSet] = None
        self.icp_template_model_surface: Optional[pv.PolyData] = None
        self.icp_template_labelmap: Optional[itk.Image] = None

        # Stage 1.5: PCA registration results (optional; enable via set_use_pca_registration(True, pca_model, number_of_pca_components))
        self.use_pca_registration = False
        self.pca_registrar: Optional[RegisterModelsPCA] = None
        self.pca_forward_point_transform: Optional[itk.Transform] = None
        self.pca_inverse_point_transform: Optional[itk.Transform] = None
        self.pca_model: Optional[dict[str, Any]] = None
        self.number_of_pca_components: int = 0
        self.pca_coefficients: Optional[np.ndarray] = None
        self.pca_template_model: Optional[pv.DataSet] = None
        self.pca_template_model_surface: Optional[pv.PolyData] = None
        self.pca_template_labelmap: Optional[itk.Image] = None
        self.use_surface: bool = False

        # Stage 2: Labelmap-to-labelmap registration results
        self.use_l2l_registration = True
        self.l2l_inverse_transform: Optional[itk.Transform] = None
        self.l2l_forward_transform: Optional[itk.Transform] = None
        self.l2l_template_model_surface: Optional[pv.PolyData] = None
        self.l2l_template_labelmap: Optional[itk.Image] = None

        # Stage 3: Labelmap-to-image registration results (disabled by default; enable via set_use_labelmap_to_image_registration(True, template_labelmap, ...))
        self.use_l2i_registration = False
        self.l2i_inverse_transform: Optional[itk.Transform] = None
        self.l2i_forward_transform: Optional[itk.Transform] = None
        self.l2i_template_model_surface: Optional[pv.PolyData] = None
        self.l2i_template_labelmap: Optional[itk.Image] = None

        self.use_ICON_registration_refinement = False

        # Final result
        self.fitted_reference_model: Optional[pv.DataSet] = None
        self.fitted_reference_mesh: Optional[pv.PolyData] = None
        self.fitted_reference_labelmap: Optional[itk.Image] = None

    def set_mask_dilation_mm(self, mask_dilation_mm: float) -> None:
        """Set dilation amount for binary registration masks.

        Args:
            mask_dilation_mm: Dilation amount in millimeters for binary registration
                mask generation. Default: 10mm
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_icp_transform_type(self, transform_type: str) -> None:
        """Set the ICP alignment applied to the template before the PCA fit.

        This has to match the alignment the model was built with -- see
        ``WorkflowCreateStatisticalModel.set_icp_transform_type`` -- because
        whatever the ICP absorbs here is variation the eigenmodes never saw.

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

    def set_distancemap_squared_max(self, distancemap_squared_max: float) -> None:
        """Set the saturation radius of the labelmap-to-labelmap distance maps.

        The radius fixes those images' intensity distribution, so it has to
        match the value the ICON weights in use were finetuned at -- see
        ``tutorials/parameters_lung_ct_dirlab.py`` and
        ``tutorials/parameters_heart_ct_kcl.py``, which carry one value per
        organ.  Left unset, it is derived from ``mask_dilation_mm``.

        Args:
            distancemap_squared_max: Maximum squared distance, in squared
                millimeters, the distance maps are normalized against.
        """
        self.distancemap_squared_max = distancemap_squared_max

    def _distancemap_squared_max(self) -> float:
        """Return the configured saturation radius, or one sized to the mask."""
        if self.distancemap_squared_max is not None:
            return self.distancemap_squared_max
        return (1.25 * self.mask_dilation_mm) ** 2

    def set_labelmap_to_labelmap_icon_weights_path(self, weights_path: str) -> None:
        """Set a finetuned ICON checkpoint for the labelmap-to-labelmap stage.

        That stage (:meth:`register_labelmap_to_labelmap`) registers distance
        maps rather than image intensities, so it benefits from weights
        finetuned on distance maps -- e.g. by
        ``tutorials/tutorial_02_lung_distancemap_finetune_icon.py``.  The
        labelmap-to-image stage keeps the stock weights: it registers the
        patient image itself.

        Args:
            weights_path: Path to an existing uniGradICON checkpoint.

        Raises:
            FileNotFoundError: If weights_path does not exist.
        """
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"ICON weights not found: {weights_path}")
        self.l2l_icon_weights_path = weights_path

    def set_use_pca_registration(
        self,
        use_pca_registration: bool,
        pca_model: Optional[dict[str, Any]] = None,
        number_of_pca_components: int = 0,
        use_surface: bool = False,
    ) -> None:
        """Set whether to use PCA-based registration and provide the PCA model.

        When enabling (True), pca_model and number_of_pca_components must be provided.

        Args:
            use_pca_registration: Whether to use PCA registration after ICP.
            pca_model: Required when use is True. PCA model dict (e.g. from
                WorkflowCreateStatisticalModel result["pca_model"]) with keys
                "eigenvalues" and "components".
            number_of_pca_components: Required when use is True. Number of PCA
                components to use. Default 0 means use all components.  A count
                larger than the model actually carries is reduced to what it
                carries; see below.
            use_surface: Whether to use the surface of the patient model for PCA registration.
        Raises:
            ValueError: If use is True and pca_model is None.
        """
        if use_pca_registration:
            if pca_model is None:
                raise ValueError(
                    "When enabling PCA registration, pca_model must be provided."
                )
            # A PCA model carries at most one fewer mode than it had samples,
            # and WorkflowCreateStatisticalModel already caps it there.  The
            # count configured for a full population is therefore too large for
            # a model built from a small one, and asking for more modes than
            # exist raises out of the optimizer partway through the fit.  Read
            # the count the model actually carries instead.
            available_components = len(pca_model.get("components", []))
            if available_components < number_of_pca_components:
                self.log_info(
                    "PCA model carries %d mode(s), fewer than the %d requested; "
                    "fitting with the %d available.",
                    available_components,
                    number_of_pca_components,
                    available_components,
                )
                number_of_pca_components = available_components
            self.pca_model = pca_model
            self.number_of_pca_components = number_of_pca_components
        else:
            self.pca_model = None
            self.number_of_pca_components = 0
        self.use_surface = use_surface
        self.use_pca_registration = use_pca_registration

    def set_use_labelmap_to_labelmap_registration(
        self, use_labelmap_to_labelmap_registration: bool
    ) -> None:
        """Set whether to use labelmap-to-labelmap registration.

        Args:
            use_labelmap_to_labelmap_registration: Whether to use labelmap-to-labelmap
                deformable registration. Default: True
        """
        self.use_l2l_registration = use_labelmap_to_labelmap_registration

    def set_use_labelmap_to_image_registration(
        self,
        use_labelmap_to_image_registration: bool,
        template_labelmap: Optional[itk.Image] = None,
        template_labelmap_organ_mesh_ids: Optional[list[int]] = None,
        template_labelmap_organ_extra_ids: Optional[list[int]] = None,
        template_labelmap_background_ids: Optional[list[int]] = None,
    ) -> None:
        """Set whether to use labelmap-to-image registration.

        When enabling (True), a template labelmap and label IDs must be provided
        so the workflow can propagate and refine the labelmap to the patient image.

        Args:
            use_labelmap_to_image_registration: Whether to use labelmap-to-image registration.
            template_labelmap: Template labelmap in template model space (same geometry
                as template_model). Required when use is True unless one was already
                supplied to the constructor, in which case that value is used.
            template_labelmap_organ_mesh_ids: Required when use is True. Label IDs for
                organ mesh in the template labelmap.
            template_labelmap_organ_extra_ids: Required when use is True. Label IDs for
                organ-extra structures in the template labelmap.
            template_labelmap_background_ids: Required when use is True. Label IDs for
                background in the template labelmap.

        Raises:
            ValueError: If use is True and any of template_labelmap or the id lists
                is None or missing.
        """
        if use_labelmap_to_image_registration:
            if template_labelmap is None:
                template_labelmap = self.template_labelmap
            if template_labelmap is None:
                raise ValueError(
                    "When enabling labelmap-to-image registration, template_labelmap must be provided."
                )
            if template_labelmap_organ_mesh_ids is None:
                raise ValueError(
                    "When enabling labelmap-to-image registration, "
                    "template_labelmap_organ_mesh_ids must be provided."
                )
            if template_labelmap_organ_extra_ids is None:
                raise ValueError(
                    "When enabling labelmap-to-image registration, "
                    "template_labelmap_organ_extra_ids must be provided."
                )
            if template_labelmap_background_ids is None:
                raise ValueError(
                    "When enabling labelmap-to-image registration, "
                    "template_labelmap_background_ids must be provided."
                )
            self.template_labelmap = template_labelmap
            self.template_labelmap_organ_mesh_ids = template_labelmap_organ_mesh_ids
            self.template_labelmap_organ_extra_ids = template_labelmap_organ_extra_ids
            self.template_labelmap_background_ids = template_labelmap_background_ids
        self.use_l2i_registration = use_labelmap_to_image_registration

    def _transform_model_dataset(
        self,
        model: pv.DataSet,
        tfm: itk.Transform,
        *,
        with_deformation_magnitude: bool = False,
    ) -> pv.DataSet:
        """Transform a model with topology-preserving handling by PyVista type."""
        if isinstance(model, pv.PolyData):
            return self.transform_tools.transform_pvcontour(
                model,
                tfm,
                with_deformation_magnitude=with_deformation_magnitude,
            )
        return self.transform_tools.transform_dataset(
            model,
            tfm,
            with_deformation_magnitude=with_deformation_magnitude,
        )

    def register_model_to_model_icp(self) -> dict:
        """Perform ICP alignment of template model to patient model.

        Uses RegisterModelsICP class for ICP alignment.

        Returns:
            dict: Dictionary containing:
                - 'forward_transform': used to warp an image from model to patient space
                - 'inverse_transform': used to warp an image from patient to model space
                - 'fitted_reference_mesh': Transformed model model surface
        """
        self.log_section("Stage 1: ICP Alignment (RegisterModelsICP)", width=70)

        # Create ICP registrar
        self.icp_registrar = RegisterModelsICP(fixed_model=self.patient_model_surface)

        # Run rigid ICP registration
        icp_result = self.icp_registrar.register(
            moving_model=self.template_model_surface,
            transform_type=self.icp_transform_type,
            max_iterations=2000,
        )

        # Store results
        # Note: Point transforms are in opposite direction from image transforms
        self.icp_forward_point_transform = icp_result["forward_point_transform"]
        self.icp_inverse_point_transform = icp_result["inverse_point_transform"]
        self.icp_template_model_surface = icp_result["registered_model"]

        self.icp_template_model = self._transform_model_dataset(
            self.template_model,
            self.icp_forward_point_transform,
        )

        if self.template_labelmap is not None:
            self.icp_template_labelmap = self.transform_tools.transform_image(
                self.template_labelmap,
                self.icp_inverse_point_transform,
                self.patient_image,
                interpolation_method="nearest",
            )
        else:
            self.icp_template_labelmap = None

        self.log_info("Stage 1 complete: ICP alignment finished.")

        self.fitted_reference_mesh = self.icp_template_model_surface
        self.fitted_reference_model = self.icp_template_model
        self.fitted_reference_labelmap = self.icp_template_labelmap

        return {
            "inverse_point_transform": self.icp_inverse_point_transform,
            "forward_point_transform": self.icp_forward_point_transform,
            "fitted_reference_model": self.icp_template_model,
            "fitted_reference_mesh": self.icp_template_model_surface,
            "fitted_reference_labelmap": self.icp_template_labelmap,
        }

    def register_model_to_model_pca(self) -> dict:
        """Perform PCA-based registration after ICP alignment.

        Uses RegisterModelsPCA to optimize shape coefficients against a distance
        map of the patient. The statistical model's modes are defined in the
        un-aligned template frame, so the registrar is given the raw template
        and the ICP alignment is passed as its ``post_pca_transform``.

        Returns:
            dict: Dictionary containing:
                - 'forward_point_transform': DisplacementFieldTransform mapping
                  un-aligned template points to their PCA-deformed positions.
                  It excludes the ICP alignment, which is applied separately.
                - 'inverse_point_transform': its inverse
                - 'pca_coefficients': PCA shape coefficients
                - 'fitted_reference_mesh': PCA-registered model surface

        Raises:
            ValueError: If PCA data has not been set
        """
        self.log_section(
            "Stage 2: PCA-Based Registration (RegisterModelsPCA)",
            width=70,
        )

        if not self.use_pca_registration or self.pca_model is None:
            self.pca_template_model = self.icp_template_model
            self.pca_template_model_surface = self.icp_template_model_surface
            self.pca_template_labelmap = self.icp_template_labelmap
            identity_transform = itk.CenteredAffineTransform[itk.D, 3].New()
            identity_transform.SetIdentity()
            self.pca_forward_point_transform = identity_transform
            self.pca_inverse_point_transform = identity_transform
            return {
                "pca_coefficients": None,
                "fitted_reference_model": self.pca_template_model,
                "fitted_reference_mesh": self.pca_template_model_surface,
                "fitted_reference_labelmap": self.pca_template_labelmap,
                "forward_point_transform": self.pca_forward_point_transform,
                "inverse_point_transform": self.pca_inverse_point_transform,
            }

        # PCA modes are directions in the statistical model's own training
        # frame, so they must be added to the un-aligned template and the ICP
        # alignment applied afterwards. Deforming the ICP-aligned template
        # instead would yield A*mean + sum(b*sigma*v) rather than
        # A*(mean + sum(b*sigma*v)), mis-rotating and mis-scaling every mode.
        pca_template_model: Optional[pv.DataSet]
        if self.use_surface:
            pca_template_model = self.template_model_surface
            fixed_model = self.patient_model_surface
            fixed_distance_map = None
        else:
            pca_template_model = self.template_model
            fixed_model = self.combined_patient_model
            if self.patient_labelmap is not None:
                fixed_distance_map = self.labelmap_tools.create_distance_map(
                    self.patient_labelmap,
                    max_distance_mm=10.0,
                    distance_scale=5.0,
                    preserve_labels=False,
                    exclude_labels=self.labelmap_interior_object_ids,
                    fill_background_only=True,
                )
            else:
                fixed_distance_map = None
        assert pca_template_model is not None, "PCA template model must be set"

        self.pca_registrar = RegisterModelsPCA.from_pca_model(
            pca_template_model=pca_template_model,
            pca_model=self.pca_model,
            pca_number_of_modes=self.number_of_pca_components,
            post_pca_transform=self.icp_forward_point_transform,
            fixed_model=fixed_model,
            fixed_distance_map=fixed_distance_map,
            reference_image=self.patient_image,
        )

        # Run complete PCA registration
        assert self.pca_registrar is not None, "PCA registrar must be initialized"
        result = self.pca_registrar.register()
        self.pca_coefficients = result["pca_coefficients"]
        registered_model = cast(pv.DataSet, result["registered_model"])
        if self.use_surface:
            self.pca_template_model_surface = cast(pv.PolyData, registered_model)
        else:
            self.pca_template_model_surface = registered_model.extract_surface(
                algorithm="dataset_surface"
            )

        # The PCA field is splatted at the *un-aligned* template's points, which
        # generally fall outside the patient image, so grid it in the template's
        # own frame; create_deformation_field drops samples that land off-grid.
        # The ICP alignment is applied separately, after this field.
        pca_field_reference_image = self.contour_tools.create_reference_image(
            pca_template_model,
            spatial_resolution=float(min(self.patient_image.GetSpacing())),
        )
        pca_transforms = self.pca_registrar.compute_pca_transforms(
            reference_image=pca_field_reference_image,
        )
        self.pca_forward_point_transform = pca_transforms["forward_point_transform"]
        self.pca_inverse_point_transform = pca_transforms["inverse_point_transform"]

        if self.log_level == logging.DEBUG:
            tfm_field = self.pca_forward_point_transform.GetDisplacementField()
            tfm_arr = itk.GetArrayFromImage(tfm_field)
            tfm_x_arr = tfm_arr[:, :, :, 0]
            tfm_y_arr = tfm_arr[:, :, :, 1]
            tfm_z_arr = tfm_arr[:, :, :, 2]
            tfm_x_img = itk.GetImageFromArray(tfm_x_arr)
            tfm_y_img = itk.GetImageFromArray(tfm_y_arr)
            tfm_z_img = itk.GetImageFromArray(tfm_z_arr)
            tfm_x_img.CopyInformation(tfm_field)
            tfm_y_img.CopyInformation(tfm_field)
            tfm_z_img.CopyInformation(tfm_field)

        if self.use_surface:
            # forward_point_transform excludes the post-PCA step and is defined
            # in the un-aligned template frame, so warp the raw volumetric
            # template with it and then apply the ICP alignment.
            assert self.icp_forward_point_transform is not None, (
                "ICP forward transform must be set"
            )
            deformed_template_model = self._transform_model_dataset(
                self.template_model,
                self.pca_forward_point_transform,
            )
            self.pca_template_model = self._transform_model_dataset(
                deformed_template_model,
                self.icp_forward_point_transform,
            )
        else:
            self.pca_template_model = registered_model

        # Store results

        self.fitted_reference_model = self.pca_template_model
        self.fitted_reference_mesh = self.pca_template_model_surface

        if self.template_labelmap is not None:
            # Resampling pulls back: a patient-grid sample is mapped by the ICP
            # inverse into the deformed-template frame and then by the PCA
            # inverse onto the un-deformed template. itk.CompositeTransform
            # applies its transforms in reverse order of addition, so the ICP
            # inverse is added last. Resampling the raw template labelmap in one
            # step also avoids a second round of nearest-neighbor sampling.
            assert self.icp_inverse_point_transform is not None, (
                "ICP inverse transform must be set"
            )
            pca_image_transform = itk.CompositeTransform[itk.D, 3].New()
            pca_image_transform.AddTransform(self.pca_inverse_point_transform)
            pca_image_transform.AddTransform(self.icp_inverse_point_transform)
            self.pca_template_labelmap = self.transform_tools.transform_image(
                self.template_labelmap,
                pca_image_transform,
                self.patient_image,
                interpolation_method="nearest",
            )
        else:
            self.pca_template_labelmap = None

        self.fitted_reference_labelmap = self.pca_template_labelmap

        self.log_info("Stage 2 complete: PCA registration finished.")

        return {
            "pca_coefficients": self.pca_coefficients,
            "forward_point_transform": self.pca_forward_point_transform,
            "inverse_point_transform": self.pca_inverse_point_transform,
            "fitted_reference_model": self.pca_template_model,
            "fitted_reference_mesh": self.pca_template_model_surface,
            "fitted_reference_labelmap": self.pca_template_labelmap,
        }

    def register_labelmap_to_labelmap(self) -> Optional[dict]:
        """Perform labelmap-based deformable registration of template model to patient model.

        Uses RegisterModelsDistanceMaps with Greedy affine followed by ICON deformable
        registration on distance maps derived from the model surfaces.

        Returns:
            dict: Dictionary containing:
                - 'forward_transform': template to patient space transform
                - 'inverse_transform': patient to template space transform
                - 'fitted_reference_mesh': Transformed template model surface
                - 'fitted_reference_labelmap': Transformed template labelmap
        """
        self.log_section(
            "Stage 3: Labelmap-to-Labelmap Deformable Registration",
            width=70,
        )

        if not self.use_l2l_registration:
            self.log_info("Labelmap-to-labelmap registration is not enabled.")
            return None

        # Create labelmap-based registrar
        assert self.pca_template_model_surface is not None, (
            "PCA template model surface must be set"
        )

        # Create a padded patient image since often the surface of interest
        # is not fully contained within the original image, which causes trouble
        # with distance map registration. The margin is physical -- it has to
        # hold the dilated masks -- so it is converted per axis rather than
        # padding a fixed voxel count on grids of any spacing.
        margin_mm = 2.5 * self.mask_dilation_mm
        spacing = np.asarray(self.patient_image.GetSpacing(), dtype=np.float64)
        pad_voxels = np.maximum(1, np.ceil(margin_mm / spacing)).astype(int).tolist()
        padded_patient_image = ImageTools().pad_image(
            self.patient_image, pad_voxels=pad_voxels, background_value=-1000
        )
        labelmap_registrar = RegisterModelsDistanceMaps(
            moving_model=self.pca_template_model_surface,
            fixed_model=self.patient_model_surface,
            reference_image=padded_patient_image,
            mask_dilation_mm=self.mask_dilation_mm,
            distance_squared_max=self._distancemap_squared_max(),
        )
        if self.l2l_icon_weights_path is not None:
            labelmap_registrar.set_icon_weights_path(self.l2l_icon_weights_path)

        # Run deformable registration
        l2l_result = labelmap_registrar.register(
            transform_type="Deformable",
        )

        # Store results
        self.l2l_forward_transform = l2l_result["forward_transform"]
        self.l2l_inverse_transform = l2l_result["inverse_transform"]
        self.l2l_template_model_surface = l2l_result["registered_model"]

        self.fitted_reference_mesh = self.l2l_template_model_surface

        if self.pca_template_labelmap is not None:
            self.l2l_template_labelmap = self.transform_tools.transform_image(
                self.pca_template_labelmap,
                self.l2l_forward_transform,
                self.patient_image,
                interpolation_method="nearest",
            )
        else:
            self.l2l_template_labelmap = None

        self.fitted_reference_labelmap = self.l2l_template_labelmap

        self.log_info("Stage 3 complete: Labelmap-to-labelmap registration finished.")

        return {
            "forward_transform": self.l2l_forward_transform,
            "inverse_transform": self.l2l_inverse_transform,
            "fitted_reference_mesh": self.l2l_template_model_surface,
            "fitted_reference_labelmap": self.l2l_template_labelmap,
        }

    def register_labelmap_to_image(
        self, use_ICON_refinement: bool = False
    ) -> Optional[dict]:
        """Perform labelmap-to-image refinement.

        Uses registration to align the propagated template labelmap to actual
        image intensities.

        Returns:
            dict: Dictionary containing:
                - 'inverse_transform': patient to template space transform
                - 'forward_transform': template to patient space transform
                - 'fitted_reference_mesh': Transformed template model surface
                - 'fitted_reference_labelmap': Transformed template labelmap
        """
        self.log_section(
            "Stage 4: Labelmap-to-Image Refinement (Icon Registration)", width=70
        )

        if (
            self.template_labelmap is None
            or self.template_labelmap_organ_mesh_ids is None
            or self.template_labelmap_organ_extra_ids is None
            or self.template_labelmap_background_ids is None
        ):
            raise ValueError(
                "Labelmap-to-image registration requires template labelmap and label IDs. "
                "Call set_use_labelmap_to_image_registration(True, template_labelmap, "
                "organ_mesh_ids, organ_extra_ids, background_ids) before process()."
            )
        propagated_labelmap = (
            self.l2l_template_labelmap
            or self.pca_template_labelmap
            or self.icp_template_labelmap
            or self.template_labelmap
        )
        if propagated_labelmap is None:
            raise ValueError(
                "Labelmap-to-image registration requires a propagated template labelmap. "
                "Provide template_labelmap via set_use_labelmap_to_image_registration(), "
                "or ensure an earlier stage (L2L, PCA, ICP) has produced one."
            )

        template_labelmap_arr = itk.GetArrayFromImage(propagated_labelmap).astype(
            np.uint16
        )
        template_labelmap_arr = np.where(
            np.isin(template_labelmap_arr, self.template_labelmap_background_ids),
            0,
            template_labelmap_arr,
        )
        template_labelmap_arr = np.where(
            np.isin(template_labelmap_arr, self.template_labelmap_organ_mesh_ids),
            1,
            template_labelmap_arr,
        )
        template_labelmap_arr = np.where(
            np.isin(template_labelmap_arr, self.template_labelmap_organ_extra_ids),
            1,
            template_labelmap_arr,
        )
        template_labelmap = itk.GetImageFromArray(template_labelmap_arr)
        template_labelmap.CopyInformation(propagated_labelmap)

        template_mask = self.labelmap_tools.convert_labelmap_to_mask(
            template_labelmap, dilation_in_mm=self.mask_dilation_mm
        )

        patient_mask = self.contour_tools.create_mask_from_mesh(
            self.patient_model_surface,
            self.patient_image,
        )
        patient_mask = self.labelmap_tools.convert_labelmap_to_mask(
            patient_mask, dilation_in_mm=self.mask_dilation_mm
        )

        self.registrar_Greedy.set_fixed_image(self.patient_image)
        self.registrar_Greedy.set_fixed_mask(patient_mask)

        result = self.registrar_Greedy.register(
            moving_image=template_labelmap, moving_mask=template_mask
        )
        self.l2i_inverse_transform = result["inverse_transform"]
        self.l2i_forward_transform = result["forward_transform"]

        if use_ICON_refinement:
            # Configure Icon registration
            self.registrar_ICON.set_fixed_image(self.patient_image)
            self.registrar_ICON.set_fixed_mask(patient_mask)

            # Perform Icon registration, refining the alignment found so far
            result = self.registrar_ICON.register_from(
                self.l2i_forward_transform,
                template_labelmap,
                moving_mask=template_mask,
            )
            self.l2i_inverse_transform = result["inverse_transform"]
            self.l2i_forward_transform = result["forward_transform"]

        # Transform model with result - use the best available pre-L2I surface.
        source_surface = (
            self.l2l_template_model_surface
            or self.pca_template_model_surface
            or self.icp_template_model_surface
        )
        if source_surface is None:
            raise ValueError(
                "Labelmap-to-image registration requires a propagated template model "
                "surface from an earlier stage (L2L, PCA, or ICP)."
            )
        self.l2i_template_model_surface = cast(
            pv.PolyData,
            self._transform_model_dataset(
                source_surface,
                self.l2i_inverse_transform,
                with_deformation_magnitude=True,
            ),
        )

        self.l2i_template_labelmap = self.transform_tools.transform_image(
            propagated_labelmap,
            self.l2i_forward_transform,
            self.patient_image,
            interpolation_method="nearest",
        )

        self.log_info("Stage 4 complete: Labelmap-to-image registration finished.")

        self.fitted_reference_mesh = self.l2i_template_model_surface

        self.fitted_reference_labelmap = self.l2i_template_labelmap

        return {
            "inverse_transform": self.l2i_inverse_transform,
            "forward_transform": self.l2i_forward_transform,
            "fitted_reference_mesh": self.l2i_template_model_surface,
            "fitted_reference_labelmap": self.l2i_template_labelmap,
        }

    def transform_model(
        self, base_model: Optional[pv.DataSet] = None
    ) -> Optional[pv.DataSet]:
        """Apply registration transforms to the model.

        Transforms the model through all registration stages.

        Args:
            base_model: Base model for generating the new model.
                If None, the template model is used.

        Returns:
            pv.DataSet: Registered model
        """
        self.log_info("Applying transforms to model...")

        if base_model is None:
            self.fitted_reference_model = self.template_model.copy(deep=True)
            assert self.fitted_reference_model is not None, (
                "Registered template model must be set"
            )
            transformed_model = self.fitted_reference_model
        else:
            transformed_model = base_model.copy(deep=True)

        transform_steps: list[tuple[str, itk.Transform]] = []
        if self.pca_coefficients is not None:
            # PCA registration runs in the un-aligned template frame and carries
            # the ICP alignment in its post-PCA transform, so ICP must not be
            # applied again here.
            assert self.pca_registrar is not None, "PCA registrar must be set"
            pca_transform = (
                self.pca_forward_point_transform
                or self.pca_registrar.forward_point_transform
            )
            if pca_transform is not None:
                transform_steps.append(("PCA", pca_transform))
            if self.pca_registrar.post_pca_transform is not None:
                transform_steps.append(
                    ("PCA post-transform", self.pca_registrar.post_pca_transform)
                )
        elif self.icp_forward_point_transform is not None:
            transform_steps.append(("ICP", self.icp_forward_point_transform))
        if self.use_l2l_registration and self.l2l_inverse_transform is not None:
            transform_steps.append(("Labelmap-to-labelmap", self.l2l_inverse_transform))
        if self.use_l2i_registration and self.l2i_inverse_transform is not None:
            transform_steps.append(("Labelmap-to-image", self.l2i_inverse_transform))

        for i, (name, tfm) in enumerate(transform_steps, start=1):
            self.log_progress(i, len(transform_steps), prefix=f"Applying {name}")
            transformed_model = self._transform_model_dataset(
                transformed_model,
                tfm,
            )

        new_points = np.asarray(transformed_model.points, dtype=float)

        self.log_info("Transform application complete.")

        if base_model is None:
            assert self.fitted_reference_model is not None, (
                "Registered template model must be set"
            )
            self.fitted_reference_model.points = new_points
            return self.fitted_reference_model
        transformed_model.points = new_points
        return transformed_model

    def process(
        self,
        use_ICON_registration_refinement: bool = False,
    ) -> dict:
        """Execute the complete multi-stage registration workflow.

        Runs registration stages in sequence:

        1. ICP alignment (RegisterModelsICP)
        2. PCA registration (PCA data was provided)
        3. Labelmap-to-labelmap deformable registration (RegisterModelsDistanceMaps)
        4. Optional labelmap-to-image refinement (Icon); requires template labelmap and IDs
            set via set_use_labelmap_to_image_registration(True, ...).

        Args:
            use_ICON_registration_refinement: Whether to apply ICON refinement in the
                labelmap-to-image stage (Stage 4). The labelmap-to-labelmap stage always
                uses Greedy affine + ICON deformable. Default: False

        Returns:
            dict with fitted_reference_model and fitted_reference_mesh
        """
        self.log_section("STARTING COMPLETE MODEL REGISTRATION WORKFLOW", width=70)

        self.use_ICON_registration_refinement = use_ICON_registration_refinement

        # Stage 1: ICP alignment
        self.register_model_to_model_icp()

        # Stage 2: Optional PCA registration (if PCA data was set)
        self.register_model_to_model_pca()

        # Stage 3: Optional Labelmap-to-labelmap deformable registration
        if self.use_l2l_registration:
            self.register_labelmap_to_labelmap()

        # Stage 4: Optional labelmap-to-image refinement
        if self.use_l2i_registration:
            self.register_labelmap_to_image(
                use_ICON_refinement=use_ICON_registration_refinement
            )

        _ = self.transform_model()

        self.log_section("REGISTRATION WORKFLOW COMPLETE", width=70)
        assert self.fitted_reference_mesh is not None, (
            "Registered template model surface must be set"
        )
        self.log_info(
            "Final registered patient model surface: %d points.",
            self.fitted_reference_mesh.n_points,
        )

        return {
            "fitted_reference_model": self.fitted_reference_model,
            "fitted_reference_mesh": self.fitted_reference_mesh,
            "fitted_reference_labelmap": self.fitted_reference_labelmap,
        }
