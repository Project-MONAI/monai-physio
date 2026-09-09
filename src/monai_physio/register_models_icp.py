"""ICP-based model-to-model registration for anatomical models.

This module provides the RegisterModelsICP class for aligning anatomical
models using Iterative Closest Point (ICP) algorithm. The workflow includes:
1. Initial centroid alignment
2. Isotropic bounding-box scaling (Similarity and Affine transform types)
3. Rigid, similarity or affine ICP alignment

The registration is particularly useful for initial rough alignment of generic
models to patient-specific anatomical data.

Key Features:
    - Centroid-based initial alignment
    - Bounding-box size matching before any ICP iteration
    - VTK ICP with rigid, similarity or affine transformation modes
    - Four-stage affine pipeline: centroid → scale → similarity ICP → affine ICP
    - Support for PyVista models
    - Automatic transform composition

Example:
    >>> import pyvista as pv
    >>> from monai_physio import RegisterModelsICP
    >>>
    >>> # Load models
    >>> moving_model = pv.read('generic_model.vtu')
    >>> fixed_model = pv.read('patient_surface.stl')
    >>>
    >>> # Run affine registration
    >>> registrar = RegisterModelsICP(fixed_model=fixed_model)
    >>> result = registrar.register(
    ...     transform_type='Affine',
    ...     moving_model=moving_model,
    ...     max_iterations=200,
    ... )
    >>>
    >>> # Access results
    >>> aligned_model = result['registered_model']
    >>> moving_to_fixed_transform = result['moving_to_fixed_transform']  # Moving to fixed
        # transform
"""

import logging
from typing import Optional

import itk
import numpy as np
import pyvista as pv
import vtk

from .monai_physio_base import MONAIPhysioBase
from .transform_tools import TransformTools


class RegisterModelsICP(MONAIPhysioBase):
    """Register anatomical models using Iterative Closest Point (ICP) algorithm.

    This class provides ICP-based alignment of 3D surface models with support for
    both rigid and affine transformation modes. The registration pipeline uses
    centroid alignment for initialization followed by VTK's ICP algorithm.

    **Registration Pipelines:**
        - **Rigid**: Centroid alignment → Rigid ICP
        - **Similarity**: Centroid alignment → bounding-box scaling →
          Similarity ICP
        - **Affine**: Centroid alignment → bounding-box scaling →
          Similarity ICP → Affine ICP

        The bounding-box scaling estimates the size ratio between the models from
        their bounding-box diagonals before ICP starts. It is skipped for 'Rigid',
        whose transform has no scale degree of freedom.

    **Transform Convention:**
        These are POINT transforms, applied with TransformPoint (e.g. via
        TransformTools.transform_pvcontour); see
        docs/developer/transform_conventions:

        - moving_to_fixed_transform: maps moving points -> fixed points; use it to
          warp the moving model/landmarks onto the fixed model.
        - fixed_to_moving_transform: maps fixed points -> moving points.

    Attributes:
        moving_model (pv.PolyData): Surface model to be aligned
        fixed_model (pv.PolyData): Target surface model
        transform_tools (TransformTools): Transform utility instance
        moving_to_fixed_transform (itk.AffineTransform): Optimized moving-to-fixed transform
        fixed_to_moving_transform (itk.AffineTransform): Optimized fixed-to-moving transform
        registered_model (pv.PolyData): Aligned moving model

    Example:
        >>> # Initialize with model
        >>> registrar = RegisterModelsICP(fixed_model=patient_surface)
        >>>
        >>> # Run rigid registration
        >>> result = registrar.register(
        ...     transform_type='Rigid',
        ...     max_iterations=200,
        ...     moving_model=model_surface,
        ... )
        >>>
        >>> # Or run affine registration
        >>> result = registrar.register(
        ...     transform_type='Affine',
        ...     max_iterations=200,
        ...     moving_model=model_surface,
        ... )
        >>>
        >>> # Get aligned model and transforms
        >>> aligned_model = result['registered_model']
        >>> moving_to_fixed_transform = result['moving_to_fixed_transform']
    """

    def __init__(
        self,
        fixed_model: pv.PolyData,
        log_level: int | str = logging.INFO,
    ):
        """Initialize ICP-based model registration.

        Args:
            moving_model: PyVista surface model to be aligned to fixed model
            fixed_model: PyVista target surface model
            log_level: Logging level (default: logging.INFO)

        Note:
            The moving_model is typically extracted from a VTU model using
            model.extract_surface(algorithm="dataset_surface") before passing to this class.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.moving_model: Optional[pv.PolyData] = None
        self.fixed_model = fixed_model
        self.transform_type = "Affine"

        # Transform utilities
        self.transform_tools = TransformTools()

        # Registration results
        self.moving_to_fixed_transform: Optional[itk.AffineTransform] = None
        self.fixed_to_moving_transform: Optional[itk.AffineTransform] = None
        self.registered_model: Optional[pv.PolyData] = None

    # ICP stages run for each transform type, in order.
    _ICP_STAGES = {
        "Rigid": ("Rigid",),
        "Similarity": ("Similarity",),
        "Affine": ("Similarity", "Affine"),
    }

    def _icp_stage(
        self, model: pv.PolyData, mode: str, max_iterations: int
    ) -> tuple[pv.PolyData, itk.AffineTransform]:
        """Run one VTK ICP stage against the fixed model.

        Args:
            model: Current state of the moving model.
            mode: Landmark-transform mode, one of ``"Rigid"``, ``"Similarity"``
                or ``"Affine"``.
            max_iterations: Maximum ICP iterations for this stage.

        Returns:
            Tuple of the transformed model and this stage's moving-to-fixed
            point transform.
        """
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(model)
        icp.SetTarget(self.fixed_model)
        landmark = icp.GetLandmarkTransform()
        if mode == "Rigid":
            landmark.SetModeToRigidBody()
        elif mode == "Similarity":
            landmark.SetModeToSimilarity()
        else:
            landmark.SetModeToAffine()
        icp.SetMaximumNumberOfIterations(max_iterations)
        icp.Update()

        stage_transform = self.transform_tools.convert_vtk_matrix_to_itk_transform(
            icp.GetMatrix()
        )
        transformed = self.transform_tools.transform_pvcontour(
            model,
            stage_transform,
            with_deformation_magnitude=False,
        )
        return transformed, stage_transform

    def _bounding_box_scale(
        self, moving_model: pv.PolyData, fixed_model: pv.PolyData
    ) -> float:
        """Return the isotropic scale matching the models' bounding-box diagonals.

        The diagonal is used rather than the three side lengths separately: the
        bounding boxes are axis-aligned, so per-axis ratios would fold any
        residual rotation between the models into an anisotropic stretch. A
        single factor changes size without distorting shape, leaving the
        remaining anisotropy to the affine ICP stage.

        Args:
            moving_model: Model whose size is being matched.
            fixed_model: Model supplying the target size.

        Returns:
            The scale factor, or ``1.0`` when either model is degenerate (a
            single point or a zero-extent bounding box), where no meaningful
            ratio exists.
        """

        def _diagonal(model: pv.PolyData) -> float:
            x_min, x_max, y_min, y_max, z_min, z_max = model.bounds
            extents = np.array(
                [x_max - x_min, y_max - y_min, z_max - z_min], dtype=np.float64
            )
            return float(np.linalg.norm(extents))

        moving_diagonal = _diagonal(moving_model)
        fixed_diagonal = _diagonal(fixed_model)
        self.log_debug(
            "Bounding-box diagonals - moving: %.4f, fixed: %.4f",
            moving_diagonal,
            fixed_diagonal,
        )
        if moving_diagonal <= 0.0 or fixed_diagonal <= 0.0:
            self.log_warning(
                "Degenerate bounding box (moving diagonal %.4f, fixed diagonal "
                "%.4f); skipping the scaling step.",
                moving_diagonal,
                fixed_diagonal,
            )
            return 1.0
        return fixed_diagonal / moving_diagonal

    def _scale_transform(self, scale: float, center: np.ndarray) -> itk.AffineTransform:
        """Build the transform scaling isotropically about ``center``.

        Args:
            scale: Isotropic scale factor.
            center: Fixed point of the scaling, in world coordinates.

        Returns:
            An ITK affine point transform mapping ``p`` to
            ``center + scale * (p - center)``.
        """
        matrix = np.eye(3, dtype=np.float64) * scale
        offset = itk.Vector[itk.D, 3]()
        for i in range(3):
            offset[i] = float(center[i]) * (1.0 - scale)

        transform = itk.AffineTransform[itk.D, 3].New()
        transform.SetIdentity()
        transform.SetMatrix(itk.Matrix[itk.D, 3, 3](itk.GetVnlMatrixFromArray(matrix)))
        transform.SetOffset(offset)
        return transform

    def register(
        self,
        moving_model: pv.PolyData,
        transform_type: str = "Affine",
        max_iterations: int = 2000,
    ) -> dict:
        """Perform ICP alignment of moving model to fixed model.

        **Rigid transform type** (rotation + translation):
            1. Centroid alignment: Translate moving model to align mass centers
            2. Rigid ICP: Refine with rigid-body transformation

        **Similarity transform type** (rotation + translation + one uniform scale):
            1. Centroid alignment: Translate moving model to align mass centers
            2. Bounding-box scaling: Scale isotropically about the fixed centroid
                so the two bounding-box diagonals match, which keeps ICP's
                closest-point search from locking onto a size mismatch
            3. Similarity ICP: Refine rotation, translation and uniform scale

        **Affine transform type** (adds anisotropic scale and shear):
            1. Centroid alignment: Translate moving model to align mass centers
            2. Bounding-box scaling: As above
            3. Similarity ICP: Refine rotation, translation and uniform scale
            4. Affine ICP: Further refine with affine transformation

        'Rigid' skips the bounding-box scaling so its result stays a pure
        rigid-body transform; use 'Similarity' when the models differ in size but
        the shape should not be distorted.

        Args:
            moving_model: PyVista surface model to be aligned to fixed model
            transform_type: Registration transform type, one of 'Rigid',
                'Similarity' or 'Affine'. Default: 'Affine'
            max_iterations: Maximum number of ICP iterations per stage. Default: 2000

        Returns:
            Dictionary containing:
                - 'registered_model': Aligned moving model (PyVista PolyData)
                - 'moving_to_fixed_transform': Moving-to-fixed transform
                    (ITK AffineTransform)
                - 'fixed_to_moving_transform': Fixed-to-moving transform
                    (ITK AffineTransform)

        Raises:
            ValueError: If transform_type is not 'Rigid', 'Similarity' or 'Affine'

        Example:
            >>> # Rigid registration
            >>> result = registrar.register(
            ...     transform_type='Rigid',
            ...     max_iterations=5000,
            ...     moving_model=moving_model,
            ... )
            >>>
            >>> # Similarity registration (rigid plus one uniform scale)
            >>> result = registrar.register(
            ...     transform_type='Similarity',
            ...     max_iterations=2000,
            ...     moving_model=moving_model,
            ... )
            >>>
            >>> # Affine registration
            >>> result = registrar.register(
            ...     transform_type='Affine',
            ...     max_iterations=2000,
            ...     moving_model=moving_model,
            ... )
        """
        if transform_type not in self._ICP_STAGES:
            raise ValueError(
                f"Invalid transform '{transform_type}'. Must be one of "
                f"{sorted(self._ICP_STAGES)}."
            )

        self.log_section("%s ICP Alignment", transform_type.upper())

        self.moving_model = moving_model
        self.transform_type = transform_type

        # Centroid alignment (common to every mode)
        registered_model = self.moving_model.copy(deep=True)

        moving_centroid = np.array(registered_model.center)
        self.log_debug("Moving model centroid: %s", moving_centroid)
        fixed_centroid = np.array(self.fixed_model.center)
        self.log_debug("Fixed model centroid: %s", fixed_centroid)
        translation = fixed_centroid - moving_centroid
        self.log_info("Translating by %s to align centroids...", translation)

        # Create ITK affine transform with translation
        moving_to_fixed_transform = itk.AffineTransform[itk.D, 3].New()
        moving_to_fixed_transform.SetIdentity()
        moving_to_fixed_transform.SetOffset(translation)

        # Apply centroid alignment to model
        registered_model = self.transform_tools.transform_pvcontour(
            registered_model,
            moving_to_fixed_transform,
            with_deformation_magnitude=False,
        )

        self.log_debug("Center after centroid alignment: %s", registered_model.center)

        # Bounding-box scaling, for the modes whose transform admits a scale. ICP
        # only searches for correspondences among nearest points, so a template
        # that differs from the patient in overall size drags the closest-point
        # matching into a local minimum. Matching the bounding-box diagonals first
        # puts the two models on the same scale before any ICP iteration runs.
        # 'Rigid' skips it: a size estimate there would make the result a
        # similarity transform, which is what 'Similarity' is for.
        if transform_type != "Rigid":
            scale = self._bounding_box_scale(registered_model, self.fixed_model)
            self.log_info(
                "Scaling by %.4f about the fixed centroid to match bounding boxes...",
                scale,
            )
            scale_transform = self._scale_transform(scale, fixed_centroid)
            moving_to_fixed_transform.Compose(scale_transform)
            registered_model = self.transform_tools.transform_pvcontour(
                registered_model,
                scale_transform,
                with_deformation_magnitude=False,
            )
            self.log_debug("Bounds after scaling: %s", registered_model.bounds)

        # ICP stages. Affine runs similarity first so the shear and anisotropic
        # scale degrees of freedom refine an already-oriented, already-sized model
        # rather than absorbing rotation and overall scale themselves.
        for stage in self._ICP_STAGES[transform_type]:
            self.log_info(
                "Performing %s ICP (max iterations: %d)...",
                stage.lower(),
                max_iterations,
            )
            registered_model, stage_transform = self._icp_stage(
                registered_model, stage, max_iterations
            )
            moving_to_fixed_transform.Compose(stage_transform)
            self.log_debug("Center after %s ICP: %s", stage, registered_model.center)

        # Compute inverse transform
        # Ths forward transform for ICP is consistent with the transform convention
        # used with images-to-images registration.
        self.registered_model = registered_model
        self.moving_to_fixed_transform = moving_to_fixed_transform
        self.fixed_to_moving_transform = moving_to_fixed_transform.GetInverseTransform()

        self.log_info("%s ICP registration complete!", transform_type.upper())

        # Return results as dictionary
        return {
            "registered_model": self.registered_model,
            "moving_to_fixed_transform": self.moving_to_fixed_transform,
            "fixed_to_moving_transform": self.fixed_to_moving_transform,
        }
