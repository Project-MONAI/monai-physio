from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import itk
import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from typing_extensions import Self

from .contour_tools import ContourTools
from .monai_physio_base import MONAIPhysioBase
from .transform_tools import TransformTools


class RegisterModelsPCA(MONAIPhysioBase):
    """Register PCA-based shape models to images by minimizing a distance metric.

    This class implements a registration pipeline for fitting statistical
    shape models to patient-specific medical images:

    **PCA Deformable Registration**
        - Optimizes PCA coefficients
        - Model equation: P = template + Σ(b_i * std_i * pca_eigenvector_i)
        - Minimizes the mean distance-to-target at the deformed model points P

    **Optimization Objective:**
        ``fixed_distance_map`` is zero on the target surface and grows with
        distance away from it (in mm), so the objective is *minimized*::

            E(b) = (1 - w) * mean_i D(P_i(b))          # model -> target
                 +       w * mean_j ||Q_j - P_nn(j)||  # target -> model
                 + lambda * Σ b_i²                     # Mahalanobis prior

        ``w`` is ``symmetric_weight`` and ``lambda`` is ``pca_prior_weight``.
        Because the deformation is linear in ``b``, the gradient is analytic and
        is supplied to the optimizer directly.

    **Coordinate frames:**
        The eigenvectors are directions in the statistical model's own training
        frame, so they are only valid when added to a template in that same
        frame. Any rigid/affine alignment to the target must be supplied as
        ``post_pca_transform``, which is applied *after* the deformation, rather
        than pre-applied to ``pca_template_model``.

    Attributes:
        pca_template_model (pv.DataSet): Mean shape model
        pca_eigenvectors (np.ndarray): PCA eigenvectors/components (modes × n_points*3)
        pca_std_deviations (np.ndarray): Standard deviations per mode (modes,)
        fixed_distance_map (itk.Image): Distance map of the target, in mm
        fixed_model (pv.DataSet): Target model, when one was supplied. Required
            for the symmetric (target-to-model) term.
        pca_number_of_modes (int): Number of PCA modes available
        registered_model_pca_coefficients (np.ndarray): Optimized PCA coefficients
        registered_model (pv.DataSet): Final registered and deformed model
        post_pca_transform (itk.Transform): Transform to apply after PCA registration
        forward_point_transform (itk.DisplacementFieldTransform): POINT transform
            mapping template points -> registered/target points; use it to warp
            the template model/landmarks onto the target. Its orientation is
            opposite to an image-registration forward_transform (see
            docs/developer/transform_conventions). Does not include the post-PCA
            transform.
        inverse_point_transform (itk.DisplacementFieldTransform): POINT transform
            mapping target points -> template points. Does not include the
            post-PCA transform.

    Example:
        >>> # Load PCA model data
        >>> pca_template_model = pv.read('pca_All_mean.vtk')
        >>> with open('pca.json', 'r') as f:
        ...     pca_data = json.load(f)
        >>> pca_group_data = pca_data['All']
        >>> pca_std_deviations = np.sqrt(np.array(pca_group_data['eigenvalues']))
        >>> pca_eigenvectors = np.array(pca_group_data['components'])
        >>>
        >>> # Initialize registrar with loaded data
        >>> registrar = RegisterModelsPCA(
        ...     pca_template_model=pca_template_model,
        ...     pca_eigenvectors=pca_eigenvectors,
        ...     pca_std_deviations=pca_std_deviations,
        ... )
        >>>
        >>> # Run full registration pipeline
        >>> result = registrar.register(pca_number_of_modes=10)
        >>>
        >>> # Save registered model
        >>> result['registered_model'].save('registered_heart.vtk')
        >>>
        >>> # Print optimization results
        >>> print(f'Final mean distance: {result["mean_distance"]:.2f}')
        >>> print(f'PCA coefficients: {result["pca_coefficients"]}')
    """

    def __init__(
        self,
        pca_template_model: pv.DataSet,
        pca_eigenvectors: np.ndarray,
        pca_std_deviations: np.ndarray,
        pca_number_of_modes: int = 0,
        pca_template_model_point_subsample: int = 4,
        post_pca_transform: Optional[itk.Transform] = None,
        fixed_distance_map: Optional[itk.Image] = None,
        fixed_model: Optional[pv.DataSet] = None,
        reference_image: Optional[itk.Image] = None,
        pca_prior_weight: float = 0.0,
        symmetric_weight: float = 0.5,
        log_level: int | str = logging.INFO,
    ):
        """Initialize the PCA-based model-to-image registration.

        Args:
            pca_template_model: PyVista model containing the mean 3D shape model
                (unstructured grid or polydata). It must be in the same frame
                the PCA modes were trained in; supply any alignment to the
                target as post_pca_transform instead of pre-applying it here.
            pca_eigenvectors: Numpy array of PCA eigenvectors/components. Shape: (modes, n_points*3)
                Each row is a flattened eigenmode with 3D displacements: [x1,y1,z1, x2,y2,z2, ...]
            pca_std_deviations: Numpy array of standard deviations per PCA mode. Shape: (modes,)
                These are the square roots of pca_eigenvalues
            pca_number_of_modes: Number of PCA modes to use. Default: 0 (use all)
            pca_template_model_point_subsample: Step size for subsampling model points. Default: 4
            post_pca_transform: Optional ITK transform to apply after PCA registration.
                Default: None
            fixed_distance_map: ITK image providing the distance map, in mm.
                Default: None
            fixed_model: PyVista model used to compute the distance map, if one isn't provided.
                Also supplies the target points for the symmetric term.
            reference_image: ITK image providing coordinate frame for computing the distance map.
            pca_prior_weight: Weight (in mm) of the Mahalanobis shape prior
                ``lambda * sum(b_i**2)``. Because b is expressed in standard
                deviations, this term is the squared Mahalanobis distance in
                shape space and makes the fit a MAP estimate rather than a pure
                data fit constrained only by the coefficient bounds.
                Default: 0.0 (prior disabled).
            symmetric_weight: Weight in [0, 1] of the target-to-model distance
                term. 0.0 measures model-to-target only, which lets the model
                satisfy the metric while covering just part of the target.
                Requires fixed_model; ignored with a warning when only a
                distance map is available. Default: 0.5
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING).
                Default: logging.INFO

        Raises:
            ValueError: If pca_eigenvector dimensions don't match model points,
                if the mode counts disagree, or if neither a distance map nor a
                fixed model plus reference image is provided.
        """
        # Initialize base class with logging
        super().__init__(class_name="RegisterModelsPCA", log_level=log_level)

        # Store model data
        self.pca_template_model: pv.DataSet = pca_template_model
        self.pca_eigenvectors: np.ndarray = np.asarray(
            pca_eigenvectors, dtype=np.float64
        )
        self.pca_std_deviations: np.ndarray = np.asarray(
            pca_std_deviations, dtype=np.float64
        )

        if self.pca_eigenvectors.ndim != 2:
            raise ValueError(
                f"pca_eigenvectors must be 2D (modes, n_points*3), got shape "
                f"{self.pca_eigenvectors.shape}"
            )
        expected_size = pca_template_model.n_points * 3
        if self.pca_eigenvectors.shape[1] != expected_size:
            raise ValueError(
                f"Component dimension mismatch: expected {expected_size} "
                f"(3 × {pca_template_model.n_points} points), got "
                f"{self.pca_eigenvectors.shape[1]}"
            )
        if self.pca_eigenvectors.shape[0] != self.pca_std_deviations.shape[0]:
            raise ValueError(
                f"Mode count mismatch: {self.pca_eigenvectors.shape[0]} eigenvectors "
                f"but {self.pca_std_deviations.shape[0]} standard deviations"
            )

        self.post_pca_transform = post_pca_transform

        self._contour_tools = ContourTools()

        self.fixed_model: Optional[pv.DataSet] = fixed_model
        self.fixed_distance_map = fixed_distance_map
        if (
            self.fixed_distance_map is None
            and fixed_model is not None
            and reference_image is not None
        ):
            self.fixed_distance_map = self._create_distance_map(
                fixed_model, reference_image
            )
        elif self.fixed_distance_map is not None and (
            fixed_model is not None or reference_image is not None
        ):
            self.log_warning(
                "A distance map was provided, so the reference image is ignored; "
                "the fixed model is retained only for the symmetric metric term."
            )
        elif self.fixed_distance_map is None and (
            fixed_model is None or reference_image is None
        ):
            self.log_error(
                "Fixed model and reference image must be provided if no distance map is provided."
            )
            raise ValueError(
                "Fixed model and reference image must be provided if no distance map is provided."
            )

        self.pca_number_of_modes: int = pca_number_of_modes
        if self.pca_number_of_modes <= 0:
            self.pca_number_of_modes = len(self.pca_std_deviations)

        self.pca_template_model_point_subsample = pca_template_model_point_subsample
        self.pca_prior_weight = pca_prior_weight
        if not 0.0 <= symmetric_weight <= 1.0:
            raise ValueError(
                f"symmetric_weight must be in [0, 1]; got {symmetric_weight}"
            )
        self.symmetric_weight = symmetric_weight

        # outputs
        self.registered_model_pca_coefficients: Optional[np.ndarray] = None
        self.registered_model: Optional[pv.DataSet] = None
        self.registered_model_mean_distance: float = 0.0
        self.registered_model_pca_deformation: Optional[np.ndarray] = None
        self.forward_point_transform: Optional[itk.DisplacementFieldTransform] = None
        self.inverse_point_transform: Optional[itk.DisplacementFieldTransform] = None

        # Sampling caches, built lazily by _prepare_sampling()
        self._sampling_ready: bool = False
        self._analytic_gradient: bool = True
        self._fixed_distance_map_max_distance: float = 0.0
        self._post_pca_affine_key: Optional[itk.Transform] = None
        self._post_pca_affine: Optional[tuple[np.ndarray, np.ndarray]] = None

        self._metric_call_count: int = 0

    def _create_distance_map(
        self, fixed_model: pv.DataSet, reference_image: itk.Image
    ) -> itk.Image:
        """Build the unsigned, un-normalized (mm) distance map of the target."""
        return self._contour_tools.create_distance_map(
            fixed_model,
            reference_image,
            squared_distance=False,
            negative_inside=False,
            zero_inside=True,
        )

    @classmethod
    def from_json(
        cls,
        pca_template_model: pv.DataSet,
        pca_json_filename: str,
        pca_number_of_modes: int = 0,
        pca_template_model_point_subsample: int = 4,
        post_pca_transform: Optional[itk.Transform] = None,
        fixed_distance_map: Optional[itk.Image] = None,
        fixed_model: Optional[pv.DataSet] = None,
        reference_image: Optional[itk.Image] = None,
        pca_prior_weight: float = 0.0,
        symmetric_weight: float = 0.5,
        log_level: int | str = logging.INFO,
    ) -> Self:
        """Create RegisterModelsPCA from PCA model JSON file.

        This method reads PCA statistical shape model data from a JSON file
        containing eigenvalues and principal component vectors.

        The JSON file must contain:
        - 'eigenvalues': Array of eigenvalues (variance) for each component
        - 'components': Array of principal component vectors (flattened shape deformations)

        Args:
            pca_template_model: Mean surface mesh to use as template
            pca_json_filename: Path to the PCA model JSON file
            pca_number_of_modes: Number of PCA modes to use. Default: 0 (use all)
            pca_template_model_point_subsample: Step size for subsampling model points. Default: 4
            post_pca_transform: Optional ITK transform to apply after PCA registration.
                Default: None
            fixed_distance_map: ITK image providing the distance values
                for registration. If None, must be set later before registration.
            fixed_model: Target surface mesh to register to. Default: None
            reference_image: Reference image defining coordinate space. Default: None
            pca_prior_weight: Weight (mm) of the Mahalanobis shape prior. Default: 0.0
            symmetric_weight: Weight of the target-to-model term. Default: 0.5
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING).
                Default: logging.INFO

        Returns:
            RegisterModelsPCA instance

        Raises:
            FileNotFoundError: If JSON file not found
            ValueError: If data format is invalid or required fields are missing

        Example:
            >>> registrar = RegisterModelsPCA.from_json(
            ...     pca_template_model=pca_template_model,
            ...     pca_json_filename='path/to/pca_model.json',
            ...     fixed_model=fixed_model,
            ...     reference_image=reference_image,
            ... )
        """
        # Create a logger for the classmethod since superclass hasn't
        # been initialized yet.
        logger = logging.getLogger("MONAIPhysio")

        json_path = Path(pca_json_filename)

        # Check if JSON file exists
        if not json_path.exists():
            logger.error(f"PCA JSON file not found: {pca_json_filename}")
            raise FileNotFoundError(f"PCA JSON file not found: {pca_json_filename}")

        logger.info("Loading PCA model data...")
        logger.info(f"  JSON file: {json_path}")

        # Load PCA data from JSON
        logger.info("Reading JSON file...")
        with open(json_path, encoding="utf-8") as f:
            pca_data = json.load(f)

        # Extract eigenvalues and convert to standard deviations
        if "eigenvalues" not in pca_data:
            raise ValueError("'eigenvalues' field not found in JSON data")
        pca_std_deviations = np.sqrt(np.array(pca_data["eigenvalues"]))
        logger.info("  Loaded %d eigenvalues", len(pca_std_deviations))

        # Extract principal component vectors
        if "components" not in pca_data:
            raise ValueError("'components' field not found in JSON data")
        pca_eigenvectors = np.array(pca_data["components"], dtype=np.float64)
        logger.info(f"  Loaded components with shape {pca_eigenvectors.shape}")

        # Validate dimensions
        expected_pca_eigenvector_size = pca_template_model.n_points * 3
        actual_pca_eigenvector_size = pca_eigenvectors.shape[1]
        if actual_pca_eigenvector_size != expected_pca_eigenvector_size:
            raise ValueError(
                f"Component dimension mismatch: "
                f"Expected {expected_pca_eigenvector_size} (3 × {pca_template_model.n_points} model points), "
                f"got {actual_pca_eigenvector_size}"
            )

        logger.info("  Data validation successful!")
        logger.info("PCA model data loaded successfully!")

        return cls.from_pca_model(
            pca_template_model=pca_template_model,
            pca_model=pca_data,
            pca_number_of_modes=pca_number_of_modes,
            pca_template_model_point_subsample=pca_template_model_point_subsample,
            post_pca_transform=post_pca_transform,
            fixed_distance_map=fixed_distance_map,
            fixed_model=fixed_model,
            reference_image=reference_image,
            pca_prior_weight=pca_prior_weight,
            symmetric_weight=symmetric_weight,
            log_level=log_level,
        )

    @classmethod
    def from_pca_model(
        cls,
        pca_template_model: pv.DataSet,
        pca_model: dict,
        pca_number_of_modes: int = 0,
        pca_template_model_point_subsample: int = 4,
        post_pca_transform: Optional[itk.Transform] = None,
        fixed_distance_map: Optional[itk.Image] = None,
        fixed_model: Optional[pv.DataSet] = None,
        reference_image: Optional[itk.Image] = None,
        pca_prior_weight: float = 0.0,
        symmetric_weight: float = 0.5,
        log_level: int | str = logging.INFO,
    ) -> Self:
        """Create RegisterModelsPCA from a PCA model dictionary.

        The dict must match the structure produced by
        :class:`WorkflowCreateStatisticalModel` (key ``pca_model``):
        ``explained_variance_ratio``, ``eigenvalues``, ``components``.

        Args:
            pca_template_model: Mean surface mesh to use as template
            pca_model: PCA model dict with 'eigenvalues' and 'components' (and optionally
                'explained_variance_ratio')
            pca_number_of_modes: Number of PCA modes to use. Default: 0 (use all)
            pca_template_model_point_subsample: Step size for subsampling model points. Default: 4
            post_pca_transform: Optional ITK transform to apply after PCA registration.
            fixed_distance_map: ITK image providing the distance values for registration.
            fixed_model: Target surface mesh to register to.
            reference_image: Reference image defining coordinate space.
            pca_prior_weight: Weight (mm) of the Mahalanobis shape prior. Default: 0.0
            symmetric_weight: Weight of the target-to-model term. Default: 0.5
            log_level: Logging level.

        Returns:
            RegisterModelsPCA instance

        Raises:
            ValueError: If required keys are missing or dimensions invalid
        """
        if "eigenvalues" not in pca_model:
            raise ValueError("'eigenvalues' field not found in pca_model")
        pca_std_deviations = np.sqrt(np.array(pca_model["eigenvalues"]))
        if "components" not in pca_model:
            raise ValueError("'components' field not found in pca_model")
        pca_eigenvectors = np.array(pca_model["components"], dtype=np.float64)
        return cls(
            pca_template_model=pca_template_model,
            pca_eigenvectors=pca_eigenvectors,
            pca_std_deviations=pca_std_deviations,
            pca_number_of_modes=pca_number_of_modes,
            pca_template_model_point_subsample=pca_template_model_point_subsample,
            post_pca_transform=post_pca_transform,
            fixed_distance_map=fixed_distance_map,
            fixed_model=fixed_model,
            reference_image=reference_image,
            pca_prior_weight=pca_prior_weight,
            symmetric_weight=symmetric_weight,
            log_level=log_level,
        )

    def set_fixed_model(
        self, fixed_model: pv.UnstructuredGrid, reference_image: Optional[itk.Image]
    ) -> None:
        """Set the fixed model for registration and rebuild its distance map.

        Args:
            fixed_model: PyVista model used to compute the distance map.
            reference_image: ITK image providing coordinate frame for computing the distance map.
        """
        if reference_image is None:
            raise ValueError(
                "reference_image must not be None when setting a fixed model"
            )

        self.fixed_model = fixed_model
        self.fixed_distance_map = self._create_distance_map(
            fixed_model, reference_image
        )
        self._sampling_ready = False

    def set_fixed_distance_map(self, fixed_distance_map: Optional[itk.Image]) -> None:
        """Set the distance map used as the registration target.

        Args:
            fixed_distance_map: ITK image providing distance data, in mm
        """
        self.fixed_distance_map = fixed_distance_map
        self._sampling_ready = False

    def set_pca_template_model(self, pca_template_model: pv.UnstructuredGrid) -> None:
        """Set the average model for registration.

        Args:
            pca_template_model: PyVista model containing the mean 3D shape model
                (unstructured grid or polydata)
        """
        self.pca_template_model = pca_template_model
        self._sampling_ready = False
        self.log_info("  Average model set successfully!")

    def _affine_of_transform(
        self, transform: itk.Transform
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Recover (matrix, offset) if ``transform`` acts affinely, else None.

        Probing the transform rather than querying ``GetMatrix()`` works for any
        ITK transform type and correctly rejects the non-affine ones (such as a
        displacement field), for which no constant Jacobian exists.
        """

        def apply(vector: np.ndarray) -> np.ndarray:
            point = itk.Point[itk.D, 3]()
            point[0], point[1], point[2] = (float(v) for v in vector)
            result = transform.TransformPoint(point)
            return np.array([result[0], result[1], result[2]], dtype=np.float64)

        # Probe inside the model's own extent: a displacement field evaluated
        # outside its grid returns no displacement, so probing at the origin and
        # the unit cube would report such a transform as the identity affine.
        bounds = np.asarray(self.pca_template_model.bounds, dtype=np.float64)
        low, high = bounds[0::2], bounds[1::2]
        center = 0.5 * (low + high)
        step = 0.25 * np.maximum(high - low, 1.0)

        base = apply(center)
        matrix = np.column_stack(
            [
                (apply(center + step[i] * basis) - base) / step[i]
                for i, basis in enumerate(np.eye(3, dtype=np.float64))
            ]
        )
        offset = base - matrix @ center
        probe = center + step * np.array([0.37, -0.61, 0.83], dtype=np.float64)
        scale = max(1.0, float(np.abs(base).max()), float(np.abs(matrix).max()))
        if not np.allclose(apply(probe), matrix @ probe + offset, atol=1e-9 * scale):
            return None
        return matrix, offset

    def _get_post_pca_affine(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return the cached (matrix, offset) of post_pca_transform, or None.

        Keyed on the transform object so that reassigning post_pca_transform
        invalidates the cache.
        """
        if self.post_pca_transform is None:
            return None
        if self._post_pca_affine_key is not self.post_pca_transform:
            self._post_pca_affine_key = self.post_pca_transform
            self._post_pca_affine = self._affine_of_transform(self.post_pca_transform)
        return self._post_pca_affine

    def _prepare_sampling(self) -> None:
        """Build the cached arrays the objective and its gradient are made of.

        Everything that does not depend on the PCA coefficients is computed once
        here: the subsampled template points, the per-mode displacement vectors
        already scaled by their standard deviation and mapped through the
        post-PCA transform, the distance map and its gradient, and the
        physical-to-index mapping used to sample them.
        """
        if self.fixed_distance_map is None:
            self.log_error("Distance map is not set.")
            raise ValueError("Distance map must be set before registering.")

        template_points = np.asarray(self.pca_template_model.points, dtype=np.float64)
        step = max(1, self.pca_template_model_point_subsample)
        self._sample_slice = slice(None, None, step)
        self._sample_points = template_points[self._sample_slice]

        # (modes, m, 3) displacement per unit coefficient, in template space.
        modes = self.pca_eigenvectors.reshape(self.pca_eigenvectors.shape[0], -1, 3)
        self._sample_modes = (
            modes[:, self._sample_slice, :] * self.pca_std_deviations[:, None, None]
        )

        # Fold the post-PCA transform into the mode directions so the gradient
        # is expressed directly in world space. A non-affine post-PCA transform
        # has no constant Jacobian, so the analytic gradient is disabled.
        affine = self._get_post_pca_affine()
        if self.post_pca_transform is not None and affine is None:
            self.log_warning(
                "post_pca_transform is not affine; falling back to a "
                "finite-difference gradient."
            )
        self._sample_modes_world = (
            self._sample_modes if affine is None else self._sample_modes @ affine[0].T
        )
        self._analytic_gradient = self.post_pca_transform is None or affine is not None

        # Physical point -> continuous index: index = affine_inv @ (p - origin).
        image = self.fixed_distance_map
        direction = itk.array_from_matrix(image.GetDirection())
        index_to_world = direction @ np.diag(np.asarray(image.GetSpacing()))
        self._index_to_world = index_to_world
        self._world_to_index = np.linalg.inv(index_to_world)
        self._image_origin = np.asarray(image.GetOrigin(), dtype=np.float64)
        size = image.GetLargestPossibleRegion().GetSize()
        self._image_size = np.array([size[0], size[1], size[2]], dtype=np.float64)

        # Array axes are (k, j, i), so index component a is array axis 2 - a.
        # The forward differences are the exact derivative of the trilinear
        # interpolant used to sample the map, which keeps the objective and its
        # gradient consistent; a central difference would not.
        self._distance_array = np.asarray(
            itk.array_view_from_image(image), dtype=np.float64
        )
        self._fixed_distance_map_max_distance = float(self._distance_array.max())
        self._distance_forward_diff = tuple(
            np.diff(self._distance_array, axis=2 - axis)
            if self._distance_array.shape[2 - axis] > 1
            else None
            for axis in range(3)
        )

        # Target points for the symmetric term.
        self._target_points: Optional[np.ndarray] = None
        if self.symmetric_weight > 0.0:
            if self.fixed_model is None:
                self.log_warning(
                    "symmetric_weight is %.3g but no fixed_model is available; "
                    "the target-to-model term is disabled.",
                    self.symmetric_weight,
                )
            else:
                self._target_points = np.asarray(
                    self.fixed_model.points, dtype=np.float64
                )[self._sample_slice]

        self._sampling_ready = True
        self.log_debug(
            "Sampling prepared: %d model points, %s target points, max distance %.3f mm",
            self._sample_points.shape[0],
            "no" if self._target_points is None else str(len(self._target_points)),
            self._fixed_distance_map_max_distance,
        )

    def _sample_distance(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Sample the distance map and its spatial gradient at world points.

        Points outside the image are clamped to the grid and charged the extra
        travel from the clamped location, which keeps the metric continuous and
        gives the optimizer a gradient that pushes such points back inside --
        unlike a constant out-of-bounds penalty, which is flat.

        Args:
            points: (n, 3) array of world-space points

        Returns:
            Tuple of (distances (n,), world-space gradients (n, 3), n_outside)
        """
        index = (points - self._image_origin) @ self._world_to_index.T
        clamped = np.clip(index, 0.0, self._image_size - 1.0)
        outside = index - clamped
        is_outside = np.any(outside != 0.0, axis=1)
        n_outside = int(np.count_nonzero(is_outside))

        # map_coordinates indexes the array as (k, j, i).
        array_coordinates = clamped[:, ::-1]
        distances = map_coordinates(
            self._distance_array, array_coordinates.T, order=1, mode="nearest"
        )

        # d/dc_a of the trilinear interpolant is the forward difference along a,
        # interpolated linearly across the other two axes within the same cell.
        gradient_index = np.zeros_like(clamped)
        for axis in range(3):
            differences = self._distance_forward_diff[axis]
            if differences is None:
                continue
            coordinates = array_coordinates.copy()
            coordinates[:, 2 - axis] = np.clip(
                np.floor(clamped[:, axis]), 0.0, self._image_size[axis] - 2.0
            )
            gradient_index[:, axis] = map_coordinates(
                differences, coordinates.T, order=1, mode="nearest"
            )
        # A clamped axis cannot change the sampled value, so it carries no
        # gradient from the map; the out-of-bounds term supplies it instead.
        gradient_index[outside != 0.0] = 0.0
        gradients = gradient_index @ self._world_to_index

        if n_outside:
            outside_world = outside @ self._index_to_world.T
            outside_distance = np.linalg.norm(outside_world, axis=1)
            safe = np.where(outside_distance > 0.0, outside_distance, 1.0)
            distances = distances + outside_distance
            gradients = gradients + outside_world / safe[:, None]

        return distances, gradients, n_outside

    def _deform(self, pca_coefficients: np.ndarray) -> np.ndarray:
        """Deform the subsampled template points into world space."""
        n_modes = len(pca_coefficients)
        points = self._sample_points + np.tensordot(
            pca_coefficients, self._sample_modes[:n_modes], axes=(0, 0)
        )
        return self._apply_post_pca_transform(points)

    def _objective_and_gradient(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate the registration objective and its gradient.

        The objective is MINIMIZED: the distance map is zero on the target
        surface and grows away from it.

        Args:
            params: PCA coefficients b, in units of standard deviations

        Returns:
            Tuple of (objective value in mm, gradient with respect to params)
        """
        if not self._sampling_ready:
            self._prepare_sampling()

        n_modes = len(params)
        modes_world = self._sample_modes_world[:n_modes]
        points = self._deform(params)

        distances, gradients, n_outside = self._sample_distance(points)
        forward_distance = float(distances.mean())
        # d/db_j of mean_i D(p_i) = mean_i grad_D(p_i) . (sigma_j * v_ij)
        forward_gradient = np.einsum("ia,jia->j", gradients, modes_world) / len(points)

        weight = self.symmetric_weight if self._target_points is not None else 0.0
        reverse_distance = 0.0
        reverse_gradient = np.zeros(n_modes, dtype=np.float64)
        if weight > 0.0:
            assert self._target_points is not None, "target points must be set"
            # Target -> model: each target point is charged its distance to the
            # nearest deformed model point, so a model that covers only part of
            # the target scores badly. The forward term alone cannot see this.
            nearest, nearest_index = cKDTree(points).query(self._target_points)
            nearest_distance = np.atleast_1d(np.asarray(nearest, dtype=np.float64))
            reverse_distance = float(nearest_distance.mean())
            safe = np.where(nearest_distance > 0.0, nearest_distance, 1.0)
            direction = (points[nearest_index] - self._target_points) / safe[:, None]
            # Accumulate each target's pull onto the model point it selected,
            # then contract once against the modes.
            pull = np.zeros_like(points)
            np.add.at(pull, nearest_index, direction)
            pull /= len(self._target_points)
            reverse_gradient = np.einsum("ia,jia->j", pull, modes_world)

        objective = (1.0 - weight) * forward_distance + weight * reverse_distance
        gradient = (1.0 - weight) * forward_gradient + weight * reverse_gradient

        prior = 0.0
        if self.pca_prior_weight > 0.0:
            prior = self.pca_prior_weight * float(np.dot(params, params))
            objective += prior
            gradient = gradient + 2.0 * self.pca_prior_weight * params

        if n_outside > 0.25 * len(points):
            self.log_warning(
                "%d of %d model points mapped outside the distance map.",
                n_outside,
                len(points),
            )

        if self.log_level <= logging.DEBUG or self._metric_call_count % 25 == 0:
            self.log_info(
                "   Metric %d: %.4f mm (model->target %.4f, target->model %.4f, "
                "prior %.4f, outside %d)",
                self._metric_call_count + 1,
                objective,
                forward_distance,
                reverse_distance,
                prior,
                n_outside,
            )
            self.log_debug("       Params %s", params)
        self._metric_call_count += 1

        return objective, gradient

    def _mean_distance_metric(self, params: np.ndarray) -> float:
        """Evaluate the registration objective at the given PCA coefficients.

        Args:
            params: PCA coefficients b, in units of standard deviations

        Returns:
            Objective value, in mm. Lower is better.
        """
        return self._objective_and_gradient(np.asarray(params, dtype=np.float64))[0]

    def _apply_post_pca_transform(self, points: np.ndarray) -> np.ndarray:
        """Apply post_pca_transform to an (n, 3) array of world points."""
        affine = self._get_post_pca_affine()
        if affine is not None:
            return np.asarray(points @ affine[0].T + affine[1], dtype=np.float64)
        if self.post_pca_transform is None:
            return points
        transformed = np.empty_like(points)
        point = itk.Point[itk.D, 3]()
        for i, source in enumerate(points):
            point[0], point[1], point[2] = (float(v) for v in source)
            result = self.post_pca_transform.TransformPoint(point)
            transformed[i] = (result[0], result[1], result[2])
        return transformed

    def _compute_pca_deformation(self, pca_coefficients: np.ndarray) -> np.ndarray:
        """Compute PCA deformation vectors for all points.

        Deformation is computed as:
            displacement = Σ(b_i * std_i * pca_eigenvector_i)

        Args:
            pca_coefficients: Array of PCA coefficients b_i. Only as many modes
                as there are coefficients contribute.

        Returns:
            Nx3 array of deformation vectors (displacement from the template)
        """
        n_modes = len(pca_coefficients)
        scaled = pca_coefficients * self.pca_std_deviations[:n_modes]
        deformation = np.asarray(
            scaled @ self.pca_eigenvectors[:n_modes], dtype=np.float64
        )
        return deformation.reshape(-1, 3)

    def _optimize_pca_coefficients(
        self,
        pca_number_of_modes: int = 0,
        pca_coefficient_bounds: float = 3.0,
        method: str = "L-BFGS-B",
        max_iterations: int = 50,
    ) -> tuple[np.ndarray, float]:
        """Optimize PCA coefficients

        Minimizes the mean distance between the deformed model and the target,
        supplying the analytic gradient of the objective to the optimizer.

        Args:
            pca_number_of_modes: Number of PCA modes to use in optimization. Using fewer
                modes provides smoother deformations. Default: 0 (use all)
            pca_coefficient_bounds: Bound on PCA coefficients in units of std deviations.
                Default: 3.0 (±3 std deviations per mode)
            method: Optimization method for scipy.optimize.minimize.
                Default: 'L-BFGS-B' (supports bounds)
            max_iterations: Maximum number of optimization iterations.
                Default: 50

        Returns:
            Tuple of (pca_coefficients, mean_distance):
                - pca_coefficients: Optimized PCA coefficients
                - mean_distance: Final objective value, in mm

        Raises:
            ValueError: If number of PCA modes to use exceeds available modes
        """
        n_available = len(self.pca_eigenvectors)
        if pca_number_of_modes <= 0:
            pca_number_of_modes = n_available
        if pca_number_of_modes > n_available:
            raise ValueError(
                f"Number of PCA modes to use ({pca_number_of_modes}) exceeds "
                f"available modes ({n_available})"
            )
        self.pca_number_of_modes = pca_number_of_modes

        self._prepare_sampling()

        self.log_info(f"Number of PCA modes: {pca_number_of_modes}")
        self.log_info(
            f"PCA coefficient bounds: ±{pca_coefficient_bounds} std deviations"
        )
        self.log_info(f"Optimization method: {method}")
        self.log_info(f"Max iterations: {max_iterations}")
        self.log_info(f"Shape prior weight: {self.pca_prior_weight}")
        self.log_info(f"Symmetric weight: {self.symmetric_weight}")

        bounds = [
            (-pca_coefficient_bounds, pca_coefficient_bounds)
            for _ in range(pca_number_of_modes)
        ]

        disp = self.log_level <= logging.INFO

        # The metric is in mm, so the default gradient tolerance is meaningful.
        # Without an analytic gradient the finite-difference step must be large
        # enough to move sample points by a useful fraction of a voxel.
        options: dict = {"maxiter": max_iterations, "disp": disp, "gtol": 1e-6}
        if not self._analytic_gradient:
            options["eps"] = 1e-2

        self.log_info("Running optimization...")
        result_pca = minimize(  # type: ignore[call-overload]
            self._objective_and_gradient
            if self._analytic_gradient
            else self._mean_distance_metric,
            np.zeros(pca_number_of_modes),
            method=method,
            jac=self._analytic_gradient,
            bounds=bounds,
            options=options,
        )

        optimized_pca_coefficients = result_pca.x
        optimized_mean_distance = float(result_pca.fun)

        self.log_info("Optimization completed!")
        self.log_info(f"Optimized PCA coefficients: {optimized_pca_coefficients}")
        self.log_info(f"Metric evaluations: {self._metric_call_count}")
        self.log_info(f"Final mean distance: {optimized_mean_distance:.4f} mm")

        return optimized_pca_coefficients, optimized_mean_distance

    def transform_template_model(self) -> pv.DataSet:
        """Create the final registered model by applying PCA deformation.

        Returns:
            Final registered and deformed model as a PyVista dataset.

        Raises:
            ValueError: If registration has not been performed
        """
        if self.registered_model_pca_coefficients is None:
            self.log_error("PCA coefficients are not set.")
            raise ValueError(
                "PCA coefficients must be set before creating registered model"
            )

        self.log_info("Creating final registered model...")

        # Compute PCA deformation
        if self.registered_model_pca_deformation is None:
            self.registered_model_pca_deformation = self._compute_pca_deformation(
                self.registered_model_pca_coefficients,
            )

        # Deform in the template frame, then map into target space.
        deformed_points = (
            np.asarray(self.pca_template_model.points, dtype=np.float64)
            + self.registered_model_pca_deformation
        )
        final_points = self._apply_post_pca_transform(deformed_points)

        # Create new model with transformed points
        self.registered_model = self.pca_template_model.copy(deep=True)
        self.registered_model.points = final_points.copy()

        self.log_info(
            f"Registered model created with {self.registered_model.n_points} points"
        )

        return self.registered_model

    def transform_point(
        self,
        point: itk.Point,
        include_post_pca_transform: bool = True,
    ) -> itk.Point:
        """Transform an arbitrary point through the PCA deformation field.

        Args:
            point: ITK point to transform (itk.Point[itk.D, 3])
            include_post_pca_transform: Also apply post_pca_transform. Default: True

        Returns:
            Transformed ITK point

        Raises:
            ValueError: If compute_pca_transforms() has not been called yet

        Notes:
            This samples the *approximated* deformation field built by
            compute_pca_transforms(), which is splatted and blurred, so it does
            not reproduce transform_template_model() exactly; the RMS of that
            difference is logged when the field is built. Points outside the
            field's reference image are not displaced.

        Example:
            >>> p = itk.Point[itk.D, 3]()
            >>> p[0], p[1], p[2] = 10.0, 20.0, 30.0
            >>> transformed_p = registrar.transform_point(p)
        """
        if self.forward_point_transform is None:
            self.log_error("Forward point transform is not set.")
            raise ValueError(
                "compute_pca_transforms() must be called before transform_point()"
            )
        transformed_point = self.forward_point_transform.TransformPoint(point)

        if include_post_pca_transform and self.post_pca_transform is not None:
            transformed_point = self.post_pca_transform.TransformPoint(
                transformed_point
            )

        return transformed_point

    def compute_pca_transforms(
        self, reference_image: itk.Image, blur_sigma: float = 2.5
    ) -> dict:
        """Compute PCA transforms.

        The field is built by splatting the per-point PCA displacements onto the
        reference grid and blurring them, so it only approximates the exact
        per-point deformation. The RMS of that approximation error, and of the
        forward/inverse round trip, are both logged.

        Args:
            reference_image: ITK image providing the coordinate frame for the field.
            blur_sigma: Sigma for Gaussian blurring of the deformation field.
                Default: 2.5

        Returns:
            Dictionary containing:
                - 'forward_point_transform': POINT transform mapping template
                  points -> target points (warps the template onto the target)
                - 'inverse_point_transform': POINT transform mapping target
                  points -> template points

        Note:
            These are point transforms, oriented opposite to image-registration
            transforms; see docs/developer/transform_conventions. Neither
            includes post_pca_transform.
        """
        assert self.registered_model_pca_deformation is not None, (
            "PCA deformation must be computed"
        )
        template_points = np.asarray(self.pca_template_model.points, dtype=np.float64)
        template_model_pca_deformation_field_image = (
            self._contour_tools.create_deformation_field(
                template_points,
                self.registered_model_pca_deformation,
                reference_image=reference_image,
                blur_sigma=blur_sigma,
                ptype=itk.D,
            )
        )

        self.forward_point_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
        self.forward_point_transform.SetDisplacementField(
            template_model_pca_deformation_field_image
        )

        transform_tools = TransformTools()
        self.inverse_point_transform = (
            transform_tools.invert_displacement_field_transform(
                self.forward_point_transform
            )
        )

        self._log_transform_fidelity(template_points)

        return {
            "forward_point_transform": self.forward_point_transform,
            "inverse_point_transform": self.inverse_point_transform,
        }

    def _log_transform_fidelity(self, template_points: np.ndarray) -> None:
        """Report how well the field reproduces the deformation and inverts."""
        assert self.forward_point_transform is not None, "forward transform must be set"
        assert self.inverse_point_transform is not None, "inverse transform must be set"
        assert self.registered_model_pca_deformation is not None, (
            "PCA deformation must be computed"
        )

        # Two TransformPoint calls per point is costly on dense templates, and a
        # strided subset reports the same RMS to within sampling noise.
        stride = max(1, len(template_points) // 5000)
        sampled = template_points[::stride]
        deformation = self.registered_model_pca_deformation[::stride]

        point = itk.Point[itk.D, 3]()
        forward = np.empty_like(sampled)
        round_trip = np.empty_like(sampled)
        for i, source in enumerate(sampled):
            point[0], point[1], point[2] = (float(v) for v in source)
            mapped = self.forward_point_transform.TransformPoint(point)
            forward[i] = (mapped[0], mapped[1], mapped[2])
            back = self.inverse_point_transform.TransformPoint(mapped)
            round_trip[i] = (back[0], back[1], back[2])

        expected = sampled + deformation
        field_rms = float(np.sqrt(np.mean(np.sum((forward - expected) ** 2, axis=1))))
        inverse_rms = float(
            np.sqrt(np.mean(np.sum((round_trip - sampled) ** 2, axis=1)))
        )
        self.log_info(
            "Deformation field RMS error: %.4f mm (approximation of the "
            "per-point deformation)",
            field_rms,
        )
        self.log_info("Forward/inverse round-trip RMS error: %.4f mm", inverse_rms)

    def register(
        self,
        pca_number_of_modes: int = 0,
        pca_coefficient_bounds: float = 3.5,
        method: str = "L-BFGS-B",
        max_iterations: int = 100,
    ) -> dict:
        """Optimize PCA coefficients to deform the model onto the target.

        Args:
            pca_number_of_modes: Number of PCA modes to use. Default: 0 (use all available modes)
            pca_coefficient_bounds: PCA coefficient bounds (±std devs). Default: 3.5
            method: Optimization method for scipy.optimize.minimize.
                Default: 'L-BFGS-B' (supports bounds)
            max_iterations: Maximum number of optimization iterations.
                Default: 100

        Returns:
            Dictionary containing:
                - 'registered_model': Final registered PyVista model
                - 'pca_coefficients': Optimized PCA coefficients
                - 'mean_distance': Final objective value, in mm

        Raises:
            ValueError: If the distance map is not set

        Example:
            >>> result = registrar.register(pca_number_of_modes=10)
            >>> result['registered_model'].save('registered_heart.vtk')
        """
        if self.fixed_distance_map is None:
            raise ValueError("A distance map must be set before registration")

        if pca_number_of_modes <= 0:
            pca_number_of_modes = self.pca_number_of_modes

        self.log_section("PCA-BASED MODEL-TO-MODEL REGISTRATION", width=70)
        self.log_info(f"Number of points: {self.pca_template_model.n_points}")
        self.log_info(f"Modes to use: {pca_number_of_modes}")

        self._metric_call_count = 0
        self.registered_model_pca_coefficients, self.registered_model_mean_distance = (
            self._optimize_pca_coefficients(
                pca_number_of_modes=pca_number_of_modes,
                pca_coefficient_bounds=pca_coefficient_bounds,
                method=method,
                max_iterations=max_iterations,
            )
        )

        # Create final registered model
        self.registered_model_pca_deformation = None
        self.registered_model = self.transform_template_model()

        # Return results as dictionary
        return {
            "registered_model": self.registered_model,
            "pca_coefficients": self.registered_model_pca_coefficients,
            "mean_distance": self.registered_model_mean_distance,
        }
