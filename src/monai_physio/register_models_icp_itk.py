import logging
from typing import Optional

import itk
import numpy as np
import pyvista as pv
from scipy.optimize import minimize

from .contour_tools import ContourTools
from .monai_physio_base import MONAIPhysioBase
from .transform_tools import TransformTools


class RegisterModelsICPITK(MONAIPhysioBase):
    """Register shape models using model to distance map minimization.

    **Optimization Objective:**
        Minimize the mean distance of the distance map sampled at model points using
        ITK's LinearInterpolateImageFunction. This aligns the model with bright
        regions in target image.

    Attributes:
        fixed_model (pv.PolyData)
        moving_model (pv.PolyData)
        reference_image (itk.Image): Patient image providing coordinate frame and
            distance data
        transform_type: Rigid or Affine
        moving_to_fixed_transform (itk.ComposeScaleSkewVersor3DTransform): Optimized
            transformation
        fixed_to_moving_transform (itk.ComposeScaleSkewVersor3DTransform): Optimized
            transformation
        registered_model (pv.PolyData): Final registered model

    Note:
        The fixed_model and moving_model are typically extracted from VTU models
        using model.extract_surface(algorithm="dataset_surface") before passing to this class.
    """

    def __init__(
        self,
        fixed_model: pv.PolyData,
        reference_image: Optional[itk.Image] = None,
        point_subsample_step: int = 4,
        log_level: int | str = logging.INFO,
    ):
        """Initialize the ICP-ITK model registration.

        Args:
            fixed_model: Target model (surface mesh) to register to
            reference_image: Optional patient image providing coordinate frame and
                distance data
            point_subsample_step: Step size for subsampling model points during
                registration (default: 4)
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING)
                (default: logging.INFO)
        """
        # Initialize base class with logging
        super().__init__(class_name="RegisterModelsICPITK", log_level=log_level)

        # Store model data
        self.fixed_model: pv.PolyData = fixed_model
        self.reference_image = reference_image

        self.moving_model: Optional[pv.PolyData] = None

        # Working transform (reused to avoid repeated memory allocation)
        self._working_transform: itk.ComposeScaleSkewVersor3DTransform[itk.D] = (
            itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        )

        self.transform_type: str = "Affine"

        # outputs
        self.moving_to_fixed_transform: Optional[
            itk.ComposeScaleSkewVersor3DTransform
        ] = None
        self.fixed_to_moving_transform: Optional[
            itk.ComposeScaleSkewVersor3DTransform
        ] = None
        self.registered_model: Optional[pv.PolyData] = None
        self.final_mean_distance = 0

        # Transform utilities
        self._contour_tools = ContourTools()
        self._transform_tools = TransformTools()

        # Image interpolator (created when needed)
        self.fixed_distance_map: Optional[itk.Image] = None
        self._interpolator: Optional[itk.LinearInterpolateImageFunction] = None
        self._max_distance: float = 0.0

        self._metric_call_count: int = 0

        # Pre-convert mean shape points to ITK format
        self.point_subsample_step = point_subsample_step
        self._moving_model_points_itk: Optional[list[itk.Point]] = None

    def _create_itk_points(self) -> None:
        """Pre-convert mean shape points to ITK Point format for efficiency.

        This method creates ITK Point objects once at initialization, avoiding
        repeated conversions during optimization iterations.
        """
        self.log_info("Converting mean shape points to ITK format...")

        assert self.moving_model is not None, (
            "moving_model must be set before creating ITK points"
        )
        self._moving_model_points_itk = []
        for point in self.moving_model.points:
            itk_point = itk.Point[itk.D, 3]()
            itk_point[0] = float(point[0])
            itk_point[1] = float(point[1])
            itk_point[2] = float(point[2])
            self._moving_model_points_itk.append(itk_point)

        self.log_info(
            f"  Converted {len(self._moving_model_points_itk)} points to ITK format"
        )

    def set_reference_image(self, reference_image: itk.Image) -> None:
        """Set the reference image for registration.

        Args:
            reference_image: ITK image providing coordinate frame and distance data
        """
        self.reference_image = reference_image
        # Clear interpolator to force recreation with new image
        self._interpolator = None
        self.fixed_distance_map = None

    def set_fixed_model(self, fixed_model: pv.PolyData) -> None:
        """Set the average model for registration.

        Args:
            fixed_model: PyVista model containing the mean 3D shape model
                (unstructured grid or polydata)
        """
        self.fixed_model = fixed_model
        self.fixed_distance_map = None
        self._interpolator = None

        self.log_info("  Fixed model set successfully!")

    def _evaluate_distance_metric(
        self,
        transform_params: np.ndarray,
    ) -> float:
        """Evaluate the optimization metric (mean distance) at model points.

        This is the objective function to be minimized during optimization.
        Higher values indicate better alignment with bright regions.

        Args:
            pca_deformation: Nx3 numpy array of PCA deformation vectors to add to
                model points.
                If None, no deformation is applied.
            transform_params: 12-element array of affine transform parameters.
                If None, no affine transformation is applied.

        Returns:
            Mean distance value across all points
        """
        if self._interpolator is None:
            if self.fixed_distance_map is None:
                self.fixed_distance_map = self._contour_tools.create_distance_map(
                    self.fixed_model,
                    self.reference_image,
                )
                self.log_debug("   Distance map created")
            ImageType = type(self.fixed_distance_map)
            self._interpolator = itk.LinearInterpolateImageFunction[
                ImageType, itk.D
            ].New()
            self._interpolator.SetInputImage(self.fixed_distance_map)
            fixed_distance_map_array = itk.GetArrayFromImage(self.fixed_distance_map)
            self._max_distance = fixed_distance_map_array.max()

            self.log_debug("   Interpolator created")

        if self._moving_model_points_itk is None:
            self._create_itk_points()

        # Update working transform if parameters provided
        if self.transform_type == "Rigid":
            itk_params = itk.OptimizerParameters[itk.D](12)
            for i in range(6):
                itk_params[i] = transform_params[i]
            for i in range(6, 9):
                itk_params[i] = 1
            for i in range(9, 12):
                itk_params[i] = 0
            self._working_transform.SetParameters(itk_params)
        else:
            itk_params = itk.OptimizerParameters[itk.D](12)
            for i in range(12):
                itk_params[i] = transform_params[i]
            self._working_transform.SetParameters(itk_params)

        # Sample intensities at each point
        n_valid_points = 0
        n_invalid_points = 0
        total_distance = 0.0
        center = np.zeros(3)
        point = itk.Point[itk.D, 3]()
        assert self.reference_image is not None, "reference_image must be set"
        assert self._moving_model_points_itk is not None, (
            "ITK points must be initialized"
        )
        image_size = self.reference_image.GetBufferedRegion().GetSize()
        for i, base_point in enumerate(self._moving_model_points_itk):
            if i % self.point_subsample_step != 0:
                continue

            point[0] = base_point[0]
            point[1] = base_point[1]
            point[2] = base_point[2]

            point = self._working_transform.TransformPoint(point)

            # Check if point is inside image bounds
            coord_index = self.reference_image.TransformPhysicalPointToContinuousIndex(
                point
            )
            if (
                0 <= coord_index[0] < image_size[0]
                and 0 <= coord_index[1] < image_size[1]
                and 0 <= coord_index[2] < image_size[2]
            ):
                center[0] += point[0]
                center[1] += point[1]
                center[2] += point[2]
                distance = self._interpolator.EvaluateAtContinuousIndex(coord_index)
                total_distance += distance
                n_valid_points += 1
            else:
                self.log_warning("   Point %d is outside image bounds (%s)", i, point)
                return self._max_distance

        if n_valid_points > n_invalid_points:
            mean_distance = total_distance / n_valid_points
            center /= n_valid_points
        else:
            mean_distance = 0.0
            self.log_warning("   *** No valid points found")

        if n_invalid_points > 0:
            self.log_warning("   %d points are outside image bounds", n_invalid_points)
            self.log_warning("   Parameters: %s", transform_params)
            if n_valid_points > n_invalid_points:
                self.log_warning("   Center: %s", center)
                self.log_warning("   Mean distance: %f", mean_distance)

        log_level_int = (
            self.log_level
            if isinstance(self.log_level, int)
            else logging.getLevelName(self.log_level)
        )
        if log_level_int <= logging.DEBUG or self._metric_call_count % 100 == 0:
            self.log_info(
                "   Metric %d: %s -> %f",
                (self._metric_call_count + 1),
                center,
                mean_distance,
            )
        self._metric_call_count += 1

        return mean_distance

    def register(
        self,
        moving_model: pv.PolyData,
        initial_transform: Optional[itk.MatrixOffsetTransformBase] = None,
        transform_type: str = "Affine",  # or 'Rigid'
        method: str = "L-BFGS-B",  # or 'Nelder-Mead'
        scale_bound: float = 0.20,
        skew_bound: float = 0.03,
        versor_bound: float = 0.15,
        translation_bound: float = 15,
        max_iterations: int = 500,
    ) -> dict:
        """Optimize affine alignment to minimize mean distance.

        to align the mean shape model with bright regions in the image.

        Args:
            initial_transform: Initial ITK ComposeScaleSkewVersor3DTransform for starting point
            method: Optimization method for scipy.optimize.minimize.
                Default: 'Nelder-Mead'
            max_iterations: Maximum number of optimization iterations.
                Default: 500

        Returns:
            Tuple of (transform, mean_distance):
                - transform: Optimized ITK ComposeScaleSkewVersor3DTransform
                - mean_distance: Final mean distance metric value

        Raises:
            ValueError: If reference image is not set
        """
        if self.reference_image is None:
            raise ValueError("Reference image must be set before optimization")

        self.log_section("Affine Alignment Optimization", width=60)

        self.moving_model = moving_model

        self.transform_type = transform_type

        # Get initial parameters from transform
        initial_params = None
        if initial_transform is not None:
            self.log_info("Using initial transform...")
            self._working_transform.SetIdentity()
            self._working_transform.SetMatrix(initial_transform.GetMatrix())
            self._working_transform.SetOffset(initial_transform.GetOffset())
            self._working_transform.SetCenter(initial_transform.GetCenter())
        else:
            self.log_info(
                "No initial transform provided, performing centroid alignment..."
            )
            moving_centroid = np.array(self.moving_model.center)
            self.log_debug("Moving model centroid: %s", moving_centroid)
            fixed_centroid = np.array(self.fixed_model.center)
            self.log_debug("Fixed model centroid: %s", fixed_centroid)
            translation = fixed_centroid - moving_centroid
            self._working_transform.SetIdentity()
            self._working_transform.SetOffset(translation)
            self._working_transform.SetCenter(moving_centroid)

        if self.transform_type == "Rigid":
            initial_params = [
                self._working_transform.GetParameters()[i] for i in range(6)
            ]
        elif self.transform_type == "Affine":
            initial_params = [
                self._working_transform.GetParameters()[i] for i in range(12)
            ]
        else:
            self.log_error("Invalid transform type: %s", self.transform_type)
            raise ValueError(f"Invalid transform type: {self.transform_type}")
        self.log_info("Initial parameters: %s", initial_params)

        bounds = []
        # Scale, Skew, Versor rotation bounds
        for v_affine in initial_params[:3]:
            bounds.append((v_affine - versor_bound, v_affine + versor_bound))
        for trans_affine in initial_params[3:6]:
            bounds.append(
                (
                    trans_affine - translation_bound,
                    trans_affine + translation_bound,
                )
            )
        if self.transform_type == "Affine":
            for s_affine in initial_params[6:9]:
                bounds.append((s_affine - scale_bound, s_affine + scale_bound))
            for k_affine in initial_params[9:12]:
                bounds.append((k_affine - skew_bound, k_affine + skew_bound))

        # Run optimization
        self.log_info("Running optimization...")
        log_level_int = (
            self.log_level
            if isinstance(self.log_level, int)
            else logging.getLevelName(self.log_level)
        )
        if log_level_int <= logging.INFO:
            disp = True
        else:
            disp = False

        result_affine = minimize(  # type: ignore[call-overload]
            self._evaluate_distance_metric,
            initial_params,
            method=method,
            bounds=bounds,
            options={"maxiter": max_iterations, "disp": disp},
        )
        self.log_info(
            "Optimization result: %s -> %f", result_affine.x, result_affine.fun
        )

        # Create optimized transform
        self.moving_to_fixed_transform = itk.ComposeScaleSkewVersor3DTransform[
            itk.D
        ].New()
        opt_itk_params = itk.OptimizerParameters[itk.D](12)
        if self.transform_type == "Rigid":
            for i in range(6):
                opt_itk_params[i] = result_affine.x[i]
            for i in range(6, 9):
                opt_itk_params[i] = 1
            for i in range(9, 12):
                opt_itk_params[i] = 0
        elif self.transform_type == "Affine":
            for i in range(12):
                opt_itk_params[i] = result_affine.x[i]
        self.moving_to_fixed_transform.SetParameters(opt_itk_params)

        self.fixed_to_moving_transform = (
            self.moving_to_fixed_transform.GetInverseTransform()
        )

        self.final_mean_distance = result_affine.fun

        self.registered_model = self._transform_tools.transform_pvcontour(
            self.moving_model,
            self.moving_to_fixed_transform,
            with_deformation_magnitude=False,
        )

        self.log_info("Optimization completed!")
        self.log_info(f"Final parameters: {result_affine.x}")
        self.log_info(f"Final mean distance: {self.final_mean_distance:.2f}")

        return {
            "registered_model": self.registered_model,
            "moving_to_fixed_transform": self.moving_to_fixed_transform,
            "fixed_to_moving_transform": self.fixed_to_moving_transform,
            "mean_distance": self.final_mean_distance,
        }
