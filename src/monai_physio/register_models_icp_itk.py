"""ICP-based model-to-model registration on ITK's v4 point-set framework.

:class:`RegisterModelsICPITK` runs the same pipeline as
:class:`RegisterModelsICP` - centroid alignment, bounding-box scaling, then
rigid, similarity or affine ICP stages - and returns the same results, but each
ICP stage minimizes ITK's ``EuclideanDistancePointSetToPointSetMetricv4`` with
a gradient-descent optimizer instead of running VTK's ICP. That lets a stage
hold chosen transform parameters fixed, which VTK's ICP cannot: with
``rotation_axis`` set, every stage rotates only about that world axis.

Example:
    >>> registrar = RegisterModelsICPITK(fixed_model=patient_surface, rotation_axis=2)
    >>> result = registrar.register(
    ...     moving_model=model_surface,
    ...     transform_type="Affine",
    ... )
    >>> aligned_model = result["registered_model"]
"""

import logging
from typing import Optional

import itk
import numpy as np
import pyvista as pv

from .register_models_icp import RegisterModelsICP

_POINT_SET_TYPE = itk.PointSet[itk.D, 3]


class RegisterModelsICPITK(RegisterModelsICP):
    """Register surface models by ICP, optionally rotating about one axis only.

    The pipelines, transform conventions and results are those of
    :class:`RegisterModelsICP`; only the ICP stages differ. Each stage optimizes
    an ITK transform whose parameters split rotation from the other degrees of
    freedom, so rotation can be restricted per axis:

        - **Rigid**: ``VersorRigid3DTransform``
        - **Similarity**: ``Similarity3DTransform``
        - **Affine**: ``ComposeScaleSkewVersor3DTransform``

    With ``rotation_axis`` set, the versor components about the other two axes
    are held at zero, and the affine stage's skews that couple ``rotation_axis``
    to another axis are held at zero too, since such a skew tilts that axis just
    as a rotation would. Scale and translation stay free along every axis.

    Each stage starts from the previous stage's result, rotates and scales about
    the moving model's current centroid, and pairs every sampled moving point
    with its closest fixed point, as ICP does.

    Attributes:
        rotation_axis (Optional[int]): World axis (0, 1 or 2) the stages may
            rotate about, or None for unrestricted rotation.
        number_of_points (int): Moving-model points sampled per stage.
    """

    def __init__(
        self,
        fixed_model: pv.PolyData,
        rotation_axis: Optional[int] = None,
        number_of_points: int = 5000,
        log_level: int | str = logging.INFO,
    ):
        """Initialize ITK ICP model registration.

        Args:
            fixed_model: PyVista target surface model.
            rotation_axis: World axis, 0, 1 or 2, that the alignment may rotate
                about; rotation about the other two is not allowed. For example
                2 for axial CT or MR, where patients lie along the scanner's
                z-axis and only in-plane rotation is expected. Default: None
                (rotation about any axis).
            number_of_points: Moving-model points, sampled evenly through the
                point list, that each stage pairs with the fixed model. All
                fixed-model points are candidates. Default: 5000
            log_level: Logging level. Default: logging.INFO

        Raises:
            ValueError: If rotation_axis is not None, 0, 1 or 2.
        """
        super().__init__(fixed_model=fixed_model, log_level=log_level)
        if rotation_axis not in (None, 0, 1, 2):
            raise ValueError(
                f"rotation_axis must be None, 0, 1 or 2; got {rotation_axis}"
            )
        self.rotation_axis = rotation_axis
        self.number_of_points = number_of_points

    @staticmethod
    def _point_set(points: np.ndarray) -> itk.PointSet:
        """Wrap an (n, 3) array as an ITK point set."""
        point_set = _POINT_SET_TYPE.New()
        point_set.SetPoints(
            itk.vector_container_from_array(
                np.ascontiguousarray(points, dtype=np.float32).ravel()
            )
        )
        return point_set

    def _stage_transform_and_weights(
        self, mode: str
    ) -> tuple[itk.MatrixOffsetTransformBase, list[float]]:
        """Return the transform a stage optimizes and its per-parameter weights.

        A weight of zero holds that parameter at its initial value. The first
        three parameters of every transform here are the versor components about
        x, y and z.
        """
        transform: itk.MatrixOffsetTransformBase
        if mode == "Rigid":
            transform = itk.VersorRigid3DTransform[itk.D].New()
        elif mode == "Similarity":
            transform = itk.Similarity3DTransform[itk.D].New()
        else:
            transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        weights = [1.0] * transform.GetNumberOfParameters()
        if self.rotation_axis is not None:
            for axis in range(3):
                if axis != self.rotation_axis:
                    weights[axis] = 0.0
            if mode == "Affine":
                # Skews 9, 10, 11 shear the (x, y), (x, z) and (y, z) planes.
                for index, plane in zip((9, 10, 11), ((0, 1), (0, 2), (1, 2))):
                    if self.rotation_axis in plane:
                        weights[index] = 0.0
        return transform, weights

    def _icp_stage(
        self, model: pv.PolyData, mode: str, max_iterations: int
    ) -> tuple[pv.PolyData, itk.AffineTransform]:
        """Run one ITK ICP stage against the fixed model.

        Args:
            model: Current state of the moving model.
            mode: Transform mode, one of ``"Rigid"``, ``"Similarity"`` or
                ``"Affine"``.
            max_iterations: Maximum optimizer iterations for this stage.

        Returns:
            Tuple of the transformed model and this stage's moving-to-fixed
            point transform.
        """
        moving_points = np.asarray(model.points, dtype=np.float64)
        step = max(1, len(moving_points) // max(1, self.number_of_points))
        sampled_points = moving_points[::step]

        transform, weights = self._stage_transform_and_weights(mode)
        transform.SetCenter([float(c) for c in moving_points.mean(axis=0)])

        # The metric pairs each of its *fixed* points with the closest of its
        # *moving* points, mapped by the inverse of the transform it optimizes.
        # Handing it the moving model as its fixed set therefore pairs every
        # model point with its closest target point, as ICP does, and the
        # optimized transform maps model points onto the target.
        metric = itk.EuclideanDistancePointSetToPointSetMetricv4[_POINT_SET_TYPE].New()
        metric.SetFixedPointSet(self._point_set(sampled_points))
        metric.SetMovingPointSet(
            self._point_set(np.asarray(self.fixed_model.points, dtype=np.float64))
        )
        metric.SetMovingTransform(transform)
        metric.Initialize()

        scales_estimator = itk.RegistrationParameterScalesFromPhysicalShift[
            type(metric)
        ].New()
        scales_estimator.SetMetric(metric)
        scales_estimator.SetTransformForward(True)
        scales_estimator.SetVirtualDomainPointSet(
            metric.GetVirtualTransformedPointSet()
        )

        parameter_weights = itk.OptimizerParameters[itk.D](len(weights))
        for index, weight in enumerate(weights):
            parameter_weights[index] = weight

        optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
        optimizer.SetMetric(metric)
        optimizer.SetScalesEstimator(scales_estimator)
        optimizer.SetWeights(parameter_weights)
        optimizer.SetLearningRate(1.0)
        optimizer.SetRelaxationFactor(0.8)
        optimizer.SetMinimumStepLength(1e-7)
        optimizer.SetGradientMagnitudeTolerance(1e-10)
        optimizer.SetNumberOfIterations(max_iterations)
        optimizer.SetReturnBestParametersAndValue(True)
        optimizer.StartOptimization()
        self.log_info(
            "%s ICP: mean distance %.4f after %d iterations (%s)",
            mode,
            optimizer.GetValue(),
            optimizer.GetCurrentIteration(),
            optimizer.GetStopConditionDescription(),
        )

        stage_transform = itk.AffineTransform[itk.D, 3].New()
        stage_transform.SetMatrix(transform.GetMatrix())
        stage_transform.SetOffset(transform.GetOffset())
        transformed = self.transform_tools.transform_pvcontour(
            model,
            stage_transform,
            with_deformation_magnitude=False,
        )
        return transformed, stage_transform
