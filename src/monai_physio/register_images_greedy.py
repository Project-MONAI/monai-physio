"""Greedy-based image registration implementation.

This module provides the RegisterImagesGreedy class, a concrete implementation of
RegisterImagesBase that uses the PICSL Greedy algorithm for image registration.
Greedy is a fast CPU-based deformable registration tool from the Penn Image
Computing and Science Lab. It supports affine and deformable registration and
can be used as an alternative to ANTs for 4D cardiac/lung CT registration.

See https://greedy.readthedocs.io/ and https://pypi.org/project/picsl-greedy/.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Optional, Union

import itk
import numpy as np
from numpy.typing import NDArray

from .image_tools import ImageTools
from .register_images_base import RegisterImagesBase
from .transform_tools import TransformTools


def _try_import_greedy() -> Any:
    """Import picsl_greedy; raise with helpful message if missing."""
    try:
        from picsl_greedy import Greedy3D

        return Greedy3D
    except ImportError as e:
        raise ImportError(
            "picsl-greedy is required for RegisterImagesGreedy. "
            "Install with: pip install picsl-greedy"
        ) from e


class RegisterImagesGreedy(RegisterImagesBase):
    """Greedy-based deformable image registration implementation.

    This class extends RegisterImagesBase to provide deformable image registration
    using the PICSL Greedy algorithm. Greedy is a fast CPU-based tool for 2D/3D
    medical image registration, supporting rigid, affine, and deformable (NCC/SSD)
    registration.

    Greedy-specific features:
    - Rigid and affine registration (-a -dof 6 or 12)
    - Deformable registration with multi-resolution (-n, -s)
    - Metrics: NMI, NCC, SSD (mapped from CC, Mattes, MeanSquares)
    - Optional mask support (-gm fixed, -mm moving when both provided)
    - SimpleITK in-memory interface via ImageTools

    Inherits from RegisterImagesBase:
    - Fixed and moving image management
    - Binary mask processing with optional dilation
    - Modality-specific parameter configuration
    - Standardized registration interface

    Attributes:
        number_of_iterations: List of iterations per level (e.g. [40, 20, 10])
        transform_type: 'Deformable', 'Affine', or 'Rigid'
        metric: 'CC' (→NCC), 'Mattes' (→NMI), or 'MeanSquares' (→SSD)
        deformable_smoothing: Smoothing sigmas for deformable (e.g. "2.0vox 0.5vox")
    """

    # picsl_greedy 0.0.12 segfaults when any metric allocates a working buffer
    # for a fixed grid larger than roughly 100M voxels (empirically: 95M voxels
    # succeeds, 104M crashes; this affects both single-channel and multi-component
    # metrics). Metric inputs are isotropically downsampled to stay under this
    # conservative cap. Greedy emits physical-space transforms, so a coarser
    # metric grid only coarsens warp sampling, not the coordinate frame.
    _MAX_METRIC_VOXELS = 90_000_000

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the Greedy image registration class.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.number_of_iterations: list[int] = [40, 20, 10]
        self.transform_type = "Deformable"
        self.metric = "CC"
        self.deformable_smoothing = "2.0vox 0.5vox"

    def set_number_of_iterations(self, number_of_iterations: list[int]) -> None:
        """Set the number of iterations per resolution level.

        Args:
            number_of_iterations: List of iterations (e.g. [40, 20, 10]).
        """
        self.number_of_iterations = number_of_iterations

    def set_transform_type(self, transform_type: str) -> None:
        """Set the type of transform: Deformable, Affine, or Rigid.

        Args:
            transform_type: 'Deformable', 'Affine', or 'Rigid'.
        """
        if transform_type not in ("Deformable", "Affine", "Rigid"):
            self.log_error("Invalid transform type: %s", transform_type)
            raise ValueError(f"Invalid transform type: {transform_type}")
        self.transform_type = transform_type

    def set_metric(self, metric: str) -> None:
        """Set the similarity metric (CC→NCC, Mattes→NMI, MeanSquares→SSD).

        This metric is used for both affine and deformable registration stages.
        Greedy recommends NCC or SSD for deformable registration; NMI works
        well for affine but is less suited to deformable.

        Args:
            metric: 'CC', 'Mattes', or 'MeanSquares'.
        """
        if metric not in ("CC", "Mattes", "MeanSquares"):
            self.log_error("Invalid metric: %s", metric)
            raise ValueError(f"Invalid metric: {metric}")
        self.metric = metric

    def _itk_to_sitk(self, itk_image: itk.Image) -> Any:
        """Convert ITK image to SimpleITK (for Greedy)."""
        image_tools = ImageTools()
        return image_tools.convert_itk_image_to_sitk(itk_image)

    def _sitk_to_itk(self, sitk_image: Any) -> itk.Image:
        """Convert SimpleITK image to ITK."""
        image_tools = ImageTools()
        return image_tools.convert_sitk_image_to_itk(sitk_image)

    def _greedy_metric(self) -> str:
        """Map base metric to Greedy metric string."""
        if self.metric == "CC":
            return "NCC 2x2x2"
        if self.metric == "Mattes":
            return "NMI"
        if self.metric == "MeanSquares":
            return "SSD"
        return "NCC 2x2x2"

    def _greedy_iterations_str(self) -> str:
        """Format iterations as Greedy -n string (e.g. 40x20x10)."""
        return "x".join(str(i) for i in self.number_of_iterations)

    def _metric_downsample_scale(self, reference_image: itk.Image) -> float:
        """Per-axis scale that keeps ``reference_image`` under the voxel cap.

        Returns ``1.0`` when the grid already fits within
        ``_MAX_METRIC_VOXELS``; otherwise returns the isotropic per-axis factor
        ``(_MAX_METRIC_VOXELS / voxels) ** (1/3)`` (always < 1.0) so the
        downsampled grid lands at or just below the cap.

        Args:
            reference_image: The fixed metric image whose voxel count drives
                the Greedy multi-component buffer size.

        Returns:
            Per-axis resampling scale in ``(0, 1]``.
        """
        size = reference_image.GetLargestPossibleRegion().GetSize()
        voxels = int(size[0]) * int(size[1]) * int(size[2])
        if voxels <= self._MAX_METRIC_VOXELS:
            return 1.0
        scale = float((self._MAX_METRIC_VOXELS / voxels) ** (1.0 / 3.0))
        self.log_info(
            "Greedy labelmap metric: downsampling %d-voxel fixed grid by "
            "%.3f/axis to stay under the %d-voxel picsl_greedy crash threshold.",
            voxels,
            scale,
            self._MAX_METRIC_VOXELS,
        )
        return scale

    def _downsample_image(
        self, image: itk.Image, scale: float, nearest: bool = False
    ) -> itk.Image:
        """Isotropically resample ``image`` by ``scale`` (no-op when >= 1.0).

        The physical extent is preserved exactly: the new per-axis spacing is
        chosen so ``new_size * new_spacing == old_size * old_spacing``, so the
        coarser grid covers the same world-space region with the same origin
        and direction.

        Args:
            image: Scalar 3D ``itk.Image`` to resample.
            scale: Per-axis factor in ``(0, 1]``; ``>= 1.0`` returns ``image``
                unchanged so the full-resolution path is untouched.
            nearest: Use nearest-neighbor interpolation (for labelmaps and
                masks) instead of linear.

        Returns:
            The resampled ``itk.Image``, or ``image`` itself when ``scale`` is
            ``>= 1.0``.
        """
        if scale >= 1.0:
            return image
        size = image.GetLargestPossibleRegion().GetSize()
        spacing = image.GetSpacing()
        new_size = [max(1, int(round(int(size[i]) * scale))) for i in range(3)]
        new_spacing = [float(spacing[i]) * int(size[i]) / new_size[i] for i in range(3)]
        kwargs: dict[str, Any] = {
            "output_origin": image.GetOrigin(),
            "output_direction": image.GetDirection(),
            "size": new_size,
            "output_spacing": new_spacing,
        }
        if nearest:
            kwargs["interpolator"] = itk.NearestNeighborInterpolateImageFunction.New(
                image
            )
        return itk.resample_image_filter(image, **kwargs)

    def _write_affine_matrix_file(self, mat_4x4: NDArray[np.float64]) -> str:
        """Write a 4x4 RAS affine matrix to a temporary Greedy ``.mat`` file.

        Greedy's in-memory interface corrupts the heap when a numpy affine
        matrix is supplied as an initial transform (``-ia``/``-it``); passing a
        file path instead avoids that native crash. Greedy reads a plain-text
        4x4 RAS matrix, which is what ``numpy.savetxt`` writes here.

        Args:
            mat_4x4: 4x4 affine matrix in RAS (Greedy) convention.

        Returns:
            Path to the written temporary ``.mat`` file. The caller is
            responsible for deleting it.
        """
        mat_4x4 = np.asarray(mat_4x4, dtype=np.float64)
        if mat_4x4.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {mat_4x4.shape}")
        fd, path = tempfile.mkstemp(suffix=".mat", prefix="greedy_aff_")
        os.close(fd)
        np.savetxt(path, mat_4x4, fmt="%.10f")
        self.log_debug("Wrote Greedy affine init matrix to %s", path)
        return path

    # RAS <-> LPS basis change: negate x and y, leave z. Its own inverse.
    _RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])

    def _matrix_to_itk_affine(self, mat_4x4: NDArray[np.float64]) -> itk.Transform:
        """Convert Greedy's 4x4 RAS affine matrix to an ITK AffineTransform.

        Greedy works in RAS (its logs print "Final RAS Transform" and its
        ``.mat`` files are RAS), while ITK -- and every transform this project
        stores or applies -- is LPS. The matrix therefore has to change basis on
        the way across: ``M_lps = F M_ras F`` and ``t_lps = F t_ras`` with
        ``F = diag(-1, -1, 1)``. Skipping this negates the x and y components of
        every Greedy result, which for a near-identity registration is a
        sub-millimetre error that is easy to miss and impossible to correct
        downstream.

        Only the affine needs this. Greedy's displacement fields come back as
        images carrying the input's own LPS metadata, so
        :meth:`_sitk_warp_to_itk_displacement_transform` passes them through
        unchanged. See docs/developer/transform_conventions.

        Args:
            mat_4x4: 4x4 affine matrix in Greedy's RAS convention.

        Returns:
            The equivalent ITK affine transform, in LPS.
        """
        mat_4x4 = np.asarray(mat_4x4, dtype=np.float64)
        if mat_4x4.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {mat_4x4.shape}")
        flip = self._RAS_TO_LPS
        M = flip @ mat_4x4[:3, :3] @ flip
        t = flip @ mat_4x4[:3, 3]
        center = itk.Point[itk.D, 3]()
        for i in range(3):
            center[i] = 0.0
        affine_tfm = itk.AffineTransform[itk.D, 3].New()
        affine_tfm.SetCenter(center)
        affine_tfm.SetMatrix(itk.GetMatrixFromArray(M))
        translation = itk.Vector[itk.D, 3]()
        for i in range(3):
            translation[i] = float(t[i])
        affine_tfm.SetTranslation(translation)
        return affine_tfm

    def _sitk_warp_to_itk_displacement_transform(
        self, warp_sitk: Any, reference_image: itk.Image
    ) -> itk.Transform:
        """Convert SimpleITK displacement field to ITK DisplacementFieldTransform."""
        field_itk = self._sitk_to_itk(warp_sitk)
        from .image_tools import ImageTools

        image_tools = ImageTools()
        arr = itk.array_from_image(field_itk)
        disp_itk = image_tools.convert_array_to_image_of_vectors(
            arr, reference_image, itk.D
        )
        disp_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
        disp_tfm.SetDisplacementField(disp_itk)
        return disp_tfm

    def _registration_method_affine_or_rigid(
        self,
        fixed_sitk: Any,
        moving_sitk: Any,
        iterations_str: str,
        metric_str: str,
        dof: int,
        fixed_mask_sitk: Optional[Any] = None,
        moving_mask_sitk: Optional[Any] = None,
        fixed_labelmap_sitk: Optional[Any] = None,
        moving_labelmap_sitk: Optional[Any] = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Run Greedy affine or rigid registration. Returns (4x4 matrix, loss)."""
        Greedy3D = _try_import_greedy()
        g = Greedy3D()

        cmd = "-d 3"
        if fixed_labelmap_sitk is not None and moving_labelmap_sitk is not None:
            cmd += " -w 0.60"
        cmd += " -i fixed moving"
        kwargs: dict[str, Any] = {
            "fixed": fixed_sitk,
            "moving": moving_sitk,
        }
        if fixed_labelmap_sitk is not None and moving_labelmap_sitk is not None:
            cmd += " -w 0.40 -i fixed_labelmap moving_labelmap"
            kwargs["fixed_labelmap"] = fixed_labelmap_sitk
            kwargs["moving_labelmap"] = moving_labelmap_sitk
        cmd += f" -a -dof {dof} -n {iterations_str} -m {metric_str} -o aff_out"
        kwargs["aff_out"] = None
        if fixed_mask_sitk is not None and moving_mask_sitk is not None:
            cmd += " -gm fixed_mask -mm moving_mask"
            kwargs["fixed_mask"] = fixed_mask_sitk
            kwargs["moving_mask"] = moving_mask_sitk
        self.log_debug("Greedy affine/rigid command: %s", cmd)
        g.execute(cmd, **kwargs)
        mat = np.array(g["aff_out"], dtype=np.float64)
        try:
            ml = g.metric_log()
            loss = float(ml[-1]["TotalPerPixelMetric"][-1]) if ml else 0.0
        except Exception:
            loss = 0.0
        self.log_info("Greedy affine/rigid registration loss: %s", loss)
        return mat, loss

    def _registration_method_deformable(
        self,
        fixed_sitk: Any,
        moving_sitk: Any,
        iterations_str: str,
        metric_str: str,
        fixed_mask_sitk: Optional[Any] = None,
        moving_mask_sitk: Optional[Any] = None,
        fixed_labelmap_sitk: Optional[Any] = None,
        moving_labelmap_sitk: Optional[Any] = None,
    ) -> tuple[Optional[NDArray[np.float64]], Any, float]:
        """Run Greedy deformable registration. Returns (affine 4x4 or None, warp_sitk, loss)."""
        Greedy3D = _try_import_greedy()
        g = Greedy3D()

        # Greedy seeds its own deformable stage with an affine pass, using the
        # configured metric.
        cmd_aff = "-d 3"
        if fixed_labelmap_sitk is not None and moving_labelmap_sitk is not None:
            cmd_aff += " -w 0.60"
        cmd_aff += " -i fixed moving"
        kwargs_aff: dict[str, Any] = {
            "fixed": fixed_sitk,
            "moving": moving_sitk,
        }
        if fixed_labelmap_sitk is not None and moving_labelmap_sitk is not None:
            cmd_aff += " -w 0.40 -i fixed_labelmap moving_labelmap"
            kwargs_aff["fixed_labelmap"] = fixed_labelmap_sitk
            kwargs_aff["moving_labelmap"] = moving_labelmap_sitk
        cmd_aff += f" -a -dof 12 -n {iterations_str} -m {metric_str} -o aff_init"
        kwargs_aff["aff_init"] = None
        if fixed_mask_sitk is not None and moving_mask_sitk is not None:
            cmd_aff += " -gm fixed_mask -mm moving_mask"
            kwargs_aff["fixed_mask"] = fixed_mask_sitk
            kwargs_aff["moving_mask"] = moving_mask_sitk
        self.log_debug("Greedy deformable affine-init command: %s", cmd_aff)
        g.execute(cmd_aff, **kwargs_aff)
        initial_affine = np.array(g["aff_init"], dtype=np.float64)
        self.log_info("Greedy deformable affine init complete")

        # Greedy crashes (heap corruption) when the affine init is passed as an
        # in-memory matrix via -it; write it to a temp file and pass the path.
        initial_affine_file = self._write_affine_matrix_file(initial_affine)
        cmd_def = "-d 3"
        if fixed_labelmap_sitk is not None and moving_labelmap_sitk is not None:
            cmd_def += " -w 0.60"
        cmd_def += " -i fixed moving"
        kwargs_def = {
            "fixed": fixed_sitk,
            "moving": moving_sitk,
        }
        if fixed_labelmap_sitk is not None and moving_labelmap_sitk is not None:
            cmd_def += " -w 0.40 -i fixed_labelmap moving_labelmap"
            kwargs_def["fixed_labelmap"] = fixed_labelmap_sitk
            kwargs_def["moving_labelmap"] = moving_labelmap_sitk
        cmd_def += (
            f" -it {initial_affine_file} -n {iterations_str}"
            f" -m {metric_str} -s {self.deformable_smoothing} -o warp_out"
        )
        kwargs_def["warp_out"] = None
        if fixed_mask_sitk is not None and moving_mask_sitk is not None:
            cmd_def += " -gm fixed_mask -mm moving_mask"
            kwargs_def["fixed_mask"] = fixed_mask_sitk
            kwargs_def["moving_mask"] = moving_mask_sitk

        self.log_debug("Greedy deformable command: %s", cmd_def)
        try:
            g.execute(cmd_def, **kwargs_def)
        finally:
            os.remove(initial_affine_file)
        warp_out = g["warp_out"]
        try:
            ml = g.metric_log()
            loss = float(ml[-1]["TotalPerPixelMetric"][-1]) if ml else 0.0
        except Exception:
            loss = 0.0
        self.log_info("Greedy deformable registration loss: %s", loss)
        return initial_affine, warp_out, loss

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Register moving image to fixed image using Greedy.

        Converts ITK images to SimpleITK, runs Greedy (affine and/or deformable),
        then converts outputs back to ITK transforms.

        Returns a dict with "forward_transform", "inverse_transform", and
        "loss". As with the other image-registration backends,
        forward_transform warps the moving image onto the fixed grid and
        inverse_transform warps the fixed image onto the moving grid; point and
        landmark warps use the opposite transform from image warps (see
        docs/developer/transform_conventions).
        """
        if self.fixed_image is None or self.fixed_image_pre is None:
            raise ValueError("Fixed image must be set before registration.")

        moving_pre = moving_image_pre if moving_image_pre is not None else moving_image

        metric_scale = self._metric_downsample_scale(self.fixed_image_pre)

        fixed_pre = self._downsample_image(self.fixed_image_pre, metric_scale)
        moving_pre = self._downsample_image(moving_pre, metric_scale)
        # warp_out lands on the (possibly downsampled) fixed grid; use the same
        # grid as the displacement-field reference so shapes match.
        displacement_reference = fixed_pre
        fixed_sitk = self._itk_to_sitk(fixed_pre)
        moving_sitk = self._itk_to_sitk(moving_pre)

        # Greedy applies one global metric to every input channel. A raw
        # integer labelmap is piecewise-constant, so NCC sees zero local
        # variance and emits NaN gradients (a native crash). Encode each
        # labelmap as a continuous label-plus-boundary-distance field instead.
        from .labelmap_tools import LabelmapTools

        labelmap_tools = LabelmapTools()
        fixed_labelmap_sitk = None
        moving_labelmap_sitk = None
        if self.fixed_labelmap is not None:
            fixed_labelmap_ds = self._downsample_image(
                self.fixed_labelmap, metric_scale, nearest=True
            )
            fixed_labelmap_dist_map = labelmap_tools.create_distance_map(
                fixed_labelmap_ds
            )
            fixed_labelmap_sitk = self._itk_to_sitk(fixed_labelmap_dist_map)
        if moving_labelmap is not None:
            moving_labelmap_ds = self._downsample_image(
                moving_labelmap, metric_scale, nearest=True
            )
            moving_labelmap_dist_map = labelmap_tools.create_distance_map(
                moving_labelmap_ds
            )
            moving_labelmap_sitk = self._itk_to_sitk(moving_labelmap_dist_map)

        fixed_mask_sitk = None
        moving_mask_sitk = None
        if self.fixed_mask is not None:
            fixed_mask_ds = self._downsample_image(
                self.fixed_mask, metric_scale, nearest=True
            )
            fixed_mask_sitk = self._itk_to_sitk(fixed_mask_ds)
        if moving_mask is not None:
            moving_mask_ds = self._downsample_image(
                moving_mask, metric_scale, nearest=True
            )
            moving_mask_sitk = self._itk_to_sitk(moving_mask_ds)

        iterations_str = self._greedy_iterations_str()
        metric_str = self._greedy_metric()

        forward_transform: itk.Transform
        inverse_transform: itk.Transform
        loss_val: float

        if self.transform_type == "Rigid":
            mat, loss_val = self._registration_method_affine_or_rigid(
                fixed_sitk,
                moving_sitk,
                fixed_mask_sitk=fixed_mask_sitk,
                moving_mask_sitk=moving_mask_sitk,
                fixed_labelmap_sitk=fixed_labelmap_sitk,
                moving_labelmap_sitk=moving_labelmap_sitk,
                iterations_str=iterations_str,
                metric_str=metric_str,
                dof=6,
            )
            forward_transform = self._matrix_to_itk_affine(mat)
            inverse_affine = itk.AffineTransform[itk.D, 3].New()
            forward_transform.GetInverse(inverse_affine)
            inverse_transform = inverse_affine
        elif self.transform_type == "Affine":
            mat, loss_val = self._registration_method_affine_or_rigid(
                fixed_sitk,
                moving_sitk,
                fixed_mask_sitk=fixed_mask_sitk,
                moving_mask_sitk=moving_mask_sitk,
                fixed_labelmap_sitk=fixed_labelmap_sitk,
                moving_labelmap_sitk=moving_labelmap_sitk,
                iterations_str=iterations_str,
                metric_str=metric_str,
                dof=12,
            )
            forward_transform = self._matrix_to_itk_affine(mat)
            inverse_affine = itk.AffineTransform[itk.D, 3].New()
            forward_transform.GetInverse(inverse_affine)
            inverse_transform = inverse_affine
        else:
            # Deformable: affine + warp
            aff_mat, warp_sitk, loss_val = self._registration_method_deformable(
                fixed_sitk,
                moving_sitk,
                fixed_mask_sitk=fixed_mask_sitk,
                moving_mask_sitk=moving_mask_sitk,
                fixed_labelmap_sitk=fixed_labelmap_sitk,
                moving_labelmap_sitk=moving_labelmap_sitk,
                iterations_str=iterations_str,
                metric_str=metric_str,
            )
            aff_tfm = (
                self._matrix_to_itk_affine(aff_mat) if aff_mat is not None else None
            )
            # warp_sitk can be displacement field (SimpleITK image) or numpy
            if hasattr(warp_sitk, "GetSize"):
                disp_tfm = self._sitk_warp_to_itk_displacement_transform(
                    warp_sitk, displacement_reference
                )
            else:
                # Assume numpy displacement field (z,y,x,3)
                from .image_tools import ImageTools

                image_tools = ImageTools()
                warp_arr = np.asarray(warp_sitk, dtype=np.float64)
                ref = displacement_reference
                disp_itk = image_tools.convert_array_to_image_of_vectors(
                    warp_arr, ref, itk.D
                )
                disp_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
                disp_tfm.SetDisplacementField(disp_itk)
            # forward_transform is consumed by transform_image(moving, ...,
            # fixed) to warp the moving image onto the fixed grid, so it holds
            # Greedy's raw affine+warp (Greedy applies the affine first, then
            # the warp). inverse_transform is the numerically inverted field,
            # used to warp the fixed image onto the moving grid. This matches
            # RegisterImagesANTS/ICON and RegisterTimeSeriesImages.
            forward_composite = itk.CompositeTransform[itk.D, 3].New()
            if aff_tfm is not None:
                forward_composite.AddTransform(aff_tfm)
            forward_composite.AddTransform(disp_tfm)
            forward_transform = forward_composite
            inv_disp = TransformTools().invert_displacement_field_transform(disp_tfm)
            inv_aff = itk.AffineTransform[itk.D, 3].New()
            if aff_tfm is not None:
                aff_tfm.GetInverse(inv_aff)
            inverse_composite = itk.CompositeTransform[itk.D, 3].New()
            inverse_composite.AddTransform(inv_disp)
            if aff_tfm is not None:
                inverse_composite.AddTransform(inv_aff)
            inverse_transform = inverse_composite

        return {
            "forward_transform": forward_transform,
            "inverse_transform": inverse_transform,
            "loss": loss_val,
        }
