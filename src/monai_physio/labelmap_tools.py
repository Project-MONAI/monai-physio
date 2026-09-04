"""
Labelmap Tools for MONAI Physio

This module provides the :class:`LabelmapTools` class with the definitive
utility for turning a multi-label (or binary) segmentation labelmap into a
binary registration mask, optionally excluding specific labels and dilating
the result by a physical radius in millimeters.
"""

import logging
from typing import Optional

import itk
import numpy as np

from .monai_physio_base import MONAIPhysioBase


class LabelmapTools(MONAIPhysioBase):
    """
    Utilities for converting segmentation labelmaps into registration masks.

    A labelmap is an ``itk.Image`` of integer labels where ``0`` is background
    and each positive value identifies an anatomical structure. A registration
    mask is a binary ``itk.Image`` where every foreground voxel is ``1``. This
    class centralizes the labelmap-to-mask conversion so that thresholding,
    label exclusion, and physically isotropic dilation are performed
    identically everywhere in the platform.

    Example:
        >>> tools = LabelmapTools()
        >>> # Binary mask of every labeled voxel, dilated 5 mm
        >>> mask = tools.convert_labelmap_to_mask(labelmap, dilation_in_mm=5.0)
        >>> # Exclude the table/background labels 8 and 9 before masking
        >>> mask = tools.convert_labelmap_to_mask(
        ...     labelmap, dilation_in_mm=5.0, exclude_labels=[8, 9]
        ... )
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize LabelmapTools.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    def convert_labelmap_to_mask(
        self,
        labelmap: itk.Image,
        dilation_in_mm: float = 0.0,
        exclude_labels: Optional[list[int]] = None,
    ) -> itk.Image:
        """Convert a labelmap into a binary registration mask.

        Any voxel whose label is in ``exclude_labels`` is set to background
        first; every remaining non-zero voxel becomes foreground (``1``). The
        binary mask is then dilated by ``dilation_in_mm`` millimeters of
        physical radius. The radius is converted into per-axis voxel counts
        from the labelmap's spacing so the dilation is physically isotropic
        even on anisotropic grids; each per-axis count is clamped to at least
        1 voxel when ``dilation_in_mm > 0``.

        Args:
            labelmap: Multi-label or binary ``itk.Image``. Any non-zero voxel
                that is not excluded is treated as foreground.
            dilation_in_mm: Physical radius of the binary dilation in
                millimeters. Pass ``0`` (or negative) to skip dilation and
                return the raw thresholded mask. Default 0.0.
            exclude_labels: Optional list of integer label values to force
                to background before thresholding. When ``None`` (the default)
                no labels are excluded.

        Returns:
            ``itk.Image[itk.UC, 3]`` binary mask in the same physical space as
            ``labelmap`` (origin, spacing, direction copied from the input).
        """
        arr = itk.array_from_image(labelmap)
        if exclude_labels:
            arr = np.where(np.isin(arr, exclude_labels), 0, arr)
        mask_arr = (arr > 0).astype(np.uint8)
        mask = itk.image_from_array(mask_arr)
        mask.CopyInformation(labelmap)

        if dilation_in_mm <= 0:
            return mask

        spacing = labelmap.GetSpacing()
        radius = itk.Size[3]()
        for i in range(3):
            radius[i] = max(1, int(round(dilation_in_mm / float(spacing[i]))))
        structuring_element = itk.FlatStructuringElement[3].Ball(radius)
        return itk.binary_dilate_image_filter(
            mask, kernel=structuring_element, foreground_value=1
        )

    def create_distance_map(
        self,
        labelmap: itk.Image,
        max_distance_mm: float = 20.0,
        distance_scale: float = 5.0,
        preserve_labels: bool = True,
        fill_background_only: bool = False,
        exclude_labels: Optional[list[int]] = None,
    ) -> itk.Image:
        """Encode a labelmap as a continuous label-plus-boundary-distance image.

        Each output voxel holds its original integer label plus a small
        fractional offset that encodes how far the voxel lies from the nearest
        boundary between two differently-labeled regions:

            value = label + min(distance_to_nearest_boundary_mm,
                                max_distance_mm) / distance_scale

        The boundary set is every voxel that 6-neighbors a voxel with a
        different label (background label ``0`` participates, so the outer
        surface of each structure is a boundary). The unsigned physical
        distance from each voxel to that set is computed with
        ``SignedMaurerDistanceMapImageFilter`` (taking the magnitude), clipped
        to ``max_distance_mm``, divided by ``distance_scale``, and added to the
        voxel's original label.

        With the defaults (``20`` mm clip, ``5`` scale) the fractional offset
        stays in ``[0.0, 4.0]``, potentially passing adjacent integer labels but
        emphasizing in medial alignment as well as boundary.

        The motivation is registration metrics such as Greedy's NCC: a raw
        integer labelmap is piecewise-constant, so the local variance inside
        each region is zero and NCC produces NaN gradients. Replacing it with
        this continuous encoding gives every region a smoothly varying signal
        while preserving label identity.

        Args:
            labelmap: Multi-label (or binary) ``itk.Image`` of integer labels.
            max_distance_mm: Distance clip, in millimeters. Default 20.0.
            distance_scale: Divisor applied to the clipped distance before it
                is added to the label. Default 5.0. With the default clip
                this bounds the fractional offset to ``[0, 4.0]``.

        Returns:
            ``itk.Image[itk.F, 3]`` in the same physical space as ``labelmap``
            (origin, spacing, direction copied from the input).
        """
        labels = itk.array_from_image(labelmap)

        if exclude_labels:
            labels = np.where(np.isin(labels, exclude_labels), 0, labels)

        if not fill_background_only:
            # A voxel is on a label boundary when it differs from a 6-connected
            # neighbor along any axis. Mark both voxels straddling each change.
            boundary = np.zeros(labels.shape, dtype=bool)
            for axis in range(labels.ndim):
                changed = np.diff(labels, axis=axis) != 0
                lower = [slice(None)] * labels.ndim
                upper = [slice(None)] * labels.ndim
                lower[axis] = slice(0, -1)
                upper[axis] = slice(1, None)
                boundary[tuple(lower)] |= changed
                boundary[tuple(upper)] |= changed
        else:
            boundary = np.zeros(labels.shape, dtype=np.float32)
            boundary[labels > 0] = 1.0

        if boundary.any():
            boundary_image = itk.image_from_array(boundary.astype(np.uint8))
            boundary_image.CopyInformation(labelmap)
            distance_filter = itk.SignedMaurerDistanceMapImageFilter.New(
                Input=boundary_image
            )
            distance_filter.SetSquaredDistance(False)
            distance_filter.SetUseImageSpacing(True)
            distance_filter.Update()
            distance = itk.array_from_image(distance_filter.GetOutput()).astype(
                np.float32
            )
            if not fill_background_only:
                distance = np.abs(distance)
        else:
            # No inter-label boundary exists (single uniform label); every
            # voxel gets a zero offset.
            distance = np.zeros(labels.shape, dtype=np.float32)

        offset = np.clip(distance, 0.0, max_distance_mm) / distance_scale
        if preserve_labels:
            encoded = labels.astype(np.float32) + offset
        else:
            encoded = offset

        encoded_image = itk.image_from_array(encoded)
        encoded_image.CopyInformation(labelmap)
        return encoded_image
