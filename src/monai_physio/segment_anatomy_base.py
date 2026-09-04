"""Base class for segmenting anatomy in CT images.

This module provides the SegmentAnatomyBase class that serves as a foundation
for implementing different anatomy CT segmentation algorithms. It handles common
preprocessing, postprocessing, and anatomical structure organization tasks.
"""

import logging
from typing import Any

import itk
import numpy as np

from .anatomy_taxonomy import AnatomyTaxonomy
from .monai_physio_base import MONAIPhysioBase


class SegmentAnatomyBase(MONAIPhysioBase):
    """Base class for anatomy segmentation that provides common functionality for
    segmenting anatomy in CT images.

    This class implements preprocessing, postprocessing, and mask creation
    methods that are shared across different anatomy segmentation
    implementations. It owns an :class:`AnatomyTaxonomy` instance that
    captures the group→organ structure (e.g. ``heart`` contains
    ``atrial_appendage_left`` at id 61); subclasses populate it via
    ``self.taxonomy.add_organ(...)`` and call
    :meth:`_finalize_other_group` once they're done.

    Extensibility
    -------------
    Each segmenter is free to define its own group names — the taxonomy does
    not hard-code a fixed set. A new subclass adds groups by calling
    ``self.taxonomy.add_organ(group_name, label_id, organ_name)`` for each
    organ; the group is created lazily on first use. To assign a custom
    OmniSurface look to a new group, register it in
    :data:`monai_physio.usd_anatomy_tools.DEFAULT_RENDER_PARAMS` (see that
    module's docstring). Groups without a registered look fall back to the
    ``"other"`` entry, so they still render.

    Attributes:
        target_spacing (float): Target isotropic spacing for resampling.
        rescale_intensity_range (bool): Whether to rescale intensity values.
        fast_mode (bool): When True, subclasses may skip auxiliary model
            passes and use faster/less-accurate models to trade segmentation
            fidelity for speed (e.g. in automated tests). Defaults to False.
        labelmap_dtype (type): NumPy integer type of the labelmap returned by
            :meth:`segment`. Defaults to ``np.uint8``; subclasses whose class
            index space exceeds 255 (e.g.
            :class:`monai_physio.SegmentNVSegmentCTMRI`) set ``np.uint16``.
        taxonomy (AnatomyTaxonomy): Group→organ mapping shared with
            :class:`monai_physio.USDAnatomyTools`.
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the SegmentAnatomyBase class.

        1. Add their organ groups via ``self.taxonomy.add_organ(...)``.
        2. Call :meth:`_finalize_other_group` to fill in unclaimed ids.

        Args:
            log_level: Logging level (default: logging.INFO).
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.target_spacing: float = 0.0

        self.rescale_intensity_range: bool = False
        self.input_intensity_scale_range: list[int] = [0, 4096]
        self.output_intensity_scale_range: list[int] = [-1024, 3071]
        self.output_intensity_clip_range: list[int] = [-1024, 3071]

        self.fast_mode: bool = False

        self.labelmap_dtype: type = np.uint8

        # Single source of truth for the anatomy hierarchy. Subclasses
        # populate this; USDAnatomyTools and ConvertVTKToUSD consume it.
        self.taxonomy = AnatomyTaxonomy()

    def _finalize_other_group(self, id_range: range = range(1, 256)) -> None:
        """Fill the ``other`` group with any unclaimed ids in *id_range*.

        Subclasses call this at the end of ``__init__`` once they have
        populated their specific groups. The consolidated all-labels view is
        available via ``self.taxonomy.all_labels()``.

        Args:
            id_range: Id space to sweep. Defaults to ``range(1, 256)``, which
                matches the default ``uint8`` :attr:`labelmap_dtype`.
                Subclasses with a larger class index space pass a wider range
                (and set :attr:`labelmap_dtype` accordingly).
        """
        self.taxonomy.fill_other_group(id_range)

    def label_to_type(self, label_name: str) -> str:
        """Return the anatomy group ('heart', 'lung', etc.) for a label name.

        Used by :class:`monai_physio.ConvertVTKToUSD` to group label-mode
        mesh prims under per-type Xforms (e.g.
        ``/World/{basename}/heart/{label_name}``). Delegates to the taxonomy.

        Args:
            label_name: Organ name (a value in the taxonomy's group organ dicts).

        Returns:
            The anatomy group name. Falls back to ``"other"`` for any label
            the segmenter doesn't recognize.
        """
        return self.taxonomy.group_for_label(label_name)

    def set_target_spacing(self, target_spacing: float) -> None:
        """Set the target isotropic spacing for image resampling.

        Args:
            target_spacing (float): Target spacing in millimeters for all three
                spatial dimensions. Set to 0.0 to disable resampling.

        Example:
            >>> segmenter.set_target_spacing(1.0)  # 1mm isotropic spacing
        """
        self.target_spacing = target_spacing

    def preprocess_input(
        self,
        input_image: Any,
    ) -> Any:
        """Preprocess the input image for segmentation.

        Performs image preprocessing including resampling to isotropic spacing
        and optional intensity rescaling. The preprocessing ensures consistent
        image characteristics for reliable segmentation.

        Args:
            input_image (itk.image): The input 3D CT image to preprocess

        Returns:
            itk.image: The preprocessed image with isotropic spacing and
                optionally rescaled intensities

        Raises:
            AssertionError: If the input image is not 3D
            ValueError: If intensity rescaling parameters are invalid

        Example:
            >>> preprocessed = segmenter.preprocess_input(ct_image)
        """

        # Check the input image
        assert len(input_image.GetSpacing()) == 3, "The input image must be 3D"

        rescale_image = False
        results_image = None
        if self.target_spacing > 0.0:
            if (
                input_image.GetSpacing()[0] != self.target_spacing
                or input_image.GetSpacing()[1] != self.target_spacing
                or input_image.GetSpacing()[2] != self.target_spacing
            ):
                rescale_image = True
            else:
                isotropy = (
                    (input_image.GetSpacing()[1] / input_image.GetSpacing()[0])
                    + (input_image.GetSpacing()[2] / input_image.GetSpacing()[0])
                ) / 2
                if isotropy < 0.9 or isotropy > 1.1:
                    rescale_image = True
                    self.target_spacing = (
                        input_image.GetSpacing()[0]
                        + input_image.GetSpacing()[1]
                        + input_image.GetSpacing()[2]
                    ) / 3
                    self.log_info(
                        "Resampling to %.3f isotropic spacing", self.target_spacing
                    )
        if rescale_image:
            self.log_warning("The input image should have isotropic spacing")
            self.log_info("Input image has spacing: %s", str(input_image.GetSpacing()))
            self.log_info("Resampling to isotropic: %.3f", self.target_spacing)
            interpolator = itk.LinearInterpolateImageFunction.New(input_image)
            results_image = itk.resample_image_filter(
                input_image,
                interpolator=interpolator,
                output_spacing=[
                    self.target_spacing,
                    self.target_spacing,
                    self.target_spacing,
                ],
                size=[
                    int(
                        input_image.GetLargestPossibleRegion().GetSize()[0]
                        * input_image.GetSpacing()[0]
                        / self.target_spacing
                    ),
                    int(
                        input_image.GetLargestPossibleRegion().GetSize()[1]
                        * input_image.GetSpacing()[1]
                        / self.target_spacing
                    ),
                    int(
                        input_image.GetLargestPossibleRegion().GetSize()[2]
                        * input_image.GetSpacing()[2]
                        / self.target_spacing
                    ),
                ],
                output_origin=input_image.GetOrigin(),
                output_direction=input_image.GetDirection(),
            )
        else:
            results_image_arr = itk.GetArrayFromImage(input_image)
            results_image = itk.GetImageFromArray(results_image_arr)
            results_image.CopyInformation(input_image)

        results_image_arr = itk.GetArrayFromImage(results_image).astype(np.float32)
        minv = results_image_arr.min()
        maxv = results_image_arr.max()
        if self.rescale_intensity_range:
            self.log_info("Rescaling intensity range...")
            if (
                self.input_intensity_scale_range is None
                or self.output_intensity_scale_range is None
                or self.output_intensity_clip_range is None
            ):
                raise ValueError(
                    "output_intensity_scale_range must be set if input_intensity_scale_range is set"
                )
            minv = self.input_intensity_scale_range[0]
            maxv = self.input_intensity_scale_range[1]
            output_minv = self.output_intensity_scale_range[0]
            output_maxv = self.output_intensity_scale_range[1]
            results_image_arr = (results_image_arr - minv) / (maxv - minv) * (
                output_maxv - output_minv
            ) + output_minv
            results_image_arr = np.clip(
                results_image_arr,
                self.output_intensity_clip_range[0],
                self.output_intensity_clip_range[1],
            )

            new_results_image = itk.GetImageFromArray(results_image_arr)
            new_results_image.CopyInformation(results_image)
            results_image = new_results_image

        return results_image

    def postprocess_labelmap(
        self,
        labelmap_image: itk.image,
        input_image: itk.image,
    ) -> itk.image:
        """
        Resample the labelmap to match the input image spacing.

        Ensures the segmentation labelmap has the same spatial properties
        as the original input image by resampling using label-specific
        interpolation that preserves discrete label values.

        Args:
            labelmap_image (itk.image): The segmentation labelmap to resample
            input_image (itk.image): The original input image providing
                target spacing and geometry

        Returns:
            itk.image: The resampled labelmap matching input image properties

        Example:
            >>> final_labels = segmenter.postprocess_labelmap(labels, original_image)
        """
        input_spacing = np.array(input_image.GetSpacing())
        label_spacing = np.array(labelmap_image.GetSpacing())
        results_image = None
        if any(input_spacing != label_spacing):
            interpolator = itk.LabelImageGaussianInterpolateImageFunction.New(
                labelmap_image
            )
            results_image = itk.resample_image_filter(
                labelmap_image,
                interpolator=interpolator,
                ReferenceImage=input_image,
                UseReferenceImage=True,
            )
            labelmap_arr = itk.GetArrayFromImage(labelmap_image)
            results_arr = itk.GetArrayFromImage(results_image)
            new_results_arr = results_arr.copy()
            if results_arr[0, :, :].sum() == 0 and labelmap_arr[0, :, :].sum() > 0:
                sumi = 1
                sum = new_results_arr[sumi, :, :].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[sumi, :, :].sum()
                for i in range(sumi):
                    new_results_arr[i, :, :] = new_results_arr[sumi, :, :]
            if results_arr[-1, :, :].sum() == 0 and labelmap_arr[-1, :, :].sum() > 0:
                sumi = 2
                sum = new_results_arr[-sumi, :, :].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[-sumi, :, :].sum()
                for i in range(1, sumi):
                    new_results_arr[-i, :, :] = new_results_arr[-sumi, :, :]
            if results_arr[:, 0, :].sum() == 0 and labelmap_arr[:, 0, :].sum() > 0:
                sumi = 1
                sum = new_results_arr[:, sumi, :].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[:, sumi, :].sum()
                for i in range(sumi):
                    new_results_arr[:, i, :] = new_results_arr[:, sumi, :]
            if results_arr[:, -1, :].sum() == 0 and labelmap_arr[:, -1, :].sum() > 0:
                sumi = 2
                sum = new_results_arr[:, -sumi, :].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[:, -sumi, :].sum()
                for i in range(1, sumi):
                    new_results_arr[:, -i, :] = new_results_arr[:, -sumi, :]
            if results_arr[:, :, 0].sum() == 0 and labelmap_arr[:, :, 0].sum() > 0:
                sumi = 1
                sum = new_results_arr[:, :, sumi].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[:, :, sumi].sum()
                for i in range(sumi):
                    new_results_arr[:, :, i] = new_results_arr[:, :, sumi]
            if results_arr[:, :, -1].sum() == 0 and labelmap_arr[:, :, -1].sum() > 0:
                sumi = 2
                sum = new_results_arr[:, :, -sumi].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[:, :, -sumi].sum()
                for i in range(1, sumi):
                    new_results_arr[:, :, -i] = new_results_arr[:, :, -sumi]
            results_image = itk.GetImageFromArray(new_results_arr)
            results_image.CopyInformation(input_image)
        else:
            results_image_arr = itk.GetArrayFromImage(labelmap_image)
            results_image = itk.GetImageFromArray(results_image_arr)
            results_image.CopyInformation(labelmap_image)

        return results_image

    def postprocess_after_labelmap(
        self, input_image: itk.image, labelmap_image: itk.image
    ) -> itk.image:
        """
        Hook for subclass-specific labelmap refinement before mask creation.

        Called by :meth:`segment` after :meth:`postprocess_labelmap`, and
        before the per-group masks are derived from the labelmap. The base
        implementation is a no-op; subclasses that offer optional features
        gated behind their own settings (e.g. TotalSegmentator's
        contrast-enhanced-study detection) override this to apply them.

        Args:
            input_image (itk.image): The original, unpreprocessed input image
            labelmap_image (itk.image): The postprocessed segmentation labelmap

        Returns:
            itk.image: The labelmap to use for mask creation
        """
        return labelmap_image

    def create_anatomy_group_labelmaps(
        self, labelmap_image: itk.image
    ) -> dict[str, itk.image]:
        """
        Create labelmaps for different anatomical groups from the labelmap.

        Generates separate labelmaps for major anatomical systems by
        grouping related anatomical structures from the detailed labelmap.
        Each group's labelmap retains the original label ids for voxels
        belonging to that group and is zero elsewhere. This is useful for
        motion analysis and visualization.

        Args:
            labelmap_image (itk.image): The detailed segmentation labelmap

        Returns:
            dict[str, itk.image]: Dictionary of labelmaps keyed by group
                name. Exactly one entry per group registered in
                :attr:`taxonomy` (plus ``"other"``). The returned key set
                is segmenter-specific — callers that need a particular
                group should check membership (``"lung" in labelmaps``)
                rather than assume a fixed schema.

        Example:
            >>> labelmaps = segmenter.create_anatomy_group_labelmaps(labelmap)
            >>> if "lung" in labelmaps:
            ...     lung_labelmap = labelmaps["lung"]
        """
        labelmap_arr = itk.GetArrayFromImage(labelmap_image)
        other_labelmap_arr = np.where(labelmap_arr > 0, labelmap_arr, 0)

        labelmaps: dict[str, itk.image] = {}
        for group_name in self.taxonomy.group_names():
            if group_name == AnatomyTaxonomy.OTHER_GROUP:
                continue
            group_ids = list(self.taxonomy.labels_in_group(group_name).keys())
            group_labelmap_arr = np.where(
                np.isin(labelmap_arr, group_ids), labelmap_arr, 0
            )
            other_labelmap_arr = np.where(group_labelmap_arr > 0, 0, other_labelmap_arr)
            group_labelmap = itk.GetImageFromArray(group_labelmap_arr)
            group_labelmap.CopyInformation(labelmap_image)
            labelmaps[group_name] = group_labelmap

        other_labelmap = itk.GetImageFromArray(other_labelmap_arr)
        other_labelmap.CopyInformation(labelmap_image)
        labelmaps[AnatomyTaxonomy.OTHER_GROUP] = other_labelmap

        return labelmaps

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """
        Abstract method for image segmentation - must be implemented by subclasses.

        This method should contain the core segmentation algorithm specific to
        each implementation (e.g., TotalSegmentator).

        Args:
            preprocessed_image (itk.image): The preprocessed input image

        Returns:
            itk.image: The segmentation labelmap

        Raises:
            NotImplementedError: If called on the base class

        Note:
            This method must be implemented by subclasses to provide the
            specific segmentation algorithm.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def segment(
        self,
        input_image: itk.image,
    ) -> dict[str, itk.image]:
        """
        Perform complete anatomy segmentation.

        This is the main segmentation method that coordinates preprocessing,
        segmentation, subclass-specific labelmap refinement, and anatomical
        group labelmap creation.

        Args:
            input_image (itk.image): The input 3D image to segment

        Returns:
            dict[str, itk.image]: Dictionary containing:
                - "labelmap": Detailed segmentation labelmap
                - one labelmap image per anatomy group, keyed by group name,
                  preserving the original label ids for that group

        Example:
            >>> result = segmenter.segment(image)
            >>> labelmap = result['labelmap']
            >>> heart_labelmap = result['heart']
        """
        preprocessed_image = self.preprocess_input(input_image)

        labelmap_image = self.segmentation_method(preprocessed_image)

        labelmap_image = self.postprocess_labelmap(labelmap_image, input_image)

        labelmap_image = self.postprocess_after_labelmap(input_image, labelmap_image)

        labelmaps = self.create_anatomy_group_labelmaps(labelmap_image)

        labelmap_image = itk.GetImageFromArray(
            itk.GetArrayFromImage(labelmap_image).astype(self.labelmap_dtype)
        )
        labelmap_image.CopyInformation(input_image)

        return {"labelmap": labelmap_image, **labelmaps}
