"""Module for segmenting contrast-enhanced chest CT images with TotalSegmentator.

This module provides the SegmentChestTotalSegmentatorWithContrast class, which
extends SegmentChestTotalSegmentator with an additional connected-component
pass that labels contrast-enhanced blood (in the heart, vessels, and lungs)
under a dedicated "contrast" taxonomy group.
"""

import logging
from typing import Optional

import itk
import numpy as np

from .image_tools import ImageTools
from .segment_chest_total_segmentator import SegmentChestTotalSegmentator


class SegmentChestTotalSegmentatorWithContrast(SegmentChestTotalSegmentator):
    """
    Chest CT segmentation using TotalSegmentator, with contrast-enhanced blood detection.

    Extends :class:`SegmentChestTotalSegmentator` with an additional
    connected-component pass that identifies contrast-enhanced blood vessels
    and cardiac chambers, labeling them under a ``"contrast"`` taxonomy
    group (label id 155). Use this class instead of
    :class:`SegmentChestTotalSegmentator` for contrast-enhanced studies.

    Attributes:
        contrast_threshold (int): Lower intensity threshold used to detect
            contrast-enhanced blood.

    Example:
        >>> segmenter = SegmentChestTotalSegmentatorWithContrast()
        >>> result = segmenter.segment(ct_image)
        >>> labelmap = result['labelmap']
        >>> contrast_labelmap = result['contrast']
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the contrast-enhanced TotalSegmentator-based segmentation.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.contrast_threshold: int = 500

    def _add_extra_taxonomy_groups(self) -> None:
        """Register the ``"contrast"`` taxonomy group (label id 155)."""
        self.taxonomy.add_organ("contrast", 155, "contrast")

    def postprocess_after_labelmap(
        self, input_image: itk.Image, labelmap_image: itk.Image
    ) -> itk.Image:
        """Run contrast-enhanced blood detection on the labelmap.

        Overrides :meth:`SegmentAnatomyBase.postprocess_after_labelmap`.

        Args:
            input_image (itk.Image): The original, unpreprocessed input image
            labelmap_image (itk.Image): The postprocessed segmentation labelmap

        Returns:
            itk.Image: The labelmap, with contrast-enhanced regions labeled
        """
        return self.segment_contrast_agent(input_image, labelmap_image)

    def segment_connected_component(
        self,
        preprocessed_image: itk.Image,
        labelmap_image: itk.Image,
        lower_threshold: int,
        upper_threshold: int,
        labelmap_ids: Optional[list[int]] = None,
        mask_id: int = 0,
        use_mid_slice: bool = True,
        hole_fill: int = 2,
    ) -> itk.Image:
        """
        Segment connected components based on intensity thresholding.

        Identifies connected regions within intensity thresholds and existing
        anatomical masks, then selects the largest component. This is useful
        for segmenting structures like contrast-enhanced blood or specific
        tissue types.

        Args:
            preprocessed_image (itk.Image): The preprocessed input image
            labelmap_image (itk.Image): Existing labelmap to constrain search
            lower_threshold (int): Lower intensity threshold
            upper_threshold (int): Upper intensity threshold
            labelmap_ids (Optional[list[int]]): List of label IDs to search within.
                If None, searches within all existing labels
            mask_id (int): ID to assign to the segmented component
            use_mid_slice (bool): If True, find largest component in middle
                slice only; if False, use entire 3D volume
            hole_fill (int): Number of pixels to dilate/erode for hole filling

        Returns:
            itk.Image: Updated labelmap with new component labeled as mask_id

        Example:
            >>> # Segment contrast-enhanced blood
            >>> updated_labels = segmenter.segment_connected_component(
            ...     preprocessed_image, labels, 700, 4000, mask_id=155
            ... )
        """
        thresh_image = itk.binary_threshold_image_filter(
            Input=preprocessed_image,
            LowerThreshold=lower_threshold,
            UpperThreshold=upper_threshold,
            InsideValue=1,
            OutsideValue=0,
        )
        thresh_arr = itk.GetArrayFromImage(thresh_image).astype(np.int16)
        thresh_image = itk.GetImageFromArray(thresh_arr)
        thresh_image.CopyInformation(preprocessed_image)

        label_arr = itk.GetArrayFromImage(labelmap_image)
        if labelmap_ids is None:
            labelmap_ids = list(self.taxonomy.all_labels().keys())
        label_arr = np.isin(label_arr, labelmap_ids)
        label_image = itk.GetImageFromArray(label_arr.astype(np.int16))
        label_image.CopyInformation(labelmap_image)

        connected_component_image = itk.connected_component_image_filter(
            Input=thresh_image,
            MaskImage=label_image,
        )

        connected_component_arr = itk.GetArrayFromImage(connected_component_image)
        if use_mid_slice:
            mid_slice = (
                connected_component_image.GetLargestPossibleRegion().GetSize()[2] // 2
            )
            tmp_connected_component_arr = connected_component_arr[mid_slice, :, :]
            ids = np.unique(tmp_connected_component_arr)
            if len(ids[ids != 0]) > 0:
                connected_component_arr = tmp_connected_component_arr

        ids = np.unique(connected_component_arr)
        ids = ids[ids != 0]
        if ids.size == 0:
            self.log_debug(
                "segment_connected_component: no connected components found "
                "in threshold [%d, %d]; returning labelmap unchanged",
                lower_threshold,
                upper_threshold,
            )
            return labelmap_image
        component_sums = [np.sum(connected_component_arr == id) for id in ids]
        largest_id = ids[np.argmax(component_sums)]
        connected_component_image = itk.binary_threshold_image_filter(
            Input=connected_component_image,
            LowerThreshold=int(largest_id),
            UpperThreshold=int(largest_id),
            InsideValue=1,
            OutsideValue=0,
        )
        image_tools = ImageTools()
        connected_component_image = image_tools.binary_dilate_image(
            connected_component_image, hole_fill, 1, 0
        )
        connected_component_image = image_tools.binary_erode_image(
            connected_component_image, hole_fill, 1, 0
        )

        labelmap_arr = itk.GetArrayFromImage(labelmap_image)
        connected_component_arr = itk.GetArrayFromImage(connected_component_image)
        connected_component_mask = connected_component_arr > 0
        mask = label_arr & connected_component_mask
        labelmap_arr = np.where(mask, mask_id, labelmap_arr)
        results_image = itk.GetImageFromArray(labelmap_arr.astype(np.uint8))
        results_image.CopyInformation(preprocessed_image)

        return results_image

    def segment_contrast_agent(
        self, preprocessed_image: itk.Image, labelmap_image: itk.Image
    ) -> itk.Image:
        """
        Include contrast-enhanced blood in the labelmap.

        Segments high-intensity regions corresponding to contrast-enhanced
        blood vessels and cardiac chambers. Uses connected component analysis
        focused on the middle slice where the heart is typically located.

        Args:
            preprocessed_image (itk.Image): The preprocessed CT image
            labelmap_image (itk.Image): Existing segmentation labelmap

        Returns:
            itk.Image: Updated labelmap with contrast-enhanced regions labeled

        Note:
            Assumes the mid-z slice of the data contains the heart.

        Example:
            >>> contrast_labels = segmenter.segment_contrast_agent(preprocessed_image, base_labels)
        """
        thoracic_ids = (
            list(self.taxonomy.labels_in_group("heart").keys())
            + list(self.taxonomy.labels_in_group("lung").keys())
            + list(self.taxonomy.labels_in_group("major_vessels").keys())
            + [0]
        )
        contrast_ids = list(self.taxonomy.labels_in_group("contrast").keys())
        if len(contrast_ids) == 0:
            self.log_warning("No contrast-enhanced regions found in the labelmap")
            return labelmap_image

        results_image = self.segment_connected_component(
            preprocessed_image,
            labelmap_image,
            lower_threshold=self.contrast_threshold,
            upper_threshold=4000,
            labelmap_ids=thoracic_ids,
            mask_id=contrast_ids[-1],
            use_mid_slice=True,
            hole_fill=3,
        )

        return results_image
