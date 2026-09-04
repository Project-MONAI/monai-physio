"""Module for heart segmentation with pulmonary/great-vessel branches trimmed.

This module provides the SegmentHeartSimplewareTrimmedBranches class, a
specialization of SegmentHeartSimpleware that clips vessel branches back to
the cardiac region after Simpleware segmentation.
"""

import logging

import itk
import numpy as np

from .image_tools import ImageTools
from .segment_heart_simpleware import SegmentHeartSimpleware


class SegmentHeartSimplewareTrimmedBranches(SegmentHeartSimpleware):
    """SegmentHeartSimpleware with pulmonary/great-vessel branches trimmed
    to the cardiac region (matches KCL-Heart-Model template extent).

    Example:
        >>> segmenter = SegmentHeartSimplewareTrimmedBranches()
        >>> result = segmenter.segment(ct_image)
        >>> labelmap = result['labelmap']
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the trimmed-branches heart segmentation.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """Run Simpleware ASCardio segmentation, then trim vessel branches.

        Args:
            preprocessed_image (itk.image): The preprocessed CT image

        Returns:
            itk.image: The segmentation labelmap with branches trimmed
        """
        labelmap_image = super().segmentation_method(preprocessed_image)
        return self.trim_branches(labelmap_image)

    def trim_branches(self, labelmap_image: itk.image) -> itk.image:
        """Trim pulmonary and great-vessel branches back to the cardiac region.

        Clips pulmonary veins and the great vessels (aorta, pulmonary artery)
        to the portions adjacent to the heart and keeps only the largest
        connected component of the left and right atria.  Reduces inter-subject
        variability in vessel extent, simplifying AI-Ready and Sim-Ready model
        fitting.  Consistent with how vessels were trimmed in the example KCL
        Heart dataset.

        Depends on the specific label IDs assigned in
        :meth:`SegmentHeartSimpleware.__init__` (1=left_ventricle,
        2=right_ventricle, 3=left_atrium, 4=right_atrium, 5=myocardium,
        6=heart) - a future change to that label scheme must keep this
        method in sync.
        """

        # Reference code for cropping aorta and pulmonary artery to
        #    portions adjacent to the heart.
        # Trim z-axis
        # z = labelmap_array.shape[2] - 1
        # z_classes = np.unique(labelmap_array[z, :, :])
        # heart_count = np.sum((c in [1, 2, 3, 4, 5]) for c in z_classes)
        # while heart_count < 3 and z > 0:
        #     z -= 1
        #     z_classes = np.unique(labelmap_array[z, :, :])
        #     heart_count = np.sum((c in [1, 2, 3, 4, 5]) for c in z_classes)
        # if z < labelmap_array.shape[2] - 3:
        # labelmap_array[(z + 3) :, :, :] = 0

        # In labelmap,
        #  if pixel is in keep_mask, was left or right atrium, then keep as
        #     left or right atrium

        #  1) Erase Heart and Myo label
        labelmap_arr = itk.array_from_image(labelmap_image)

        heart_arr = itk.array_from_image(labelmap_image)
        heart_arr[heart_arr == 6] = 0
        heart_arr[heart_arr == 5] = 0

        image_tools = ImageTools()
        spacing = labelmap_image.GetSpacing()

        #  2) Erode then Dilate Left Atrium label to clip vessels
        #  3) Erode then Dilate Right Atrium label to clip vessels
        #
        #  Each label is isolated into its own binary mask before the
        #  open (erode-then-dilate) operation so that the opening of one
        #  label can never bleed into a neighboring label.
        simple_arr = heart_arr.copy()
        for label_id in (3, 4):
            label_mask_arr = (heart_arr == label_id).astype(np.uint8)
            label_mask_img = itk.image_from_array(label_mask_arr)
            label_mask_img.CopyInformation(labelmap_image)
            radius = round(7 / spacing[0])
            label_mask_img = image_tools.binary_erode_image(
                label_mask_img, radius, 1, 0
            )
            label_mask_img = image_tools.binary_dilate_image(
                label_mask_img, radius, 1, 0
            )

            #  Keep only the largest connected component of this label
            label_mask_img = image_tools.keep_largest_connected_component(
                label_mask_img, foreground_value=1
            )
            label_mask_arr = itk.array_from_image(label_mask_img)

            simple_arr[simple_arr == label_id] = 0
            simple_arr[label_mask_arr > 0] = label_id

        #  4) Dilate all others = keep_mask
        keep_mask_arr = heart_arr.copy()
        keep_mask_arr[keep_mask_arr == 2] = 1
        keep_mask_arr[keep_mask_arr == 5] = 1
        keep_mask_arr[keep_mask_arr != 1] = 0
        keep_mask = itk.image_from_array(keep_mask_arr.astype(np.uint8))
        keep_mask.CopyInformation(labelmap_image)
        keep_mask = image_tools.binary_dilate_image(
            keep_mask, round(7 / spacing[0]), 1, 0
        )
        keep_mask_arr = itk.array_from_image(keep_mask)

        #  Add the left and right atrium labels to the keep_mask
        heart_arr = heart_arr * keep_mask_arr
        heart_arr[simple_arr == 3] = 3
        heart_arr[simple_arr == 4] = 4
        heart_img = itk.image_from_array(heart_arr)
        heart_img.CopyInformation(labelmap_image)

        #  Dilate the keep_mask to simulate 3mm (heart)
        keep_mask_arr = heart_arr.copy()
        keep_mask_arr[keep_mask_arr == 1] = 0
        keep_mask_arr[keep_mask_arr > 0] = 1
        keep_mask = itk.image_from_array(keep_mask_arr.astype(np.uint8))
        keep_mask.CopyInformation(labelmap_image)
        keep_mask = image_tools.binary_dilate_image(
            keep_mask, round(5 / spacing[0]), 1, 0
        )
        heart_mask = image_tools.binary_erode_image(
            keep_mask, round(2 / spacing[0]), 1, 0
        )

        #  Insert the heart and myo labels back into the labelmap
        heart_mask_arr = itk.array_from_image(heart_mask)
        heart_mask_arr[heart_arr > 0] = 0
        heart_arr[heart_mask_arr > 0] = 6
        heart_arr_myo = itk.array_from_image(labelmap_image)
        heart_arr[heart_arr_myo == 5] = 5
        heart_arr[heart_arr_myo == 1] = 1
        heart_img = itk.image_from_array(heart_arr)
        heart_img.CopyInformation(labelmap_image)

        #  Add in missing pieces / gaps of the myocardium
        lv_arr = heart_arr.copy()
        lv_arr[lv_arr != 1] = 0
        lv_img = itk.image_from_array(lv_arr.astype(np.uint8))
        lv_img.CopyInformation(labelmap_image)
        lv_img = image_tools.binary_dilate_image(lv_img, round(2 / spacing[0]), 1, 0)
        lv_arr = itk.array_from_image(lv_img)
        lv_arr = lv_arr * 5  # Myocardium label is 5

        #  Add the gap-filled myocardium back into the labelmap
        heart_arr = np.where(heart_arr == 0, lv_arr, heart_arr)
        # Eliminate overlap with other labels
        heart_arr = np.where(labelmap_arr > 6, 0, heart_arr)
        heart_img = itk.image_from_array(heart_arr)
        heart_img.CopyInformation(labelmap_image)

        return heart_img
