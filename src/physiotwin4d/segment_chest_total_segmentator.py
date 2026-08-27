"""Module for segmenting chest CT images using TotalSegmentator.

This module provides the SegmentChestTotalSegmentator class that implements
chest CT segmentation using the TotalSegmentator deep learning model. It inherits
from SegmentAnatomyBase and defines anatomical structure mappings specific to
TotalSegmentator's output labels.
"""

import logging
import os
import tempfile

import itk
import nibabel as nib
import numpy as np

from .image_tools import ImageTools
from .segment_anatomy_base import SegmentAnatomyBase


class SegmentChestTotalSegmentator(SegmentAnatomyBase):
    """
    Chest CT segmentation using TotalSegmentator deep learning model.

    This class implements chest CT segmentation using the TotalSegmentator
    neural network, which provides detailed anatomical structure segmentation
    including organs, bones, and vessels. It maps TotalSegmentator's output
    labels to physiological groups for motion analysis.

    TotalSegmentator provides segmentation for 117 anatomical structures
    including detailed organ, bone, and vessel segmentation. This implementation
    combines the 'total' task (main organs and structures) with the 'body' task
    (body outline) to ensure complete coverage.

    Anatomy groups (heart, lung, bone, major_vessels, soft_tissue) are
    populated into :attr:`SegmentAnatomyBase.taxonomy` so downstream
    consumers (``ConvertVTKToUSD``, ``USDAnatomyTools``) see a single,
    consistent group→organ mapping.

    For contrast-enhanced studies (CT with contrast-enhanced blood in the
    heart/vessels), use :class:`SegmentChestTotalSegmentatorWithContrast`
    instead, which subclasses this class and adds a connected-component
    pass to label contrast-enhanced blood under a ``"contrast"`` taxonomy
    group.

    Attributes:
        target_spacing (float): Target spacing set to 1.0mm for TotalSegmentator.

    Example:
        >>> segmenter = SegmentChestTotalSegmentator()
        >>> result = segmenter.segment(ct_image)
        >>> labelmap = result['labelmap']
        >>> heart_labelmap = result['heart']
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the TotalSegmentator-based chest segmentation.

        Populates :attr:`SegmentAnatomyBase.taxonomy` with the
        TotalSegmentator class index space, then calls
        :meth:`SegmentAnatomyBase._finalize_other_group` so unclaimed ids end
        up in the ``other`` group.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.target_spacing = 1.0

        # TotalSegmentator class indices, grouped by anatomy.
        for group_name, organs in (
            (
                "heart",
                {
                    51: "heart",
                    61: "atrial_appendage_left",
                    140: "highres_myocardium",
                    141: "highres_atrium_left",
                    142: "highres_ventricle_left",
                    143: "highres_atrium_right",
                    144: "highres_ventricle_right",
                    146: "highres_pulmonary_artery",
                },
            ),
            (
                "major_vessels",
                {
                    52: "aorta",
                    145: "highres_aorta",
                    53: "pulmonary_vein",
                    54: "brachiocephalic_trunk",
                    55: "right_subclavian_artery",
                    56: "left_subclavian_artery",
                    57: "common_carotid_artery_right",
                    58: "common_carotid_artery_left",
                    59: "brachiocephalic_vein_left",
                    60: "brachiocephalic_vein_right",
                    62: "superior_vena_cava",
                    63: "inferior_vena_cava",
                },
            ),
            (
                "lung",
                {
                    10: "lung_upper_lobe_left",
                    11: "lung_lower_lobe_left",
                    12: "lung_upper_lobe_right",
                    13: "lung_middle_lobe_right",
                    14: "lung_lower_lobe_right",
                    120: "lung_arteries",
                    121: "lung_veins",
                    122: "lung_airways",
                    123: "lung_airways_wall",
                },
            ),
            (
                "bone",
                {
                    26: "vertebra_S1",
                    27: "vertebra_L5",
                    28: "vertebra_L4",
                    29: "vertebrae_L3",
                    30: "vertebrae_L2",
                    31: "vertebrae_L1",
                    32: "vertebrae_T12",
                    33: "vertebrae_T11",
                    34: "vertebrae_T10",
                    35: "vertebrae_T9",
                    36: "vertebrae_T8",
                    37: "vertebrae_T7",
                    38: "vertebrae_T6",
                    39: "vertebrae_T5",
                    40: "vertebrae_T4",
                    41: "vertebrae_T3",
                    42: "vertebrae_T2",
                    43: "vertebrae_T1",
                    44: "vertebrae_C7",
                    45: "vertebrae_C6",
                    46: "vertebrae_C5",
                    47: "vertebrae_C4",
                    48: "vertebrae_C3",
                    49: "vertebrae_C2",
                    50: "vertebrae_C1",
                    69: "humerus_left",
                    70: "humerus_right",
                    71: "scapula_left",
                    72: "scapula_right",
                    73: "clavicula_left",
                    74: "clavicula_right",
                    75: "femur_left",
                    76: "femur_right",
                    77: "hip_left",
                    78: "hip_right",
                    91: "skull",
                    92: "rib_left_1",
                    93: "rib_left_2",
                    94: "rib_left_3",
                    95: "rib_left_4",
                    96: "rib_left_5",
                    97: "rib_left_6",
                    98: "rib_left_7",
                    99: "rib_left_8",
                    100: "rib_left_9",
                    101: "rib_left_10",
                    102: "rib_left_11",
                    103: "rib_left_12",
                    104: "rib_right_1",
                    105: "rib_right_2",
                    106: "rib_right_3",
                    107: "rib_right_4",
                    108: "rib_right_5",
                    109: "rib_right_6",
                    110: "rib_right_7",
                    111: "rib_right_8",
                    112: "rib_right_9",
                    113: "rib_right_10",
                    114: "rib_right_11",
                    115: "rib_right_12",
                    116: "sternum",
                    117: "costal_cartilages",
                    25: "sacrum",
                },
            ),
            (
                "soft_tissue",
                {
                    1: "spleen",
                    2: "kidney_right",
                    3: "kidney_left",
                    4: "gallbladder",
                    5: "liver",
                    6: "stomach",
                    7: "pancreas",
                    8: "adrenal_gland_right",
                    9: "adrenal_gland_left",
                    17: "thyroid_gland",
                    18: "small_bowel",
                    19: "duodenum",
                    20: "colon",
                    21: "urinary_bladder",
                    22: "prostate",
                    80: "gluteus_maximus_left",
                    81: "gluteus_maximus_right",
                    82: "gluteus_medius_left",
                    83: "gluteus_medius_right",
                    84: "gluteus_minimus_left",
                    85: "gluteus_minimus_right",
                    90: "brain",
                    15: "esophagus",
                    16: "trachea",
                    133: "body_skin",  # 4 in body task
                    134: "tissue_subcutaneous_fat",  # tissue_4_types
                    135: "tissue_torso_fat",
                    136: "tissue_skeletal_muscle",
                    137: "tissue_intermuscular_fat",
                },
            ),
        ):
            for label_id, organ_name in organs.items():
                self.taxonomy.add_organ(group_name, label_id, organ_name)

        self._add_extra_taxonomy_groups()
        self._finalize_other_group()

        self.has_academic_license = False

    @staticmethod
    def _academic_license_is_valid() -> bool:
        """Return True when TotalSegmentator reports an installed license.

        Deliberately the same offline check ``show_license_info`` performs
        before a licensed task, so this predicts exactly whether that call
        would exit.  The offline check only tests that a license number is
        configured and 18 characters long, so a stale or revoked key of the
        right length still reads as installed; TotalSegmentator would then
        exit while downloading the licensed weights, which no pre-check of
        ours can prevent.

        ``has_valid_license`` would catch that by asking the backend, but it
        reports a network failure as ``invalid_license`` too, so a runner that
        is merely offline would silently segment without the licensed tasks
        and quietly produce different anatomy.  Wrongly degrading a valid
        licensed run is worse than the revoked-key case this misses.
        """
        from totalsegmentator.libs import (  # noqa: PLC0415
            has_valid_license_offline,
        )

        status, _ = has_valid_license_offline()
        return bool(status == "yes")

    def set_has_academic_license(self, has_academic_license: bool) -> None:
        """Request the licensed tasks, if a license is actually installed.

        ``heartchambers_highres`` and ``tissue_4_types`` are not openly
        available.  Asking for them without a license makes
        ``totalsegmentator`` print its licensing notice and call
        ``sys.exit(1)`` from inside the segmentation, which surfaces as a bare
        ``SystemExit`` partway through whatever workflow was running.  Check
        here instead, so that a machine without a key segments the heart as
        one structure rather than aborting the run.  The fallback is logged,
        because it is a coarser segmentation than a licensed run produces.

        Args:
            has_academic_license (bool): Whether the academic license is available
        """
        if has_academic_license and not self._academic_license_is_valid():
            self.log_warning(
                "No valid TotalSegmentator license found; skipping the "
                "'heartchambers_highres' and 'tissue_4_types' tasks, so the "
                "heart is segmented as a single structure and no chamber "
                "labels (141-144) are produced. Install one with "
                "'totalseg_set_license -l <key>'; a free academic license is "
                "at https://backend.totalsegmentator.com/license-academic/"
            )
            has_academic_license = False
        self.has_academic_license = has_academic_license

    def _add_extra_taxonomy_groups(self) -> None:
        """Hook for subclasses to add taxonomy groups before finalization.

        Called at the end of :meth:`__init__`, before
        :meth:`SegmentAnatomyBase._finalize_other_group` claims unclaimed ids
        into the ``other`` group. Subclasses (e.g.
        :class:`SegmentChestTotalSegmentatorWithContrast`) override this to
        register additional groups without duplicating the base class's
        organ mapping.
        """

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """
        Run TotalSegmentator on the preprocessed image and return result.

        This implementation always runs the 'total' task (major organs and
        structures). Outside fast mode it also runs the 'lung_vessels' overlay
        and the 'body' task; when ``has_academic_license`` is set it additionally
        runs the 'heartchambers_highres' and 'tissue_4_types' tasks. The 'body'
        task contributes only its skin outline (the skin label) into remaining
        background regions; it does not fill gaps with soft tissue.

        The method uses temporary files for coordinate system conversion between
        ITK (LPS) and nibabel (RAS) formats, which is required for proper
        integration with TotalSegmentator.

        Args:
            preprocessed_image (itk.image): The preprocessed CT image with
                isotropic spacing and appropriate intensity scaling

        Returns:
            itk.image: The segmentation labelmap with TotalSegmentator labels.
                The 'body' task's skin label is written into the remaining
                background regions as the skin outline.

        Note:
            Requires GPU acceleration (device="gpu:0") for reasonable performance.
            The method automatically handles coordinate system conversions between
            ITK and nibabel formats.

        Example:
            >>> labelmap = segmenter.segmentation_method(preprocessed_ct)
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            from totalsegmentator.python_api import totalsegmentator  # noqa: PLC0415

            # ITK and Nibabel use different coordinate systems (LPS vs RAS).
            # The safest conversion is via a temporary file. This approach
            # still reduces I/O compared to the original implementation.
            tmp_file = os.path.join(tmp_dir, "in.nii.gz")
            itk.imwrite(preprocessed_image, tmp_file, compression=True)
            nib_image = nib.load(tmp_file)

            # fast_mode trades accuracy for speed (e.g. for automated tests):
            # it runs only the 'total' task with TotalSegmentator's faster
            # model, skipping the 'body' background-fill and 'lung_vessels'
            # overlay passes below.
            # nr_thr_resamp defaults to 1; TotalSegmentator's post-prediction
            # resampling back to native resolution is CPU-bound and benefits
            # from parallelizing across the available cores.
            resamp_threads = min(12, os.cpu_count() or 1)
            output_nib_image_total = totalsegmentator(
                nib_image,
                task="total",
                device="gpu:0",
                fast=self.fast_mode,
                nr_thr_resamp=resamp_threads,
            )
            labelmap_arr_total = output_nib_image_total.get_fdata().astype(np.uint8)

            final_arr = labelmap_arr_total

            if not self.fast_mode:
                if self.has_academic_license:
                    self.log_info("Running heart chambers task")
                    output_nib_image_heart = totalsegmentator(
                        nib_image,
                        task="heartchambers_highres",
                        device="gpu:0",
                        nr_thr_resamp=resamp_threads,
                    )
                    labelmap_arr_heart = output_nib_image_heart.get_fdata().astype(
                        np.uint8
                    )
                    # labelmap_arr_heart contains: 1=myocardium, 2=atrium_left, 3=ventricle_left,
                    #     4=atrium_right, 5=ventricle_right, 6=aorta, 7=pulmonary_artery
                    final_arr = np.where(labelmap_arr_heart == 1, 140, final_arr)
                    final_arr = np.where(labelmap_arr_heart == 2, 141, final_arr)
                    final_arr = np.where(labelmap_arr_heart == 3, 142, final_arr)
                    final_arr = np.where(labelmap_arr_heart == 4, 143, final_arr)
                    final_arr = np.where(labelmap_arr_heart == 5, 144, final_arr)
                    final_arr = np.where(labelmap_arr_heart == 7, 146, final_arr)
                    # final_arr = np.where(labelmap_arr_heart == 6, 145, final_arr)
                    #  Aorta is not included in heart model.
                    #  Should include only a portion of the aorta in the heart model.

                    self.log_info("Running tissue_4_types task")
                    output_nib_image_tissue_4_types = totalsegmentator(
                        nib_image,
                        task="tissue_4_types",
                        device="gpu:0",
                        nr_thr_resamp=resamp_threads,
                    )
                    labelmap_arr_tissue_4_types = (
                        output_nib_image_tissue_4_types.get_fdata().astype(np.uint8)
                    )
                    # 134: "subcutaneous_fat", # tissue_4_types
                    # 135: "torso_fat",
                    # 136: "skeletal_muscle",
                    # 137: "intermuscular_fat"
                    final_arr = np.where(
                        labelmap_arr_tissue_4_types == 1, 134, final_arr
                    )
                    final_arr = np.where(
                        labelmap_arr_tissue_4_types == 2, 135, final_arr
                    )
                    final_arr = np.where(
                        labelmap_arr_tissue_4_types == 3, 136, final_arr
                    )
                    final_arr = np.where(
                        labelmap_arr_tissue_4_types == 4, 137, final_arr
                    )

                self.log_info("Running lung vessels task")
                output_nib_image_lung = totalsegmentator(
                    nib_image,
                    task="lung_vessels",
                    device="gpu:0",
                    nr_thr_resamp=resamp_threads,
                )
                labelmap_arr_lung = output_nib_image_lung.get_fdata().astype(np.uint8)
                # labelmap_arr_lung contains: 1=arteries, 2=veins, 3=airways,
                #     4=airways_wall
                final_arr = np.where(labelmap_arr_lung == 1, 120, final_arr)
                final_arr = np.where(labelmap_arr_lung == 2, 121, final_arr)
                final_arr = np.where(labelmap_arr_lung == 3, 122, final_arr)
                # final_arr = np.where(labelmap_arr_lung == 4, 123, final_arr)
                # Airway wall segmentation is too zealous.  Fills right atrium

                self.log_info("Running body task")
                output_nib_image_body = totalsegmentator(
                    nib_image, task="body", device="gpu:0", nr_thr_resamp=resamp_threads
                )
                labelmap_arr_body = output_nib_image_body.get_fdata().astype(np.uint8)
                # labelmap_arr_body contains: 1=body, 2=body_trunc, 3=body_extremities,
                #     4=skin
                # Only overwrite the background with body labels
                # mask = final_arr > 0
                # labelmap_arr_body[mask] = 0
                final_arr = np.where(
                    labelmap_arr_body == 4, 133, final_arr
                )  # body_skin

            # To create an ITK image, we save the result and read it back with
            # ITK. This correctly handles the coordinate system and data
            # layout conversions.
            out_tmp_file = os.path.join(tmp_dir, "out.nii.gz")
            # Use the affine from one of the outputs to preserve spatial info
            result_nib = nib.Nifti1Image(final_arr, output_nib_image_total.affine)
            nib.save(result_nib, out_tmp_file)
            labelmap_image = itk.imread(out_tmp_file)
            labelmap_arr = itk.array_from_image(labelmap_image).astype(np.uint8)

            # Add heart around interior regions.
            if self.has_academic_license:
                interior_mask = np.isin(labelmap_arr, [141, 142, 143, 144])
                # Binarize to foreground value 1 so the dilate/erode calls
                # below (which use foreground=1) operate on the mask.
                interior_arr = interior_mask.astype(np.uint8)
                interior_image = itk.GetImageFromArray(interior_arr)
                interior_image.CopyInformation(preprocessed_image)
                imMath = ImageTools()
                spacing = interior_image.GetSpacing()
                exterior_image = imMath.binary_dilate_image(
                    interior_image, round(7 / spacing[0]), 1, 0
                )
                exterior_image = imMath.binary_erode_image(
                    exterior_image, round(4 / spacing[0]), 1, 0
                )
                exterior_arr = itk.GetArrayFromImage(exterior_image)
                mask_id = 51  # Heart mask id
                exterior_arr = exterior_arr * mask_id
                labelmap_arr = np.where(labelmap_arr == 0, exterior_arr, labelmap_arr)
                skin_mask = np.isin(labelmap_arr, [133, 134, 135, 136, 137])
                replace_arr = np.where(skin_mask, exterior_arr, 0)
                labelmap_arr = np.where(replace_arr > 0, exterior_arr, labelmap_arr)

            labelmap_image = itk.image_from_array(labelmap_arr)
            labelmap_image.CopyInformation(preprocessed_image)

        return labelmap_image
