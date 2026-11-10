"""Module for segmenting heart from chest CT images using Simpleware Medical.

This module provides the SegmentHeartSimpleware class that implements
heart segmentation using Synopsys Simpleware Medical's ASCardio module.
It inherits from SegmentAnatomyBase and provides heart-specific anatomical
structure mappings.
"""

import csv
import logging
import os
import tempfile

import itk
import numpy as np

from .process_images import ProcessImages
from .segment_anatomy_base import SegmentAnatomyBase
from .simpleware_medical import DEFAULT_SIMPLEWARE_EXE_PATH, run_simpleware_script


def _simpleware_global_to_lps(
    point: tuple[float, float, float], reference_image: itk.Image
) -> tuple[float, float, float]:
    """Convert a Simpleware Medical global-frame point to reference_image's frame.

    Measured on ConsoleSimplewareMedical Y-2026.03: an imported image with
    direction ``D`` and origin ``O`` gets the image-to-global matrix
    ``[D | D @ O]``, applied to image-space coordinates ``index * spacing``.
    Exported measurements (``landmarks.csv``) are in that global frame, so
    ``g = D @ (O + index * spacing)`` and the physical point is
    ``g - D @ O + O``. This holds for any orthonormal ``D``, oblique included,
    and is unaffected by cropping or resampling the Simpleware document.

    Args:
        point: Point ``g`` as exported by Simpleware Medical, in mm.
        reference_image: The image that was imported into Simpleware.

    Returns:
        The point in reference_image's physical space.
    """
    direction = itk.array_from_matrix(reference_image.GetDirection())
    origin = np.array(reference_image.GetOrigin())
    lps = np.asarray(point) - direction @ origin + origin
    return (float(lps[0]), float(lps[1]), float(lps[2]))


class SegmentHeartSimpleware(SegmentAnatomyBase):
    """
    Heart CT segmentation using Simpleware Medical's ASCardio module.

    This class implements heart segmentation using Synopsys Simpleware Medical,
    a commercial medical image processing platform. It specifically leverages
    the ASCardio module for automated cardiac segmentation. The class handles
    the external process communication with Simpleware Medical and converts
    between ITK and Simpleware image formats.

    Simpleware Medical provides high-quality segmentation for cardiac
    structures including chambers, myocardium, and major vessels. The
    segmentation is performed by launching Simpleware Medical as an external
    process and running a Python script within the Simpleware environment.

    The class maintains specific ID mappings for:
    - Heart structures (left/right atrium, left/right ventricle, myocardium)
    - Major vessels (aorta, pulmonary artery)

    Attributes:
        target_spacing (float): Target spacing set to 1.0mm for Simpleware.
        simpleware_exe_path (str): Path to Simpleware Medical executable.
        simpleware_script_path (str): Path to Simpleware Python script.

    The heart and major-vessel labels populated by this class are accessed
    through the inherited :attr:`SegmentAnatomyBase.taxonomy`
    (``taxonomy.labels_in_group("heart")`` etc.).

    Example:
        >>> segmenter = SegmentHeartSimpleware()
        >>> result = segmenter.segment(ct_image)
        >>> labelmap = result['labelmap']
        >>> heart_labelmap = result['heart']

    See :class:`SegmentHeartSimplewareTrimmedBranches` for a variant that
    additionally clips pulmonary/great-vessel branches to the cardiac region.
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the Simpleware Medical based heart segmentation.

        Sets up the Simpleware-specific anatomical structure ID mappings
        and processing parameters. The target spacing is set to 1.0mm which
        is optimal for cardiac segmentation in Simpleware Medical.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.landmarks: dict[str, tuple[float, float, float]] = {}

        self.target_spacing = 1.0

        # Heart and major-vessel labels from Simpleware Medical ASCardio.
        # Lung, bone, soft_tissue and contrast are not segmented by ASCardio,
        # and SegmentAnatomyBase seeds no defaults for them, so those groups
        # stay empty and never reach a result. Ids ASCardio does emit but no
        # group claims are folded into 'other' by _finalize_other_group().
        for group_name, organs in (
            (
                "heart",
                {
                    1: "left_ventricle",
                    2: "right_ventricle",
                    3: "left_atrium",
                    4: "right_atrium",
                    5: "myocardium",
                    6: "heart",
                },
            ),
            (
                "major_vessels",
                {
                    7: "aorta",
                    8: "pulmonary_artery",
                    9: "right_coronary_artery",
                    10: "left_coronary_artery",
                },
            ),
        ):
            for label_id, organ_name in organs.items():
                self.taxonomy.add_organ(group_name, label_id, organ_name)

        self._finalize_other_group()

        # Path to Simpleware Medical console executable
        self.simpleware_exe_path = DEFAULT_SIMPLEWARE_EXE_PATH

        # Path to the Simpleware Python script for heart segmentation
        self.simpleware_script_path = os.path.join(
            os.path.dirname(__file__),
            "simpleware_medical",
            "SimplewareScript_heart_segmentation.py",
        )

    def set_simpleware_executable_path(self, path: str) -> None:
        """Set the path to the Simpleware Medical console executable.

        The landmarks' coordinate mapping was measured on Y-2026.03; other
        versions may place them elsewhere.

        Args:
            path (str): Full path to ConsoleSimplewareMedical.exe

        Example:
            >>> segmenter.set_simpleware_executable_path(
            ...     "C:/Program Files/Synopsys/Simpleware Medical/Y-2026.03/ConsoleSimplewareMedical.exe"
            ... )
        """
        self.simpleware_exe_path = path

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """
        Run Simpleware Medical ASCardio segmentation on the preprocessed image.

        This implementation calls Simpleware Medical as an external process,
        passing the preprocessed image via a temporary file. The Simpleware
        Python script (SimplewareScript_heart_segmentation.py) runs within the
        Simpleware environment and uses the ASCardio module for heart
        segmentation. The results are written as per-structure MHD mask files and assembled
        into a labelmap, then read back as an ITK image.

        Args:
            preprocessed_image (itk.image): The preprocessed CT image with
                isotropic spacing and appropriate intensity scaling

        Returns:
            itk.image: The segmentation labelmap with heart and vessel labels
                from the ASCardio module

        Raises:
            FileNotFoundError: If Simpleware Medical executable is not found
            RuntimeError: If Simpleware Medical process fails
            ValueError: If output segmentation is not produced

        Note:
            Requires a valid installation of Synopsys Simpleware Medical
            with the ASCardio module. The method creates temporary files
            for input/output communication with Simpleware.

        Example:
            >>> labelmap = segmenter.segmentation_method(preprocessed_ct)
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save preprocessed image to temporary file
            tmp_input_image_file = os.path.join(tmp_dir, "input_image.nii.gz")

            self.log_info("Writing input image to: %s", tmp_input_image_file)
            itk.imwrite(preprocessed_image, tmp_input_image_file, compression=True)

            self.log_info("Running Simpleware Medical ASCardio segmentation...")
            stdout, stderr = run_simpleware_script(
                self.simpleware_exe_path,
                tmp_input_image_file,
                tmp_dir,
                self.simpleware_script_path,
            )
            if stdout:
                self.log_info("Simpleware stdout:\n%s", stdout)
            if stderr:
                self.log_warning("Simpleware stderr:\n%s", stderr)

            # Simpleware's right ventricle, left atrium, right atrium correspond to
            #   the interior of those regions.
            mask_ids_of_interior_regions = [1, 2, 3, 4]

            # Check if output file was created
            sz = [s for s in preprocessed_image.GetLargestPossibleRegion().GetSize()]
            sz = sz[::-1]
            labelmap_array = np.zeros(sz, dtype=np.uint8)
            interior_array = np.zeros(sz, dtype=np.uint8)
            mask_image = None
            for mask_id, mask_name in self.taxonomy.all_labels().items():
                output_file = os.path.join(tmp_dir, f"mask_{mask_name}.mhd")
                if os.path.exists(output_file):
                    mask_image = itk.imread(output_file)
                    mask_array = itk.GetArrayFromImage(mask_image).astype(np.uint8)
                    tmp_array = (mask_array > 128).astype(np.uint8)
                    if mask_id in mask_ids_of_interior_regions:
                        interior_array = np.where(
                            interior_array == 0, tmp_array, interior_array
                        )
                    mask_array = tmp_array * mask_id
                    labelmap_array = np.where(
                        labelmap_array == 0, mask_array, labelmap_array
                    )

            # landmarks.csv is optional: Simpleware Medical's ASCardio module
            # writes it for some valid inputs and omits it for others (e.g.
            # very small ROIs or unsupported acquisitions). Treat its
            # absence as "no landmarks for this case" rather than a hard
            # failure, so callers that only need the labelmap still succeed.
            # Landmarks are exported in Simpleware's global frame; see
            # _simpleware_global_to_lps for the conversion.
            landmarks_file = os.path.join(tmp_dir, "landmarks.csv")
            self.landmarks.clear()
            if os.path.exists(landmarks_file):
                with open(landmarks_file, newline="", encoding="utf-8-sig") as fh:
                    next(fh)  # skip line 1 (file header)
                    for row in csv.DictReader(fh):
                        coords = row["Measurement"].replace(" mm", "").split(",")
                        self.landmarks[row["Name"]] = _simpleware_global_to_lps(
                            (float(coords[0]), float(coords[1]), float(coords[2])),
                            preprocessed_image,
                        )
            else:
                self.log_warning(
                    "Simpleware did not write landmarks.csv (looked at %s); "
                    "continuing with an empty landmark set.",
                    landmarks_file,
                )

            # Dilate the interior regions to simulate 3mm myocardium (heart)
            interior_image = itk.GetImageFromArray(interior_array.astype(np.uint8))
            interior_image.CopyInformation(preprocessed_image)
            imMath = ProcessImages()
            spacing = interior_image.GetSpacing()
            exterior_image = imMath.binary_dilate_image(
                interior_image, round(7 / spacing[0]), 1, 0
            )
            exterior_image = imMath.binary_erode_image(
                exterior_image, round(4 / spacing[0]), 1, 0
            )
            exterior_array = itk.GetArrayFromImage(exterior_image)
            mask_id = 6  # Heart mask id
            exterior_array = exterior_array * mask_id
            labelmap_array = np.where(
                labelmap_array == 0, exterior_array, labelmap_array
            )

            if not np.any(labelmap_array != 0):
                raise ValueError(
                    "Simpleware Medical produced no segmentation output: no mask_*.mhd "
                    "files found or all masks are empty. Check Simpleware logs above and "
                    "ensure the ASCardio module ran successfully."
                )

            # Y-2026.03 mask exports carry preprocessed_image's direction, so
            # this flip is a no-op there; it is kept for older exports
            # (X-2025.06) whose direction could differ.
            if mask_image is not None:
                in_direction = itk.array_from_matrix(preprocessed_image.GetDirection())
                out_direction = itk.array_from_matrix(mask_image.GetDirection())
                flip = [False, False, False]
                for i in range(3):
                    if np.sign(out_direction[i, i]) != np.sign(in_direction[i, i]):
                        self.log_info(f"Flipping labelmap array along {i}-axis")
                        labelmap_array = np.flip(labelmap_array, axis=(2 - i))
                        flip[i] = True

            labelmap_image = itk.GetImageFromArray(labelmap_array.astype(np.uint8))
            labelmap_image.CopyInformation(preprocessed_image)

        return labelmap_image

    def get_landmarks(self) -> dict[str, tuple[float, float, float]]:
        """Get the landmarks from the last run, in the input image's physical space.

        The mapping from Simpleware's frame was measured on Y-2026.03.
        """
        return self.landmarks
