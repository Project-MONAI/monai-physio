"""Module for segmenting CT images using NVIDIA NV-Segment-CT.

This module provides the SegmentNVSegmentCT class, which implements CT
segmentation using NVIDIA's NV-Segment-CT model (a VISTA3D derivative
finetuned on CT scans). Model weights are downloaded on first use from
https://huggingface.co/nvidia/NV-Segment-CT.

The labelmap ids emitted by this class are the model's own published class
indices, taken verbatim from ``NV-Segment-CT/configs/label_dict.json`` in
https://github.com/NVIDIA-Medtech/NV-Segment-CTMR (e.g. 6 = aorta,
115 = heart). Those indices run to 132, and the model omits several ids
(2, 16, 18, 20, 21) as deprecated, so unclaimed ids in ``[1, 133)`` end up in
the ``other`` group.
"""

import glob
import logging
import os
import sys
import tempfile
from typing import Any, Optional

import itk

from .segment_anatomy_base import SegmentAnatomyBase


class SegmentNVSegmentCT(SegmentAnatomyBase):
    """CT segmentation using NVIDIA's NV-Segment-CT model.

    NV-Segment-CT is a VISTA3D-architecture network finetuned on CT scans. It
    covers 117 "segment everything" classes (132 addressable via label
    prompt) and, like NV-Segment-CTMR, supports only automatic (label-prompt)
    segmentation - there is no point-click interactive branch exposed here.

    Model weights (~872 MB) are downloaded from :attr:`hf_repo_id` on the
    first call to :meth:`segmentation_method` and cached by ``huggingface_hub``
    thereafter.

    Labelmap ids are the model's published class indices, used verbatim.

    Anatomy groups (heart, major_vessels, lung, bone, soft_tissue) are
    populated into :attr:`SegmentAnatomyBase.taxonomy`, reusing the names the
    TotalSegmentator and NV-Segment-CTMR backends use so downstream consumers
    see the same group keys.

    Licensing:
        The NV-Segment-CT *weights* are released under the NVIDIA Open Model
        License Agreement (research use only, not for clinical use); the
        surrounding bundle code is Apache 2.0. :attr:`license_warning` is
        logged at ``WARNING`` on the first call to :meth:`segmentation_method`.

    Attributes:
        target_spacing (float): 1.5mm, matching the model bundle's internal
            resampling, so the image is interpolated once rather than twice.
        model_cache_dir (Optional[str]): Download destination passed to
            ``huggingface_hub``. ``None`` uses the default Hugging Face cache.
        hf_repo_id (str): Hugging Face repository holding the bundle and
            weights.
        hf_revision (str): Pinned commit of :attr:`hf_repo_id` to download.
        hf_allow_patterns (tuple[str, ...]): Files pulled from
            :attr:`hf_repo_id`.
        license_warning (str): Banner logged at ``WARNING`` on first use.

    The anatomy labels populated by this class are accessed through the
    inherited :attr:`SegmentAnatomyBase.taxonomy`
    (``taxonomy.labels_in_group("heart")`` etc.).

    Note:
        :attr:`SegmentAnatomyBase.fast_mode` is ignored: this model has a
        single network and no reduced-accuracy variant.

    Example:
        >>> segmenter = SegmentNVSegmentCT()
        >>> result = segmenter.segment(ct_image)
        >>> labelmap = result['labelmap']
        >>> heart_labelmap = result['heart']
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the NV-Segment-CT-based segmentation.

        Populates :attr:`SegmentAnatomyBase.taxonomy` with the model's class
        indices, then calls
        :meth:`SegmentAnatomyBase._finalize_other_group` over the model's full
        ``[1, 133)`` class index space so unclaimed ids end up in the ``other``
        group. Constructing the class downloads nothing; weights are fetched
        lazily by :meth:`segmentation_method`.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        # The bundle resamples to 1.5mm isotropic internally (Spacingd), so
        # preprocessing to the same spacing avoids a second interpolation.
        self.target_spacing = 1.5

        self.hf_repo_id = "nvidia/NV-Segment-CT"

        # Pinned to a commit rather than tracking main: the repo publishes no
        # tags, and an unpinned download would silently swap the weights (and
        # the bundle's pipeline code, which is imported and executed here)
        # whenever upstream pushes. Bump deliberately after re-testing.
        self.hf_revision = "afb51518689f71e6abb367ee6301b2cd0225c66a"

        # model.safetensors and model_monai1.3.pt are deliberately excluded:
        # the former holds the same weights under the raw MONAI keys (no
        # 'network.' prefix), so it is unusable here, and the latter is an
        # older checkpoint format. Pulling either would bloat the download.
        self.hf_allow_patterns = (
            "*.py",
            "config.json",
            "metadata.json",
            "scripts/*.py",
            "vista3d_pretrained_model/config.json",
            "vista3d_pretrained_model/model.pt",
        )

        self.model_cache_dir: Optional[str] = None

        # The weights carry a more restrictive license than the rest of this
        # repository, so the restriction is surfaced at run time rather than
        # left to the class docstring.
        self.license_warning = (
            "\n"
            "  ==============================================================\n"
            "  RESEARCH-ONLY LICENSE\n"
            "  --------------------------------------------------------------\n"
            "  NV-Segment-CT weights are released under the NVIDIA Open Model\n"
            "  License Agreement: research use only, not for clinical use.\n"
            "  This is more restrictive than the rest of MONAI Physio.\n"
            "  https://huggingface.co/nvidia/NV-Segment-CT\n"
            "  =============================================================="
        )

        # NV-Segment-CT class indices, grouped by anatomy. Ids omitted from
        # NV-Segment-CT/configs/label_dict.json (2, 16, 18, 20, 21, and
        # everything above 128 except 132) are deprecated or unused and are
        # never emitted by the bundle.
        for group_name, organs in (
            (
                "heart",
                {
                    108: "atrial_appendage_left",
                    115: "heart",
                },
            ),
            (
                "major_vessels",
                {
                    6: "aorta",
                    7: "inferior_vena_cava",
                    17: "portal_vein_and_splenic_vein",
                    25: "hepatic_vessel",
                    58: "iliac_artery_left",
                    59: "iliac_artery_right",
                    60: "iliac_vena_left",
                    61: "iliac_vena_right",
                    109: "brachiocephalic_trunk",
                    110: "brachiocephalic_vein_left",
                    111: "brachiocephalic_vein_right",
                    112: "common_carotid_artery_left",
                    113: "common_carotid_artery_right",
                    119: "pulmonary_vein",
                    123: "subclavian_artery_left",
                    124: "subclavian_artery_right",
                    125: "superior_vena_cava",
                },
            ),
            (
                "lung",
                {
                    23: "lung_tumor",
                    28: "lung_upper_lobe_left",
                    29: "lung_lower_lobe_left",
                    30: "lung_upper_lobe_right",
                    31: "lung_middle_lobe_right",
                    32: "lung_lower_lobe_right",
                    132: "airway",
                },
            ),
            (
                "bone",
                {
                    33: "vertebrae_l5",
                    34: "vertebrae_l4",
                    35: "vertebrae_l3",
                    36: "vertebrae_l2",
                    37: "vertebrae_l1",
                    38: "vertebrae_t12",
                    39: "vertebrae_t11",
                    40: "vertebrae_t10",
                    41: "vertebrae_t9",
                    42: "vertebrae_t8",
                    43: "vertebrae_t7",
                    44: "vertebrae_t6",
                    45: "vertebrae_t5",
                    46: "vertebrae_t4",
                    47: "vertebrae_t3",
                    48: "vertebrae_t2",
                    49: "vertebrae_t1",
                    50: "vertebrae_c7",
                    51: "vertebrae_c6",
                    52: "vertebrae_c5",
                    53: "vertebrae_c4",
                    54: "vertebrae_c3",
                    55: "vertebrae_c2",
                    56: "vertebrae_c1",
                    63: "rib_1_left",
                    64: "rib_2_left",
                    65: "rib_3_left",
                    66: "rib_4_left",
                    67: "rib_5_left",
                    68: "rib_6_left",
                    69: "rib_7_left",
                    70: "rib_8_left",
                    71: "rib_9_left",
                    72: "rib_10_left",
                    73: "rib_11_left",
                    74: "rib_12_left",
                    75: "rib_1_right",
                    76: "rib_2_right",
                    77: "rib_3_right",
                    78: "rib_4_right",
                    79: "rib_5_right",
                    80: "rib_6_right",
                    81: "rib_7_right",
                    82: "rib_8_right",
                    83: "rib_9_right",
                    84: "rib_10_right",
                    85: "rib_11_right",
                    86: "rib_12_right",
                    87: "humerus_left",
                    88: "humerus_right",
                    89: "scapula_left",
                    90: "scapula_right",
                    91: "clavicula_left",
                    92: "clavicula_right",
                    93: "femur_left",
                    94: "femur_right",
                    95: "hip_left",
                    96: "hip_right",
                    97: "sacrum",
                    114: "costal_cartilages",
                    120: "skull",
                    122: "sternum",
                    127: "vertebrae_s1",
                    128: "bone_lesion",
                },
            ),
            (
                "soft_tissue",
                {
                    1: "liver",
                    3: "spleen",
                    4: "pancreas",
                    5: "kidney_right",
                    8: "adrenal_gland_right",
                    9: "adrenal_gland_left",
                    10: "gallbladder",
                    11: "esophagus",
                    12: "stomach",
                    13: "duodenum",
                    14: "kidney_left",
                    15: "bladder",
                    19: "small_bowel",
                    22: "brain",
                    24: "pancreatic_tumor",
                    26: "hepatic_tumor",
                    27: "colon_cancer_primaries",
                    57: "trachea",
                    62: "colon",
                    98: "gluteus_maximus_left",
                    99: "gluteus_maximus_right",
                    100: "gluteus_medius_left",
                    101: "gluteus_medius_right",
                    102: "gluteus_minimus_left",
                    103: "gluteus_minimus_right",
                    104: "autochthon_left",
                    105: "autochthon_right",
                    106: "iliopsoas_left",
                    107: "iliopsoas_right",
                    116: "kidney_cyst_left",
                    117: "kidney_cyst_right",
                    118: "prostate",
                    121: "spinal_cord",
                    126: "thyroid_gland",
                },
            ),
        ):
            for label_id, organ_name in organs.items():
                self.taxonomy.add_organ(group_name, label_id, organ_name)

        self._finalize_other_group(range(1, 133))

        self._snapshot_dir: Optional[str] = None
        self._pipeline: Optional[Any] = None

    def _ensure_model(self) -> str:
        """Download the NV-Segment-CT bundle if needed and return its path.

        The snapshot is fetched once per instance and cached on disk by
        ``huggingface_hub``, so repeated calls are cheap. Logs
        :attr:`license_warning` on the first call, before the weights are
        obtained.

        Returns:
            str: Local directory holding the downloaded bundle.
        """
        if self._snapshot_dir is None:
            from huggingface_hub import snapshot_download

            self.log_warning(self.license_warning)
            self.log_info("Downloading %s (cached after first use)", self.hf_repo_id)
            self._snapshot_dir = snapshot_download(
                repo_id=self.hf_repo_id,
                revision=self.hf_revision,
                cache_dir=self.model_cache_dir,
                allow_patterns=list(self.hf_allow_patterns),
            )
        return self._snapshot_dir

    def _ensure_pipeline(self) -> Any:
        """Build the VISTA3D pipeline if needed and return it.

        Weight loading takes seconds and the pipeline is stateless across
        calls, so it is built once per instance and reused for every
        subsequent image or timepoint.

        Returns:
            Any: The bundle's ``VISTA3DPipeline`` on the current CUDA device.
        """
        if self._pipeline is None:
            snapshot_dir = self._ensure_model()

            # The bundle ships hugging_face_pipeline / vista3d_pipeline as
            # top-level modules inside the snapshot rather than as an installed
            # package, so the snapshot directory has to be importable. Move it
            # to the front rather than just ensuring it's present: if the
            # other backend's snapshot dir precedes it, that one would still
            # win the import even after the cache purge below.
            if snapshot_dir in sys.path:
                sys.path.remove(snapshot_dir)
            sys.path.insert(0, snapshot_dir)

            # NV-Segment-CT and NV-Segment-CTMR ship modules under these same
            # top-level names; if the other backend already imported them
            # from its own snapshot dir, drop the cached entries so this
            # backend's copy loads instead.
            for module_name in ("vista3d_config", "vista3d_model", "vista3d_pipeline"):
                cached = sys.modules.get(module_name)
                if cached is not None and not (
                    getattr(cached, "__file__", "") or ""
                ).startswith(snapshot_dir):
                    del sys.modules[module_name]

            import torch
            from vista3d_config import VISTA3DConfig
            from vista3d_model import VISTA3DModel
            from vista3d_pipeline import VISTA3DPipeline

            # The bundle's HuggingFacePipelineHelper builds the model through
            # PreTrainedModel.from_pretrained, which reads only
            # model.safetensors. That file stores the weights under the raw
            # MONAI keys, so loading it leaves every parameter of
            # VISTA3DModel.network randomly initialized. Load model.pt into the
            # network directly instead.
            model = VISTA3DModel(VISTA3DConfig())
            model.network.load_state_dict(
                torch.load(
                    os.path.join(snapshot_dir, "vista3d_pretrained_model", "model.pt"),
                    map_location="cpu",
                    weights_only=True,
                )
            )

            # Unindexed, so the pipeline follows torch.cuda.set_device: under a
            # distributed launcher each rank segments on its own GPU instead of
            # every rank piling onto GPU 0.  Identical in a single process,
            # where the current device is 0.
            self._pipeline = VISTA3DPipeline(model, device=torch.device("cuda"))
        return self._pipeline

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """Run NV-Segment-CT on the preprocessed image and return the result.

        The model's Hugging Face pipeline reads and writes NIfTI files, so the
        image is written to a temporary file and the prediction read back with
        ITK. That round trip also handles the coordinate-system conversion
        between ITK (LPS) and the bundle's internal RAS orientation.

        The bundle inverts its own preprocessing before saving, so the
        prediction is returned on the same grid as *preprocessed_image*.

        Args:
            preprocessed_image (itk.image): The preprocessed CT image with
                isotropic spacing

        Returns:
            itk.image: The segmentation labelmap with NV-Segment-CT class
                indices, as ``uint8``.

        Raises:
            RuntimeError: If the model pipeline produced no output volume.

        Note:
            Requires a CUDA GPU; the segmentation runs on whichever CUDA
            device is current in this process.

        Example:
            >>> labelmap = segmenter.segmentation_method(preprocessed_ct)
        """
        pipeline = self._ensure_pipeline()

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "in.nii.gz")
            out_dir = os.path.join(tmp_dir, "out")
            itk.imwrite(preprocessed_image, in_file, compression=True)

            self.log_info("Running NV-Segment-CT")
            pipeline([{"image": in_file}], output_dir=out_dir)

            # The bundle saves with separate_folder=True and its own postfix,
            # so locate the result rather than reconstructing its name.
            out_files = glob.glob(
                os.path.join(out_dir, "**", "*.nii.gz"), recursive=True
            )
            # One input dict in, so exactly one output is expected; anything
            # else means the bundle's output layout changed and picking a file
            # would be a guess.
            if len(out_files) != 1:
                raise RuntimeError(
                    f"NV-Segment-CT produced {len(out_files)} outputs in "
                    f"{out_dir}, expected 1."
                )

            labelmap_arr = itk.array_from_image(itk.imread(out_files[0])).astype(
                self.labelmap_dtype
            )

        # The bundle's postprocessing maps unpredicted voxels to 255 via
        # nan_to_num(nan=255). 255 is not a valid NV-Segment-CT class (ids run
        # to 132), so it can always be cleared.
        labelmap_arr[labelmap_arr == 255] = 0

        labelmap_image = itk.image_from_array(labelmap_arr)
        labelmap_image.CopyInformation(preprocessed_image)

        return labelmap_image
