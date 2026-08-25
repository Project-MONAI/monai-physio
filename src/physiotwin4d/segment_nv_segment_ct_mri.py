"""Module for segmenting CT and MRI images using NVIDIA NV-Segment-CTMR.

This module provides the SegmentNVSegmentCTMRI class, which implements CT and
MRI segmentation using NVIDIA's NV-Segment-CTMR model (a VISTA3D derivative
finetuned on 30K+ CT and MRI scans). Model weights are downloaded on first use
from https://huggingface.co/nvidia/NV-Segment-CTMR.

The labelmap ids emitted by this class are the model's own published class
indices, taken verbatim from ``configs/label_dict.json`` in
https://github.com/NVIDIA-Medtech/NV-Segment-CTMR (e.g. 6 = aorta,
115 = heart). Those indices run to 345, which does not fit in ``uint8``, so
this class sets :attr:`SegmentAnatomyBase.labelmap_dtype` to ``np.uint16``.
"""

import glob
import logging
import os
import sys
import tempfile
from typing import Any, Optional

import itk
import numpy as np

from .segment_anatomy_base import SegmentAnatomyBase


class SegmentNVSegmentCTMRI(SegmentAnatomyBase):
    """CT and MRI segmentation using NVIDIA's NV-Segment-CTMR model.

    NV-Segment-CTMR is a VISTA3D-architecture network finetuned on more than
    30,000 CT and MRI scans. It covers 345 classes across three modalities
    (``CT_BODY``, ``MRI_BODY``, ``MRI_BRAIN``) and, unlike VISTA3D, supports
    only automatic (label-prompt) segmentation — there is no point-click
    interactive branch.

    Model weights (~872 MB) are downloaded from :attr:`hf_repo_id` on the
    first call to :meth:`segmentation_method` and cached by ``huggingface_hub``
    thereafter.

    Labelmap ids are the model's published class indices, used verbatim.
    Because those run to 345, :attr:`SegmentAnatomyBase.labelmap_dtype` is
    ``np.uint16`` rather than the ``np.uint8`` used by the other segmenters.

    Anatomy groups (heart, major_vessels, lung, bone, soft_tissue,
    brain_parcellation) are populated into
    :attr:`SegmentAnatomyBase.taxonomy`. The first five reuse the names the
    TotalSegmentator backend uses, so downstream consumers see the same group
    keys; ``brain_parcellation`` is new and renders with the grey-matter entry
    registered for it in
    :data:`physiotwin4d.usd_anatomy_tools.DEFAULT_RENDER_PARAMS`, plus
    organ-level overrides for the tissues that differ (white matter, CSF-filled
    ventricles, brainstem, cerebellum, pallidum).

    Licensing:
        The NV-Segment-CTMR *weights* are released under the NVIDIA OneWay
        Non-Commercial License (academic research use only); the surrounding
        bundle code is Apache 2.0. This is more restrictive than the rest of
        this repository. NV-Segment-CT (CT only, 132 classes) is the
        commercially licensed alternative. :attr:`license_warning` is logged
        at ``WARNING`` on the first call to :meth:`segmentation_method`.

    Attributes:
        target_spacing (float): 1.5mm, matching the model bundle's internal
            resampling, so the image is interpolated once rather than twice.
        modality (str): One of :attr:`modalities`; selects the model's
            predefined "segment everything" class list. Defaults to
            ``"CT_BODY"``.
        modalities (tuple[str, ...]): Modalities the model accepts in place of
            an explicit ``label_prompt``.
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
        >>> segmenter = SegmentNVSegmentCTMRI()
        >>> result = segmenter.segment(ct_image)
        >>> labelmap = result['labelmap']
        >>> heart_labelmap = result['heart']
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the NV-Segment-CTMR-based segmentation.

        Populates :attr:`SegmentAnatomyBase.taxonomy` with the model's class
        indices, then calls
        :meth:`SegmentAnatomyBase._finalize_other_group` over the model's full
        ``[1, 346)`` class index space so unclaimed ids end up in the ``other``
        group. Constructing the class downloads nothing; weights are fetched
        lazily by :meth:`segmentation_method`.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.labelmap_dtype = np.uint16

        # The bundle resamples to 1.5mm isotropic internally (Spacingd), so
        # preprocessing to the same spacing avoids a second interpolation.
        self.target_spacing = 1.5

        self.modalities = ("CT_BODY", "MRI_BODY", "MRI_BRAIN")
        self.modality = "CT_BODY"

        self.hf_repo_id = "nvidia/NV-Segment-CTMR"

        # Pinned to a commit rather than tracking main: the repo publishes no
        # tags, and an unpinned download would silently swap the weights (and
        # the bundle's pipeline code, which is imported and executed here)
        # whenever upstream pushes. Bump deliberately after re-testing.
        self.hf_revision = "4fb8b4a6b2532be9f1c449a3726fe5440ab4213a"

        # model.safetensors is deliberately excluded: it holds the same weights
        # as model.pt under the raw MONAI keys (no 'network.' prefix), so it is
        # unusable here, and pulling both would double the ~872 MB download.
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
            "  NON-COMMERCIAL LICENSE\n"
            "  --------------------------------------------------------------\n"
            "  NV-Segment-CTMR weights are released under the NVIDIA OneWay\n"
            "  Non-Commercial License: academic research use ONLY. This is\n"
            "  more restrictive than the rest of PhysioTwin4D. Commercial use\n"
            "  requires a different model (e.g. NV-Segment-CT).\n"
            "  https://huggingface.co/nvidia/NV-Segment-CTMR\n"
            "  =============================================================="
        )

        # NV-Segment-CTMR class indices, grouped by anatomy. Ids marked
        # "(deprecated)" in the model's label_dict.json are omitted: the bundle
        # rejects them as prompts and never emits them.
        for group_name, organs in (
            (
                "heart",
                {
                    108: "atrial_appendage_left",
                    115: "heart",
                    149: "atrium_left",
                    151: "ventricle_left",
                    152: "ventricle_right",
                    153: "atrium_right",
                    154: "ventricle_myocardium_left",
                    169: "heart_tissue",
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
                    170: "celiac_trunk",
                    171: "pulmonary_artery",
                },
            ),
            (
                "lung",
                {
                    20: "lung",
                    23: "lung_tumor",
                    28: "lung_upper_lobe_left",
                    29: "lung_lower_lobe_left",
                    30: "lung_upper_lobe_right",
                    31: "lung_middle_lobe_right",
                    32: "lung_lower_lobe_right",
                    132: "airway",
                    135: "lung_left",
                    136: "lung_right",
                },
            ),
            (
                "bone",
                {
                    21: "bone",
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
                    134: "intervertebral_discs",
                    146: "vertebrae",
                    196: "ethmoid_bone_left",
                    197: "ethmoid_bone_right",
                    200: "mandible_left",
                    201: "mandible_right",
                    206: "mastoid_left",
                    207: "mastoid_right",
                    208: "temporomandibular_joint_left",
                    209: "temporomandibular_joint_right",
                },
            ),
            (
                "soft_tissue",
                {
                    1: "liver",
                    2: "kidney",
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
                    18: "rectum",
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
                    147: "prostate_transitional_zone",
                    148: "prostate_peripheral_zone",
                    150: "white_matter_hyperintensity",
                    156: "muscles",
                    157: "fat",
                    158: "abdominal_tissue",
                    159: "mediastinal_tissue",
                    160: "gonads",
                    161: "uterocervix",
                    163: "breast_left",
                    164: "breast_right",
                    165: "thyroid_left",
                    166: "thyroid_right",
                    167: "thymus",
                    168: "skin",
                    172: "cheek_left",
                    173: "cheek_right",
                    174: "eyeball_left",
                    175: "eyeball_right",
                    176: "brain_tumor",
                    177: "chiasm",
                    178: "temporal_lobe_left",
                    179: "temporal_lobe_right",
                    180: "eye_left",
                    181: "eye_right",
                    182: "lens_left",
                    183: "lens_right",
                    184: "optic_nerve_left",
                    185: "optic_nerve_right",
                    186: "middle_ear_left",
                    187: "middle_ear_right",
                    188: "internal_auditory_canal_left",
                    189: "internal_auditory_canal_right",
                    190: "tympanic_cavity_left",
                    191: "tympanic_cavity_right",
                    192: "vestibular_semicircular_canals_left",
                    193: "vestibular_semicircular_canals_right",
                    194: "cochlea_left",
                    195: "cochlea_right",
                    198: "pituitary",
                    199: "oral_cavity",
                    202: "submandibular_left",
                    203: "submandibular_right",
                    204: "parotid_left",
                    205: "parotid_right",
                    210: "larynx",
                    211: "larynx_glottic",
                    212: "larynx_supraglot",
                    213: "pharynx_constrictor",
                },
            ),
            (
                "brain_parcellation",
                {
                    214: "3rd_ventricle",
                    215: "4th_ventricle",
                    216: "accumbens_area_right",
                    217: "accumbens_area_left",
                    218: "amygdala_right",
                    219: "amygdala_left",
                    220: "brain_stem",
                    221: "caudate_right",
                    222: "caudate_left",
                    223: "cerebellum_exterior_right",
                    224: "cerebellum_exterior_left",
                    225: "cerebellum_white_matter_right",
                    226: "cerebellum_white_matter_left",
                    227: "cerebral_white_matter_right",
                    228: "cerebral_white_matter_left",
                    229: "hippocampus_right",
                    230: "hippocampus_left",
                    231: "inf_lat_vent_right",
                    232: "inf_lat_vent_left",
                    233: "lateral_ventricle_right",
                    234: "lateral_ventricle_left",
                    235: "pallidum_right",
                    236: "pallidum_left",
                    237: "putamen_right",
                    238: "putamen_left",
                    239: "thalamus_proper_right",
                    240: "thalamus_proper_left",
                    241: "ventral_dc_right",
                    242: "ventral_dc_left",
                    243: "cerebellar_vermal_lobules_i_v",
                    244: "cerebellar_vermal_lobules_vi_vii",
                    245: "cerebellar_vermal_lobules_viii_x",
                    246: "basal_forebrain_left",
                    247: "basal_forebrain_right",
                    248: "acgg_anterior_cingulate_gyrus_right",
                    249: "acgg_anterior_cingulate_gyrus_left",
                    250: "ains_anterior_insula_right",
                    251: "ains_anterior_insula_left",
                    252: "aorg_anterior_orbital_gyrus_right",
                    253: "aorg_anterior_orbital_gyrus_left",
                    254: "ang_angular_gyrus_right",
                    255: "ang_angular_gyrus_left",
                    256: "calc_calcarine_cortex_right",
                    257: "calc_calcarine_cortex_left",
                    258: "co_central_operculum_right",
                    259: "co_central_operculum_left",
                    260: "cun_cuneus_right",
                    261: "cun_cuneus_left",
                    262: "ent_entorhinal_area_right",
                    263: "ent_entorhinal_area_left",
                    264: "fo_frontal_operculum_right",
                    265: "fo_frontal_operculum_left",
                    266: "frp_frontal_pole_right",
                    267: "frp_frontal_pole_left",
                    268: "fug_fusiform_gyrus_right",
                    269: "fug_fusiform_gyrus_left",
                    270: "gre_gyrus_rectus_right",
                    271: "gre_gyrus_rectus_left",
                    272: "iog_inferior_occipital_gyrus_right",
                    273: "iog_inferior_occipital_gyrus_left",
                    274: "itg_inferior_temporal_gyrus_right",
                    275: "itg_inferior_temporal_gyrus_left",
                    276: "lig_lingual_gyrus_right",
                    277: "lig_lingual_gyrus_left",
                    278: "lorg_lateral_orbital_gyrus_right",
                    279: "lorg_lateral_orbital_gyrus_left",
                    280: "mcgg_middle_cingulate_gyrus_right",
                    281: "mcgg_middle_cingulate_gyrus_left",
                    282: "mfc_medial_frontal_cortex_right",
                    283: "mfc_medial_frontal_cortex_left",
                    284: "mfg_middle_frontal_gyrus_right",
                    285: "mfg_middle_frontal_gyrus_left",
                    286: "mog_middle_occipital_gyrus_right",
                    287: "mog_middle_occipital_gyrus_left",
                    288: "morg_medial_orbital_gyrus_right",
                    289: "morg_medial_orbital_gyrus_left",
                    290: "mpog_postcentral_gyrus_right",
                    291: "mpog_postcentral_gyrus_left",
                    292: "mprg_precentral_gyrus_right",
                    293: "mprg_precentral_gyrus_left",
                    294: "msfg_superior_frontal_gyrus_right",
                    295: "msfg_superior_frontal_gyrus_left",
                    296: "mtg_middle_temporal_gyrus_right",
                    297: "mtg_middle_temporal_gyrus_left",
                    298: "ocp_occipital_pole_right",
                    299: "ocp_occipital_pole_left",
                    300: "ofug_occipital_fusiform_gyrus_right",
                    301: "ofug_occipital_fusiform_gyrus_left",
                    302: "opifg_opercular_part_of_the_ifg_right",
                    303: "opifg_opercular_part_of_the_ifg_left",
                    304: "orifg_orbital_part_of_the_ifg_right",
                    305: "orifg_orbital_part_of_the_ifg_left",
                    306: "pcgg_posterior_cingulate_gyrus_right",
                    307: "pcgg_posterior_cingulate_gyrus_left",
                    308: "pcu_precuneus_right",
                    309: "pcu_precuneus_left",
                    310: "phg_parahippocampal_gyrus_right",
                    311: "phg_parahippocampal_gyrus_left",
                    312: "pins_posterior_insula_right",
                    313: "pins_posterior_insula_left",
                    314: "po_parietal_operculum_right",
                    315: "po_parietal_operculum_left",
                    316: "pog_postcentral_gyrus_right",
                    317: "pog_postcentral_gyrus_left",
                    318: "porg_posterior_orbital_gyrus_right",
                    319: "porg_posterior_orbital_gyrus_left",
                    320: "pp_planum_polare_right",
                    321: "pp_planum_polare_left",
                    322: "prg_precentral_gyrus_right",
                    323: "prg_precentral_gyrus_left",
                    324: "pt_planum_temporale_right",
                    325: "pt_planum_temporale_left",
                    326: "sca_subcallosal_area_right",
                    327: "sca_subcallosal_area_left",
                    328: "sfg_superior_frontal_gyrus_right",
                    329: "sfg_superior_frontal_gyrus_left",
                    330: "smc_supplementary_motor_cortex_right",
                    331: "smc_supplementary_motor_cortex_left",
                    332: "smg_supramarginal_gyrus_right",
                    333: "smg_supramarginal_gyrus_left",
                    334: "sog_superior_occipital_gyrus_right",
                    335: "sog_superior_occipital_gyrus_left",
                    336: "spl_superior_parietal_lobule_right",
                    337: "spl_superior_parietal_lobule_left",
                    338: "stg_superior_temporal_gyrus_right",
                    339: "stg_superior_temporal_gyrus_left",
                    340: "tmp_temporal_pole_right",
                    341: "tmp_temporal_pole_left",
                    342: "trifg_triangular_part_of_the_ifg_right",
                    343: "trifg_triangular_part_of_the_ifg_left",
                    344: "ttg_transverse_temporal_gyrus_right",
                    345: "ttg_transverse_temporal_gyrus_left",
                },
            ),
        ):
            for label_id, organ_name in organs.items():
                self.taxonomy.add_organ(group_name, label_id, organ_name)

        self._finalize_other_group(range(1, 346))

        self._snapshot_dir: Optional[str] = None
        self._pipeline: Optional[Any] = None

    def set_modality(self, modality: str) -> None:
        """Set the modality whose predefined class list the model segments.

        Args:
            modality (str): One of :attr:`modalities`. ``CT_BODY`` segments the
                117-class CT set, ``MRI_BODY`` the 50-class body MR set, and
                ``MRI_BRAIN`` the 132-class LUMIR brain parcellation.

        Raises:
            ValueError: If *modality* is not one of :attr:`modalities`.

        Note:
            ``MRI_BRAIN`` expects a T1 volume that has already been
            skull-stripped and affinely aligned to the LUMIR template. This
            class does not perform that preprocessing.

        Example:
            >>> segmenter.set_modality("MRI_BODY")
        """
        if modality not in self.modalities:
            raise ValueError(
                f"Unknown modality: {modality}. "
                f"Must be one of: {', '.join(self.modalities)}."
            )
        self.modality = modality

    def _ensure_model(self) -> str:
        """Download the NV-Segment-CTMR bundle if needed and return its path.

        The snapshot is fetched once per instance and cached on disk by
        ``huggingface_hub``, so repeated calls are cheap. Logs
        :attr:`license_warning` on the first call, before the weights are
        obtained.

        Returns:
            str: Local directory holding the downloaded bundle.
        """
        if self._snapshot_dir is None:
            from huggingface_hub import snapshot_download  # noqa: PLC0415

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
        calls (modality is passed per input), so it is built once per instance
        and reused for every subsequent image or timepoint.

        Returns:
            Any: The bundle's ``VISTA3DPipeline`` on the current CUDA device.
        """
        if self._pipeline is None:
            snapshot_dir = self._ensure_model()

            # The bundle ships hugging_face_pipeline / vista3d_pipeline as
            # top-level modules inside the snapshot rather than as an installed
            # package, so the snapshot directory has to be importable.
            if snapshot_dir not in sys.path:
                sys.path.insert(0, snapshot_dir)

            import torch  # noqa: PLC0415
            from vista3d_config import VISTA3DConfig  # noqa: PLC0415
            from vista3d_model import VISTA3DModel  # noqa: PLC0415
            from vista3d_pipeline import VISTA3DPipeline  # noqa: PLC0415

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
        """Run NV-Segment-CTMR on the preprocessed image and return the result.

        The model's Hugging Face pipeline reads and writes NIfTI files, so the
        image is written to a temporary file and the prediction read back with
        ITK. That round trip also handles the coordinate-system conversion
        between ITK (LPS) and the bundle's internal RAS orientation.

        The bundle inverts its own preprocessing before saving, so the
        prediction is returned on the same grid as *preprocessed_image*.

        Args:
            preprocessed_image (itk.image): The preprocessed CT or MR image
                with isotropic spacing

        Returns:
            itk.image: The segmentation labelmap with NV-Segment-CTMR class
                indices, as ``uint16``.

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

            self.log_info("Running NV-Segment-CTMR (%s)", self.modality)
            pipeline(
                [{"image": in_file, "modality": self.modality}],
                output_dir=out_dir,
            )

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
                    f"NV-Segment-CTMR produced {len(out_files)} outputs in "
                    f"{out_dir}, expected 1."
                )

            labelmap_arr = itk.array_from_image(itk.imread(out_files[0])).astype(
                np.uint16
            )

        # The bundle's postprocessing maps unpredicted voxels to 255 via
        # nan_to_num(nan=255), not to 0. 255 is a real class only in the brain
        # parcellation, so it can be cleared for the body modalities.
        if self.modality == "MRI_BRAIN":
            self.log_warning(
                "MRI_BRAIN output leaves id 255 ambiguous: it is both the "
                "model's unpredicted-voxel sentinel and %r",
                self.taxonomy.labels_in_group("brain_parcellation")[255],
            )
        else:
            labelmap_arr[labelmap_arr == 255] = 0

        labelmap_image = itk.image_from_array(labelmap_arr)
        labelmap_image.CopyInformation(preprocessed_image)

        return labelmap_image
