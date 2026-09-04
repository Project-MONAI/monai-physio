"""
This module contains the USDAnatomyTools class, which is used to enhance
the anatomy meshes in a USD file.

Extensibility
-------------
The default OmniSurface look for each anatomy group/organ lives in the
module-level :data:`DEFAULT_RENDER_PARAMS` dict. A new segmenter that
introduces a new group (e.g. ``"brain"``, ``"tumor"``) can register a
matching look in one of three ways:

1. **Globally**, before instantiating any ``USDAnatomyTools``::

       from monai_physio.usd_anatomy_tools import DEFAULT_RENDER_PARAMS
       DEFAULT_RENDER_PARAMS["brain"] = {"name": "Brain", ...}

   Every subsequent ``USDAnatomyTools`` instance picks up the new entry.

2. **Per-instance**, after construction::

       tools = USDAnatomyTools(stage)
       tools.render_params["brain"] = {"name": "Brain", ...}

3. **By subclassing**, overriding ``__init__`` to populate
   ``self.render_params`` with project-specific defaults.

Group lookup falls back to ``render_params["other"]`` when a group has no
registered entry, so any group present in the segmenter's
:class:`monai_physio.AnatomyTaxonomy` will still render *something*.
"""

import logging
from typing import Any, Mapping, Optional

from pxr import Sdf, UsdGeom, UsdShade

from .monai_physio_base import MONAIPhysioBase

# Brain grey matter: cortical gyri and deep nuclei alike. Shared by the
# brain_parcellation group entry and the "basal_forebrain" override below,
# which exists only to keep the basal forebrain off the whole-organ "brain"
# look ("brain" is a substring of "forebrain").
_GREY_MATTER_RENDER_PARAMS: dict[str, Any] = {
    "name": "Grey_Matter",
    "enable_diffuse_transmission": True,
    "diffuse_reflection_weight": 0.17,
    "diffuse_reflection_color": (0.7, 0.56, 0.55),
    "diffuse_reflection_roughness": 0.55,
    "metalness": 0.0,
    "specular_reflection_weight": 0.35,
    "specular_reflection_roughness": 0.55,
    "subsurface_transmission_color": (0.85, 0.7, 0.68),
    "subsurface_scattering_color": (0.85, 0.7, 0.68),
    "subsurface_weight": 0.09,
    "subsurface_scale": 0.25,
    "coat_weight": 0.08,
}

# CSF-filled ventricular spaces. Shared by the four ventricle override keys
# below rather than repeated inline: the ventricles are one tissue (clear
# cerebrospinal fluid) split across labels, and sharing the dict also keeps
# them on a single "CSF" OmniSurface material in the exported stage.
_CSF_RENDER_PARAMS: dict[str, Any] = {
    "name": "CSF",
    "enable_diffuse_transmission": True,
    "diffuse_reflection_weight": 0.05,
    "diffuse_reflection_color": (0.72, 0.84, 0.9),
    "diffuse_reflection_roughness": 0.15,
    "metalness": 0.0,
    "specular_reflection_weight": 0.7,
    "specular_reflection_roughness": 0.1,
    "subsurface_transmission_color": (0.8, 0.92, 0.97),
    "subsurface_scattering_color": (0.8, 0.92, 0.97),
    "subsurface_weight": 0.3,
    "subsurface_scale": 3.0,
    "coat_weight": 0.3,
}

# Default OmniSurface render parameters keyed by group name (matching
# :class:`monai_physio.AnatomyTaxonomy.group_names`) and by organ-level
# overrides (e.g. ``liver``, ``spleen``, ``kidney``). ``enhance_meshes``
# consults the organ-level entries first (matching an override key when it is
# a substring of the organ name, so ``kidney`` covers ``kidney_left`` and
# ``kidney_right``), then falls back to the containing group's entry, and
# finally to ``"other"``. Module-level so CLIs and tests can enumerate the
# supported types without constructing a USD stage.
DEFAULT_RENDER_PARAMS: dict[str, dict[str, Any]] = {
    "heart": {
        "name": "Heart",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.2, 0.01, 0.01),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.8, 0.4, 0.4),
        "subsurface_scattering_color": (0.8, 0.4, 0.4),
        "subsurface_weight": 0.1,
        "subsurface_scale": 2.0,
        "coat_weight": 0.0,
    },
    "lung": {
        "name": "Lung",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.125,
        "diffuse_reflection_color": (0.34, 0.0, 0.0),
        "diffuse_reflection_roughness": 0.6,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.9, 0.7, 0.7),
        "subsurface_scattering_color": (0.9, 0.7, 0.7),
        "subsurface_weight": 0.1,
        "subsurface_scale": 0.2,
        "coat_weight": 0.0,
    },
    "bone": {
        "name": "Bone",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.2,
        "diffuse_reflection_color": (0.8, 0.8, 0.9),
        "diffuse_reflection_roughness": 0.3,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.95, 0.9, 0.7),
        "subsurface_scattering_color": (0.95, 0.9, 0.7),
        "subsurface_weight": 0.03,
        "subsurface_scale": 0.05,
        "coat_weight": 0.0,
    },
    "major_vessels": {
        "name": "Major_Vessels",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.35,
        "diffuse_reflection_color": (0.2, 0.01, 0.01),
        "diffuse_reflection_roughness": 0.3,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.7, 0.15, 0.18),
        "subsurface_scattering_color": (0.7, 0.15, 0.18),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.22,
        "coat_weight": 0.12,
    },
    "contrast": {
        "name": "Contrast",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.2, 0.01, 0.01),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.9, 0.4, 0.4),
        "subsurface_scattering_color": (0.9, 0.4, 0.4),
        "subsurface_weight": 0.1,
        "subsurface_scale": 2.0,
        "coat_weight": 0.0,
    },
    "soft_tissue": {
        "name": "Soft_Tissue",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.17,
        "diffuse_reflection_color": (0.7, 0.5, 0.4),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.95, 0.9, 0.9),
        "subsurface_scattering_color": (0.95, 0.9, 0.9),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.3,
        "coat_weight": 0.12,
    },
    # Grey matter is the default for the brain_parcellation group of
    # SegmentNVSegmentCTMRI: most of its labels are cortical gyri or deep grey
    # nuclei (caudate, putamen, thalamus, amygdala, hippocampus), which in a
    # fresh specimen all share the same pink-grey, matte, weakly translucent
    # look. Only the tissues that genuinely differ get overrides below.
    "brain_parcellation": _GREY_MATTER_RENDER_PARAMS,
    "other": {
        "name": "Other",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.1,
        "diffuse_reflection_color": (0.7, 0.5, 0.4),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.95, 0.5, 0.4),
        "subsurface_scattering_color": (0.95, 0.5, 0.4),
        "subsurface_weight": 0.1,
        "subsurface_scale": 0.3,
        "coat_weight": 0.12,
    },
    # Organ-level overrides. enhance_meshes consults these before falling
    # back to the containing group's params, matching an override key when it
    # is a substring of the organ name, so e.g. liver/spleen/kidney get their
    # dedicated look despite being in the soft_tissue group of the taxonomy.
    # The overrides below span abdominal/endocrine organs, vessels (artery vs.
    # vein), tissue types (skin/fat/muscle/cartilage), and the oxygenation-
    # coded heart chambers; when several keys are substrings of one name the
    # longest (most specific) one wins.
    "liver": {
        "name": "Liver",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.1,
        "diffuse_reflection_color": (0.16, 0.01, 0.01),
        "diffuse_reflection_roughness": 0.4,
        "metalness": 0.0,
        "specular_reflection_weight": 0.01,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.7, 0.2, 0.15),
        "subsurface_scattering_color": (0.7, 0.2, 0.15),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.15,
        "coat_weight": 0.01,
    },
    "spleen": {
        "name": "Spleen",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.1,
        "diffuse_reflection_color": (0.45, 0.08, 0.15),
        "diffuse_reflection_roughness": 0.4,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.6, 0.15, 0.22),
        "subsurface_scattering_color": (0.6, 0.15, 0.22),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.15,
        "coat_weight": 0.1,
    },
    "kidney": {
        "name": "Kidney",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.1,
        "diffuse_reflection_color": (0.45, 0.13, 0.12),
        "diffuse_reflection_roughness": 0.35,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.6, 0.18, 0.15),
        "subsurface_scattering_color": (0.6, 0.18, 0.15),
        "subsurface_weight": 0.085,
        "subsurface_scale": 0.18,
        "coat_weight": 0.1,
    },
    # Skin: tan/pink dermis. Skin is the canonical strong-subsurface tissue
    # (Jensen et al., "A Practical Model for Subsurface Light Transport",
    # SIGGRAPH 2001), so it carries the highest subsurface_weight here plus a
    # thin oily coat for the epidermal sheen. Matches organs named "*skin*".
    "skin": {
        "name": "Skin",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.2,
        "diffuse_reflection_color": (0.75, 0.52, 0.42),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.4,
        "specular_reflection_roughness": 0.45,
        "subsurface_transmission_color": (0.85, 0.5, 0.4),
        "subsurface_scattering_color": (0.9, 0.55, 0.45),
        "subsurface_weight": 0.12,
        "subsurface_scale": 0.4,
        "coat_weight": 0.15,
    },
    # Airway: pale-pink tracheobronchial mucosa. Wet mucus surface -> strong,
    # low-roughness specular plus a heavy coat for the glossy sheen. Matches
    # organs named "*airway*" (e.g. lung_airways, lung_airways_wall).
    "airway": {
        "name": "Airway",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.18,
        "diffuse_reflection_color": (0.85, 0.6, 0.58),
        "diffuse_reflection_roughness": 0.4,
        "metalness": 0.0,
        "specular_reflection_weight": 0.6,
        "specular_reflection_roughness": 0.35,
        "subsurface_transmission_color": (0.9, 0.7, 0.7),
        "subsurface_scattering_color": (0.9, 0.7, 0.7),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.15,
        "coat_weight": 0.2,
    },
    # Vein: deoxygenated blood -> dark, desaturated purplish-red (blue channel
    # lifted above green). Deoxy-hemoglobin absorbs green/red more than oxy-Hb,
    # shifting venous vessels darker and bluer than arteries. Matches organs
    # named "*vein*" (pulmonary_vein, brachiocephalic_vein, lung_veins).
    "vein": {
        "name": "Vein",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.2,
        "diffuse_reflection_color": (0.3, 0.03, 0.12),
        "diffuse_reflection_roughness": 0.4,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.4,
        "subsurface_transmission_color": (0.5, 0.1, 0.2),
        "subsurface_scattering_color": (0.5, 0.1, 0.2),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.2,
        "coat_weight": 0.12,
    },
    # Artery: oxygenated blood -> brighter, more saturated red than veins. Key
    # is the substring "arter" so it matches both "artery" and "arteries"
    # (subclavian/carotid arteries, lung_arteries, pulmonary_artery).
    "arter": {
        "name": "Artery",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.55, 0.03, 0.03),
        "diffuse_reflection_roughness": 0.35,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.4,
        "subsurface_transmission_color": (0.8, 0.2, 0.2),
        "subsurface_scattering_color": (0.8, 0.2, 0.2),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.22,
        "coat_weight": 0.12,
    },
    # Fat: adipose tissue -> pale yellow, translucent, with a greasy coat.
    # Matches organs named "*fat*" (subcutaneous, torso, intermuscular).
    "fat": {
        "name": "Fat",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.2,
        "diffuse_reflection_color": (0.9, 0.75, 0.35),
        "diffuse_reflection_roughness": 0.45,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.4,
        "subsurface_transmission_color": (0.95, 0.85, 0.5),
        "subsurface_scattering_color": (0.95, 0.85, 0.5),
        "subsurface_weight": 0.1,
        "subsurface_scale": 0.3,
        "coat_weight": 0.15,
    },
    # Muscle: skeletal muscle -> deep red-brown, matte (fibrous striations),
    # only lightly moist. Matches organs named "*muscle*" (skeletal_muscle);
    # the substring does not appear in "intermuscular_fat", so fat is
    # unaffected. Darker and less saturated than arterial blood.
    "muscle": {
        "name": "Muscle",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.18,
        "diffuse_reflection_color": (0.42, 0.08, 0.07),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.4,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.6, 0.15, 0.13),
        "subsurface_scattering_color": (0.6, 0.15, 0.13),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.18,
        "coat_weight": 0.05,
    },
    # Aorta: largest systemic artery, oxygenated -> bright saturated red, same
    # optical look as the generic "arter" override but kept as its own entry
    # because "aorta" contains no "arter" substring. Matches "aorta",
    # "highres_aorta".
    "aorta": {
        "name": "Aorta",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.55, 0.03, 0.03),
        "diffuse_reflection_roughness": 0.35,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.4,
        "subsurface_transmission_color": (0.8, 0.2, 0.2),
        "subsurface_scattering_color": (0.8, 0.2, 0.2),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.22,
        "coat_weight": 0.12,
    },
    # Vena cava: large systemic vein, deoxygenated -> dark purplish-red, same
    # optical look as the generic "vein" override but kept as its own entry
    # because "vena_cava" contains no "vein" substring. Matches
    # "superior_vena_cava", "inferior_vena_cava".
    "cava": {
        "name": "Vena_Cava",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.2,
        "diffuse_reflection_color": (0.3, 0.03, 0.12),
        "diffuse_reflection_roughness": 0.4,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.4,
        "subsurface_transmission_color": (0.5, 0.1, 0.2),
        "subsurface_scattering_color": (0.5, 0.1, 0.2),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.2,
        "coat_weight": 0.12,
    },
    # --- Abdominal / endocrine organ overrides (soft_tissue group). These
    # organs otherwise all collapse onto the generic tan soft_tissue look.
    # Colors are realistic but nudged apart so neighboring organs stay
    # distinguishable during cropped visual inspection. ---
    # Gallbladder: bile-filled -> characteristic olive/green, the single most
    # recognizable abdominal cue. Matches "gallbladder".
    "gallbladder": {
        "name": "Gallbladder",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.17,
        "diffuse_reflection_color": (0.35, 0.45, 0.15),
        "diffuse_reflection_roughness": 0.45,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.5, 0.6, 0.25),
        "subsurface_scattering_color": (0.5, 0.6, 0.25),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.25,
        "coat_weight": 0.15,
    },
    # Pancreas: lobulated -> pale salmon-tan. Matches "pancreas".
    "pancreas": {
        "name": "Pancreas",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.17,
        "diffuse_reflection_color": (0.8, 0.52, 0.44),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.4,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.9, 0.65, 0.55),
        "subsurface_scattering_color": (0.9, 0.65, 0.55),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.25,
        "coat_weight": 0.1,
    },
    # Stomach: muscular wall -> pink-red, redder than the pale bowel/pancreas.
    # Matches "stomach".
    "stomach": {
        "name": "Stomach",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.17,
        "diffuse_reflection_color": (0.72, 0.4, 0.38),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.85, 0.5, 0.45),
        "subsurface_scattering_color": (0.85, 0.5, 0.45),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.3,
        "coat_weight": 0.12,
    },
    # Brain: soft neural tissue -> light pink-grey with a little more
    # subsurface glow than firm organs. Matches "brain".
    "brain": {
        "name": "Brain",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.17,
        "diffuse_reflection_color": (0.82, 0.72, 0.68),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.4,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.9, 0.82, 0.8),
        "subsurface_scattering_color": (0.9, 0.82, 0.8),
        "subsurface_weight": 0.1,
        "subsurface_scale": 0.3,
        "coat_weight": 0.1,
    },
    # Thyroid: highly vascular gland -> deep red-brown. Matches "thyroid".
    "thyroid": {
        "name": "Thyroid",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.15,
        "diffuse_reflection_color": (0.55, 0.22, 0.2),
        "diffuse_reflection_roughness": 0.45,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.7, 0.3, 0.25),
        "subsurface_scattering_color": (0.7, 0.3, 0.25),
        "subsurface_weight": 0.085,
        "subsurface_scale": 0.2,
        "coat_weight": 0.1,
    },
    # Adrenal gland: cortex -> yellow-orange. More saturated/orange than the
    # pale yellow "fat" look so the two stay distinct. Matches "adrenal".
    "adrenal": {
        "name": "Adrenal_Gland",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.17,
        "diffuse_reflection_color": (0.82, 0.62, 0.32),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.4,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.9, 0.75, 0.5),
        "subsurface_scattering_color": (0.9, 0.75, 0.5),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.25,
        "coat_weight": 0.1,
    },
    # Urinary bladder: thin, fluid-filled wall -> pale pink, lightly wet.
    # Key "bladder" also occurs in "gallbladder", but override matching is
    # longest-first, so "gallbladder" (above) wins for the gallbladder and
    # "bladder" claims only "urinary_bladder".
    "bladder": {
        "name": "Urinary_Bladder",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.15,
        "diffuse_reflection_color": (0.8, 0.62, 0.6),
        "diffuse_reflection_roughness": 0.45,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.45,
        "subsurface_transmission_color": (0.9, 0.75, 0.72),
        "subsurface_scattering_color": (0.9, 0.75, 0.72),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.25,
        "coat_weight": 0.15,
    },
    # Gluteus muscles: skeletal muscle -> deep red-brown. Shares the "muscle"
    # look but needs its own key because the gluteus label names contain no
    # "muscle" substring. Matches "gluteus_maximus/medius/minimus_*".
    "gluteus": {
        "name": "Gluteus_Muscle",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.18,
        "diffuse_reflection_color": (0.42, 0.08, 0.07),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.4,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.6, 0.15, 0.13),
        "subsurface_scattering_color": (0.6, 0.15, 0.13),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.18,
        "coat_weight": 0.05,
    },
    # Costal cartilage: cartilage, not cortical bone -> translucent grey-white
    # with more subsurface than the opaque "bone" look. Bone-group organ, but
    # organ overrides win over the group. Matches "costal_cartilages".
    "cartilage": {
        "name": "Costal_Cartilage",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.2,
        "diffuse_reflection_color": (0.82, 0.86, 0.82),
        "diffuse_reflection_roughness": 0.35,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.85, 0.9, 0.85),
        "subsurface_scattering_color": (0.85, 0.9, 0.85),
        "subsurface_weight": 0.1,
        "subsurface_scale": 0.2,
        "coat_weight": 0.05,
    },
    # --- Heart-chamber overrides (heart group). Oxygenation-coded so cardiac
    # anatomy reads at a glance: left (systemic, oxygenated) = brighter red;
    # right (pulmonary, deoxygenated) = darker, bluer red; myocardium = deep
    # red-brown wall. Keys carry the full "_left"/"_right" suffix because the
    # two sides are not substrings of one another. ---
    "myocardium": {
        "name": "Myocardium",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.4, 0.08, 0.07),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.6, 0.25, 0.22),
        "subsurface_scattering_color": (0.6, 0.25, 0.22),
        "subsurface_weight": 0.1,
        "subsurface_scale": 1.5,
        "coat_weight": 0.0,
    },
    "atrium_left": {
        "name": "Atrium_Left",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.5, 0.06, 0.06),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.85, 0.4, 0.4),
        "subsurface_scattering_color": (0.85, 0.4, 0.4),
        "subsurface_weight": 0.1,
        "subsurface_scale": 2.0,
        "coat_weight": 0.0,
    },
    "ventricle_left": {
        "name": "Ventricle_Left",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.45, 0.03, 0.03),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.8, 0.35, 0.35),
        "subsurface_scattering_color": (0.8, 0.35, 0.35),
        "subsurface_weight": 0.1,
        "subsurface_scale": 2.0,
        "coat_weight": 0.0,
    },
    "atrium_right": {
        "name": "Atrium_Right",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.32, 0.05, 0.12),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.6, 0.25, 0.35),
        "subsurface_scattering_color": (0.6, 0.25, 0.35),
        "subsurface_weight": 0.1,
        "subsurface_scale": 2.0,
        "coat_weight": 0.0,
    },
    "ventricle_right": {
        "name": "Ventricle_Right",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.28, 0.03, 0.11),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.55, 0.2, 0.32),
        "subsurface_scattering_color": (0.55, 0.2, 0.32),
        "subsurface_weight": 0.1,
        "subsurface_scale": 2.0,
        "coat_weight": 0.0,
    },
    # Left atrial appendage: part of the left atrium -> oxygenated look. Key
    # "atrial" is not a substring of "atrium", so it picks up only
    # "atrial_appendage_left".
    "atrial": {
        "name": "Atrial_Appendage",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.5, 0.06, 0.06),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.85, 0.4, 0.4),
        "subsurface_scattering_color": (0.85, 0.4, 0.4),
        "subsurface_weight": 0.1,
        "subsurface_scale": 2.0,
        "coat_weight": 0.0,
    },
    # --- Brain-tissue overrides (brain_parcellation group of
    # SegmentNVSegmentCTMRI). The group entry above paints everything as grey
    # matter; these are the tissues whose natural gross appearance is actually
    # different from cortex, so they are the ones worth separating. ---
    # White matter: myelin -> glossy, creamy off-white, markedly brighter and
    # shinier than grey matter. Key matches "cerebral_white_matter_*" and
    # "cerebellum_white_matter_*"; being longer than "cerebell" it also wins
    # over the cerebellar-cortex override for the cerebellar white matter.
    "white_matter": {
        "name": "White_Matter",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.22,
        "diffuse_reflection_color": (0.92, 0.88, 0.8),
        "diffuse_reflection_roughness": 0.3,
        "metalness": 0.0,
        "specular_reflection_weight": 0.55,
        "specular_reflection_roughness": 0.35,
        "subsurface_transmission_color": (0.96, 0.93, 0.86),
        "subsurface_scattering_color": (0.96, 0.93, 0.86),
        "subsurface_weight": 0.1,
        "subsurface_scale": 0.35,
        "coat_weight": 0.12,
    },
    # White-matter hyperintensity is a gliotic lesion, not normal myelin, and
    # it lives in the soft_tissue group rather than brain_parcellation. It
    # needs its own key only because "white_matter" is a substring of its
    # name; the longer key wins, keeping the lesion dull and matte instead of
    # inheriting the glossy white-matter look.
    "hyperintensity": {
        "name": "White_Matter_Hyperintensity",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.18,
        "diffuse_reflection_color": (0.78, 0.76, 0.68),
        "diffuse_reflection_roughness": 0.6,
        "metalness": 0.0,
        "specular_reflection_weight": 0.25,
        "specular_reflection_roughness": 0.6,
        "subsurface_transmission_color": (0.85, 0.83, 0.75),
        "subsurface_scattering_color": (0.85, 0.83, 0.75),
        "subsurface_weight": 0.07,
        "subsurface_scale": 0.2,
        "coat_weight": 0.05,
    },
    # CSF spaces: clear fluid, so heavy subsurface transport and a wet,
    # low-roughness surface -- the most visually distinct tissue in the group.
    # Four keys because the labels share no single safe substring: "ventricle"
    # alone is shorter than the heart's "ventricle_left"/"ventricle_right"
    # overrides, which would otherwise claim "lateral_ventricle_left/right"
    # and paint the ventricles arterial red.
    "3rd_ventricle": _CSF_RENDER_PARAMS,
    "4th_ventricle": _CSF_RENDER_PARAMS,
    "lateral_ventricle": _CSF_RENDER_PARAMS,
    "inf_lat_vent": _CSF_RENDER_PARAMS,
    # Basal forebrain: grey-matter nuclei. Present only to outrank the
    # whole-organ "brain" override, which "forebrain" would otherwise match.
    "basal_forebrain": _GREY_MATTER_RENDER_PARAMS,
    # Globus pallidus: myelin-rich, so distinctly paler than the neighboring
    # putamen and caudate it is embedded in -- the one deep nucleus that reads
    # differently from the rest by eye. Matches "pallidum_left/right".
    "pallidum": {
        "name": "Pallidum",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.2,
        "diffuse_reflection_color": (0.82, 0.74, 0.68),
        "diffuse_reflection_roughness": 0.45,
        "metalness": 0.0,
        "specular_reflection_weight": 0.4,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.9, 0.84, 0.78),
        "subsurface_scattering_color": (0.9, 0.84, 0.78),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.3,
        "coat_weight": 0.1,
    },
    # Brainstem: dominated by descending/ascending fiber tracts, so it reads
    # pale like white matter rather than pink like cortex. Key is longer than
    # the whole-organ "brain" override, so it wins for "brain_stem".
    "brain_stem": {
        "name": "Brain_Stem",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.2,
        "diffuse_reflection_color": (0.86, 0.79, 0.72),
        "diffuse_reflection_roughness": 0.4,
        "metalness": 0.0,
        "specular_reflection_weight": 0.45,
        "specular_reflection_roughness": 0.45,
        "subsurface_transmission_color": (0.92, 0.86, 0.8),
        "subsurface_scattering_color": (0.92, 0.86, 0.8),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.3,
        "coat_weight": 0.1,
    },
    # Cerebellar cortex: darker and browner than cerebral cortex, and finely
    # foliated rather than smoothly gyral, so it is given a rougher, more
    # matte surface. The "cerebell" stem matches both "cerebellum_exterior_*"
    # and "cerebellar_vermal_lobules_*".
    "cerebell": {
        "name": "Cerebellar_Cortex",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.16,
        "diffuse_reflection_color": (0.62, 0.5, 0.46),
        "diffuse_reflection_roughness": 0.65,
        "metalness": 0.0,
        "specular_reflection_weight": 0.3,
        "specular_reflection_roughness": 0.6,
        "subsurface_transmission_color": (0.78, 0.64, 0.6),
        "subsurface_scattering_color": (0.78, 0.64, 0.6),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.2,
        "coat_weight": 0.06,
    },
}

# Canonical AnatomyTaxonomy group names that carry a group-level entry in
# DEFAULT_RENDER_PARAMS. Every other key is an organ-level override. Used by
# :meth:`USDAnatomyTools._resolve_render_params` to mirror ``enhance_meshes``,
# where an organ override always beats the containing group on a substring
# match (e.g. "lung_veins" -> the "vein" override, not the "lung" group).
GROUP_RENDER_KEYS: frozenset[str] = frozenset(
    {
        "heart",
        "lung",
        "bone",
        "major_vessels",
        "contrast",
        "soft_tissue",
        "brain_parcellation",
        "other",
    }
)


class USDAnatomyTools(MONAIPhysioBase):
    """Apply OmniSurface materials to anatomy mesh prims in a USD stage.

    The instance attribute :attr:`render_params` is initialized from the
    module-level :data:`DEFAULT_RENDER_PARAMS` (deep copy per instance, so
    in-place edits stay local). See the module docstring for how to add new
    groups/organs.
    """

    def __init__(self, stage: Any, log_level: int | str = logging.INFO) -> None:
        """Initialize USDAnatomyTools.

        Args:
            stage: USD stage to work with. May be ``None`` when the instance
                is only used for color look-ups (e.g. via
                :meth:`get_anatomy_diffuse_color`).
            log_level: Logging level (default: logging.INFO).
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.stage = stage
        # Per-instance copy so per-instance mutations don't leak into other
        # USDAnatomyTools instances.
        self.render_params: dict[str, dict[str, Any]] = {
            key: dict(params) for key, params in DEFAULT_RENDER_PARAMS.items()
        }

    def get_anatomy_types(self) -> list[str]:
        """Return list of registered render-param keys (groups + organ overrides)."""
        return list(self.render_params.keys())

    def _resolve_render_params(self, anatomy_type: str) -> Optional[dict[str, Any]]:
        """Resolve *anatomy_type* to a render-params entry, or ``None``.

        Matching mirrors :meth:`enhance_meshes`: an exact (case-insensitive)
        key wins first; otherwise a key that is a substring of *anatomy_type*
        matches (so ``"kidney_left"`` resolves to the ``"kidney"`` entry).
        Organ-level overrides are tried before group-level keys (so
        ``"lung_veins"`` resolves to the ``"vein"`` override, not the ``"lung"``
        group), and within each set the longest (most specific) key wins.

        Args:
            anatomy_type: A group/organ name or registered render-params key.

        Returns:
            The matching render-params dict, or ``None`` if nothing matches.
        """
        name = anatomy_type.lower()
        exact = self.render_params.get(name)
        if exact is not None:
            return exact
        override_keys = [k for k in self.render_params if k not in GROUP_RENDER_KEYS]
        group_keys = [k for k in self.render_params if k in GROUP_RENDER_KEYS]
        for keys in (override_keys, group_keys):
            for key in sorted(keys, key=len, reverse=True):
                if key in name:
                    return self.render_params[key]
        return None

    def resolve_anatomy_type(self, anatomy_type: str) -> Optional[str]:
        """Return the material name *anatomy_type* selects, or ``None``.

        Lets callers test a group/organ name before applying it, instead of
        catching the :exc:`ValueError` raised by
        :meth:`apply_anatomy_material_to_mesh`. Matching is the same as that
        method's (see :meth:`_resolve_render_params`), so ``"kidney_left"``
        returns ``"Kidney"``.

        Args:
            anatomy_type: A group/organ name or registered render-params key.

        Returns:
            The matching material name (e.g. ``"Heart"``), or ``None`` when
            nothing matches.
        """
        params = self._resolve_render_params(anatomy_type)
        return str(params["name"]) if params is not None else None

    def get_anatomy_diffuse_color(
        self, anatomy_type: str
    ) -> tuple[float, float, float]:
        """Return the diffuse reflection RGB color for the given group/organ.

        This accessor does not require a USD stage and may be called on an instance
        created with ``stage=None`` purely for color look-up purposes.

        Args:
            anatomy_type: A group/organ name or registered render-params key
                (e.g. ``"heart"``, ``"lung"``, ``"liver"``). Resolved via
                exact-then-substring matching, so ``"kidney_left"`` maps to the
                ``"kidney"`` entry (see :meth:`_resolve_render_params`).

        Returns:
            RGB tuple of floats in ``[0, 1]``.

        Raises:
            ValueError: If *anatomy_type* matches no registered entry.
        """
        params = self._resolve_render_params(anatomy_type)
        if params is None:
            raise ValueError(
                f"Unknown anatomy_type '{anatomy_type}'. "
                f"Supported: {', '.join(self.get_anatomy_types())}"
            )
        color = params["diffuse_reflection_color"]
        return (float(color[0]), float(color[1]), float(color[2]))

    def apply_anatomy_material_to_mesh(self, mesh_path: str, anatomy_type: str) -> None:
        """Apply an anatomic OmniSurface material to a single mesh prim by type.

        Args:
            mesh_path: USD path to the mesh prim (e.g. "/World/Meshes/MyMesh").
            anatomy_type: A group/organ name or registered render-params key.
                Resolved via exact-then-substring matching, so ``"kidney_left"``
                maps to the ``"kidney"`` entry (see
                :meth:`_resolve_render_params`). See :meth:`get_anatomy_types`.

        Raises:
            ValueError: If mesh_path is invalid or anatomy_type matches no entry.
        """
        params = self._resolve_render_params(anatomy_type)
        if params is None:
            raise ValueError(
                f"Unknown anatomy_type '{anatomy_type}'. "
                f"Supported: {', '.join(self.get_anatomy_types())}"
            )
        prim = self.stage.GetPrimAtPath(mesh_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim at path: {mesh_path}")
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Prim at {mesh_path} is not a Mesh")
        self.apply_anatomy_material_to_prim(prim, params)

    def apply_anatomy_material_to_prim(
        self, prim: Any, material_params: Mapping[str, Any]
    ) -> None:
        """Corrected material application with Omniverse-specific fixes"""

        # 1. Unique material path using prim's full path hierarchy
        material_path = Sdf.Path(f"/World/Looks/OmniSurface_{material_params['name']}")
        shader_path = material_path.AppendPath("Shader")

        material = UsdShade.Material.Get(self.stage, material_path)
        if not material or not material.GetPrim().IsValid():
            # prim.CreateDisplayColorAttr().Set(material_params["diffuse_reflection_color"])

            material = UsdShade.Material.Define(self.stage, material_path)
            shader = UsdShade.Shader.Define(self.stage, shader_path)

            # 2. MDL-Context Shader Definition (REQUIRED for Omniverse)
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset("OmniSurface.mdl", "mdl")
            shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")

            # 3. Set the parameters
            shader.CreateInput(
                "enable_diffuse_transmission", Sdf.ValueTypeNames.Bool
            ).Set(material_params["enable_diffuse_transmission"])
            shader.CreateInput(
                "diffuse_reflection_weight", Sdf.ValueTypeNames.Float
            ).Set(material_params["diffuse_reflection_weight"])
            shader.CreateInput(
                "diffuse_reflection_color", Sdf.ValueTypeNames.Color3f
            ).Set(material_params["diffuse_reflection_color"])
            shader.CreateInput(
                "diffuse_reflection_roughness", Sdf.ValueTypeNames.Float
            ).Set(material_params["diffuse_reflection_roughness"])
            shader.CreateInput("metalness", Sdf.ValueTypeNames.Float).Set(
                material_params["metalness"]
            )
            shader.CreateInput(
                "specular_reflection_weight", Sdf.ValueTypeNames.Float
            ).Set(material_params["specular_reflection_weight"])
            shader.CreateInput(
                "specular_reflection_roughness", Sdf.ValueTypeNames.Float
            ).Set(material_params["specular_reflection_roughness"])
            shader.CreateInput(
                "subsurface_transmission_color", Sdf.ValueTypeNames.Color3f
            ).Set(material_params["subsurface_transmission_color"])
            shader.CreateInput(
                "subsurface_scattering_color", Sdf.ValueTypeNames.Color3f
            ).Set(material_params["subsurface_scattering_color"])
            shader.CreateInput("subsurface_weight", Sdf.ValueTypeNames.Float).Set(
                material_params["subsurface_weight"]
            )
            shader.CreateInput("subsurface_scale", Sdf.ValueTypeNames.Float).Set(
                material_params["subsurface_scale"]
            )
            shader.CreateInput("coat_weight", Sdf.ValueTypeNames.Float).Set(
                material_params["coat_weight"]
            )

            # 4. Connect the shader's output to the material's surface output for the MDL render context.
            material.CreateSurfaceOutput("mdl").ConnectToSource(
                shader.ConnectableAPI(), "out"
            )
            material.CreateDisplacementOutput("mdl").ConnectToSource(
                shader.ConnectableAPI(), "out"
            )

        # 5. Bind the material to the mesh primitive.
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(material)

    def enhance_meshes(self, segmentator: Any) -> None:
        """Apply per-organ OmniSurface materials to every matching mesh prim.

        Walks the segmenter's :class:`AnatomyTaxonomy` and applies a material
        to each mesh prim whose leaf name matches an organ name in any group.
        An organ-level entry in :attr:`render_params` (e.g. ``"liver"``,
        ``"spleen"``, ``"kidney"``) takes precedence over the entry for the
        containing group. An organ-level key matches when it is a substring of
        the organ name, so ``"kidney"`` covers both ``kidney_left`` and
        ``kidney_right``; when several keys match, the longest (most specific)
        one wins.

        Anatomy grouping is performed upstream by ConvertVTKToUSD, which
        writes labeled prims under ``/World/{basename}/{type}/{label_name}``.
        This method only needs to apply materials; it does not move prims.

        Args:
            segmentator: A :class:`SegmentAnatomyBase` instance whose
                ``taxonomy`` attribute holds the group/organ structure.
        """
        taxonomy = segmentator.taxonomy

        # Organ-level override keys are every registered render-params key
        # that is neither a group name nor the "other" fallback. They match an
        # organ when the key is a substring of the organ name (e.g. "kidney"
        # matches "kidney_left"). Sorted longest-first so a more specific key
        # wins when several are substrings of the same organ name.
        group_names = set(taxonomy.group_names())
        override_keys = sorted(
            (k for k in self.render_params if k not in group_names and k != "other"),
            key=len,
            reverse=True,
        )

        # Build organ_name -> render params dict in one pass. Organ-level
        # overrides win over the containing group's params; if neither
        # applies, fall back to the "other" entry (always present in
        # DEFAULT_RENDER_PARAMS, so the lookup is safe).
        organ_params: dict[str, dict[str, Any]] = {}
        default_params: dict[str, Any] = self.render_params["other"]
        for group_name in taxonomy.group_names():
            group_params = self.render_params.get(group_name, default_params)
            for organ_name in taxonomy.labels_in_group(group_name).values():
                selected = group_params
                for key in override_keys:
                    if key in organ_name.lower():
                        selected = self.render_params[key]
                        break
                organ_params[organ_name] = selected

        for prim in self.stage.Traverse():
            mesh_prim = UsdGeom.Mesh(prim)
            if not mesh_prim:
                continue
            prim_name = prim.GetName()
            # ConvertVTKToUSD may prefix prim names with an index like
            # "frame0_<organ>"; accept the suffix as a match too.
            prim_sub_name = "_".join(prim_name.split("_")[1:])
            params = organ_params.get(prim_name) or organ_params.get(prim_sub_name)
            if params is None:
                continue
            self.apply_anatomy_material_to_prim(prim, params)
