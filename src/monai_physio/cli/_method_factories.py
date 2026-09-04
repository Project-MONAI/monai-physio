"""Shared string-to-instance factories for CLI segmentation/registration flags.

CLI scripts expose segmentation/registration backends as string choices for
usability, then build the corresponding instance via these factories before
passing it to the library's instance-based workflow API.
"""

from monai_physio import (
    RegisterImagesBase,
    RegisterImagesGreedy,
    RegisterImagesGreedyICON,
    RegisterImagesICON,
    SegmentAnatomyBase,
    SegmentChestTotalSegmentator,
    SegmentChestTotalSegmentatorWithContrast,
    SegmentHeartSimpleware,
    SegmentHeartSimplewareTrimmedBranches,
    SegmentNVSegmentCTMRI,
)

#: Segmentation backend string choices exposed by CLI flags.
SEGMENTATION_METHODS: tuple[str, ...] = (
    "ChestTotalSegmentator",
    "HeartSimpleware",
    "HeartSimplewareTrimmedBranches",
    "NVSegmentCTMR",
)

#: Registration backend string choices exposed by CLI flags.
REGISTRATION_METHODS: tuple[str, ...] = ("Greedy", "ICON", "Greedy_ICON")


def build_segmentation_method(name: str, contrast: bool = False) -> SegmentAnatomyBase:
    """Build a SegmentAnatomyBase instance for a CLI --segmentation-method choice.

    Args:
        name: One of SEGMENTATION_METHODS.
        contrast: If True, build the contrast-enhanced variant instead of the
            plain backend. Only supported for "ChestTotalSegmentator".

    Returns:
        A new, unconfigured segmentation backend instance.

    Raises:
        ValueError: If name is not one of SEGMENTATION_METHODS, or if
            contrast=True is requested for a backend that has no
            contrast-enhanced variant.
    """
    if name == "ChestTotalSegmentator":
        if contrast:
            return SegmentChestTotalSegmentatorWithContrast()
        return SegmentChestTotalSegmentator()
    if contrast:
        raise ValueError(
            f"contrast=True is not supported for segmentation method: {name}. "
            "Only ChestTotalSegmentator has a contrast-enhanced variant."
        )
    if name == "HeartSimpleware":
        return SegmentHeartSimpleware()
    if name == "HeartSimplewareTrimmedBranches":
        return SegmentHeartSimplewareTrimmedBranches()
    if name == "NVSegmentCTMR":
        return SegmentNVSegmentCTMRI()
    raise ValueError(
        f"Unknown segmentation method: {name}. "
        f"Must be one of: {', '.join(SEGMENTATION_METHODS)}."
    )


def build_registration_method(name: str) -> RegisterImagesBase:
    """Build a RegisterImagesBase instance for a CLI --registration-method choice.

    Args:
        name: One of REGISTRATION_METHODS.

    Returns:
        A new, unconfigured registration backend instance.

    Raises:
        ValueError: If name is not one of REGISTRATION_METHODS.
    """
    if name == "Greedy":
        return RegisterImagesGreedy()
    if name == "ICON":
        return RegisterImagesICON()
    if name == "Greedy_ICON":
        return RegisterImagesGreedyICON()
    raise ValueError(
        f"Unknown registration method: {name}. "
        f"Must be one of: {', '.join(REGISTRATION_METHODS)}."
    )
