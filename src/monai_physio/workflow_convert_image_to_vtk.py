"""Workflow for segmenting a CT image and converting anatomy groups to VTK surfaces.

The workflow segments a 3D CT image using a chosen backend, then extracts one
VTP surface per non-empty anatomy group.  Each output surface carries anatomy
metadata and solid color from :class:`USDAnatomyTools` as field and cell data
so that downstream tools (PyVista, Paraview, USD pipeline) can use them
directly.

Typical usage::

    import itk
    from monai_physio import (
        SegmentChestTotalSegmentatorWithContrast,
        WorkflowConvertImageToVTK,
    )

    ct = itk.imread("chest_ct.nii.gz")
    segmenter = SegmentChestTotalSegmentatorWithContrast()
    workflow = WorkflowConvertImageToVTK(segmentation_method=segmenter)
    result = workflow.process(ct, surface_reduction_rate=0.5)

    # Combined single-file output (default)
    ContourTools.save_combined_surfaces(result["surfaces"], "./out/patient.vtp")

    # Per-group split output
    ContourTools.save_surfaces(result["surfaces"], "./out", prefix="patient")

    # Per-label split output (one VTP per individual anatomical structure)
    result = workflow.process(ct, extract_label_surfaces=True)
    ContourTools.save_surfaces(result["label_surfaces"], "./out", prefix="patient")
"""

import logging
from typing import Any, Optional

import itk
import numpy as np
import pyvista as pv

from .contour_tools import ContourTools
from .monai_physio_base import MONAIPhysioBase
from .segment_anatomy_base import SegmentAnatomyBase
from .segment_chest_total_segmentator_with_contrast import (
    SegmentChestTotalSegmentatorWithContrast,
)
from .usd_anatomy_tools import USDAnatomyTools


class WorkflowConvertImageToVTK(MONAIPhysioBase):
    """Segment a CT image and produce per-anatomy-group VTK surfaces.

    ``segmentation_method`` accepts a pre-configured
    :class:`SegmentAnatomyBase` instance (e.g. :class:`SegmentChestTotalSegmentator`,
    :class:`SegmentChestTotalSegmentatorWithContrast` for contrast-enhanced
    studies, :class:`SegmentHeartSimpleware`, or
    :class:`SegmentHeartSimplewareTrimmedBranches` for cardiac-only
    segmentation with pulmonary/great-vessel branches trimmed). Defaults to
    a new :class:`SegmentChestTotalSegmentator` when omitted.

    **Output anatomy groups**

    Determined by the active segmenter's :attr:`SegmentAnatomyBase.taxonomy`
    (see :attr:`anatomy_groups`).  Groups that are empty after segmentation
    are silently skipped.  Pass ``extract_label_surfaces=True`` to
    :meth:`process` to additionally extract one surface per individual
    structure (label) within each group, e.g. ``left_ventricle`` separately
    from ``right_ventricle`` within ``heart``.

    **VTK object annotation**

    Each :class:`pyvista.PolyData` surface returned by :meth:`process` carries:

    - ``field_data['AnatomyGroup']`` — anatomy group name, e.g. ``'heart'``.
    - ``field_data['SegmentationLabelNames']`` — individual structure names within the
      group (e.g. ``['left_ventricle', 'right_ventricle', …]``).
    - ``field_data['SegmentationLabelIds']`` — corresponding integer label IDs.
    - ``field_data['AnatomyColor']`` — RGB float color from :class:`USDAnatomyTools`.
    - ``cell_data['Color']`` — RGBA uint8 array (n_cells × 4) for direct VTK rendering.

    **I/O contract**

    :meth:`process` performs *no* file I/O.  Use
    :class:`ContourTools`'s static helpers
    :meth:`ContourTools.save_surfaces` and
    :meth:`ContourTools.save_combined_surfaces` — or the CLI
    ``monai-physio-convert-image-to-vtk`` — to write results to disk.
    """

    def __init__(
        self,
        segmentation_method: Optional[SegmentAnatomyBase] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """Initialize the workflow.

        Args:
            segmentation_method: Segmentation backend instance. Defaults to
                a new :class:`SegmentChestTotalSegmentatorWithContrast` when None.
            log_level: Logging level.  Default: ``logging.INFO``.

        Raises:
            TypeError: If segmentation_method is neither None nor a
                SegmentAnatomyBase instance.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        if segmentation_method is None:
            segmentation_method = SegmentChestTotalSegmentatorWithContrast(
                log_level=log_level
            )
        elif not isinstance(segmentation_method, SegmentAnatomyBase):
            raise TypeError(
                "segmentation_method must be a SegmentAnatomyBase instance or None"
            )
        self._segmenter: SegmentAnatomyBase = segmentation_method
        self._contour_tools: ContourTools = ContourTools(log_level=log_level)

        #: Anatomy group names registered by the active segmenter's taxonomy,
        #: in the order they were first added.
        self.anatomy_groups: tuple[str, ...] = tuple(
            self._segmenter.taxonomy.group_names()
        )

        # Build anatomy-group → RGB color from USDAnatomyTools.
        # USDAnatomyTools sets up its color dicts entirely in __init__ without
        # accessing the stage, so stage=None is safe for this lookup-only use.
        _anatomy_tools = USDAnatomyTools(stage=None, log_level=log_level)
        supported_types = set(_anatomy_tools.get_anatomy_types())
        self._anatomy_color_map: dict[str, tuple[float, float, float]] = {
            group: _anatomy_tools.get_anatomy_diffuse_color(group)
            for group in self.anatomy_groups
            if group in supported_types
        }

    # ─────────────────────────── Internal helpers ──────────────────────────

    def _get_label_info_for_group(self, group: str) -> tuple[list[str], list[int]]:
        """Return ``(label_names, label_ids)`` for *group* from the active segmenter.

        Reads the segmenter's :class:`AnatomyTaxonomy`. Returns empty lists if
        the group is not present (e.g. HeartSimpleware does not register
        lung/bone).
        """
        mask_ids = self._segmenter.taxonomy.labels_in_group(group)
        return list(mask_ids.values()), list(mask_ids.keys())

    @staticmethod
    def _annotate(
        vtk_obj: pv.DataSet,
        group: str,
        label_names: list[str],
        label_ids: list[int],
        color_rgb: tuple[float, float, float],
    ) -> None:
        """Attach anatomy metadata and solid RGBA color to a VTK object **in-place**.

        Sets:

        - ``field_data['AnatomyGroup']`` — group name.
        - ``field_data['SegmentationLabelNames']`` — individual label names.
        - ``field_data['SegmentationLabelIds']`` — integer label IDs (int32).
        - ``field_data['AnatomyColor']`` — RGB float32 color.
        - ``cell_data['Color']`` — RGBA uint8 solid color (n_cells × 4).
        """
        vtk_obj.field_data["AnatomyGroup"] = np.array([group])
        vtk_obj.field_data["SegmentationLabelNames"] = np.array(
            label_names if label_names else [group]
        )
        vtk_obj.field_data["SegmentationLabelIds"] = np.array(label_ids, dtype=np.int32)
        vtk_obj.field_data["AnatomyColor"] = np.array(color_rgb, dtype=np.float32)

        r, g, b = color_rgb
        rgba = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        if vtk_obj.n_cells > 0:
            vtk_obj.cell_data["Color"] = np.tile(rgba, (vtk_obj.n_cells, 1))

    def _extract_surface(self, mask_image: Any) -> Optional[pv.PolyData]:
        """Extract a smoothed triangulated surface (VTP) from a binary mask image.

        Delegates to :meth:`ContourTools.extract_contours`.

        Returns:
            Smoothed :class:`pyvista.PolyData`, or ``None`` if the mask is empty.
        """
        arr = itk.GetArrayFromImage(mask_image)
        if int(arr.sum()) == 0:
            return None
        return self._contour_tools.extract_contours(mask_image)

    def _extract_label_surface(
        self, labelmap_image: Any, labelmap_arr: np.ndarray, label_id: int
    ) -> Optional[pv.PolyData]:
        """Extract a smoothed triangulated surface for one individual label.

        Isolates *label_id* out of *labelmap_arr* into its own binary mask
        before delegating to :meth:`ContourTools.extract_contours`.

        Args:
            labelmap_image: Source image, used only for ``CopyInformation``
                (origin/spacing/direction) on the isolated label mask.
            labelmap_arr: ``labelmap_image``'s voxel array. Callers extracting
                multiple labels from the same labelmap (as :meth:`process`
                does) should compute this once — e.g. via
                ``itk.GetArrayViewFromImage`` — and pass the same array to
                every call, rather than re-deriving it per label.
            label_id: Integer label id to isolate.

        Returns:
            Smoothed :class:`pyvista.PolyData`, or ``None`` if *label_id* has
            no voxels in *labelmap_arr*.
        """
        label_mask_arr = (labelmap_arr == label_id).astype(np.uint8)
        if int(label_mask_arr.sum()) == 0:
            return None
        label_mask = itk.GetImageFromArray(label_mask_arr)
        label_mask.CopyInformation(labelmap_image)
        return self._contour_tools.extract_contours(label_mask)

    # ─────────────────────────── Main workflow ─────────────────────────────

    def process(
        self,
        input_image: Any,
        anatomy_groups: Optional[list[str]] = None,
        surface_reduction_rate: float = 0.0,
        extract_label_surfaces: bool = False,
    ) -> dict[str, Any]:
        """Segment the CT image and extract per-anatomy-group VTK surfaces.

        Args:
            input_image: Input 3D CT image (``itk.Image``).
            anatomy_groups: Subset of anatomy groups to process.  ``None`` (default)
                processes all non-empty groups.  Valid names are given by
                :attr:`anatomy_groups`, derived from the active segmenter's
                taxonomy.
            surface_reduction_rate: Fraction in ``[0, 1)`` of surface
                triangles to remove via ``decimate_pro(surface_reduction_rate,
                preserve_topology=True)``.  ``0.0`` (default) skips decimation.
                Applied to both group and (when requested) label surfaces.
            extract_label_surfaces: When ``True``, also extract one surface per
                individual anatomical structure (label) within each processed
                group — e.g. ``left_ventricle`` and ``right_ventricle``
                separately within the ``heart`` group — in addition to the
                per-group surfaces.  ``False`` (default) skips this and leaves
                ``'label_surfaces'`` empty.

        Returns:
            ``dict`` with the following keys:

            - ``'surfaces'`` — ``dict[str, pv.PolyData]``: smoothed surface per group.
            - ``'label_surfaces'`` — ``dict[str, pv.PolyData]``: smoothed surface per
              individual label, populated only when *extract_label_surfaces* is True.
            - ``'labelmap'`` — ``itk.Image``: detailed per-structure segmentation
              labelmap from the segmenter.
            - ``'segmentation_masks'`` — ``dict[str, itk.Image]``: per-group binary
              masks used to produce the VTK objects.

        Raises:
            ValueError: If any name in *anatomy_groups* is invalid.
        """
        self.log_section("STARTING IMAGE TO VTK WORKFLOW")

        # Validate requested groups
        if anatomy_groups is not None:
            invalid = [g for g in anatomy_groups if g not in self.anatomy_groups]
            if invalid:
                raise ValueError(
                    f"Unknown anatomy groups: {invalid}. "
                    f"Valid: {list(self.anatomy_groups)}"
                )
            groups_to_process: list[str] = list(anatomy_groups)
        else:
            groups_to_process = list(self.anatomy_groups)

        # Run segmenter
        self.log_info("Running segmenter: %s", type(self._segmenter).__name__)

        self.log_section("Running segmentation")
        seg_result: dict[str, Any] = self._segmenter.segment(input_image)

        # A zero-copy view, computed once and shared across every label
        # surface extracted below so a multi-group/multi-label run doesn't
        # repeatedly re-copy the (often large) labelmap volume.
        labelmap_arr = itk.GetArrayViewFromImage(seg_result["labelmap"])

        # Extract VTK objects per anatomy group
        self.log_section("Extracting VTK objects")
        surfaces: dict[str, pv.PolyData] = {}
        label_surfaces: dict[str, pv.PolyData] = {}
        seg_masks: dict[str, Any] = {}

        for group in groups_to_process:
            if group not in seg_result:
                self.log_warning(
                    "Group %s absent from segmentation result — skipping", group
                )
                continue

            mask_image = seg_result[group]
            if int(itk.GetArrayFromImage(mask_image).sum()) == 0:
                self.log_info("Group %s is empty — skipping", group)
                continue

            self.log_info("Processing anatomy group: %s", group)
            seg_masks[group] = mask_image

            label_names, label_ids = self._get_label_info_for_group(group)
            color = self._anatomy_color_map.get(group, (0.7, 0.7, 0.7))

            self.log_info("  Extracting surface for: %s", group)
            base_surface = self._extract_surface(mask_image)
            if base_surface is None:
                continue

            export_surface = base_surface
            if surface_reduction_rate > 0.0:
                export_surface = export_surface.decimate_pro(
                    surface_reduction_rate, preserve_topology=True
                )
            self._annotate(export_surface, group, label_names, label_ids, color)
            surfaces[group] = export_surface

            if extract_label_surfaces:
                self.log_info("  Extracting label surfaces for: %s", group)
                for label_id, label_name in zip(label_ids, label_names, strict=True):
                    label_surface = self._extract_label_surface(
                        seg_result["labelmap"], labelmap_arr, label_id
                    )
                    if label_surface is None:
                        continue
                    if surface_reduction_rate > 0.0:
                        label_surface = label_surface.decimate_pro(
                            surface_reduction_rate, preserve_topology=True
                        )
                    self._annotate(
                        label_surface, group, [label_name], [label_id], color
                    )
                    label_surfaces[label_name] = label_surface

        self.log_section("IMAGE TO VTK WORKFLOW COMPLETE")
        self.log_info("Surfaces extracted:       %d", len(surfaces))
        self.log_info("Label surfaces extracted: %d", len(label_surfaces))

        return {
            "surfaces": surfaces,
            "label_surfaces": label_surfaces,
            "labelmap": seg_result["labelmap"],
            "segmentation_masks": seg_masks,
        }
