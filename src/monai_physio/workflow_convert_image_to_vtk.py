"""Workflow for turning a CT image or a labelmap into anatomy VTK surfaces.

The workflow segments a 3D CT image with a chosen backend, or takes a labelmap
that backend already produced, then extracts one closed surface per anatomy
group and, optionally, per structure, and a tetrahedral mesh of each.  Each
output carries anatomy metadata and the :class:`ProcessUSDAnatomy` color as
field and cell data so that downstream tools (PyVista, Paraview, USD pipeline)
can use them directly.

Typical usage::

    import itk
    from monai_physio import (
        SegmentChestTotalSegmentatorWithContrast,
        WorkflowConvertImageToVTK,
    )

    ct = itk.imread("chest_ct.nii.gz")
    segmenter = SegmentChestTotalSegmentatorWithContrast()
    workflow = WorkflowConvertImageToVTK(segmentation_method=segmenter)
    result = workflow.process(ct, extract_label_surfaces=True)

    # Per-label surfaces in one file, per-group surfaces one file each
    WorkflowConvertImageToVTK.save_combined_surfaces(
        result["label_surfaces"], "./out/patient_surfaces.vtp"
    )
    WorkflowConvertImageToVTK.save_surfaces(result["surfaces"], "./out", prefix="patient")
"""

import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

import itk
import numpy as np
import pyvista as pv

from .monai_physio_base import MONAIPhysioBase
from .process_surface_mesh import ProcessSurfaceMesh
from .process_usd_anatomy import ProcessUSDAnatomy
from .process_volumetric_mesh import ProcessVolumetricMesh
from .segment_anatomy_base import SegmentAnatomyBase
from .segment_chest_total_segmentator_with_contrast import (
    SegmentChestTotalSegmentatorWithContrast,
)


class WorkflowConvertImageToVTK(MONAIPhysioBase):
    """Turn a CT image or a labelmap into per-group and per-label VTK surfaces.

    ``segmentation_method`` accepts a pre-configured
    :class:`SegmentAnatomyBase` instance (e.g. :class:`SegmentChestTotalSegmentator`,
    :class:`SegmentChestTotalSegmentatorWithContrast` for contrast-enhanced
    studies, :class:`SegmentHeartSimpleware`, or
    :class:`SegmentHeartSimplewareTrimmedBranches` for cardiac-only
    segmentation with pulmonary/great-vessel branches trimmed). Defaults to
    a new :class:`SegmentChestTotalSegmentatorWithContrast` when omitted.  Its
    taxonomy names the labels and groups, so it is also needed when
    :meth:`process` is given a labelmap rather than an image.

    **Surfaces**

    Every surface is closed and outward-oriented, from
    :meth:`ProcessSurfaceMesh.extract_label_surfaces`.  The per-label surfaces
    are extracted together, so neighbors share the wall between them; the group
    surfaces are extracted together in a second pass, so neighboring groups do
    too.  Groups default to the segmenter's taxonomy (see
    :attr:`anatomy_groups`); pass ``groups`` to :meth:`process` for others, such
    as a heart without its chambers.  Empty groups and labels are skipped.

    **VTK object annotation**

    Each surface and tetrahedral mesh returned by :meth:`process` carries:

    - ``field_data['AnatomyGroup']`` - anatomy group name, e.g. ``'heart'``.
    - ``field_data['SegmentationLabelNames']`` - structure names it holds
      (e.g. ``['left_ventricle', 'right_ventricle', ...]``).
    - ``field_data['SegmentationLabelIds']`` - corresponding integer label IDs.
    - ``field_data['AnatomyColor']`` - RGB float color from :class:`ProcessUSDAnatomy`.
    - ``cell_data['Color']`` - RGBA uint8 array (n_cells x 4) for direct VTK rendering.

    **I/O contract**

    :meth:`process` performs *no* file I/O.  Use :meth:`save_surfaces` and
    :meth:`save_combined_surfaces` - or the CLI
    ``monai-physio-convert-image-to-vtk`` - to write results to disk.
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
        self._surface_tools = ProcessSurfaceMesh(log_level=log_level)
        self._volume_tools = ProcessVolumetricMesh(log_level=log_level)
        # ProcessUSDAnatomy builds its color tables in __init__ without touching
        # the stage, so stage=None is safe for this lookup-only use.
        self._anatomy_tools = ProcessUSDAnatomy(stage=None, log_level=log_level)

        #: Anatomy group names registered by the active segmenter's taxonomy,
        #: in the order they were first added.
        self.anatomy_groups: tuple[str, ...] = tuple(
            self._segmenter.taxonomy.group_names()
        )

    # ─────────────────────────── Internal helpers ──────────────────────────

    def _annotate(
        self,
        vtk_obj: pv.DataSet,
        group: str,
        label_names: list[str],
        label_ids: list[int],
    ) -> None:
        """Attach anatomy metadata and solid color to a VTK object **in-place**.

        Sets ``field_data['AnatomyGroup']``, ``field_data['SegmentationLabelNames']``
        and ``field_data['SegmentationLabelIds']`` (int32), then the color
        :meth:`ProcessUSDAnatomy.apply_anatomy_color` picks for the most specific
        name: the label for a single-label object, otherwise the group.
        """
        vtk_obj.field_data["AnatomyGroup"] = np.array([group])
        vtk_obj.field_data["SegmentationLabelNames"] = np.array(
            label_names if label_names else [group]
        )
        vtk_obj.field_data["SegmentationLabelIds"] = np.array(label_ids, dtype=np.int32)
        names = [label_names[0], group] if len(label_names) == 1 else [group]
        self._anatomy_tools.apply_anatomy_color(vtk_obj, names)

    def _tetrahedra(
        self, mask: itk.Image, surface: pv.PolyData, element_size_mm: float
    ) -> Optional[pv.UnstructuredGrid]:
        """Return *mask*'s tetrahedral mesh relaxed onto *surface*, or ``None``.

        Resampling to the element size eats a structure thinner than it, so
        the size is halved until the mesh holds most of the surface's volume
        or is as fine as the mask itself.  ``None`` means the structure is
        thinner than one element even then.
        """
        finest_spacing = float(np.min(np.asarray(mask.GetSpacing())))
        while True:
            mesh = self._volume_tools.extract_tetrahedra(mask, element_size_mm)
            volume = float(np.sum(mesh.compute_cell_sizes(volume=True)["Volume"]))
            if volume > 0.5 * surface.volume or element_size_mm <= finest_spacing:
                break
            element_size_mm *= 0.5
        if mesh.n_cells == 0:
            return None
        return self._volume_tools.trim_tetrahedra_to_surface(mesh, surface)

    @staticmethod
    def _mask(keep: np.ndarray, like: itk.Image) -> itk.Image:
        """Return *keep* as a binary mask with *like*'s geometry."""
        mask = itk.GetImageFromArray(keep.astype(np.uint8))
        mask.CopyInformation(like)
        return mask

    # ─────────────────────────── Main workflow ─────────────────────────────

    def process(
        self,
        input_image: Optional[itk.Image] = None,
        anatomy_groups: Optional[list[str]] = None,
        surface_reduction_rate: float = 0.0,
        extract_label_surfaces: bool = False,
        *,
        labelmap_image: Optional[itk.Image] = None,
        extract_group_surfaces: bool = True,
        groups: Optional[Mapping[str, list[int]]] = None,
        isotropic_spacing_mm: Optional[float] = None,
        smoothing_iterations: int = 30,
        tetrahedra_element_size_mm: Optional[float] = None,
    ) -> dict[str, Any]:
        """Extract per-group and per-label surfaces from an image or a labelmap.

        Args:
            input_image: CT image to segment.  Give this or *labelmap_image*.
            anatomy_groups: Subset of the taxonomy's groups to process.
                ``None`` (default) processes all of them.  Valid names are given
                by :attr:`anatomy_groups`.
            surface_reduction_rate: Fraction in ``[0, 1)`` of each surface's
                triangles to remove by ACVD remeshing.  ``0.0`` (default) keeps
                them all, and only then do neighboring surfaces share their
                walls exactly; see
                :meth:`ProcessSurfaceMesh.extract_label_surfaces`.
            extract_label_surfaces: Also extract one surface per structure
                (label) of the processed groups, e.g. ``left_ventricle`` and
                ``right_ventricle`` separately within ``heart``.
            labelmap_image: Labelmap produced by this workflow's segmenter, used
                in place of segmenting *input_image*.
            extract_group_surfaces: Extract one surface per group (default).
            groups: Group name → label ids, each extracted as one surface.
                ``None`` uses the processed taxonomy groups.  The groups must not
                share a label.
            isotropic_spacing_mm: Grid pitch the surfaces are contoured on,
                which sets their triangle count.  ``None`` uses the labelmap's
                finest spacing.
            smoothing_iterations: Taubin smoothing iterations (0 disables).
            tetrahedra_element_size_mm: When set, also build a tetrahedral mesh
                of every surface at this element size, relaxed onto the surface.

        Returns:
            ``dict`` with the following keys:

            - ``'surfaces'`` - group name → surface.
            - ``'label_surfaces'`` - label name → surface; empty unless
              *extract_label_surfaces*.
            - ``'tetrahedra'`` / ``'label_tetrahedra'`` - group / label name →
              tetrahedral mesh; empty unless *tetrahedra_element_size_mm*.
            - ``'labelmap'`` - ``itk.Image``: per-structure labelmap.

        Raises:
            ValueError: If not exactly one of *input_image* and *labelmap_image*
                is given, if any name in *anatomy_groups* is invalid, or if two
                *groups* share a label.
        """
        self.log_section("STARTING IMAGE TO VTK WORKFLOW")

        if (input_image is None) == (labelmap_image is None):
            raise ValueError("Give exactly one of input_image and labelmap_image")
        if anatomy_groups is not None:
            invalid = [g for g in anatomy_groups if g not in self.anatomy_groups]
            if invalid:
                raise ValueError(
                    f"Unknown anatomy groups: {invalid}. "
                    f"Valid: {list(self.anatomy_groups)}"
                )
        groups_to_process = list(anatomy_groups or self.anatomy_groups)

        if labelmap_image is None:
            self.log_info("Running segmenter: %s", type(self._segmenter).__name__)
            labelmap_image = self._segmenter.segment(input_image)["labelmap"]
        labels = itk.GetArrayViewFromImage(labelmap_image)

        taxonomy = self._segmenter.taxonomy
        label_names = taxonomy.all_labels()
        present_ids = [int(value) for value in np.unique(labels) if value != 0]
        processed_ids = [
            label_id
            for label_id in present_ids
            if taxonomy.group_for_id(label_id) in groups_to_process
        ]
        if groups is None:
            groups = {
                group: [i for i in processed_ids if taxonomy.group_for_id(i) == group]
                for group in groups_to_process
            }
        group_ids = [label_id for ids in groups.values() for label_id in ids]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(f"Groups must not share a label: {dict(groups)}")

        extract_options: dict[str, Any] = {
            "isotropic_spacing_mm": isotropic_spacing_mm,
            "smoothing_iterations": smoothing_iterations,
            "surface_reduction_rate": surface_reduction_rate,
        }
        surfaces: dict[str, pv.PolyData] = {}
        label_surfaces: dict[str, pv.PolyData] = {}
        tetrahedra: dict[str, pv.UnstructuredGrid] = {}
        label_tetrahedra: dict[str, pv.UnstructuredGrid] = {}

        if extract_group_surfaces:
            self.log_section("Extracting group surfaces")
            # One id per group, so the groups are extracted together and the
            # wall between two groups is shared like the wall between labels.
            group_map = np.zeros(labels.shape, dtype=np.uint16)
            for index, ids in enumerate(groups.values(), start=1):
                group_map[np.isin(labels, ids)] = index
            group_image = itk.GetImageFromArray(group_map)
            group_image.CopyInformation(labelmap_image)
            extracted = self._surface_tools.extract_label_surfaces(
                group_image, **extract_options
            )
            for index, (group, ids) in enumerate(groups.items(), start=1):
                if index not in extracted:
                    continue
                names = [label_names.get(i, f"label_{i}") for i in ids]
                self._annotate(extracted[index], group, names, list(ids))
                surfaces[group] = extracted[index]
                if tetrahedra_element_size_mm is not None:
                    mesh = self._tetrahedra(
                        self._mask(group_map == index, labelmap_image),
                        extracted[index],
                        tetrahedra_element_size_mm,
                    )
                    if mesh is not None:
                        self._annotate(mesh, group, names, list(ids))
                        tetrahedra[group] = mesh

        if extract_label_surfaces:
            self.log_section("Extracting label surfaces")
            extracted = self._surface_tools.extract_label_surfaces(
                labelmap_image, label_ids=processed_ids, **extract_options
            )
            for label_id, surface in extracted.items():
                name = label_names.get(label_id, f"label_{label_id}")
                group = taxonomy.group_for_id(label_id)
                self._annotate(surface, group, [name], [label_id])
                label_surfaces[name] = surface
                if tetrahedra_element_size_mm is not None:
                    mesh = self._tetrahedra(
                        self._mask(labels == label_id, labelmap_image),
                        surface,
                        tetrahedra_element_size_mm,
                    )
                    if mesh is not None:
                        self._annotate(mesh, group, [name], [label_id])
                        label_tetrahedra[name] = mesh

        self.log_section("IMAGE TO VTK WORKFLOW COMPLETE")
        self.log_info("Surfaces extracted:       %d", len(surfaces))
        self.log_info("Label surfaces extracted: %d", len(label_surfaces))

        return {
            "surfaces": surfaces,
            "label_surfaces": label_surfaces,
            "tetrahedra": tetrahedra,
            "label_tetrahedra": label_tetrahedra,
            "labelmap": labelmap_image,
        }

    # ─────────────────────────── I/O helpers ───────────────────────────────

    @staticmethod
    def save_surfaces(
        surfaces: Mapping[str, pv.DataSet],
        output_dir: str,
        prefix: str = "",
    ) -> dict[str, str]:
        """Save each named surface or mesh to its own VTK file.

        Args:
            surfaces: Mapping of name → mesh, e.g. a value :meth:`process`
                returns.  Surfaces are written as ``.vtp``, volume meshes as
                ``.vtu``.
            output_dir: Directory to write files into (created if absent).
            prefix: Optional filename prefix.  Each file is named
                ``{prefix}_{name}`` (or ``{name}`` when *prefix* is empty).

        Returns:
            Mapping of name → path of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved: dict[str, str] = {}
        for name, surface in surfaces.items():
            stem = f"{prefix}_{name}" if prefix else name
            suffix = ".vtp" if isinstance(surface, pv.PolyData) else ".vtu"
            path = os.path.join(output_dir, f"{stem}{suffix}")
            surface.save(path)
            saved[name] = path
        return saved

    @staticmethod
    def save_combined_surfaces(
        surfaces: Mapping[str, pv.PolyData], output_filename: str
    ) -> str:
        """Write :meth:`ProcessSurfaceMesh.combine_surfaces` of *surfaces* to one file.

        The file keeps each cell's structure as ``cell_data['SegmentationLabelIds']``
        and its color as ``cell_data['Color']``, but not the per-object
        ``field_data``; use :meth:`save_surfaces` when structure *names* must
        be recoverable from the saved files.

        Args:
            surfaces: Mapping of name → surface.
            output_filename: Path of the VTP file to write, including its
                directory.  Any missing parent directories are created.

        Returns:
            Path to the saved VTP file.

        Raises:
            ValueError: If *surfaces* is empty.
        """
        combined = ProcessSurfaceMesh.combine_surfaces(surfaces)
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        combined.save(output_filename)
        return output_filename
