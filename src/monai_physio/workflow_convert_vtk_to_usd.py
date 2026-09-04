"""
VTK to USD conversion workflow and batch runner.

Implements the pipeline from the Convert_VTK_To_USD experiment notebooks:
take one or more meshes, optionally split by connectivity or cell type,
convert to USD, then apply a chosen appearance (solid color, anatomic material,
or colormap from a primvar with auto or specified intensity range).
"""

import logging
import re
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import pyvista as pv
import vtk

from .convert_vtk_to_usd import ConvertVTKToUSD
from .monai_physio_base import MONAIPhysioBase
from .segment_anatomy_base import SegmentAnatomyBase
from .usd_anatomy_tools import USDAnatomyTools
from .usd_tools import USDTools

AppearanceKind = Literal["solid", "anatomy", "colormap"]


class WorkflowConvertVTKToUSD(MONAIPhysioBase):
    """
    Workflow to convert one or more meshes to USD with configurable
    splitting and appearance (solid color, anatomic material, or colormap).
    """

    def __init__(
        self,
        input_meshes: Sequence[Union[pv.DataSet, vtk.vtkDataSet]],
        usd_project_name: str,
        output_directory: Union[str, Path],
        *,
        separate_by_connectivity: bool = True,
        separate_by_cell_type: bool = False,
        frames_per_second: float = 60.0,
        extract_surface: bool = True,
        static_merge: bool = False,
        time_codes: Optional[list[float]] = None,
        appearance: AppearanceKind = "solid",
        solid_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
        anatomy_type: Optional[str] = None,
        object_names: Optional[Sequence[str]] = None,
        label_names: Optional[Mapping[int, str]] = None,
        segmenter: Optional[SegmentAnatomyBase] = None,
        colormap_primvar: Optional[str] = None,
        colormap_name: str = "viridis",
        colormap_intensity_range: Optional[tuple[float, float]] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """
        Initialize the VTK-to-USD workflow.

        Args:
            input_meshes: One or more PyVista/VTK meshes. A single mesh, or
                static_merge=True, produces a static scene; multiple meshes
                with static_merge=False (default) are treated as ordered
                time-series frames, in list order.
            usd_project_name: Project name; used as the root USD prim name
                (/World/{usd_project_name}) and the output filename. A
                trailing USD extension (.usd, .usda, .usdc) is stripped from
                the prim name but preserved for the output filename; if
                omitted, ".usd" is used.
            output_directory: Directory where the output USD file is written.
            separate_by_connectivity: If True, split mesh into separate objects by connectivity.
            separate_by_cell_type: If True, split mesh by cell type (triangle/quad/...).
                Cannot be True when separate_by_connectivity is True.
            frames_per_second: FPS for time-varying data.
            extract_surface: For volumetric meshes, extract surface before conversion.
            static_merge: If True, input_meshes is not a time series - each mesh is
                written as a separate static object with no time samples (see
                ConvertVTKToUSD).
            time_codes: Explicit time codes aligned to input_meshes, used when
                static_merge is False. If None, uses sequential integers [0, 1, 2, ...].
            appearance: "solid" | "anatomy" | "colormap".
            solid_color: RGB in [0,1] when appearance == "solid".
            anatomy_type: Anatomy material name applied to every mesh when
                appearance == "anatomy" (e.g. heart, lung, bone, soft_tissue).
                None (default) instead resolves a material per mesh prim from
                that prim's name, so a stage whose objects are named after the
                structures they hold gets per-structure materials (e.g.
                ventricle_left vs. myocardium). A name matching no material
                falls back to the object's ``field_data['AnatomyGroup']``
                (so "rib_left_3" still reaches the bone material) and then to
                the "other" material.
            object_names: Prim names aligned to input_meshes, used when
                static_merge is True. None (default) derives them from each
                mesh's ``field_data['SegmentationLabelNames']`` when that holds
                exactly one name (as written by
                :class:`WorkflowConvertImageToVTK`), and falls back to
                ``{usd_project_name}_{index}`` otherwise.
            label_names: Mapping of label id → structure name. Each input mesh
                is then split on its per-cell ``SegmentationLabelIds`` (or
                ``boundary_labels``) array, so every structure becomes its own
                prim at ``/World/{usd_project_name}/{group}/{structure}``:
                time-varying across frames, or static from a single mesh. This
                is the only way structure identity survives a time series;
                without it, parts are named by connectivity-component order,
                which is positional per frame. None (default) reads the ids off
                the meshes themselves when they carry the per-cell array, and
                names them from *segmenter*'s taxonomy — so passing a mesh
                merged by :meth:`ContourTools.save_combined_surfaces` splits by
                structure without further arguments. ``static_merge`` accepts
                only one labeled mesh: several would collide on one prim path
                per label.
            segmenter: Segmenter whose taxonomy groups the labels of
                *label_names* by anatomy type. Also selects each structure's
                material when appearance == "anatomy", through
                :meth:`USDAnatomyTools.enhance_meshes`, which falls back to the
                containing group for a structure with no material of its own.
            colormap_primvar: Primvar name for coloring when appearance == "colormap"
                (e.g. vtk_point_stress_c0). If None, a candidate is auto-picked when possible.
            colormap_name: Matplotlib colormap name when appearance == "colormap".
            colormap_intensity_range: Optional (vmin, vmax) for colormap; None = auto from data.
            log_level: Logging level.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.input_meshes = list(input_meshes)
        suffix = Path(usd_project_name).suffix
        if suffix.lower() in {".usd", ".usda", ".usdc"}:
            self.usd_project_name = usd_project_name[: -len(suffix)]
            self._usd_extension = suffix
        else:
            self.usd_project_name = usd_project_name
            self._usd_extension = ".usd"
        self.output_directory = Path(output_directory)
        self.separate_by_connectivity = separate_by_connectivity
        self.separate_by_cell_type = separate_by_cell_type
        self.frames_per_second = frames_per_second
        self.extract_surface = extract_surface
        self.static_merge = static_merge
        self.time_codes = time_codes
        self.appearance = appearance
        self.solid_color = solid_color
        self.anatomy_type = anatomy_type
        self.object_names = list(object_names) if object_names is not None else None
        self.label_names = dict(label_names) if label_names is not None else None
        self.segmenter = segmenter
        self.colormap_primvar = colormap_primvar
        self.colormap_name = colormap_name
        self.colormap_intensity_range = colormap_intensity_range

        if separate_by_connectivity and separate_by_cell_type:
            raise ValueError(
                "separate_by_connectivity and separate_by_cell_type cannot both be True"
            )

    @staticmethod
    def _as_pyvista(
        mesh: Union[pv.DataSet, vtk.vtkDataSet],
    ) -> Optional[pv.DataSet]:
        """Return *mesh* as a PyVista dataset, or ``None`` if it is not one."""
        if not isinstance(mesh, pv.DataSet) and isinstance(mesh, vtk.vtkDataSet):
            mesh = pv.wrap(mesh)
        return mesh if isinstance(mesh, pv.DataSet) else None

    def _resolve_label_names(self) -> Optional[dict[int, str]]:
        """Return the label ids to split every input mesh on, or ``None``.

        An explicit ``label_names`` is used as given. Otherwise the meshes are
        searched for the per-cell label array that
        :meth:`ContourTools.save_combined_surfaces` writes on a merge, and that
        contouring a multi-label labelmap leaves behind. That array is
        preferred wherever it exists because it survives merging, which the
        per-object ``field_data`` naming does not — so a combined surface file
        splits back into its structures instead of collapsing onto one prim.

        Ids are named from *segmenter*'s taxonomy first, then from the
        ``field_data`` of any input holding exactly one structure, and finally
        as ``label_{id}``. Ids that no source can name are not worth splitting
        on, so a set where none resolve falls back to per-object naming.

        Returns:
            The id → name mapping, or ``None`` when no mesh carries the array
            or none of its ids can be named, in which case prims are named per
            object as before.
        """
        if self.label_names is not None:
            return self.label_names

        label_ids: set[int] = set()
        field_names: dict[int, str] = {}
        for mesh in self.input_meshes:
            pv_mesh = self._as_pyvista(mesh)
            if pv_mesh is None:
                continue
            for array_name in ("SegmentationLabelIds", "boundary_labels"):
                if array_name in pv_mesh.cell_data:
                    label_ids.update(
                        int(value) for value in np.unique(pv_mesh.cell_data[array_name])
                    )
                    break
            ids = pv_mesh.field_data.get("SegmentationLabelIds")
            names = pv_mesh.field_data.get("SegmentationLabelNames")
            if ids is not None and names is not None and len(ids) == len(names) == 1:
                field_names[int(ids[0])] = str(names[0])

        # 0 tags the cells save_combined_surfaces could not attribute to one
        # structure, so it names nothing.
        label_ids.discard(0)
        if not label_ids:
            return None

        taxonomy_names = (
            self.segmenter.taxonomy.all_labels() if self.segmenter is not None else {}
        )
        named = {
            label_id: taxonomy_names.get(label_id) or field_names.get(label_id)
            for label_id in sorted(label_ids)
        }
        if not any(named.values()):
            # Ids nobody can name would only produce "label_37" prims, which
            # carry less meaning than the object names they would replace.
            self.log_debug(
                "Per-cell labels %s match no name; naming per object instead",
                sorted(label_ids),
            )
            return None
        resolved = {
            label_id: name or f"label_{label_id}" for label_id, name in named.items()
        }
        self.log_info(
            "Splitting on the per-cell label array: %s", ", ".join(resolved.values())
        )
        return resolved

    def _read_object_annotations(self) -> list[tuple[Optional[str], Optional[str]]]:
        """Return ``(structure name, anatomy group)`` per input mesh.

        Both come from the annotation :class:`WorkflowConvertImageToVTK` writes
        onto each surface: the name from ``field_data['SegmentationLabelNames']``
        when it holds exactly one entry, the group from
        ``field_data['AnatomyGroup']``. Either is ``None`` when absent.
        """
        annotations: list[tuple[Optional[str], Optional[str]]] = []
        for mesh in self.input_meshes:
            pv_mesh = self._as_pyvista(mesh)
            if pv_mesh is None:
                annotations.append((None, None))
                continue
            mesh = pv_mesh
            label_names = mesh.field_data.get("SegmentationLabelNames")
            groups = mesh.field_data.get("AnatomyGroup")
            name = (
                str(label_names[0])
                if label_names is not None and len(label_names) == 1
                else None
            )
            group = str(groups[0]) if groups is not None and len(groups) else None
            annotations.append((name, group))
        return annotations

    def _anatomy_candidates(
        self, mesh_path: str, object_groups: Mapping[str, str]
    ) -> list[str]:
        """Return the anatomy names to try for *mesh_path*, best match first.

        With ``anatomy_type`` set, that one name is the only candidate. Without
        it, the prim's own name is tried first, then the anatomy group of the
        object it came from — so ``"rib_left_3"``, which matches no material of
        its own, still lands on the bone material through its group.
        """
        if self.anatomy_type is not None:
            return [self.anatomy_type]
        # Connectivity/cell-type splitting appends "_objectN" to the object
        # name; strip it to recover the name object_groups is keyed by.
        leaf = mesh_path.rsplit("/", 1)[-1]
        object_name = re.sub(r"_object\d+$", "", leaf)
        group = object_groups.get(object_name)
        return [object_name] if group is None else [object_name, group]

    def process(self) -> dict[str, Any]:
        """
        Run the full workflow: convert meshes to USD, then apply the chosen appearance.

        Returns:
            Dict with the results of the workflow:
                - "usd_file" (str): Path to the created USD file.
        """
        self.log_section("VTK to USD conversion workflow")

        if not self.input_meshes:
            raise ValueError("input_meshes must not be empty")

        n_frames = len(self.input_meshes)
        if self.static_merge:
            time_codes = None
        elif self.time_codes is None:
            time_codes = [float(i) for i in range(n_frames)]
        else:
            time_codes = self.time_codes

        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_usd = (
            self.output_directory / f"{self.usd_project_name}{self._usd_extension}"
        )

        self.log_info("Input: %d mesh(es)", n_frames)
        if self.static_merge:
            self.log_info(
                "static_merge=True; outputting static scene (no time samples)"
            )
        self.log_info("Output: %s", output_usd)

        separate_by: Literal["none", "connectivity", "cell_type"] = (
            "connectivity"
            if self.separate_by_connectivity
            else "cell_type"
            if self.separate_by_cell_type
            else "none"
        )

        # Per-cell labels, when the meshes carry them, name the prims instead:
        # one per structure, in both the static and the time-series layout.
        label_names = self._resolve_label_names()
        name_objects = self.static_merge and label_names is None

        # Object names only name prims in the static-merge layout; a time
        # series writes one prim per part across all frames instead.
        annotations = self._read_object_annotations()
        object_names = None
        if name_objects:
            object_names = self.object_names
            if object_names is None and any(name for name, _ in annotations):
                object_names = [
                    name or f"{self.usd_project_name}_{index}"
                    for index, (name, _) in enumerate(annotations)
                ]
            if object_names is not None:
                self.log_info("Naming objects: %s", ", ".join(object_names))

        # Anatomy group per object name, used as the fallback when the name
        # itself matches no material (e.g. "rib_left_3" -> the bone group).
        # Keyed by the prim names ConvertVTKToUSD will actually emit, which fall
        # back to "<project>_<index>" when no object_names were derived.
        object_groups: dict[str, str] = {}
        if name_objects:
            group_keys = object_names or [
                f"{self.usd_project_name}_{index}" for index in range(len(annotations))
            ]
            for object_name, (_, group) in zip(group_keys, annotations):
                if group is not None:
                    object_groups[object_name] = group

        converter = ConvertVTKToUSD(
            data_basename=self.usd_project_name,
            input_polydata=self.input_meshes,
            mask_ids=label_names,
            segmenter=self.segmenter,
            convert_to_surface=self.extract_surface,
            separate_by=separate_by,
            frames_per_second=self.frames_per_second,
            solid_color=self.solid_color,
            static_merge=self.static_merge,
            time_codes=time_codes,
            object_names=object_names,
            log_level=self.log_level,
        )
        stage = converter.convert(str(output_usd))

        # Post-process: apply chosen appearance to all meshes under /World/{usd_project_name}
        usd_tools = USDTools(log_level=self.log_level)
        mesh_paths = usd_tools.list_mesh_paths_under(
            str(output_usd), parent_path=f"/World/{self.usd_project_name}"
        )
        if not mesh_paths:
            self.log_warning(
                "No mesh prims found under /World/%s", self.usd_project_name
            )
            return {"usd_file": str(output_usd)}

        # Static merge has no time samples; pass None so only default time is used
        appearance_time_codes = None if self.static_merge else time_codes

        self.log_info(
            "Applying appearance '%s' to %d mesh(es)", self.appearance, len(mesh_paths)
        )

        if self.appearance == "solid":
            for mesh_path in mesh_paths:
                usd_tools.set_solid_display_color(
                    str(output_usd),
                    mesh_path,
                    self.solid_color,
                    time_codes=appearance_time_codes,
                    bind_vertex_color_material=True,
                )

        elif (
            self.appearance == "anatomy"
            and label_names is not None
            and self.segmenter is not None
        ):
            # The label layout names each prim after its structure, and the
            # segmenter's taxonomy supplies the group to fall back on when the
            # structure has no material of its own.
            USDAnatomyTools(stage, log_level=self.log_level).enhance_meshes(
                self.segmenter
            )
            stage.Save()

        elif self.appearance == "anatomy":
            anatomy_tools = USDAnatomyTools(stage, log_level=self.log_level)
            for mesh_path in mesh_paths:
                candidates = self._anatomy_candidates(mesh_path, object_groups)
                selected = next(
                    (
                        candidate
                        for candidate in candidates
                        if anatomy_tools.resolve_anatomy_type(candidate) is not None
                    ),
                    None,
                )
                if selected is None:
                    self.log_warning(
                        "No anatomy material matches %s; using 'other'",
                        " or ".join(candidates),
                    )
                    selected = "other"
                anatomy_tools.apply_anatomy_material_to_mesh(mesh_path, selected)
            stage.Save()

        elif self.appearance == "colormap":
            primvar = self.colormap_primvar
            for mesh_path in mesh_paths:
                if primvar is None:
                    primvars = usd_tools.list_mesh_primvars(str(output_usd), mesh_path)
                    primvar = usd_tools.pick_color_primvar(primvars)
                if primvar is None:
                    self.log_warning(
                        "No color primvar found for %s; skip colormap", mesh_path
                    )
                    primvar = self.colormap_primvar
                    continue
                self.log_info(
                    "Applying colormap to %s from primvar %s", mesh_path, primvar
                )
                usd_tools.apply_colormap_from_primvar(
                    str(output_usd),
                    mesh_path,
                    primvar,
                    cmap=self.colormap_name,
                    intensity_range=self.colormap_intensity_range,
                    write_default_at_t0=True,
                    bind_vertex_color_material=True,
                )
                if self.colormap_primvar is None:
                    primvar = None  # next mesh: auto-pick again

        self.log_info("Workflow complete: %s", output_usd)
        return {"usd_file": str(output_usd)}
