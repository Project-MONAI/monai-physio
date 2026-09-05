"""Unified VTK to USD converter with advanced features.

This module provides a high-level interface for converting VTK/PyVista meshes to USD,
with support for:
- Time-series animation
- Anatomical region labeling (mask_ids)
- Colormap visualization
- Automatic topology change detection
- Both surface and volumetric meshes

Uses the vtk_to_usd library internally for core conversion functionality.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

import numpy as np
import pyvista as pv
import vtk
from pxr import Sdf, Usd, UsdGeom

from .monai_physio_base import MONAIPhysioBase
from .segment_anatomy_base import SegmentAnatomyBase
from .vtk_to_usd import (
    ConversionSettings,
    DataType,
    GenericArray,
    MaterialData,
    MaterialManager,
    MeshData,
    UsdMeshConverter,
    add_framing_camera,
    cell_type_name_for_vertex_count,
    read_vtk_file,
    split_mesh_data_by_cell_type,
    split_mesh_data_by_connectivity,
    validate_time_series_topology,
)


class ConvertVTKToUSD(MONAIPhysioBase):
    """
    Advanced VTK to USD converter with colormap and anatomical labeling support.

    This class extends the basic vtk_to_usd library with:
    - Support for VTK/PyVista objects (not just files)
    - Anatomical region labeling via mask_ids
    - Colormap-based visualization
    - Automatic topology change detection
    - Time-series animation

    Example Usage:
        >>> # Create converter with time-series meshes
        >>> converter = ConvertVTKToUSD(
        ...     data_basename='CardiacModel',
        ...     input_polydata=meshes,  # List of PyVista/VTK meshes
        ...     mask_ids={1: 'ventricle', 2: 'atrium'}
        ... )
        >>>
        >>> # Configure colormap visualization
        >>> converter.set_colormap(
        ...     color_by_array='transmembrane_potential',
        ...     colormap='rainbow',
        ...     intensity_range=(-80.0, 20.0)
        ... )
        >>>
        >>> # Convert to USD
        >>> stage = converter.convert('output.usd')
    """

    def __init__(
        self,
        data_basename: str,
        input_polydata: Sequence[pv.DataSet | vtk.vtkDataSet],
        mask_ids: Optional[dict[int, str]] = None,
        compute_normals: bool = False,
        convert_to_surface: bool = True,
        frames_per_second: float = 24.0,
        separate_by: Literal["none", "connectivity", "cell_type"] = "none",
        solid_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
        static_merge: bool = False,
        time_codes: Optional[list[float]] = None,
        object_names: Optional[Sequence[str]] = None,
        segmenter: Optional[SegmentAnatomyBase] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        """
        Initialize converter.

        Args:
            data_basename: Base name for USD data (used in prim paths). A
                trailing USD extension (.usd, .usda, .usdc) is stripped.
            input_polydata: Sequence of PyVista/VTK meshes (one per time step, or
                one per static object when static_merge is True)
            mask_ids: Optional mapping of label IDs to anatomical region names.
                     If provided, meshes are split by labeled region, one prim
                     per region, and the static_merge layout is not used: with
                     a single mesh that yields one static prim per structure,
                     with several it yields one time-varying prim per structure.
            compute_normals: Whether to compute vertex normals
            convert_to_surface: If True, extract surface from volumetric meshes
            frames_per_second: Time codes per second (default 24.0).
                            For medical imaging time series where each frame = 1 second, use 1.0.
            separate_by: How to split the mesh into sub-prims.
                        'none' keeps the mesh as-is, 'connectivity' splits by connected
                        component, 'cell_type' splits by face vertex count.
            solid_color: Default RGB diffuse color in [0, 1] used when no colormap is set.
            static_merge: If True, treat each mesh in input_polydata as a separate
                          object in a single static scene (no time samples) instead
                          of a time series.
            time_codes: Explicit time codes aligned to input_polydata, used when
                        static_merge is False. If None, uses sequential integers
                        [0, 1, 2, ...].
            object_names: Optional prim names aligned to input_polydata, used
                        when static_merge is True. If None, objects are named
                        ``{data_basename}_{index}``. Naming objects after the
                        structure they hold (e.g. "heart_ventricle_left") makes
                        the stage self-describing and lets downstream material
                        assignment key off the prim name.
            segmenter: Optional SegmentAnatomyBase instance used to classify each
                       mask_ids label into an anatomy group (heart / lung / bone /
                       major_vessels / contrast / soft_tissue / other) so labeled
                       prims are written under ``/World/{data_basename}/{type}/{label}``
                       and materials under ``/World/Looks/{type}/{label}_material``.
                       When None, labels are grouped under a single ``Anatomy`` Xform.
            log_level: Logging level

        Raises:
            ValueError: If time_codes or object_names is not None and its length
                does not match input_polydata, or if time_codes values are not
                non-decreasing.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        suffix = Path(data_basename).suffix
        self.data_basename = (
            data_basename[: -len(suffix)]
            if suffix.lower() in {".usd", ".usda", ".usdc"}
            else data_basename
        )
        self.input_polydata = list(input_polydata)
        self.mask_ids = mask_ids
        self.compute_normals = compute_normals
        self.convert_to_surface = convert_to_surface
        self.separate_by = separate_by
        self.solid_color = solid_color
        self.segmenter = segmenter

        # Colormap settings
        self.color_by_array: Optional[str] = None
        self.colormap: str = "plasma"
        self.intensity_range: Optional[tuple[float, float]] = None

        if static_merge and mask_ids and len(self.input_polydata) > 1:
            raise ValueError(
                "static_merge with mask_ids is only defined for a single mesh: "
                "several static objects each holding the same labels would "
                "collide on one prim path per label. Merge them first, or drop "
                "static_merge to treat them as frames."
            )
        if not static_merge and time_codes is not None:
            if len(time_codes) != len(self.input_polydata):
                raise ValueError(
                    f"time_codes length ({len(time_codes)}) must match "
                    f"input_polydata length ({len(self.input_polydata)})"
                )
            if len(time_codes) > 1 and any(
                time_codes[i] > time_codes[i + 1] for i in range(len(time_codes) - 1)
            ):
                raise ValueError(
                    "time_codes must be in non-decreasing order; "
                    "got values that decrease between consecutive frames"
                )
        if object_names is not None:
            if len(object_names) != len(self.input_polydata):
                raise ValueError(
                    f"object_names length ({len(object_names)}) must match "
                    f"input_polydata length ({len(self.input_polydata)})"
                )
            # Each name becomes a prim path component, so it has to be a legal
            # USD identifier and unique or prims silently collide.
            invalid = [n for n in object_names if not Sdf.Path.IsValidIdentifier(n)]
            counts = Counter(object_names)
            duplicated = sorted(n for n, count in counts.items() if count > 1)
            if invalid or duplicated:
                raise ValueError(
                    "object_names must be unique and valid USD prim names "
                    "(letter or underscore followed by letters, digits or "
                    f"underscores); invalid: {invalid}, duplicated: {duplicated}"
                )
        self._is_static_merge: bool = static_merge
        self._time_codes: Optional[list[float]] = time_codes
        self.object_names: Optional[list[str]] = (
            list(object_names) if object_names is not None else None
        )
        # Pre-converted MeshData for each time step; populated by from_files() so
        # _convert_unified() can reuse the topology-validation work instead of
        # calling _vtk_to_mesh_data() a second time.
        self._cached_mesh_data: Optional[list[MeshData]] = None

        # Conversion settings
        self.settings = ConversionSettings(
            triangulate_meshes=True,
            compute_normals=compute_normals,
            preserve_point_arrays=True,
            preserve_cell_arrays=True,
            meters_per_unit=1.0,
            up_axis="Y",
            frames_per_second=frames_per_second,
        )

        self.logger.info(
            f"Initialized converter with {len(input_polydata)} time steps, "
            f"mask_ids={'enabled' if mask_ids else 'disabled'}, "
            f"separate_by='{separate_by}'"
        )

    @classmethod
    def from_files(
        cls,
        data_basename: str,
        vtk_files: Sequence[Path | str],
        *,
        extract_surface: bool = True,
        separate_by: Literal["none", "connectivity", "cell_type"] = "none",
        frames_per_second: float = 24.0,
        solid_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
        time_codes: Optional[list[float]] = None,
        static_merge: bool = False,
        mask_ids: Optional[dict[int, str]] = None,
        segmenter: Optional[SegmentAnatomyBase] = None,
        log_level: int | str = logging.INFO,
    ) -> "ConvertVTKToUSD":
        """Create a converter by loading VTK files from disk.

        Accepts .vtk (legacy), .vtp (PolyData), and .vtu (UnstructuredGrid) files.
        For time-series input, pass files ordered by time and supply time_codes.
        For a static scene with multiple disconnected meshes, set static_merge=True.

        Args:
            data_basename: Base name for USD prim paths. A trailing USD
                extension (.usd, .usda, .usdc) is stripped.
            vtk_files: Paths to VTK files; one file = one time step (or one static mesh).
            extract_surface: If True, extract surface from UnstructuredGrid (.vtu) meshes.
            separate_by: How to split each mesh into sub-prims.
            frames_per_second: FPS for time-varying animation.
            solid_color: Default RGB diffuse color in [0, 1].
            time_codes: Explicit time codes aligned to vtk_files. If None, uses
                        sequential integers [0, 1, 2, ...].
            static_merge: If True, treat each file as a separate mesh object in a
                          single static scene (no time samples).
            mask_ids: Optional anatomical label mapping.
            segmenter: Optional SegmentAnatomyBase used to group labeled prims
                       by anatomy type. See the ConvertVTKToUSD constructor.
            log_level: Logging level.

        Returns:
            ConvertVTKToUSD instance ready to call .convert().
        """
        file_list = [Path(f) for f in vtk_files]
        if not file_list:
            raise ValueError("vtk_files must not be empty")

        meshes: list[pv.DataSet | vtk.vtkDataSet] = []
        for path in file_list:
            mesh = pv.read(str(path))
            if extract_surface and isinstance(mesh, pv.UnstructuredGrid):
                mesh = mesh.extract_surface(algorithm="dataset_surface")
            meshes.append(mesh)

        resolved_time_codes = (
            time_codes
            if time_codes is not None
            else [float(i) for i in range(len(meshes))]
        )

        instance = cls(
            data_basename=data_basename,
            input_polydata=meshes,
            mask_ids=mask_ids,
            separate_by=separate_by,
            convert_to_surface=extract_surface,
            frames_per_second=frames_per_second,
            solid_color=solid_color,
            static_merge=static_merge,
            time_codes=resolved_time_codes,
            segmenter=segmenter,
            log_level=log_level,
        )

        # Validate topology consistency for multi-frame time series and cache the
        # converted MeshData so _convert_unified() can reuse it without a second
        # round of _vtk_to_mesh_data() calls.
        if len(meshes) > 1 and not static_merge:
            mesh_data_seq = [
                instance._vtk_to_mesh_data(m, i) for i, m in enumerate(meshes)
            ]
            instance._cached_mesh_data = mesh_data_seq
            try:
                report = validate_time_series_topology(mesh_data_seq)
                for w in report.get("warnings", []):
                    instance.log_warning("%s", w)
            except Exception as exc:
                instance.log_debug("Topology validation skipped: %s", exc)

        return instance

    def supports_mesh_type(self, mesh: pv.DataSet | vtk.vtkDataSet) -> bool:
        """
        Check if mesh type is supported for conversion.

        Args:
            mesh: PyVista or VTK mesh to check

        Returns:
            bool: True if mesh type is supported
        """
        # Wrap VTK objects
        if isinstance(
            mesh, (vtk.vtkPolyData, vtk.vtkUnstructuredGrid, vtk.vtkImageData)
        ):
            mesh = pv.wrap(mesh)

        # Support most PyVista types
        return isinstance(
            mesh,
            (
                pv.PolyData,
                pv.UnstructuredGrid,
                pv.StructuredGrid,
                pv.ImageData,
                pv.RectilinearGrid,
            ),
        )

    @classmethod
    def inspect_file(
        cls,
        vtk_file: Union[Path, str],
        *,
        extract_surface: bool = True,
    ) -> dict[str, Any]:
        """Summarize a VTK file using the same low-level reader as conversion.

        This method is intended for experiments, workflows, and CLIs that need
        diagnostics without importing the advanced vtk_to_usd subpackage
        directly.

        Args:
            vtk_file: Path to a VTK file (.vtk, .vtp, or .vtu).
            extract_surface: If True, extract surfaces from volumetric meshes.

        Returns:
            Dictionary containing geometry counts, bounds, data arrays, and
            surface cell type counts.
        """
        mesh_data = read_vtk_file(vtk_file, extract_surface=extract_surface)
        is_empty = len(mesh_data.points) == 0
        if is_empty:
            bbox_min = np.zeros(3, dtype=np.float64)
            bbox_max = np.zeros(3, dtype=np.float64)
        else:
            bbox_min = np.min(mesh_data.points, axis=0)
            bbox_max = np.max(mesh_data.points, axis=0)

        unique_counts, num_each = np.unique(
            mesh_data.face_vertex_counts, return_counts=True
        )

        arrays = []
        for array in mesh_data.generic_arrays:
            array_min = None
            array_max = None
            if array.data.size > 0:
                array_min = float(np.min(array.data))
                array_max = float(np.max(array.data))
            arrays.append(
                {
                    "name": array.name,
                    "data_type": array.data_type.value,
                    "num_components": array.num_components,
                    "interpolation": array.interpolation,
                    "num_elements": len(array.data),
                    "shape": tuple(array.data.shape),
                    "range": (array_min, array_max),
                }
            )

        return {
            "is_empty": is_empty,
            "points": len(mesh_data.points),
            "faces": len(mesh_data.face_vertex_counts),
            "has_normals": mesh_data.normals is not None,
            "has_colors": mesh_data.colors is not None,
            "bounds_min": tuple(float(v) for v in bbox_min),
            "bounds_max": tuple(float(v) for v in bbox_max),
            "bounds_size": tuple(float(v) for v in bbox_max - bbox_min),
            "arrays": arrays,
            "cell_types": [
                {
                    "name": cell_type_name_for_vertex_count(int(count)),
                    "vertex_count": int(count),
                    "num_faces": int(total),
                }
                for count, total in zip(unique_counts, num_each, strict=False)
            ],
        }

    def list_available_arrays(self) -> dict:
        """
        List all point data arrays available across all time steps.

        Returns:
            dict: Dictionary with array names as keys and metadata as values.
                  Metadata includes: 'n_components', 'dtype', 'range', 'present_in_steps'
        """
        available_arrays: dict[str, dict[str, Any]] = {}

        for time_idx, mesh in enumerate(self.input_polydata):
            # Wrap VTK objects
            if isinstance(mesh, (vtk.vtkPolyData, vtk.vtkUnstructuredGrid)):
                mesh = pv.wrap(mesh)

            # Get point data arrays
            if hasattr(mesh, "point_data"):
                for array_name in mesh.point_data.keys():
                    if array_name not in available_arrays:
                        array_data = mesh.point_data[array_name]
                        available_arrays[array_name] = {
                            "n_components": int(
                                array_data.shape[1] if array_data.ndim > 1 else 1
                            ),
                            "dtype": str(array_data.dtype),
                            "range": (
                                float(np.min(array_data)),
                                float(np.max(array_data)),
                            ),
                            "present_in_steps": [time_idx],
                        }
                    else:
                        array_data = mesh.point_data[array_name]
                        meta = available_arrays[array_name]
                        current_min, current_max = cast(
                            tuple[float, float], meta["range"]
                        )
                        meta["range"] = (
                            min(current_min, float(np.min(array_data))),
                            max(current_max, float(np.max(array_data))),
                        )
                        cast(list[int], meta["present_in_steps"]).append(time_idx)

        self.logger.debug(f"Found {len(available_arrays)} data arrays")
        return available_arrays

    def set_colormap(
        self,
        color_by_array: Optional[str] = None,
        colormap: str = "plasma",
        intensity_range: Optional[tuple[float, float]] = None,
    ) -> "ConvertVTKToUSD":
        """
        Configure colormap for visualization.

        Args:
            color_by_array: Name of point data array to visualize. If None, uses solid colors.
            colormap: Colormap name. Supports all matplotlib colormaps plus aliases:
                - 'plasma', 'viridis', 'inferno', 'magma' (perceptually uniform)
                - 'rainbow', 'jet' (spectral)
                - 'hot', 'heat' (heat map, 'heat' is alias for 'hot')
                - 'coolwarm', 'seismic' (diverging)
                - 'gray', 'grayscale', 'grey', 'greyscale' (grayscale)
                - 'random', 'tab20' (categorical/discrete colors)
            intensity_range: Manual (vmin, vmax) range. If None, auto-computed from data.

        Returns:
            self: For method chaining
        """
        self.color_by_array = color_by_array
        self.colormap = colormap
        self.intensity_range = intensity_range

        self.logger.info(
            f"Colormap configured: array='{color_by_array}', "
            f"colormap='{colormap}', range={intensity_range}"
        )

        return self

    def compute_von_mises_stress(
        self,
        stress_array_name: str = "stress",
        output_name: str = "von_mises_stress",
    ) -> "ConvertVTKToUSD":
        """Add a scalar von Mises stress array derived from a 9-component
        stress tensor on every input mesh.

        For each mesh in ``self.input_polydata``, looks up ``stress_array_name``
        in ``cell_data`` first (FE stress is typically per-cell), then in
        ``point_data``, reduces the 9-component tensor to scalar von Mises, and
        writes the result back to the same data dict under ``output_name``.

        Call this between ``from_files()`` and ``convert()``. The new array
        becomes a USD primvar at convert time (``vtk_cell_<output_name>`` or
        ``vtk_point_<output_name>``) and can be selected as the
        ``color_by_array`` for set_colormap or as the target primvar for
        ``USDTools.apply_colormap_from_primvar``.

        Tensor layout (row-major)::

            [s_xx, s_xy, s_xz, s_yx, s_yy, s_yz, s_zx, s_zy, s_zz]

        Off-diagonal pairs are averaged to symmetrize, which is a no-op for an
        already-symmetric Cauchy stress tensor.

        Formula::

            sigma_VM = sqrt(0.5 * [(sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2]
                            + 3.0 * (sxy^2 + syz^2 + szx^2))

        Args:
            stress_array_name: Source array name on the input meshes.
                Defaults to ``"stress"``.
            output_name: Name under which to store the resulting scalar
                array. Defaults to ``"von_mises_stress"``.

        Returns:
            self: For method chaining.

        Raises:
            ValueError: If no input mesh contains a 9-component array named
                ``stress_array_name`` in either cell_data or point_data.
        """
        found_any = False
        for mesh in self.input_polydata:
            pv_mesh = mesh if isinstance(mesh, pv.DataSet) else pv.wrap(mesh)
            for data_dict in (pv_mesh.cell_data, pv_mesh.point_data):
                if stress_array_name not in data_dict:
                    continue
                source = np.asarray(data_dict[stress_array_name])
                if source.ndim == 1 and source.size % 9 == 0:
                    source = source.reshape(-1, 9)
                if source.ndim != 2 or source.shape[1] != 9:
                    continue

                vm = self._von_mises_from_tensor(source)
                data_dict[output_name] = vm
                found_any = True
                break

        if not found_any:
            raise ValueError(
                f"No input mesh has a 9-component array named "
                f"'{stress_array_name}' in point_data or cell_data."
            )

        # Invalidate cached MeshData so convert() picks up the new array.
        self._cached_mesh_data = None
        self.logger.info(
            "Computed %s from %s on %d input mesh(es)",
            output_name,
            stress_array_name,
            len(self.input_polydata),
        )
        return self

    @staticmethod
    def _von_mises_from_tensor(stress_tensor: np.ndarray) -> np.ndarray:
        """Scalar von Mises stress from a row-major 9-component tensor field.

        Args:
            stress_tensor: Float array of shape ``(N, 9)``.

        Returns:
            Float32 array of shape ``(N,)``.
        """
        arr = np.asarray(stress_tensor, dtype=np.float64)
        sxx, sxy, sxz = arr[:, 0], arr[:, 1], arr[:, 2]
        syx, syy, syz = arr[:, 3], arr[:, 4], arr[:, 5]
        szx, szy, szz = arr[:, 6], arr[:, 7], arr[:, 8]
        sym_xy = 0.5 * (sxy + syx)
        sym_yz = 0.5 * (syz + szy)
        sym_zx = 0.5 * (sxz + szx)
        deviatoric = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        shear = 3.0 * (sym_xy**2 + sym_yz**2 + sym_zx**2)
        result: np.ndarray = np.sqrt(np.maximum(deviatoric + shear, 0.0)).astype(
            np.float32
        )
        return result

    def convert(
        self,
        output_usd_file: str,
        convert_to_surface: Optional[bool] = None,
        compute_normals: Optional[bool] = None,
    ) -> Usd.Stage:
        """
        Convert VTK meshes to USD.

        Args:
            output_usd_file: Path to output USD file
            convert_to_surface: Override convert_to_surface setting
            compute_normals: Override compute_normals setting

        Returns:
            Usd.Stage: Created USD stage

        Raises:
            ValueError: If no valid meshes found
        """
        if convert_to_surface is not None:
            self.convert_to_surface = convert_to_surface
        if compute_normals is not None:
            self.settings.compute_normals = compute_normals

        self.logger.info(
            f"Converting {len(self.input_polydata)} meshes to {output_usd_file}"
        )

        # Remove existing file
        output_path = Path(output_usd_file)
        if output_path.exists():
            output_path.unlink()
            self.logger.debug(f"Removed existing file: {output_path}")

        # USD caches layers globally by identifier, so a prior call in the
        # same Python session can block CreateNew even after the file is
        # gone. Evict any stale in-memory layer for this path.
        stale_layer = Sdf.Layer.Find(str(output_path))
        if stale_layer is not None:
            stale_layer.Clear()
            del stale_layer

        # Create USD stage
        stage = Usd.Stage.CreateNew(str(output_path))
        UsdGeom.SetStageMetersPerUnit(stage, self.settings.meters_per_unit)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        # Create root
        root_path = f"/World/{self.data_basename}"
        UsdGeom.Xform.Define(stage, root_path)
        root_prim = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(root_prim)

        # Set time range for animation (not for static merge)
        if len(self.input_polydata) > 1 and not self._is_static_merge:
            time_codes = self._time_codes or [
                float(i) for i in range(len(self.input_polydata))
            ]
            stage.SetStartTimeCode(time_codes[0])
            stage.SetEndTimeCode(time_codes[-1])
            stage.SetTimeCodesPerSecond(self.settings.frames_per_second)

        # Initialize managers
        material_mgr = MaterialManager(stage)
        mesh_converter = UsdMeshConverter(stage, self.settings, material_mgr)

        # Process meshes. Labels win over the static layout: a per-cell label
        # array names the structures outright, which the static layout's one
        # prim per input mesh cannot.
        if self.mask_ids:
            # Split by anatomical regions
            self._convert_with_labels(stage, root_path, material_mgr, mesh_converter)
        elif self._is_static_merge:
            self._convert_static_merge(stage, root_path, material_mgr, mesh_converter)
        else:
            # Single mesh (or time series) conversion
            self._convert_unified(stage, root_path, material_mgr, mesh_converter)

        # Add a framing camera with tight near-clip so Omniverse Kit and other
        # USD viewers can zoom close without geometry vanishing.
        if add_framing_camera(stage) is None:
            self.logger.debug("Skipped framing camera: no stage bounds available")

        # Save stage
        stage.Save()
        self.logger.info(f"Saved USD file: {output_path}")

        return stage

    def _convert_unified(
        self,
        stage: Usd.Stage,
        root_path: str,
        material_mgr: MaterialManager,
        mesh_converter: UsdMeshConverter,
    ) -> None:
        """Convert all meshes as a single mesh (or split by connectivity/cell_type)."""
        self.logger.debug("Converting mesh(es), separate_by='%s'", self.separate_by)

        # Reuse pre-converted data built during topology validation in from_files();
        # fall back to computing fresh when called without the file-based factory.
        mesh_data_sequence = self._cached_mesh_data or [
            self._vtk_to_mesh_data(m, i) for i, m in enumerate(self.input_polydata)
        ]

        time_codes = self._time_codes or [
            float(i) for i in range(len(mesh_data_sequence))
        ]

        if self.separate_by == "none":
            # Single prim path for all time steps
            parts_per_frame = [[(md, "Mesh")] for md in mesh_data_sequence]
        elif self.separate_by == "connectivity":
            parts_per_frame = [
                split_mesh_data_by_connectivity(md, self.data_basename)
                for md in mesh_data_sequence
            ]
        else:  # cell_type
            parts_per_frame = [
                split_mesh_data_by_cell_type(md, self.data_basename)
                for md in mesh_data_sequence
            ]

        # Collect all part names across frames for stable prim paths
        all_part_names: list[str] = []
        for parts in parts_per_frame:
            for _, name in parts:
                if name not in all_part_names:
                    all_part_names.append(name)

        for part_name in all_part_names:
            material = self._create_material_from_colormap(f"{part_name}_material")
            material_mgr.get_or_create_material(material)

            # Collect frames that contain this part
            part_sequence = []
            part_time_codes = []
            for frame_idx, parts in enumerate(parts_per_frame):
                for md, name in parts:
                    if name == part_name:
                        md.material_id = material.name
                        part_sequence.append(md)
                        part_time_codes.append(time_codes[frame_idx])
                        break

            if not part_sequence:
                continue

            mesh_path = f"{root_path}/{part_name}"
            # Always use create_time_varying_mesh so the prim carries explicit time
            # samples and is only visible at the frames it was present in, even when
            # a part appears in only one frame.
            mesh_converter.create_time_varying_mesh(
                part_sequence, mesh_path, part_time_codes, bind_material=True
            )

    def _convert_static_merge(
        self,
        stage: Usd.Stage,
        root_path: str,
        material_mgr: MaterialManager,
        mesh_converter: UsdMeshConverter,
    ) -> None:
        """Write each input mesh as a separate prim with no time samples.

        Used when multiple files don't match a time-series pattern.
        """
        self.logger.debug(
            "Static merge: writing %d mesh(es) as separate prims",
            len(self.input_polydata),
        )
        for i, vtk_mesh in enumerate(self.input_polydata):
            mesh_data = self._vtk_to_mesh_data(vtk_mesh, i)
            frame_name = (
                self.object_names[i]
                if self.object_names is not None
                else f"{self.data_basename}_{i}"
            )

            if self.separate_by == "none":
                parts = [(mesh_data, frame_name)]
            elif self.separate_by == "connectivity":
                parts = split_mesh_data_by_connectivity(mesh_data, frame_name)
            else:  # cell_type
                parts = split_mesh_data_by_cell_type(mesh_data, frame_name)

            for part_md, part_name in parts:
                material = self._create_material_from_colormap(f"{part_name}_material")
                material_mgr.get_or_create_material(material)
                part_md.material_id = material.name
                mesh_converter.create_mesh(
                    part_md, f"{root_path}/{part_name}", bind_material=True
                )

    def _convert_with_labels(
        self,
        stage: Usd.Stage,
        root_path: str,
        material_mgr: MaterialManager,
        mesh_converter: UsdMeshConverter,
    ) -> None:
        """Convert meshes split by anatomical labels."""
        mask_ids = self.mask_ids
        assert mask_ids is not None
        self.logger.debug(f"Converting with {len(mask_ids)} anatomical labels")

        # Extract labeled meshes for each time step
        labeled_meshes_by_time = []
        for time_idx, vtk_mesh in enumerate(self.input_polydata):
            labeled_meshes = self._split_by_labels(vtk_mesh, time_idx)
            labeled_meshes_by_time.append(labeled_meshes)

        # Get all unique labels
        all_labels: set[str] = set()
        for labeled_meshes in labeled_meshes_by_time:
            all_labels.update(labeled_meshes.keys())

        # Track per-type Xforms/Scopes already defined so we only Define each
        # once even when multiple labels share a type.
        defined_type_groups: set[str] = set()

        # Convert each label separately
        for label_name in sorted(all_labels):
            self.logger.debug(f"Processing label: {label_name}")

            # Collect mesh data for this label across time, tracking which
            # original frame indices contribute so time codes stay aligned.
            label_mesh_sequence = []
            label_frame_indices: list[int] = []
            for time_idx, labeled_meshes in enumerate(labeled_meshes_by_time):
                if label_name in labeled_meshes:
                    label_mesh_sequence.append(labeled_meshes[label_name])
                    label_frame_indices.append(time_idx)
                else:
                    # Label not present in this time step - skip
                    self.logger.warning(f"Label '{label_name}' missing in time step")

            if not label_mesh_sequence:
                continue

            # Group labeled prims under an anatomy-type Xform so a 100-label
            # output is structured (heart/, lung/, bone/, ...) instead of flat.
            type_name = (
                self.segmenter.label_to_type(label_name)
                if self.segmenter is not None
                else "Anatomy"
            )
            if type_name not in defined_type_groups:
                UsdGeom.Xform.Define(stage, f"{root_path}/{type_name}")
                UsdGeom.Scope.Define(stage, f"/World/Looks/{type_name}")
                defined_type_groups.add(type_name)

            # Create material for this label. The "/" in the material name
            # makes MaterialManager author it at /World/Looks/{type}/{label}_material,
            # so material paths mirror the mesh hierarchy.
            material = self._create_material_from_colormap(
                f"{type_name}/{label_name}_material"
            )

            # Convert to USD
            mesh_path = f"{root_path}/{type_name}/{label_name}"
            if self._time_codes is not None:
                label_time_codes = [self._time_codes[i] for i in label_frame_indices]
            else:
                label_time_codes = [float(i) for i in label_frame_indices]
            for md in label_mesh_sequence:
                md.material_id = material.name
            material_mgr.get_or_create_material(material)
            mesh_converter.create_time_varying_mesh(
                label_mesh_sequence, mesh_path, label_time_codes, bind_material=True
            )

    def _vtk_to_mesh_data(
        self, vtk_mesh: pv.DataSet | vtk.vtkDataSet, time_idx: int
    ) -> MeshData:
        """Convert VTK/PyVista mesh to MeshData."""
        # Wrap VTK objects
        if isinstance(vtk_mesh, vtk.vtkDataSet):
            vtk_mesh = pv.wrap(vtk_mesh)

        # Extract surface if needed
        if self.convert_to_surface and not isinstance(vtk_mesh, pv.PolyData):
            if isinstance(vtk_mesh, pv.UnstructuredGrid):
                vtk_mesh = vtk_mesh.extract_surface(algorithm="dataset_surface")
            elif hasattr(vtk_mesh, "extract_surface"):
                vtk_mesh = vtk_mesh.extract_surface(algorithm="dataset_surface")
            elif hasattr(vtk_mesh, "extract_geometry"):
                vtk_mesh = vtk_mesh.extract_geometry()

        # Get points
        points = np.array(vtk_mesh.points, dtype=np.float64)

        # Get faces
        if hasattr(vtk_mesh, "faces"):
            faces = vtk_mesh.faces
            if len(faces) == 0 and not self.convert_to_surface:
                raise ValueError("Mesh has no faces - surface extraction may be needed")
            # Parse VTK face format: [n_points, i0, i1, ..., n_points, j0, j1, ...]
            face_counts_list: list[int] = []
            face_indices_list: list[int] = []
            idx = 0
            while idx < len(faces):
                n = int(faces[idx])
                face_counts_list.append(n)
                face_indices_list.extend([int(v) for v in faces[idx + 1 : idx + 1 + n]])
                idx += n + 1
            face_counts = np.array(face_counts_list, dtype=np.int32)
            face_indices = np.array(face_indices_list, dtype=np.int32)
        elif self.convert_to_surface:
            face_counts = np.array([], dtype=np.int32)
            face_indices = np.array([], dtype=np.int32)
        else:
            # No faces - might be point cloud or volumetric
            raise ValueError("Mesh has no faces - surface extraction may be needed")

        # Get normals
        normals = None
        if "Normals" in vtk_mesh.point_data:
            normals = np.array(vtk_mesh.point_data["Normals"], dtype=np.float64)

        # Get colors if using colormap
        colors = None
        if self.color_by_array and self.color_by_array in vtk_mesh.point_data:
            colors = self._apply_colormap(vtk_mesh.point_data[self.color_by_array])

        # Extract generic arrays from both point and cell data.
        # Point arrays are vertex-interpolated; cell arrays are uniform
        # (per-face) and are required for downstream colormap workflows that
        # target simulation fields such as stress or strain.
        generic_arrays = []
        for source, interpolation in (
            (vtk_mesh.point_data, "vertex"),
            (vtk_mesh.cell_data, "uniform"),
        ):
            for array_name in source.keys():
                array_data = source[array_name]
                num_components = int(array_data.shape[1] if array_data.ndim > 1 else 1)

                if array_data.dtype in [np.float32, np.float64]:
                    data_type = DataType.FLOAT
                elif array_data.dtype in [np.int32, np.int64]:
                    data_type = DataType.INT
                else:
                    data_type = DataType.FLOAT

                generic_arrays.append(
                    GenericArray(
                        name=array_name,
                        data=array_data,
                        num_components=num_components,
                        data_type=data_type,
                        interpolation=interpolation,
                    )
                )

        return MeshData(
            points=points,
            face_vertex_counts=face_counts,
            face_vertex_indices=face_indices,
            normals=normals,
            colors=colors,
            generic_arrays=generic_arrays,
        )

    def _split_by_labels(
        self, vtk_mesh: pv.DataSet | vtk.vtkDataSet, time_idx: int
    ) -> dict[str, MeshData]:
        """Split mesh by anatomical labels."""
        mask_ids = self.mask_ids
        assert mask_ids is not None
        # Wrap VTK objects
        if isinstance(vtk_mesh, (vtk.vtkPolyData, vtk.vtkUnstructuredGrid)):
            vtk_mesh = pv.wrap(vtk_mesh)

        # Extract surface if needed
        if isinstance(vtk_mesh, pv.UnstructuredGrid) and self.convert_to_surface:
            vtk_mesh = vtk_mesh.extract_surface(algorithm="dataset_surface")

        # Get per-cell label IDs. 'SegmentationLabelIds' is written by
        # ContourTools.save_combined_surfaces when merging per-label surfaces;
        # 'boundary_labels' comes from contouring a multi-label labelmap.
        if "SegmentationLabelIds" in vtk_mesh.cell_data:
            label_array = vtk_mesh.cell_data["SegmentationLabelIds"]
        elif "boundary_labels" in vtk_mesh.cell_data:
            label_array = vtk_mesh.cell_data["boundary_labels"]
        else:
            self.log_warning(
                "No 'SegmentationLabelIds' or 'boundary_labels' array found "
                "- using unified mesh"
            )
            return {"default": self._vtk_to_mesh_data(vtk_mesh, time_idx)}

        # Create submeshes for each label
        labeled_meshes = {}
        for label_id, label_name in mask_ids.items():
            # Extract cells with this label
            mask = label_array == label_id
            if not np.any(mask):
                continue

            # Create submesh
            cell_ids = np.where(mask)[0].astype(int).tolist()
            submesh = vtk_mesh.extract_cells(cell_ids)

            # Convert to MeshData
            labeled_meshes[label_name] = self._vtk_to_mesh_data(submesh, time_idx)

        return labeled_meshes

    def _apply_colormap(self, scalar_data: np.ndarray) -> np.ndarray:
        """Apply colormap to scalar data."""
        from matplotlib import colormaps

        # Map common/intuitive names to actual matplotlib colormap names
        colormap_aliases = {
            "heat": "hot",
            "grayscale": "gray",
            "greyscale": "grey",
            "jet": "jet",
            "random": "tab20",  # Good for categorical data
        }

        # Flatten to 1D if needed
        if scalar_data.ndim > 1:
            scalar_data = np.linalg.norm(scalar_data, axis=1)

        # Normalize
        if self.intensity_range:
            vmin, vmax = self.intensity_range
        else:
            vmin, vmax = np.min(scalar_data), np.max(scalar_data)

        if vmax > vmin:
            normalized = (scalar_data - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0.0, 1.0)
        else:
            normalized = np.ones_like(scalar_data) * 0.5

        # Get colormap name (use alias if available)
        cmap_name = colormap_aliases.get(self.colormap, self.colormap)

        # Apply colormap with fallback
        try:
            cmap = colormaps[cmap_name]
        except KeyError:
            self.logger.warning(
                f"Colormap '{self.colormap}' not found, falling back to 'viridis'"
            )
            cmap = colormaps["viridis"]

        colors_rgba = cmap(normalized)

        # Return RGB (drop alpha). The intermediate variable pins the type so
        # mypy doesn't lose track through .astype() - matplotlib's colormap
        # return signature is loose.
        colors_rgb: np.ndarray = colors_rgba[:, :3].astype(np.float32)
        return colors_rgb

    def _create_material_from_colormap(self, name: str) -> MaterialData:
        """Create material based on colormap settings."""
        if self.color_by_array:
            return MaterialData(
                name=name,
                diffuse_color=self.solid_color,
                roughness=0.5,
                metallic=0.0,
                use_vertex_colors=True,
            )
        else:
            return MaterialData(
                name=name,
                diffuse_color=self.solid_color,
                roughness=0.5,
                metallic=0.0,
                use_vertex_colors=False,
            )
