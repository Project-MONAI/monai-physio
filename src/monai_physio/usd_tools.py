"""
This module contains the USDTools class for manipulating USD objects and files.

This module provides utilities for working with Universal Scene Description (USD)
files in the context of medical visualization. It includes functions for merging
USD files, arranging objects in grids, computing bounding boxes, and preserving
materials and animations for visualization in NVIDIA Omniverse.

The tools are specifically designed for medical imaging workflows where multiple
anatomical structures need to be organized and visualized together.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pvtk
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from .monai_physio_base import MONAIPhysioBase
from .vtk_to_usd import add_framing_camera


class USDTools(MONAIPhysioBase):
    """
    Utilities for manipulating Universal Scene Description (USD) files.

    This class provides tools for working with USD files in medical visualization
    contexts, including merging multiple USD files, arranging objects in spatial
    grids, computing bounding boxes, and preserving materials and animations.

    USD (Universal Scene Description) is the foundation for 3D content in
    NVIDIA Omniverse and other modern 3D pipelines. This class facilitates
    the creation of complex medical visualizations by organizing anatomical
    structures from multiple sources.

    Key capabilities:
    - Merge multiple USD files while preserving hierarchy and materials
    - Arrange objects in spatial grids for comparison or overview
    - Compute bounding boxes for spatial layout
    - Preserve time-varying animation data
    - Handle material bindings and shader networks

    The class is designed to work with USD files generated from medical
    imaging data, particularly anatomical structures extracted from CT
    and MR images.

    Example:
        >>> usd_tools = USDTools()
        >>> # Merge multiple anatomical USD files
        >>> usd_tools.merge_usd_files(
        ...     'combined_anatomy.usd', ['heart.usd', 'lungs.usd', 'bones.usd']
        ... )
        >>> # Create grid arrangement for comparison
        >>> usd_tools.save_usd_file_arrangement(
        ...     'comparison_grid.usd', ['patient1.usd', 'patient2.usd', 'patient3.usd']
        ... )
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the USDTools class.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    def load_usd_as_vtk(
        self,
        usd_file: str | Path,
        prim_path: str = "/World",
        time_code: float | None = None,
    ) -> pvtk.PolyData:
        """Load USD mesh geometry as a PyVista ``PolyData``.

        Evaluates mesh points at ``time_code``, applies each mesh prim's
        local-to-world transform, and stores RGB colors in
        ``point_data['openusd_rgb']``. Authored ``displayColor`` is used when
        available; otherwise points are colored red. Coordinates are returned
        in the USD stage coordinate system.

        Args:
            usd_file: Path to a USD file.
            prim_path: Root prim path to traverse. Defaults to ``/World``.
            time_code: Optional time code for animated meshes. ``None`` reads
                default values and falls back to the first authored point time
                sample.

        Returns:
            A merged PyVista ``PolyData`` containing all mesh prims under
            ``prim_path``.

        Raises:
            FileNotFoundError: If ``usd_file`` does not exist.
            ValueError: If the stage, prim path, or mesh geometry is invalid.
        """
        usd_path = Path(usd_file)
        if not usd_path.exists():
            raise FileNotFoundError(f"USD file not found: {usd_path}")

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise ValueError(f"Could not open USD file: {usd_path}")

        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            raise ValueError(f"USD prim path not found: {prim_path}")

        if time_code is None:
            usd_time = Usd.TimeCode.Default()
        else:
            usd_time = Usd.TimeCode(time_code)
        xform_cache = UsdGeom.XformCache(usd_time)

        meshes: list[pvtk.PolyData] = []
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue

            mesh = UsdGeom.Mesh(prim)
            points_value = mesh.GetPointsAttr().Get(usd_time)
            if points_value is None and time_code is None:
                point_samples = mesh.GetPointsAttr().GetTimeSamples()
                if point_samples:
                    sample_time = Usd.TimeCode(point_samples[0])
                    points_value = mesh.GetPointsAttr().Get(sample_time)
            if points_value is None or len(points_value) == 0:
                continue

            face_counts = mesh.GetFaceVertexCountsAttr().Get(usd_time)
            face_indices = mesh.GetFaceVertexIndicesAttr().Get(usd_time)
            if face_counts is None or face_indices is None:
                continue

            world_matrix = xform_cache.GetLocalToWorldTransform(prim)
            # Vectorize the local-to-world transform: USD's Gf.Matrix4d uses
            # the convention `world_point = local_point_row_vec * M`, where
            # the matrix is row-major and the translation row is the last
            # row. Building a (N, 4) homogeneous-point block and multiplying
            # once is dramatically faster than calling Transform() per point
            # for large meshes.
            mat_array = np.array(
                [[float(world_matrix[i][j]) for j in range(4)] for i in range(4)],
                dtype=np.float64,
            )
            local_points = np.asarray(points_value, dtype=np.float64)
            homogeneous = np.empty((local_points.shape[0], 4), dtype=np.float64)
            homogeneous[:, :3] = local_points
            homogeneous[:, 3] = 1.0
            points = (homogeneous @ mat_array)[:, :3].astype(np.float32)
            if len(points) == 0:
                continue

            faces: list[int] = []
            index_offset = 0
            for count in face_counts:
                count_int = int(count)
                faces.append(count_int)
                faces.extend(
                    int(face_indices[index_offset + i]) for i in range(count_int)
                )
                index_offset += count_int
            if not faces:
                continue

            pv_mesh = pvtk.PolyData(points, np.asarray(faces, dtype=np.int64))
            rgb = self._usd_display_color(mesh, len(points), usd_time)
            pv_mesh.point_data["openusd_rgb"] = rgb
            meshes.append(pv_mesh)

        if not meshes:
            raise ValueError(f"No mesh geometry found in {usd_path} under {prim_path}")

        if len(meshes) == 1:
            return meshes[0]
        # pvtk.merge is loosely typed (it can return several DataSet subclasses
        # depending on inputs). All callers here pass PolyData with
        # merge_points=False, so the result is always PolyData.
        return cast(pvtk.PolyData, pvtk.merge(meshes, merge_points=False))

    @staticmethod
    def _usd_display_color(
        mesh: UsdGeom.Mesh,
        n_points: int,
        time_code: Usd.TimeCode,
    ) -> np.ndarray:
        """Return point RGB colors from ``displayColor`` or the red fallback."""
        fallback = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (n_points, 1))
        primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("displayColor")
        if not primvar:
            return fallback

        color_value = primvar.Get(time_code)
        if color_value is None:
            color_value = primvar.Get()
        if color_value is None or len(color_value) == 0:
            return fallback

        colors = np.asarray(color_value, dtype=np.float32)
        if colors.ndim != 2 or colors.shape[1] < 3:
            return fallback

        colors = colors[:, :3]
        interpolation = primvar.GetInterpolation()
        if interpolation in (UsdGeom.Tokens.constant, "constant"):
            colors = np.tile(colors[0], (n_points, 1))
        elif len(colors) == 1:
            colors = np.tile(colors[0], (n_points, 1))
        elif len(colors) != n_points:
            return fallback

        return np.asarray(np.clip(colors, 0.0, 1.0) * 255.0, dtype=np.uint8)

    def get_subtree_bounding_box(
        self, prim: UsdGeom.Xform
    ) -> tuple[Gf.Vec3f, Gf.Vec3f]:
        """
        Compute the axis-aligned bounding box of a USD primitive subtree.

        Recursively traverses a USD primitive hierarchy and computes the
        combined bounding box of all mesh geometry within the subtree.
        This is useful for spatial layout and positioning operations.

        Args:
            prim (UsdGeom.Xform): The root primitive of the subtree to analyze.
                Should be a UsdGeom.Xform or similar transformable primitive

        Returns:
            tuple[Gf.Vec3f, Gf.Vec3f]: Tuple containing:
                - bbox_min: Minimum corner of the bounding box
                - bbox_max: Maximum corner of the bounding box

        Example:
            >>> stage = Usd.Stage.Open('anatomy.usd')
            >>> heart_prim = stage.GetPrimAtPath('/World/Heart')
            >>> bbox_min, bbox_max = usd_tools.get_subtree_bounding_box(heart_prim)
            >>> center = (bbox_min + bbox_max) / 2
        """
        first_bbox = True
        bbox_min = np.array([0, 0, 0])
        bbox_max = np.array([0, 0, 0])

        def traverse_prim(current_prim: Any) -> None:
            nonlocal bbox_min, bbox_max, first_bbox
            if current_prim.IsA(UsdGeom.Mesh):
                bbox = UsdGeom.Boundable.ComputeExtentFromPlugins(
                    UsdGeom.Boundable(current_prim), Usd.TimeCode.Default()
                )
                # Skip if bbox computation returned None
                if bbox is None or len(bbox) != 2:
                    return

                if first_bbox:
                    bbox_min = bbox[0]
                    bbox_max = bbox[1]
                    first_bbox = False
                else:
                    bbox_min = np.minimum(bbox_min, bbox[0])
                    bbox_max = np.maximum(bbox_max, bbox[1])

            # Recursively traverse all children
            for child in current_prim.GetAllChildren():
                traverse_prim(child)

        traverse_prim(prim)

        # If no valid bounding boxes were found, return default values
        if first_bbox:
            self.log_warning(f"No valid bounding box found for prim: {prim.GetPath()}")
            return np.array([0, 0, 0]), np.array([0, 0, 0])

        return bbox_min, bbox_max

    def save_usd_file_arrangement(
        self, new_stage_name: str, usd_file_names: list[str]
    ) -> None:
        """
        Create a spatial grid arrangement of objects from multiple USD files.

        Takes a list of USD files and arranges them in a regular grid pattern
        for comparison or overview visualization. Each USD file is referenced
        into the new stage and positioned to avoid overlap. This is useful
        for comparing anatomical structures from different patients or time
        points.

        The grid layout is automatically computed based on the number of
        input files, creating approximately square arrangements. Objects
        are centered at their computed positions and spaced to avoid overlap.

        Args:
            new_stage_name (str): Path for the output USD file containing
                the arranged objects
            usd_file_names (list[str]): List of paths to USD files to arrange.
                Each file should contain anatomical structures under /World

        Note:
            The method preserves material bindings from the source files and
            applies spatial transforms to position objects in the grid.
            The first USD file in the list is used as the template for the
            new stage structure.

        Example:
            >>> # Create comparison grid of cardiac models
            >>> usd_tools.save_usd_file_arrangement(
            ...     'cardiac_comparison.usd',
            ...     [
            ...         'patient_001_heart.usd',
            ...         'patient_002_heart.usd',
            ...         'patient_003_heart.usd',
            ...         'patient_004_heart.usd',
            ...     ],
            ... )
        """
        new_stage = Usd.Stage.Open(usd_file_names[0])

        n_objects = len(usd_file_names)
        n_rows = int(np.floor(np.sqrt(n_objects)))
        n_cols = int(np.ceil(n_objects / n_rows))
        self.log_info("Grid layout: %d rows x %d cols", n_rows, n_cols)
        x_spacing = 0.4
        y_spacing = 0.4
        x_offset = -x_spacing * (n_cols - 1) / 2
        y_offset = -y_spacing * (n_rows - 1) / 2

        for i, usd_file_name in enumerate(usd_file_names):
            source_stage = Usd.Stage.Open(usd_file_name, Usd.Stage.LoadAll)

            source_root = source_stage.GetPrimAtPath("/World")
            children = source_root.GetChildren()
            for child in children:
                self.log_info("Copying %s:%s", usd_file_name, child.GetPrimPath())
                new_stage.DefinePrim(child.GetPrimPath()).GetReferences().AddReference(
                    assetPath=usd_file_name,
                    primPath=child.GetPrimPath(),
                )
                # Apply translation to t
                for grandchild in child.GetAllChildren():
                    self.log_debug("   Bounding box of %s", grandchild.GetPrimPath())
                    bbox_min, bbox_max = self.get_subtree_bounding_box(grandchild)
                    bbox_center = (bbox_min + bbox_max) / 2
                    self.log_debug("   Bounding box center: %s", bbox_center)

                    xform = UsdGeom.Xformable(grandchild)
                    if not xform.GetOrderedXformOps():
                        xform.AddTranslateOp()
                    xform_op = xform.GetOrderedXformOps()[
                        -1
                    ]  # Get the last transform op

                    # Calculate translation to position object center at grid position
                    grid_x = (i % n_cols) * x_spacing + x_offset
                    grid_y = (i // n_cols) * y_spacing + y_offset
                    translate = (
                        grid_x - bbox_center[0],
                        grid_y - bbox_center[1],
                        -bbox_center[2],
                    )
                    self.log_debug(
                        "   Translating %s to %s", grandchild.GetPrimPath(), translate
                    )
                    xform_op.Set(translate, Usd.TimeCode.Default())

            # Note: Material bindings are preserved through references/payloads,
            # so we don't need to explicitly rebind them. The code below is
            # commented out to avoid cross-layer material binding issues.
            #
            # for prim in source_stage.Traverse():
            #     if prim.IsA(UsdGeom.Mesh):
            #         bindingAPI = UsdShade.MaterialBindingAPI(prim)
            #         mesh_material = bindingAPI.ComputeBoundMaterial()
            #         if bool(mesh_material):
            #             material_path = (
            #                 str(mesh_material[0].GetPath())
            #                 if isinstance(mesh_material, tuple)
            #                 and len(mesh_material) > 0
            #                 else str(mesh_material.GetPath())
            #             )
            #             self.log_debug(
            #                 "   Mesh %s has material %s",
            #                 prim.GetPrimPath(),
            #                 material_path,
            #             )
            #             new_prim = new_stage.GetPrimAtPath(prim.GetPrimPath())
            #             material = UsdShade.Material.Get(new_stage, material_path)
            #             if new_prim is not None and new_prim.IsValid() and material:
            #                 binding_api = UsdShade.MaterialBindingAPI.Apply(new_prim)
            #                 binding_api.Bind(material)
            #             else:
            #                 self.log_warning(
            #                     "      Cannot bind. No new prim found for %s",
            #                     prim.GetPrimPath(),
            #                 )

        # Framing camera with tight near-clip for Omniverse Kit viewer ergonomics.
        add_framing_camera(new_stage)

        self.log_info("Exporting stage...")
        new_stage.Export(new_stage_name)

    def merge_usd_files(
        self, output_filename: str, input_filenames_list: list[str]
    ) -> None:
        """
        Merge multiple USD files into a single comprehensive USD file.

        Combines multiple USD files while preserving all essential data
        including object hierarchies, transforms, materials, shaders,
        and time-varying animation data. This is useful for creating
        complete anatomical scenes from individually processed structures.

        The merging process:
        1. Creates a new USD stage with proper metadata
        2. Copies all primitive hierarchies from input files
        3. Preserves all attributes including time-sampled data
        4. Maintains material bindings and shader networks
        5. Handles coordinate system and units consistently

        Args:
            output_filename (str): Path for the merged output USD file.
                Should have .usd or .usda extension
            input_filenames_list (list[str]): List of input USD file paths
                to merge. Each should be a valid USD file with compatible
                coordinate systems

        Note:
            The merged file stores coordinates in meters (metersPerUnit=1.0)
            with upAxis="Y", which are standard for Omniverse.
            Time-varying data (animations) are preserved across all time samples.

        Example:
            >>> # Merge anatomical components into complete scene
            >>> usd_tools.merge_usd_files(
            ...     'complete_anatomy.usd', ['heart_dynamic.usd', 'lungs_static.usd', 'skeleton.usd']
            ... )
        """
        # Remove any existing file and evict any stale in-memory USD layer.
        # USD caches layers globally by identifier, so a prior call in the
        # same Python session can block CreateNew even after the file is gone.
        output_path = Path(output_filename)
        if output_path.exists():
            output_path.unlink()
        stale_layer = Sdf.Layer.Find(str(output_path))
        if stale_layer is not None:
            stale_layer.Clear()
            del stale_layer

        # Create new stage with meters as units (standard USD configuration)
        stage = Usd.Stage.CreateNew(output_filename)
        stage.SetMetadata("metersPerUnit", 1.0)
        stage.SetMetadata("upAxis", "Y")

        # Define root prim for organization
        root_prim = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(root_prim)

        # Track time range across all input files for stage metadata
        global_start_time = float("inf")
        global_end_time = float("-inf")
        time_codes_per_second = None
        frames_per_second = None

        for i, input_path in enumerate(input_filenames_list):
            # Open input stage with time-sampling enabled
            input_stage = Usd.Stage.Open(input_path, Usd.Stage.LoadAll)

            # Track time range from this input file
            start_time = input_stage.GetStartTimeCode()
            end_time = input_stage.GetEndTimeCode()
            global_start_time = min(global_start_time, start_time)
            global_end_time = max(global_end_time, end_time)

            # Capture time codes per second from first file
            if time_codes_per_second is None:
                time_codes_per_second = input_stage.GetTimeCodesPerSecond()
                frames_per_second = input_stage.GetFramesPerSecond()

            # Copy all root prims from input
            for prim in input_stage.GetPseudoRoot().GetAllChildren():
                new_path = "/" + prim.GetName()
                self.log_info("Copying %s to %s", prim.GetPrimPath(), new_path)

                # Recursively copy prim hierarchy with all attributes and time samples
                def _copy_prim(src_prim: Any, target_path: str) -> None:
                    # Create new prim with same type
                    new_prim = stage.DefinePrim(target_path, src_prim.GetTypeName())

                    # Copy properties and metadata
                    for attr in src_prim.GetAttributes():
                        if attr.GetName() == "deformationMagnitude":
                            continue

                        new_attr = new_prim.CreateAttribute(
                            attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom()
                        )

                        # Copy default value if it exists
                        if attr.HasValue():
                            value = attr.Get()
                            # Skip if value is None or invalid
                            if value is not None:
                                try:
                                    new_attr.Set(value)
                                except Exception as e:
                                    self.log_warning(
                                        f"Failed to copy attribute {attr.GetName()}: {e}"
                                    )

                        # Copy all time samples for time-varying attributes
                        time_samples = attr.GetTimeSamples()
                        if time_samples:
                            for time in time_samples:
                                new_attr.Set(attr.Get(time), time)

                        # Copy attribute connections (critical for material shader networks)
                        connections = attr.GetConnections()
                        if connections:
                            new_attr.SetConnections(connections)

                    # Copy relationships (important for material connections)
                    for rel in src_prim.GetRelationships():
                        new_rel = new_prim.CreateRelationship(
                            rel.GetName(), custom=rel.IsCustom()
                        )
                        targets = rel.GetTargets()
                        if targets:
                            new_rel.SetTargets(targets)

                    # Copy transforms if applicable
                    if src_prim.IsA(UsdGeom.Xformable):
                        xform = UsdGeom.Xformable(new_prim)
                        xform_op = xform.GetTransformOp()
                        if src_prim.HasAttribute("xformOp:transform"):
                            src_xform = UsdGeom.Xformable(src_prim)
                            xform_op.Set(src_xform.GetLocalTransformation())

                    # Recurse through children
                    for child in src_prim.GetChildren():
                        child_path = f"{target_path}/{child.GetName()}"
                        _copy_prim(child, child_path)

                _copy_prim(prim, new_path)

            # Copy material bindings from source stage to target stage
            for prim in input_stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    bindingAPI = UsdShade.MaterialBindingAPI(prim)
                    mesh_material = bindingAPI.ComputeBoundMaterial()
                    if bool(mesh_material):
                        # Get material path from source
                        material_path = (
                            str(mesh_material[0].GetPath())
                            if isinstance(mesh_material, tuple)
                            and len(mesh_material) > 0
                            else str(mesh_material.GetPath())
                        )
                        self.log_debug(
                            "   Binding material %s to %s",
                            material_path,
                            prim.GetPrimPath(),
                        )
                        # Get corresponding mesh prim and material in target stage
                        new_prim = stage.GetPrimAtPath(prim.GetPrimPath())
                        material = UsdShade.Material.Get(stage, material_path)
                        if new_prim is not None and new_prim.IsValid():
                            if material and material.GetPrim().IsValid():
                                binding_api = UsdShade.MaterialBindingAPI.Apply(
                                    new_prim
                                )
                                binding_api.Bind(material)
                            else:
                                self.log_warning(
                                    "      Material not found at %s in target stage",
                                    material_path,
                                )
                        else:
                            self.log_warning(
                                "      Cannot bind material. No mesh prim found at %s",
                                prim.GetPrimPath(),
                            )

        # Set stage time range metadata for animation playback
        if global_start_time != float("inf") and global_end_time != float("-inf"):
            stage.SetStartTimeCode(global_start_time)
            stage.SetEndTimeCode(global_end_time)
            if time_codes_per_second is not None:
                stage.SetTimeCodesPerSecond(time_codes_per_second)
            if frames_per_second is not None:
                stage.SetFramesPerSecond(frames_per_second)
            self.log_info(
                "Set stage time range: %.1f to %.1f", global_start_time, global_end_time
            )
            self.log_info(
                "Time codes per second: %s, Frames per second: %s",
                time_codes_per_second,
                frames_per_second,
            )

        # Framing camera with tight near-clip for Omniverse Kit viewer ergonomics.
        add_framing_camera(stage)

        # Save with USDA format
        # stage.GetRootLayer().Export(output_path, args=['--usdFormat', 'usda'])
        stage.Export(output_filename)

    def merge_usd_files_flattened(
        self, output_filename: str, input_filenames_list: list[str]
    ) -> None:
        """
        Merge multiple USD files using references and flattening.

        This method uses USD's native composition system (references) and then flattens
        the result into a self-contained file. This approach is simpler (~50 lines vs
        ~150 lines) and leverages USD's built-in composition engine.

        The method properly preserves:
            - All materials and MDL shader networks
            - Time-varying animation data with correct time codes
            - Material bindings to geometry
            - Stage metadata (TimeCodesPerSecond, time range, etc.)

        Args:
            output_filename (str): Path for the merged output USD file.
                Should have .usd or .usda extension
            input_filenames_list (list[str]): List of input USD file paths
                to merge. Each should be a valid USD file with compatible
                coordinate systems

        Comparison to merge_usd_files():
            - **merge_usd_files()**: More control, can skip specific attributes
            - **merge_usd_files_flattened()**: Simpler, faster, USD-native approach

        Both methods produce equivalent results for most use cases. Use the flattened
        method unless you need fine-grained control over what gets copied.

        Example:
            >>> usd_tools = USDTools()
            >>> usd_tools.merge_usd_files_flattened(
            ...     'complete_anatomy.usd', ['heart_dynamic.usd', 'lungs_static.usd']
            ... )
        """
        # Create temporary in-memory stage for composition
        temp_stage = Usd.Stage.CreateInMemory()

        # Set standard metadata (meters and Y-up for Omniverse)
        temp_stage.SetMetadata("metersPerUnit", 1.0)
        temp_stage.SetMetadata("upAxis", "Y")

        # Define root prim for organization
        root_prim = temp_stage.DefinePrim("/World", "Xform")
        temp_stage.SetDefaultPrim(root_prim)

        # Track time range across all input files for stage metadata
        global_start_time = float("inf")
        global_end_time = float("-inf")
        time_codes_per_second = None
        frames_per_second = None

        # Add references to all input files
        num_files = len(input_filenames_list)
        for idx, input_path in enumerate(input_filenames_list):
            self.log_progress(idx + 1, num_files, prefix="Referencing files")
            input_stage = Usd.Stage.Open(input_path, Usd.Stage.LoadAll)

            # Track time range from this input file
            start_time = input_stage.GetStartTimeCode()
            end_time = input_stage.GetEndTimeCode()
            global_start_time = min(global_start_time, start_time)
            global_end_time = max(global_end_time, end_time)

            # Capture time codes per second from first file
            if time_codes_per_second is None:
                time_codes_per_second = input_stage.GetTimeCodesPerSecond()
                frames_per_second = input_stage.GetFramesPerSecond()

            # Reference each top-level prim from the input file
            for prim in input_stage.GetPseudoRoot().GetAllChildren():
                new_path = "/" + prim.GetName()
                self.log_debug(
                    "  Adding reference: %s -> %s", prim.GetPrimPath(), new_path
                )

                # Create prim and add reference to source file
                temp_stage.DefinePrim(new_path).GetReferences().AddReference(
                    assetPath=input_path, primPath=prim.GetPrimPath()
                )

        # Set time range metadata on temporary stage before flattening
        if global_start_time != float("inf") and global_end_time != float("-inf"):
            temp_stage.SetStartTimeCode(global_start_time)
            temp_stage.SetEndTimeCode(global_end_time)
            if time_codes_per_second is not None:
                temp_stage.SetTimeCodesPerSecond(time_codes_per_second)
            if frames_per_second is not None:
                temp_stage.SetFramesPerSecond(frames_per_second)
            self.log_info(
                "Time range: %.1f to %.1f", global_start_time, global_end_time
            )
            self.log_info(
                "Time codes per second: %s, Frames per second: %s",
                time_codes_per_second,
                frames_per_second,
            )

        # Flatten the composed stage into a single layer
        # This resolves all references and bakes everything into one file
        self.log_info("Flattening composed stage...")
        flattened_layer = temp_stage.Flatten()

        # Create output stage from flattened layer
        output_stage = Usd.Stage.Open(flattened_layer)

        # Set time metadata on the output stage (must be done AFTER flattening)
        # This is critical - the flattened layer doesn't inherit metadata from temp_stage
        if global_start_time != float("inf") and global_end_time != float("-inf"):
            output_stage.SetStartTimeCode(global_start_time)
            output_stage.SetEndTimeCode(global_end_time)
            if time_codes_per_second is not None:
                output_stage.SetTimeCodesPerSecond(time_codes_per_second)
                self.log_info(
                    "Set output TimeCodesPerSecond: %s", time_codes_per_second
                )
            if frames_per_second is not None:
                output_stage.SetFramesPerSecond(frames_per_second)
                self.log_info("Set output FramesPerSecond: %s", frames_per_second)

        # Framing camera with tight near-clip for Omniverse Kit viewer ergonomics.
        add_framing_camera(output_stage)

        # Export the flattened layer with corrected metadata
        self.log_info("Exporting to %s", output_filename)
        output_stage.Export(output_filename)

    def list_mesh_primvars(
        self,
        stage_or_path: Usd.Stage | str,
        mesh_path: str,
        time_code: float | None = None,
    ) -> list[dict]:
        """
        List all primvars on a USD mesh with metadata.

        Inspects a mesh and returns information about each primvar including
        name, type, interpolation, time samples, and value range when feasible.
        This is useful for understanding what simulation data is available on
        the mesh for visualization.

        Args:
            stage_or_path: USD Stage or path to USD file
            mesh_path: Path to mesh prim (e.g., "/World/Meshes/MyMesh")
            time_code: Optional time code to sample values. If None, uses default.

        Returns:
            list[dict]: List of primvar metadata dictionaries containing:
                - name: Primvar name
                - type_name: USD type name (e.g., "float[]", "color3f[]")
                - interpolation: Interpolation mode ("vertex", "uniform", "constant")
                - num_time_samples: Number of time samples (0 if static)
                - elements: Number of elements in the array
                - range: Tuple (min, max) for numeric arrays, None otherwise

        Example:
            >>> usd_tools = USDTools()
            >>> primvars = usd_tools.list_mesh_primvars("valve.usd", "/World/Meshes/Valve")
            >>> for pv in primvars:
            ...     print(f"{pv['name']}: {pv['interpolation']}, {pv['elements']} elements")
        """
        # Open stage if needed
        if isinstance(stage_or_path, str):
            stage = Usd.Stage.Open(stage_or_path)
        else:
            stage = stage_or_path

        # Get mesh prim
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid():
            raise ValueError(f"Invalid mesh prim at path: {mesh_path}")

        if not mesh_prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Prim at {mesh_path} is not a Mesh")

        mesh = UsdGeom.Mesh(mesh_prim)
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        primvars = primvars_api.GetPrimvars()

        # Use provided time code or default
        tc = (
            Usd.TimeCode(time_code) if time_code is not None else Usd.TimeCode.Default()
        )

        result = []
        for primvar in primvars:
            pv_info = {
                "name": primvar.GetPrimvarName(),
                "type_name": str(primvar.GetTypeName()),
                "interpolation": primvar.GetInterpolation(),
                "num_time_samples": primvar.GetAttr().GetNumTimeSamples(),
                "elements": 0,
                "range": None,
            }

            # Get value at time code
            try:
                value = primvar.Get(tc)
                if value is not None:
                    pv_info["elements"] = len(value) if hasattr(value, "__len__") else 1

                    # Compute range for numeric types
                    if hasattr(value, "__iter__") and len(value) > 0:
                        try:
                            # Convert to numpy for easy min/max
                            arr = np.asarray(value)
                            if np.issubdtype(arr.dtype, np.number):
                                pv_info["range"] = (
                                    float(np.min(arr)),
                                    float(np.max(arr)),
                                )
                        except (TypeError, ValueError):
                            pass  # Skip range for non-numeric data
            except Exception as e:
                self.log_debug(
                    f"Could not get value for primvar {pv_info['name']}: {e}"
                )

            result.append(pv_info)

        return result

    def pick_color_primvar(
        self,
        primvar_infos: list[dict[str, Any]],
        keywords: tuple[str, ...] = ("strain", "stress"),
    ) -> str | None:
        """
        Select a primvar for coloring based on keywords and preferences.

        Examines a list of primvar metadata and picks the best candidate for
        default coloring visualization. Prefers primvars containing keywords
        like "strain" or "stress" that are commonly used in biomechanical
        simulations.

        Selection priority:
        1. Name contains first keyword ("strain") over later keywords ("stress")
        2. Vertex interpolation preferred over uniform (face) interpolation
        3. Alphabetically first if multiple candidates tie

        Args:
            primvar_infos: List of primvar metadata dicts (from list_mesh_primvars)
            keywords: Tuple of keywords to search for in primvar names (case-insensitive)

        Returns:
            str | None: Name of selected primvar, or None if no candidates found

        Example:
            >>> primvars = usd_tools.list_mesh_primvars("valve.usd", "/World/Meshes/Valve")
            >>> color_primvar = usd_tools.pick_color_primvar(primvars)
            >>> print(f"Selected for coloring: {color_primvar}")
        """
        candidates: list[tuple[dict[str, Any], int]] = []

        for pv in primvar_infos:
            name_lower = pv["name"].lower()
            for keyword_idx, keyword in enumerate(keywords):
                if keyword in name_lower:
                    candidates.append((pv, keyword_idx))
                    break

        if not candidates:
            return None

        # Sort by: keyword index, interpolation (vertex=0, else=1), name
        def sort_key(item: tuple[dict[str, Any], int]) -> tuple[int, int, str]:
            pv, kw_idx = item
            interp_priority = 0 if str(pv.get("interpolation")) == "vertex" else 1
            return (int(kw_idx), int(interp_priority), str(pv.get("name")))

        candidates.sort(key=sort_key)
        name_obj = candidates[0][0].get("name")
        if name_obj is None:
            return None
        return str(name_obj)

    def apply_colormap_from_primvar(
        self,
        stage_or_path: Usd.Stage | str,
        mesh_path: str,
        source_primvar: str,
        *,
        cmap: str = "viridis",
        time_codes: list[float] | None = None,
        intensity_range: tuple[float, float] | None = None,
        use_sigmoid_scale: bool = False,
        write_default_at_t0: bool = True,
        bind_vertex_color_material: bool = True,
    ) -> None:
        """
        Apply colormap visualization by converting a primvar to displayColor.

        Reads numeric data from a source primvar (like vtk_cell_stress or
        vtk_point_displacement) and generates RGB vertex colors using a matplotlib
        colormap. Writes these colors to the mesh's displayColor primvar and
        optionally binds a material that uses vertex colors for rendering.

        This is especially useful for post-processing USD files to add default
        visualization colors based on simulation data like stress or strain fields.

        Key features:

        - Handles multi-component data (vectors/tensors) by computing magnitude
        - Converts uniform (per-face) data to vertex data by averaging
        - Computes global value range across all time samples for consistent coloring
          (or uses intensity_range when provided)
        - Writes both default and time-sampled displayColor for Omniverse compatibility

        Args:
            stage_or_path: USD Stage or path to USD file
            mesh_path: Path to mesh prim (e.g., "/World/Meshes/MyMesh")
            source_primvar: Name of primvar to visualize (e.g., "vtk_cell_stress")
            cmap: Matplotlib colormap name (default: "viridis")
            time_codes: List of time codes to process. If None, uses stage time range.
            intensity_range: Optional (vmin, vmax) for colormap. If None,
                computed from data.
            use_sigmoid_scale: If True, use sigmoid scale for colormap normalization.
            write_default_at_t0: If True, also write default value at t=0
            bind_vertex_color_material: If True, create/bind material using displayColor

        Raises:
            ValueError: If mesh or primvar not found
            ImportError: If matplotlib is not available

        Example:
            >>> usd_tools = USDTools()
            >>> usd_tools.apply_colormap_from_primvar(
            ...     "valve.usd",
            ...     "/World/Meshes/Valve",
            ...     "vtk_cell_stress",
            ...     cmap="plasma"
            ... )
        """
        # Check matplotlib availability
        try:
            from matplotlib import colormaps as mpl_colormaps
        except ImportError:
            raise ImportError(
                "matplotlib is required for colormap coloring. "
                "Install with: pip install matplotlib"
            )

        # Open stage if needed
        if isinstance(stage_or_path, str):
            stage = Usd.Stage.Open(stage_or_path)
            stage_path = stage_or_path
        else:
            stage = stage_or_path
            stage_path = None

        # Get mesh prim
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid():
            raise ValueError(f"Invalid mesh prim at path: {mesh_path}")

        if not mesh_prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Prim at {mesh_path} is not a Mesh")

        mesh = UsdGeom.Mesh(mesh_prim)

        # Get source primvar
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        source_pv = primvars_api.GetPrimvar(source_primvar)
        if not source_pv:
            raise ValueError(
                f"Primvar '{source_primvar}' not found on mesh {mesh_path}"
            )

        # Determine time codes to process
        if time_codes is None:
            # Prefer the source primvar's authored samples (avoid inventing in-between frames).
            pv_samples = list(source_pv.GetAttr().GetTimeSamples())
            if pv_samples:
                time_codes = pv_samples
            else:
                # Fallback to points samples; last resort is default time.
                pts_samples = list(mesh.GetPointsAttr().GetTimeSamples())
                if pts_samples:
                    time_codes = pts_samples
                elif stage.HasAuthoredTimeCodeRange():
                    time_codes = [float(stage.GetStartTimeCode())]
                else:
                    time_codes = [Usd.TimeCode.Default().GetValue()]

        # Get mesh topology (needed for uniform->vertex conversion)
        # For time-varying meshes, get topology at the first time code
        first_time = (
            Usd.TimeCode(time_codes[0]) if time_codes else Usd.TimeCode.Default()
        )
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(first_time)
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(first_time)
        points_attr = mesh.GetPointsAttr()
        points_data = points_attr.Get(first_time)
        if points_data is None:
            self.log_error(f"Cannot get points data for mesh at {mesh_path}")
            return
        n_points = len(points_data)

        source_interp = source_pv.GetInterpolation()
        element_size = int(source_pv.GetElementSize() or 1)

        # Process all time samples to compute global range
        self.log_info(
            f"Processing {len(time_codes)} time samples for primvar '{source_primvar}'"
        )
        scalar_samples: list[tuple[float, np.ndarray]] = []
        n_faces = len(face_vertex_counts) if face_vertex_counts is not None else 0

        for tc in time_codes:
            time_code = Usd.TimeCode(tc)
            values = source_pv.Get(time_code)

            if values is None:
                self.log_warning(
                    f"No values for primvar '{source_primvar}' at time {tc}"
                )
                continue

            # Convert to numpy array
            arr = np.asarray(values)

            # If the primvar is stored as a flattened array with an elementSize, reshape it
            # back to (N, elementSize) so multi-component reduction works.
            if arr.ndim == 1:
                inferred = None
                if element_size > 1 and len(arr) % element_size == 0:
                    inferred = element_size
                else:
                    # Try to infer element size from expected element count.
                    expected = n_points if source_interp == "vertex" else n_faces
                    if expected and len(arr) % expected == 0 and len(arr) != expected:
                        inferred = len(arr) // expected
                if inferred and inferred > 1 and len(arr) % inferred == 0:
                    arr = arr.reshape(-1, int(inferred))

            # Reduce multi-component to scalar magnitude
            if arr.ndim == 2 and arr.shape[1] > 1:
                scalar = np.linalg.norm(arr, axis=1)
            elif arr.ndim == 1:
                scalar = arr
            else:
                scalar = arr.flatten()

            # Convert uniform (per-face) to vertex (per-point)
            if source_interp == "uniform":
                if len(scalar) != n_faces:
                    self.log_warning(
                        f"Skipping time {tc} for primvar '{source_primvar}': "
                        f"size mismatch (got {len(scalar)}, expected {n_faces} faces)"
                    )
                    continue
                vertex_scalar = self._uniform_to_vertex_scalar(
                    scalar, face_vertex_counts, face_vertex_indices, n_points
                )
            elif source_interp == "vertex":
                if len(scalar) != n_points:
                    self.log_warning(
                        f"Skipping time {tc} for primvar '{source_primvar}': "
                        f"size mismatch (got {len(scalar)}, expected {n_points} points)"
                    )
                    continue
                vertex_scalar = scalar
            else:
                raise ValueError(
                    f"Unsupported interpolation '{source_interp}' for primvar '{source_primvar}'"
                )

            scalar_samples.append(
                (float(tc), np.asarray(vertex_scalar, dtype=np.float32))
            )

        if not scalar_samples:
            raise ValueError(f"No valid data found for primvar '{source_primvar}'")

        # Value range: use provided intensity_range or compute from data
        if intensity_range is not None:
            try:
                vmin, vmax = float(intensity_range[0]), float(intensity_range[1])
            except (TypeError, IndexError) as e:
                raise ValueError(
                    "intensity_range must be a sequence of two floats (vmin, vmax)"
                ) from e
            if not (np.isfinite(vmin) and np.isfinite(vmax)):
                raise ValueError(
                    f"intensity_range values must be finite; got ({vmin}, {vmax})"
                )
            if vmin >= vmax:
                vmin, vmax = vmax, vmin
                self.log_info(
                    f"intensity_range was (vmax, vmin); swapped to {vmin:.6g} to {vmax:.6g}"
                )
            self.log_info(f"Using specified intensity range: {vmin:.6g} to {vmax:.6g}")
        else:
            all_values = np.concatenate([s for _, s in scalar_samples])
            vmin = float(np.min(all_values))
            vmax = float(np.max(all_values))
            self.log_info(f"Value range: {vmin:.6g} to {vmax:.6g}")

        # Apply colormap to each time sample
        try:
            cmap_obj = mpl_colormaps[cmap]
        except KeyError:
            raise ValueError(
                f"Colormap '{cmap}' not found. "
                f"Available: {', '.join(list(mpl_colormaps.keys())[:10])}..."
            )

        # Create or get displayColor primvar
        from pxr import Gf, Sdf, Vt

        display_color_pv = primvars_api.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        # If we're rewriting displayColor, clear any previously-authored time samples first.
        # This prevents leaving behind stale/corrupt samples at times we no longer author.
        try:
            dc_attr = display_color_pv.GetAttr()
            for t in list(dc_attr.GetTimeSamples()):
                dc_attr.ClearAtTime(t)
        except Exception:
            # Silently ignore errors (e.g., if attribute doesn't exist yet or has no samples).
            # This is expected on first-time creation or when no time samples are present.
            pass

        for idx, (tc, scalar) in enumerate(scalar_samples):
            # Normalize to [0, 1]

            if vmax > vmin:
                normalized = (scalar - vmin) / (vmax - vmin)
            else:
                normalized = np.full_like(scalar, 0.5)

            if use_sigmoid_scale:
                normalized = 1 / (1 + np.exp(-4 * (normalized - 0.5)))

            normalized = np.clip(normalized, 0.0, 1.0)

            # Apply colormap
            rgba = cmap_obj(normalized)
            rgb = rgba[:, :3].astype(np.float32)
            if len(rgb) != n_points:
                self.log_warning(
                    f"Skipping displayColor write at time {tc}: "
                    f"color length {len(rgb)} != n_points {n_points}"
                )
                continue

            # Convert to USD Vec3f array
            color_array = Vt.Vec3fArray(
                [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in rgb]
            )

            time_code = Usd.TimeCode(tc)

            # Write default at t=0 for Omniverse compatibility
            if write_default_at_t0 and idx == 0:
                display_color_pv.Set(color_array)

            # Write time sample
            display_color_pv.Set(color_array, time_code)

        self.log_info(f"Wrote displayColor primvar with {len(time_codes)} time samples")

        # Bind vertex color material if requested
        if bind_vertex_color_material:
            self._ensure_vertex_color_material(stage, mesh_prim)

        # Save stage if we opened it from a path
        if stage_path:
            stage.Save()
            self.log_info(f"Saved USD file: {stage_path}")

    def set_solid_display_color(
        self,
        stage_or_path: Usd.Stage | str,
        mesh_path: str,
        color: tuple[float, float, float],
        *,
        time_codes: list[float] | None = None,
        bind_vertex_color_material: bool = True,
    ) -> None:
        """
        Set a constant (solid) displayColor for a mesh.

        Fills the mesh's displayColor primvar with the same RGB for every vertex,
        optionally at each time code for animated meshes, and binds the vertex
        color material so the color is visible in Omniverse.

        Args:
            stage_or_path: USD Stage or path to USD file
            mesh_path: Path to mesh prim (e.g., "/World/Meshes/MyMesh")
            color: RGB tuple in [0, 1] (e.g., (1, 0, 0) for red)
            time_codes: If provided, set displayColor at each time. If None, set default only.
            bind_vertex_color_material: If True, bind material that uses displayColor
        """
        from pxr import Gf, Sdf, Vt

        if isinstance(stage_or_path, str):
            stage = Usd.Stage.Open(stage_or_path)
            stage_path = stage_or_path
        else:
            stage = stage_or_path
            stage_path = None

        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Invalid mesh prim at path: {mesh_path}")

        mesh = UsdGeom.Mesh(mesh_prim)
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        points_attr = mesh.GetPointsAttr()

        # Resolve time codes: default only or at each sample
        vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))

        display_color_pv = primvars_api.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        display_color_attr = display_color_pv.GetAttr()
        # Clear any existing authored default and time samples to avoid stale colors
        if display_color_attr:
            display_color_attr.Clear()

        if time_codes is None:
            # Default time: get points and set primvar without an explicit time code
            pts = points_attr.Get()
            n_points = len(pts) if pts is not None else 0
            if n_points > 0:
                color_array = Vt.Vec3fArray([vec] * n_points)
                display_color_pv.Set(color_array)
        else:
            default_point_count: int | None = None
            for tc in time_codes:
                # Normalize to a Usd.TimeCode
                usd_tc = tc if isinstance(tc, Usd.TimeCode) else Usd.TimeCode(tc)

                # Get point count at this time
                pts = points_attr.Get(usd_tc)
                n_points = len(pts) if pts is not None else 0
                if n_points == 0 and usd_tc.IsDefault():
                    # Fallback: use time-independent points if default has no sample
                    pts = points_attr.Get()
                    n_points = len(pts) if pts is not None else 0
                if n_points == 0:
                    continue
                if default_point_count is None:
                    default_point_count = n_points
                color_array = Vt.Vec3fArray([vec] * n_points)
                if usd_tc.IsDefault():
                    display_color_pv.Set(color_array)
                else:
                    display_color_pv.Set(color_array, usd_tc)

            # Author a default (time-independent) value so consumers that query the
            # default when not time-scrubbing still see the solid color.
            if default_point_count is not None:
                default_color_array = Vt.Vec3fArray([vec] * default_point_count)
                display_color_pv.Set(default_color_array)  # , Usd.TimeCode.Default())

        if bind_vertex_color_material:
            self._ensure_vertex_color_material(stage, mesh_prim)
        if stage_path:
            stage.Save()
        self.log_info(f"Set solid displayColor on {mesh_path}")

    def list_mesh_paths_under(
        self, stage_or_path: Usd.Stage | str, parent_path: str = "/World/Meshes"
    ) -> list[str]:
        """
        List paths of all mesh prims at any depth under a parent path.

        Descends the whole subtree, so meshes written under an intermediate
        Xform - as :class:`ConvertVTKToUSD` writes labeled structures, at
        ``/World/{basename}/{anatomy_group}/{structure}`` - are found too.

        Args:
            stage_or_path: USD Stage or path to USD file
            parent_path: Parent prim path (default: /World/Meshes)

        Returns:
            List of mesh prim paths (e.g. ["/World/Meshes/Mesh0", "/World/Meshes/Mesh1"])
        """
        if isinstance(stage_or_path, str):
            stage = Usd.Stage.Open(stage_or_path)
        else:
            stage = stage_or_path

        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            return []
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(parent)
            if prim.IsA(UsdGeom.Mesh)
        ]

    def repair_mesh_primvar_element_sizes(
        self,
        stage_or_path: Usd.Stage | str,
        mesh_path: str,
        *,
        time_code: float | None = None,
        save: bool = True,
    ) -> dict:
        """
        Repair missing/incorrect primvar elementSize metadata for a mesh.

        Some multi-component primvars (e.g. 9-component stress tensors) may be authored
        as a flat array (float[]) but require primvar elementSize > 1 so that viewers
        interpret them as tuples-per-point rather than extra points. This can prevent
        Omniverse/Hydra crashes during animation evaluation.

        Heuristic:
        - For vertex primvars: infer elementSize if raw_len % n_points == 0
        - For uniform primvars: infer elementSize if raw_len % n_faces == 0
        - Only updates when inferred elementSize > 1

        Returns:
            dict with keys: updated (list), skipped (list)
        """
        if isinstance(stage_or_path, str):
            stage = Usd.Stage.Open(stage_or_path)
            stage_path = stage_or_path
        else:
            stage = stage_or_path
            stage_path = None

        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Invalid mesh prim at path: {mesh_path}")

        mesh = UsdGeom.Mesh(mesh_prim)
        tc = (
            Usd.TimeCode(time_code) if time_code is not None else Usd.TimeCode.Default()
        )

        pts = mesh.GetPointsAttr().Get(tc)
        if pts is None:
            samples = mesh.GetPointsAttr().GetTimeSamples()
            if samples:
                pts = mesh.GetPointsAttr().Get(Usd.TimeCode(samples[0]))
        n_points = len(pts) if pts is not None else 0

        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        n_faces = len(face_counts) if face_counts is not None else 0

        updated: list[dict] = []
        skipped: list[dict] = []

        api = UsdGeom.PrimvarsAPI(mesh)
        for pv in api.GetPrimvars():
            interp = pv.GetInterpolation()
            if interp not in ("vertex", "uniform"):
                skipped.append({"name": pv.GetName(), "reason": f"interp={interp}"})
                continue

            exp = n_points if interp == "vertex" else n_faces
            if exp <= 0:
                skipped.append({"name": pv.GetName(), "reason": "no topology"})
                continue

            ts = pv.GetAttr().GetTimeSamples()
            t0 = Usd.TimeCode(ts[0]) if ts else tc
            v = pv.Get(t0)
            if v is None:
                skipped.append({"name": pv.GetName(), "reason": "no value"})
                continue

            raw_len = len(v)
            current_elem = int(pv.GetElementSize() or 1)
            eff_len = raw_len // current_elem if current_elem else raw_len

            if eff_len == exp:
                skipped.append({"name": pv.GetName(), "reason": "already consistent"})
                continue

            if raw_len % exp != 0:
                skipped.append(
                    {
                        "name": pv.GetName(),
                        "reason": f"not divisible (raw={raw_len}, exp={exp})",
                    }
                )
                continue

            inferred = raw_len // exp
            if inferred <= 1:
                skipped.append({"name": pv.GetName(), "reason": "inferred<=1"})
                continue

            try:
                pv.SetElementSize(int(inferred))
                updated.append(
                    {
                        "name": pv.GetName(),
                        "interp": interp,
                        "raw_len": raw_len,
                        "exp": exp,
                        "old_elementSize": current_elem,
                        "new_elementSize": int(inferred),
                    }
                )
            except Exception as e:
                skipped.append(
                    {"name": pv.GetName(), "reason": f"SetElementSize failed: {e}"}
                )

        if stage_path and save:
            stage.Save()
            self.log_info(f"Saved USD file: {stage_path}")

        return {"updated": updated, "skipped": skipped}

    def _uniform_to_vertex_scalar(
        self,
        face_scalar: np.ndarray,
        face_vertex_counts: Sequence[int] | np.ndarray,
        face_vertex_indices: Sequence[int] | np.ndarray,
        n_points: int,
    ) -> np.ndarray:
        """
        Convert per-face scalar data to per-vertex by averaging incident faces.

        Args:
            face_scalar: Scalar value per face
            face_vertex_counts: Number of vertices per face
            face_vertex_indices: Flattened vertex indices for all faces
            n_points: Total number of vertices in mesh

        Returns:
            np.ndarray: Scalar value per vertex
        """
        counts_arr = np.asarray(face_vertex_counts, dtype=np.int32)
        indices_arr = np.asarray(face_vertex_indices, dtype=np.int32)

        # Create face ID for each vertex reference
        face_ids = np.repeat(np.arange(len(counts_arr)), counts_arr)

        # Accumulate values at each vertex
        acc = np.zeros(n_points, dtype=np.float64)
        cnt = np.zeros(n_points, dtype=np.int32)

        np.add.at(acc, indices_arr, face_scalar[face_ids])
        np.add.at(cnt, indices_arr, 1)

        # Average
        vertex_scalar = acc / np.maximum(cnt, 1)
        result: np.ndarray = vertex_scalar.astype(np.float32)
        return result

    def _ensure_vertex_color_material(
        self, stage: Usd.Stage, mesh_prim: Usd.Prim
    ) -> None:
        """
        Create or reuse a vertex color material and bind it to the mesh.

        Creates a UsdPreviewSurface material that reads displayColor via
        UsdPrimvarReader_float3, following Omniverse best practices.

        Args:
            stage: USD Stage
            mesh_prim: Mesh prim to bind material to
        """
        from pxr import Sdf

        material_name = "VertexColorMaterial"
        material_path = f"/World/Looks/{material_name}"

        # Check if material already exists
        material_prim = stage.GetPrimAtPath(material_path)
        if material_prim.IsValid() and material_prim.IsA(UsdShade.Material):
            material = UsdShade.Material(material_prim)
            self.log_debug(f"Reusing existing material: {material_path}")
        else:
            # Create material scope if needed
            looks_prim = stage.GetPrimAtPath("/World/Looks")
            if not looks_prim.IsValid():
                stage.DefinePrim("/World/Looks", "Scope")

            # Create material
            material = UsdShade.Material.Define(stage, material_path)

            # Create PreviewSurface shader
            shader_path = f"{material_path}/PreviewSurface"
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")

            # Create PrimvarReader for displayColor
            reader_path = f"{material_path}/PrimvarReader_displayColor"
            reader = UsdShade.Shader.Define(stage, reader_path)
            reader.CreateIdAttr("UsdPrimvarReader_float3")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")

            # Connect reader output to shader diffuseColor input
            reader_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Color3f)
            diffuse_input = shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            )
            diffuse_input.ConnectToSource(reader_output)

            # Set other shader properties
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

            # Connect shader to material surface
            surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(surface_output)

            self.log_info(f"Created vertex color material: {material_path}")

        # Bind material to mesh
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        binding_api.Bind(material)
        self.log_debug(f"Bound material to mesh: {mesh_prim.GetPath()}")
