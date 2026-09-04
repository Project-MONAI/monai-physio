"""USD Mesh converter for creating UsdGeomMesh from MeshData.

Handles geometry, normals, colors, primvars, and time-varying attributes.
"""

import logging
from typing import Optional

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from .data_structures import ConversionSettings, GenericArray, MeshData
from .material_manager import MaterialManager
from .usd_utils import (
    compute_mesh_extent,
    create_primvar,
    lps_normals_to_usd,
    lps_points_to_usd,
    triangulate_face,
)

logger = logging.getLogger(__name__)


class UsdMeshConverter:
    """Converts MeshData to UsdGeomMesh with full feature support.

    Handles:
    - Geometry (points, faces, normals)
    - Vertex colors and display color primvars
    - Generic data arrays as primvars
    - Time-varying attributes
    - Material binding
    """

    def __init__(
        self,
        stage: Usd.Stage,
        settings: ConversionSettings,
        material_mgr: MaterialManager,
    ):
        """Initialize mesh converter.

        Args:
            stage: USD stage
            settings: Conversion settings
            material_mgr: Material manager for material binding
        """
        self.stage = stage
        self.settings = settings
        self.material_mgr = material_mgr

    def _resolve_topology(
        self, mesh_data: MeshData
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return the face counts, indices and triangulation map to author.

        Triangulation only happens when it was asked for and the mesh holds
        faces that are not triangles; the map is ``None`` otherwise.

        Args:
            mesh_data: Mesh whose topology is being written.

        Returns:
            ``(face_vertex_counts, face_vertex_indices,
            triangulation_face_map)``, the last mapping each triangulated face
            back to its source face.
        """
        face_counts = mesh_data.face_vertex_counts
        face_indices = mesh_data.face_vertex_indices
        if self.settings.triangulate_meshes and not all(
            count == 3 for count in face_counts
        ):
            logger.debug("Triangulating mesh faces")
            return triangulate_face(face_counts, face_indices)
        return face_counts, face_indices, None

    def create_mesh(
        self,
        mesh_data: MeshData,
        mesh_path: str,
        time_code: Optional[float] = None,
        bind_material: bool = True,
    ) -> UsdGeom.Mesh:
        """Create a UsdGeomMesh from MeshData.

        Args:
            mesh_data: Mesh data to convert
            mesh_path: USD path for the mesh
            time_code: Optional time code for time-varying data
            bind_material: Whether to create and bind material

        Returns:
            UsdGeom.Mesh: Created USD mesh
        """
        logger.info(f"Creating USD mesh at: {mesh_path}")

        # Create mesh prim
        mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)

        # Convert points to USD coordinates
        usd_points = lps_points_to_usd(mesh_data.points)

        # Handle triangulation if requested
        face_counts, face_indices, triangulation_face_map = self._resolve_topology(
            mesh_data
        )

        # Convert to Vt arrays
        face_counts_vt = Vt.IntArray(face_counts.tolist())
        face_indices_vt = Vt.IntArray(face_indices.tolist())

        # Set topology as the default value. create_time_varying_mesh() adds
        # time samples on top of this when a series changes topology.
        mesh.CreateFaceVertexCountsAttr(face_counts_vt)
        mesh.CreateFaceVertexIndicesAttr(face_indices_vt)

        # Set points (time-varying if time_code provided). Also author a
        # default value for readers that inspect the prim without a time code.
        points_attr = mesh.CreatePointsAttr()
        if time_code is not None:
            if points_attr.Get() is None:
                points_attr.Set(usd_points)
            points_attr.Set(usd_points, time_code)
        else:
            points_attr.Set(usd_points)

        # Set extent (bounding box)
        extent = compute_mesh_extent(usd_points)
        extent_attr = mesh.CreateExtentAttr()
        if time_code is not None:
            if extent_attr.Get() is None:
                extent_attr.Set(extent)
            extent_attr.Set(extent, time_code)
        else:
            extent_attr.Set(extent)

        # Set mesh attributes
        mesh.CreateSubdivisionSchemeAttr("none")  # No subdivision
        mesh.CreateDoubleSidedAttr(True)  # Visible from both sides

        # Handle normals
        if mesh_data.normals is not None:
            logger.debug("Adding normals to mesh")
            usd_normals = lps_normals_to_usd(mesh_data.normals)
            normals_attr = mesh.CreateNormalsAttr()
            normals_attr.SetMetadata("interpolation", UsdGeom.Tokens.vertex)
            if time_code is not None:
                if normals_attr.Get() is None:
                    normals_attr.Set(usd_normals)
                normals_attr.Set(usd_normals, time_code)
            else:
                normals_attr.Set(usd_normals)
        elif self.settings.compute_normals:
            logger.debug("Computing normals for mesh")
            # Normals will be computed by renderer or in post-process
            pass

        # Handle vertex colors
        if mesh_data.colors is not None:
            logger.debug("Adding vertex colors to mesh")
            self._add_vertex_colors(mesh, mesh_data.colors, time_code)

        # Handle generic arrays (primvars). Pass the triangulation face-map so
        # uniform (per-source-face) arrays are expanded to match the
        # post-triangulation face count; otherwise USD would drop them on size
        # mismatch.
        if self.settings.preserve_point_arrays or self.settings.preserve_cell_arrays:
            self._add_generic_arrays(mesh, mesh_data, time_code, triangulation_face_map)

        # Bind material (if material_id is provided and material exists in cache)
        if bind_material and mesh_data.material_id:
            if mesh_data.material_id in self.material_mgr.material_cache:
                material = self.material_mgr.material_cache[mesh_data.material_id]
                self.material_mgr.bind_material(mesh, material)

        logger.info(
            f"Created mesh with {len(mesh_data.points)} points, "
            f"{len(face_counts)} faces"
        )

        return mesh

    def _add_vertex_colors(
        self, mesh: UsdGeom.Mesh, colors: Vt.Vec3fArray, time_code: Optional[float]
    ) -> None:
        """Add vertex colors to mesh as displayColor primvar.

        Args:
            mesh: USD mesh
            colors: Color array (N, 3) or (N, 4)
            time_code: Optional time code
        """
        # Convert to Vec3f if needed
        if colors.shape[1] == 4:
            # RGBA -> RGB
            colors_rgb = colors[:, :3]
        else:
            colors_rgb = colors

        # Create displayColor primvar
        display_color_primvar = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)

        # Convert to Vt.Vec3fArray (convert numpy float32 to Python float)
        color_array = Vt.Vec3fArray(
            [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in colors_rgb]
        )

        if time_code is not None:
            # Author a default value for viewers that don't evaluate time samples unless
            # an explicit time is set (common in some Omniverse/Kit workflows).
            if float(time_code) == 0.0:
                display_color_primvar.Set(color_array)
            display_color_primvar.Set(color_array, time_code)
        else:
            display_color_primvar.Set(color_array)

        # Handle opacity if RGBA
        if colors.shape[1] == 4:
            display_opacity_primvar = mesh.CreateDisplayOpacityPrimvar(
                UsdGeom.Tokens.vertex
            )
            opacity_array = Vt.FloatArray(colors[:, 3].tolist())
            if time_code is not None:
                if float(time_code) == 0.0:
                    display_opacity_primvar.Set(opacity_array)
                display_opacity_primvar.Set(opacity_array, time_code)
            else:
                display_opacity_primvar.Set(opacity_array)

    def _add_generic_arrays(
        self,
        mesh: UsdGeom.Mesh,
        mesh_data: MeshData,
        time_code: Optional[float],
        triangulation_face_map: Optional[np.ndarray] = None,
    ) -> None:
        """Add generic data arrays as primvars.

        Args:
            mesh: USD mesh
            mesh_data: Mesh data containing arrays
            time_code: Optional time code
            triangulation_face_map: Optional int32 array mapping each
                triangulated face back to its source face. When provided,
                uniform-interpolation arrays sized to the source face count
                are expanded so they match the triangulated face count.
        """
        for array in mesh_data.generic_arrays:
            if triangulation_face_map is not None and array.interpolation == "uniform":
                data = np.asarray(array.data)
                if len(data) == triangulation_face_map.shape[0]:
                    # Already triangle-aligned (e.g. derived primvar that was
                    # built post-triangulation). Leave it alone.
                    pass
                elif (
                    len(data) > 0
                    and triangulation_face_map.size > 0
                    and triangulation_face_map.max() < len(data)
                ):
                    expanded_data = data[triangulation_face_map]
                    array = GenericArray(
                        name=array.name,
                        data=expanded_data,
                        num_components=array.num_components,
                        data_type=array.data_type,
                        interpolation=array.interpolation,
                    )
            # Avoid authoring large multi-component tensors as flat float[] vertex primvars.
            # Omniverse/Hydra can be unstable when such primvars have elementSize > 1.
            # Instead, split into multiple primvars with <= 3 components each.
            if array.num_components > 4:
                try:
                    data = np.asarray(array.data)
                    # Data should already be normalized to 2D by GenericArray.__post_init__
                    # (or 1D for scalar arrays with num_components=1, but we're in num_components>4 branch)
                    if data.ndim != 2 or data.shape[1] != array.num_components:
                        logger.warning(
                            "Skipping primvar %s: unexpected shape %s for num_components=%d",
                            array.name,
                            data.shape,
                            array.num_components,
                        )
                        continue

                    # Determine prefix based on interpolation
                    if array.interpolation == "vertex":
                        prefix = self.settings.point_array_prefix
                    elif array.interpolation == "uniform":
                        prefix = self.settings.cell_array_prefix
                    else:
                        prefix = ""

                    # Split into chunks of 3 components (last chunk may be 1 or 2)
                    for chunk_idx, start in enumerate(
                        range(0, array.num_components, 3)
                    ):
                        chunk = data[:, start : start + 3]
                        if chunk.size == 0:
                            continue
                        chunk_name = f"{array.name}_c{chunk_idx}"
                        chunk_arr = GenericArray(
                            name=chunk_name,
                            data=chunk,
                            num_components=int(chunk.shape[1]),
                            data_type=array.data_type,
                            interpolation=array.interpolation,
                        )
                        create_primvar(mesh, chunk_arr, prefix, time_code)

                except Exception as e:
                    logger.warning("Failed to split primvar %s: %s", array.name, e)
                continue

            # Determine prefix based on interpolation
            if array.interpolation == "vertex":
                prefix = self.settings.point_array_prefix
            elif array.interpolation == "uniform":
                prefix = self.settings.cell_array_prefix
            else:
                prefix = ""

            # Skip if not preserving this type of array
            if (
                array.interpolation == "vertex"
                and not self.settings.preserve_point_arrays
            ):
                continue
            if (
                array.interpolation == "uniform"
                and not self.settings.preserve_cell_arrays
            ):
                continue

            try:
                create_primvar(mesh, array, prefix, time_code)
            except Exception as e:
                logger.warning(f"Failed to create primvar for {array.name}: {e}")

    def create_time_varying_mesh(
        self,
        mesh_data_sequence: list[MeshData],
        mesh_path: str,
        time_codes: list[float],
        bind_material: bool = True,
    ) -> UsdGeom.Mesh:
        """Create a mesh with time-varying attributes.

        A series whose frames share one topology, as a surface propagated
        through a deformation does, authors that topology once and time-samples
        only the point positions, so viewers interpolate between samples.  A
        series whose frames were built independently, and so agree on neither
        point count nor triangulation, additionally time-samples
        ``faceVertexCounts`` and ``faceVertexIndices``; USD holds those samples
        rather than interpolating them, so such a mesh snaps from frame to
        frame.

        Args:
            mesh_data_sequence: List of MeshData for each time step
            mesh_path: USD path for the mesh
            time_codes: List of time codes
            bind_material: Whether to create and bind material

        Returns:
            UsdGeom.Mesh: Created USD mesh with time samples
        """
        if len(mesh_data_sequence) != len(time_codes):
            raise ValueError(
                f"Number of mesh data ({len(mesh_data_sequence)}) must match "
                f"number of time codes ({len(time_codes)})"
            )

        if len(mesh_data_sequence) == 0:
            raise ValueError("Empty mesh data sequence")

        logger.info(
            f"Creating time-varying mesh at: {mesh_path} "
            f"with {len(time_codes)} time steps"
        )

        topologies = [self._resolve_topology(md) for md in mesh_data_sequence]
        first_counts, first_indices, _ = topologies[0]
        topology_varies = any(
            not np.array_equal(counts, first_counts)
            or not np.array_equal(indices, first_indices)
            for counts, indices, _ in topologies[1:]
        )

        # Create mesh with first time step
        first_mesh_data = mesh_data_sequence[0]
        mesh = self.create_mesh(
            first_mesh_data, mesh_path, time_codes[0], bind_material=bind_material
        )

        if topology_varies:
            logger.warning(
                "Topology changes across the %d frames of %s; authoring it per "
                "time sample, which viewers hold rather than interpolate",
                len(time_codes),
                mesh_path,
            )
            # A time sample wins over the default at every time, so the first
            # frame has to be sampled too or it would resolve to the last one.
            counts_attr = mesh.GetFaceVertexCountsAttr()
            indices_attr = mesh.GetFaceVertexIndicesAttr()
            for (counts, indices, _), time_code in zip(
                topologies, time_codes, strict=False
            ):
                counts_attr.Set(Vt.IntArray(counts.tolist()), time_code)
                indices_attr.Set(Vt.IntArray(indices.tolist()), time_code)

        # Add time samples for subsequent steps
        for frame_index, (mesh_data, time_code) in enumerate(
            zip(mesh_data_sequence[1:], time_codes[1:], strict=False), start=1
        ):
            # Update points
            usd_points = lps_points_to_usd(mesh_data.points)
            mesh.GetPointsAttr().Set(usd_points, time_code)

            # Update extent
            extent = compute_mesh_extent(usd_points)
            mesh.GetExtentAttr().Set(extent, time_code)

            # Update normals if present
            if mesh_data.normals is not None:
                usd_normals = lps_normals_to_usd(mesh_data.normals)
                mesh.GetNormalsAttr().Set(usd_normals, time_code)

            # Update colors if present
            if mesh_data.colors is not None:
                self._add_vertex_colors(mesh, mesh_data.colors, time_code)

            # Update generic arrays, expanding uniform ones with this frame's
            # own triangulation map.
            if (
                self.settings.preserve_point_arrays
                or self.settings.preserve_cell_arrays
            ):
                self._add_generic_arrays(
                    mesh,
                    mesh_data,
                    time_code,
                    topologies[frame_index][2],
                )

        logger.info(f"Created time-varying mesh with {len(time_codes)} time samples")

        return mesh
