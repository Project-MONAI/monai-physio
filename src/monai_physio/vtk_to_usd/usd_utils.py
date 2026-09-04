"""USD utility functions for VTK to USD conversion.

Provides helper functions for coordinate conversion, primvar creation, and USD type mapping.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, Vt

from .data_structures import DataType, GenericArray

logger = logging.getLogger(__name__)


def lps_to_usd(point: NDArray | tuple | list) -> Gf.Vec3f:
    """Convert LPS (Left-Posterior-Superior) coordinates to USD's right-handed Y-up frame.

    MONAI Physio keeps images and surfaces in ITK's native LPS world space.
    ``itk.imread`` normalizes every supported input (DICOM, NIfTI, MHA, NRRD)
    to LPS, and ``itk.vtk_image_from_image`` preserves that frame when
    handing data to PyVista/VTK, so meshes extracted via ``contour_labels``
    or ``threshold`` arrive here in LPS.

    The USD frame produced by this conversion is right-handed Y-up with:

    - USD +X = patient Left      (LPS +x)
    - USD +Y = patient Superior  (LPS +z)  — "up" in Omniverse
    - USD +Z = patient Anterior  (−LPS +y) — toward the viewer when the
      camera looks at the patient from the front

    Conversion: ``USD(x, y, z) = LPS(x, z, −y) * 0.001``  (mm → m)

    Args:
        point: Point in LPS coordinates [x, y, z] in millimeters

    Returns:
        Gf.Vec3f: Point in USD coordinates in meters
    """
    if isinstance(point, (tuple, list)):
        return Gf.Vec3f(
            float(point[0]) * 0.001,
            float(point[2]) * 0.001,
            float(-point[1]) * 0.001,
        )
    else:
        return Gf.Vec3f(
            float(point[0]) * 0.001,
            float(point[2]) * 0.001,
            float(-point[1]) * 0.001,
        )


def lps_points_to_usd(points: NDArray) -> Vt.Vec3fArray:
    """Convert array of LPS points (mm) to USD Y-up coordinates (m).

    Applies the LPS → Y-up axis swap defined in :func:`lps_to_usd` and scales
    millimeters to meters (* 0.001).

    Args:
        points: Array of points with shape (N, 3) in millimeters

    Returns:
        Vt.Vec3fArray: Points in USD Y-up coordinates in meters
    """
    if points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")

    # Vectorized: USD(x, y, z) = LPS(x, z, -y) * 0.001  (mm → m)
    usd_points = np.empty(points.shape, dtype=np.float32)
    usd_points[:, 0] = points[:, 0] * 0.001
    usd_points[:, 1] = points[:, 2] * 0.001
    usd_points[:, 2] = -points[:, 1] * 0.001

    return Vt.Vec3fArray.FromNumpy(usd_points)


def lps_normals_to_usd(normals: NDArray) -> Vt.Vec3fArray:
    """Convert array of LPS normals to USD Y-up coordinates.

    Applies only the axis swap from :func:`lps_to_usd` — normals are unit
    direction vectors and must not be scaled by the mm→m factor.

    Args:
        normals: Array of normals with shape (N, 3)

    Returns:
        Vt.Vec3fArray: Normals in USD Y-up coordinates (unit length preserved)
    """
    if normals.shape[1] != 3:
        raise ValueError(f"Normals must have shape (N, 3), got {normals.shape}")

    usd_normals = np.empty(normals.shape, dtype=np.float32)
    usd_normals[:, 0] = normals[:, 0]
    usd_normals[:, 1] = normals[:, 2]
    usd_normals[:, 2] = -normals[:, 1]

    return Vt.Vec3fArray.FromNumpy(usd_normals)


def numpy_to_vt_array(array: NDArray, data_type: DataType) -> Any:
    """Convert numpy array to appropriate VtArray type.

    Args:
        array: Numpy array to convert
        data_type: Target data type

    Returns:
        Appropriate VtArray based on data_type and array shape
    """
    # Ensure contiguous array for efficient conversion
    array = np.ascontiguousarray(array)

    # Determine number of components
    if array.ndim == 1:
        num_components = 1
    elif array.ndim == 2:
        num_components = array.shape[1]
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

    # Convert based on type and components
    if data_type in [DataType.FLOAT, DataType.DOUBLE]:
        array_f = array.astype(np.float32)
        if num_components == 1:
            return Vt.FloatArray.FromNumpy(array_f)
        elif num_components == 2:
            return Vt.Vec2fArray.FromNumpy(array_f)
        elif num_components == 3:
            return Vt.Vec3fArray.FromNumpy(array_f)
        elif num_components == 4:
            return Vt.Vec4fArray.FromNumpy(array_f)
        else:
            # Fallback: flatten to float array
            return Vt.FloatArray.FromNumpy(array_f.flatten())

    elif data_type in [DataType.INT, DataType.LONG]:
        array_i = array.astype(np.int32)
        if num_components == 1:
            return Vt.IntArray.FromNumpy(array_i)
        elif num_components == 2:
            return Vt.Vec2iArray.FromNumpy(array_i)
        elif num_components == 3:
            return Vt.Vec3iArray.FromNumpy(array_i)
        elif num_components == 4:
            return Vt.Vec4iArray.FromNumpy(array_i)
        else:
            return Vt.IntArray.FromNumpy(array_i.flatten())

    elif data_type in [DataType.UINT, DataType.ULONG]:
        array_ui = array.astype(np.uint32)
        if num_components == 1:
            return Vt.UIntArray.FromNumpy(array_ui)
        else:
            # No Vec types for uint, flatten
            return Vt.UIntArray.FromNumpy(array_ui.flatten())

    elif data_type in [DataType.UCHAR, DataType.CHAR]:
        array_uc = array.astype(np.uint8)
        return Vt.UCharArray.FromNumpy(array_uc.flatten())

    else:  # DataType.SHORT, DataType.USHORT
        # Convert to int
        array_i = array.astype(np.int32)
        return Vt.IntArray.FromNumpy(array_i.flatten())


def get_sdf_value_type(data_type: DataType, num_components: int) -> Sdf.ValueTypeName:
    """Get appropriate SDF value type for primvar creation.

    Args:
        data_type: Data type
        num_components: Number of components (1, 2, 3, or 4)

    Returns:
        Sdf.ValueTypeName: Appropriate USD type
    """
    if data_type in [DataType.FLOAT, DataType.DOUBLE]:
        if num_components == 1:
            return Sdf.ValueTypeNames.FloatArray
        elif num_components == 2:
            return Sdf.ValueTypeNames.Float2Array
        elif num_components == 3:
            return Sdf.ValueTypeNames.Float3Array
        elif num_components == 4:
            return Sdf.ValueTypeNames.Float4Array
        else:
            return Sdf.ValueTypeNames.FloatArray

    elif data_type in [DataType.INT, DataType.LONG]:
        if num_components == 1:
            return Sdf.ValueTypeNames.IntArray
        elif num_components == 2:
            return Sdf.ValueTypeNames.Int2Array
        elif num_components == 3:
            return Sdf.ValueTypeNames.Int3Array
        elif num_components == 4:
            return Sdf.ValueTypeNames.Int4Array
        else:
            return Sdf.ValueTypeNames.IntArray

    elif data_type in [DataType.UINT, DataType.ULONG]:
        return Sdf.ValueTypeNames.UIntArray

    elif data_type in [DataType.UCHAR, DataType.CHAR]:
        return Sdf.ValueTypeNames.UCharArray

    else:  # DataType.SHORT, DataType.USHORT
        return Sdf.ValueTypeNames.IntArray


def sanitize_primvar_name(name: str) -> str:
    """Sanitize a name to be USD-compliant.

    USD attribute names must:
    - Start with a letter or underscore
    - Contain only letters, numbers, and underscores
    - Not contain dots, spaces, or special characters

    Args:
        name: Original name

    Returns:
        str: Sanitized name safe for USD
    """
    import re

    # Replace dots with underscores
    name = name.replace(".", "_")

    # Replace spaces and other special characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Ensure it starts with a letter or underscore
    if name and name[0].isdigit():
        name = "_" + name

    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)

    # Remove trailing underscores
    name = name.rstrip("_")

    return name


def create_primvar(
    geom: UsdGeom.Gprim,
    array: GenericArray,
    array_name_prefix: str = "",
    time_code: float | None = None,
) -> UsdGeom.Primvar | None:
    """Create a USD primvar from a GenericArray.

    Args:
        geom: USD geometry prim (Mesh, Points, etc.)
        array: GenericArray containing data
        array_name_prefix: Prefix for primvar name (e.g., ``"vtk_point_"``)
        time_code: Optional time code for time-varying data

    Returns:
        UsdGeom.Primvar: Created primvar, or None if validation failed
    """
    # Sanitize the array name to be USD-compliant
    sanitized_name = sanitize_primvar_name(array.name)
    primvar_name = f"{array_name_prefix}{sanitized_name}"

    # Log if name was changed
    if sanitized_name != array.name:
        logger.debug(f"Sanitized primvar name: '{array.name}' -> '{sanitized_name}'")

    # Validate array size for meshes
    if isinstance(geom, UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(geom)

        # Check size matches expected count based on interpolation
        if array.interpolation == "vertex":
            # Get number of points
            points_attr = mesh.GetPointsAttr()
            if points_attr:
                points = points_attr.Get(
                    time_code if time_code is not None else Usd.TimeCode.Default()
                )
                if points and len(array.data) != len(points):
                    logger.warning(
                        f"Skipping primvar '{primvar_name}': size mismatch "
                        f"(got {len(array.data)}, expected {len(points)} vertices)"
                    )
                    return None

        elif array.interpolation == "uniform":
            # Get number of faces
            face_counts_attr = mesh.GetFaceVertexCountsAttr()
            if face_counts_attr:
                face_counts = face_counts_attr.Get(
                    time_code if time_code is not None else Usd.TimeCode.Default()
                )
                if face_counts and len(array.data) != len(face_counts):
                    logger.warning(
                        f"Skipping primvar '{primvar_name}': size mismatch "
                        f"(got {len(array.data)}, expected {len(face_counts)} faces)"
                    )
                    return None

    # Skip if array has no data
    if len(array.data) == 0:
        logger.debug(f"Skipping empty primvar '{primvar_name}'")
        return None

    # Get primvars API
    primvars_api = UsdGeom.PrimvarsAPI(geom)

    # Get appropriate USD type
    sdf_type = get_sdf_value_type(array.data_type, array.num_components)

    # Create primvar
    primvar = primvars_api.CreatePrimvar(primvar_name, sdf_type)

    # Set interpolation
    if array.interpolation == "vertex":
        primvar.SetInterpolation(UsdGeom.Tokens.vertex)
    elif array.interpolation == "uniform":
        primvar.SetInterpolation(UsdGeom.Tokens.uniform)
    elif array.interpolation == "constant":
        primvar.SetInterpolation(UsdGeom.Tokens.constant)
    else:
        primvar.SetInterpolation(UsdGeom.Tokens.vertex)

    # If this is a multi-component array that we're storing in a scalar array type
    # (e.g. FloatArray for >4 components), preserve the component grouping via elementSize.
    # This makes downstream tools (and USDTools.apply_colormap_from_primvar) able to reshape.
    if array.num_components > 1 and sdf_type in (
        Sdf.ValueTypeNames.FloatArray,
        Sdf.ValueTypeNames.IntArray,
        Sdf.ValueTypeNames.UIntArray,
        Sdf.ValueTypeNames.UCharArray,
    ):
        try:
            primvar.SetElementSize(int(array.num_components))
        except Exception:
            # Not fatal; continue without elementSize.
            pass

    # Convert data to VtArray
    vt_array = numpy_to_vt_array(array.data, array.data_type)

    # Set value (with or without time code)
    if time_code is not None:
        primvar.Set(vt_array, time_code)
    else:
        primvar.Set(vt_array)

    logger.debug(
        f"Created primvar '{primvar_name}' with {len(array.data)} elements, "
        f"{array.num_components} components, type {array.data_type.value}"
    )

    return primvar


def triangulate_face(
    face_counts: NDArray, face_indices: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Triangulate polygonal faces using fan triangulation.

    Args:
        face_counts: Array of vertex counts per source face (length F).
        face_indices: Flat array of vertex indices.

    Returns:
        ``(tri_counts, tri_indices, source_face_index_per_triangle)``:
        - ``tri_counts``: int32 array, all entries equal to 3.
        - ``tri_indices``: int32 flat array of triangle vertex indices.
        - ``source_face_index_per_triangle``: int32 array mapping each output
          triangle back to its source face in ``face_counts``. Length matches
          ``tri_counts``. Use this to expand uniform (per-face) primvar data
          to match the triangulated face count: ``new_data = old_data[mapping]``.
    """
    tri_counts: list[int] = []
    tri_indices: list[int] = []
    source_face_index: list[int] = []

    idx = 0
    for face_idx, count in enumerate(face_counts):
        if count == 3:
            tri_counts.append(3)
            tri_indices.extend(face_indices[idx : idx + 3])
            source_face_index.append(face_idx)
        elif count == 4:
            v0, v1, v2, v3 = face_indices[idx : idx + 4]
            tri_counts.extend([3, 3])
            tri_indices.extend([v0, v1, v2, v0, v2, v3])
            source_face_index.extend([face_idx, face_idx])
        else:
            v0 = face_indices[idx]
            for i in range(1, count - 1):
                tri_counts.append(3)
                tri_indices.extend(
                    [v0, face_indices[idx + i], face_indices[idx + i + 1]]
                )
                source_face_index.append(face_idx)

        idx += count

    return (
        np.array(tri_counts, dtype=np.int32),
        np.array(tri_indices, dtype=np.int32),
        np.array(source_face_index, dtype=np.int32),
    )


def compute_mesh_extent(points: Vt.Vec3fArray) -> Vt.Vec3fArray:
    """Compute bounding box extent for a mesh.

    Args:
        points: Array of points

    Returns:
        Vt.Vec3fArray: Extent as [min_point, max_point]
    """
    return UsdGeom.Mesh.ComputeExtent(points)


def add_framing_camera(
    stage: Usd.Stage,
    *,
    parent_path: str = "/World",
    name: str = "Camera",
    light_name: str = "DistantLight",
    light_intensity: float = 3000.0,
    bounds_min: tuple[float, float, float] | None = None,
    bounds_max: tuple[float, float, float] | None = None,
    focal_length_mm: float = 50.0,
    horizontal_aperture_mm: float = 36.0,
    distance_factor: float = 3.0,
) -> UsdGeom.Camera | None:
    """Define a USD camera that frames stage geometry with tight clipping planes.

    Adds a ``UsdGeom.Camera`` prim at ``{parent_path}/{name}`` positioned along
    +Z to view the supplied (or stage-computed) bounding box. Sets a tight
    ``clippingRange`` so users can zoom close in Omniverse Kit and other USD
    viewers without geometry vanishing at the near plane.

    A ``UsdLux.DistantLight`` is added at ``{parent_path}/{light_name}`` sharing
    the camera's orientation, so the anatomy is lit from the viewing direction
    in stages that carry no other light. The light is created only when valid,
    non-degenerate bounds exist; otherwise no light is authored.

    Bounds must be expressed in stage coordinates (post axis-swap and unit
    scaling). For time-varying stages, bounds are sampled at the start time
    code.

    Args:
        stage: The USD stage. Must already contain geometry when bounds are not
            supplied; world bounds are then computed from the stage.
        parent_path: Parent prim path. Defaults to ``"/World"``.
        name: Camera prim name. Defaults to ``"Camera"``.
        light_name: Distant light prim name. Defaults to ``"DistantLight"``.
        light_intensity: Distant light intensity. Defaults to ``3000.0``, which
            reads as a neutral key light under the RTX renderers.
        bounds_min: Optional min corner ``(x, y, z)`` in stage coordinates.
        bounds_max: Optional max corner ``(x, y, z)`` in stage coordinates.
        focal_length_mm: Camera focal length. USD camera lens parameters are
            always in millimeters regardless of ``metersPerUnit``.
        horizontal_aperture_mm: Camera horizontal aperture in millimeters.
        distance_factor: Camera distance from bbox center as a multiple of the
            bounding-box diagonal. ``3.0`` gives generous framing.

    Returns:
        The created Camera prim, or ``None`` if no valid bounds could be found.
    """
    if bounds_min is None or bounds_max is None:
        if stage.HasAuthoredTimeCodeRange():
            time_code = Usd.TimeCode(stage.GetStartTimeCode())
        else:
            time_code = Usd.TimeCode.Default()
        bbox_cache = UsdGeom.BBoxCache(
            time_code, includedPurposes=[UsdGeom.Tokens.default_]
        )
        world_bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
        bbox_range = world_bbox.ComputeAlignedRange()
        if bbox_range.IsEmpty():
            return None
        bounds_min = tuple(bbox_range.GetMin())
        bounds_max = tuple(bbox_range.GetMax())

    bmin = np.asarray(bounds_min, dtype=np.float64)
    bmax = np.asarray(bounds_max, dtype=np.float64)
    center = 0.5 * (bmin + bmax)
    size = bmax - bmin
    diagonal = float(np.linalg.norm(size))
    if diagonal <= 0.0:
        return None

    distance = diagonal * distance_factor

    camera_path = f"{parent_path.rstrip('/')}/{name}"
    camera = UsdGeom.Camera.Define(stage, camera_path)

    # Compute an eye point offset along an axis perpendicular to the stage up
    # axis, then build a full look-at transform so the camera both moves and
    # rotates to point at the bbox center. A pure translation only frames the
    # geometry on a Y-up stage (where the default -Z look direction lines up);
    # on a Z-up stage it would leave the camera staring horizontally past the
    # geometry, so we author orientation as well via a single TransformOp.
    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis == UsdGeom.Tokens.z:
        eye = center + np.array([0.0, -distance, 0.0])
        up_world = Gf.Vec3d(0.0, 0.0, 1.0)
    else:
        eye = center + np.array([0.0, 0.0, distance])
        up_world = Gf.Vec3d(0.0, 1.0, 0.0)

    view = Gf.Matrix4d()
    view.SetLookAt(
        Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])),
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])),
        up_world,
    )
    camera_to_world = view.GetInverse()

    # Idempotent: clear any prior xformOpOrder (e.g. a translate op carried in
    # from a merged source USD that already had a /World/Camera) and author a
    # single transform op describing the look-at placement.
    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    camera.AddTransformOp().Set(camera_to_world)

    near = max(diagonal * 0.001, 1e-6)
    far = max(diagonal * 1000.0, distance * 10.0)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(float(near), float(far)))

    camera.CreateFocalLengthAttr().Set(float(focal_length_mm))
    camera.CreateHorizontalApertureAttr().Set(float(horizontal_aperture_mm))
    camera.CreateFocusDistanceAttr().Set(float(distance))

    # A DistantLight emits along its local -Z, the same direction the camera
    # looks, so reusing the camera transform lights whatever the camera frames.
    light_path = f"{parent_path.rstrip('/')}/{light_name}"
    light = UsdLux.DistantLight.Define(stage, light_path)
    light_xformable = UsdGeom.Xformable(light.GetPrim())
    light_xformable.ClearXformOpOrder()
    light_xformable.AddTransformOp().Set(camera_to_world)
    light.CreateIntensityAttr().Set(float(light_intensity))
    # Sun-like angular diameter: soft enough to avoid hard-edged shadows.
    light.CreateAngleAttr().Set(0.53)

    return camera
