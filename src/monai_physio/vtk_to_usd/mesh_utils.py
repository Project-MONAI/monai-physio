"""Mesh utilities for VTK to USD conversion.

Includes splitting meshes by cell type (face vertex count) or by connectivity
for separate USD prims.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from .data_structures import GenericArray, MeshData

# Map face vertex count to cell type name (matches VTK semantics: triangle=3, quad=4, tetra=4, hex=8)
# For 4 we use "Quad" (surface); volume tet would also be 4 - we don't distinguish here.
CELL_TYPE_NAME_BY_VERTEX_COUNT: dict[int, str] = {
    3: "Triangle",
    4: "Quad",
    5: "Pentagon",
    6: "Wedge",
    8: "Hexahedron",
}


def cell_type_name_for_vertex_count(count: int) -> str:
    """Return a readable name for a cell type given its vertex count."""
    return CELL_TYPE_NAME_BY_VERTEX_COUNT.get(count, f"Cell_{count}")


def split_mesh_data_by_cell_type(
    mesh_data: MeshData, mesh_name: str
) -> list[tuple[MeshData, str]]:
    """Split MeshData into one mesh per distinct face vertex count (cell type).

    Each part is named as mesh_name plus the cell type (e.g. MeshName_Triangle,
    MeshName_Quad).

    Args:
        mesh_data: Single mesh that may contain mixed cell types.
        mesh_name: Name of the source mesh; used as prefix in returned base_name.

    Returns:
        List of (MeshData, base_name) for each cell type present. base_name is
        mesh_name + "_" + cell type name (e.g. "MeshName_Triangle", "MeshName_Quad").
    """
    counts = np.asarray(mesh_data.face_vertex_counts, dtype=np.int32)
    indices = np.asarray(mesh_data.face_vertex_indices, dtype=np.int32)
    points = np.asarray(mesh_data.points)
    n_points = len(points)
    n_faces = len(counts)

    if n_faces == 0:
        return [(mesh_data, f"{mesh_name}_Empty")]

    cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    unique_counts = np.unique(counts)
    if len(unique_counts) <= 1:
        # Single cell type: return one mesh with that type name
        name = cell_type_name_for_vertex_count(int(counts[0])) if n_faces else "Mesh"
        return [(mesh_data, f"{mesh_name}_{name}")]

    result: list[tuple[MeshData, str]] = []

    for count in unique_counts:
        count = int(count)
        face_mask = counts == count
        face_idxs = np.where(face_mask)[0]
        num_faces = len(face_idxs)

        # Gather vertex indices used by these faces
        seg_starts = cum[face_idxs]
        seg_ends = cum[face_idxs + 1]
        used = np.concatenate(
            [indices[seg_starts[i] : seg_ends[i]] for i in range(num_faces)]
        )
        unique_pts = np.unique(used)
        old_to_new = np.full(n_points, -1, dtype=np.int32)
        old_to_new[unique_pts] = np.arange(len(unique_pts), dtype=np.int32)

        new_points = points[unique_pts]
        new_counts = np.full(num_faces, count, dtype=np.int32)
        new_indices_list: list[int] = []
        for i in range(num_faces):
            seg = indices[seg_starts[i] : seg_ends[i]]
            new_indices_list.extend(old_to_new[seg].tolist())
        new_indices = np.array(new_indices_list, dtype=np.int32)

        # Subset normals (per-vertex)
        new_normals = None
        if mesh_data.normals is not None:
            arr = np.asarray(mesh_data.normals)
            if arr.shape[0] == n_points and arr.ndim == 2:
                new_normals = arr[unique_pts]
            elif arr.shape[0] == cum[-1] and arr.ndim == 2:
                flat = np.concatenate(
                    [arr[seg_starts[j] : seg_ends[j]] for j in range(num_faces)]
                )
                new_normals = flat

        # Subset colors (per-vertex)
        new_colors = None
        if mesh_data.colors is not None:
            arr = np.asarray(mesh_data.colors)
            if arr.shape[0] == n_points:
                new_colors = arr[unique_pts]

        # Subset generic arrays: vertex by point index, uniform by face index
        new_arrays: list[GenericArray] = []
        for generic_array in mesh_data.generic_arrays:
            data = np.asarray(generic_array.data)
            if generic_array.interpolation == "vertex":
                if data.shape[0] == n_points:
                    new_data = data[unique_pts]
                else:
                    continue
            else:
                if data.shape[0] == n_faces:
                    new_data = data[face_idxs]
                else:
                    continue
            new_arrays.append(
                GenericArray(
                    name=generic_array.name,
                    data=new_data,
                    num_components=generic_array.num_components,
                    data_type=generic_array.data_type,
                    interpolation=generic_array.interpolation,
                )
            )

        part = MeshData(
            points=new_points,
            face_vertex_counts=new_counts,
            face_vertex_indices=new_indices,
            normals=new_normals,
            uvs=None,
            colors=new_colors,
            generic_arrays=new_arrays,
            material_id=mesh_data.material_id,
        )
        name = cell_type_name_for_vertex_count(count)
        result.append((part, f"{mesh_name}_{name}"))

    return result


def _connected_components_face_indices(
    n_faces: int,
    indices: NDArray,
    cum: NDArray,
) -> list[list[int]]:
    """Return list of face-index lists, one per connected component.

    Two faces are in the same component if they share at least one vertex.
    Uses union-find on face indices.
    """
    # vertex -> list of face indices that use that vertex
    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    for i in range(n_faces):
        start, end = int(cum[i]), int(cum[i + 1])
        for k in range(start, end):
            v = int(indices[k])
            vertex_to_faces[v].append(i)

    # Union-find for faces
    parent = list(range(n_faces))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for face_list in vertex_to_faces.values():
        if len(face_list) < 2:
            continue
        r = find(face_list[0])
        for f in face_list[1:]:
            union(r, find(f))

    # Group face indices by component root
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n_faces):
        components[find(i)].append(i)

    # Return as list of lists, sorted by min face index for stable order
    return sorted(components.values(), key=lambda x: min(x))


def _extract_mesh_part_by_face_indices(
    mesh_data: MeshData,
    face_idxs: list[int],
    n_points: int,
    n_faces: int,
    counts: NDArray,
    indices: NDArray,
    cum: NDArray,
    points: NDArray,
) -> MeshData:
    """Build a new MeshData containing only the given faces (and their points)."""
    face_idxs_arr = np.asarray(face_idxs, dtype=np.int32)
    num_faces = len(face_idxs)

    seg_starts = cum[face_idxs_arr]
    seg_ends = cum[face_idxs_arr + 1]
    used = np.concatenate(
        [indices[seg_starts[i] : seg_ends[i]] for i in range(num_faces)]
    )
    unique_pts = np.unique(used)
    old_to_new = np.full(n_points, -1, dtype=np.int32)
    old_to_new[unique_pts] = np.arange(len(unique_pts), dtype=np.int32)

    new_points = points[unique_pts]
    new_counts = counts[face_idxs_arr]
    new_indices_list: list[int] = []
    for i in range(num_faces):
        seg = indices[seg_starts[i] : seg_ends[i]]
        new_indices_list.extend(old_to_new[seg].tolist())
    new_indices = np.array(new_indices_list, dtype=np.int32)

    new_normals = None
    if mesh_data.normals is not None:
        arr = np.asarray(mesh_data.normals)
        if arr.shape[0] == n_points and arr.ndim == 2:
            new_normals = arr[unique_pts]
        elif arr.shape[0] == cum[-1] and arr.ndim == 2:
            flat = np.concatenate(
                [arr[seg_starts[j] : seg_ends[j]] for j in range(num_faces)]
            )
            new_normals = flat

    new_colors = None
    if mesh_data.colors is not None:
        arr = np.asarray(mesh_data.colors)
        if arr.shape[0] == n_points:
            new_colors = arr[unique_pts]

    new_arrays: list[GenericArray] = []
    for generic_array in mesh_data.generic_arrays:
        data = np.asarray(generic_array.data)
        if generic_array.interpolation == "vertex":
            if data.shape[0] == n_points:
                new_data = data[unique_pts]
            else:
                continue
        else:
            if data.shape[0] == n_faces:
                new_data = data[face_idxs_arr]
            else:
                continue
        new_arrays.append(
            GenericArray(
                name=generic_array.name,
                data=new_data,
                num_components=generic_array.num_components,
                data_type=generic_array.data_type,
                interpolation=generic_array.interpolation,
            )
        )

    return MeshData(
        points=new_points,
        face_vertex_counts=new_counts,
        face_vertex_indices=new_indices,
        normals=new_normals,
        uvs=None,
        colors=new_colors,
        generic_arrays=new_arrays,
        material_id=mesh_data.material_id,
    )


def split_mesh_data_by_connectivity(
    mesh_data: MeshData, mesh_name: str
) -> list[tuple[MeshData, str]]:
    """Split MeshData into one mesh per connected component.

    A connected component is a maximal set of cells that share vertices (directly
    or transitively). Components are named mesh_name_object1, mesh_name_object2, etc.

    Args:
        mesh_data: Single mesh that may contain multiple disconnected parts.
        mesh_name: Name of the source mesh; used as prefix in returned base_name.

    Returns:
        List of (MeshData, base_name) for each component. base_name is
        mesh_name + "_objectN" (e.g. "MeshName_object1", "MeshName_object2", ...).
    """
    counts = np.asarray(mesh_data.face_vertex_counts, dtype=np.int32)
    indices = np.asarray(mesh_data.face_vertex_indices, dtype=np.int32)
    points = np.asarray(mesh_data.points)
    n_points = len(points)
    n_faces = len(counts)

    if n_faces == 0:
        return [(mesh_data, f"{mesh_name}_object1")]

    cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    component_face_lists = _connected_components_face_indices(n_faces, indices, cum)
    if len(component_face_lists) <= 1:
        return [(mesh_data, f"{mesh_name}_object1")]

    result: list[tuple[MeshData, str]] = []
    for k, face_idxs in enumerate(component_face_lists, start=1):
        part = _extract_mesh_part_by_face_indices(
            mesh_data, face_idxs, n_points, n_faces, counts, indices, cum, points
        )
        result.append((part, f"{mesh_name}_object{k}"))

    return result
