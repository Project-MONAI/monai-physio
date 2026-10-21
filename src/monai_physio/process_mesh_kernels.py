"""
GPU primitives behind the surface and volumetric mesh tools.

This module provides the :class:`ProcessMeshKernels` class: closest-point and
inside/outside queries against a triangle surface, the tetrahedron scaled
Jacobian, exact Euclidean distance maps, a blur matching ITK's recursive
Gaussian, and label-vote resampling.  The warp kernels stay at module scope,
where warp builds them; the class methods prepare their inputs and launch
them.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import itk
import numpy as np
import pyvista as pv
import vtk
import warp as wp

from .monai_physio_base import MONAIPhysioBase

if TYPE_CHECKING:
    import torch


@wp.kernel
def _closest_point_kernel(
    mesh_id: wp.uint64,
    queries: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    closest: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
) -> None:
    index = wp.tid()
    query = queries[index]
    hit = wp.mesh_query_point_no_sign(mesh_id, query, 1.0e6)  # type: ignore[arg-type, unused-ignore]
    if hit.result:  # type: ignore[attr-defined]
        closest[index] = wp.mesh_eval_position(
            mesh_id,
            hit.face,  # type: ignore[attr-defined]
            hit.u,  # type: ignore[attr-defined]
            hit.v,  # type: ignore[attr-defined]
        )
    else:
        closest[index] = query


@wp.kernel
def _inside_surface_kernel(
    mesh_id: wp.uint64,
    edge_band: wp.float32,
    indices: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    face_normals: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    face_neighbors: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    vertex_starts: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    vertex_faces: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    queries: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    inside: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
    undecided: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
) -> None:
    index = wp.tid()
    query = queries[index]
    hit = wp.mesh_query_point_no_sign(mesh_id, query, 1.0e6)  # type: ignore[arg-type, unused-ignore]
    inside[index] = False
    undecided[index] = False
    if hit.result:  # type: ignore[attr-defined]
        face = hit.face  # type: ignore[attr-defined]
        closest = wp.mesh_eval_position(
            mesh_id,
            face,
            hit.u,  # type: ignore[attr-defined]
            hit.v,  # type: ignore[attr-defined]
        )
        offset = query - closest
        is_inside = wp.dot(offset, face_normals[face]) < 0.0
        inside[index] = is_inside
        weights = wp.vec3(
            hit.u,  # type: ignore[attr-defined]
            hit.v,  # type: ignore[attr-defined]
            1.0 - hit.u - hit.v,  # type: ignore[attr-defined]
        )
        zero_count = wp.int32(0)
        zero_corner = wp.int32(0)
        far_corner = wp.int32(0)
        for corner in range(3):
            if weights[corner] < edge_band:
                zero_count += 1
                zero_corner = corner  # type: ignore[assignment]
            else:
                far_corner = corner  # type: ignore[assignment]
        if zero_count == 1:
            neighbor = face_neighbors[face * 3 + zero_corner]
            if neighbor >= 0:  # noqa: SIM102
                if (wp.dot(offset, face_normals[neighbor]) < 0.0) != is_inside:
                    undecided[index] = True
        elif zero_count == 2:
            vertex = indices[face * 3 + far_corner]
            for slot in range(vertex_starts[vertex], vertex_starts[vertex + 1]):
                other = vertex_faces[slot]
                if (wp.dot(offset, face_normals[other]) < 0.0) != is_inside:
                    undecided[index] = True


@wp.kernel
def _squared_distance_line_kernel(
    values: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    no_feature_limit: wp.float32,
    spacing_squared: wp.float32,
    position: wp.array2d(dtype=wp.int32),  # type: ignore[valid-type]
    bound: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    result: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
) -> None:
    # Lower envelope of the parabolas rooted at every finite entry of one line
    # (Felzenszwalb and Huttenlocher), one line per thread; columns of the 2-D
    # arrays are lines, so neighboring threads read neighboring addresses.
    line = wp.tid()
    length = values.shape[0]
    top = wp.int32(-1)
    for q in range(length):
        value = values[q, line]
        if value < no_feature_limit:
            offset = value + spacing_squared * float(q) * float(q)
            crossing = wp.float32(-1.0e30)
            placed = wp.int32(0)
            while placed == 0:
                if top < 0:
                    placed = 1  # type: ignore[assignment]
                else:
                    p = position[top, line]
                    other = values[p, line] + spacing_squared * float(p) * float(p)
                    gap = 2.0 * spacing_squared * float(q - p)
                    crossing = (offset - other) / gap
                    if crossing <= bound[top, line]:
                        top = top - 1
                    else:
                        placed = 1  # type: ignore[assignment]
            top = top + 1
            position[top, line] = q
            bound[top, line] = crossing
    reach = wp.int32(0)
    for q in range(length):
        if top < 0:
            result[q, line] = values[q, line]
        else:
            while reach < top and bound[reach + 1, line] < float(q):
                reach = reach + 1
            p = position[reach, line]
            gap = float(q - p)
            result[q, line] = values[p, line] + spacing_squared * gap * gap


@functools.cache
def _recursive_gaussian_taps(sigma: float) -> np.ndarray:
    """Return the impulse response of ITK's recursive Gaussian, in voxels.

    ITK's recursive filter only approximates a Gaussian (its variance falls
    short of ``sigma ** 2`` by several percent), and a level set read off a
    blurred distance map moves with that difference, so the GPU blur applies
    the response ITK's own filter has rather than a true Gaussian.  The
    response is read off ITK for a unit impulse and cut where it dies away.
    """
    # Half-width, in sigmas, of the window over which the response is read, and
    # the size under which a tap of it is dropped.
    impulse_sigmas = 16.0
    tap_floor = 1.0e-9
    radius = math.ceil(impulse_sigmas * sigma)
    impulse = np.zeros((1, 1, 2 * radius + 1), dtype=np.float64)
    impulse[0, 0, radius] = 1.0
    image = itk.GetImageFromArray(impulse)
    recursive = itk.RecursiveGaussianImageFilter[type(image), type(image)].New()
    recursive.SetInput(image)
    recursive.SetSigma(sigma)
    recursive.SetDirection(0)
    recursive.SetNormalizeAcrossScale(False)
    recursive.Update()
    response = itk.GetArrayFromImage(recursive.GetOutput())[0, 0]
    reach = int(np.flatnonzero(np.abs(response) > tap_floor).max())
    half = max(1, reach - radius)
    return np.asarray(response[radius - half : radius + half + 1])


@wp.kernel
def _banded_sum_kernel(
    source: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
    weights: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
    first: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    last: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    result: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
) -> None:
    row, column = wp.tid()  # type: ignore[misc]
    total = wp.float64(0.0)
    for index in range(first[row], last[row]):
        total += weights[row, index] * source[index, column]
    result[row, column] = total


class ProcessMeshKernels(MONAIPhysioBase):
    """
    GPU primitives shared by :class:`ProcessSurfaceMesh` and
    :class:`ProcessVolumetricMesh`.

    Surface queries take a :class:`warp.Mesh` built once by
    :meth:`build_surface_mesh` and reused across calls.  Distance maps and
    blurs take and return torch tensors on :attr:`device`;
    :meth:`vote_resample_labelmap` takes and returns ITK images.

    Example:
        >>> kernels = ProcessMeshKernels()
        >>> mesh = kernels.build_surface_mesh(surface)
        >>> closest = kernels.closest_points_on_surface(mesh, points)
    """

    #: Squared distance given to a voxel no feature can reach.
    NO_FEATURE = 1.0e30

    #: Squared distance at or above which a voxel had no feature in reach.
    NO_FEATURE_LIMIT = 1.0e20

    #: ITK's label-vote interpolator weighs a voxel by the Gaussian's mass over
    #: it and ignores voxels farther than this many sigmas from the sample.
    VOTE_ALPHA = 3.0

    #: A closest point whose barycentric weight on a triangle corner is below
    #: this counts as lying on the opposite edge (on a vertex, for two such
    #: weights); float32 positions at a patient's coordinates blur the weights
    #: by about 1e-3.
    EDGE_BAND = 2.0e-3

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the ProcessMeshKernels class.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.device = "cuda:0"

    def build_surface_mesh(self, surface: pv.PolyData) -> wp.Mesh:
        """Upload *surface*'s triangles to the GPU for closest-point queries."""
        import torch

        wp.init()
        triangles = surface if surface.is_all_triangles else surface.triangulate()
        points = torch.as_tensor(
            np.asarray(triangles.points, dtype=np.float32), device=self.device
        )
        indices = torch.as_tensor(
            triangles.faces.reshape(-1, 4)[:, 1:].astype(np.int32).ravel(),
            device=self.device,
        )
        return wp.Mesh(
            points=wp.from_torch(points, dtype=wp.vec3),  # type: ignore[arg-type]
            indices=wp.from_torch(indices, dtype=wp.int32),  # type: ignore[arg-type]
        )

    def closest_points_on_surface(
        self, mesh: wp.Mesh, queries: np.ndarray
    ) -> np.ndarray:
        """Return each row of *queries*' closest point on the surface of *mesh*."""
        import torch

        query_tensor = torch.as_tensor(
            np.ascontiguousarray(queries, dtype=np.float32), device=self.device
        )
        closest = torch.empty_like(query_tensor)
        torch.cuda.synchronize()
        wp.launch(
            _closest_point_kernel,
            dim=len(query_tensor),
            inputs=[
                mesh.id,
                wp.from_torch(query_tensor, dtype=wp.vec3),
                wp.from_torch(closest, dtype=wp.vec3),
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return closest.cpu().numpy().astype(np.float64)

    def points_inside_surface(self, mesh: wp.Mesh, points: np.ndarray) -> np.ndarray:
        """Return which rows of *points* lie inside the closed, outward-wound *mesh*.

        The side is the one ``vtkImplicitPolyDataDistance`` reports: that of the
        normal of the closest face.  Where the closest point lies on an edge or a
        vertex, VTK takes whichever of the faces meeting there its search finds
        first.  If they all put the point on one side the GPU has the answer; on a
        thin, tubular surface they sometimes do not, and those few points go to VTK.
        """
        import torch

        vertices = wp.to_torch(mesh.points)
        corners = wp.to_torch(mesh.indices).long().reshape(-1, 3)
        face_normals = torch.nn.functional.normalize(
            torch.cross(
                vertices[corners[:, 1]] - vertices[corners[:, 0]],
                vertices[corners[:, 2]] - vertices[corners[:, 0]],
                dim=1,
            ),
            dim=1,
        )

        # Slot 3 * face + corner holds the face across the edge opposite that corner.
        edge_keys = torch.stack(
            [
                torch.minimum(
                    corners[:, (corner + 1) % 3], corners[:, (corner + 2) % 3]
                )
                * len(vertices)
                + torch.maximum(
                    corners[:, (corner + 1) % 3], corners[:, (corner + 2) % 3]
                )
                for corner in range(3)
            ],
            dim=1,
        ).reshape(-1)
        order = torch.argsort(edge_keys)
        paired = edge_keys[order][1:] == edge_keys[order][:-1]
        first, second = order[:-1][paired], order[1:][paired]
        face_neighbors = torch.full_like(edge_keys, -1)
        face_neighbors[first] = second // 3
        face_neighbors[second] = first // 3

        corner_order = torch.argsort(corners.reshape(-1))
        vertex_starts = torch.zeros(
            len(vertices) + 1, dtype=torch.int64, device=self.device
        )
        vertex_starts[1:] = torch.cumsum(
            torch.bincount(corners.reshape(-1), minlength=len(vertices)), dim=0
        )

        query_tensor = torch.as_tensor(
            np.ascontiguousarray(points, dtype=np.float32), device=self.device
        )
        inside = torch.empty(len(query_tensor), dtype=torch.bool, device=self.device)
        undecided = torch.empty_like(inside)
        torch.cuda.synchronize()
        wp.launch(
            _inside_surface_kernel,
            dim=len(query_tensor),
            inputs=[
                mesh.id,
                self.EDGE_BAND,
                mesh.indices,
                wp.from_torch(face_normals, dtype=wp.vec3),
                wp.from_torch(face_neighbors.int()),
                wp.from_torch(vertex_starts.int()),
                wp.from_torch((corner_order // 3).int()),
                wp.from_torch(query_tensor, dtype=wp.vec3),
                wp.from_torch(inside),
                wp.from_torch(undecided),
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        result = inside.cpu().numpy()
        to_vtk = undecided.cpu().numpy()
        if to_vtk.any():
            surface = pv.PolyData.from_regular_faces(
                vertices.cpu().numpy().astype(np.float64), corners.cpu().numpy()
            )
            distance = vtk.vtkImplicitPolyDataDistance()
            distance.SetInput(surface)
            values = vtk.vtkDoubleArray()
            distance.FunctionValue(pv.convert_array(points[to_vtk]), values)
            result[to_vtk] = pv.convert_array(values) < 0.0
        return result

    def scaled_jacobian(
        self, points: torch.Tensor, connectivity: torch.Tensor
    ) -> torch.Tensor:
        """Return VTK's tetrahedron scaled Jacobian (Verdict) for every cell."""
        import torch

        p0, p1, p2, p3 = points[connectivity].unbind(dim=1)
        side0, side1, side2 = p1 - p0, p2 - p1, p0 - p2
        side3, side4, side5 = p3 - p0, p3 - p1, p3 - p2
        jacobian = (side3 * torch.cross(side2, side0, dim=1)).sum(dim=1)
        l0, l1, l2, l3, l4, l5 = (
            torch.linalg.vector_norm(side, dim=1)
            for side in (side0, side1, side2, side3, side4, side5)
        )
        length_product = torch.maximum(
            torch.maximum(l0 * l2 * l3, l0 * l1 * l4),
            torch.maximum(l1 * l2 * l5, l3 * l4 * l5),
        )
        scaled = jacobian * (2.0**0.5) / length_product
        return torch.where(
            length_product > torch.finfo(torch.float64).tiny, scaled, 0.0
        )

    def squared_distance_to_features(
        self, features: torch.Tensor, spacing: Sequence[float]
    ) -> torch.Tensor:
        """Return every voxel's squared distance, in mm squared, to a feature voxel.

        The distance is the exact Euclidean one, taken between voxel centers, with
        *spacing* the voxel pitch along each axis of *features*.
        """
        import torch

        wp.init()
        values = torch.where(features, 0.0, self.NO_FEATURE)
        for axis in range(3):
            lines = values.movedim(axis, 0).contiguous()
            flat = lines.reshape(lines.shape[0], -1)
            position = torch.empty(flat.shape, dtype=torch.int32, device=self.device)
            bound = torch.empty_like(flat)
            result = torch.empty_like(flat)
            torch.cuda.synchronize()
            wp.launch(
                _squared_distance_line_kernel,
                dim=flat.shape[1],
                inputs=[
                    wp.from_torch(flat),
                    self.NO_FEATURE_LIMIT,
                    float(spacing[axis]) ** 2,
                    wp.from_torch(position),
                    wp.from_torch(bound),
                    wp.from_torch(result),
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
            values = result.reshape(lines.shape).movedim(0, axis)
        return values.contiguous()

    def _edge_depth_mm(
        self, shape: Sequence[int], spacing: Sequence[float]
    ) -> torch.Tensor:
        """Return how far ITK's signed Maurer map puts each voxel's own edge inside it.

        ITK measures a voxel to itself as the gap between its coordinate taken as a
        float product and as a rounded double product, so an edge voxel sits a
        rounding error inside its label rather than at zero, and only at some
        voxels.  An isocontour at zero reads an exact zero as outside, so the
        pattern decides which voxels a contour passes between.
        """
        import torch

        depth = torch.zeros(shape, device=self.device)
        for axis, size in enumerate(shape):
            index = torch.arange(size, dtype=torch.float64, device=self.device)
            pitch = float(spacing[axis])
            rounded = (index * pitch).to(torch.float32)
            product = index.to(torch.float32) * torch.tensor(
                pitch, dtype=torch.float32, device=self.device
            )
            gap = ((product - rounded) ** 2).reshape(
                [-1 if other == axis else 1 for other in range(len(shape))]
            )
            depth = depth + gap
        return torch.sqrt(depth)

    def gaussian_blur(
        self, field: torch.Tensor, spacing: Sequence[float], sigma_mm: float
    ) -> torch.Tensor:
        """Blur *field* as ITK's recursive Gaussian does, repeating its edge values."""
        import torch
        from torch.nn import functional

        blurred = field[None, None]
        for axis in range(3):
            taps = torch.as_tensor(
                _recursive_gaussian_taps(sigma_mm / float(spacing[axis])),
                dtype=field.dtype,
                device=field.device,
            )
            radius = len(taps) // 2
            padding = [0] * 6
            padding[2 * (2 - axis)] = padding[2 * (2 - axis) + 1] = radius
            shape = [1, 1, 1, 1, 1]
            shape[2 + axis] = -1
            blurred = functional.conv3d(
                functional.pad(blurred, padding, mode="replicate"), taps.view(shape)
            )
        return blurred[0, 0]

    def _vote_weights(
        self, input_size: int, output_size: int, step: float, sigma: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return how much each input voxel counts toward each output voxel's vote.

        A voxel counts by the mass of a Gaussian of *sigma* (in input voxels) over
        it, centered on the output voxel; output voxels are *step* input voxels
        apart.  Voxels beyond ``VOTE_ALPHA`` sigmas of the center do not count.
        Also returns, per output voxel, the first and one-past-last input voxel
        that counts, since the weights are banded.
        """
        import torch

        centers = torch.arange(output_size, dtype=torch.float64, device=self.device)
        voxels = torch.arange(input_size, dtype=torch.float64, device=self.device)
        offsets = voxels[None, :] - centers[:, None] * step
        scale = 1.0 / (math.sqrt(2.0) * sigma)
        mass = torch.erf((offsets + 0.5) * scale) - torch.erf((offsets - 0.5) * scale)
        counted = offsets.abs() < self.VOTE_ALPHA * sigma + 0.5
        weights = torch.where(counted, mass, 0.0)
        occupied = weights != 0.0
        first = torch.where(
            occupied.any(dim=1), occupied.to(torch.int8).argmax(dim=1), 0
        ).to(torch.int32)
        last = torch.where(
            occupied.any(dim=1),
            input_size - occupied.flip(1).to(torch.int8).argmax(dim=1),
            0,
        ).to(torch.int32)
        return weights, first, last

    def _banded_sum(
        self,
        field: torch.Tensor,
        banded: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        axis: int,
    ) -> torch.Tensor:
        """Return *field* with its *axis* replaced by the weighted sums over it."""
        import torch

        weights, first, last = banded
        wp.init()
        lines = field.movedim(axis, 0).contiguous()
        flat = lines.reshape(lines.shape[0], -1)
        result = torch.empty(
            (weights.shape[0], flat.shape[1]), dtype=torch.float64, device=self.device
        )
        torch.cuda.synchronize()
        wp.launch(
            _banded_sum_kernel,
            dim=result.shape,
            inputs=[
                wp.from_torch(flat),
                wp.from_torch(weights),
                wp.from_torch(first),
                wp.from_torch(last),
                wp.from_torch(result),
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return result.reshape(weights.shape[0], *lines.shape[1:]).movedim(0, axis)

    def vote_resample_labelmap(
        self,
        labelmap_image: itk.image,
        isotropic_spacing_mm: float,
        sigma_mm: Optional[np.ndarray] = None,
    ) -> itk.image:
        """Resample a labelmap onto an isotropic grid, label boundaries intact.

        This reproduces ITK's ``LabelImageGaussianInterpolateImageFunction``
        on the GPU, so labels stay whole numbers and keep sharing their walls.
        Each output voxel takes the label with the largest Gaussian-weighted
        vote, where a voxel's weight is the Gaussian's mass over it.  The
        weights separate along the axes, so every label's votes come from
        three banded sums.  Exact ties, which ITK settles by rounding noise, go
        to the smaller label.

        Args:
            labelmap_image: Labelmap to resample.
            isotropic_spacing_mm: Edge length of the output voxels.
            sigma_mm: Per-axis width of the voting kernel.  ``None`` uses one
                input voxel along each axis, which is what lets a coarse axis
                interpolate between its slices.
        """
        import torch

        spacing = np.asarray(labelmap_image.GetSpacing(), dtype=np.float64)
        size = itk.size(labelmap_image)
        sigma = spacing if sigma_mm is None else sigma_mm
        output_size = [
            int((size[axis] - 1) * spacing[axis] / isotropic_spacing_mm) + 1
            for axis in range(3)
        ]
        banded_zyx = [
            self._vote_weights(
                size[axis],
                output_size[axis],
                isotropic_spacing_mm / spacing[axis],
                sigma[axis] / spacing[axis],
            )
            for axis in (2, 1, 0)
        ]
        input_arr = itk.GetArrayViewFromImage(labelmap_image)
        labels = torch.as_tensor(input_arr.astype(np.int32), device=self.device)

        output_shape = (output_size[2], output_size[1], output_size[0])
        best_vote = torch.full(
            output_shape, -1.0, dtype=torch.float64, device=self.device
        )
        best_label = torch.zeros(output_shape, dtype=torch.int32, device=self.device)
        for label_id in torch.unique(labels).tolist():
            votes = (labels == label_id).to(torch.float64)
            for axis, banded in enumerate(banded_zyx):
                votes = self._banded_sum(votes, banded, axis)
            wins = votes > best_vote
            best_vote = torch.where(wins, votes, best_vote)
            best_label = torch.where(wins, label_id, best_label)

        resampled = itk.GetImageFromArray(
            np.ascontiguousarray(best_label.cpu().numpy().astype(input_arr.dtype))
        )
        resampled.SetSpacing([isotropic_spacing_mm] * 3)
        resampled.SetOrigin(labelmap_image.GetOrigin())
        resampled.SetDirection(labelmap_image.GetDirection())
        return resampled

    def signed_distance_mm(
        self,
        labels: torch.Tensor,
        spacing: Sequence[float],
        label_id: int,
        sigma_mm: float,
    ) -> torch.Tensor:
        """Return the signed distance, in mm, to *label_id*, negative inside.

        The map measures to voxel centers, so its zero level is faceted at the
        voxel pitch; *sigma_mm* of blur takes those facets out, at the cost of
        pulling the level set in by roughly ``sigma_mm ** 2`` times the surface
        curvature.  ``label_id`` of ``0`` measures to the background, whose
        distance is the negated distance to the union of every label.

        Like ITK's signed Maurer map, distances run to the voxels of the label
        that touch a voxel outside it, so a voxel on the label's edge is at
        distance zero.  *labels* sits on the GPU and *spacing* is its voxel
        pitch along each axis.
        """
        import torch
        from torch.nn import functional

        inside = labels != 0 if label_id == 0 else labels == label_id
        neighborhood_minimum = -functional.max_pool3d(
            -inside[None, None].to(torch.float32), 3, 1, 1
        )[0, 0]
        edge = inside & (neighborhood_minimum < 0.5)
        distance = torch.sqrt(self.squared_distance_to_features(edge, spacing))
        distance = torch.where(
            edge, self._edge_depth_mm(labels.shape, spacing), distance
        )
        distance = torch.where(inside, -distance, distance)
        if sigma_mm > 0.0:
            distance = self.gaussian_blur(distance, spacing, sigma_mm)
        return -distance if label_id == 0 else distance
