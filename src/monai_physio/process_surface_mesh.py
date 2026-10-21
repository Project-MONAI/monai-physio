"""
Tools for creating and manipulating surface meshes.

This module provides the :class:`ProcessSurfaceMesh` class: surfaces from
labelmaps and masks, surface remeshing, smoothing and combining, and the
conversions from surfaces back to images (masks, labelmaps, distance maps,
deformation fields).  Tetrahedral meshes live in
:class:`ProcessVolumetricMesh`; the GPU primitives both use live in
:class:`ProcessMeshKernels`.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Optional, cast

import itk
import numpy as np
import pyacvd
import pyvista as pv
import trimesh

from .monai_physio_base import MONAIPhysioBase
from .process_images import ProcessImages
from .process_mesh_kernels import ProcessMeshKernels
from .process_transforms import ProcessTransforms


class ProcessSurfaceMesh(MONAIPhysioBase):
    """
    Tools for creating and manipulating surface meshes.
    """

    # Background voxels kept around the labels in extract_label_surfaces, enough
    # for the isotropic grid to hold a full voxel of background on every side.
    LABEL_SURFACE_PAD = 3

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize ProcessSurfaceMesh.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self._image_tools = ProcessImages(log_level=log_level)
        self._kernels = ProcessMeshKernels(log_level=log_level)

    def extract_label_surfaces(
        self,
        labelmap_image: itk.image,
        isotropic_spacing_mm: Optional[float] = None,
        distance_sigma_mm: Optional[float] = None,
        smoothing_iterations: int = 30,
        label_ids: Optional[Sequence[int]] = None,
        surface_reduction_rate: float = 0.0,
    ) -> dict[int, pv.PolyData]:
        """Extract each label's closed surface, conforming with its neighbors.

        Contouring each label on its own traces its voxel block edges: on
        anisotropic data the result terraces at the slice pitch, and a wall
        between two labels is meshed twice, in copies that do not match.  This
        extracts the labels together instead:

        1. The labelmap is resampled onto an isotropic grid with a label-aware
           Gaussian vote over a physical-space kernel rather than a nearest
           voxel.  That is what removes the terracing: a boundary between two
           slices lands between them instead of on one of them.
        2. Every label, plus the background, gets a signed distance map on that
           grid, and each voxel goes to the label whose map is smallest.  The
           assignment is a partition, so no gap or overlap can arise.
        3. Label ``L``'s surface is the zero level of the margin between the
           closest and the second closest map, negated where ``L`` is the
           closest.  On a wall between ``L`` and ``M`` that field is the
           negation of ``M``'s, so marching cubes puts identical vertices on
           both surfaces and the two meet exactly, including where three
           labels meet.
        4. The surfaces are merged, which welds those coincident vertices, then
           Taubin-smoothed as one mesh.  Smoothing therefore moves a shared
           vertex once and the surfaces stay in contact.

        Structures thinner than the interpolation kernel lose volume, coronary
        arteries most of all; the fraction of each label's voxel volume that
        the surface encloses is logged at debug level.

        Args:
            labelmap_image: Multi-label image; every non-zero label present is
                extracted.  A binary mask yields a single surface.
            isotropic_spacing_mm: Edge length of the isotropic grid the
                surfaces are contoured on, which sets both their smoothness and
                their triangle count.  ``None`` uses the labelmap's finest
                spacing.
            distance_sigma_mm: Blur applied to the distance maps, which is what
                takes the voxel facets out of the contoured surface.  ``None``
                uses the isotropic spacing; raising it smooths further and
                thins the smallest structures.
            smoothing_iterations: Taubin smoothing iterations (0 disables).
            label_ids: Labels to extract; every other label is treated as
                background.  ``None`` extracts every label.  To extract a union
                of labels as one surface, relabel them to one id first.
            surface_reduction_rate: Fraction of each surface's triangles to
                remove by :meth:`remesh_and_smooth_surface` (0.0 disables).
                Each label is remeshed on its own, so neighboring surfaces no
                longer share their vertices; lower the triangle count with a
                coarser *isotropic_spacing_mm* instead to keep them matching.

        Returns:
            Label id → that label's closed, outward-oriented surface, whose
            ``field_data['SegmentationLabelIds']`` holds the id.  A label too
            small to survive the isotropic grid is left out, so the mapping is
            empty when the labelmap holds no non-zero label and may be missing
            labels that it does hold.
        """
        import torch

        if label_ids is not None:
            all_labels = itk.GetArrayViewFromImage(labelmap_image)
            selected = itk.GetImageFromArray(
                np.where(np.isin(all_labels, label_ids), all_labels, 0)
            )
            selected.CopyInformation(labelmap_image)
            labelmap_image = selected
        labels = itk.GetArrayViewFromImage(labelmap_image)
        present, voxel_counts = torch.unique(
            torch.as_tensor(labels.astype(np.int32), device=self._kernels.device),
            return_counts=True,
        )
        label_voxels = dict(zip(present.tolist(), voxel_counts.tolist()))
        present_ids = [value for value in label_voxels if value != 0]
        if not present_ids:
            self.log_warning("Labelmap holds no non-zero label")
            return {}

        spacing = np.asarray(labelmap_image.GetSpacing(), dtype=np.float64)
        iso = (
            float(np.min(spacing))
            if isotropic_spacing_mm is None
            else isotropic_spacing_mm
        )
        sigma = iso if distance_sigma_mm is None else distance_sigma_mm
        # The pad closes structures that reach the image border, which would
        # otherwise contour to an open surface, and gives the distance maps
        # room to fall away from the outermost label.
        cropped = self._image_tools.crop_to_nonzero(
            labelmap_image, self.LABEL_SURFACE_PAD
        )
        fine = self._kernels.vote_resample_labelmap(cropped, iso)
        fine_labels = torch.as_tensor(
            itk.GetArrayViewFromImage(fine).astype(np.int32),
            device=self._kernels.device,
        )
        fine_spacing = [iso] * 3

        # Pass one: the closest and second closest label at every voxel, one
        # distance map at a time, so memory does not grow with the label count.
        closest = torch.full(fine_labels.shape, torch.inf, device=self._kernels.device)
        runner_up = torch.full_like(closest, torch.inf)
        closest_index = torch.zeros(
            fine_labels.shape, dtype=torch.int16, device=self._kernels.device
        )
        for index, label_id in enumerate(present_ids + [0]):
            distance = self._kernels.signed_distance_mm(
                fine_labels, fine_spacing, label_id, sigma
            )
            is_closer = distance < closest
            runner_up = torch.where(
                is_closer, closest, torch.minimum(runner_up, distance)
            )
            closest = torch.where(is_closer, distance, closest)
            closest_index = torch.where(is_closer, index, closest_index)

        # Pass two: one surface per label, tagged so the merged mesh can be
        # split again after smoothing.  Label L's field is the margin between
        # the closest and second closest maps, negative where L is closest.  On
        # a wall between L and M that field is the negation of M's, so marching
        # cubes puts identical vertices on both surfaces; it does so where three
        # labels meet too, since only the two closest labels ever enter it.
        margin = runner_up - closest
        parts: list[pv.PolyData] = []
        for index, label_id in enumerate(present_ids):
            field = itk.GetImageFromArray(
                np.ascontiguousarray(
                    torch.where(closest_index == index, -margin, margin).cpu().numpy()
                )
            )
            field.CopyInformation(fine)
            part = cast(
                pv.PolyData,
                pv.wrap(itk.vtk_image_from_image(field)).contour(
                    [0.0], method="flying_edges"
                ),
            )
            # Voxels where two labels tie exactly contour to zero-area
            # triangles.  clean turns those into line cells rather than
            # dropping them, so the polygons are then taken on their own.
            part = part.clean().triangulate()
            part = pv.PolyData(part.points, faces=part.faces)
            part.cell_data["LabelId"] = np.full(part.n_cells, label_id, dtype=np.int32)
            parts.append(part)

        merged = cast(pv.PolyData, pv.merge(parts, merge_points=True))
        if smoothing_iterations > 0:
            # Every wall between two labels is meshed twice, so its edges are
            # non-manifold; without non_manifold_smoothing VTK pins them and
            # nothing moves.
            merged = merged.smooth_taubin(
                n_iter=smoothing_iterations, non_manifold_smoothing=True
            )

        voxel_volume = float(np.prod(spacing))
        merged_ids = np.asarray(merged.cell_data["LabelId"])
        surfaces: dict[int, pv.PolyData] = {}
        for label_id in present_ids:
            # Kept as an array rather than a list: a label with no cells gives an
            # empty selection, and an empty list has no integer dtype for
            # extract_cells to recognize it by.  Empty here means no surface,
            # which is what the next branch reports.
            cell_ids = cast(Sequence[int], np.flatnonzero(merged_ids == label_id))
            surface = self.extract_surface(merged.extract_cells(cell_ids)).triangulate()
            if surface.n_cells == 0:
                # A label smaller than the isotropic grid loses its vote to its
                # neighbors, so no voxel is assigned to it and its field never
                # crosses zero.  It has no surface to return.
                self.log_warning(
                    "Label %d is too small for a %.3g mm grid; it has no surface",
                    label_id,
                    iso,
                )
                continue
            # The bookkeeping arrays of the merge and the split.
            for array_name in ("LabelId", "vtkOriginalCellIds", "vtkOriginalPointIds"):
                surface.cell_data.pop(array_name, None)
                surface.point_data.pop(array_name, None)
            if surface_reduction_rate > 0.0:
                surface = self.remesh_and_smooth_surface(
                    surface, surface_reduction_rate
                )
            # VTK winds faces for a right-handed direction matrix, so an LPS
            # image with a negative-determinant direction comes out inward.
            surface = surface.compute_normals(
                auto_orient_normals=True, consistent_normals=True
            )
            surface.field_data["SegmentationLabelIds"] = np.array(
                [label_id], dtype=np.int32
            )
            surfaces[label_id] = surface
            if self.logger.isEnabledFor(logging.DEBUG):
                self.log_debug(
                    "Label %d: %d triangles, %.3f of its voxel volume",
                    label_id,
                    surfaces[label_id].n_cells,
                    float(surfaces[label_id].volume)
                    / (label_voxels[label_id] * voxel_volume),
                )
        return surfaces

    @staticmethod
    def is_watertight(surface: pv.PolyData) -> bool:
        """Report whether every edge of *surface* is shared by exactly two faces.

        A surface with no faces has no edge that fails the test, so it is
        reported as not watertight rather than vacuously watertight.
        """
        faces = surface.triangulate().faces.reshape(-1, 4)[:, 1:]
        if len(faces) == 0:
            return False
        edges = np.sort(
            np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    def remesh_and_smooth_surface(
        self,
        surface: pv.PolyData,
        surface_reduction_rate: float = 0.0,
        smoothing_iterations: int = 0,
    ) -> pv.PolyData:
        """Optionally remesh then smooth a surface (no-op when disabled).

        Reduction is isotropic remeshing (ACVD, via ``pyacvd``) rather than
        decimation: the surface is re-tiled with uniform, well-shaped triangles
        at the requested resolution.  ``decimate_pro`` reaches the same triangle
        count but leaves a watertight input non-watertight; ACVD does not.

        Remeshing rebuilds the topology and so discards cell data, exactly as
        ``decimate_pro`` did: per-cell ``SegmentationLabelIds`` (which structure
        each triangle belongs to, as :meth:`combine_surfaces` writes it) is
        transferred back onto the new cells from their nearest original cell so
        anatomy materials and structure ids still apply.  Smoothing uses
        non-shrinking Taubin smoothing, which only moves points and therefore
        preserves cells and their labels.  It is told to move non-manifold
        points too, since on a combined surface the walls where structures meet
        are non-manifold and VTK pins those points otherwise; on a manifold
        surface the setting has nothing to act on.

        Args:
            surface: Input surface.
            surface_reduction_rate: Fraction of triangles to remove (0.0 disables).
            smoothing_iterations: Taubin smoothing iterations (0 disables).

        Returns:
            The remeshed and/or smoothed surface.
        """
        conditioned = surface
        if surface_reduction_rate > 0.0:
            original = conditioned
            triangulated = conditioned.triangulate()
            # ACVD's clustering graph assumes a single connected component; a
            # surface merged from disjoint parts (e.g. left and right lung,
            # merged by combine_surfaces) crashes
            # pyacvd's native clustering step otherwise. Remesh each connected
            # component on its own and recombine.
            regions = triangulated.connectivity(extraction_mode="all")
            remeshed_pieces = []
            for region_id in np.unique(regions.cell_data["RegionId"]):
                piece = regions.threshold(
                    (region_id, region_id), scalars="RegionId", preference="cell"
                ).extract_surface(algorithm="geometry")
                # pyacvd 0.4.0 still uses a pyvista-deprecated private API.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*_connectivity_array.*")
                    clustering = pyacvd.Clustering(piece)
                    # One cluster per retained point.  A closed surface carries
                    # about twice as many triangles as points, so scaling the
                    # point count by (1 - rate) scales the triangle count by the
                    # same fraction; four is the fewest clusters that can still
                    # close a surface.
                    clustering.cluster(
                        max(4, round(piece.n_points * (1.0 - surface_reduction_rate)))
                    )
                    remeshed_pieces.append(clustering.create_mesh())
            conditioned = pv.merge(remeshed_pieces)
            if "SegmentationLabelIds" in original.cell_data:
                nearest = original.find_closest_cell(conditioned.cell_centers().points)
                conditioned.cell_data["SegmentationLabelIds"] = np.asarray(
                    original.cell_data["SegmentationLabelIds"]
                )[nearest]
        if smoothing_iterations > 0:
            conditioned = conditioned.smooth_taubin(
                n_iter=smoothing_iterations, non_manifold_smoothing=True
            )
        return conditioned

    @staticmethod
    def extract_surface(mesh: pv.DataSet) -> pv.PolyData:
        """Extract the surface of a mesh.

        Args:
            mesh: Input mesh (PolyData is returned unchanged; any other DataSet
                is passed through ``extract_surface``).

        Returns:
            pv.PolyData: The surface of the mesh.
        """
        if isinstance(mesh, pv.PolyData):
            return mesh
        return mesh.extract_surface(algorithm="dataset_surface")

    @staticmethod
    def transform_contours(
        contours: pv.PolyData,
        tfm: itk.Transform,
        with_deformation_magnitude: bool = False,
    ) -> pv.PolyData:
        """
        Transform contours using a given transform.

        Args:
            tfm (itk.Transform): The transform to use

        Returns:
            pv.PolyData: The transformed contours with deformation magnitude
        """
        new_contours = ProcessTransforms().transform_pvcontour(
            contours, tfm, with_deformation_magnitude=with_deformation_magnitude
        )

        return new_contours

    def merge_meshes(
        self, meshes: list[pv.PolyData]
    ) -> tuple[pv.PolyData, list[pv.PolyData]]:
        """
        Merge multiple fixed meshes into a single mesh.

        Returns
        -------
        pv.PolyData
            Merged mesh
        """
        self.log_info("Merging meshes...")
        trimesh_meshes: list[trimesh.Trimesh] = []
        if hasattr(meshes[0], "n_faces_strict"):
            trimesh_meshes = [
                trimesh.Trimesh(
                    vertices=mesh.points,
                    faces=mesh.faces.reshape((mesh.n_faces_strict, 4))[:, 1:],
                )
                for mesh in meshes
            ]
        else:
            trimesh_meshes = [
                trimesh.Trimesh(
                    vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:4]
                )
                for mesh in meshes
            ]

        # Merge meshes
        merged_trimesh = trimesh.util.concatenate(trimesh_meshes)
        flip_matrix = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        merged_trimesh.apply_transform(flip_matrix)  # Apply flip transformation
        for mesh in trimesh_meshes:
            mesh.apply_transform(flip_matrix)

        merged_mesh = pv.wrap(merged_trimesh)
        pv_meshes = [pv.wrap(mesh) for mesh in trimesh_meshes]

        return merged_mesh, pv_meshes

    @staticmethod
    def create_reference_image(
        mesh: pv.DataSet,
        spatial_resolution: float = 0.5,
        buffer_factor: float = 0.25,
        ptype: type = itk.F,
    ) -> itk.Image:
        """
        Create a reference image from a mesh.
        """
        points = np.array(mesh.points)
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        min_bounds = min_bounds - buffer_factor * (max_bounds - min_bounds)
        max_bounds = max_bounds + buffer_factor * (max_bounds - min_bounds)
        region = (
            ((max_bounds - min_bounds) / spatial_resolution + 1)
            .astype(np.int32)
            .tolist()
        )
        itk_region = itk.ImageRegion[3]()
        itk_region.SetSize(region)
        reference_image = itk.Image[ptype, 3].New()
        reference_image.SetRegions(itk_region)
        reference_image.SetSpacing([spatial_resolution] * 3)
        reference_image.SetOrigin(min_bounds.tolist())
        reference_image.Allocate()
        return reference_image

    @staticmethod
    def create_mask_from_mesh(
        mesh: pv.DataSet | pv.UnstructuredGrid,
        reference_image: itk.Image,
    ) -> itk.Image:
        ref_spacing = np.array(reference_image.GetSpacing())

        # Create trimesh object with LPS coordinates
        if isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.extract_surface(algorithm="dataset_surface")

        if hasattr(mesh, "n_faces_strict"):
            # PyVista PolyData
            num_points_per_face = len(mesh.faces) // mesh.n_faces_strict
            faces = mesh.faces.reshape((mesh.n_faces_strict, num_points_per_face))[
                :, 1:
            ]
        else:
            # Handle other mesh types
            faces = mesh.faces.reshape((-1, 4))[:, 1:]

        trimesh_mesh = trimesh.Trimesh(vertices=mesh.points, faces=faces)

        # Determine voxel spacing (use minimum spacing from reference)
        voxel_pitch = float(np.min(ref_spacing))

        # Voxelize the mesh
        # trimesh.voxelized() creates a grid aligned with the mesh's bounding box
        # The voxel grid origin is at the minimum corner of the bounding box
        vox = trimesh_mesh.voxelized(pitch=voxel_pitch)
        binary_array = vox.matrix.astype(np.uint8)

        # Get the physical origin of the voxel grid in LPS space
        # trimesh voxel grids use a transformation matrix, and the voxel grid starts
        # at the mesh's minimum bounds. The physical origin is where voxel [0,0,0]
        # center is located.
        # Get mesh bounds in LPS coordinates
        mesh_bounds_lps = (
            trimesh_mesh.bounds
        )  # shape (2, 3): [[x_min, y_min, z_min], [x_max, y_max, z_max]]

        # The voxel grid origin is at the minimum corner, but ITK origin is the CENTER
        # of voxel (0,0,0)
        # So we need to add half a voxel pitch to each dimension
        voxel_grid_origin_lps = mesh_bounds_lps[0] + voxel_pitch / 2.0
        voxel_grid_origin_lps[2] = (
            voxel_grid_origin_lps[2] + voxel_pitch * binary_array.shape[2]
        )

        # transpose to match trimesh XYZ convention
        binary_array_zyx = np.transpose(binary_array, (2, 1, 0))
        binary_array_flip = np.flip(binary_array_zyx, axis=0)
        binary_image = itk.GetImageFromArray(binary_array_flip)

        # Set ITK image metadata in LPS coordinates
        # Origin: where the center of voxel (0,0,0) is located in physical space
        binary_image.SetOrigin(voxel_grid_origin_lps)

        # Spacing: uniform voxel pitch in all directions
        binary_image.SetSpacing([voxel_pitch] * 3)

        # Direction: use identity for now (axis-aligned), will be handled by resampling
        # Flip Z axis to match ITK convention
        ref_dir = itk.array_from_matrix(binary_image.GetDirection())
        ref_dir[2, 2] = -ref_dir[2, 2]
        binary_image.SetDirection(ref_dir)

        # Fill holes to create solid mask
        ImageType = type(binary_image)
        fill_filter = itk.BinaryFillholeImageFilter[ImageType].New()
        fill_filter.SetInput(binary_image)
        fill_filter.SetForegroundValue(1)
        fill_filter.Update()
        mask_image = fill_filter.GetOutput()

        resampler = itk.ResampleImageFilter.New(Input=mask_image)
        resampler.SetReferenceImage(reference_image)
        resampler.SetUseReferenceImage(True)
        resampler.SetInterpolator(
            itk.NearestNeighborInterpolateImageFunction.New(mask_image)
        )
        resampler.SetDefaultPixelValue(0)
        resampler.Update()
        mask_image = resampler.GetOutput()

        return mask_image

    def create_labelmap_from_meshes(
        self,
        meshes: list[pv.DataSet | pv.UnstructuredGrid],
        reference_image: itk.Image,
    ) -> itk.Image:
        """
        Create a labelmap from a list of meshes.
        """
        labelmap_arr = np.zeros(
            (
                reference_image.GetLargestPossibleRegion().GetSize()[2],
                reference_image.GetLargestPossibleRegion().GetSize()[1],
                reference_image.GetLargestPossibleRegion().GetSize()[0],
            ),
            dtype=np.uint16,
        )
        for i, mesh in enumerate(meshes):
            mask_image = self.create_mask_from_mesh(mesh, reference_image)
            mask_arr = itk.GetArrayFromImage(mask_image)
            labelmap_arr[mask_arr > 0] = i + 1

        labelmap_image = itk.GetImageFromArray(labelmap_arr)
        labelmap_image.CopyInformation(reference_image)

        return labelmap_image

    @staticmethod
    def sample_mesh_faces(mesh: pv.DataSet, max_spacing: float) -> np.ndarray:
        """Return mesh points supplemented by samples across the triangle faces.

        Rasterizing vertices alone leaves gaps between them on meshes that are
        coarse relative to the voxel size, which makes a distance map built from
        them ripple. Adding barycentric samples dense enough that consecutive
        samples are closer than ``max_spacing`` closes those gaps.

        Args:
            mesh: Source mesh; its surface is triangulated if needed.
            max_spacing: Target spacing between samples, in mm.

        Returns:
            (n, 3) array of sample points, starting with the mesh's own points.
        """
        points = np.asarray(mesh.points, dtype=np.float64)
        surface = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh
        surface = surface.triangulate()
        if surface.faces.size == 0:
            return points
        faces = surface.faces.reshape(-1, 4)[:, 1:]

        vertices = np.asarray(surface.points, dtype=np.float64)
        corners = vertices[faces]  # (n_faces, 3, 3)
        edge_lengths = np.linalg.norm(
            corners - np.roll(corners, 1, axis=1), axis=2
        ).max(axis=1)

        # Group faces by how finely they need to be subdivided so each division
        # level is generated as one vectorized batch.
        divisions = np.maximum(1, np.ceil(edge_lengths / max(max_spacing, 1e-6)))
        divisions = np.minimum(divisions, 64).astype(np.int64)

        samples = [points]
        for level in np.unique(divisions):
            if level < 2:
                continue
            selected = corners[divisions == level]
            # Barycentric lattice with `level` divisions per edge.
            steps = np.arange(level + 1, dtype=np.float64) / level
            u, v = np.meshgrid(steps, steps, indexing="ij")
            mask = (u + v) <= 1.0
            weights = np.column_stack([1.0 - u[mask] - v[mask], u[mask], v[mask]])
            samples.append(np.einsum("fca,kc->fka", selected, weights).reshape(-1, 3))

        return np.concatenate(samples, axis=0)

    def create_distance_map(
        self,
        mesh: pv.DataSet | pv.UnstructuredGrid,
        reference_image: itk.Image,
        squared_distance: bool = False,
        negative_inside: bool = True,
        zero_inside: bool = False,
        norm_to_max_distance: float = 0.0,
        sample_faces: bool = True,
    ) -> itk.Image:
        """Compute a distance map of a mesh on the reference image's grid.

        Args:
            mesh: Mesh whose surface the distances are measured to.
            reference_image: Image defining the output grid.
            squared_distance: Sign-preserving square of the result. Default: False
            negative_inside: Keep the signed output. Default: True
            zero_inside: Clip negative values to zero before anything else.
                Default: False
            norm_to_max_distance: If non-zero, divide by this value and clip to
                [-1, 1]. Default: 0.0 (distances stay in mm)
            sample_faces: Rasterize samples across the triangle faces as well as
                the vertices, so that coarse meshes do not leave gaps in the
                rasterized surface. Default: True

        Returns:
            ITK image of distances on the reference grid.
        """
        self.log_info("Computing signed distance map...")

        size = reference_image.GetLargestPossibleRegion().GetSize()

        if sample_faces:
            points = self.sample_mesh_faces(
                mesh, 0.5 * float(min(reference_image.GetSpacing()))
            )
            self.log_debug(
                "Distance map: %d face samples from %d mesh points",
                len(points),
                mesh.n_points,
            )
        else:
            points = np.asarray(mesh.points, dtype=np.float64)

        # NumPy convention is (z, y, x); ITK GetSize() returns (x, y, z)
        tmp_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)

        # Bulk equivalent of TransformPhysicalPointToIndex, which rounds half up.
        index_to_world = itk.array_from_matrix(
            reference_image.GetDirection()
        ) @ np.diag(np.asarray(reference_image.GetSpacing()))
        origin = np.asarray(reference_image.GetOrigin(), dtype=np.float64)
        indices = np.floor(
            (points - origin) @ np.linalg.inv(index_to_world).T + 0.5
        ).astype(np.int64)
        size_arr = np.array([size[0], size[1], size[2]], dtype=np.int64)
        inside = np.all((indices >= 0) & (indices < size_arr), axis=1)
        indices = indices[inside]
        point_count = len(indices)
        if point_count:
            tmp_arr[indices[:, 2], indices[:, 1], indices[:, 0]] = 1

        self.log_info(
            "Distance map: %d/%d surface samples within reference image",
            point_count,
            len(points),
        )
        if point_count == 0:
            self.log_warning(
                "No surface points fall within the reference image! "
                "Distance map will be constant. "
                "Mesh bounds: %s  Image origin: %s  Image size: %s  Image spacing: %s",
                str(mesh.bounds),
                str(reference_image.GetOrigin()),
                str(size),
                str(reference_image.GetSpacing()),
            )
        elif not inside.all():
            # Distances near the dropped region are measured to whatever samples
            # remain in the grid, so they are larger than the true distance.
            self.log_warning(
                "%d of %d surface samples fall outside the reference image; "
                "distances near that boundary are overestimated.",
                len(points) - point_count,
                len(points),
            )

        tmp_binary_image = itk.GetImageFromArray(tmp_arr)
        tmp_binary_image.CopyInformation(reference_image)
        assert (
            tmp_binary_image.GetLargestPossibleRegion().GetSize()
            == reference_image.GetLargestPossibleRegion().GetSize()
        )

        distance_filter = itk.SignedMaurerDistanceMapImageFilter.New(
            Input=tmp_binary_image
        )
        distance_filter.SetSquaredDistance(False)
        distance_filter.SetUseImageSpacing(True)
        distance_filter.Update()
        distance_image = distance_filter.GetOutput()

        distance_arr = itk.GetArrayFromImage(distance_image).astype(np.float32)
        if zero_inside:
            distance_arr = np.clip(distance_arr, 0.0, None)
        if not negative_inside:
            distance_arr = np.abs(distance_arr)
        if squared_distance:
            distance_arr = np.sign(distance_arr) * distance_arr**2
        if norm_to_max_distance != 0.0:
            distance_arr = distance_arr / norm_to_max_distance
            distance_arr = np.clip(distance_arr, -1.0, 1.0)
        distance_image = itk.GetImageFromArray(distance_arr)
        distance_image.CopyInformation(reference_image)

        return distance_image

    @staticmethod
    def create_deformation_field(
        points: np.ndarray,
        point_displacements: np.ndarray,
        reference_image: itk.Image,
        blur_sigma: float = 2.5,
        ptype: type = itk.D,
    ) -> itk.Image:
        """
        Create a displacement map from model points and displacements.
        """
        size = reference_image.GetLargestPossibleRegion().GetSize()
        norm_map = np.zeros((size[2], size[1], size[0])).astype(np.float32)
        displacement_map_x = np.zeros((size[2], size[1], size[0])).astype(np.float32)
        displacement_map_y = np.zeros((size[2], size[1], size[0])).astype(np.float32)
        displacement_map_z = np.zeros((size[2], size[1], size[0])).astype(np.float32)
        itk_point = itk.Point[itk.D, 3]()
        for i, point in enumerate(points):
            itk_point[0] = float(point[0])
            itk_point[1] = float(point[1])
            itk_point[2] = float(point[2])
            indx = reference_image.TransformPhysicalPointToIndex(itk_point)
            if (
                indx[0] < 0
                or indx[1] < 0
                or indx[2] < 0
                or indx[0] >= size[0]
                or indx[1] >= size[1]
                or indx[2] >= size[2]
            ):
                continue
            displacement_map_x[int(indx[2]), int(indx[1]), int(indx[0])] = (
                point_displacements[i, 0]
            )
            displacement_map_y[int(indx[2]), int(indx[1]), int(indx[0])] = (
                point_displacements[i, 1]
            )
            displacement_map_z[int(indx[2]), int(indx[1]), int(indx[0])] = (
                point_displacements[i, 2]
            )
            norm_map[int(indx[2]), int(indx[1]), int(indx[0])] = 1

        norm_img = itk.GetImageFromArray(norm_map)
        norm_img.CopyInformation(reference_image)
        assert (
            norm_img.GetLargestPossibleRegion().GetSize()
            == reference_image.GetLargestPossibleRegion().GetSize()
        )

        blurred_norm = itk.smoothing_recursive_gaussian_image_filter(
            Input=norm_img, Sigma=blur_sigma
        )
        blurred_norm_arr = itk.GetArrayFromImage(blurred_norm)
        blurred_norm_arr = np.where(blurred_norm_arr < 1.0e-4, 1.0e-4, blurred_norm_arr)

        deformation_field_x_img = itk.GetImageFromArray(displacement_map_x)
        deformation_field_x_img.CopyInformation(reference_image)
        deformation_field_x_img = itk.smoothing_recursive_gaussian_image_filter(
            Input=deformation_field_x_img, Sigma=blur_sigma
        )

        deformation_field_y_img = itk.GetImageFromArray(displacement_map_y)
        deformation_field_y_img.CopyInformation(reference_image)
        deformation_field_y_img = itk.smoothing_recursive_gaussian_image_filter(
            Input=deformation_field_y_img, Sigma=blur_sigma
        )

        deformation_field_z_img = itk.GetImageFromArray(displacement_map_z)
        deformation_field_z_img.CopyInformation(reference_image)
        deformation_field_z_img = itk.smoothing_recursive_gaussian_image_filter(
            Input=deformation_field_z_img, Sigma=blur_sigma
        )

        deformation_field_x = (
            itk.GetArrayFromImage(deformation_field_x_img) / blurred_norm_arr
        )
        deformation_field_y = (
            itk.GetArrayFromImage(deformation_field_y_img) / blurred_norm_arr
        )
        deformation_field_z = (
            itk.GetArrayFromImage(deformation_field_z_img) / blurred_norm_arr
        )

        deformation_field_x = np.where(
            blurred_norm_arr > 1.0e-3, deformation_field_x, 0.0
        )
        deformation_field_y = np.where(
            blurred_norm_arr > 1.0e-3, deformation_field_y, 0.0
        )
        deformation_field_z = np.where(
            blurred_norm_arr > 1.0e-3, deformation_field_z, 0.0
        )

        deformation_field = np.stack(
            [deformation_field_x, deformation_field_y, deformation_field_z], axis=-1
        )

        image_tools = ProcessImages()
        deformation_field_img = image_tools.convert_array_to_image_of_vectors(
            deformation_field, reference_image, ptype=ptype
        )

        return deformation_field_img

    @staticmethod
    def combine_surfaces(surfaces: Mapping[Any, pv.PolyData]) -> pv.PolyData:
        """Merge *surfaces* into one mesh that still tells its structures apart.

        Each cell gains a ``SegmentationLabelIds`` (int32) entry naming the
        label it came from, taken from its surface's single-entry
        ``field_data['SegmentationLabelIds']``; a surface holding several ids,
        such as a group surface, is tagged ``0``.  :class:`ConvertVTKToUSD`
        splits on that array, giving one prim and one anatomy material per
        structure.  The per-cell ``Color`` survives the merge; per-object
        ``field_data`` (``AnatomyGroup``, ``SegmentationLabelNames``,
        ``AnatomyColor``) cannot, since one mesh holds one value per key.

        Raises:
            ValueError: If *surfaces* is empty.
        """
        if not surfaces:
            raise ValueError("No surfaces to combine.")
        # Shallow copies so tagging does not add an array to the caller's
        # surfaces; the point/cell arrays themselves stay shared.
        tagged: list[pv.PolyData] = []
        for surface in surfaces.values():
            label_ids = surface.field_data.get("SegmentationLabelIds")
            label_id = (
                int(label_ids[0])
                if label_ids is not None and len(label_ids) == 1
                else 0
            )
            tagged_surface = surface.copy(deep=False)
            tagged_surface.cell_data["SegmentationLabelIds"] = np.full(
                tagged_surface.n_cells, label_id, dtype=np.int32
            )
            tagged.append(tagged_surface)
        return cast(pv.PolyData, pv.merge(tagged, merge_points=False))
