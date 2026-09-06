"""
Tools for creating and manipulating contours.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence, cast

import itk
import numpy as np
import pyacvd
import pyvista as pv
import trimesh

from .image_tools import ImageTools
from .monai_physio_base import MONAIPhysioBase
from .transform_tools import TransformTools
from .usd_anatomy_tools import USDAnatomyTools

# VTK_VOXEL lists its eight corners in (i, j, k) raster order; VTK_HEXAHEDRON
# wants the bottom quad wound consistently, then the matching top quad.
_VOXEL_TO_HEX = [0, 1, 3, 2, 4, 5, 7, 6]
# When the image direction matrix is left-handed (negative determinant), the
# above winding yields inverted (negative-Jacobian) hexahedra, so swap the
# bottom and top quads to restore a positive volume.
_VOXEL_TO_HEX_FLIPPED = [4, 5, 7, 6, 0, 1, 3, 2]

# trim_tetrahedra_to_surface halves a point's move until its cells clear the
# quality bound; below this fraction of the original move the point is put back
# where it started instead, which both ends the search and keeps the mesh free
# of moves too small to matter.
_MIN_TRIM_DAMPING = 2.0**-10

# Background voxels kept around the labels in extract_label_surfaces, enough
# for the isotropic grid to hold a full voxel of background on every side.
_LABEL_SURFACE_PAD = 3

# extract_contours resamples a labelmap whose coarsest spacing is more than
# this multiple of its finest one, because a contour built on the coarse axis
# terraces at its pitch.  Below it the resample costs more than it buys.
_CONTOUR_ANISOTROPY_LIMIT = 1.5


class ContourTools(MONAIPhysioBase):
    """
    Tools for creating and manipulating contours.
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize ContourTools.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        # USDAnatomyTools builds its color tables in __init__ without touching
        # the stage, so stage=None is safe for these lookup-only uses.
        self._anatomy_tools = USDAnatomyTools(stage=None, log_level=log_level)

    def apply_anatomy_color(
        self, mesh: pv.DataSet, anatomy_names: Sequence[str]
    ) -> None:
        """Attach a structure's :class:`USDAnatomyTools` color **in-place**.

        Sets, as :meth:`WorkflowConvertImageToVTK._annotate` does, so geometry
        from here colors the same way in Paraview, PyVista, and the USD
        exporter:

        - ``field_data['AnatomyColor']`` - RGB float32 color.
        - ``cell_data['Color']`` - RGBA uint8 solid color (n_cells x 4).

        Args:
            mesh: Surface or volume mesh to annotate.
            anatomy_names: Names tried in order, most specific first, e.g. an
                organ name followed by its anatomy group.  ``USDAnatomyTools``
                carries overrides for some organs (``myocardium``) but not
                others (``left_ventricle``), so a group name is the usual
                second entry.  Falls back to ``'other'`` when none resolves.
        """
        name = next(
            (
                candidate
                for candidate in anatomy_names
                if self._anatomy_tools.resolve_anatomy_type(candidate) is not None
            ),
            None,
        )
        if name is None:
            self.log_warning(
                "No anatomy color matches %s; using 'other'",
                " or ".join(anatomy_names),
            )
            name = "other"

        color_rgb = self._anatomy_tools.get_anatomy_diffuse_color(name)
        mesh.field_data["AnatomyColor"] = np.array(color_rgb, dtype=np.float32)
        rgba = np.array(
            [int(channel * 255) for channel in color_rgb] + [255], dtype=np.uint8
        )
        if mesh.n_cells > 0:
            mesh.cell_data["Color"] = np.tile(rgba, (mesh.n_cells, 1))

    def extract_contours(
        self,
        labelmap_image: itk.image,
        smoothing_iterations: int = 10,
        smoothing_scale: float = 1.0,
        surface_reduction_rate: float = 0.0,
        taubin_iterations: int = 20,
    ) -> pv.PolyData:
        """
        Make contours from a labelmap image.

        Every label boundary is emitted, including the internal boundaries
        between adjacent labels, so the result is a multi-material surface and
        is not watertight: an edge where three labels meet is shared by three
        faces.  Use :meth:`extract_watertight_surface` when a single label's
        closed surface is needed.

        Two passes keep the result off the voxel block edges the labelmap is
        drawn on.  An anisotropic labelmap is first resampled onto an isotropic
        grid of its finest pitch, the way :meth:`extract_label_surfaces` does,
        so that a boundary between two thick slices lands between them instead
        of terracing at one of them.  The contour is then Taubin-smoothed,
        which the surface net's own constrained smoothing cannot substitute
        for: that one may not move a point more than about a voxel, so it
        rounds the blocks without removing them.  Taubin does not shrink the
        surface, and the surface net shares its points between neighboring
        labels, so smoothing the mesh as one moves a shared point once and the
        labels stay in contact.

        Args:
            labelmap_image (itk.image): The labelmap image to create contours from
            smoothing_iterations: Surface-net smoothing iterations.
            smoothing_scale: Surface-net smoothing scale.
            surface_reduction_rate: Fraction of triangles to remove afterwards
                (0.0 disables).
            taubin_iterations: Taubin smoothing iterations applied after
                reduction, so they act on evenly sized triangles (0 disables).

        Returns:
            pv.PolyData: The contours as a PyVista PolyData object
        """
        spacing = np.asarray(labelmap_image.GetSpacing(), dtype=np.float64)
        if float(np.max(spacing) / np.min(spacing)) > _CONTOUR_ANISOTROPY_LIMIT:
            self.log_info(
                "Contouring on a %.3g mm isotropic grid rather than the "
                "labelmap's %s mm one",
                float(np.min(spacing)),
                " x ".join(f"{value:.3g}" for value in spacing),
            )
            labelmap_image = self._resample_labelmap_isotropic(
                labelmap_image, float(np.min(spacing))
            )

        labels = pv.wrap(itk.vtk_image_from_image(labelmap_image))
        contours = cast(
            pv.PolyData,
            labels.contour_labels(
                boundary_style="all",
                pad_background=False,
                smoothing=True,
                smoothing_iterations=smoothing_iterations,
                smoothing_scale=smoothing_scale,
                output_mesh_type="triangles",
            ),
        )

        return self.remesh_and_smooth_surface(
            contours, surface_reduction_rate, taubin_iterations
        )

    def extract_watertight_surface(
        self,
        mask_image: itk.image,
        smoothing_iterations: int = 10,
        gaussian_sigma_mm: float = 0.5,
        surface_reduction_rate: float = 0.0,
        anatomy_names: Optional[Sequence[str]] = None,
    ) -> pv.PolyData:
        """Extract one binary mask's closed, outward-oriented surface.

        :meth:`extract_contours` cannot produce a watertight surface: its
        surface nets pinch where a mask self-touches across a voxel diagonal,
        leaving edges shared by four faces, and they leave the mask open where
        it reaches the image border.  Isocontouring a continuous field cannot
        pinch, so this pads the mask with one voxel of background, blurs it, and
        runs marching cubes at the half-way isovalue instead.

        Reduction goes through :meth:`remesh_and_smooth_surface`, whose ACVD
        remeshing keeps a watertight input watertight where the VTK decimators
        do not.  That is a property of the remesher rather than a guarantee, so
        a reduced result is still checked and a warning logged if it degrades.

        Args:
            mask_image: Binary mask holding the single structure to extract.
            smoothing_iterations: Taubin smoothing iterations (0 disables).
            gaussian_sigma_mm: Blur applied before isocontouring, in
                millimeters, so it is independent of the voxel pitch.
            surface_reduction_rate: Fraction of triangles to remove (0.0
                disables).
            anatomy_names: Names passed to :meth:`apply_anatomy_color`, most
                specific first.  ``None`` leaves the surface uncolored.

        Returns:
            The structure's surface, with outward normals.
        """
        spacing = np.asarray(mask_image.GetSpacing(), dtype=np.float64)
        direction = itk.array_from_matrix(mask_image.GetDirection())
        # One voxel of background all around, so a structure reaching the image
        # border still closes; the origin steps back one voxel to match.
        padded_arr = np.pad(itk.GetArrayViewFromImage(mask_image).astype(np.float32), 1)
        padded = itk.GetImageFromArray(np.ascontiguousarray(padded_arr))
        padded.SetSpacing(mask_image.GetSpacing())
        padded.SetDirection(mask_image.GetDirection())
        padded.SetOrigin(
            np.asarray(mask_image.GetOrigin(), dtype=np.float64) + direction @ -spacing
        )

        blurred = itk.smoothing_recursive_gaussian_image_filter(
            padded, sigma=gaussian_sigma_mm
        )
        surface = cast(
            pv.PolyData,
            pv.wrap(itk.vtk_image_from_image(blurred)).contour(
                [0.5], method="flying_edges"
            ),
        )
        if smoothing_iterations > 0:
            surface = surface.smooth_taubin(n_iter=smoothing_iterations)
        # VTK winds faces for a right-handed direction matrix, so an LPS image
        # with a negative-determinant direction comes out with inward normals.
        surface = surface.compute_normals(
            auto_orient_normals=True, consistent_normals=True
        )

        if surface_reduction_rate > 0.0:
            surface = self.remesh_and_smooth_surface(surface, surface_reduction_rate, 0)
            if not self.is_watertight(surface):
                self.log_warning(
                    "Remeshing by %.2f made the surface non-watertight",
                    surface_reduction_rate,
                )

        if anatomy_names is not None:
            self.apply_anatomy_color(surface, anatomy_names)
        return surface

    def extract_label_surfaces(
        self,
        labelmap_image: itk.image,
        isotropic_spacing_mm: Optional[float] = None,
        distance_sigma_mm: Optional[float] = None,
        smoothing_iterations: int = 30,
    ) -> dict[int, pv.PolyData]:
        """Extract every label's surface, smooth and conforming with its neighbors.

        :meth:`extract_watertight_surface`, run per label, traces each label's
        own voxel block edges: on anisotropic data the result terraces at the
        slice pitch, and neighboring labels are contoured independently, so
        their shared wall is meshed twice and the two copies do not match.
        This extracts all labels together instead:

        1. The labelmap is resampled onto an isotropic grid with ITK's
           label-aware Gaussian interpolator, which votes over the labels in a
           physical-space kernel rather than picking a nearest voxel.  That is
           what removes the terracing: a boundary between two slices lands
           between them instead of on one of them.
        2. Every label, plus the background, gets a signed distance map on that
           grid, and each voxel is assigned to the label whose map is smallest.
           The assignment is a partition, so no gap or overlap can arise.
        3. Label ``L``'s surface is the zero level of ``D_L`` minus the smallest
           of the other maps.  On a wall between ``L`` and ``M`` that field is
           the negation of ``M``'s, so marching cubes puts identical vertices on
           both surfaces and the two meet exactly.
        4. The surfaces are merged, which welds those coincident vertices, then
           Taubin-smoothed as one mesh.  Smoothing therefore moves a shared
           vertex once and the surfaces stay in contact.

        Structures thinner than the interpolation kernel lose volume, coronary
        arteries most of all; the fraction of each label's voxel volume that
        the surface encloses is logged.

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

        Returns:
            Label id → that label's closed, outward-oriented surface.  A label
            too small to survive the isotropic grid is left out, so the mapping
            is empty when the labelmap holds no non-zero label and may be
            missing labels that it does hold.
        """
        labels = itk.GetArrayViewFromImage(labelmap_image)
        label_ids = [int(value) for value in np.unique(labels) if value != 0]
        if not label_ids:
            self.log_warning("Labelmap holds no non-zero label")
            return {}

        spacing = np.asarray(labelmap_image.GetSpacing(), dtype=np.float64)
        iso = (
            float(np.min(spacing))
            if isotropic_spacing_mm is None
            else isotropic_spacing_mm
        )
        sigma = iso if distance_sigma_mm is None else distance_sigma_mm
        cropped = self._crop_to_labels(labelmap_image, labels)
        fine = self._resample_labelmap_isotropic(cropped, iso)
        fine_labels = itk.GetArrayViewFromImage(fine)

        # Pass one: the closest and second closest label at every voxel.  Two
        # values are needed because a label's own map is the closest one inside
        # it, and its surface is measured against the next closest.
        closest = np.full(fine_labels.shape, np.inf, dtype=np.float32)
        runner_up = np.full(fine_labels.shape, np.inf, dtype=np.float32)
        closest_index = np.zeros(fine_labels.shape, dtype=np.int16)
        for index, label_id in enumerate(label_ids + [0]):
            distance = self._signed_distance_mm(fine, fine_labels, label_id, sigma)
            is_closer = distance < closest
            runner_up = np.where(is_closer, closest, np.minimum(runner_up, distance))
            closest = np.where(is_closer, distance, closest)
            closest_index[is_closer] = index

        # Pass two: one surface per label, tagged so the merged mesh can be
        # split again after smoothing.  The distance maps are recomputed rather
        # than kept, which costs one more pass but not one array per label.
        parts: list[pv.PolyData] = []
        for index, label_id in enumerate(label_ids):
            distance = self._signed_distance_mm(fine, fine_labels, label_id, sigma)
            others = np.where(closest_index == index, runner_up, closest)
            field = itk.GetImageFromArray(np.ascontiguousarray(distance - others))
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
        for label_id in label_ids:
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
            # The bookkeeping arrays of the merge and the split; the label is
            # the key of the returned mapping, so it is not data on the mesh.
            for array_name in ("LabelId", "vtkOriginalCellIds", "vtkOriginalPointIds"):
                surface.cell_data.pop(array_name, None)
                surface.point_data.pop(array_name, None)
            # VTK winds faces for a right-handed direction matrix, so an LPS
            # image with a negative-determinant direction comes out inward.
            surfaces[label_id] = surface.compute_normals(
                auto_orient_normals=True, consistent_normals=True
            )
            self.log_debug(
                "Label %d: %d triangles, %.3f of its voxel volume",
                label_id,
                surfaces[label_id].n_cells,
                float(surfaces[label_id].volume)
                / (int(np.count_nonzero(labels == label_id)) * voxel_volume),
            )
        return surfaces

    @staticmethod
    def _crop_to_labels(labelmap_image: itk.image, labels: np.ndarray) -> itk.image:
        """Return *labelmap_image* cropped to its labels and padded with background.

        The pad closes structures that reach the image border, which would
        otherwise contour to an open surface, and gives the distance maps room
        to fall away from the outermost label.
        """
        spacing = np.asarray(labelmap_image.GetSpacing(), dtype=np.float64)
        direction = itk.array_from_matrix(labelmap_image.GetDirection())
        # labels' axes are reversed relative to the ITK image, so the extents
        # come back as (z, y, x) and are flipped to (x, y, z) for the origin.
        extents = np.nonzero(labels)
        start = np.array([int(axis.min()) for axis in extents])
        stop = np.array([int(axis.max()) + 1 for axis in extents])
        cropped_arr = np.pad(
            labels[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]],
            _LABEL_SURFACE_PAD,
        )
        cropped = itk.GetImageFromArray(
            np.ascontiguousarray(cropped_arr.astype(np.uint16))
        )
        cropped.SetSpacing(labelmap_image.GetSpacing())
        cropped.SetDirection(labelmap_image.GetDirection())
        cropped.SetOrigin(
            np.asarray(labelmap_image.GetOrigin(), dtype=np.float64)
            + direction @ (spacing * (start[::-1] - _LABEL_SURFACE_PAD))
        )
        return cropped

    @staticmethod
    def _resample_labelmap_isotropic(
        labelmap_image: itk.image,
        isotropic_spacing_mm: float,
        sigma_mm: Optional[np.ndarray] = None,
    ) -> itk.image:
        """Resample a labelmap onto an isotropic grid, label boundaries intact.

        ``LabelImageGaussianInterpolateImageFunction`` gives each output voxel
        the label with the largest Gaussian-weighted vote among its neighbors,
        so labels stay whole numbers, keep sharing their walls, and their
        boundaries move to where the vote turns over rather than snapping to an
        input voxel.

        Args:
            labelmap_image: Labelmap to resample.
            isotropic_spacing_mm: Edge length of the output voxels.
            sigma_mm: Per-axis width of the voting kernel.  ``None`` uses one
                input voxel along each axis, which is what lets a coarse axis
                interpolate between its slices.
        """
        spacing = np.asarray(labelmap_image.GetSpacing(), dtype=np.float64)
        size = itk.size(labelmap_image)
        interpolator = itk.LabelImageGaussianInterpolateImageFunction.New(
            labelmap_image
        )
        sigma = spacing if sigma_mm is None else sigma_mm
        interpolator.SetSigma([float(value) for value in sigma])
        interpolator.SetAlpha(3.0)
        return itk.resample_image_filter(
            labelmap_image,
            size=[
                int((size[axis] - 1) * spacing[axis] / isotropic_spacing_mm) + 1
                for axis in range(3)
            ],
            output_spacing=[isotropic_spacing_mm] * 3,
            output_origin=list(labelmap_image.GetOrigin()),
            output_direction=labelmap_image.GetDirection(),
            interpolator=interpolator,
            default_pixel_value=0,
        )

    @staticmethod
    def _signed_distance_mm(
        reference: itk.image, labels: np.ndarray, label_id: int, sigma_mm: float
    ) -> np.ndarray:
        """Return the signed distance, in mm, to *label_id*, negative inside.

        The map measures to voxel centers, so its zero level is faceted at the
        voxel pitch; *sigma_mm* of blur takes those facets out, at the cost of
        pulling the level set in by roughly ``sigma_mm ** 2`` times the surface
        curvature.  ``label_id`` of ``0`` measures to the background, whose
        distance is the negated distance to the union of every label.
        """
        inside = labels != 0 if label_id == 0 else labels == label_id
        mask = itk.GetImageFromArray(np.ascontiguousarray(inside.astype(np.uint8)))
        mask.CopyInformation(reference)
        distance = itk.signed_maurer_distance_map_image_filter(
            mask,
            use_image_spacing=True,
            squared_distance=False,
            inside_is_positive=False,
        )
        if sigma_mm > 0.0:
            distance = itk.smoothing_recursive_gaussian_image_filter(
                distance, sigma=sigma_mm
            )
        distance_arr = np.asarray(itk.GetArrayFromImage(distance), dtype=np.float32)
        return -distance_arr if label_id == 0 else distance_arr

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

    def extract_tetrahedra(
        self,
        mask_image: itk.image,
        element_size_mm: Optional[float] = None,
        anatomy_names: Optional[Sequence[str]] = None,
    ) -> pv.UnstructuredGrid:
        """Build a tetrahedral mesh filling one binary mask.

        Every retained voxel becomes a hexahedron, which VTK then splits into
        six tetrahedra sharing the hexahedra's points, so the result is a
        conforming mesh whose boundary is the voxel staircase rather than the
        smooth surface :meth:`extract_label_surfaces` returns.  Pass the result
        through :meth:`trim_tetrahedra_to_surface` to relax that staircase onto
        the surface.

        Args:
            mask_image: Binary mask holding the single structure to fill.
            element_size_mm: Edge length of the isotropic voxels the mask is
                resampled to before meshing, which is the resulting element
                size.  ``None`` meshes the mask's own voxels, so on anisotropic
                data the elements inherit that anisotropy.  A size above the
                thinnest part of the structure drops that part.
            anatomy_names: Names passed to :meth:`apply_anatomy_color`, most
                specific first.  ``None`` leaves the mesh uncolored.

        Returns:
            The structure's tetrahedral mesh, empty if the mask is empty or
            *element_size_mm* is too coarse to keep any of it.
        """
        mask_arr = itk.GetArrayViewFromImage(mask_image) != 0
        if not mask_arr.any():
            self.log_warning("Mask holds no voxel to mesh; its mesh is empty")
            return pv.UnstructuredGrid()
        # mask_arr axes are reversed relative to the ITK image, so the per-axis
        # extents come back as (z, y, x) and are flipped to (x, y, z).
        starts, stops = [], []
        for axis_extent in np.nonzero(mask_arr):
            starts.append(int(axis_extent.min()))
            stops.append(int(axis_extent.max()) + 1)
        start_zyx, stop_zyx = np.array(starts), np.array(stops)
        cropped_arr = mask_arr[
            start_zyx[0] : stop_zyx[0],
            start_zyx[1] : stop_zyx[1],
            start_zyx[2] : stop_zyx[2],
        ]

        spacing = np.asarray(mask_image.GetSpacing(), dtype=np.float64)
        direction = itk.array_from_matrix(mask_image.GetDirection())
        # Cropping only translates the image, so the direction is unchanged and
        # the winding correction below still applies after any resampling.
        cropped = itk.GetImageFromArray(
            np.ascontiguousarray(cropped_arr.astype(np.uint16))
        )
        cropped.SetSpacing(mask_image.GetSpacing())
        cropped.SetDirection(mask_image.GetDirection())
        cropped.SetOrigin(
            np.asarray(mask_image.GetOrigin(), dtype=np.float64)
            + direction @ (spacing * start_zyx[::-1])
        )

        if element_size_mm is not None:
            # A vote over the mask rather than a nearest neighbor, and one
            # taken over at least an output voxel, so that coarsening keeps
            # thin walls instead of sampling through them.
            cropped = self._resample_labelmap_isotropic(
                cropped,
                element_size_mm,
                np.maximum(spacing, element_size_mm),
            )
            cropped_arr = itk.GetArrayViewFromImage(cropped) != 0
            spacing = np.full(3, element_size_mm, dtype=np.float64)
            if not cropped_arr.any():
                self.log_warning(
                    "Elements of %.3g mm are too coarse for this structure; "
                    "its mesh is empty",
                    element_size_mm,
                )
                return pv.UnstructuredGrid()

        # Corner-point grid: one more point than voxels along each axis, with
        # the origin backed off half a voxel to reach the first voxel's corner.
        grid = pv.ImageData(
            dimensions=tuple(int(n) + 1 for n in cropped_arr.shape[::-1]),
            spacing=tuple(spacing),
            origin=tuple(
                np.asarray(cropped.GetOrigin(), dtype=np.float64)
                + direction @ (spacing * -0.5)
            ),
        )
        grid.direction_matrix = direction
        grid.cell_data["label"] = cropped_arr.ravel().astype(np.uint8)
        voxels = grid.threshold(0.5, scalars="label")

        order = (
            _VOXEL_TO_HEX if np.linalg.det(direction) > 0.0 else _VOXEL_TO_HEX_FLIPPED
        )
        connectivity = voxels.cells.reshape(-1, 9)[:, 1:][:, order]
        cells = np.hstack(
            [np.full((len(connectivity), 1), 8, dtype=connectivity.dtype), connectivity]
        )
        hexahedra = pv.UnstructuredGrid(
            cells.ravel(),
            np.full(len(connectivity), pv.CellType.HEXAHEDRON, dtype=np.uint8),
            voxels.points,
        )
        tetrahedra = cast(pv.UnstructuredGrid, hexahedra.triangulate())

        if anatomy_names is not None:
            self.apply_anatomy_color(tetrahedra, anatomy_names)
        return tetrahedra

    def trim_tetrahedra_to_surface(
        self,
        tetrahedra: pv.UnstructuredGrid,
        surface: pv.PolyData,
        iterations: int = 5,
        relaxation: float = 0.6,
        min_scaled_jacobian: float = 0.1,
    ) -> pv.UnstructuredGrid:
        """Relax a tetrahedral mesh onto *surface*, keeping every cell whole.

        :meth:`extract_tetrahedra` meshes voxels, so its boundary is a
        staircase that both protrudes through the smooth surface
        :meth:`extract_label_surfaces` builds from the same mask and falls short
        of it elsewhere.  Cutting the mesh at the surface with ``clip_surface``
        would follow it exactly but shatters the boundary tetrahedra into
        slivers (about a tenth of the cells drop below a scaled Jacobian of
        0.1), and neither VTK nor any current dependency can repair those.

        So nothing is cut.  A crinkle clip drops the cells that lie entirely
        outside while leaving every surviving cell intact, then the mesh is
        relaxed: each sweep moves every point part of the way toward the
        average of its neighbors, and every boundary point instead toward its
        closest point on *surface*.  The interior smoothing is what makes room
        for the boundary to reach the surface -- projecting the boundary alone
        flattens the cells behind it, and the quality bound below then undoes
        the move, which is why one projection pass leaves the staircase in
        place.

        A move that would still wreck a cell is backed off: every point of a
        cell below *min_scaled_jacobian* has that sweep's step halved,
        repeatedly, until the whole mesh clears the bound.

        Args:
            tetrahedra: Volume mesh to relax; its cell and field data survive.
            surface: Closed surface to relax onto, in the same frame.
            iterations: Relaxation sweeps.  The boundary reaches the surface in
                the first few and then stops moving: on the Duke heart labels,
                sweeps beyond the default leave the mean boundary-to-surface
                distance and the worst cell quality where they already were,
                and only cost time.
            relaxation: Fraction of the way to its target a point moves per
                sweep.  ``1.0`` moves the whole way and oscillates.
            min_scaled_jacobian: Cell-quality bound every tetrahedron must meet
                after each sweep.  ``0.0`` only rules out flattened and
                inverted cells; the default also rules out slivers.

        Returns:
            The relaxed mesh.
        """
        # crinkle keeps whole cells, so this only discards, never subdivides.
        relaxed = cast(
            pv.UnstructuredGrid,
            tetrahedra.clip_surface(surface, invert=True, crinkle=True),
        )
        if relaxed.n_cells == 0:
            self.log_warning("Trimming against the surface removed every cell")
            return relaxed

        connectivity = relaxed.cells_dict[np.uint8(pv.CellType.TETRA)]
        # Both directions of every tetrahedron edge, so a point's neighbors are
        # the second column of the rows its id occupies in the first.
        edges = np.vstack(
            [
                connectivity[:, pair]
                for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ]
        )
        starts = np.concatenate([edges[:, 0], edges[:, 1]])
        ends = np.concatenate([edges[:, 1], edges[:, 0]])
        point_count = relaxed.n_points
        neighbor_counts = np.bincount(starts, minlength=point_count).clip(1)
        boundary_ids = np.asarray(
            relaxed.extract_surface(algorithm="dataset_surface").point_data[
                "vtkOriginalPointIds"
            ]
        )

        points = np.array(relaxed.points)
        for _ in range(iterations):
            target = (
                np.column_stack(
                    [
                        np.bincount(
                            starts, weights=points[ends, axis], minlength=point_count
                        )
                        for axis in range(3)
                    ]
                )
                / neighbor_counts[:, np.newaxis]
            )
            _, closest = cast(
                "tuple[np.ndarray, np.ndarray]",
                surface.find_closest_cell(
                    target[boundary_ids], return_closest_point=True
                ),
            )
            target[boundary_ids] = closest
            step = relaxation * (target - points)

            # Backing one point off can push a neighboring cell below the bound
            # that the full move had cleared, so this repeats until the whole
            # mesh passes.  The step bottoms out at zero, so the loop
            # terminates even when the mesh holds cells below the bound that no
            # amount of backing off can rescue.
            damping = np.ones(point_count)
            while True:
                relaxed.points = points + damping[:, np.newaxis] * step
                quality = np.asarray(
                    relaxed.cell_quality(["scaled_jacobian"]).cell_data[
                        "scaled_jacobian"
                    ]
                )
                below_bound = quality < min_scaled_jacobian
                if not np.any(below_bound):
                    break
                damped_ids = np.unique(connectivity[below_bound])
                damped_ids = damped_ids[damping[damped_ids] > 0.0]
                if damped_ids.size == 0:
                    break
                halved = damping[damped_ids] * 0.5
                damping[damped_ids] = np.where(halved < _MIN_TRIM_DAMPING, 0.0, halved)
            points = np.array(relaxed.points)
        return relaxed

    def repair_inverted_tetrahedra(
        self,
        tetrahedra: pv.UnstructuredGrid,
        max_iterations: int = 20,
    ) -> pv.UnstructuredGrid:
        """Relax the nodes of any inverted or degenerate tetrahedron.

        :meth:`trim_tetrahedra_to_surface` holds every cell of the *template*
        above a scaled Jacobian of 0.1, but warping that template onto a
        specific subject -- a statistical-model fit, say -- carries no such
        constraint, and can flip a handful of elements even though the
        template it started from was clean. Only the nodes touching a bad
        element are moved, each to the mean of its mesh neighbors, which
        leaves geometry the fit got right alone.

        Args:
            tetrahedra: Volume mesh to repair; its cell and field data
                survive.
            max_iterations: Smoothing passes to attempt before giving up.

        Returns:
            The repaired mesh, or *tetrahedra* unchanged if every cell
            already clears zero volume.

        Raises:
            ValueError: If *tetrahedra* has no TETRA cells, or if elements
                are still inverted or degenerate after *max_iterations*
                passes.
        """
        if np.uint8(pv.CellType.TETRA) not in tetrahedra.cells_dict:
            raise ValueError(
                "tetrahedra has no TETRA cells to repair; got cell types "
                f"{sorted(tetrahedra.cells_dict)}."
            )
        connectivity = tetrahedra.cells_dict[np.uint8(pv.CellType.TETRA)]

        def volumes(points: np.ndarray) -> np.ndarray:
            corners = points[connectivity]
            edges = corners[:, 1:, :] - corners[:, 0:1, :]
            return cast(np.ndarray, np.linalg.det(edges) / 6.0)

        points = np.array(tetrahedra.points)
        n_inverted = int(np.sum(volumes(points) <= 0.0))
        if n_inverted == 0:
            return tetrahedra

        edges = np.vstack(
            [
                connectivity[:, pair]
                for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ]
        )
        starts = np.concatenate([edges[:, 0], edges[:, 1]])
        ends = np.concatenate([edges[:, 1], edges[:, 0]])
        order = np.argsort(starts, kind="stable")
        sorted_starts = starts[order]
        sorted_ends = ends[order]
        split_points = np.searchsorted(sorted_starts, np.arange(len(points) + 1))
        neighbors = {
            node: np.unique(sorted_ends[split_points[node] : split_points[node + 1]])
            for node in np.unique(connectivity)
        }

        for _ in range(max_iterations):
            bad = volumes(points) <= 0.0
            if not np.any(bad):
                break
            snapshot = points.copy()
            for node in np.unique(connectivity[bad]):
                neighbor_points = snapshot[neighbors[node]]
                if len(neighbor_points):
                    points[node] = neighbor_points.mean(axis=0)

        still_bad = int(np.sum(volumes(points) <= 0.0))
        if still_bad:
            raise ValueError(
                f"{still_bad} of {len(connectivity)} tetrahedra are still "
                f"inverted or degenerate after {max_iterations} repair "
                "passes; the fitted mesh needs a real re-fit, not just "
                "smoothing."
            )

        self.log_warning(
            "%d of %d tetrahedra were inverted or degenerate; repaired by "
            "relaxing their nodes onto their neighbors' mean.",
            n_inverted,
            len(connectivity),
        )
        repaired = tetrahedra.copy()
        repaired.points = points
        return repaired

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
        ``decimate_pro`` did: per-cell ``boundary_labels`` (needed for anatomy
        splitting downstream) and ``SegmentationLabelIds`` (which structure each
        triangle belongs to) are transferred back onto the new cells from their
        nearest original cell so anatomy materials and structure ids still apply.  Uniform
        triangles cannot represent a label patch smaller than one of them,
        though, so such a patch is absorbed by its neighbours and its label pair
        disappears -- a warning names the pairs lost.  ``decimate_pro`` kept
        those patches by being non-uniform, which is the trade being made here.
        Smoothing uses non-shrinking Taubin smoothing, which only moves points
        and therefore preserves cells and their labels.  It is told to move
        non-manifold points too, since on a multi-material surface every edge
        where three labels meet is non-manifold and VTK pins those points
        otherwise; on a manifold surface the setting has nothing to act on.

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
            clustering = pyacvd.Clustering(conditioned.triangulate())
            # One cluster per retained point.  A closed surface carries about
            # twice as many triangles as points, so scaling the point count by
            # (1 - rate) scales the triangle count by the same fraction; four
            # is the fewest clusters that can still close a surface.
            clustering.cluster(
                max(4, round(original.n_points * (1.0 - surface_reduction_rate)))
            )
            conditioned = clustering.create_mesh()
            carried = [
                name
                for name in ("boundary_labels", "SegmentationLabelIds")
                if name in original.cell_data
            ]
            if carried:
                nearest = original.find_closest_cell(conditioned.cell_centers().points)
                for name in carried:
                    conditioned.cell_data[name] = np.asarray(original.cell_data[name])[
                        nearest
                    ]

            if "boundary_labels" in original.cell_data:
                labels = np.asarray(original.cell_data["boundary_labels"])
                pairs = labels.reshape(len(labels), -1)
                before = {tuple(row) for row in np.unique(pairs, axis=0).tolist()}
                after = {
                    tuple(row) for row in np.unique(pairs[nearest], axis=0).tolist()
                }
                if before - after:
                    self.log_warning(
                        "Remeshing by %.2f dropped %d of %d boundary label pairs, "
                        "each covering less than one output triangle: %s",
                        surface_reduction_rate,
                        len(before - after),
                        len(before),
                        sorted(before - after),
                    )
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
        new_contours = TransformTools().transform_pvcontour(
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

        image_tools = ImageTools()
        deformation_field_img = image_tools.convert_array_to_image_of_vectors(
            deformation_field, reference_image, ptype=ptype
        )

        return deformation_field_img

    # ─────────────────────────── I/O helpers ───────────────────────────────

    @staticmethod
    def save_surfaces(
        surfaces: dict[str, pv.PolyData],
        output_dir: str,
        prefix: str = "",
    ) -> dict[str, str]:
        """Save each named surface to its own VTP file.

        Args:
            surfaces: Mapping of name → surface (e.g. the ``'surfaces'``
                value from :meth:`WorkflowConvertImageToVTK.process`).
            output_dir: Directory to write files into (created if absent).
            prefix: Optional filename prefix.  Each file is named
                ``{prefix}_{name}.vtp`` (or ``{name}.vtp`` when *prefix* is empty).

        Returns:
            Mapping of name → absolute path of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved: dict[str, str] = {}
        for name, surface in surfaces.items():
            stem = f"{prefix}_{name}" if prefix else name
            path = os.path.join(output_dir, f"{stem}.vtp")
            surface.save(path)
            saved[name] = path
        return saved

    @staticmethod
    def save_combined_surfaces(
        surfaces: dict[str, pv.PolyData],
        output_filename: str,
    ) -> str:
        """Merge all named surfaces into a single VTP file.

        The merged mesh retains per-cell ``Color`` (RGBA uint8) from each
        surface's annotation, enabling colour-by-anatomy rendering in
        Paraview, PyVista, etc.

        It also gains a per-cell ``SegmentationLabelIds`` (int32) array, which
        carries each cell's originating label ID so structure identity survives
        the merge.  Downstream, :class:`ConvertVTKToUSD` splits on this array
        when given ``mask_ids``, giving one prim (and one anatomy material) per
        structure.  A surface whose ``field_data['SegmentationLabelIds']`` does
        not hold exactly one ID has no per-cell attribution - that is the case
        for the per-group surfaces of :class:`WorkflowConvertImageToVTK`, which
        are contoured from a merged binary mask - so its cells are tagged ``0``.
        Pass the per-label surfaces (``'label_surfaces'``) to get real IDs.

        Per-object ``field_data`` is *not* preserved: it is per-object, so a
        single merged mesh cannot carry one value per input surface.  The
        remaining keys set by :meth:`WorkflowConvertImageToVTK._annotate` are
        therefore lost:

        - ``AnatomyGroup`` - group name, e.g. ``'heart'``.
        - ``SegmentationLabelNames`` - structure names within the group.
        - ``AnatomyColor`` - RGB float color (survives indirectly as the
          per-cell ``Color`` array).

        Use :meth:`save_surfaces` instead when structure *names* must be
        recoverable from the saved files.

        Args:
            surfaces: Mapping of name → surface.
            output_filename: Path of the VTP file to write, including its
                directory.  Any missing parent directories are created.

        Returns:
            Path to the saved VTP file.

        Raises:
            ValueError: If *surfaces* is empty.
        """
        if not surfaces:
            raise ValueError("No surfaces to save.")
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Shallow copies so tagging does not add an array to the caller's
        # surfaces; the point/cell arrays themselves stay shared.
        tagged: list[pv.PolyData] = []
        for surface in surfaces.values():
            label_ids = surface.field_data.get("SegmentationLabelIds")
            if label_ids is not None and len(label_ids) == 1:
                label_id = int(label_ids[0])
            else:
                label_id = 0
            tagged_surface = surface.copy(deep=False)
            tagged_surface.cell_data["SegmentationLabelIds"] = np.full(
                tagged_surface.n_cells, label_id, dtype=np.int32
            )
            tagged.append(tagged_surface)
        merged = cast(pv.PolyData, pv.merge(tagged, merge_points=False))
        merged.save(output_filename)
        return output_filename
