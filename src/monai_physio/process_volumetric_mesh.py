"""
Tools for creating and relaxing tetrahedral (volumetric) meshes.

This module provides the :class:`ProcessVolumetricMesh` class: tetrahedral
meshes filling a binary mask, their relaxation onto a smooth surface, and the
repair of inverted elements.  Surfaces come from :class:`ProcessSurfaceMesh`;
the GPU primitives live in :class:`ProcessMeshKernels`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import ClassVar, Optional, cast

import itk
import numpy as np
import pyvista as pv
from scipy import sparse

from .monai_physio_base import MONAIPhysioBase
from .process_images import ProcessImages
from .process_mesh_kernels import ProcessMeshKernels
from .process_usd_anatomy import ProcessUSDAnatomy


class ProcessVolumetricMesh(MONAIPhysioBase):
    """
    Tools for creating and relaxing tetrahedral meshes.

    Example:
        >>> surfaces = ProcessSurfaceMesh().extract_label_surfaces(mask)
        >>> volume_tools = ProcessVolumetricMesh()
        >>> mesh = volume_tools.extract_tetrahedra(mask, element_size_mm=1.5)
        >>> mesh = volume_tools.trim_tetrahedra_to_surface(mesh, surfaces[1])
    """

    # VTK_VOXEL lists its eight corners in (i, j, k) raster order; VTK_HEXAHEDRON
    # wants the bottom quad wound consistently, then the matching top quad.
    VOXEL_TO_HEX: ClassVar[list[int]] = [0, 1, 3, 2, 4, 5, 7, 6]
    # When the image direction matrix is left-handed (negative determinant), the
    # above winding yields inverted (negative-Jacobian) hexahedra, so swap the
    # bottom and top quads to restore a positive volume.
    VOXEL_TO_HEX_FLIPPED: ClassVar[list[int]] = [4, 5, 7, 6, 0, 1, 3, 2]

    # trim_tetrahedra_to_surface halves a point's move until its cells clear the
    # quality bound; below this fraction of the original move the point is put
    # back where it started instead, which both ends the search and keeps the
    # mesh free of moves too small to matter.
    MIN_TRIM_DAMPING = 2.0**-10

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize ProcessVolumetricMesh.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        # ProcessUSDAnatomy builds its color tables in __init__ without touching
        # the stage, so stage=None is safe for this lookup-only use.
        self._anatomy_tools = ProcessUSDAnatomy(stage=None, log_level=log_level)
        self._image_tools = ProcessImages(log_level=log_level)
        self._kernels = ProcessMeshKernels(log_level=log_level)

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
        smooth surface :meth:`ProcessSurfaceMesh.extract_label_surfaces` returns.  Pass the result
        through :meth:`trim_tetrahedra_to_surface` to relax that staircase onto
        the surface.

        Args:
            mask_image: Binary mask holding the single structure to fill.
            element_size_mm: Edge length of the isotropic voxels the mask is
                resampled to before meshing, which is the resulting element
                size.  ``None`` meshes the mask's own voxels, so on anisotropic
                data the elements inherit that anisotropy.  A size above the
                thinnest part of the structure drops that part.
            anatomy_names: Names passed to
                :meth:`ProcessUSDAnatomy.apply_anatomy_color`, most specific
                first.  ``None`` leaves the mesh uncolored.

        Returns:
            The structure's tetrahedral mesh, empty if the mask is empty or
            *element_size_mm* is too coarse to keep any of it.
        """
        if not (itk.GetArrayViewFromImage(mask_image) != 0).any():
            self.log_warning("Mask holds no voxel to mesh; its mesh is empty")
            return pv.UnstructuredGrid()
        # Cropping only translates the image, so the direction is unchanged and
        # the winding correction below still applies after any resampling.
        cropped = self._image_tools.crop_to_nonzero(mask_image)
        cropped_arr = itk.GetArrayViewFromImage(cropped) != 0
        spacing = np.asarray(mask_image.GetSpacing(), dtype=np.float64)
        direction = itk.array_from_matrix(mask_image.GetDirection())

        if element_size_mm is not None:
            # A vote over the mask rather than a nearest neighbor, and one
            # taken over at least an output voxel, so that coarsening keeps
            # thin walls instead of sampling through them.
            cropped = self._kernels.vote_resample_labelmap(
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
            self.VOXEL_TO_HEX
            if np.linalg.det(direction) > 0.0
            else self.VOXEL_TO_HEX_FLIPPED
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
            self._anatomy_tools.apply_anatomy_color(tetrahedra, anatomy_names)
        return tetrahedra

    @staticmethod
    def _edge_endpoints(connectivity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return both directions of every tetrahedron edge as (starts, ends).

        A point's neighbors are then the ends of the rows its id starts.
        """
        edges = np.vstack(
            [
                connectivity[:, pair]
                for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ]
        )
        return (
            np.concatenate([edges[:, 0], edges[:, 1]]),
            np.concatenate([edges[:, 1], edges[:, 0]]),
        )

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
        :meth:`ProcessSurfaceMesh.extract_label_surfaces` builds from the same mask and falls short
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
        surface_mesh = self._kernels.build_surface_mesh(surface)
        # A crinkle clip keeps every cell with a point inside the surface, whole,
        # so this only discards, never subdivides.
        inside = self._kernels.points_inside_surface(
            surface_mesh, np.asarray(tetrahedra.points)
        )
        kept = inside[tetrahedra.cells.reshape(-1, 5)[:, 1:]].any(axis=1)
        crinkled = tetrahedra.copy(deep=False)
        scalars = crinkled.active_scalars_info
        crinkled.cell_data["cell_ids"] = np.arange(tetrahedra.n_cells, dtype=np.int64)
        relaxed = cast(
            pv.UnstructuredGrid,
            crinkled.extract_cells(kept, pass_cell_ids=False, pass_point_ids=False),
        )
        if relaxed.n_cells == 0:
            self.log_warning("Trimming against the surface removed every cell")
            return relaxed
        relaxed.set_active_scalars(
            scalars.name,
            preference=scalars.association,  # type: ignore[arg-type]
        )

        connectivity = cast(np.ndarray, relaxed.cells_dict[np.uint8(pv.CellType.TETRA)])
        starts, ends = self._edge_endpoints(connectivity)
        point_count = relaxed.n_points
        neighbor_counts = np.bincount(starts, minlength=point_count).clip(1)
        neighbor_sums = sparse.csr_matrix(
            (np.ones(len(starts)), (starts, ends)), shape=(point_count, point_count)
        )
        boundary_ids = np.asarray(
            relaxed.extract_surface(algorithm="dataset_surface").point_data[
                "vtkOriginalPointIds"
            ]
        )

        import torch

        gpu_connectivity = torch.as_tensor(
            connectivity.astype(np.int64), device=self._kernels.device
        )
        points = np.array(relaxed.points)
        for _ in range(iterations):
            target = (neighbor_sums @ points) / neighbor_counts[:, np.newaxis]
            target[boundary_ids] = self._kernels.closest_points_on_surface(
                surface_mesh, target[boundary_ids]
            )
            step = torch.as_tensor(
                relaxation * (target - points), device=self._kernels.device
            )
            gpu_points = torch.as_tensor(
                points, dtype=torch.float64, device=self._kernels.device
            )

            # Backing one point off can push a neighboring cell below the bound
            # that the full move had cleared, so this repeats until the whole
            # mesh passes.  The step bottoms out at zero, so the loop
            # terminates even when the mesh holds cells below the bound that no
            # amount of backing off can rescue.
            damping = torch.ones(
                point_count, dtype=torch.float64, device=self._kernels.device
            )
            while True:
                moved = gpu_points + damping[:, None] * step
                below_bound = (
                    self._kernels.scaled_jacobian(moved, gpu_connectivity)
                    < min_scaled_jacobian
                )
                if not bool(below_bound.any()):
                    break
                damped_ids = torch.unique(gpu_connectivity[below_bound])
                damped_ids = damped_ids[damping[damped_ids] > 0.0]
                if damped_ids.numel() == 0:
                    break
                halved = damping[damped_ids] * 0.5
                damping[damped_ids] = torch.where(
                    halved < self.MIN_TRIM_DAMPING, torch.zeros_like(halved), halved
                )
            points = moved.cpu().numpy()
        if iterations > 0:
            relaxed.points = points
        return relaxed

    def repair_inverted_tetrahedra(
        self,
        tetrahedra: pv.UnstructuredGrid,
        max_iterations: int = 100,
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
        connectivity = cast(
            np.ndarray, tetrahedra.cells_dict[np.uint8(pv.CellType.TETRA)]
        )

        def volumes(points: np.ndarray) -> np.ndarray:
            corners = points[connectivity]
            edges = corners[:, 1:, :] - corners[:, 0:1, :]
            return cast(np.ndarray, np.linalg.det(edges) / 6.0)

        points = np.array(tetrahedra.points)
        n_inverted = int(np.sum(volumes(points) <= 0.0))
        if n_inverted == 0:
            return tetrahedra

        starts, ends = self._edge_endpoints(connectivity)
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

        final_volumes = volumes(points)
        still_bad_mask = final_volumes <= 0.0
        still_bad = int(np.sum(still_bad_mask))
        if still_bad:
            _MAX_LISTED = 20
            bad_cell_ids = np.nonzero(still_bad_mask)[0]
            details = "; ".join(
                f"cell {cell_id} (nodes {connectivity[cell_id].tolist()}): "
                f"volume={final_volumes[cell_id]:.3e}"
                for cell_id in bad_cell_ids[:_MAX_LISTED]
            )
            if len(bad_cell_ids) > _MAX_LISTED:
                details += f"; ... and {len(bad_cell_ids) - _MAX_LISTED} more"
            raise ValueError(
                f"{still_bad} of {len(connectivity)} tetrahedra are still "
                f"inverted or degenerate after {max_iterations} repair "
                f"passes ({details}); the fitted mesh needs a real re-fit, "
                "not just smoothing."
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
