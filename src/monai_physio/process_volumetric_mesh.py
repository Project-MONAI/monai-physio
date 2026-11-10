"""
Tools for creating and relaxing tetrahedral (volumetric) meshes.

This module provides the :class:`ProcessVolumetricMesh` class: one tetrahedral
mesh per label of a labelmap (voxel-based, netgen, or Simpleware Medical),
their relaxation onto a smooth surface, and the repair of inverted elements.
Surfaces come from :class:`ProcessSurfaceMesh`; the GPU primitives live in
:class:`ProcessMeshKernels`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Optional, cast

import itk
import numpy as np
import pyvista as pv
from scipy import sparse

from .monai_physio_base import MONAIPhysioBase
from .process_images import ProcessImages
from .process_mesh_kernels import ProcessMeshKernels
from .process_surface_mesh import ProcessSurfaceMesh
from .simpleware_medical import DEFAULT_SIMPLEWARE_EXE_PATH, run_simpleware_script

# Run by ProcessVolumetricMesh._retile_and_fill_netgen in a child process:
# argv is the ASCII STL surface, the .npz to write, and the maximum element
# size ("" for netgen's default).  netgen numbers points from 1.
_NETGEN_STL_SCRIPT = """
import sys

import numpy as np
from netgen.stl import STLGeometry

stl_file, result_file, element_size = sys.argv[1:4]
options = {"maxh": float(element_size)} if element_size else {}
mesh = STLGeometry(stl_file).GenerateMesh(**options)
connectivity = (
    np.asarray(mesh.Elements3D().NumPy()["nodes"])[:, :4].astype(np.int64) - 1
    if mesh.ne
    else np.zeros((0, 4), dtype=np.int64)
)
np.savez(
    result_file,
    points=np.asarray(mesh.Coordinates(), dtype=np.float64),
    connectivity=connectivity,
)
"""


class ProcessVolumetricMesh(MONAIPhysioBase):
    """
    Tools for creating and relaxing tetrahedral meshes.

    Example:
        >>> surfaces = ProcessSurfaceMesh().extract_label_surfaces(labelmap)
        >>> volume_tools = ProcessVolumetricMesh()
        >>> meshes = volume_tools.extract_label_tetrahedra_vtk(labelmap, 1.5)
        >>> meshes = {
        ...     label_id: volume_tools.trim_tetrahedra_to_surface(
        ...         mesh, surfaces[label_id]
        ...     )
        ...     for label_id, mesh in meshes.items()
        ... }
        >>> meshes = volume_tools.extract_label_tetrahedra_netgen(labelmap, 1.5)
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
        self._image_tools = ProcessImages(log_level=log_level)
        self._kernels = ProcessMeshKernels(log_level=log_level)
        self._surface_tools = ProcessSurfaceMesh(log_level=log_level)

        self.simpleware_exe_path = DEFAULT_SIMPLEWARE_EXE_PATH
        self.simpleware_script_path = os.path.join(
            os.path.dirname(__file__),
            "simpleware_medical",
            "SimplewareScript_tetrahedral_mesh.py",
        )

    def set_simpleware_executable_path(self, path: str) -> None:
        """Set the path to the Simpleware Medical console executable.

        Args:
            path: Full path to ``ConsoleSimplewareMedical.exe``.
        """
        self.simpleware_exe_path = path

    def _select_labels(
        self, labelmap_image: itk.image, label_ids: Optional[Sequence[int]]
    ) -> tuple[itk.image, list[int]]:
        """Return *labelmap_image* holding only *label_ids*, and the labels left.

        ``None`` keeps every label.  The labels are the non-zero ids present,
        in ascending order.
        """
        import torch

        if label_ids is not None:
            all_labels = itk.GetArrayViewFromImage(labelmap_image)
            selected = itk.GetImageFromArray(
                np.where(np.isin(all_labels, label_ids), all_labels, 0)
            )
            selected.CopyInformation(labelmap_image)
            labelmap_image = selected
        present = torch.unique(
            torch.as_tensor(
                itk.GetArrayViewFromImage(labelmap_image).astype(np.int32),
                device=self._kernels.device,
            )
        ).tolist()
        return labelmap_image, [int(value) for value in present if value != 0]

    @staticmethod
    def _tag_label(mesh: pv.UnstructuredGrid, label_id: int) -> pv.UnstructuredGrid:
        """Record *label_id* in *mesh*'s ``field_data['SegmentationLabelIds']``."""
        mesh.field_data["SegmentationLabelIds"] = np.array([label_id], dtype=np.int32)
        return mesh

    def extract_label_tetrahedra_vtk(
        self,
        labelmap_image: itk.image,
        element_size_mm: Optional[float] = None,
        label_ids: Optional[Sequence[int]] = None,
    ) -> dict[int, pv.UnstructuredGrid]:
        """Build one tetrahedral mesh filling each label of a labelmap.

        Every labeled voxel becomes a hexahedron, which VTK then splits into
        six tetrahedra sharing the hexahedra's points, and each label keeps
        the tetrahedra of its own voxels.  The labels are meshed on one grid,
        so neighboring labels' meshes have coincident nodes along their shared
        walls (each mesh holds its own copy of them).  Each boundary is the
        voxel staircase rather than the smooth surface
        :meth:`ProcessSurfaceMesh.extract_label_surfaces` returns; pass each
        mesh through :meth:`trim_tetrahedra_to_surface` with its label's
        surface to relax that staircase onto it.  Relaxing moves each mesh on
        its own, so the shared nodes then match only as closely as the two
        surfaces do.

        Args:
            labelmap_image: Multi-label image; every non-zero label present is
                meshed.  A binary mask yields a single mesh.
            element_size_mm: Edge length of the isotropic voxels the labelmap
                is resampled to, with a label-aware vote, before meshing,
                which is the resulting element size.  ``None`` meshes the
                labelmap's own voxels, so on anisotropic data the elements
                inherit that anisotropy.  A size above the thinnest part of a
                structure drops that part.
            label_ids: Labels to mesh; every other label is treated as
                background.  ``None`` meshes every label.  To mesh a union of
                labels as one structure, relabel them to one id first.

        Returns:
            Label id → that label's tetrahedral mesh, whose
            ``field_data['SegmentationLabelIds']`` holds the id.  A label that
            *element_size_mm* is too coarse to keep is left out, so the mapping
            is empty when the labelmap holds no non-zero label.
        """
        labelmap_image, present = self._select_labels(labelmap_image, label_ids)
        if not present:
            self.log_warning("Labelmap holds no non-zero label; there are no meshes")
            return {}
        # Cropping only translates the image, so the direction is unchanged and
        # the winding correction below still applies after any resampling.
        cropped = self._image_tools.crop_to_nonzero(labelmap_image)
        spacing = np.asarray(labelmap_image.GetSpacing(), dtype=np.float64)
        direction = itk.array_from_matrix(labelmap_image.GetDirection())

        if element_size_mm is not None:
            # A vote over the labels rather than a nearest neighbor, and one
            # taken over at least an output voxel, so that coarsening keeps
            # thin walls instead of sampling through them.
            cropped = self._kernels.vote_resample_labelmap(
                cropped,
                element_size_mm,
                np.maximum(spacing, element_size_mm),
            )
            spacing = np.full(3, element_size_mm, dtype=np.float64)
        cropped_arr = itk.GetArrayViewFromImage(cropped)

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
        grid.cell_data["label"] = cropped_arr.ravel().astype(np.int32)
        voxels = grid.threshold(0.5, scalars="label")

        meshes: dict[int, pv.UnstructuredGrid] = {}
        if voxels.n_cells > 0:
            order = (
                self.VOXEL_TO_HEX
                if np.linalg.det(direction) > 0.0
                else self.VOXEL_TO_HEX_FLIPPED
            )
            connectivity = voxels.cells.reshape(-1, 9)[:, 1:][:, order]
            cells = np.hstack(
                [
                    np.full((len(connectivity), 1), 8, dtype=connectivity.dtype),
                    connectivity,
                ]
            )
            hexahedra = pv.UnstructuredGrid(
                cells.ravel(),
                np.full(len(connectivity), pv.CellType.HEXAHEDRON, dtype=np.uint8),
                voxels.points,
            )
            hexahedra.cell_data["label"] = voxels.cell_data["label"]
            tetrahedra = cast(pv.UnstructuredGrid, hexahedra.triangulate())
            tetrahedron_labels = np.asarray(tetrahedra.cell_data["label"])
            for label_id in present:
                # An array rather than a list: an empty list has no integer
                # dtype for extract_cells to recognize it by.
                cell_ids = cast(
                    Sequence[int], np.flatnonzero(tetrahedron_labels == label_id)
                )
                if len(cell_ids) == 0:
                    continue
                mesh = tetrahedra.extract_cells(cell_ids)
                # The bookkeeping arrays of the triangulation and the split.
                for array_name in ("label", "vtkOriginalCellIds"):
                    mesh.cell_data.pop(array_name, None)
                mesh.point_data.pop("vtkOriginalPointIds", None)
                meshes[label_id] = self._tag_label(mesh, label_id)
        for label_id in present:
            if label_id not in meshes:
                self.log_warning(
                    "Elements of %.3g mm are too coarse for label %d; it has no mesh",
                    element_size_mm,
                    label_id,
                )
        return meshes

    def _build_tetrahedra(
        self, points: np.ndarray, connectivity: np.ndarray
    ) -> pv.UnstructuredGrid:
        """Build a TETRA grid of the points *connectivity* uses.

        External meshers do not share VTK's corner order, and a left-handed
        image direction mirrors whatever order they used, so two corners of
        every inverted cell are swapped rather than the whole mesh flipped at
        once.  Points no cell uses are dropped.
        """
        used, connectivity = np.unique(
            np.asarray(connectivity, dtype=np.int64), return_inverse=True
        )
        connectivity = connectivity.reshape(-1, 4)
        points = points[used]
        corners = points[connectivity]
        inverted = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
        connectivity[inverted] = connectivity[inverted][:, [0, 1, 3, 2]]
        return pv.UnstructuredGrid({pv.CellType.TETRA: connectivity}, points)

    def extract_label_tetrahedra_netgen(
        self,
        labelmap_image: itk.image,
        element_size_mm: Optional[float] = None,
        smoothing_sigma_mm: Optional[float] = None,
        label_ids: Optional[Sequence[int]] = None,
        smoothing_iterations: int = 30,
        shared_interfaces: bool = True,
        contour_spacing_mm: Optional[float] = None,
    ) -> dict[int, pv.UnstructuredGrid]:
        """Build one unstructured tetrahedral mesh filling each label with netgen.

        The labels' smooth surfaces from
        :meth:`ProcessSurfaceMesh.extract_label_surfaces` are handed to netgen
        one at a time as the mesh boundary and netgen fills each with
        tetrahedra, keeping the boundary's triangles.  With
        *shared_interfaces* the surfaces are used as contoured: they are
        triangulated identically where two labels meet, so neighboring
        labels' meshes have coincident nodes along their shared walls (each
        mesh holds its own copy of them).  Without it, netgen first re-tiles
        each surface on its own into triangles about *element_size_mm*
        across, so the meshes are independent and may overlap or gap slightly
        where labels meet.  Unlike
        :meth:`extract_label_tetrahedra_vtk` the boundaries are already smooth,
        so no :meth:`trim_tetrahedra_to_surface` pass is needed.

        Requires the optional ``netgen`` extra (``netgen-mesher``).

        Args:
            labelmap_image: Multi-label image; every non-zero label present is
                meshed.  A binary mask yields a single mesh.
            element_size_mm: Target edge length.  It caps netgen's interior
                element size and, unless *contour_spacing_mm* is given, sets
                the isotropic grid the surfaces are contoured on, and so their
                triangle size.  The contour's smoothing
                grows with it, so coarse sizes shrink the meshes: on the
                1.5 mm heart test labels, 3 mm loses about 5% of the voxel
                volume where the labelmap's own spacing loses about 1%.
                ``None`` contours at the labelmap's finest spacing and lets
                netgen grade the interior freely from the surface.
            smoothing_sigma_mm: Blur of the labels' distance maps before their
                surfaces are contoured, passed to
                :meth:`ProcessSurfaceMesh.extract_label_surfaces` as
                ``distance_sigma_mm``.  ``None`` uses the contour grid spacing
                (*contour_spacing_mm*, else *element_size_mm*, else the
                labelmap's finest spacing), the
                same default as :meth:`extract_label_tetrahedra_simpleware`;
                ``0`` disables it, leaving only the surfaces' Taubin smoothing.
            label_ids: Labels to mesh; every other label is treated as
                background.  ``None`` meshes every label.
            smoothing_iterations: Taubin smoothing iterations applied to the
                surfaces, passed to
                :meth:`ProcessSurfaceMesh.extract_label_surfaces` (0
                disables).  Smoothing the labels' surfaces together can pinch
                a thin wall between two labels until a surface touches
                itself, which netgen rejects; 0 avoids that, leaving the blur
                alone to smooth the surfaces, at the cost of more irregular
                triangles.
            shared_interfaces: Keep the contoured surfaces, so neighboring
                labels' meshes share their interface nodes (default), or have
                netgen re-tile each label's surface on its own to triangles
                about *element_size_mm* across.
            contour_spacing_mm: Isotropic grid the surfaces are contoured on;
                ``None`` uses *element_size_mm*.  The surfaces carry steps of
                this size where the labelmap's voxels are coarser than the
                smoothing can round off, so a finer grid gives smaller steps.
                With *shared_interfaces* netgen keeps every contoured
                triangle, so halving the spacing roughly quadruples the
                boundary triangles and the mesh; without it netgen's re-tiling
                brings the triangles back to about *element_size_mm*.

        Returns:
            Label id → that label's tetrahedral mesh, whose
            ``field_data['SegmentationLabelIds']`` holds the id.  A label too
            thin to keep a surface is left out, so the mapping is empty when
            the labelmap holds no non-zero label.

        Raises:
            RuntimeError: If netgen produces no tetrahedra from a surface, or,
                without *shared_interfaces*, cannot re-tile one; a surface
                that intersects itself, as heavy smoothing can make of a thin
                wall, is the usual cause.
        """
        from netgen.meshing import FaceDescriptor, Mesh

        surfaces = self._surface_tools.extract_label_surfaces(
            labelmap_image,
            isotropic_spacing_mm=(
                element_size_mm if contour_spacing_mm is None else contour_spacing_mm
            ),
            distance_sigma_mm=smoothing_sigma_mm,
            smoothing_iterations=smoothing_iterations,
            label_ids=label_ids,
        )
        size_options = {} if element_size_mm is None else {"maxh": element_size_mm}
        meshes: dict[int, pv.UnstructuredGrid] = {}
        for label_id, surface in surfaces.items():
            if shared_interfaces:
                # The surface is closed and outward-oriented, so the domain
                # inside it is domain 1 and the outside is netgen's domain 0.
                mesh = Mesh(dim=3)
                mesh.AddPoints(np.ascontiguousarray(surface.points, dtype=np.float64))
                face = mesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))
                mesh.AddElements(
                    dim=2,
                    index=face,
                    data=np.ascontiguousarray(surface.regular_faces, dtype=np.int32),
                    base=0,
                )
                mesh.GenerateVolumeMesh(**size_options)
                # netgen numbers points from 1.
                connectivity = np.asarray(mesh.Elements3D().NumPy()["nodes"])[:, :4] - 1
                points = np.asarray(mesh.Coordinates(), dtype=np.float64)
            else:
                points, connectivity = self._retile_and_fill_netgen(
                    surface, element_size_mm, label_id
                )
            if len(connectivity) == 0:
                raise RuntimeError(
                    f"netgen produced no tetrahedra from label {label_id}'s surface "
                    f"({surface.n_points} points, {surface.n_cells} triangles)."
                )

            meshes[label_id] = self._tag_label(
                self._build_tetrahedra(points, connectivity), label_id
            )
        return meshes

    @staticmethod
    def _retile_and_fill_netgen(
        surface: pv.PolyData, element_size_mm: Optional[float], label_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Have netgen re-tile *surface* and fill it, in a child process.

        netgen's STL mesher ends the whole process, rather than raising, when
        the surface intersects itself, so it runs in a child Python process
        whose failure is reported as an exception here.

        Returns:
            The tetrahedra's points and zero-based connectivity.

        Raises:
            RuntimeError: If the child process fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            stl_file = Path(tmp) / "surface.stl"
            result_file = Path(tmp) / "tetrahedra.npz"
            # netgen's binary STL reader misreads VTK's files.
            pv.PolyData(surface.points, faces=surface.faces).save(
                stl_file, binary=False
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _NETGEN_STL_SCRIPT,
                    str(stl_file),
                    str(result_file),
                    "" if element_size_mm is None else str(element_size_mm),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 or not result_file.exists():
                tail = "\n".join(
                    (completed.stderr or completed.stdout).strip().splitlines()[-5:]
                )
                raise RuntimeError(
                    f"netgen failed to re-tile label {label_id}'s surface "
                    f"({surface.n_points} points, {surface.n_cells} triangles; "
                    f"exit code {completed.returncode}); a surface that "
                    f"intersects itself is the usual cause:\n{tail}"
                )
            with np.load(result_file) as result:
                return result["points"], result["connectivity"]

    def extract_label_tetrahedra_simpleware(
        self,
        labelmap_image: itk.image,
        element_size_mm: Optional[float] = None,
        smoothing_sigma_mm: Optional[float] = None,
        label_ids: Optional[Sequence[int]] = None,
        shared_interfaces: bool = True,
        timeout_s: int = 600,
    ) -> dict[int, pv.UnstructuredGrid]:
        """Build one unstructured tetrahedral mesh filling each label with Simpleware.

        Runs Synopsys Simpleware Medical's +FE Free mesher on the labelmap, as
        one external process (see :attr:`simpleware_exe_path`), and reads back
        the linear tetrahedra it exports.  Each label becomes a mask that is
        blurred inside Simpleware (its recursive Gaussian filter) before
        meshing: a mask carries no sub-voxel boundary position, so +FE Free
        would otherwise mesh the voxel staircase.  No
        :meth:`trim_tetrahedra_to_surface` pass is needed.

        Simpleware receives the labelmap with an identity direction and a zero
        origin, and its points are mapped back through the labelmap's own
        origin and direction, so the result lies in the labelmap's world frame
        whatever Simpleware's convention for reading image orientation.

        Requires a Simpleware Medical installation licensed for +FE Free.

        Args:
            labelmap_image: Multi-label image; every non-zero label present is
                meshed.  A binary mask yields a single mesh.
            element_size_mm: Target edge length: the maximum edge length, with
                half of it as the target minimum.  ``None`` keeps Simpleware's
                defaults.
            smoothing_sigma_mm: Standard deviation of the Gaussian blur applied
                to each label's mask before meshing.  ``None`` uses
                *element_size_mm*, or the labelmap's finest spacing when that
                is ``None`` too -- the same blur
                :meth:`extract_label_tetrahedra_netgen`'s surfaces get.  ``0``
                meshes the unblurred voxel staircase.  On the 1.5 mm Slicer
                heart mask at 2 mm elements, 1.5 mm cut the boundary roughness
                by a quarter and lost under 2% of the volume; 3 mm smoothed
                further but lost 4%, thinning small structures.
            label_ids: Labels to mesh; every other label is treated as
                background.  ``None`` meshes every label.
            shared_interfaces: ``True`` (default) meshes all labels as the
                parts of one +FE Free model, so neighboring labels' meshes
                have coincident nodes along their shared walls (each mesh
                holds its own copy of them); where blurred masks overlap,
                Simpleware's part priority decides which label keeps the
                voxel.  ``False`` meshes each label in a model of its own, so
                the meshes are independent and may overlap or gap slightly
                where labels meet.
            timeout_s: Seconds before the Simpleware process is killed.

        Returns:
            Label id → that label's tetrahedral mesh, whose
            ``field_data['SegmentationLabelIds']`` holds the id.  A label
            Simpleware exports no tetrahedra for is left out, so the mapping
            is empty when the labelmap holds no non-zero label.

        Raises:
            FileNotFoundError: If the Simpleware executable or script is
                missing.
            RuntimeError: If Simpleware fails, times out, or exports no mesh.
        """
        labelmap_image, present = self._select_labels(labelmap_image, label_ids)
        if not present:
            self.log_warning("Labelmap holds no non-zero label; there are no meshes")
            return {}
        finest_spacing = float(np.min(labelmap_image.GetSpacing()))
        if smoothing_sigma_mm is None:
            smoothing_sigma_mm = (
                finest_spacing if element_size_mm is None else element_size_mm
            )
        # Zero voxels around the crop keep the structures, and their blur, off
        # the image border, where Simpleware would leave their surfaces open.
        pad_voxels = 1 + int(np.ceil(3.0 * smoothing_sigma_mm / finest_spacing))
        cropped = self._image_tools.crop_to_nonzero(
            labelmap_image, pad_voxels=pad_voxels
        )
        # Simpleware sees the labels as parts 1..N, in ascending label order.
        cropped_labels = itk.GetArrayViewFromImage(cropped)
        parts = (np.searchsorted(present, cropped_labels) + 1).astype(np.uint16)
        parts[cropped_labels == 0] = 0
        staged = itk.GetImageFromArray(parts)
        staged.SetSpacing(cropped.GetSpacing())

        exported: dict[int, pv.UnstructuredGrid] = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = os.path.join(tmp_dir, "labelmap.mha")
            itk.imwrite(staged, input_file, compression=True)
            with open(
                os.path.join(tmp_dir, "params.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(
                    {
                        "element_size_mm": element_size_mm,
                        "smoothing_sigma_mm": smoothing_sigma_mm,
                        "part_count": len(present),
                        "shared_interfaces": shared_interfaces,
                    },
                    fh,
                )

            self.log_info(
                "Running Simpleware Medical +FE Free meshing of %d label(s)...",
                len(present),
            )
            stdout, stderr = run_simpleware_script(
                self.simpleware_exe_path,
                input_file,
                tmp_dir,
                self.simpleware_script_path,
                timeout_s=timeout_s,
            )
            if stdout:
                self.log_debug("Simpleware stdout:\n%s", stdout)
            if stderr:
                self.log_warning("Simpleware stderr:\n%s", stderr)

            # One file holding every part, tagged by its cell array PartId,
            # or one file per part.
            output_files = (
                ["tetrahedra.vtk"]
                if shared_interfaces
                else [f"tetrahedra_{part}.vtk" for part in range(1, len(present) + 1)]
            )
            for index, output_name in enumerate(output_files):
                output_file = os.path.join(tmp_dir, output_name)
                if not os.path.exists(output_file):
                    raise RuntimeError(
                        f"Simpleware Medical exported no mesh to {output_file}."
                    )
                exported[index + 1] = cast(pv.UnstructuredGrid, pv.read(output_file))

        direction = itk.array_from_matrix(cropped.GetDirection())
        meshes: dict[int, pv.UnstructuredGrid] = {}
        for part, label_id in enumerate(present, start=1):
            grid = exported[1 if shared_interfaces else part]
            if set(np.unique(grid.celltypes)) - {pv.CellType.TETRA}:
                self.log_warning(
                    "Simpleware Medical exported cell types %s; keeping only "
                    "the tetrahedra.",
                    sorted(int(t) for t in np.unique(grid.celltypes)),
                )
            grid = grid.extract_cells_by_type(pv.CellType.TETRA)
            connectivity = np.asarray(grid.cells.reshape(-1, 5)[:, 1:])
            if shared_interfaces:
                connectivity = connectivity[
                    np.asarray(grid.cell_data["PartId"]) == part
                ]
            if len(connectivity) == 0:
                self.log_warning(
                    "Simpleware Medical exported no tetrahedra for label %d; "
                    "it has no mesh",
                    label_id,
                )
                continue
            points = (
                np.asarray(cropped.GetOrigin(), dtype=np.float64)
                + np.asarray(grid.points, dtype=np.float64) @ direction.T
            )
            meshes[label_id] = self._tag_label(
                self._build_tetrahedra(points, connectivity), label_id
            )
        return meshes

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

        :meth:`extract_label_tetrahedra_vtk` meshes voxels, so each label's
        boundary is a staircase that both protrudes through the smooth surface
        :meth:`ProcessSurfaceMesh.extract_label_surfaces` builds for the same
        label and falls short of it elsewhere.  Cutting the mesh at the surface with ``clip_surface``
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

    def remove_boundary_slivers(
        self,
        tetrahedra: pv.UnstructuredGrid,
        min_scaled_jacobian: float = 0.15,
    ) -> pv.UnstructuredGrid:
        """Drop flat tetrahedra lying entirely on the mesh's boundary.

        :meth:`trim_tetrahedra_to_surface` presses boundary cells against the
        surface until they reach its quality bound, which leaves flat caps
        whose four corners all sit on the surface.  Each holds almost no
        volume, but warping the mesh -- through a registration, say -- carries
        its corners along nearly together, and a modest compression across
        the cap is enough to invert it.  :meth:`repair_inverted_tetrahedra`
        cannot help: averaging a boundary node with its neighbors pulls the
        cap flatter, not open.  Removing the caps before the warp avoids that.

        Removing a cap exposes the cell behind it, so this repeats until no
        cell below the bound has all four corners on the boundary.

        Args:
            tetrahedra: Volume mesh to clean; its point, cell and field data
                survive on the cells kept.
            min_scaled_jacobian: Cells whose four corners are all on the
                boundary are removed when their scaled Jacobian is below this.
                The default sits just above
                :meth:`trim_tetrahedra_to_surface`'s own 0.1 bound.

        Returns:
            The mesh without those cells, or *tetrahedra* unchanged if it had
            none.
        """
        import torch

        cleaned = tetrahedra
        removed = 0
        while cleaned.n_cells > 0:
            connectivity = cast(
                np.ndarray, cleaned.cells_dict[np.uint8(pv.CellType.TETRA)]
            )
            on_boundary = np.zeros(cleaned.n_points, dtype=bool)
            on_boundary[
                np.asarray(
                    cleaned.extract_surface(
                        pass_pointid=True, algorithm="dataset_surface"
                    ).point_data["vtkOriginalPointIds"]
                )
            ] = True
            quality = (
                self._kernels.scaled_jacobian(
                    torch.as_tensor(
                        np.asarray(cleaned.points),
                        dtype=torch.float64,
                        device=self._kernels.device,
                    ),
                    torch.as_tensor(
                        connectivity.astype(np.int64), device=self._kernels.device
                    ),
                )
                .cpu()
                .numpy()
            )
            slivers = on_boundary[connectivity].all(axis=1) & (
                quality < min_scaled_jacobian
            )
            if not slivers.any():
                break
            removed += int(slivers.sum())
            cleaned = cast(
                pv.UnstructuredGrid,
                cleaned.extract_cells(
                    ~slivers, pass_cell_ids=False, pass_point_ids=False
                ),
            )
        if removed:
            self.log_info(
                "Removed %d boundary sliver tetrahedra (scaled Jacobian < %g).",
                removed,
                min_scaled_jacobian,
            )
        return cleaned

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
