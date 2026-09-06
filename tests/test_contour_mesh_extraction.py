"""Tests for ContourTools' per-label surface and tetrahedral mesh extraction.

These use ``data/test/slicer_heart_small``, whose direction matrix is
right-handed, plus a synthetic left-handed image for the cases that only a
negative-determinant direction can exercise.
"""

from __future__ import annotations

from pathlib import Path

import itk
import numpy as np
import pytest
import pyvista as pv

from monai_physio.contour_tools import ContourTools

#: A label that reaches the volume border, so its surface only closes if the
#: mask is padded first.
BORDER_LABEL = 5


@pytest.fixture(scope="module")
def heart_labelmap() -> itk.Image:
    """Read one small labelmap."""
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "test"
        / "slicer_heart_small"
        / "slice_000_labelmap.mha"
    )
    if not path.exists():
        pytest.skip(f"Test labelmap not available: {path}")
    return itk.imread(str(path))


def _label_mask(labelmap: itk.Image, label_id: int) -> itk.Image:
    """Isolate one label of *labelmap* as a binary mask."""
    arr = itk.GetArrayViewFromImage(labelmap)
    mask = itk.GetImageFromArray((arr == label_id).astype(np.uint8))
    mask.CopyInformation(labelmap)
    return mask


def _left_handed_box() -> itk.Image:
    """Build a solid box in an image whose direction determinant is negative."""
    arr = np.zeros((6, 6, 6), dtype=np.uint8)
    arr[1:5, 1:5, 1:5] = 1
    mask = itk.GetImageFromArray(arr)
    mask.SetSpacing([1.0, 2.0, 3.0])
    mask.SetDirection(itk.matrix_from_array(np.diag([1.0, 1.0, -1.0])))
    return mask


def _touching_boxes() -> itk.Image:
    """Build two boxes sharing a wall, sampled coarsely along the slice axis.

    The 4 mm slice pitch against the 1 mm in-plane pitch is what makes an
    independently contoured surface terrace, and the negative-determinant
    direction is the LPS case that inverts VTK's face winding.
    """
    arr = np.zeros((14, 48, 60), dtype=np.uint16)
    arr[3:11, 8:40, 6:30] = 1
    arr[3:11, 8:40, 30:54] = 2
    labelmap = itk.GetImageFromArray(arr)
    labelmap.SetSpacing([1.0, 1.0, 4.0])
    labelmap.SetDirection(itk.matrix_from_array(np.diag([1.0, -1.0, 1.0])))
    return labelmap


def _point_keys(surface: pv.PolyData) -> set[tuple[float, ...]]:
    """Return *surface*'s points as hashable, rounded coordinate triples."""
    return {tuple(point) for point in np.round(np.asarray(surface.points), 6).tolist()}


def _roughness(surface: pv.PolyData) -> float:
    """Return how far a point sits from its neighbors' centroid, per edge length.

    Dividing by the mean edge length makes the measure comparable between
    surfaces triangulated at different resolutions.
    """
    mesh = surface.triangulate()
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    points = np.asarray(mesh.points)
    total = np.zeros_like(points)
    count = np.zeros(len(points))
    for first, second in ((0, 1), (1, 2), (2, 0)):
        for source, target in ((first, second), (second, first)):
            np.add.at(total, faces[:, source], points[faces[:, target]])
            np.add.at(count, faces[:, source], 1)
    offset = total / np.maximum(count, 1)[:, np.newaxis] - points
    edge_length = np.linalg.norm(points[faces[:, 0]] - points[faces[:, 1]], axis=1)
    return float(np.linalg.norm(offset, axis=1).mean() / edge_length.mean())


def _label_pairs(contours: pv.PolyData) -> set[tuple[int, ...]]:
    """Return the set of label pairs the contour's cells separate."""
    pairs = np.asarray(contours.cell_data["boundary_labels"])
    return {tuple(row) for row in np.unique(pairs.reshape(len(pairs), -1), axis=0)}


class TestExtractContours:
    """The multi-label contour must be smooth and keep its labels in contact."""

    def test_smoothing_takes_the_voxel_blocks_out(
        self, contour_tools: ContourTools
    ) -> None:
        """Taubin smoothing leaves the surface markedly less faceted.

        These boxes are flat over most of their area, where there is nothing to
        take out, so the margin here is smaller than on anatomy: the same
        measure halves on a segmented chest CT.
        """
        blocky = contour_tools.extract_contours(_touching_boxes(), taubin_iterations=0)
        smooth = contour_tools.extract_contours(_touching_boxes(), taubin_iterations=20)

        assert _roughness(smooth) < 0.75 * _roughness(blocky)

    def test_smoothing_keeps_every_cell_and_its_labels(
        self, contour_tools: ContourTools
    ) -> None:
        """Smoothing only moves points, so the label of each cell survives.

        Downstream splits a contour into one prim per structure on
        ``boundary_labels``, so a lost pair is a lost structure.
        """
        blocky = contour_tools.extract_contours(_touching_boxes(), taubin_iterations=0)
        smooth = contour_tools.extract_contours(_touching_boxes(), taubin_iterations=20)

        assert smooth.n_cells == blocky.n_cells
        assert smooth.n_points == blocky.n_points
        assert _label_pairs(smooth) == _label_pairs(blocky)

    def test_labels_stay_in_contact(self, contour_tools: ContourTools) -> None:
        """The wall between the labels is not torn open by smoothing.

        Regression guard for ``non_manifold_smoothing``: the surface net shares
        its points between neighboring labels, so smoothing the mesh as one
        keeps them together, and the edges where three labels meet must move
        with the rest instead of being pinned.
        """
        blocky = contour_tools.extract_contours(_touching_boxes(), taubin_iterations=0)
        smooth = contour_tools.extract_contours(_touching_boxes(), taubin_iterations=20)

        def open_edges(contours: pv.PolyData) -> int:
            return int(
                contours.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    manifold_edges=False,
                    non_manifold_edges=False,
                ).n_cells
            )

        assert open_edges(smooth) == open_edges(blocky)

    def test_anisotropic_labelmap_is_contoured_isotropically(
        self, contour_tools: ContourTools
    ) -> None:
        """Boundaries land between the slices rather than terracing at them.

        Contoured on the labelmap's own 4 mm grid, every vertex of a face
        parallel to the slices lies on a slice plane; the isotropic resample is
        what lets one land in between.
        """
        labelmap = _touching_boxes()
        origin_z = float(np.asarray(labelmap.GetOrigin())[2])

        contours = contour_tools.extract_contours(labelmap)

        offsets = (np.asarray(contours.points)[:, 2] - origin_z) / 4.0
        assert np.any(np.abs(offsets - np.round(offsets)) > 0.1)


class TestExtractLabelSurfaces:
    """Labels extracted together must stay closed and stay in contact."""

    def test_surfaces_are_watertight_and_outward(
        self, contour_tools: ContourTools
    ) -> None:
        """Every label closes and encloses a positive volume."""
        surfaces = contour_tools.extract_label_surfaces(_touching_boxes())

        assert sorted(surfaces) == [1, 2]
        for label_id, surface in surfaces.items():
            assert contour_tools.is_watertight(surface), (
                f"label {label_id} must be closed"
            )
            assert surface.volume > 0.0, f"label {label_id} must face outward"

    def test_neighbors_share_the_wall_between_them(
        self, contour_tools: ContourTools
    ) -> None:
        """The two surfaces meet on identical vertices, smoothing included.

        Regression guard for the merge that welds the wall and for smoothing
        the welded mesh as one: smoothing the labels separately, or without
        ``non_manifold_smoothing``, pulls the two copies of the wall apart.
        """
        surfaces = contour_tools.extract_label_surfaces(
            _touching_boxes(), smoothing_iterations=20
        )

        shared = _point_keys(surfaces[1]) & _point_keys(surfaces[2])
        assert len(shared) > 100, "the shared wall must be meshed identically"
        distances = (
            surfaces[1].copy().compute_implicit_distance(surfaces[2])
        ).point_data["implicit_distance"]
        assert np.count_nonzero(np.abs(distances) < 1e-9) == len(shared)

    def test_volume_matches_the_labelmap(self, contour_tools: ContourTools) -> None:
        """Neither label is thinned or fattened by the smoothing.

        Not to the voxel exactly: the distance map is zero at the outermost
        labeled voxel's center rather than half a voxel beyond it, so the
        surface sits just inside the block of voxels it came from.
        """
        labelmap = _touching_boxes()
        labels = itk.GetArrayViewFromImage(labelmap)
        voxel_volume = float(np.prod(np.asarray(labelmap.GetSpacing())))

        surfaces = contour_tools.extract_label_surfaces(labelmap)

        for label_id, surface in surfaces.items():
            voxels = int(np.count_nonzero(labels == label_id))
            assert surface.volume == pytest.approx(voxels * voxel_volume, rel=0.15)


class TestExtractWatertightSurface:
    """The per-label surface must be closed and outward-oriented."""

    def test_border_label_surface_is_watertight(
        self, contour_tools: ContourTools, heart_labelmap: itk.Image
    ) -> None:
        """A label touching the volume border still closes.

        Regression guard for the one-voxel background pad: without it the
        isosurface is cut open where the structure reaches the image edge.
        """
        surface = contour_tools.extract_watertight_surface(
            _label_mask(heart_labelmap, BORDER_LABEL)
        )

        assert surface.n_cells > 0
        assert contour_tools.is_watertight(surface), (
            "every edge must be shared by exactly two faces"
        )

    def test_normals_point_outward(self, contour_tools: ContourTools) -> None:
        """A left-handed direction still yields a positive enclosed volume.

        Regression guard for ``auto_orient_normals``: VTK winds faces for a
        right-handed direction matrix, so LPS images with a negative-determinant
        direction otherwise come out with inward normals.
        """
        surface = contour_tools.extract_watertight_surface(_left_handed_box())

        volume = surface.triangulate().compute_cell_sizes(volume=True)
        assert float(np.sum(volume["Volume"])) == pytest.approx(0.0, abs=1e-9), (
            "cell volumes of a surface sum to zero; enclosed volume is checked below"
        )
        assert surface.volume > 0.0, "enclosed volume must be positive"

    def test_empty_surface_is_not_watertight(self, contour_tools: ContourTools) -> None:
        """A surface with no face fails the test rather than passing it vacuously."""
        assert not contour_tools.is_watertight(pv.PolyData())

    def test_decimation_reduces_triangle_count(
        self, contour_tools: ContourTools, heart_labelmap: itk.Image
    ) -> None:
        """surface_reduction_rate removes roughly that fraction of triangles."""
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        full = contour_tools.extract_watertight_surface(mask)
        reduced = contour_tools.extract_watertight_surface(
            mask, surface_reduction_rate=0.5
        )

        assert reduced.n_cells == pytest.approx(full.n_cells / 2, rel=0.1)


class TestExtractTetrahedra:
    """The per-label volume mesh must be tetrahedral and positively oriented."""

    def test_cells_are_tetrahedra(
        self, contour_tools: ContourTools, heart_labelmap: itk.Image
    ) -> None:
        """Every cell is a tetrahedron, six per labeled voxel."""
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        mesh = contour_tools.extract_tetrahedra(mask)

        assert set(mesh.celltypes) == {pv.CellType.TETRA}
        voxels = int(np.count_nonzero(itk.GetArrayViewFromImage(mask)))
        assert mesh.n_cells == 6 * voxels

    def test_left_handed_direction_yields_positive_volumes(
        self, contour_tools: ContourTools
    ) -> None:
        """No tetrahedron is inverted when the direction determinant is negative.

        Regression guard for the hexahedron winding flip: triangulating the raw
        voxel cells of a left-handed image inverts every tetrahedron.
        """
        mesh = contour_tools.extract_tetrahedra(_left_handed_box())

        volumes = mesh.compute_cell_sizes(volume=True)["Volume"]
        assert np.all(volumes > 0.0), "no tetrahedron may have negative volume"
        # 4 x 4 x 4 voxels of 1 x 2 x 3 mm.
        assert float(np.sum(volumes)) == pytest.approx(4 * 4 * 4 * 6.0)

    def test_volume_survives_coarsening(
        self, contour_tools: ContourTools, heart_labelmap: itk.Image
    ) -> None:
        """Elements twice the voxel keep the volume while shedding cells cubically."""
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        voxel = float(np.min(np.asarray(mask.GetSpacing())))
        full = contour_tools.extract_tetrahedra(mask, element_size_mm=voxel)
        coarse = contour_tools.extract_tetrahedra(mask, element_size_mm=2.0 * voxel)

        full_volume = float(np.sum(full.compute_cell_sizes(volume=True)["Volume"]))
        coarse_volume = float(np.sum(coarse.compute_cell_sizes(volume=True)["Volume"]))
        assert coarse_volume == pytest.approx(full_volume, rel=0.1)
        assert coarse.n_cells == pytest.approx(full.n_cells * 0.125, rel=0.3)

    def test_empty_mask_yields_an_empty_mesh(self, contour_tools: ContourTools) -> None:
        """A mask with nothing in it has no bounding box to mesh."""
        empty = itk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.uint8))

        assert contour_tools.extract_tetrahedra(empty).n_cells == 0

    def test_elements_are_isotropic(self, contour_tools: ContourTools) -> None:
        """The requested element size is what the mesh is built on.

        Regression guard for the anisotropic default: meshing a mask's own
        voxels inherits their slice pitch, which is what made these meshes
        blocky along one axis.
        """
        mesh = contour_tools.extract_tetrahedra(_left_handed_box(), element_size_mm=1.0)

        volumes = np.asarray(mesh.compute_cell_sizes(volume=True)["Volume"])
        # Six tetrahedra fill a 1 mm cube.
        assert float(np.median(volumes)) == pytest.approx(1.0 / 6.0, rel=1e-6)


class TestTrimTetrahedraToSurface:
    """Relaxing must land the boundary on the surface without wrecking cells."""

    @staticmethod
    def _boundary_gap(mesh: pv.UnstructuredGrid, surface: pv.PolyData) -> float:
        """Return the mean distance, in mm, from *mesh*'s boundary to *surface*."""
        boundary = mesh.extract_surface(algorithm="dataset_surface")
        distance = np.asarray(
            boundary.compute_implicit_distance(surface).point_data["implicit_distance"]
        )
        return float(np.abs(distance).mean())

    def test_boundary_lands_on_the_surface(
        self, contour_tools: ContourTools, heart_labelmap: itk.Image
    ) -> None:
        """The voxel staircase is relaxed onto the surface, not merely clipped.

        Regression guard for the interior smoothing: projecting the boundary
        alone flattens the cells behind it, the quality bound undoes the move,
        and the staircase survives.
        """
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = contour_tools.extract_tetrahedra(mask)
        relaxed = contour_tools.trim_tetrahedra_to_surface(mesh, surface)

        before = self._boundary_gap(mesh, surface)
        assert before > 0.0, "the staircase must start off the surface"
        assert self._boundary_gap(relaxed, surface) < 0.25 * before

    def test_cells_are_kept_whole_and_well_shaped(
        self, contour_tools: ContourTools, heart_labelmap: itk.Image
    ) -> None:
        """Cells are only dropped, never cut into slivers.

        Regression guard against swapping the crinkle clip for a real
        ``clip_surface`` cut: cutting subdivides the boundary tetrahedra and
        drives about a tenth of them below a scaled Jacobian of 0.1.
        """
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = contour_tools.extract_tetrahedra(mask)
        relaxed = contour_tools.trim_tetrahedra_to_surface(mesh, surface)

        assert set(relaxed.celltypes) == {pv.CellType.TETRA}
        assert 0 < relaxed.n_cells <= mesh.n_cells
        quality = np.asarray(
            relaxed.cell_quality(["scaled_jacobian"]).cell_data["scaled_jacobian"]
        )
        assert np.min(quality) >= 0.1, "no tetrahedron may be left a sliver"

    def test_anatomy_color_survives(self, contour_tools: ContourTools) -> None:
        """Cell and field data attached at extraction are carried through."""
        mask = _left_handed_box()
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = contour_tools.extract_tetrahedra(mask, anatomy_names=["heart"])
        relaxed = contour_tools.trim_tetrahedra_to_surface(mesh, surface)

        assert "Color" in relaxed.cell_data
        assert np.array_equal(
            relaxed.field_data["AnatomyColor"], mesh.field_data["AnatomyColor"]
        )


class TestRepairInvertedTetrahedra:
    """Node relaxation must fix a flippable mesh and give up on one that can't."""

    @staticmethod
    def _single_tetra(points: np.ndarray) -> pv.UnstructuredGrid:
        """Build a one-cell tetrahedral mesh from four *points*."""
        cells = np.array([4, 0, 1, 2, 3])
        return pv.UnstructuredGrid(cells, [pv.CellType.TETRA], points)

    def test_repairs_an_inverted_tetrahedron(self, contour_tools: ContourTools) -> None:
        """Swapping two corners inverts the cell; relaxation must restore it."""
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        mesh = self._single_tetra(points)

        repaired = contour_tools.repair_inverted_tetrahedra(mesh)

        corners = repaired.points[repaired.cells_dict[np.uint8(pv.CellType.TETRA)]][0]
        edges = corners[1:, :] - corners[0:1, :]
        assert np.linalg.det(edges) / 6.0 > 0.0

    def test_raises_when_unrecoverable(self, contour_tools: ContourTools) -> None:
        """A degenerate cell with no neighbors to average toward can't be fixed."""
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        mesh = self._single_tetra(points)

        with pytest.raises(ValueError, match="still inverted or degenerate"):
            contour_tools.repair_inverted_tetrahedra(mesh, max_iterations=2)

    def test_raises_when_no_tetra_cells(self, contour_tools: ContourTools) -> None:
        """A mesh without TETRA cells fails with a clear message, not a KeyError."""
        mesh = pv.UnstructuredGrid()

        with pytest.raises(ValueError, match="no TETRA cells"):
            contour_tools.repair_inverted_tetrahedra(mesh)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
