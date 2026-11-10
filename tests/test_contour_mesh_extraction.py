"""Tests for ProcessSurfaceMesh and ProcessVolumetricMesh label extraction.

These use ``data/test/slicer_heart_small``, whose direction matrix is
right-handed, plus a synthetic left-handed image for the cases that only a
negative-determinant direction can exercise.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, cast

import itk
import numpy as np
import pytest
import pyvista as pv
import torch
import vtk
from scipy import ndimage

from monai_physio.process_images import ProcessImages
from monai_physio.process_mesh_kernels import ProcessMeshKernels
from monai_physio.process_surface_mesh import ProcessSurfaceMesh
from monai_physio.process_volumetric_mesh import ProcessVolumetricMesh

#: A label that reaches the volume border, so its surface only closes if the
#: mask is padded first.
BORDER_LABEL = 5


@pytest.fixture(scope="module")
def heart_labelmap(test_labelmaps: list[dict[str, Any]]) -> itk.Image:
    """Return one small labelmap."""
    return test_labelmaps[0]["labelmap"]


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


def _point_keys(surface: pv.DataSet) -> set[tuple[float, ...]]:
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


@pytest.mark.requires_gpu
class TestLabelSurfaceSmoothing:
    """The label surfaces must lose the voxel blocks they are drawn on."""

    def test_smoothing_takes_the_voxel_blocks_out(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """Taubin smoothing leaves the surface markedly less faceted."""
        blocky = contour_tools.extract_label_surfaces(
            _touching_boxes(), distance_sigma_mm=0.0, smoothing_iterations=0
        )
        smooth = contour_tools.extract_label_surfaces(
            _touching_boxes(), distance_sigma_mm=0.0, smoothing_iterations=20
        )

        for label_id in (1, 2):
            assert _roughness(smooth[label_id]) < _roughness(blocky[label_id])

    def test_anisotropic_labelmap_is_contoured_isotropically(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """Boundaries land between the slices rather than terracing at them.

        Contoured on the labelmap's own 4 mm grid, every vertex of a face
        parallel to the slices lies on a slice plane; the isotropic resample is
        what lets one land in between.
        """
        labelmap = _touching_boxes()
        origin_z = float(np.asarray(labelmap.GetOrigin())[2])

        surface = contour_tools.extract_label_surfaces(labelmap)[1]

        offsets = (np.asarray(surface.points)[:, 2] - origin_z) / 4.0
        assert np.any(np.abs(offsets - np.round(offsets)) > 0.1)


@pytest.mark.requires_gpu
class TestExtractLabelSurfaces:
    """Labels extracted together must stay closed and stay in contact."""

    def test_surfaces_are_watertight_and_outward(
        self, contour_tools: ProcessSurfaceMesh
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
        self, contour_tools: ProcessSurfaceMesh
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

    def test_volume_matches_the_labelmap(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """Neither label is thinned or fattened by the smoothing.

        Not to the voxel exactly: the distance-map blur rounds the boxes'
        edges and corners, which on 4 mm slices costs a few percent.
        """
        labelmap = _touching_boxes()
        labels = itk.GetArrayViewFromImage(labelmap)
        voxel_volume = float(np.prod(np.asarray(labelmap.GetSpacing())))

        surfaces = contour_tools.extract_label_surfaces(labelmap)

        for label_id, surface in surfaces.items():
            voxels = int(np.count_nonzero(labels == label_id))
            assert surface.volume == pytest.approx(voxels * voxel_volume, rel=0.08)

    @pytest.mark.parametrize("spacing_mm", [0.5, 1.0])
    def test_sphere_keeps_its_volume(
        self, contour_tools: ProcessSurfaceMesh, spacing_mm: float
    ) -> None:
        """A 12 mm sphere encloses its analytic volume.

        The background-facing wall must sit midway between the label's and the
        background's edge voxels; on the label's own edge voxels it loses half
        a voxel all round, a fifth of this sphere on a 1 mm grid.
        """
        radius_mm = 12.0
        n = int(2 * radius_mm / spacing_mm) + 8
        center = (n - 1) / 2.0
        zz, yy, xx = np.indices((n, n, n))
        arr = (
            np.sqrt((zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2)
            * spacing_mm
            <= radius_mm
        ).astype(np.uint8)
        labelmap = itk.GetImageFromArray(arr)
        labelmap.SetSpacing([spacing_mm] * 3)

        surface = contour_tools.extract_label_surfaces(labelmap)[1]

        assert surface.volume == pytest.approx(
            4.0 / 3.0 * np.pi * radius_mm**3, rel=0.05
        )

    def test_thin_shell_stays_in_one_piece(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """A wall three voxels thick contours to a single closed surface."""
        n = 40
        center = (n - 1) / 2.0
        zz, yy, xx = np.indices((n, n, n))
        radius = np.sqrt((zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2)
        arr = ((radius >= 12.0) & (radius < 15.0)).astype(np.uint8)
        labelmap = itk.GetImageFromArray(arr)

        surface = contour_tools.extract_label_surfaces(labelmap)[1]

        assert surface.connectivity().point_data["RegionId"].max() == 1, (
            "an outer and an inner surface, nothing else"
        )

    def test_label_thinner_than_the_grid_returns_no_surface(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """A wall too thin to survive the resample is reported, not a KeyError."""
        n = 40
        center = (n - 1) / 2.0
        zz, yy, xx = np.indices((n, n, n))
        radius = np.sqrt((zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2)
        arr = ((radius >= 12.0) & (radius < 12.6)).astype(np.uint8)

        assert contour_tools.extract_label_surfaces(itk.GetImageFromArray(arr)) == {}

    def test_walls_match_where_three_labels_meet(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """A vertex lying on a neighbor's surface is one of that neighbor's vertices.

        Label 5 overlaps the wall between labels 1 and 2, so the three meet
        along lines; the margin field keeps every wall exact there too.
        """
        surfaces = contour_tools.extract_label_surfaces(
            _blob_labelmap((1.0, 1.0, 1.0), 1.0)
        )

        for first, second in ((1, 2), (1, 5), (2, 5)):
            on_wall = (
                np.abs(
                    surfaces[first]
                    .copy()
                    .compute_implicit_distance(surfaces[second])
                    .point_data["implicit_distance"]
                )
                < 1e-3
            )
            assert np.count_nonzero(on_wall) > 0, f"{first} and {second} must touch"
            wall_points = np.round(np.asarray(surfaces[first].points)[on_wall], 6)
            assert {tuple(p) for p in wall_points.tolist()} <= _point_keys(
                surfaces[second]
            )

    def test_label_ids_treat_other_labels_as_background(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """A subset gives the surfaces of a labelmap holding only that subset."""
        labelmap = _blob_labelmap((1.0, 1.0, 1.0), 1.0)
        arr = itk.GetArrayFromImage(labelmap)
        subset_only = itk.GetImageFromArray(np.where(np.isin(arr, [1, 2]), arr, 0))
        subset_only.CopyInformation(labelmap)

        actual = contour_tools.extract_label_surfaces(labelmap, label_ids=[1, 2])
        expected = contour_tools.extract_label_surfaces(subset_only)

        assert sorted(actual) == sorted(expected) == [1, 2]
        for label_id, surface in actual.items():
            assert np.array_equal(surface.points, expected[label_id].points)

    def test_surfaces_carry_their_label_id(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        surfaces = contour_tools.extract_label_surfaces(_touching_boxes())

        for label_id, surface in surfaces.items():
            assert surface.field_data["SegmentationLabelIds"].tolist() == [label_id]

    def test_reduction_lowers_the_triangle_count(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        full = contour_tools.extract_label_surfaces(_touching_boxes())
        reduced = contour_tools.extract_label_surfaces(
            _touching_boxes(), surface_reduction_rate=0.5
        )

        for label_id, surface in reduced.items():
            assert surface.n_cells == pytest.approx(full[label_id].n_cells / 2, rel=0.1)
            assert surface.field_data["SegmentationLabelIds"].tolist() == [label_id]
            assert surface.volume > 0.0


class TestSingleMaskSurface:
    """A binary mask's surface must be closed and outward-oriented."""

    @pytest.mark.requires_gpu
    def test_border_label_surface_is_watertight(
        self, contour_tools: ProcessSurfaceMesh, heart_labelmap: itk.Image
    ) -> None:
        """A label touching the volume border still closes.

        Regression guard for the background pad: without it the isosurface is
        cut open where the structure reaches the image edge.
        """
        surface = contour_tools.extract_label_surfaces(
            _label_mask(heart_labelmap, BORDER_LABEL)
        )[1]

        assert surface.n_cells > 0
        assert contour_tools.is_watertight(surface), (
            "every edge must be shared by exactly two faces"
        )

    @pytest.mark.requires_gpu
    def test_normals_point_outward(self, contour_tools: ProcessSurfaceMesh) -> None:
        """A left-handed direction still yields a positive enclosed volume.

        Regression guard for ``auto_orient_normals``: VTK winds faces for a
        right-handed direction matrix, so LPS images with a negative-determinant
        direction otherwise come out with inward normals.
        """
        surface = contour_tools.extract_label_surfaces(_left_handed_box())[1]

        assert surface.volume > 0.0, "enclosed volume must be positive"

    def test_empty_surface_is_not_watertight(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """A surface with no face fails the test rather than passing it vacuously."""
        assert not contour_tools.is_watertight(pv.PolyData())

    @pytest.mark.requires_gpu
    def test_reduction_keeps_the_surface_closed(
        self, contour_tools: ProcessSurfaceMesh, heart_labelmap: itk.Image
    ) -> None:
        """ACVD remeshing removes triangles without opening the surface."""
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        reduced = contour_tools.extract_label_surfaces(
            mask, surface_reduction_rate=0.5
        )[1]

        assert contour_tools.is_watertight(reduced)


@pytest.mark.requires_gpu
class TestExtractLabelTetrahedraVtk:
    """The per-label volume mesh must be tetrahedral and positively oriented."""

    @pytest.mark.requires_gpu
    def test_cells_are_tetrahedra(
        self, volume_tools: ProcessVolumetricMesh, heart_labelmap: itk.Image
    ) -> None:
        """Every cell is a tetrahedron, six per labeled voxel."""
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        mesh = volume_tools.extract_label_tetrahedra_vtk(mask)[1]

        assert set(mesh.celltypes) == {pv.CellType.TETRA}
        voxels = int(np.count_nonzero(itk.GetArrayViewFromImage(mask)))
        assert mesh.n_cells == 6 * voxels

    def test_left_handed_direction_yields_positive_volumes(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """No tetrahedron is inverted when the direction determinant is negative.

        Regression guard for the hexahedron winding flip: triangulating the raw
        voxel cells of a left-handed image inverts every tetrahedron.
        """
        mesh = volume_tools.extract_label_tetrahedra_vtk(_left_handed_box())[1]

        volumes = mesh.compute_cell_sizes(volume=True)["Volume"]
        assert np.all(volumes > 0.0), "no tetrahedron may have negative volume"
        # 4 x 4 x 4 voxels of 1 x 2 x 3 mm.
        assert float(np.sum(volumes)) == pytest.approx(4 * 4 * 4 * 6.0)

    @pytest.mark.requires_gpu
    def test_volume_survives_coarsening(
        self, volume_tools: ProcessVolumetricMesh, heart_labelmap: itk.Image
    ) -> None:
        """Elements twice the voxel keep the volume while shedding cells cubically."""
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        voxel = float(np.min(np.asarray(mask.GetSpacing())))
        full = volume_tools.extract_label_tetrahedra_vtk(mask, element_size_mm=voxel)[1]
        coarse = volume_tools.extract_label_tetrahedra_vtk(
            mask, element_size_mm=2.0 * voxel
        )[1]

        full_volume = float(np.sum(full.compute_cell_sizes(volume=True)["Volume"]))
        coarse_volume = float(np.sum(coarse.compute_cell_sizes(volume=True)["Volume"]))
        assert coarse_volume == pytest.approx(full_volume, rel=0.1)
        assert coarse.n_cells == pytest.approx(full.n_cells * 0.125, rel=0.3)

    def test_empty_mask_yields_an_empty_mesh(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """A mask with nothing in it has no bounding box to mesh."""
        empty = itk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.uint8))

        assert volume_tools.extract_label_tetrahedra_vtk(empty) == {}

    def test_each_label_gets_its_own_mesh(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """Touching labels are meshed apart, tagged, and meet on coincident nodes."""
        labelmap = _touching_boxes()
        meshes = volume_tools.extract_label_tetrahedra_vtk(labelmap)

        assert sorted(meshes) == [1, 2]
        labels = itk.GetArrayViewFromImage(labelmap)
        for label_id, mesh in meshes.items():
            assert mesh.field_data["SegmentationLabelIds"].tolist() == [label_id]
            assert mesh.n_cells == 6 * int(np.count_nonzero(labels == label_id))
        # The shared wall is 32 x 8 voxels, so 33 x 9 corner nodes.
        assert len(_point_keys(meshes[1]) & _point_keys(meshes[2])) == 33 * 9

    def test_label_ids_select_the_labels(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """Labels left out of *label_ids* are treated as background."""
        meshes = volume_tools.extract_label_tetrahedra_vtk(
            _touching_boxes(), label_ids=[2]
        )

        assert sorted(meshes) == [2]

    @pytest.mark.requires_gpu
    def test_elements_are_isotropic(self, volume_tools: ProcessVolumetricMesh) -> None:
        """The requested element size is what the mesh is built on.

        Regression guard for the anisotropic default: meshing a mask's own
        voxels inherits their slice pitch, which is what made these meshes
        blocky along one axis.
        """
        mesh = volume_tools.extract_label_tetrahedra_vtk(
            _left_handed_box(), element_size_mm=1.0
        )[1]

        volumes = np.asarray(mesh.compute_cell_sizes(volume=True)["Volume"])
        # Six tetrahedra fill a 1 mm cube.
        assert float(np.median(volumes)) == pytest.approx(1.0 / 6.0, rel=1e-6)


def _volume_and_centroid(mesh: pv.UnstructuredGrid) -> tuple[float, np.ndarray]:
    """Return *mesh*'s total cell volume and volume-weighted centroid."""
    volumes = np.asarray(mesh.compute_cell_sizes(volume=True)["Volume"])
    centers = np.asarray(mesh.cell_centers().points)
    total = float(np.sum(volumes))
    return total, (centers * volumes[:, np.newaxis]).sum(axis=0) / total


_TETRAHEDRA_MESHERS = [
    pytest.param(
        "extract_label_tetrahedra_netgen",
        marks=[
            pytest.mark.requires_gpu,
            pytest.mark.skipif(
                importlib.util.find_spec("netgen") is None,
                reason="netgen extra (netgen-mesher) not installed",
            ),
        ],
        id="netgen",
    ),
    pytest.param(
        "extract_label_tetrahedra_simpleware",
        marks=[pytest.mark.requires_gpu, pytest.mark.requires_simpleware],
        id="simpleware",
    ),
]


@pytest.mark.parametrize("mesher", _TETRAHEDRA_MESHERS)
class TestExtractLabelTetrahedraExternal:
    """External meshers must fill the mask, in its frame, with valid tetrahedra."""

    def test_mesh_fills_the_mask(
        self,
        volume_tools: ProcessVolumetricMesh,
        heart_labelmap: itk.Image,
        mesher: str,
    ) -> None:
        """Cells are positive tetrahedra matching the voxel mesh's volume and place."""
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        mesh = getattr(volume_tools, mesher)(mask, element_size_mm=3.0)[1]

        assert set(mesh.celltypes) == {pv.CellType.TETRA}
        volumes = mesh.compute_cell_sizes(volume=True)["Volume"]
        assert np.all(volumes > 0.0), "no tetrahedron may be inverted"
        volume, centroid = _volume_and_centroid(mesh)
        voxel_volume, voxel_centroid = _volume_and_centroid(
            volume_tools.extract_label_tetrahedra_vtk(mask)[1]
        )
        # netgen meshes a surface smoothed at the element size, which sheds
        # about 5% of this label's volume at 3 mm.
        assert volume == pytest.approx(voxel_volume, rel=0.1)
        assert np.linalg.norm(centroid - voxel_centroid) < float(
            np.max(mask.GetSpacing())
        )

    def test_left_handed_direction_keeps_frame_and_orientation(
        self, volume_tools: ProcessVolumetricMesh, mesher: str
    ) -> None:
        """A negative-determinant direction neither inverts nor moves the mesh."""
        mask = _left_handed_box()
        mesh = getattr(volume_tools, mesher)(mask, element_size_mm=1.0)[1]

        volumes = mesh.compute_cell_sizes(volume=True)["Volume"]
        assert mesh.n_cells > 0
        assert np.all(volumes > 0.0), "no tetrahedron may be inverted"
        _, centroid = _volume_and_centroid(mesh)
        _, voxel_centroid = _volume_and_centroid(
            volume_tools.extract_label_tetrahedra_vtk(mask)[1]
        )
        assert np.linalg.norm(centroid - voxel_centroid) < 1.0

    def test_empty_mask_yields_an_empty_mesh(
        self, volume_tools: ProcessVolumetricMesh, mesher: str
    ) -> None:
        """A mask with nothing in it is not handed to the mesher."""
        empty = itk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.uint8))

        assert getattr(volume_tools, mesher)(empty) == {}

    def test_touching_labels_meet_on_coincident_nodes(
        self, volume_tools: ProcessVolumetricMesh, mesher: str
    ) -> None:
        """Each label gets its own valid mesh, and touching labels' meshes meet."""
        meshes = getattr(volume_tools, mesher)(_touching_boxes(), element_size_mm=2.0)

        assert sorted(meshes) == [1, 2]
        for label_id, mesh in meshes.items():
            assert mesh.field_data["SegmentationLabelIds"].tolist() == [label_id]
            volumes = mesh.compute_cell_sizes(volume=True)["Volume"]
            assert np.all(volumes > 0.0), "no tetrahedron may be inverted"
        assert len(_point_keys(meshes[1]) & _point_keys(meshes[2])) > 100


@pytest.mark.requires_gpu
@pytest.mark.requires_simpleware
def test_simpleware_independent_labels_share_no_nodes(
    volume_tools: ProcessVolumetricMesh,
) -> None:
    """``shared_interfaces=False`` meshes each label in a model of its own."""
    meshes = volume_tools.extract_label_tetrahedra_simpleware(
        _touching_boxes(), element_size_mm=2.0, shared_interfaces=False
    )

    assert sorted(meshes) == [1, 2]
    assert not _point_keys(meshes[1]) & _point_keys(meshes[2])


@pytest.mark.requires_gpu
@pytest.mark.requires_simpleware
def test_simpleware_blur_smooths_the_boundary(
    volume_tools: ProcessVolumetricMesh, heart_labelmap: itk.Image
) -> None:
    """The default blur takes the voxel staircase out of Simpleware's boundary.

    A staircase shows as boundary faces whose normals lie along an image
    axis; on this label the blur drops their share from about 47% to 41%.
    """
    mask = _label_mask(heart_labelmap, BORDER_LABEL)
    staircase = volume_tools.extract_label_tetrahedra_simpleware(
        mask, element_size_mm=3.0, smoothing_sigma_mm=0.0
    )[1]
    smoothed = volume_tools.extract_label_tetrahedra_simpleware(
        mask, element_size_mm=3.0
    )[1]

    def axis_aligned_fraction(mesh: pv.UnstructuredGrid) -> float:
        boundary = mesh.extract_surface(algorithm="dataset_surface").compute_normals(
            cell_normals=True, point_normals=False
        )
        normals = np.abs(np.asarray(boundary.cell_data["Normals"]))
        return float((normals.max(axis=1) > np.cos(np.radians(10.0))).mean())

    assert axis_aligned_fraction(smoothed) < 0.95 * axis_aligned_fraction(staircase)


@pytest.mark.requires_gpu
@pytest.mark.skipif(
    importlib.util.find_spec("netgen") is None,
    reason="netgen extra (netgen-mesher) not installed",
)
def test_netgen_blur_reaches_the_surface(
    volume_tools: ProcessVolumetricMesh, heart_labelmap: itk.Image
) -> None:
    """``smoothing_sigma_mm`` sets the blur of the surface netgen fills.

    A wider blur pulls the contour in; on this label 6 mm sheds about 6% more
    of the volume than no blur.
    """
    mask = _label_mask(heart_labelmap, BORDER_LABEL)
    unblurred, _ = _volume_and_centroid(
        volume_tools.extract_label_tetrahedra_netgen(
            mask, element_size_mm=3.0, smoothing_sigma_mm=0.0
        )[1]
    )
    blurred, _ = _volume_and_centroid(
        volume_tools.extract_label_tetrahedra_netgen(
            mask, element_size_mm=3.0, smoothing_sigma_mm=6.0
        )[1]
    )

    assert blurred < 0.97 * unblurred


@pytest.mark.requires_gpu
@pytest.mark.skipif(
    importlib.util.find_spec("netgen") is None,
    reason="netgen extra (netgen-mesher) not installed",
)
def test_netgen_smoothing_iterations_reach_the_surface(
    volume_tools: ProcessVolumetricMesh, heart_labelmap: itk.Image
) -> None:
    """``smoothing_iterations`` sets the Taubin smoothing of the surface netgen fills."""
    mask = _label_mask(heart_labelmap, BORDER_LABEL)

    def boundary_roughness(smoothing_iterations: int) -> float:
        mesh = volume_tools.extract_label_tetrahedra_netgen(
            mask, element_size_mm=3.0, smoothing_iterations=smoothing_iterations
        )[1]
        return _roughness(mesh.extract_surface(algorithm="dataset_surface"))

    assert boundary_roughness(30) < boundary_roughness(0)


@pytest.mark.requires_gpu
@pytest.mark.skipif(
    importlib.util.find_spec("netgen") is None,
    reason="netgen extra (netgen-mesher) not installed",
)
def test_netgen_independent_labels_share_few_nodes(
    volume_tools: ProcessVolumetricMesh,
) -> None:
    """``shared_interfaces=False`` re-tiles each label's surface on its own.

    netgen keeps the vertices of sharp feature edges, such as the rim where
    the boxes' shared wall meets their outer faces, so a few nodes still
    coincide, but the wall's interior is tiled independently.
    """

    def shared_nodes(shared_interfaces: bool) -> int:
        meshes = volume_tools.extract_label_tetrahedra_netgen(
            _touching_boxes(),
            element_size_mm=2.0,
            shared_interfaces=shared_interfaces,
            contour_spacing_mm=1.0,
        )
        assert sorted(meshes) == [1, 2]
        for mesh in meshes.values():
            volumes = mesh.compute_cell_sizes(volume=True)["Volume"]
            assert np.all(volumes > 0.0), "no tetrahedron may be inverted"
        return len(_point_keys(meshes[1]) & _point_keys(meshes[2]))

    assert shared_nodes(False) < 0.2 * shared_nodes(True)


@pytest.mark.skipif(
    importlib.util.find_spec("netgen") is None,
    reason="netgen extra (netgen-mesher) not installed",
)
def test_netgen_retiling_a_self_intersecting_surface_raises(
    volume_tools: ProcessVolumetricMesh,
) -> None:
    """netgen's re-tiling ends its process on such a surface; ours survives."""
    overlapping_boxes = pv.merge(
        [
            pv.Box(bounds=(0, 10, 0, 10, 0, 10), level=4).triangulate(),
            pv.Box(bounds=(5, 15, 5, 15, 5, 15), level=4).triangulate(),
        ]
    ).extract_surface()

    with pytest.raises(RuntimeError, match="intersects itself"):
        volume_tools._retile_and_fill_netgen(overlapping_boxes, 2.0, 1)


@pytest.mark.requires_gpu
@pytest.mark.skipif(
    importlib.util.find_spec("netgen") is None,
    reason="netgen extra (netgen-mesher) not installed",
)
def test_netgen_contour_spacing_sets_the_boundary_triangles(
    volume_tools: ProcessVolumetricMesh, heart_labelmap: itk.Image
) -> None:
    """A finer contour grid multiplies the shared boundary's triangles.

    Re-tiling the surfaces on their own brings them back to about the
    element size.
    """
    mask = _label_mask(heart_labelmap, BORDER_LABEL)

    def boundary_triangles(**kwargs: Any) -> int:
        mesh = volume_tools.extract_label_tetrahedra_netgen(
            mask, element_size_mm=3.0, **kwargs
        )[1]
        return int(mesh.extract_surface(algorithm="dataset_surface").n_cells)

    coarse = boundary_triangles()
    fine = boundary_triangles(contour_spacing_mm=1.5)
    retiled = boundary_triangles(contour_spacing_mm=1.5, shared_interfaces=False)

    assert fine > 3 * coarse
    assert 0.5 * coarse < retiled < 1.5 * coarse


@pytest.mark.requires_gpu
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
        self,
        contour_tools: ProcessSurfaceMesh,
        volume_tools: ProcessVolumetricMesh,
        heart_labelmap: itk.Image,
    ) -> None:
        """The voxel staircase is relaxed onto the surface, not merely clipped.

        Regression guard for the interior smoothing: projecting the boundary
        alone flattens the cells behind it, the quality bound undoes the move,
        and the staircase survives.
        """
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = volume_tools.extract_label_tetrahedra_vtk(mask)[1]
        relaxed = volume_tools.trim_tetrahedra_to_surface(mesh, surface)

        before = self._boundary_gap(mesh, surface)
        assert before > 0.0, "the staircase must start off the surface"
        assert self._boundary_gap(relaxed, surface) < 0.25 * before

    def test_cells_are_kept_whole_and_well_shaped(
        self,
        contour_tools: ProcessSurfaceMesh,
        volume_tools: ProcessVolumetricMesh,
        heart_labelmap: itk.Image,
    ) -> None:
        """Cells are only dropped, never cut into slivers.

        Regression guard against swapping the crinkle clip for a real
        ``clip_surface`` cut: cutting subdivides the boundary tetrahedra and
        drives about a tenth of them below a scaled Jacobian of 0.1.
        """
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = volume_tools.extract_label_tetrahedra_vtk(mask)[1]
        relaxed = volume_tools.trim_tetrahedra_to_surface(mesh, surface)

        assert set(relaxed.celltypes) == {pv.CellType.TETRA}
        assert 0 < relaxed.n_cells <= mesh.n_cells
        quality = np.asarray(
            relaxed.cell_quality(["scaled_jacobian"]).cell_data["scaled_jacobian"]
        )
        assert np.min(quality) >= 0.1, "no tetrahedron may be left a sliver"

    def test_cell_and_field_data_survive(
        self, contour_tools: ProcessSurfaceMesh, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """Cell and field data on the mesh are carried through."""
        mask = _left_handed_box()
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = volume_tools.extract_label_tetrahedra_vtk(mask)[1]
        mesh.cell_data["SegmentationLabelIds"] = np.ones(mesh.n_cells, np.int32)
        relaxed = volume_tools.trim_tetrahedra_to_surface(mesh, surface)

        assert "SegmentationLabelIds" in relaxed.cell_data
        assert relaxed.field_data["SegmentationLabelIds"].tolist() == [1]


@pytest.mark.requires_gpu
class TestRemoveBoundarySlivers:
    """Flat caps on the boundary go; well-shaped cells, even on it, stay."""

    def test_removes_a_flat_cap_and_keeps_a_good_cell(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """Every node of a two-cell mesh is on its boundary; only quality decides."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                # Barely below the shared face: a flat cap glued to it.
                [0.4, 0.4, -0.02],
            ]
        )
        cells = np.array([4, 0, 1, 2, 3, 4, 0, 2, 1, 4])
        mesh = pv.UnstructuredGrid(cells, [pv.CellType.TETRA] * 2, points)
        mesh.cell_data["cell_ids"] = np.array([0, 1])

        cleaned = volume_tools.remove_boundary_slivers(mesh)

        assert cleaned.n_cells == 1
        assert cleaned.cell_data["cell_ids"].tolist() == [0]

    def test_mesh_of_only_slivers_comes_back_empty(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.4, 0.4, 0.02]]
        )
        mesh = pv.UnstructuredGrid(
            np.array([4, 0, 1, 2, 3]), [pv.CellType.TETRA], points
        )

        assert volume_tools.remove_boundary_slivers(mesh).n_cells == 0

    def test_leaves_a_mesh_without_slivers_unchanged(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        mesh = volume_tools.extract_label_tetrahedra_vtk(
            _left_handed_box(), element_size_mm=1.0
        )[1]

        assert volume_tools.remove_boundary_slivers(mesh) is mesh

    def test_trimmed_heart_mesh_loses_its_caps_but_not_its_shape(
        self,
        contour_tools: ProcessSurfaceMesh,
        volume_tools: ProcessVolumetricMesh,
        heart_labelmap: itk.Image,
    ) -> None:
        """The caps trimming leaves are removed at a small cost in volume."""
        mask = _label_mask(heart_labelmap, BORDER_LABEL)
        surface = contour_tools.extract_label_surfaces(mask)[1]
        relaxed = volume_tools.trim_tetrahedra_to_surface(
            volume_tools.extract_label_tetrahedra_vtk(mask)[1], surface
        )

        cleaned = volume_tools.remove_boundary_slivers(relaxed)

        connectivity = cast(np.ndarray, cleaned.cells_dict[np.uint8(pv.CellType.TETRA)])
        boundary = np.zeros(cleaned.n_points, dtype=bool)
        boundary[
            cleaned.extract_surface(
                pass_pointid=True, algorithm="dataset_surface"
            ).point_data["vtkOriginalPointIds"]
        ] = True
        quality = np.asarray(
            cleaned.cell_quality(["scaled_jacobian"]).cell_data["scaled_jacobian"]
        )
        assert not np.any(boundary[connectivity].all(axis=1) & (quality < 0.15))
        before = float(np.sum(relaxed.compute_cell_sizes(volume=True)["Volume"]))
        after = float(np.sum(cleaned.compute_cell_sizes(volume=True)["Volume"]))
        assert after > 0.95 * before


class TestRepairInvertedTetrahedra:
    """Node relaxation must fix a flippable mesh and give up on one that can't."""

    @staticmethod
    def _single_tetra(points: np.ndarray) -> pv.UnstructuredGrid:
        """Build a one-cell tetrahedral mesh from four *points*."""
        cells = np.array([4, 0, 1, 2, 3])
        return pv.UnstructuredGrid(cells, [pv.CellType.TETRA], points)

    def test_repairs_an_inverted_tetrahedron(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """Swapping two corners inverts the cell; relaxation must restore it."""
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        mesh = self._single_tetra(points)

        repaired = volume_tools.repair_inverted_tetrahedra(mesh)

        tets = cast(np.ndarray, repaired.cells_dict[np.uint8(pv.CellType.TETRA)])
        corners = repaired.points[tets][0]
        edges = corners[1:, :] - corners[0:1, :]
        assert np.linalg.det(edges) / 6.0 > 0.0

    def test_raises_when_unrecoverable(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """A degenerate cell with no neighbors to average toward can't be fixed."""
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        mesh = self._single_tetra(points)

        with pytest.raises(ValueError, match="still inverted or degenerate"):
            volume_tools.repair_inverted_tetrahedra(mesh, max_iterations=2)

    def test_raises_when_no_tetra_cells(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """A mesh without TETRA cells fails with a clear message, not a KeyError."""
        mesh = pv.UnstructuredGrid()

        with pytest.raises(ValueError, match="no TETRA cells"):
            volume_tools.repair_inverted_tetrahedra(mesh)


def _frozen_base_trim(
    tetrahedra: pv.UnstructuredGrid,
    surface: pv.PolyData,
    iterations: int = 5,
    relaxation: float = 0.6,
    min_scaled_jacobian: float = 0.1,
) -> tuple[pv.UnstructuredGrid, int]:
    """Run the pre-S1 relaxation; return the mesh and its most damping passes."""
    relaxed = tetrahedra.clip_surface(surface, invert=True, crinkle=True)
    connectivity = np.asarray(relaxed.cells_dict[np.uint8(pv.CellType.TETRA)])
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
    most_passes = 0
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
            surface.find_closest_cell(target[boundary_ids], return_closest_point=True),
        )
        target[boundary_ids] = closest
        step = relaxation * (target - points)
        damping = np.ones(point_count)
        passes = 0
        while True:
            passes += 1
            relaxed.points = points + damping[:, np.newaxis] * step
            quality = np.asarray(
                relaxed.cell_quality(["scaled_jacobian"]).cell_data["scaled_jacobian"]
            )
            below_bound = quality < min_scaled_jacobian
            if not np.any(below_bound):
                break
            damped_ids = np.unique(connectivity[below_bound])
            damped_ids = damped_ids[damping[damped_ids] > 0.0]
            if damped_ids.size == 0:
                break
            halved = damping[damped_ids] * 0.5
            damping[damped_ids] = np.where(halved < 2.0**-10, 0.0, halved)
        most_passes = max(most_passes, passes)
        points = np.array(relaxed.points)
    return relaxed, most_passes


def _ball_mask(spacing: tuple[float, float, float], direction_sign: float) -> itk.Image:
    """Build a ball whose surface is smooth but whose voxel mesh is a staircase."""
    zz, yy, xx = np.indices((26, 26, 26))
    arr = ((xx - 12.5) ** 2 + (yy - 12.5) ** 2 + (zz - 12.5) ** 2 < 9.0**2).astype(
        np.uint8
    )
    mask = itk.GetImageFromArray(arr)
    mask.SetSpacing(list(spacing))
    mask.SetDirection(itk.matrix_from_array(np.diag([1.0, 1.0, direction_sign])))
    return mask


def _point_rmse(first: pv.DataSet, second: pv.DataSet) -> float:
    """Return the point-to-point RMSE between two meshes of equal point count."""
    assert first.n_points == second.n_points
    delta = np.asarray(first.points, dtype=np.float64) - np.asarray(
        second.points, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))


def _min_scaled_jacobian(mesh: pv.UnstructuredGrid) -> float:
    """Return the worst scaled Jacobian over *mesh*'s cells."""
    quality = mesh.cell_quality(["scaled_jacobian"]).cell_data["scaled_jacobian"]
    return float(np.min(quality))


@pytest.mark.requires_gpu
class TestS1TrimMatchesFrozenBase:
    """The GPU relaxation must land where the pre-S1 CPU relaxation did."""

    @pytest.mark.parametrize(
        ("spacing", "direction_sign"),
        [((1.0, 1.0, 1.0), 1.0), ((1.0, 1.5, 2.0), -1.0)],
    )
    def test_s1_2_trim_matches_frozen_base(
        self,
        contour_tools: ProcessSurfaceMesh,
        volume_tools: ProcessVolumetricMesh,
        spacing: tuple[float, float, float],
        direction_sign: float,
    ) -> None:
        """Default arguments, both direction handednesses."""
        mask = _ball_mask(spacing, direction_sign)
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = volume_tools.extract_label_tetrahedra_vtk(mask)[1]

        expected, _ = _frozen_base_trim(mesh, surface)
        relaxed = volume_tools.trim_tetrahedra_to_surface(mesh, surface)

        assert relaxed.n_cells == expected.n_cells
        assert np.array_equal(relaxed.cells, expected.cells)
        assert _point_rmse(relaxed, expected) < 0.1
        assert _min_scaled_jacobian(relaxed) >= 0.1

    def test_s1_2_damping_loop_iterates_more_than_once(
        self, contour_tools: ProcessSurfaceMesh, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """A tight bound and a long step force the back-off to repeat."""
        mask = _ball_mask((1.0, 1.0, 1.0), 1.0)
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = volume_tools.extract_label_tetrahedra_vtk(mask)[1]

        expected, passes = _frozen_base_trim(
            mesh, surface, relaxation=0.9, min_scaled_jacobian=0.4
        )
        relaxed = volume_tools.trim_tetrahedra_to_surface(
            mesh, surface, relaxation=0.9, min_scaled_jacobian=0.4
        )

        assert passes > 2, "the frozen base must damp more than once"
        assert relaxed.n_cells == expected.n_cells
        assert _point_rmse(relaxed, expected) < 0.1
        assert _min_scaled_jacobian(relaxed) >= 0.4

    def test_s1_2_zero_iterations_leave_points_alone(
        self, contour_tools: ProcessSurfaceMesh, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """No sweep means only the crinkle clip is applied."""
        mask = _ball_mask((1.0, 1.0, 1.0), 1.0)
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = volume_tools.extract_label_tetrahedra_vtk(mask)[1]

        expected, _ = _frozen_base_trim(mesh, surface, iterations=0)
        relaxed = volume_tools.trim_tetrahedra_to_surface(mesh, surface, iterations=0)

        assert np.array_equal(np.asarray(relaxed.points), np.asarray(expected.points))

    def test_s1_2_scaled_jacobian_matches_vtk(self) -> None:
        """The GPU quality equals VTK's on random, flat and inverted tetrahedra."""
        import torch

        rng = np.random.default_rng(0)
        points = rng.normal(size=(400, 3)) * 3.0
        connectivity = rng.integers(0, 400, size=(2000, 4))
        connectivity = connectivity[
            np.all(np.diff(np.sort(connectivity, axis=1), axis=1) > 0, axis=1)
        ]
        points[connectivity[0, 3]] = (
            points[connectivity[0, 0]]
            + points[connectivity[0, 1]]
            - points[connectivity[0, 2]]
        )
        cells = np.hstack([np.full((len(connectivity), 1), 4), connectivity]).ravel()
        grid = pv.UnstructuredGrid(
            cells, np.full(len(connectivity), pv.CellType.TETRA, np.uint8), points
        )
        expected = np.asarray(
            grid.cell_quality(["scaled_jacobian"]).cell_data["scaled_jacobian"]
        )

        actual = (
            ProcessMeshKernels()
            .scaled_jacobian(
                torch.as_tensor(points, device="cuda:0"),
                torch.as_tensor(connectivity, device="cuda:0"),
            )
            .cpu()
        )

        assert np.max(np.abs(actual.numpy() - expected)) < 1e-9

    def test_s1_2_closest_points_match_vtk(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """Warp's closest surface point equals the VTK locator's."""
        kernels = ProcessMeshKernels()

        surface = contour_tools.extract_label_surfaces(
            _ball_mask((1.0, 1.0, 1.0), 1.0)
        )[1]
        rng = np.random.default_rng(1)
        queries = rng.uniform(-2.0, 28.0, size=(500, 3))
        queries = queries + np.asarray(surface.center) - 13.0

        _, expected = cast(
            "tuple[np.ndarray, np.ndarray]",
            surface.find_closest_cell(queries, return_closest_point=True),
        )
        actual = kernels.closest_points_on_surface(
            kernels.build_surface_mesh(surface), queries
        )

        gap = np.linalg.norm(actual - queries, axis=1)
        assert np.max(np.abs(gap - np.linalg.norm(expected - queries, axis=1))) < 1e-4
        assert np.max(np.abs(actual - expected)) < 5e-3


def _itk_signed_distance(
    labels: np.ndarray, spacing: Sequence[float], label_id: int, sigma_mm: float
) -> np.ndarray:
    """Run ITK's signed Maurer map and recursive Gaussian blur on one label."""
    inside = labels == label_id
    mask = itk.GetImageFromArray(np.ascontiguousarray(inside.astype(np.uint8)))
    mask.SetSpacing([float(value) for value in spacing[::-1]])
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
    return np.asarray(itk.GetArrayFromImage(distance), dtype=np.float32)


def _itk_resample_labelmap_isotropic(
    labelmap_image: itk.Image,
    isotropic_spacing_mm: float,
    sigma_mm: Optional[np.ndarray] = None,
) -> itk.Image:
    """Run ITK's label-vote resample, which vote_resample_labelmap reproduces."""
    spacing = np.asarray(labelmap_image.GetSpacing(), dtype=np.float64)
    size = itk.size(labelmap_image)
    interpolator = itk.LabelImageGaussianInterpolateImageFunction.New(labelmap_image)
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


class _ItkKernels(ProcessMeshKernels):
    """ProcessMeshKernels with the pre-S2 ITK distance maps."""

    def signed_distance_mm(
        self,
        labels: torch.Tensor,
        spacing: Sequence[float],
        label_id: int,
        sigma_mm: float,
    ) -> torch.Tensor:
        return torch.as_tensor(
            _itk_signed_distance(labels.cpu().numpy(), spacing, label_id, sigma_mm),
            device=labels.device,
        )


def _itk_surface_tools() -> ProcessSurfaceMesh:
    """Return surface tools running on the pre-S2 ITK distance maps."""
    tools = ProcessSurfaceMesh()
    tools._kernels = _ItkKernels()
    return tools


def _itk_volume_tools() -> ProcessVolumetricMesh:
    """Return volume tools whose kernels run on ITK maps."""
    tools = ProcessVolumetricMesh()
    tools._kernels = _ItkKernels()
    return tools


def _blob_labelmap(
    spacing: tuple[float, float, float], direction_sign: float
) -> itk.Image:
    """Build three touching labels, plus a one-voxel label too small to survive."""
    zz, yy, xx = np.indices((24, 30, 34)).astype(np.float64)
    arr = np.zeros(zz.shape, dtype=np.uint16)
    arr[(xx - 11) ** 2 + (yy - 14) ** 2 + (zz - 12) ** 2 < 8.0**2] = 1
    arr[(xx - 22) ** 2 + (yy - 14) ** 2 + (zz - 12) ** 2 < 7.0**2] = 2
    arr[(xx - 16) ** 2 + (yy - 14) ** 2 + (zz - 12) ** 2 < 3.5**2] = 5
    arr[3, 3, 3] = 9
    labelmap = itk.GetImageFromArray(arr)
    labelmap.SetSpacing(list(spacing))
    labelmap.SetOrigin([5.0, -3.0, 2.0])
    labelmap.SetDirection(itk.matrix_from_array(np.diag([1.0, 1.0, direction_sign])))
    return labelmap


def _irregular_mask(
    spacing: tuple[float, float, float], direction_sign: float
) -> itk.Image:
    """Build a lumpy blob with no mirror symmetry for a vote to tie across."""
    noise = ndimage.gaussian_filter(
        np.random.default_rng(5).normal(size=(30, 34, 38)), 3.0
    )
    mask = itk.GetImageFromArray((noise > 0.04).astype(np.uint8))
    mask.SetSpacing(list(spacing))
    mask.SetDirection(itk.matrix_from_array(np.diag([1.0, 1.0, direction_sign])))
    return mask


def _surface_rmse(first: pv.PolyData, second: pv.PolyData) -> float:
    """Return the RMSE of each surface's points to the other, both pooled."""
    squared = []
    for source, target in ((first, second), (second, first)):
        points = np.asarray(source.points, dtype=np.float64)
        _, closest = cast(
            "tuple[np.ndarray, np.ndarray]",
            target.find_closest_cell(points, return_closest_point=True),
        )
        squared.append(np.sum((points - closest) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(squared))))


_S2_GEOMETRIES = [
    pytest.param((1.0, 1.0, 1.0), 1.0, id="isotropic-right-handed"),
    pytest.param((0.8, 1.0, 2.0), -1.0, id="anisotropic-left-handed"),
]


class TestS2GpuMatchesItk:
    """The GPU distance maps must do what the ITK calls did."""

    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("label_id", [1, 2, 5, 0])
    @pytest.mark.parametrize("sigma_mm", [0.0, 0.8])
    def test_s2_2_signed_distance_matches_itk(
        self, label_id: int, sigma_mm: float
    ) -> None:
        """Same sign, same metric, same blur kernel."""
        labelmap = _blob_labelmap((0.8, 1.0, 2.0), -1.0)
        fine = _itk_resample_labelmap_isotropic(labelmap, 0.8)
        labels = itk.GetArrayFromImage(fine)
        spacing = [0.8, 0.8, 0.8]

        expected = _itk_signed_distance(labels, spacing, label_id, sigma_mm)
        actual = (
            ProcessMeshKernels()
            .signed_distance_mm(
                torch.as_tensor(labels.astype(np.int32), device="cuda:0"),
                spacing,
                label_id,
                sigma_mm,
            )
            .cpu()
            .numpy()
        )

        assert np.max(np.abs(actual - expected)) < 1e-4

    @pytest.mark.requires_gpu
    def test_s2_2_squared_distance_matches_scipy(self) -> None:
        """The line kernel is the exact anisotropic Euclidean transform."""

        rng = np.random.default_rng(3)
        features = rng.random((12, 17, 23)) > 0.97
        spacing = (0.7, 1.1, 2.3)

        actual = (
            ProcessMeshKernels()
            .squared_distance_to_features(
                torch.as_tensor(features, device="cuda:0"), spacing
            )
            .cpu()
            .numpy()
        )
        expected = (
            np.asarray(ndimage.distance_transform_edt(~features, sampling=spacing)) ** 2
        )

        assert np.allclose(actual, expected, rtol=1e-5, atol=1e-5)
        empty = ProcessMeshKernels().squared_distance_to_features(
            torch.zeros((4, 5, 6), dtype=torch.bool, device="cuda:0"), spacing
        )
        assert bool(torch.all(empty > ProcessMeshKernels.NO_FEATURE_LIMIT))

    def test_crop_to_nonzero_keeps_voxel_positions(self) -> None:
        """The crop spans the nonzero bounding box and keeps physical positions."""
        rng = np.random.default_rng(4)
        occupied = np.zeros((9, 11, 13), dtype=np.uint8)
        occupied[2:7, 3:10, 4:5] = rng.random((5, 7, 1)) > 0.3
        occupied[3, 3, 12] = 1
        image = itk.GetImageFromArray(occupied)
        image.SetSpacing([0.5, 1.0, 2.0])
        image.SetOrigin([3.0, -1.0, 4.0])
        image.SetDirection(itk.matrix_from_array(np.diag([1.0, -1.0, 1.0])))

        cropped = ProcessImages().crop_to_nonzero(image, pad_voxels=2)

        extents = np.nonzero(occupied)
        start = [int(axis.min()) for axis in extents]
        stop = [int(axis.max()) + 1 for axis in extents]
        assert itk.GetArrayFromImage(cropped).shape == tuple(
            b - a + 4 for a, b in zip(start, stop)
        )
        first = image.TransformIndexToPhysicalPoint([start[2], start[1], start[0]])
        assert np.allclose(cropped.TransformIndexToPhysicalPoint([2, 2, 2]), first)

    @pytest.mark.requires_gpu
    @pytest.mark.parametrize(("spacing", "direction_sign"), _S2_GEOMETRIES)
    def test_s2_2_label_surfaces_match_itk_pipeline(
        self,
        contour_tools: ProcessSurfaceMesh,
        spacing: tuple[float, float, float],
        direction_sign: float,
    ) -> None:
        """Three labels give the same surfaces; the one-voxel label is dropped."""
        labelmap = _blob_labelmap(spacing, direction_sign)

        expected = _itk_surface_tools().extract_label_surfaces(labelmap)
        actual = contour_tools.extract_label_surfaces(labelmap)

        assert sorted(actual) == sorted(expected) == [1, 2, 5]
        for label_id, surface in actual.items():
            assert _surface_rmse(surface, expected[label_id]) < 0.1
            assert surface.is_all_triangles
            assert surface.volume > 0.0

    @pytest.mark.requires_gpu
    def test_s2_2_label_surfaces_with_options_match_itk_pipeline(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """Non-default grid, blur and smoothing, as the other callers pass them."""
        labelmap = _blob_labelmap((0.8, 1.0, 2.0), 1.0)

        expected = _itk_surface_tools().extract_label_surfaces(
            labelmap,
            isotropic_spacing_mm=1.0,
            distance_sigma_mm=1.2,
            smoothing_iterations=5,
        )
        actual = contour_tools.extract_label_surfaces(
            labelmap,
            isotropic_spacing_mm=1.0,
            distance_sigma_mm=1.2,
            smoothing_iterations=5,
        )

        assert sorted(actual) == sorted(expected)
        for label_id, surface in actual.items():
            assert _surface_rmse(surface, expected[label_id]) < 0.1

    @pytest.mark.requires_gpu
    def test_s2_2_empty_labelmap_has_no_surface(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """A labelmap with no label returns an empty mapping."""
        labelmap = itk.GetImageFromArray(np.zeros((5, 6, 7), dtype=np.uint16))

        assert contour_tools.extract_label_surfaces(labelmap) == {}

    @pytest.mark.requires_gpu
    @pytest.mark.parametrize(("spacing", "direction_sign"), _S2_GEOMETRIES)
    def test_s2_2_tetrahedra_match_itk_pipeline(
        self,
        volume_tools: ProcessVolumetricMesh,
        spacing: tuple[float, float, float],
        direction_sign: float,
    ) -> None:
        """A coarser element size votes to the same tetrahedral mesh."""
        mask = _irregular_mask(spacing, direction_sign)

        expected = _itk_volume_tools().extract_label_tetrahedra_vtk(
            mask, element_size_mm=1.5
        )[1]
        actual = volume_tools.extract_label_tetrahedra_vtk(mask, element_size_mm=1.5)[1]

        assert abs(actual.n_cells - expected.n_cells) <= 0.01 * expected.n_cells
        assert set(actual.celltypes) == {int(pv.CellType.TETRA)}
        assert (
            _surface_rmse(
                actual.extract_surface(algorithm="dataset_surface"),
                expected.extract_surface(algorithm="dataset_surface"),
            )
            < 0.1
        )

    @pytest.mark.requires_gpu
    def test_s2_2_tetrahedra_too_coarse_are_empty(
        self, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """An element larger than the structure drops it, as before."""
        arr = np.zeros((26, 26, 26), dtype=np.uint8)
        for step in range(10):
            arr[step + 5, step + 5, step + 5] = 1
        thin_line = itk.GetImageFromArray(arr)

        expected = _itk_volume_tools().extract_label_tetrahedra_vtk(
            thin_line, element_size_mm=8.0
        )
        actual = volume_tools.extract_label_tetrahedra_vtk(
            thin_line, element_size_mm=8.0
        )

        assert actual == expected == {}

    @pytest.mark.requires_gpu
    def test_s2_2_closest_point_distance_matches_implicit_distance(
        self, contour_tools: ProcessSurfaceMesh, volume_tools: ProcessVolumetricMesh
    ) -> None:
        """The mean surface-to-mesh distance equals VTK's implicit distance."""
        kernels = ProcessMeshKernels()

        mask = _ball_mask((1.0, 1.0, 1.0), 1.0)
        surface = contour_tools.extract_label_surfaces(mask)[1]
        mesh = volume_tools.extract_label_tetrahedra_vtk(mask, element_size_mm=1.5)[1]
        boundary = mesh.extract_surface(algorithm="dataset_surface")

        expected = float(
            np.abs(
                surface.copy().compute_implicit_distance(boundary)["implicit_distance"]
            ).mean()
        )
        points = np.asarray(surface.points, dtype=np.float64)
        closest = kernels.closest_points_on_surface(
            kernels.build_surface_mesh(boundary), points
        )
        actual = float(np.linalg.norm(closest - points, axis=1).mean())

        assert actual == pytest.approx(expected, abs=1e-4)


@pytest.mark.requires_gpu
class TestS2UnsmoothedMatchesItk:
    """Without a blur, an edge voxel's side of zero decides the contour."""

    @pytest.mark.parametrize("label_id", [1, 2, 0])
    def test_s2_2_edge_voxels_sit_on_the_same_side_of_zero_as_itk(
        self, label_id: int
    ) -> None:
        """Exact zeros and rounding-error depths fall on the same voxels."""
        labels = itk.GetArrayFromImage(_blob_labelmap((0.4, 0.4, 0.4), 1.0))
        spacing = [0.4, 0.4, 0.4]

        expected = _itk_signed_distance(labels, spacing, label_id, 0.0)
        actual = (
            ProcessMeshKernels()
            .signed_distance_mm(
                torch.as_tensor(labels.astype(np.int32), device="cuda:0"),
                spacing,
                label_id,
                0.0,
            )
            .cpu()
            .numpy()
        )

        assert np.array_equal(expected == 0.0, actual == 0.0)
        assert np.array_equal(expected < 0.0, actual < 0.0)
        assert np.max(np.abs(actual - expected)) < 1e-5

    def test_s2_2_unsmoothed_label_surfaces_match_itk_pipeline(
        self, contour_tools: ProcessSurfaceMesh
    ) -> None:
        """No blur and no smoothing leaves the contour at the labels' own edges."""
        labelmap = _blob_labelmap((0.4, 0.4, 1.2), 1.0)

        expected = _itk_surface_tools().extract_label_surfaces(
            labelmap, distance_sigma_mm=0.0, smoothing_iterations=0
        )
        actual = contour_tools.extract_label_surfaces(
            labelmap, distance_sigma_mm=0.0, smoothing_iterations=0
        )

        assert sorted(actual) == sorted(expected) == [1, 2, 5]
        for label_id, surface in actual.items():
            assert _surface_rmse(surface, expected[label_id]) < 0.01

    @pytest.mark.parametrize(("spacing", "direction_sign"), _S2_GEOMETRIES)
    def test_s2_2_tetrahedra_keep_the_itk_vote_voxels(
        self,
        volume_tools: ProcessVolumetricMesh,
        spacing: tuple[float, float, float],
        direction_sign: float,
    ) -> None:
        """Tetrahedra fill exactly the voxels ITK's vote keeps, ties included."""
        mask = _irregular_mask(spacing, direction_sign)

        mesh = volume_tools.extract_label_tetrahedra_vtk(mask, element_size_mm=1.5)[1]

        cropped = ProcessImages().crop_to_nonzero(mask)
        voted = _itk_resample_labelmap_isotropic(
            cropped, 1.5, np.maximum(np.asarray(spacing), 1.5)
        )
        assert mesh.n_cells == 6 * int(np.count_nonzero(itk.GetArrayFromImage(voted)))


@pytest.mark.requires_gpu
@pytest.mark.parametrize(("spacing", "direction_sign"), _S2_GEOMETRIES)
@pytest.mark.parametrize("isotropic_spacing_mm", [0.6, 1.5])
def test_s3_1_vote_resample_matches_itk_but_for_exact_ties(
    spacing: tuple[float, float, float],
    direction_sign: float,
    isotropic_spacing_mm: float,
) -> None:
    """The GPU vote keeps ITK's grid and its labels, bar a few exact ties."""
    labelmap = _blob_labelmap(spacing, direction_sign)

    expected = _itk_resample_labelmap_isotropic(labelmap, isotropic_spacing_mm)
    actual = ProcessMeshKernels().vote_resample_labelmap(labelmap, isotropic_spacing_mm)

    assert itk.size(actual) == itk.size(expected)
    assert tuple(actual.GetSpacing()) == tuple(expected.GetSpacing())
    assert tuple(actual.GetOrigin()) == tuple(expected.GetOrigin())
    assert np.array_equal(
        itk.array_from_matrix(actual.GetDirection()),
        itk.array_from_matrix(expected.GetDirection()),
    )
    expected_arr = itk.GetArrayViewFromImage(expected)
    actual_arr = itk.GetArrayViewFromImage(actual)
    assert actual_arr.dtype == expected_arr.dtype
    assert np.mean(actual_arr != expected_arr) < 1e-3


@pytest.mark.requires_gpu
def test_s3_1_vote_resample_with_a_wide_kernel_matches_itk() -> None:
    """A per-axis kernel wider than the voxels, as the tetrahedra use."""
    mask = _irregular_mask((0.8, 1.0, 2.0), -1.0)
    sigma = np.maximum(np.asarray(mask.GetSpacing()), 1.5)

    expected = _itk_resample_labelmap_isotropic(mask, 1.5, sigma)
    actual = ProcessMeshKernels().vote_resample_labelmap(mask, 1.5, sigma)

    expected_arr = itk.GetArrayViewFromImage(expected)
    actual_arr = itk.GetArrayViewFromImage(actual)
    assert actual_arr.shape == expected_arr.shape
    assert np.mean(actual_arr != expected_arr) < 1e-3


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ("spacing", "direction_sign"),
    [((1.0, 1.0, 1.0), 1.0), ((1.0, 1.5, 2.0), -1.0)],
)
def test_s4_1_gpu_crinkle_matches_vtk_clip(
    contour_tools: ProcessSurfaceMesh,
    volume_tools: ProcessVolumetricMesh,
    spacing: tuple[float, float, float],
    direction_sign: float,
) -> None:
    """The GPU cell selection keeps the cells, arrays and scalars VTK's does."""
    mask = _ball_mask(spacing, direction_sign)
    surface = contour_tools.extract_label_surfaces(mask)[1]
    mesh = volume_tools.extract_label_tetrahedra_vtk(mask)[1]

    expected = mesh.clip_surface(surface, invert=True, crinkle=True)
    relaxed = volume_tools.trim_tetrahedra_to_surface(mesh, surface, iterations=0)

    assert 0 < relaxed.n_cells < mesh.n_cells
    assert np.array_equal(relaxed.cells, expected.cells)
    assert np.array_equal(relaxed.points, expected.points)
    assert relaxed.cell_data.keys() == expected.cell_data.keys()
    assert np.array_equal(relaxed.cell_data["cell_ids"], expected.cell_data["cell_ids"])
    assert relaxed.point_data.keys() == expected.point_data.keys()
    assert relaxed.active_scalars_info.name == expected.active_scalars_info.name


@pytest.mark.requires_gpu
def test_s4_1_gpu_inside_test_matches_vtk_distance_sign(
    contour_tools: ProcessSurfaceMesh,
) -> None:
    """Points off the surface are inside exactly where VTK's distance is negative."""
    kernels = ProcessMeshKernels()

    surface = contour_tools.extract_label_surfaces(_ball_mask((1.0, 1.0, 1.0), 1.0))[1]
    rng = np.random.default_rng(2)
    queries = rng.uniform(-3.0, 29.0, size=(4000, 3)) + np.asarray(surface.center) - 13
    distance = np.asarray(
        pv.PolyData(queries).compute_implicit_distance(surface)["implicit_distance"]
    )
    away = np.abs(distance) > 1e-3

    inside = kernels.points_inside_surface(kernels.build_surface_mesh(surface), queries)

    assert away.sum() > 3000
    assert 0 < inside.sum() < len(queries)
    assert np.array_equal(inside[away], distance[away] < 0.0)


@pytest.mark.requires_gpu
def test_s4_1_trim_keeps_the_empty_result_when_the_surface_misses(
    contour_tools: ProcessSurfaceMesh,
    volume_tools: ProcessVolumetricMesh,
) -> None:
    """A surface that holds no point of the mesh trims every cell away."""
    mask = _ball_mask((1.0, 1.0, 1.0), 1.0)
    surface = contour_tools.extract_label_surfaces(mask)[1]
    mesh = volume_tools.extract_label_tetrahedra_vtk(mask)[1]
    far = surface.translate((500.0, 0.0, 0.0), inplace=False)

    relaxed = volume_tools.trim_tetrahedra_to_surface(mesh, far)

    assert relaxed.n_cells == 0


@pytest.mark.requires_gpu
def test_s5_1_trim_keeps_vtk_cells_on_a_thin_tube(
    volume_tools: ProcessVolumetricMesh,
) -> None:
    """A thin tube, far from the origin, is trimmed to the cells VTK keeps."""
    kernels = ProcessMeshKernels()

    angle = np.linspace(0.0, 2.5, 40)
    spine = np.column_stack([3.0 * np.cos(angle), 3.0 * np.sin(angle), 0.4 * angle])
    tube = (
        pv.Spline(spine + (-20.0, 35.0, 1800.0), 120)
        .tube(radius=0.15, n_sides=7, capping=True)
        .triangulate()
        .clean()
    )
    assert tube.n_open_edges == 0
    grid = pv.ImageData(
        dimensions=(30, 30, 20),
        spacing=(0.17, 0.17, 0.11),
        origin=np.asarray(tube.bounds[::2]) - 0.2,
    ).triangulate()

    expected = grid.clip_surface(tube, invert=True, crinkle=True)
    function = vtk.vtkImplicitPolyDataDistance()
    function.SetInput(tube)
    values = np.array([function.EvaluateFunction(point) for point in grid.points])
    inside = kernels.points_inside_surface(
        kernels.build_surface_mesh(tube), grid.points
    )
    relaxed = volume_tools.trim_tetrahedra_to_surface(grid, tube, iterations=0)

    assert 0 < expected.n_cells < grid.n_cells
    assert np.array_equal(inside, values < 0.0)
    assert np.array_equal(relaxed.cells, expected.cells)
    assert np.array_equal(relaxed.cell_data["cell_ids"], expected.cell_data["cell_ids"])


@pytest.mark.requires_gpu
def test_tutorial_4_frame_writes_every_output(tmp_path: Path) -> None:
    """One Tutorial 4 (Duke Heart) frame writes its surfaces and meshes."""
    tutorials = Path(__file__).resolve().parents[1] / "tutorials"
    if str(tutorials) not in sys.path:
        sys.path.insert(0, str(tutorials))
    tutorial = importlib.import_module(
        "tutorial_04_convert_to_vtk_duke_heart_4d_labelmaps"
    )
    labelmap_file = tmp_path / "pm0001_g000_labelmap.nii.gz"
    itk.imwrite(_blob_labelmap((1.0, 1.0, 1.0), 1.0), str(labelmap_file))

    displacements = tutorial.convert_frame(
        labelmap_file, tmp_path, full=True, log_level=logging.WARNING
    )

    written = sorted(path.name for path in tmp_path.glob("pm0001_g000_*.vt?"))
    assert "pm0001_g000_heart_minus_interior_chambers.vtp" in written
    assert "pm0001_g000_heart_minus_interior_chambers.vtu" in written
    assert "pm0001_g000_surfaces.vtp" in written
    assert len(displacements) == len([n for n in written if n.endswith(".vtu")])
    assert all(0.0 <= value < 1.0 for value in displacements)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
