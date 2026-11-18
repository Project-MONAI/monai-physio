"""Prepare the rigid aortic lumen mesh for 3D flow.

The Tutorial 13 aorta is frozen at sample 69, cut into a five-port lumen by
``../BloodFlow-1D-Rigid-Warp/aorta_lumen.py``, voxelized and meshed with
Netgen, with finer elements around the narrow ports. Boundary triangles are
labelled 0 (wall) or by the region id of their port. The inlet waveform is
written by the 1D experiment's ``prepare.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import itk
import numpy as np
import pyvista as pv

from monai_physio.process_volumetric_mesh import ProcessVolumetricMesh

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
# The Tutorial 13 lumen preparation is shared with the 1D experiment.
sys.path.insert(0, str(ROOT.parent / "BloodFlow-1D-Rigid-Warp"))
from aorta_lumen import AortaLumen


def outer_faces(tets: np.ndarray) -> np.ndarray:
    """Return the outward oriented boundary triangles of positive tetrahedra."""
    faces = np.concatenate(
        [tets[:, p] for p in ([1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1])]
    )
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    return np.asarray(faces[counts[inverse] == 1])


def six_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return six times the signed volume of every tetrahedron."""
    q = points[tets]
    return np.asarray(
        np.einsum(
            "ij,ij->i",
            q[:, 1] - q[:, 0],
            np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]),
        )
    )


# USD (x, y, z) metres with Y up maps to LPS (x, -z, y) millimetres.
USD_TO_LPS_MM = np.array([1000.0, -1000.0, 1000.0])


def port_element_sizes(
    caps: dict[str, pv.PolyData], ports: dict[str, Any], config: dict[str, Any]
) -> tuple[Optional[itk.Image], dict[str, float]]:
    """Return element sizes that put ``port_cells_across`` elements across ports.

    A port whose diameter / ``port_cells_across`` is under 90% of
    ``element_size_mm`` gets elements of that size within ``port_zone_radii``
    of its radius from its centre, sampled every ``port_size_grid_mm`` in the
    LPS millimetres the lumen is meshed in; where zones overlap, the finer
    size wins.  Without that, a narrow opening is tiled by one or two
    triangles whose no-slip rim pins every velocity node, so it carries
    little or no flow.
    """
    zones = {}
    for name, cap in caps.items():
        radius = float(np.sqrt(cap.area * 1e6 / np.pi))
        size = 2 * radius / config["port_cells_across"]
        if size < 0.9 * config["element_size_mm"]:
            center = np.asarray(ports[name]["center_m"])[[0, 2, 1]] * USD_TO_LPS_MM
            zones[name] = (center, config["port_zone_radii"] * radius, size)
    # Coarse zones first, so a finer zone overwrites any overlap.
    zones = dict(sorted(zones.items(), key=lambda item: -item[1][2]))
    if not zones:
        return None, {}
    spacing = config["port_size_grid_mm"]
    low = np.min([c - r for c, r, _ in zones.values()], axis=0)
    high = np.max([c + r for c, r, _ in zones.values()], axis=0)
    axes = [
        low[a] + spacing * np.arange(int((high - low)[a] / spacing) + 2)
        for a in range(3)
    ]
    labels = np.zeros([len(axes[2]), len(axes[1]), len(axes[0])], dtype=np.uint8)
    for label, (center, zone_radius, _) in enumerate(zones.values(), 1):
        distance2 = (
            (axes[0] - center[0])[None, None, :] ** 2
            + (axes[1] - center[1])[None, :, None] ** 2
            + (axes[2] - center[2])[:, None, None] ** 2
        )
        labels[distance2 <= zone_radius**2] = label
    zone_image = itk.GetImageFromArray(labels)
    zone_image.SetOrigin(low)
    zone_image.SetSpacing([spacing] * 3)
    size_image = ProcessVolumetricMesh.create_element_size_image(
        zone_image, {label: zone[2] for label, zone in enumerate(zones.values(), 1)}
    )
    return size_image, {name: zone[2] for name, zone in zones.items()}


def mesh_lumen(
    surface: pv.PolyData, config: dict[str, Any], size_image: Optional[itk.Image]
) -> pv.UnstructuredGrid:
    """Voxelize the closed lumen in LPS millimetres and fill it with Netgen."""
    lps = surface.copy()
    lps.points = np.asarray(surface.points)[:, [0, 2, 1]] * USD_TO_LPS_MM
    voxels = lps.voxelize_binary_mask(spacing=config["voxel_spacing_mm"])
    dimensions = tuple(int(v) for v in voxels.dimensions)
    array = np.asarray(voxels.active_scalars).reshape(dimensions, order="F")
    mask = itk.GetImageFromArray((array.transpose(2, 1, 0) > 0).astype(np.uint8))
    mask.SetOrigin(voxels.origin)
    mask.SetSpacing(voxels.spacing)
    meshes = ProcessVolumetricMesh().extract_label_tetrahedra_netgen(
        mask,
        element_size_mm=config["element_size_mm"],
        contour_spacing_mm=config["contour_spacing_mm"],
        smoothing_sigma_mm=config["smoothing_sigma_mm"],
        smoothing_iterations=config["smoothing_iterations"],
        shared_interfaces=False,
        label_ids=[1],
        element_size_image=size_image,
    )
    return meshes[1]


def prepare(source_path: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Write the frozen anatomy and the tagged tetrahedral lumen."""
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    tools = AortaLumen()
    source = tools.load_source(source_path, config["frozen_sample"])
    boundary = tools.prepare_boundaries(source, config["distal_margin_m"])
    surface = boundary.surface()
    caps = {
        name: surface.extract_cells(
            np.flatnonzero(boundary.regions == port["region_id"]).tolist()
        ).extract_surface()
        for name, port in boundary.ports.items()
    }
    size_image, port_sizes = port_element_sizes(caps, boundary.ports, config)

    grid = mesh_lumen(surface, config, size_image)
    if not np.all(grid.celltypes == pv.CellType.TETRA):
        raise ValueError("Netgen returned non-tetrahedral cells")
    lps = np.asarray(grid.points, dtype=np.float64)
    points = lps[:, [0, 2, 1]] * [0.001, 0.001, -0.001]
    tets = np.array(grid.cells.reshape(-1, 5)[:, 1:], dtype=np.int64)
    negative = six_volumes(points, tets) < 0
    tets[negative] = tets[negative][:, [1, 0, 2, 3]]
    if np.any(six_volumes(points, tets) <= 0):
        raise ValueError("Degenerate tetrahedra")
    faces = outer_faces(tets)

    # A mesh face joins a port when its nearest source triangle belongs to
    # that cap, it lies within 0.8 mm of the cap plane and faces the same way.
    centroids = points[faces].mean(axis=1)
    nearest = boundary.regions[surface.find_closest_cell(centroids)]
    normals = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    areas = np.linalg.norm(normals, axis=1) / 2
    normals /= 2 * areas[:, None]
    regions = np.zeros(len(faces), dtype=np.int64)
    ports: dict[str, Any] = {}
    for name, port in boundary.ports.items():
        rid = port["region_id"]
        cap = caps[name]
        normal = np.asarray(cap.compute_normals(cell_normals=True).cell_normals)
        normal = normal.mean(axis=0) / np.linalg.norm(normal.mean(axis=0))
        origin = np.asarray(cap.points).mean(axis=0)
        on_plane = np.abs((centroids - origin) @ normal) < 0.0008
        selected = (nearest == rid) & on_plane & (normals @ normal > 0.5)
        regions[selected] = rid
        ports[name] = {
            "region_id": rid,
            "role": port["role"],
            "center_m": port["center_m"],
            "normal": normal.tolist(),
            "mesh_area_m2": float(areas[selected].sum()),
            "source_area_m2": float(cap.area),
            "mesh_faces": int(selected.sum()),
            "element_size_mm": port_sizes.get(name, config["element_size_mm"]),
        }
    # Write only after meshing succeeds, so a failure leaves any earlier
    # preparation whole.
    surface.save(output / "computational_surface.vtp")
    frozen = source.aorta[source.frame]
    pv.PolyData(frozen, np.c_[np.full(len(source.faces), 3), source.faces]).save(
        output / "source.vtp"
    )
    np.savez_compressed(
        output / "geometry.npz",
        points_m=points,
        tetrahedra=tets,
        boundary_faces=faces,
        boundary_regions=regions,
    )
    (output / "ports.json").write_text(json.dumps(ports, indent=2), encoding="utf-8")
    skin = pv.PolyData(points, np.c_[np.full(len(faces), 3), faces])
    skin.cell_data["region"] = regions
    skin.save(output / "boundary.vtp")
    lumen_volume = tools.signed_surface_volume(boundary.points, boundary.faces)
    report = {
        "source": str(source.path),
        "frozen_sample": source.frame,
        "distal_crop_y_m": float(frozen[:, 1].min()),
        "points": len(points),
        "tetrahedra": len(tets),
        "mesh_volume_relative_error": float(
            six_volumes(points, tets).sum() / 6 / lumen_volume - 1
        ),
        "ports": ports,
        "config": config,
        "preparation_seconds": time.perf_counter() - started,
    }
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    """Prepare the 3D lumen from the configured or given source USD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--source", type=Path, help="Tutorial 13 USD path")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "prepared")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.source or REPOSITORY / config["source"]
    report = prepare(source, args.output.resolve(), config)
    summary = {key: report[key] for key in ("points", "tetrahedra", "ports")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
