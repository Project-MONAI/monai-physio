"""Extract the rigid aortic trunk centreline and section areas for 1D flow.

The Tutorial 13 aorta is frozen at sample 69. A radius-weighted interior
geodesic joins the centroids of the root and distal cuts that
``aorta_lumen.py`` also makes for the 3D experiment, and each section area is
the closed plane section of the source surface normal to the curve.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from aorta_lumen import AortaLumen
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]


def section(
    surface: pv.PolyData, origin: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the closed plane section nearest ``origin`` and its area."""
    cut = surface.slice(normal=normal, origin=origin).strip()
    lines = np.asarray(cut.lines, dtype=np.int64)
    loops: list[np.ndarray] = []
    index = 0
    while index < len(lines):
        count = int(lines[index])
        ids = lines[index + 1 : index + 1 + count]
        index += count + 1
        if count >= 4 and ids[0] == ids[-1]:
            loops.append(np.asarray(cut.points[ids[:-1]], dtype=np.float64))
    if not loops:
        raise ValueError(f"No closed source section at {origin.tolist()}")
    ring = min(loops, key=lambda p: float(np.linalg.norm(p.mean(axis=0) - origin)))
    centered = ring - ring.mean(axis=0)
    area = abs(np.cross(centered, np.roll(centered, -1, axis=0)).sum(0) @ normal) / 2
    return ring, float(area)


def medial_path(
    surface: pv.PolyData, start: np.ndarray, end: np.ndarray, spacing: float
) -> np.ndarray:
    """Trace a radius-weighted interior voxel geodesic between two points."""
    image = surface.voxelize_binary_mask(spacing=spacing)
    dimensions = tuple(int(n) for n in image.dimensions)
    mask = np.asarray(image.active_scalars).reshape(dimensions, order="F") > 0
    radius = distance_transform_edt(mask, sampling=image.spacing)
    inside = np.argwhere(mask)
    world = np.asarray(image.origin) + inside * np.asarray(image.spacing)
    endpoints = inside[np.asarray(cKDTree(world).query(np.stack((start, end)))[1])]
    costs = np.full(dimensions, np.inf)
    costs[mask] = 1.0 / np.maximum(radius[mask], spacing / 2.0) ** 2
    search = MCP_Geometric(costs, fully_connected=True, sampling=image.spacing)
    search.find_costs([tuple(endpoints[0])], [tuple(endpoints[1])])
    indices = np.asarray(search.traceback(tuple(endpoints[1])))
    path = np.asarray(image.origin) + indices * np.asarray(image.spacing)
    # Smoothing removes voxel stair steps; check_inside rejects wall exits.
    path = gaussian_filter1d(path, sigma=2.0, axis=0, mode="nearest")
    path[0], path[-1] = start, end
    return np.asarray(path)


def resample(path: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline uniformly by arc length."""
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-12]
    path, distance = path[keep], distance[keep]
    samples = np.linspace(0.0, distance[-1], int(np.ceil(distance[-1] / spacing)) + 1)
    curve = np.column_stack(
        [np.interp(samples, distance, path[:, i]) for i in range(3)]
    )
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    return curve, arc


def check_inside(surface: pv.PolyData, curve: np.ndarray) -> None:
    """Reject a curve whose dense interpolation leaves the lumen."""
    dense, _ = resample(curve, 0.00025)
    selected = pv.PolyData(dense).select_interior_points(surface)
    outside = int((np.asarray(selected["selected_points"]) == 0).sum())
    if outside:
        raise ValueError(f"Centreline leaves the lumen at {outside} points")


def prepare(source_path: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Write the frozen anatomy, trunk geometry and LV interval flux."""
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    tools = AortaLumen()
    source = tools.load_source(
        source_path, config["source_sha256"], config["frozen_sample"]
    )
    ports = tools.prepare_boundaries(source, config["distal_margin_m"]).ports
    root, distal = ports["root_inlet"], ports["distal_crop_outlet"]
    surface = pv.PolyData(
        source.aorta[source.frame],
        np.c_[np.full(len(source.faces), 3), source.faces].ravel(),
    )
    surface.save(output / "source.vtp")
    start, end = np.asarray(root["center_m"]), np.asarray(distal["center_m"])
    path = medial_path(surface, start, end, config["voxel_spacing_m"])
    curve, arc = resample(path, config["centerline_spacing_m"])
    check_inside(surface, curve)
    tangents = np.gradient(curve, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    rings, areas = zip(*(section(surface, c, t) for c, t in zip(curve, tangents)))
    offsets = np.r_[0, np.cumsum([len(ring) for ring in rings])]
    inlet = tools.infer_inlet(source)
    np.savez_compressed(
        output / "geometry.npz",
        centerline_m=curve,
        arc_length_m=arc,
        area_m2=np.asarray(areas),
        tangent=tangents,
        contour_points_m=np.concatenate(rings),
        contour_offsets=offsets,
    )
    np.savez_compressed(
        output / "waveform.npz",
        times_s=source.seconds,
        interval_end_times_s=source.seconds + tools.SAMPLE_INTERVAL_S,
        lv_volume_m3=np.asarray(inlet["lv_volume_m3"]),
        inlet_interval_flux_m3_s=np.asarray(inlet["inlet_interval_flux_m3_s"]),
        ejection_gate=np.asarray(inlet["ejection_gate"]),
    )
    report = {
        "source": str(source.path),
        "source_sha256": source.sha256,
        "frozen_sample": source.frame,
        "distal_crop_y_m": float(source.aorta[source.frame][:, 1].min()),
        "root_plane": {k: root[k] for k in ("center_m", "plane_normal")},
        "distal_plane": {k: distal[k] for k in ("center_m", "plane_normal")},
        "centerline_nodes": len(curve),
        "length_m": float(arc[-1]),
        "area_range_mm2": [float(min(areas) * 1e6), float(max(areas) * 1e6)],
        "stroke_volumes_ml": [v * 1e6 for v in inlet["stroke_volumes_m3"]],
        "config": config,
        "preparation_seconds": time.perf_counter() - started,
    }
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    """Prepare the trunk from the configured or given source USD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--source", type=Path, help="Tutorial 13 USD path")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "prepared")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.source or REPOSITORY / config["source"]
    report = prepare(source, args.output.resolve(), config)
    keys = ("centerline_nodes", "length_m", "area_range_mm2", "stroke_volumes_ml")
    print(json.dumps({key: report[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
