"""Prepare the candidate-20 blood and wall meshes for the FSI demonstration.

The Simpleware candidate-20 ``geometry.npz`` holds one point array shared by
a blood (fluid) and a wall (solid) tetrahedral mesh that meet on exactly
shared triangles. Only that file and ``waveform.npz`` are read; the stale
per-mesh fields of its ``ports.json`` are not. Both meshes are compacted to
their own vertices, ports are recomputed from the face regions, and each port
gets a clamped wall collar: the wall vertices at or beyond the cap plane,
within ``collar_margin_m`` of the rim, that connect to it (the wall overhangs
the caps by up to about 2 mm).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]


def compact(tetrahedra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the used point ids and the tetrahedra renumbered over them."""
    ids, inverse = np.unique(tetrahedra, return_inverse=True)
    return ids, inverse.reshape(tetrahedra.shape)


def collar(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    rim: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    margin: float,
) -> np.ndarray:
    """Wall vertices beyond the cap plane near the rim and connected to it.

    A vertex qualifies when its cap-plane offset is at least the rim's lowest
    offset and at most ``margin`` beyond its highest, and its distance from
    the port axis exceeds the rim radius by at most ``margin``.
    """
    relative = points - center
    offset = relative @ normal
    lateral = np.linalg.norm(relative - offset[:, None] * normal, axis=1)
    beyond = (
        (offset >= offset[rim].min())
        & (offset <= offset[rim].max() + margin)
        & (lateral <= lateral[rim].max() + margin)
    )
    pairs = tetrahedra[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]]
    pairs = pairs.reshape(-1, 2)
    pairs = pairs[beyond[pairs].all(axis=1)]
    graph = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(len(points), len(points)),
    )
    labels = csgraph.connected_components(graph, directed=False)[1]
    return np.asarray(beyond & np.isin(labels, labels[rim]))


def prepare(source: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Write compact meshes, port data, wall clamps and the inlet waveform."""
    data = np.load(source / "geometry.npz")
    points = data["points_m"]
    fluid_ids, fluid_tets = compact(data["fluid_tetrahedra"])
    solid_ids, solid_tets = compact(data["solid_tetrahedra"])
    to_fluid = np.full(len(points), -1)
    to_fluid[fluid_ids] = np.arange(len(fluid_ids))
    to_solid = np.full(len(points), -1)
    to_solid[solid_ids] = np.arange(len(solid_ids))
    faces = data["fluid_boundary_faces"]
    regions = np.maximum(data["fluid_boundary_regions"], 0)  # Wall -1 becomes 0.
    shared = np.zeros(len(data["solid_boundary_faces"]), dtype=bool)
    shared[data["shared_face_pairs"][:, 1]] = True
    outer = to_solid[data["solid_boundary_faces"][~shared]]

    clamped = np.zeros(len(solid_ids), dtype=bool)
    ports = {}
    for name, region in config["ports"].items():
        selected = faces[regions == region]
        corners = points[selected]
        area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        total = area.sum(axis=0) / 2
        normal = total / np.linalg.norm(total)
        center = corners.reshape(-1, 3).mean(axis=0)
        rim = to_solid[np.intersect1d(selected, solid_ids)]
        held = collar(
            points[solid_ids],
            solid_tets,
            rim,
            center,
            normal,
            config["collar_margin_m"],
        )
        clamped |= held
        depth = (points[solid_ids][held] - center) @ normal
        ports[name] = {
            "region_id": region,
            "role": "inlet" if name == config["inlet"] else "outlet",
            "center_m": center.tolist(),
            "normal": normal.tolist(),
            "area_m2": float(np.linalg.norm(total)),
            "faces": len(selected),
            "rim_vertices": len(rim),
            "clamped_wall_vertices": int(held.sum()),
            "collar_depth_m": float(depth.max()),
        }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "mesh.npz",
        fluid_points_m=points[fluid_ids],
        fluid_tetrahedra=fluid_tets,
        fluid_boundary_faces=to_fluid[faces],
        fluid_boundary_regions=regions,
        solid_points_m=points[solid_ids],
        solid_tetrahedra=solid_tets,
        solid_clamped=clamped,
        solid_outer_faces=outer,
    )
    shutil.copyfile(source / "waveform.npz", output / "waveform.npz")
    (output / "ports.json").write_text(json.dumps(ports, indent=2), encoding="utf-8")
    report = {
        "source": str(source),
        "fluid_vertices": len(fluid_ids),
        "fluid_tetrahedra": len(fluid_tets),
        "solid_vertices": len(solid_ids),
        "solid_tetrahedra": len(solid_tets),
        "interface_vertices": len(np.intersect1d(fluid_ids, solid_ids)),
        "outer_wall_faces": len(outer),
        "clamped_wall_vertices": int(clamped.sum()),
        "ports": ports,
    }
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    """Prepare the meshes from the configured or given candidate-20 folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--source", type=Path, help="Candidate-20 folder")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "prepared")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.source or REPOSITORY / config["source"]
    print(json.dumps(prepare(source, args.output.resolve(), config), indent=2))


if __name__ == "__main__":
    main()
