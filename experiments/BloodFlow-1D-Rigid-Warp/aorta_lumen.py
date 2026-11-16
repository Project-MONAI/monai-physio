"""Tutorial 13 aortic lumen and LV interval flux for the blood-flow experiments.

The five-port lumen is a computational truncation of the Tutorial 13 aorta: a
root inlet (not a valve), three arch-branch stump outlets and a distal outlet
above the CT crop. Nothing here connects missing vessels or infers pressure.
The prim paths, landmarks and timing are specific to the Tutorial 13 USD, so
this module lives with the experiments (``prepare.py`` here and in
``../BloodFlow-3D-Rigid-Warp``) rather than in the library. The smoothed
inlet flow is :func:`monai_physio.warp_navier_stokes.inlet_flow`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import pyvista as pv
from pxr import Usd, UsdGeom
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from monai_physio.process_usd import ProcessUSD


@dataclass
class AortaSource:
    """Source samples in the USD world frame (metres, Y-up).

    ``aorta`` and ``lv`` hold every time sample; ``branches`` are read at the
    frozen sample ``frame`` that the computational boundary uses.
    """

    path: Path
    sha256: str
    times: np.ndarray
    seconds: np.ndarray
    aorta: np.ndarray
    faces: np.ndarray
    lv: np.ndarray
    lv_faces: np.ndarray
    branches: dict[str, pv.PolyData]
    metadata: dict[str, Any]
    frame: int = 69


@dataclass
class AortaBoundary:
    """Closed computational boundary with sparse source-vertex provenance."""

    points: np.ndarray
    faces: np.ndarray
    source_weights: sparse.csr_matrix
    source_face_ids: np.ndarray
    regions: np.ndarray
    ports: dict[str, dict[str, Any]]
    discarded_faces: dict[str, list[int]] = field(default_factory=dict)

    def surface(self) -> pv.PolyData:
        """Return the tagged computational surface."""
        mesh = pv.PolyData(
            self.points, np.c_[np.full(len(self.faces), 3), self.faces].ravel()
        )
        mesh.cell_data["region"] = self.regions
        mesh.cell_data["source_face_id"] = self.source_face_ids
        return mesh


class AortaLumen:
    """Prepare the rigid five-port aortic lumen and the LV interval flux.

    Input points are read through the public ProcessUSD loader; authored
    topology and time metadata are inspected directly through USD. Three
    source beats are retimed to 3 s (60 bpm, 0.03 s per source sample).
    """

    AORTA = "/World/heart_and_lung_motion/major_vessels/aorta"
    LV = "/World/heart_and_lung_motion/heart/ventricle_left"
    BRANCHES: ClassVar[dict[str, str]] = {
        "BCT_outlet": "brachiocephalic_trunk",
        "LSA_outlet": "subclavian_artery_left",
        "LCCA_stump_outlet": "common_carotid_artery_left",
    }
    # Landmarks inspected on sample 0 and transported to the frozen sample.
    ROOT_REFERENCE_M: ClassVar[tuple[float, float, float]] = (-0.006, -0.125, 0.003)
    LCCA_TIP_M: ClassVar[tuple[float, float, float]] = (
        -0.0003928,
        -0.05196689,
        -0.00595526,
    )
    LCCA_NORMAL: ClassVar[tuple[float, float, float]] = (0.123, 0.990, -0.063)
    SAMPLE_INTERVAL_S = 0.03

    @staticmethod
    def file_sha256(path: Path) -> str:
        """Hash a file in blocks."""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_source(
        self, path: Path, expected_sha256: Optional[str] = None, frame: int = 69
    ) -> AortaSource:
        """Read all aorta and LV samples and the branches at ``frame``.

        Reject changing topology, non-SI stage units or unauthored physical
        timing. Sample 69 is the nearest sample to 70% of the 0..99 range.
        """
        path = Path(path).resolve()
        digest = self.file_sha256(path)
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise ValueError("Source USD SHA256 differs from the expected input")
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise ValueError("Cannot open source USD")
        if not stage.HasAuthoredMetadata("timeCodesPerSecond"):
            raise ValueError("Source has no authored physical time rate")
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
            raise ValueError("This boundary plan requires a stage in meters")
        if UsdGeom.GetStageUpAxis(stage) != "Y":
            raise ValueError("This boundary plan requires the source Y-up stage")
        tools = ProcessUSD()
        arrays: list[np.ndarray] = []
        topologies: list[np.ndarray] = []
        times = np.arange(100, dtype=float)
        for prim_path in (self.AORTA, self.LV):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsA(UsdGeom.Mesh):
                raise ValueError(f"Missing actual source mesh: {prim_path}")
            mesh = UsdGeom.Mesh(prim)
            samples = np.asarray(mesh.GetPointsAttr().GetTimeSamples(), dtype=float)
            if not np.array_equal(samples, times):
                raise ValueError("Expected the full 100-sample Tutorial 13 sequence")
            frames: list[np.ndarray] = []
            faces: Optional[np.ndarray] = None
            for t in times:
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(float(t)))
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(float(t)))
                if not np.all(counts == 3):
                    raise ValueError("Source mesh must contain only triangles")
                current = indices.reshape(-1, 3)
                if faces is None:
                    faces = current
                elif not np.array_equal(faces, current):
                    raise ValueError("Source topology changes during the sequence")
                points = np.asarray(
                    tools.load_usd_as_vtk(path, prim_path, float(t)).points,
                    dtype=np.float64,
                )
                if frames and points.shape != frames[0].shape:
                    raise ValueError("Source point count changes during the sequence")
                frames.append(points)
            assert faces is not None
            arrays.append(np.stack(frames))
            topologies.append(faces)
        branches = {
            name: tools.load_usd_as_vtk(
                path, f"/World/heart_and_lung_motion/major_vessels/{prim}", float(frame)
            )
            for name, prim in self.BRANCHES.items()
        }
        rate = float(stage.GetTimeCodesPerSecond())
        metadata = {
            "time_codes_per_second": rate,
            "source_loop_seconds": 100.0 / rate,
            "retimed_loop_seconds": 100 * self.SAMPLE_INTERVAL_S,
            "assumed_bpm": 60.0,
            "aorta_prim": self.AORTA,
            "lv_prim": self.LV,
            "frozen_sample": frame,
        }
        return AortaSource(
            path,
            digest,
            times,
            np.arange(100) * self.SAMPLE_INTERVAL_S,
            arrays[0],
            topologies[0],
            arrays[1],
            topologies[1],
            branches,
            metadata,
            frame,
        )

    @staticmethod
    def boundary_loops(faces: np.ndarray) -> list[list[int]]:
        """Return ordered rims, rejecting nonmanifold edges and branch vertices."""
        edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Nonmanifold surface edges")
        adjacency: dict[int, list[int]] = {}
        for a, b in unique[counts == 1]:
            adjacency.setdefault(int(a), []).append(int(b))
            adjacency.setdefault(int(b), []).append(int(a))
        if any(len(v) != 2 for v in adjacency.values()):
            raise ValueError("A cut rim is not a simple closed loop")
        remaining = set(adjacency)
        loops: list[list[int]] = []
        while remaining:
            start = min(remaining)
            path = [start]
            previous, current = -1, start
            while True:
                nxt = next(v for v in adjacency[current] if v != previous)
                if nxt == start:
                    break
                if nxt in path:
                    raise ValueError("A cut rim crosses itself topologically")
                path.append(nxt)
                previous, current = current, nxt
            remaining.difference_update(path)
            loops.append(path)
        return loops

    @staticmethod
    def section(
        points: np.ndarray,
        faces: np.ndarray,
        origin: np.ndarray,
        normal: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the plane-section loop nearest ``target``.

        Returns the ordered rim positions, the cut source edges and the weight
        of each edge's second vertex.
        """
        edges = np.unique(
            np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0
        )
        distances = (points - origin) @ normal
        cross = edges[np.prod(distances[edges], axis=1) < 0]
        weights = distances[cross[:, 0]] / (
            distances[cross[:, 0]] - distances[cross[:, 1]]
        )
        positions = (
            points[cross[:, 0]] * (1 - weights[:, None])
            + points[cross[:, 1]] * weights[:, None]
        )
        lookup = {tuple(e): i for i, e in enumerate(cross)}
        segments = []
        for face in faces:
            ids = [
                lookup[key]
                for a, b in zip(face, np.roll(face, -1))
                if (key := (min(int(a), int(b)), max(int(a), int(b)))) in lookup
            ]
            if len(ids) == 2:
                segments.append(ids)
        # Reuse the rim routine by adding a private apex to each segment.
        apex = len(positions)
        loops = AortaLumen.boundary_loops(
            np.c_[np.asarray(segments), np.full(len(segments), apex)]
        )
        loop = min(loops, key=lambda x: np.linalg.norm(positions[x].mean(0) - target))
        return positions[loop], cross[loop], weights[loop]

    @staticmethod
    def polygon_centroid(ring: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Area centroid of a planar polygon, independent of its tessellation."""
        a, b = ring[1:-1], ring[2:]
        area = np.cross(a - ring[0], b - ring[0]) @ normal
        return np.asarray((((ring[0] + a + b) / 3) * area[:, None]).sum(0) / area.sum())

    def cut_planes(
        self, source: AortaSource, distal_margin_m: float
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return (origin, outward normal, target) of the three planar cuts.

        The root plane passes through the section centroid at the transported
        root reference, normal to the local axis estimated from sections 5 mm
        above and below. The LCCA stump is cut 0.15 mm inside its transported
        tip. The distal plane is ``distal_margin_m`` above the inferior CT crop.
        """
        p, f = source.aorta[source.frame], source.faces
        reference = source.aorta[0]
        tree = cKDTree(reference)
        up = np.array([0.0, 1.0, 0.0])

        def transport(point: np.ndarray) -> tuple[np.ndarray, int]:
            index = int(tree.query(point)[1])
            return point + p[index] - reference[index], index

        root, _ = transport(np.asarray(self.ROOT_REFERENCE_M))
        centers = []
        for dy in (0.005, 0.0, -0.005):
            origin = root + dy * up
            ring, _, _ = self.section(p, f, origin, up, origin)
            centers.append(self.polygon_centroid(ring, up))
        root_normal = centers[2] - centers[0]
        root_normal /= np.linalg.norm(root_normal)
        tip, index = transport(np.asarray(self.LCCA_TIP_M))
        neighbors = tree.query(reference[index], k=24)[1]
        deformation = np.linalg.lstsq(
            reference[neighbors] - reference[index],
            p[neighbors] - p[index],
            rcond=None,
        )[0]
        lcca_normal = np.linalg.solve(deformation, np.asarray(self.LCCA_NORMAL))
        lcca_normal /= np.linalg.norm(lcca_normal)
        distal_y = float(p[:, 1].min() + distal_margin_m)
        below = p[p[:, 1] < distal_y]
        distal = np.array([below[:, 0].mean(), distal_y, below[:, 2].mean()])
        return {
            "root_inlet": (centers[1], root_normal, centers[1]),
            "LCCA_stump_outlet": (tip - lcca_normal * 0.00015, lcca_normal, tip),
            "distal_crop_outlet": (distal, -up, distal),
        }

    def prepare_boundaries(
        self, source: AortaSource, distal_margin_m: float = 0.010
    ) -> AortaBoundary:
        """Construct the five-port closed lumen at the frozen sample.

        BCT/LSA openings remove the exact triangles shared with the named
        branch meshes. Root, LCCA stump and distal openings are the planar cuts
        of :meth:`cut_planes`. Region 0 is the wall; ports are 1..5.
        """
        p, f = source.aorta[source.frame], source.faces
        ports: dict[str, dict[str, Any]] = {}
        remove: dict[str, list[int]] = {}
        for name in ("BCT_outlet", "LSA_outlet"):
            branch = source.branches[name]
            distances, ids_value = cKDTree(p).query(branch.points)
            ids = np.asarray(ids_value, dtype=np.int64)
            ids[np.asarray(distances) > 1e-8] = -1
            branch_faces = branch.faces.reshape(-1, 4)[:, 1:]
            keys = {tuple(sorted(t)) for t in ids[branch_faces] if np.min(t) >= 0}
            shared = [i for i, t in enumerate(f) if tuple(sorted(t)) in keys]
            if not shared:
                raise ValueError(f"No shared anatomical interface for {name}")
            rims = self.boundary_loops(f[shared])
            if len(rims) != 1:
                raise ValueError(f"Ambiguous shared interface for {name}")
            remove[name] = shared
            ports[name] = {
                "center_m": p[rims[0]].mean(0).tolist(),
                "role": "outlet",
                "region_id": len(ports) + 1,
            }
        all_edges = np.unique(
            np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0
        )
        cut: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for name, (origin, normal, target) in self.cut_planes(
            source, distal_margin_m
        ).items():
            rim, _, _ = self.section(p, f, origin, normal, target)
            distances = (p - origin) @ normal
            edges = all_edges[(distances[all_edges] > 0).all(axis=1)]
            matrix = sparse.coo_matrix(
                (
                    np.ones(len(edges) * 2),
                    (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
                ),
                shape=(len(p), len(p)),
            )
            _, labels = connected_components(matrix, directed=False)
            components = [
                np.flatnonzero((labels == r) & (distances > 0))
                for r in np.unique(labels[distances > 0])
            ]
            positive = min(
                (x for x in components if len(x) > 1),
                key=lambda x: np.linalg.norm(p[x].mean(0) - rim.mean(0)),
            )
            affected = np.flatnonzero(np.isin(f, positive).any(axis=1))
            if any(int(i) in cut for i in affected):
                raise ValueError("Computational cuts overlap")
            cut.update({int(i): (origin, normal) for i in affected})
            remove[name] = affected.tolist()
            ports[name] = {
                "center_m": self.polygon_centroid(rim, normal).tolist(),
                "role": "inlet" if name == "root_inlet" else "outlet",
                "region_id": len(ports) + 1,
                "plane_origin_m": origin.tolist(),
                "plane_normal": normal.tolist(),
            }
        points = p.tolist()
        source_ids = [[i] for i in range(len(p))]
        source_weights = [[1.0] for _ in p]
        new_edges: dict[tuple[int, int], int] = {}
        faces: list[list[int]] = []
        parents: list[int] = []
        omitted = set(remove["BCT_outlet"] + remove["LSA_outlet"])
        for i, triangle in enumerate(f):
            if i in omitted:
                continue
            polygon = triangle.tolist()
            if i in cut:
                origin, normal = cut[i]
                distances = (p[triangle] - origin) @ normal
                polygon = []
                for j in range(3):
                    a, b = int(triangle[j]), int(triangle[(j + 1) % 3])
                    da, db = distances[j], distances[(j + 1) % 3]
                    if da <= 0:
                        polygon.append(a)
                    if da * db < 0:
                        key = (min(a, b), max(a, b))
                        if key not in new_edges:
                            w = float(da / (da - db))
                            new_edges[key] = len(points)
                            points.append((p[a] * (1 - w) + p[b] * w).tolist())
                            source_ids.append([a, b])
                            source_weights.append([1 - w, w])
                        polygon.append(new_edges[key])
            for j in range(1, len(polygon) - 1):
                faces.append([polygon[0], polygon[j], polygon[j + 1]])
                parents.append(i)
        triangles = np.asarray(faces)
        used = np.unique(triangles)
        inverse = np.full(len(points), -1)
        inverse[used] = np.arange(len(used))
        triangles = inverse[triangles]
        q = np.asarray(points)[used]
        rows, columns, values = [], [], []
        for row, old in enumerate(used):
            for col, value in zip(source_ids[old], source_weights[old]):
                rows.append(row)
                columns.append(col)
                values.append(value)
        mapping = sparse.csr_matrix((values, (rows, columns)), shape=(len(q), len(p)))
        loops = self.boundary_loops(triangles)
        if len(loops) != 5:
            raise ValueError(f"Expected five openings; found {len(loops)}")
        tags = [0] * len(triangles)
        matched = set()
        for loop in loops:
            name = min(
                ports,
                key=lambda n: np.linalg.norm(
                    np.asarray(ports[n]["center_m"]) - q[loop].mean(0)
                ),
            )
            if name in matched:
                raise ValueError("Ambiguous cut-to-anatomy association")
            matched.add(name)
            if name == "root_inlet":
                # A central fan avoids long diagonals across the wide root cap.
                center_id = len(q)
                q = np.vstack([q, q[loop].mean(0)])
                center_weights = sparse.csr_matrix(
                    mapping[loop].sum(axis=0) / len(loop)
                )
                mapping = sparse.csr_matrix(
                    sparse.vstack([mapping, center_weights], format="csr")
                )
                cap_faces = np.c_[
                    loop, np.roll(loop, -1), np.full(len(loop), center_id)
                ]
            else:
                polygon_mesh = pv.PolyData(
                    q[loop], np.r_[len(loop), np.arange(len(loop))]
                )
                cap = polygon_mesh.triangulate()
                cap_faces = np.asarray(loop)[cap.faces.reshape(-1, 4)[:, 1:]]
            # A cap must oppose the orientation of the adjacent wall edge.
            a, b = loop[:2]
            forward = np.any(np.all(triangles[:, [0, 1]] == [a, b], axis=1))
            forward |= np.any(np.all(triangles[:, [1, 2]] == [a, b], axis=1))
            forward |= np.any(np.all(triangles[:, [2, 0]] == [a, b], axis=1))
            if forward:
                cap_faces = cap_faces[:, ::-1]
            triangles = np.vstack([triangles, cap_faces])
            tags.extend([ports[name]["region_id"]] * len(cap_faces))
            parents.extend([-1] * len(cap_faces))
        boundary = AortaBoundary(
            q, triangles, mapping, np.asarray(parents), np.asarray(tags), ports, remove
        )
        if self.boundary_loops(boundary.faces):
            raise ValueError("Capped lumen is not closed")
        if self.signed_surface_volume(boundary.points, boundary.faces) < 0:
            boundary.faces = boundary.faces[:, ::-1]
        return boundary

    @staticmethod
    def signed_surface_volume(points: np.ndarray, faces: np.ndarray) -> float:
        """Compute the oriented enclosed volume of a closed triangle surface."""
        triangles = points[faces] - points.mean(0)
        return float(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            ).sum()
            / 6
        )

    def infer_inlet(self, source: AortaSource) -> dict[str, Any]:
        """Infer the LV ejection flux of each 0.03 s source interval.

        An ideal non-regurgitant aortic valve and a closed mitral valve during
        ejection are assumed: each interval's flux is its positive LV-volume
        loss divided by 0.03 s, and filling intervals carry no flux. Three
        ejection episodes must be separated by filling. Small disconnected LV
        fragments are reported and excluded.
        """
        faces = source.lv_faces
        edges = np.unique(
            np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0
        )
        graph = sparse.coo_matrix(
            (
                np.ones(len(edges) * 2),
                (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
            ),
            shape=(source.lv.shape[1], source.lv.shape[1]),
        )
        count, labels = connected_components(graph, directed=False)
        volumes = []
        for i in range(count):
            selected = faces[(labels[faces] == i).all(axis=1)]
            if self.boundary_loops(selected):
                raise ValueError("LV component is not closed")
            volumes.append(
                np.asarray(
                    [
                        abs(self.signed_surface_volume(points, selected))
                        for points in source.lv
                    ]
                )
            )
        dominant = int(np.argmax([v.max() for v in volumes]))
        cavity = volumes[dominant]
        fragments = [v for i, v in enumerate(volumes) if i != dominant]
        if any(v.max() > cavity.min() * 1e-4 for v in fragments):
            raise ValueError("A discarded LV component is not negligible")
        loss = cavity - np.roll(cavity, -1)
        gate = loss > 0
        starts = np.flatnonzero(gate & ~np.roll(gate, 1))
        if len(starts) != 3:
            raise ValueError(
                f"Expected three gated ejection episodes; found {len(starts)}"
            )
        flux = np.where(gate, loss / self.SAMPLE_INTERVAL_S, 0.0)
        return {
            "lv_volume_m3": cavity.tolist(),
            "inlet_interval_flux_m3_s": flux.tolist(),
            "ejection_gate": gate.tolist(),
            "ejection_start_samples": starts.tolist(),
            "stroke_volumes_m3": [
                float(flux[np.arange(a, b) % len(flux)].sum() * self.SAMPLE_INTERVAL_S)
                for a, b in zip(starts, np.r_[starts[1:], starts[0] + len(flux)])
            ],
            "discarded_component_volume_range_m3": [
                [float(v.min()), float(v.max())] for v in fragments
            ],
            "assumptions": "60 bpm, ideal non-regurgitant valve, closed mitral",
        }
