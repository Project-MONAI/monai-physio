"""Incompressible Navier-Stokes flow on tetrahedral meshes with Warp FEM.

Velocity is quadratic and pressure linear (Taylor-Hood P2/P1). Each time step
uses backward Euler, the symmetric viscous stress and a single-step
semi-Lagrangian backtrace of the previous velocity. Walls and the inlet are
Dirichlet boundaries; :func:`inlet_flow` builds a smooth periodic inlet flow
from interval fluxes. Every outlet carries a three-element Windkessel (RCR)
traction, coupled implicitly, plus backflow stabilization. GMRES solves the
saddle-point system with a block-triangular Schur-complement preconditioner
whose pressure block is factored once on the GPU by NVIDIA cuDSS.

Internally lengths are scaled by 0.1 m, velocities by 0.1 m/s and pressure by
rho (0.1 m/s)^2. All inputs and outputs are SI. The FEM operators and the
linear solve run in float32 (default) or float64; returned arrays are float64.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import site
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import warp as wp
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import ndtr
from warp import fem
from warp.optim.linear import LinearOperator, gmres
from warp.sparse import (
    bsr_axpy,
    bsr_copy,
    bsr_from_triplets,
    bsr_get_diag,
    bsr_mv,
    bsr_transposed,
)

from .monai_physio_base import MONAIPhysioBase

LENGTH = 0.1
SPEED = 0.1
# Scalar, velocity and 3 x 3 block types of each supported precision, and its
# default relative GMRES tolerance. Float32 residuals of the aorta stagnate
# near 1e-6, so float32 stops at 1e-5.
PRECISIONS: dict[str, tuple[Any, Any, Any, float]] = {
    "float32": (wp.float32, wp.vec3f, wp.mat33f, 1e-5),
    "float64": (wp.float64, wp.vec3d, wp.mat33d, 1e-6),
}
# Warp 1.17 bilinear assembly overflows 32-bit offsets into its 3 x 3 block
# triplets beyond about 298,000 quadratic tetrahedra, so the velocity operator
# is assembled over chunks of at most this many cells and summed.
CELL_CHUNK = 200_000


def tetrahedron_edges(tetrahedra: np.ndarray) -> np.ndarray:
    """Return the sorted unique edges of tetrahedra (the P2 edge node order)."""
    pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    return np.unique(np.sort(tetrahedra[:, pairs].reshape(-1, 2), axis=1), axis=0)


def triangle_nodes(
    faces: np.ndarray, edges: np.ndarray, vertex_count: int
) -> np.ndarray:
    """Return the six P2 nodes (v0, v1, v2, e01, e12, e20) of each triangle."""
    keys = edges[:, 0] * vertex_count + edges[:, 1]
    pairs = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
    edge_ids = np.searchsorted(keys, pairs[:, :, 0] * vertex_count + pairs[:, :, 1])
    return np.asarray(np.c_[faces, vertex_count + edge_ids])


@dataclass
class Windkessel:
    """Three-element Windkessel (RCR) outlet.

    The outlet pressure is ``P = R_p Q + P_c``, where the distal capacitor
    pressure obeys ``C dP_c/dt = Q - P_c / R_d``. A zero distal resistance
    gives a pure resistance ``R_p``; zero resistances give zero traction.

    Attributes:
        proximal_resistance: R_p in Pa s/m^3.
        distal_resistance: R_d in Pa s/m^3.
        compliance: C in m^3/Pa.
        pressure: Capacitor pressure P_c in Pa.
    """

    proximal_resistance: float
    distal_resistance: float
    compliance: float
    pressure: float

    @classmethod
    def from_total(
        cls,
        fraction: float,
        total_resistance: float,
        total_compliance: float,
        proximal_fraction: float,
        pressure: float,
    ) -> Windkessel:
        """Split whole-body values to an outlet receiving ``fraction`` of flow.

        Resistances scale with ``1 / fraction`` and compliance with
        ``fraction``, so every outlet has the same time constant.
        """
        resistance = total_resistance / fraction
        return cls(
            proximal_fraction * resistance,
            (1 - proximal_fraction) * resistance,
            total_compliance * fraction,
            pressure,
        )

    def _decay(self, dt: float) -> float:
        rc = self.distal_resistance * self.compliance
        return rc / (rc + dt)

    def implicit(self, dt: float) -> tuple[float, float]:
        """Return ``(R, P0)`` with backward-Euler pressure ``R Q^{n+1} + P0``."""
        decay = self._decay(dt)
        return (
            self.proximal_resistance + decay * dt / self.compliance,
            decay * self.pressure,
        )

    def advance(self, flow: float, dt: float) -> float:
        """Advance ``P_c`` with the new outlet flow and return the outlet pressure."""
        resistance, offset = self.implicit(dt)
        self.pressure = self._decay(dt) * (self.pressure + dt * flow / self.compliance)
        return resistance * flow + offset


def inlet_flow(
    times: np.ndarray,
    interval_flux: np.ndarray,
    interval_s: float,
    sigma_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smoothed periodic inlet flow Q(t) and dQ/dt.

    ``interval_flux`` holds the mean flux of consecutive source intervals of
    ``interval_s`` seconds (for example the left-ventricle volume loss between
    frames of a cardiac image series) and repeats with period
    ``len(interval_flux) * interval_s``. This staircase has unphysical jumps,
    so it is convolved with a periodic Gaussian of standard deviation
    ``sigma_s``, evaluated exactly with the normal CDF. A nonnegative kernel
    of unit integral keeps Q >= 0 and preserves the volume of every beat whose
    window boundaries lie in diastole, many ``sigma_s`` from ejection.

    Args:
        times: Evaluation times in seconds.
        interval_flux: Flux of each source interval in m^3/s.
        interval_s: Duration of each source interval in seconds.
        sigma_s: Gaussian standard deviation in seconds, much less than one
            period (three periodic images are summed).

    Returns:
        Flow in m^3/s and its derivative in m^3/s^2 at ``times``.
    """
    flux = np.asarray(interval_flux, dtype=np.float64)
    period = len(flux) * interval_s
    edges = np.arange(len(flux) + 1) * interval_s
    t = np.mod(np.asarray(times, dtype=np.float64), period)[:, None]
    flow = np.zeros(len(t))
    derivative = np.zeros(len(t))
    for shift in (-period, 0.0, period):
        start = (t - edges[:-1] - shift) / sigma_s
        end = (t - edges[1:] - shift) / sigma_s
        flow += (ndtr(start) - ndtr(end)) @ flux
        density = np.exp(-0.5 * start**2) - np.exp(-0.5 * end**2)
        derivative += density @ flux / (sigma_s * np.sqrt(2 * np.pi))
    return flow, derivative


def find_cudss_library() -> Path:
    """Locate the NVIDIA cuDSS shared library.

    ``CUDSS_LIBRARY`` names the library file explicitly. Otherwise the NVIDIA
    cuDSS Python wheels in site-packages and then the system library search
    path are used.

    Raises:
        RuntimeError: If cuDSS cannot be found.
    """
    explicit = os.environ.get("CUDSS_LIBRARY")
    if explicit:
        if not Path(explicit).is_file():
            raise RuntimeError(f"CUDSS_LIBRARY is not a file: {explicit}")
        return Path(explicit)
    pattern = "cudss64_*.dll" if os.name == "nt" else "libcudss.so*"
    for root in [*site.getsitepackages(), site.getusersitepackages()]:
        matches = sorted(Path(root, "nvidia").glob(f"**/{pattern}"))
        if matches:
            return matches[-1]
    found = ctypes.util.find_library("cudss64_1" if os.name == "nt" else "cudss")
    if found:
        return Path(found)
    raise RuntimeError(
        "NVIDIA cuDSS was not found. Install the cuDSS wheel for your CUDA "
        "version (for example nvidia-cudss-cu13) or set CUDSS_LIBRARY to the "
        "cuDSS shared library."
    )


class CuDssFactor:
    """Sparse direct factorization on the GPU with NVIDIA cuDSS.

    The CSR matrix is analyzed and factored once on the current Warp stream.
    Each :meth:`solve` overwrites the persistent ``solution`` array from the
    persistent ``rhs`` array, so callers can fill and read them in place.
    Values, ``rhs`` and ``solution`` use ``dtype`` (Warp float32 or float64).
    """

    MATRIX_TYPES: ClassVar[dict[str, int]] = {"general": 0, "symmetric": 1, "spd": 3}
    ANALYSIS, FACTORIZATION, SOLVE = 3, 4, 1008
    VALUE_TYPES: ClassVar[dict[Any, int]] = {wp.float32: 0, wp.float64: 1}

    def __init__(
        self,
        matrix: sparse.csr_matrix,
        matrix_type: str = "spd",
        dtype: Any = wp.float64,
    ) -> None:
        """Upload ``matrix`` and factor it.

        Args:
            matrix: Square CSR matrix with 32-bit indices.
            matrix_type: "general", "symmetric" or "spd".
            dtype: Value type, ``wp.float32`` or ``wp.float64``.
        """
        library = find_cudss_library()
        self._directories: list[Any] = []
        if sys.platform == "win32":
            folders = [library.parent]
            cuda = os.environ.get("CUDA_PATH")
            if cuda:
                folders += [Path(cuda, "bin"), Path(cuda, "bin", "x64")]
            for root in site.getsitepackages():
                folders += [
                    p.parent for p in Path(root, "nvidia").glob("*/bin/**/*.dll")
                ]
            self._directories = [
                os.add_dll_directory(str(folder))
                for folder in dict.fromkeys(folders)
                if folder.is_dir()
            ]
        self.lib: Any = ctypes.CDLL(str(library))
        p, i, q, z = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_size_t
        signatures = {
            "cudssCreate": [ctypes.POINTER(p)],
            "cudssDestroy": [p],
            "cudssConfigCreate": [ctypes.POINTER(p)],
            "cudssConfigDestroy": [p],
            "cudssDataCreate": [p, ctypes.POINTER(p)],
            "cudssDataDestroy": [p, p],
            "cudssSetStream": [p, p],
            "cudssMatrixCreateCsr": [ctypes.POINTER(p), q, q, q, p, p, p, p]
            + [i] * 6,  # offset/index/value types, matrix type, view, base
            "cudssMatrixCreateDn": [ctypes.POINTER(p), q, q, q, p, i, i],
            "cudssMatrixDestroy": [p],
            "cudssExecute": [p, i, p, p, p, p, p],
            "cudssDataGet": [p, p, i, p, z, ctypes.POINTER(z)],
        }
        for name, arguments in signatures.items():
            getattr(self.lib, name).argtypes = arguments
            getattr(self.lib, name).restype = i
        self.handle, self.config, self.data = p(), p(), p()
        self.a, self.b, self.x = p(), p(), p()
        matrix = matrix.tocsr()
        self.n = int(matrix.shape[0])
        self.offsets: Any = wp.array(matrix.indptr.astype(np.int32), dtype=wp.int32)
        self.columns: Any = wp.array(matrix.indices.astype(np.int32), dtype=wp.int32)
        self.values: Any = wp.array(matrix.data, dtype=dtype)
        self.rhs = wp.zeros(self.n, dtype=dtype)
        self.solution = wp.zeros(self.n, dtype=dtype)
        try:
            self._check("cudssCreate", ctypes.byref(self.handle))
            self._check("cudssSetStream", self.handle, p(wp.get_stream().cuda_stream))
            self._check("cudssConfigCreate", ctypes.byref(self.config))
            self._check("cudssDataCreate", self.handle, ctypes.byref(self.data))
            # Offsets and columns CUDA_R_32I=10, values CUDA_R_32F=0 or
            # CUDA_R_64F=1, full (not triangular) view, zero index base.
            self._check(
                "cudssMatrixCreateCsr",
                ctypes.byref(self.a),
                self.n,
                self.n,
                int(matrix.nnz),
                p(self.offsets.ptr),
                None,
                p(self.columns.ptr),
                p(self.values.ptr),
                10,
                10,
                self.VALUE_TYPES[dtype],
                self.MATRIX_TYPES[matrix_type],
                0,
                0,
            )
            for handle, array in ((self.b, self.rhs), (self.x, self.solution)):
                self._check(
                    "cudssMatrixCreateDn",
                    ctypes.byref(handle),
                    self.n,
                    1,
                    self.n,
                    p(array.ptr),
                    self.VALUE_TYPES[dtype],
                    0,
                )
            self._execute(self.ANALYSIS)
            self._execute(self.FACTORIZATION)
            wp.synchronize_device()
            self.check_info()
        except BaseException:
            self.close()
            raise

    def _check(self, name: str, *arguments: Any) -> None:
        status = int(getattr(self.lib, name)(*arguments))
        if status:
            raise RuntimeError(f"{name} failed with cuDSS status {status}")

    def _execute(self, phase: int) -> None:
        self._check(
            "cudssExecute",
            self.handle,
            phase,
            self.config,
            self.data,
            self.a,
            self.x,
            self.b,
        )

    def solve(self) -> None:
        """Enqueue ``solution = A^-1 rhs`` on the current Warp stream."""
        self._execute(self.SOLVE)

    def check_info(self) -> None:
        """Raise if cuDSS reported a numerical error (call after synchronizing)."""
        info, written = ctypes.c_int(), ctypes.c_size_t()
        self._check(
            "cudssDataGet",
            self.handle,
            self.data,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(written),
        )
        if info.value:
            raise RuntimeError(f"cuDSS numerical info={info.value}")

    def close(self) -> None:
        """Release native objects before their GPU buffers."""
        for matrix in (self.x, self.b, self.a):
            if matrix.value:
                self.lib.cudssMatrixDestroy(matrix)
                matrix.value = None
        if self.data.value:
            self.lib.cudssDataDestroy(self.handle, self.data)
            self.data.value = None
        if self.config.value:
            self.lib.cudssConfigDestroy(self.config)
            self.config.value = None
        if self.handle.value:
            self.lib.cudssDestroy(self.handle)
            self.handle.value = None
        for directory in self._directories:
            directory.close()
        self._directories = []


# Warp 1.17 FEM treats every annotation, including return, as an input. The
# integrands and kernels are generic over the precision: ``x.dtype`` of a Warp
# vector or matrix is its scalar type, which converts the float64 coefficients.
@fem.integrand
def velocity_form(  # type: ignore[no-untyped-def]
    s: fem.Sample, u: fem.Field, v: fem.Field, dt: wp.float64, nu: wp.float64
):
    """Backward-Euler mass and symmetric viscous stress."""
    value = u(s)  # type: ignore[operator]
    mass = wp.dot(value, v(s)) / value.dtype(dt)  # type: ignore[operator]
    return mass + value.dtype(nu + nu) * wp.ddot(fem.D(u, s), fem.D(v, s))


@fem.integrand
def divergence_form(  # type: ignore[no-untyped-def]
    s: fem.Sample, u: fem.Field, q: fem.Field
):
    """Incompressibility constraint and its adjoint pressure-gradient coupling."""
    return -q(s) * fem.div(u, s)  # type: ignore[operator]


@fem.integrand
def transported_form(  # type: ignore[no-untyped-def]
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    w: fem.Field,
    v: fem.Field,
    dt: wp.float64,
):
    """Single-step backtrace of the previous velocity along ``u - w``."""
    velocity = u(s) - w(s)  # type: ignore[operator]
    step = velocity.dtype(dt)
    position = domain(s) - step * velocity  # type: ignore[operator]
    departure = fem.lookup(domain, position, s)
    return wp.dot(u(departure), v(s)) / step  # type: ignore[operator]


@fem.integrand
def backflow_form(  # type: ignore[no-untyped-def]
    s: fem.Sample,
    domain: fem.Domain,
    previous: fem.Field,
    mesh: fem.Field,
    u: fem.Field,
    v: fem.Field,
    beta: wp.float64,
):
    """Outflow stabilization ``beta |min((u_n - w) . n, 0)| u . v`` on boundary sides.

    The flux is relative to the mesh velocity ``w``, so it vanishes on a moving
    no-slip wall and acts only where blood re-enters through a port.
    """
    normal = fem.normal(domain, s)
    flux = wp.dot(previous(s) - mesh(s), normal)  # type: ignore[operator]
    inflow = wp.min(flux, normal.dtype(0.0))
    return -normal.dtype(beta) * inflow * wp.dot(u(s), v(s))  # type: ignore[operator]


@wp.kernel
def constrain_velocity_matrix(
    offsets: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    columns: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    values: wp.array(dtype=Any),  # type: ignore[valid-type]
    constrained: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
) -> None:
    """Eliminate constrained rows and columns, keeping a unit diagonal."""
    row = int(wp.tid())  # type: ignore[arg-type]
    for index in range(offsets[row], offsets[row + 1]):
        col = columns[index]
        if constrained[row] != 0 or constrained[col] != 0:
            block = values.dtype(0.0)
            if row == col:
                # Needed by the Warp 1.17 stubs, not by 1.19.
                block = wp.identity(n=3, dtype=block.dtype)  # type: ignore[arg-type,unused-ignore]
            values[index] = block


@wp.kernel
def constrain_divergence_matrix(
    offsets: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    columns: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    values: wp.array(dtype=Any),  # type: ignore[valid-type]
    constrained: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
) -> None:
    """Remove constrained velocity columns from the divergence operator."""
    row = int(wp.tid())  # type: ignore[arg-type]
    for index in range(offsets[row], offsets[row + 1]):
        if constrained[columns[index]] != 0:
            values[index] = values.dtype(0.0)


@wp.kernel
def zero_constrained(
    values: wp.array(dtype=Any),  # type: ignore[valid-type]
    constrained: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
) -> None:
    """Zero constrained entries of a velocity vector."""
    row = int(wp.tid())  # type: ignore[arg-type]
    if constrained[row] != 0:
        values[row] = values.dtype(0.0)


@wp.kernel
def diagonal_apply(
    inverse: wp.array(dtype=Any),  # type: ignore[valid-type]
    x: wp.array(dtype=Any),  # type: ignore[valid-type]
    y: wp.array(dtype=Any),  # type: ignore[valid-type]
) -> None:
    """Apply the inverse block diagonal of the velocity operator."""
    i = int(wp.tid())  # type: ignore[arg-type]
    y[i] = inverse[i] * x[i]


@wp.kernel
def velocity_finish(
    inverse: wp.array(dtype=Any),  # type: ignore[valid-type]
    provisional: wp.array(dtype=Any),  # type: ignore[valid-type]
    gradient: wp.array(dtype=Any),  # type: ignore[valid-type]
    y: wp.array(dtype=Any),  # type: ignore[valid-type]
    z: wp.array(dtype=Any),  # type: ignore[valid-type]
    alpha: Any,
    beta: Any,
) -> None:
    """Velocity part of the block-triangular preconditioner."""
    i = int(wp.tid())  # type: ignore[arg-type]
    value = alpha * (provisional[i] - inverse[i] * gradient[i])
    if beta != type(beta)(0.0):
        value += beta * y[i]
    z[i] = value


@wp.kernel
def pressure_finish(
    solution: wp.array(dtype=Any),  # type: ignore[valid-type]
    y: wp.array(dtype=Any),  # type: ignore[valid-type]
    z: wp.array(dtype=Any),  # type: ignore[valid-type]
    alpha: Any,
    beta: Any,
) -> None:
    """Pressure part of the block-triangular preconditioner."""
    i = int(wp.tid())  # type: ignore[arg-type]
    value = alpha * solution[i]
    if beta != type(beta)(0.0):
        value += beta * y[i]
    z[i] = value


class WarpNavierStokes(MONAIPhysioBase):
    """P2/P1 incompressible flow with a Dirichlet inlet and RCR outlets.

    The mesh is given by vertex positions, linear tetrahedra and outward
    oriented boundary triangles labelled 0 (no-slip wall) or by a port
    region id. Velocity arrays exchanged with callers are in the canonical
    P2 order: vertices, then the sorted unique tetrahedron edges
    (``edges``). Pressure arrays are per vertex.

    Moving walls (ALE) for fluid-structure coupling: before each step call
    :meth:`move` with the vertex displacement and mesh velocity (ports must
    stay fixed; elements stay straight-sided), pass the P2 wall velocity to
    :meth:`step` through ``wall_velocity``, then read the fluid load on the
    wall with :meth:`wall_force`. :meth:`move` reassembles the operators;
    :meth:`update_preconditioner` refactors the Schur block if GMRES
    iterations grow.
    """

    def __init__(
        self,
        points: np.ndarray,
        tetrahedra: np.ndarray,
        boundary_faces: np.ndarray,
        boundary_regions: np.ndarray,
        *,
        inlet_region: int,
        outlets: dict[int, Windkessel],
        dt: float,
        density: float = 1060.0,
        viscosity: float = 0.004,
        backflow: float = 1.0,
        blunt_fraction: float = 0.2,
        tolerance: Optional[float] = None,
        max_iterations: int = 500,
        precision: str = "float32",
    ) -> None:
        """Build spaces, boundary data and operators, and factor the Schur block.

        Args:
            points: Vertex positions in metres.
            tetrahedra: Positively oriented vertex indices of each tetrahedron.
            boundary_faces: Outward oriented boundary triangles.
            boundary_regions: 0 for wall faces, otherwise the port region id.
            inlet_region: Region id of the Dirichlet inlet.
            outlets: Windkessel of every outlet region; updated in place.
            dt: Time step in seconds.
            density: Fluid density in kg/m^3.
            viscosity: Dynamic viscosity in Pa s.
            backflow: Backflow stabilization coefficient beta (0 disables it).
            blunt_fraction: The inlet speed rises linearly from the wall over
                this fraction of the inlet's equivalent radius, then is flat.
            tolerance: Relative GMRES tolerance; None uses 1e-5 in float32
                and 1e-6 in float64. Where the float32 residual stagnates is
                problem dependent, so other meshes may need a looser value.
            max_iterations: GMRES iteration limit.
            precision: "float32" or "float64" for the FEM operators, the
                linear solve and the cuDSS factorization.
        """
        super().__init__(class_name=self.__class__.__name__)
        started = time.perf_counter()
        regions = np.asarray(boundary_regions)
        ports = {inlet_region, *outlets}
        if set(np.unique(regions)) != {0} | ports or not outlets:
            raise ValueError("Every port must be the inlet or an outlet")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
        self.scalar, self.vector, self.block, default = PRECISIONS[precision]
        self.tolerance = float(default if tolerance is None else tolerance)
        self.dt, self.density = float(dt), float(density)
        self.viscosity, self.outlets = float(viscosity), outlets
        self.backflow = float(backflow)
        self.max_iterations = int(max_iterations)
        self.points = np.asarray(points, dtype=np.float64)
        self.tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
        self.center = self.points.mean(axis=0)
        scaled = (self.points - self.center) / LENGTH
        self.edges = tetrahedron_edges(self.tetrahedra)
        self.geometry: Any = fem.Tetmesh(
            tet_vertex_indices=wp.array(self.tetrahedra, dtype=wp.int32),
            positions=wp.array(scaled, dtype=self.vector),
            build_bvh=True,
        )
        self.u_space: Any = fem.make_polynomial_space(
            self.geometry, degree=2, dtype=self.vector
        )
        self.p_space: Any = fem.make_polynomial_space(
            self.geometry, degree=1, dtype=self.scalar
        )
        self.dof_points = np.concatenate(
            [self.points, self.points[self.edges].mean(axis=1)]
        )
        self.u_index = self._match(
            (self.dof_points - self.center) / LENGTH, self.u_space
        )
        self.p_index = self._match(scaled, self.p_space)

        faces = np.asarray(boundary_faces, dtype=np.int64)
        self.face_dofs = triangle_nodes(faces, self.edges, len(self.points))
        corners = self.points[faces]
        self.area_vectors = (
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]) / 2
        )
        cells = self.points[self.tetrahedra]
        volume = np.einsum(
            "ij,ij->",
            cells[:, 1] - cells[:, 0],
            np.cross(cells[:, 2] - cells[:, 0], cells[:, 3] - cells[:, 0]),
        )
        flux_volume = np.einsum("ij,ij->", corners.mean(axis=1), self.area_vectors)
        if abs(flux_volume * 2 / volume - 1) > 1e-8:
            raise ValueError("Boundary faces are not closed and outward oriented")
        self.wall_dofs = np.unique(self.face_dofs[regions == 0])
        constrained = np.zeros(len(self.dof_points), dtype=np.int32)
        constrained[self.wall_dofs] = 1
        constrained[np.unique(self.face_dofs[regions == inlet_region])] = 1
        self.constrained: Any = wp.array(self._to_fem(constrained), dtype=wp.int32)
        # Edge DOFs carry a flat face's flux: the vertex P2 basis integrates to 0.
        self.port_flux = {}
        for region in sorted(ports):
            vector = np.zeros((len(self.dof_points), 3))
            selected = regions == region
            for k in range(3, 6):
                np.add.at(
                    vector,
                    self.face_dofs[selected, k],
                    self.area_vectors[selected] / 3,
                )
            self.port_flux[region] = vector
        self.inlet_shape = self._inlet_shape(regions, inlet_region, blunt_fraction)

        domain = fem.Cells(self.geometry)
        boundary = fem.BoundarySides(self.geometry)
        self.u_test = fem.make_test(space=self.u_space, domain=domain)
        self.u_trial = fem.make_trial(space=self.u_space, domain=domain)
        self.u_chunks: list[tuple[Any, Any]] = []
        for indices in np.array_split(
            np.arange(len(self.tetrahedra)), -(-len(self.tetrahedra) // CELL_CHUNK)
        ):
            chunk = fem.Subdomain(domain, element_indices=wp.array(indices, dtype=int))
            self.u_chunks.append(
                (
                    fem.make_trial(space=self.u_space, domain=chunk),
                    fem.make_test(space=self.u_space, domain=chunk),
                )
            )
        self.p_test = fem.make_test(space=self.p_space, domain=domain)
        self.side_test = fem.make_test(space=self.u_space, domain=boundary)
        self.side_trial = fem.make_trial(space=self.u_space, domain=boundary)
        self.u_field: Any = self.u_space.make_field()
        self.w_field: Any = self.u_space.make_field()
        self.pressure_dofs: Any = wp.zeros(self.p_space.node_count(), dtype=self.scalar)
        self.n_u, self.n_p = self.u_space.node_count(), self.p_space.node_count()
        self.rcr_matrix = self._rcr_matrix()
        self.assemble()
        self.a = self._constrained_velocity(self.a_static)
        self.operator = LinearOperator(
            (3 * self.n_u + self.n_p,) * 2,
            dtype=self.scalar,
            device=wp.get_device(),
            matvec=self._saddle_matvec,
        )
        self.factor: Optional[CuDssFactor] = None
        self.update_preconditioner()
        self.log_info(
            "%d P2 velocity nodes, %d pressure nodes, setup %.1f s",
            self.n_u,
            self.n_p,
            time.perf_counter() - started,
        )

    @staticmethod
    def _match(positions: np.ndarray, space: Any) -> np.ndarray:
        """Map canonical nodes to Warp space nodes by position."""
        distances, index = cKDTree(space.node_positions().numpy()).query(positions)
        index = np.asarray(index, dtype=np.int64)
        # 1e-5 (1 um) lies above float32 rounding and far below node spacing.
        if np.max(distances) > 1e-5 or len(np.unique(index)) != len(positions):
            raise ValueError("Warp FEM nodes do not match the canonical P2 nodes")
        return index

    def _to_fem(self, values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values)
        result[self.u_index] = values
        return result

    def _inlet_shape(
        self, regions: np.ndarray, region: int, blunt_fraction: float
    ) -> np.ndarray:
        """Blunt inflow profile carrying 1 m^3/s, in canonical order."""
        selected = regions == region
        area = self.area_vectors[selected].sum(axis=0)
        normal = area / np.linalg.norm(area)
        radius = np.sqrt(np.linalg.norm(area) / np.pi)
        free = np.setdiff1d(np.unique(self.face_dofs[selected]), self.wall_dofs)
        clearance = cKDTree(self.dof_points[self.wall_dofs]).query(
            self.dof_points[free]
        )[0]
        shape = np.zeros((len(self.dof_points), 3))
        shape[free] = (
            -np.minimum(clearance / (blunt_fraction * radius), 1)[:, None] * normal
        )
        inflow = -np.einsum("ij,ij->", self.port_flux[region], shape)
        if inflow <= 0:
            raise ValueError("Inlet has no free velocity nodes")
        return np.asarray(shape / inflow)

    def _rcr_matrix(self) -> Optional[Any]:
        """Implicit outlet coupling sum_i R_i b_i b_i^T in Warp node order."""
        rows, columns, blocks = [], [], []
        for region, windkessel in self.outlets.items():
            resistance = windkessel.implicit(self.dt)[0]
            if resistance == 0:
                continue
            scaled = resistance * LENGTH**2 / (self.density * SPEED)
            ids = np.flatnonzero(np.any(self.port_flux[region] != 0, axis=1))
            vector = self.port_flux[region][ids] / LENGTH**2
            nodes = self.u_index[ids]
            rows.append(np.repeat(nodes, len(nodes)))
            columns.append(np.tile(nodes, len(nodes)))
            blocks.append(
                (scaled * vector[:, None, :, None] * vector[None, :, None, :]).reshape(
                    -1, 3, 3
                )
            )
        if not rows:
            return None
        return bsr_from_triplets(
            self.n_u,
            self.n_u,
            wp.array(np.concatenate(rows), dtype=wp.int32),
            wp.array(np.concatenate(columns), dtype=wp.int32),
            wp.array(np.concatenate(blocks), dtype=self.block),
        )

    def assemble(self) -> None:
        """Assemble mass, viscous, outlet and divergence operators."""
        dt = self.dt * SPEED / LENGTH
        nu = self.viscosity / self.density / (LENGTH * SPEED)
        for index, (trial, test) in enumerate(self.u_chunks):
            part = fem.integrate(
                velocity_form,
                fields={"u": trial, "v": test},
                values={"dt": wp.float64(dt), "nu": wp.float64(nu)},
                output_dtype=self.scalar,
            )
            if index == 0:
                self.a_static = part
            else:
                bsr_axpy(part, self.a_static)
        if self.rcr_matrix is not None:
            bsr_axpy(self.rcr_matrix, self.a_static)
        self.b_raw = fem.integrate(
            divergence_form,
            fields={"u": self.u_trial, "q": self.p_test},
            output_dtype=self.scalar,
        )
        self.bt_raw = bsr_transposed(self.b_raw)
        self.b = bsr_copy(self.b_raw)
        wp.launch(
            constrain_divergence_matrix,
            dim=self.b.nrow,
            inputs=[self.b.offsets, self.b.columns, self.b.values, self.constrained],
        )
        self.bt = bsr_transposed(self.b)

    def _constrained_velocity(self, matrix: Any) -> Any:
        result = bsr_copy(matrix)
        wp.launch(
            constrain_velocity_matrix,
            dim=result.nrow,
            inputs=[result.offsets, result.columns, result.values, self.constrained],
        )
        return result

    def update_preconditioner(self) -> None:
        """Factor ``B diag(A)^-1 B^T`` of the current operators with cuDSS."""
        if self.factor is not None:
            self.factor.close()
        rows = self.b.nnz_sync()
        values = self.b.values.numpy()[:rows].reshape(-1, 3)
        columns = self.b.columns.numpy()[:rows]
        offsets = self.b.offsets.numpy()[: self.b.nrow + 1]
        divergence = sparse.csr_matrix(
            (
                values.reshape(-1),
                (columns[:, None] * 3 + np.arange(3)).reshape(-1),
                offsets * 3,
            ),
            shape=(self.n_p, 3 * self.n_u),
        )
        diagonal: Any = bsr_get_diag(self.a)
        inverse = np.linalg.inv(diagonal.numpy())
        block_inverse = sparse.bsr_matrix(
            (inverse, np.arange(self.n_u), np.arange(self.n_u + 1)),
            shape=(3 * self.n_u, 3 * self.n_u),
        ).tocsr()
        schur = (divergence @ block_inverse @ divergence.T).tocsr()
        schur.eliminate_zeros()
        schur.sort_indices()
        self.factor = CuDssFactor(schur, "spd", self.scalar)
        self.inverse_diagonal: Any = wp.array(inverse, dtype=self.block)
        self.provisional = wp.zeros(self.n_u, dtype=self.vector)
        self.gradient = wp.zeros(self.n_u, dtype=self.vector)
        self.preconditioner = LinearOperator(
            self.operator.shape,
            dtype=self.scalar,
            device=wp.get_device(),
            matvec=self._schur_matvec,
        )

    def _velocity(self, array: Any) -> Any:
        return wp.array(
            ptr=array.ptr,
            dtype=self.vector,
            shape=self.n_u,
            device=array.device,
            copy=False,
        )

    def _pressure(self, array: Any) -> Any:
        return wp.array(
            ptr=array.ptr + 3 * self.n_u * wp.types.type_size_in_bytes(self.scalar),
            dtype=self.scalar,
            shape=self.n_p,
            device=array.device,
            copy=False,
        )

    def _saddle_matvec(self, x: Any, y: Any, z: Any, alpha: float, beta: float) -> None:
        """``z = alpha [A B^T; B 0] x + beta y`` with the constrained operators."""
        if beta != 0.0 and y.ptr != z.ptr:
            wp.copy(dest=z, src=y)
        x_u, x_p, z_u, z_p = (
            self._velocity(x),
            self._pressure(x),
            self._velocity(z),
            self._pressure(z),
        )
        bsr_mv(self.a, x_u, z_u, alpha=alpha, beta=beta)
        bsr_mv(self.bt, x_p, z_u, alpha=alpha, beta=1.0)
        bsr_mv(self.b, x_u, z_p, alpha=alpha, beta=beta)

    def _schur_matvec(self, x: Any, y: Any, z: Any, alpha: float, beta: float) -> None:
        """Exact inverse of ``[D B^T; B 0]`` with D the block diagonal of A."""
        assert self.factor is not None
        wp.launch(
            diagonal_apply,
            dim=self.n_u,
            inputs=[self.inverse_diagonal, self._velocity(x), self.provisional],
        )
        wp.copy(dest=self.factor.rhs, src=self._pressure(x))
        bsr_mv(self.b, self.provisional, self.factor.rhs, alpha=1.0, beta=-1.0)
        self.factor.solve()
        bsr_mv(self.bt, self.factor.solution, self.gradient, alpha=1.0, beta=0.0)
        wp.launch(
            velocity_finish,
            dim=self.n_u,
            inputs=[
                self.inverse_diagonal,
                self.provisional,
                self.gradient,
                self._velocity(y),
                self._velocity(z),
                self.scalar(alpha),
                self.scalar(beta),
            ],
        )
        wp.launch(
            pressure_finish,
            dim=self.n_p,
            inputs=[
                self.factor.solution,
                self._pressure(y),
                self._pressure(z),
                self.scalar(alpha),
                self.scalar(beta),
            ],
        )

    def step(
        self, inlet_flow: float, wall_velocity: Optional[np.ndarray] = None
    ) -> dict[str, Any]:
        """Advance one time step and update the outlet Windkessels.

        Args:
            inlet_flow: Volume flow entering through the inlet in m^3/s.
            wall_velocity: Canonical P2 velocity in m/s; only wall nodes are
                used. None keeps the wall at rest.

        Returns:
            GMRES iterations, the relative residual ``|b - A x| / |b|`` of the
            linear solve, the outward flow of every port in m^3/s and the
            pressure of every outlet in Pa.
        """
        lifting_values = self.inlet_shape * inlet_flow
        if wall_velocity is not None:
            lifting_values[self.wall_dofs] = wall_velocity[self.wall_dofs]
        lifting: Any = wp.array(self._to_fem(lifting_values) / SPEED, dtype=self.vector)
        self.a_full = bsr_copy(self.a_static)
        if self.backflow > 0:
            stabilization = fem.integrate(
                backflow_form,
                fields={
                    "previous": self.u_field.trace(),
                    "mesh": self.w_field.trace(),
                    "u": self.side_trial,
                    "v": self.side_test,
                },
                values={"beta": wp.float64(self.backflow)},
                output_dtype=self.scalar,
            )
            bsr_axpy(stabilization, self.a_full)
        self.a = self._constrained_velocity(self.a_full)
        rhs_u = fem.integrate(
            transported_form,
            fields={"u": self.u_field, "w": self.w_field, "v": self.u_test},
            values={"dt": wp.float64(self.dt * SPEED / LENGTH)},
            output_dtype=self.vector,
        )
        load = np.zeros_like(lifting_values)
        for region, windkessel in self.outlets.items():
            load -= windkessel.implicit(self.dt)[1] * self.port_flux[region]
        newton = self.density * SPEED**2 * LENGTH**2
        fem.linalg.array_axpy(
            x=wp.array(self._to_fem(load) / newton, dtype=self.vector), y=rhs_u
        )
        self.rhs_full = wp.clone(rhs_u)
        bsr_mv(self.a_full, lifting, rhs_u, alpha=-1.0, beta=1.0)
        wp.launch(zero_constrained, dim=self.n_u, inputs=[rhs_u, self.constrained])
        rhs = wp.empty(self.operator.shape[0], dtype=self.scalar)
        x = wp.zeros_like(rhs)
        wp.copy(dest=self._velocity(rhs), src=rhs_u)
        bsr_mv(self.b_raw, lifting, self._pressure(rhs), alpha=-1.0, beta=0.0)
        wp.copy(dest=self._velocity(x), src=self.u_field.dof_values)
        wp.launch(
            zero_constrained, dim=self.n_u, inputs=[self._velocity(x), self.constrained]
        )
        wp.copy(dest=self._pressure(x), src=self.pressure_dofs)
        # Warp runs whole restart cycles; one 25-step cycle meets the default
        # tolerances on the aorta in both precisions.
        iterations, error, threshold = gmres(
            self.operator,
            b=rhs,
            x=x,
            M=self.preconditioner,
            tol=self.tolerance,
            atol=0.0,
            maxiter=self.max_iterations,
            restart=25,
            check_every=5,
            use_cuda_graph=False,
        )
        # With right preconditioning Warp recomputes the true residual
        # |b - A x| after every restart cycle; threshold is tolerance |b|,
        # zero at rest with no inflow, where the solution is zero.
        size = float(threshold) / self.tolerance
        residual = float(error) / size if size > 0 else float(error)
        if not residual <= self.tolerance:
            self.log_warning(
                "GMRES stopped after %d iterations at relative residual %.3g",
                iterations,
                residual,
            )
        velocity = self._velocity(x).numpy() + lifting.numpy()
        self.u_field.dof_values.assign(velocity)
        wp.copy(dest=self.pressure_dofs, src=self._pressure(x))
        physical = velocity[self.u_index] * SPEED
        flows = {
            region: float(np.einsum("ij,ij->", vector, physical))
            for region, vector in self.port_flux.items()
        }
        pressures = {
            region: windkessel.advance(flows[region], self.dt)
            for region, windkessel in self.outlets.items()
        }
        return {
            "iterations": int(iterations),
            "residual": residual,
            "flows_m3_s": flows,
            "pressures_pa": pressures,
        }

    def velocity(self) -> np.ndarray:
        """Return the current velocity in m/s at the canonical P2 nodes."""
        velocity = self.u_field.dof_values.numpy()[self.u_index]
        return np.asarray(velocity, dtype=np.float64) * SPEED

    def pressure(self) -> np.ndarray:
        """Return the current pressure in Pa at the vertices."""
        scale = self.density * SPEED**2
        pressure = self.pressure_dofs.numpy()[self.p_index]
        return np.asarray(pressure, dtype=np.float64) * scale

    def wall_force(self) -> np.ndarray:
        """Return the force in N that the fluid exerts on each wall P2 node.

        This is the consistent reaction of the last step's discrete momentum
        equation (pressure, viscous stress and inertia), in canonical order
        and zero away from the wall.
        """
        reaction = wp.zeros(self.n_u, dtype=self.vector)
        bsr_mv(self.a_full, self.u_field.dof_values, reaction, alpha=1.0, beta=0.0)
        bsr_mv(self.bt_raw, self.pressure_dofs, reaction, alpha=1.0, beta=1.0)
        newton = self.density * SPEED**2 * LENGTH**2
        force = (self.rhs_full.numpy() - reaction.numpy())[self.u_index]
        force = np.asarray(force, dtype=np.float64) * newton
        result = np.zeros_like(force)
        result[self.wall_dofs] = force[self.wall_dofs]
        return np.asarray(result)

    def move(self, displacement: np.ndarray, velocity: np.ndarray) -> None:
        """Move the mesh vertices and set the ALE mesh velocity, then reassemble.

        Args:
            displacement: Vertex displacement from the input points in m.
            velocity: Vertex mesh velocity in m/s, used by the backtrace along
                ``u - w``; edge nodes take the mean of their two vertices.
        """
        moved = self.points + displacement
        self.geometry.positions.assign((moved - self.center) / LENGTH)
        mesh_velocity = np.concatenate([velocity, velocity[self.edges].mean(axis=1)])
        self.w_field.dof_values.assign(self._to_fem(mesh_velocity) / SPEED)
        self.geometry.update_bvh()
        self.assemble()

    def close(self) -> None:
        """Release the cuDSS factorization."""
        if self.factor is not None:
            self.factor.close()
            self.factor = None
