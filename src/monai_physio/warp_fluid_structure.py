"""Two-way coupling of incompressible flow and a hyperelastic wall with Warp FEM.

:class:`WarpHyperelasticWall` is a quasi-static compressible Neo-Hookean solid
on quadratic (P2) tetrahedra. It is clamped at chosen vertices, optionally
tethered to its surroundings by springs on its outer surface, and can carry a
prestress so that the given geometry is in equilibrium under a given load.

:class:`WarpFluidStructure` couples it to :class:`WarpNavierStokes` across a
conforming interface whose P2 nodes coincide. Each time step moves the fluid
mesh once, by a harmonic (Laplace) extension of the predicted wall
displacement, and then iterates Dirichlet-Neumann: the fluid takes the wall
velocity, the wall takes the fluid force, and an interface quasi-Newton
method with an inverse Jacobian from a least-squares model (IQN-ILS) updates
the interface displacement.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import torch
import warp as wp
from scipy import sparse
from scipy.spatial import cKDTree
from warp import fem

from .monai_physio_base import MONAIPhysioBase
from .warp_navier_stokes import (
    CuDssFactor,
    WarpNavierStokes,
    tetrahedron_edges,
    triangle_nodes,
)

# P2 triangle mass matrix times 180 / area, nodes (v0, v1, v2, e01, e12, e20).
TRIANGLE_MASS = (
    np.array(
        [
            [6, -1, -1, 0, -4, 0],
            [-1, 6, -1, 0, 0, -4],
            [-1, -1, 6, -4, 0, 0],
            [0, 0, -4, 32, 16, 16],
            [-4, 0, 0, 16, 32, 16],
            [0, -4, 0, 16, 16, 32],
        ]
    )
    / 180.0
)


# Warp 1.17 FEM treats every annotation, including return, as an input.
@fem.integrand
def wall_stress(  # type: ignore[no-untyped-def]
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
    prestress: wp.array(dtype=wp.mat33d),  # type: ignore[valid-type]
    mu: wp.float64,
    lam: wp.float64,
):
    """Prestressed Neo-Hookean first Piola stress ``F S0 + P(F)`` against grad v."""
    f = wp.identity(n=3, dtype=wp.float64) + fem.grad(u, s)  # type: ignore[type-var,arg-type]
    inverse_t = wp.transpose(wp.inverse(f))
    stress = (
        f * prestress[s.qp_index]
        + mu * (f - inverse_t)
        + lam * wp.log(wp.determinant(f)) * inverse_t
    )
    return wp.ddot(stress, fem.grad(v, s))


@fem.integrand
def wall_tangent(  # type: ignore[no-untyped-def]
    s: fem.Sample,
    d: fem.Field,
    u: fem.Field,
    v: fem.Field,
    prestress: wp.array(dtype=wp.mat33d),  # type: ignore[valid-type]
    mu: wp.float64,
    lam: wp.float64,
):
    """Tangent of :func:`wall_stress` at the displacement ``d``."""
    f = wp.identity(n=3, dtype=wp.float64) + fem.grad(d, s)  # type: ignore[type-var,arg-type]
    inverse_t = wp.transpose(wp.inverse(f))
    h = fem.grad(u, s)
    stress = (
        h * prestress[s.qp_index]
        + mu * h
        + (mu - lam * wp.log(wp.determinant(f)))
        * inverse_t
        * wp.transpose(h)
        * inverse_t
        + lam * wp.ddot(inverse_t, h) * inverse_t
    )
    return wp.ddot(stress, fem.grad(v, s))


@fem.integrand
def linear_stress(  # type: ignore[no-untyped-def]
    s: fem.Sample, u: fem.Field, mu: wp.float64, lam: wp.float64
):
    """Small-strain stress, used as the prestress S0."""
    h = fem.grad(u, s)
    identity = wp.identity(n=3, dtype=wp.float64)  # type: ignore[type-var,arg-type]
    return mu * (h + wp.transpose(h)) + lam * wp.trace(h) * identity


def to_scipy(matrix: Any) -> sparse.csr_matrix:
    """Convert a Warp BSR matrix of 3 x 3 blocks to SciPy CSR."""
    count = matrix.nnz_sync()
    return sparse.bsr_matrix(
        (
            matrix.values.numpy()[:count],
            matrix.columns.numpy()[:count],
            matrix.offsets.numpy()[: matrix.nrow + 1],
        ),
        shape=(3 * matrix.nrow, 3 * matrix.ncol),
    ).tocsr()


class WarpHyperelasticWall(MONAIPhysioBase):
    """Quasi-static compressible Neo-Hookean wall on P2 tetrahedra.

    The first Piola stress is ``F S0 + mu (F - F^-T) + lambda ln(J) F^-T``.
    ``S0`` is a prestress stored at every quadrature point (zero until
    :meth:`prestress`). Nonlinear solves use modified Newton iterations with
    a tangent factored by cuDSS and refreshed only when convergence slows,
    so most solves reuse one factorization. Everything runs in float64: in
    float32 the internal force is too inexact for the Newton tolerance.
    Nodal arrays are in the canonical P2 order: vertices, then :attr:`edges`.
    """

    def __init__(
        self,
        points: np.ndarray,
        tetrahedra: np.ndarray,
        clamped: np.ndarray,
        *,
        young_modulus: float,
        poisson_ratio: float,
        tether_faces: Optional[np.ndarray] = None,
        tether_stiffness: float = 0.0,
        quadrature_order: int = 4,
    ) -> None:
        """Build the P2 space and the boundary operators.

        Args:
            points: Vertex positions in metres.
            tetrahedra: Positively oriented vertex indices of each tetrahedron.
            clamped: True for vertices held fixed; an edge node is fixed when
                both of its vertices are.
            young_modulus: E in Pa.
            poisson_ratio: nu.
            tether_faces: Boundary triangles carrying the tether springs.
            tether_stiffness: Spring stiffness per area in Pa/m.
            quadrature_order: Order of the cell quadrature.
        """
        super().__init__(class_name=self.__class__.__name__)
        self.points = np.asarray(points, dtype=np.float64)
        self.tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
        self.edges = tetrahedron_edges(self.tetrahedra)
        self.dof_points = np.concatenate(
            [self.points, self.points[self.edges].mean(axis=1)]
        )
        self.mu = young_modulus / (2 * (1 + poisson_ratio))
        self.lam = (
            young_modulus
            * poisson_ratio
            / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        )
        self.geometry: Any = fem.Tetmesh(
            tet_vertex_indices=wp.array(self.tetrahedra, dtype=wp.int32),
            positions=wp.array(self.points, dtype=wp.vec3d),
        )
        self.space: Any = fem.make_polynomial_space(
            self.geometry, degree=2, dtype=wp.vec3d
        )
        distance, index = cKDTree(self.space.node_positions().numpy()).query(
            self.dof_points
        )
        self.index = np.asarray(index, dtype=np.int64)
        if np.max(distance) > 1e-12 or len(np.unique(self.index)) != len(self.index):
            raise ValueError("Warp FEM nodes do not match the canonical P2 nodes")
        mask = np.asarray(clamped, dtype=bool)
        self.clamped = np.r_[mask, mask[self.edges].all(axis=1)]
        fixed = np.zeros(len(self.dof_points), dtype=bool)
        fixed[self.index[self.clamped]] = True
        self.free = np.flatnonzero(np.repeat(~fixed, 3))

        domain = fem.Cells(self.geometry)
        self.quadrature = fem.RegularQuadrature(domain, order=quadrature_order)
        self.test = fem.make_test(space=self.space, domain=domain)
        self.trial = fem.make_trial(space=self.space, domain=domain)
        self.field: Any = self.space.make_field()
        self.s0: Any = wp.zeros(self.quadrature.total_point_count(), dtype=wp.mat33d)
        self.tether = self._tether(tether_faces, tether_stiffness)
        self.factor: Optional[CuDssFactor] = None

    def _tether(
        self, faces: Optional[np.ndarray], stiffness: float
    ) -> sparse.csr_matrix:
        """Spring stiffness ``k int d . v dA`` over the tether faces, Warp order."""
        size = 3 * len(self.dof_points)
        if faces is None or stiffness == 0:
            return sparse.csr_matrix((size, size))
        faces = np.asarray(faces, dtype=np.int64)
        nodes = self.index[triangle_nodes(faces, self.edges, len(self.points))]
        corners = self.points[faces]
        area = (
            np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
            / 2
        )
        # Entry (3 i + c, 3 j + c) of face f, broadcast to shape (f, i, j, c).
        shape = (len(faces), 6, 6, 3)
        values = stiffness * area[:, None, None, None] * TRIANGLE_MASS[..., None]
        rows = 3 * nodes[:, :, None, None] + np.arange(3)
        columns = 3 * nodes[:, None, :, None] + np.arange(3)
        return sparse.coo_matrix(
            (
                np.broadcast_to(values, shape).ravel(),
                (
                    np.broadcast_to(rows, shape).ravel(),
                    np.broadcast_to(columns, shape).ravel(),
                ),
            ),
            shape=(size, size),
        ).tocsr()

    def _tangent(self, tether: bool) -> sparse.csr_matrix:
        """Tangent at the current displacement on the free degrees of freedom."""
        matrix: Any = to_scipy(
            fem.integrate(
                wall_tangent,
                quadrature=self.quadrature,
                fields={"d": self.field, "u": self.trial, "v": self.test},
                values={"prestress": self.s0, "mu": self.mu, "lam": self.lam},
                output_dtype=wp.float64,
            )
        )
        if tether:
            matrix = matrix + self.tether
        return sparse.csr_matrix(matrix[self.free][:, self.free])

    def _refactor(self, matrix: sparse.csr_matrix) -> None:
        if self.factor is not None:
            self.factor.close()
        matrix.sort_indices()
        self.factor = CuDssFactor(matrix, "spd")

    def _warp_order(self, values: np.ndarray) -> np.ndarray:
        result = np.zeros((len(self.dof_points), 3))
        result[self.index] = values
        return result.reshape(-1)

    def pressure_load(self, faces: np.ndarray, pressure: float) -> np.ndarray:
        """Nodal force in N of a uniform pressure pushing along the face normals.

        On a flat triangle the P2 vertex functions integrate to zero and each
        edge function to a third of the area.
        """
        faces = np.asarray(faces, dtype=np.int64)
        corners = self.points[faces]
        area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        load = np.zeros((len(self.dof_points), 3))
        nodes = triangle_nodes(faces, self.edges, len(self.points))
        for k in range(3, 6):
            np.add.at(load, nodes[:, k], pressure * area / 6)
        return load

    def prestress(self, load: np.ndarray) -> None:
        """Store a prestress that balances ``load`` in the given geometry.

        ``S0`` is the small-strain stress of the clamped wall under ``load``
        without tethers, which is in exact discrete equilibrium with ``load``
        at zero displacement. The tangent is then refactored with ``S0``.
        """
        started = time.perf_counter()
        self.s0.zero_()
        self.field.dof_values.zero_()
        self._refactor(self._tangent(tether=False))
        assert self.factor is not None
        self.factor.rhs.assign(self._warp_order(load)[self.free])
        self.factor.solve()
        solution = np.zeros(3 * len(self.dof_points))
        solution[self.free] = self.factor.solution.numpy()
        self.field.dof_values.assign(solution.reshape(-1, 3))
        fem.interpolate(
            linear_stress,
            dest=self.s0,
            at=self.quadrature,
            fields={"u": self.field},
            values={"mu": self.mu, "lam": self.lam},
        )
        self.field.dof_values.zero_()
        self._refactor(self._tangent(tether=True))
        self.log_info(
            "Prestress: max displacement without it %.3g mm, %.1f s",
            1e3 * np.abs(solution).max(),
            time.perf_counter() - started,
        )

    def residual(self, load: np.ndarray) -> np.ndarray:
        """Return ``load - internal force - tether force`` on the free DOFs."""
        internal = fem.integrate(
            wall_stress,
            quadrature=self.quadrature,
            fields={"u": self.field, "v": self.test},
            values={"prestress": self.s0, "mu": self.mu, "lam": self.lam},
            output_dtype=wp.vec3d,
        )
        displacement = self.field.dof_values.numpy().reshape(-1)
        force = (
            self._warp_order(load)
            - internal.numpy().reshape(-1)
            - self.tether @ displacement
        )
        return np.asarray(force[self.free])

    def solve(
        self, load: np.ndarray, tolerance: float = 1e-6, max_iterations: int = 30
    ) -> dict[str, Any]:
        """Find the equilibrium displacement under nodal forces ``load`` (N).

        Modified Newton starts from the current displacement and stops when
        the free-DOF residual is below ``tolerance`` times the load norm, or
        warns after ``max_iterations``. The
        factored tangent is reused across solves and refactored at the
        current displacement whenever an iteration fails to halve the
        residual.
        """
        if self.factor is None:
            self._refactor(self._tangent(tether=True))
        scale = max(float(np.linalg.norm(self._warp_order(load)[self.free])), 1e-300)
        displacement = self.field.dof_values.numpy().reshape(-1)
        previous, updates = np.inf, 0
        for iteration in range(max_iterations + 1):
            residual = self.residual(load)
            relative = float(np.linalg.norm(residual)) / scale
            if not np.isfinite(relative):
                raise RuntimeError("Wall solve diverged")
            if relative <= tolerance or iteration == max_iterations:
                break
            if relative > 0.5 * previous:
                self._refactor(self._tangent(tether=True))
                updates += 1
            previous = relative
            assert self.factor is not None
            self.factor.rhs.assign(residual)
            self.factor.solve()
            displacement[self.free] += self.factor.solution.numpy()
            self.field.dof_values.assign(displacement.reshape(-1, 3))
        if relative > tolerance:
            self.log_warning(
                "Wall solve stopped after %d iterations at relative residual %.3g",
                iteration,
                relative,
            )
        return {
            "iterations": iteration,
            "residual": relative,
            "tangent_updates": updates,
        }

    def displacement(self) -> np.ndarray:
        """Return the displacement in m at the canonical P2 nodes."""
        return np.asarray(self.field.dof_values.numpy()[self.index])

    def close(self) -> None:
        """Release the cuDSS factorization."""
        if self.factor is not None:
            self.factor.close()
            self.factor = None


def laplacian(points: np.ndarray, tetrahedra: np.ndarray) -> sparse.csr_matrix:
    """Linear (P1) finite-element Laplacian of a tetrahedral mesh."""
    corners = points[tetrahedra]
    edges = corners[:, 1:] - corners[:, :1]
    inverse = np.linalg.inv(edges)
    gradients = np.concatenate([-inverse.sum(axis=2, keepdims=True), inverse], axis=2)
    volume = np.abs(np.linalg.det(edges)) / 6
    local = np.einsum("tki,tkj->tij", gradients, gradients) * volume[:, None, None]
    rows = np.repeat(tetrahedra, 4, axis=1)
    columns = np.tile(tetrahedra, (1, 4))
    size = len(points)
    return sparse.coo_matrix(
        (local.ravel(), (rows.ravel(), columns.ravel())), shape=(size, size)
    ).tocsr()


def quasi_newton_update(
    residual: torch.Tensor,
    columns: list[tuple[torch.Tensor, torch.Tensor, int]],
    relaxation: float,
    filter_tolerance: float,
) -> torch.Tensor:
    """Change of the interface guess for ``residual`` (IQN-ILS, relaxed).

    ``columns`` holds (residual difference, response difference, step) pairs,
    newest first. Columns whose part orthogonal to the newer ones is below
    ``filter_tolerance`` of their norm are removed from it in place. See
    :class:`WarpFluidStructure` for the method.
    """
    update = relaxation * residual
    while columns:
        v = torch.stack([column[0] for column in columns], dim=1)
        q, r = torch.linalg.qr(v)
        norms = torch.linalg.vector_norm(v, dim=0)
        small = r.diagonal().abs() <= filter_tolerance * norms
        if small.any():
            del columns[int(torch.nonzero(small)[0])]
            continue
        c = torch.linalg.solve_triangular(r, -(q.T @ residual)[:, None], upper=True)
        w = torch.stack([column[1] for column in columns], dim=1)
        update += ((w - (1 - relaxation) * v) @ c)[:, 0]
        break
    return update


class WarpFluidStructure(MONAIPhysioBase):
    """Partitioned two-way coupling of :class:`WarpNavierStokes` and a wall.

    The wall is prestressed so that the given geometry is its equilibrium at
    ``reference_pressure`` (uniform on the interface). Each :meth:`step`:

    1. predicts the interface displacement by linear extrapolation, extends
       it harmonically into the fluid and moves the fluid mesh (once per
       step; interface caps stay fixed);
    2. repeats until the interface displacement converges: fluid step with
       the wall velocity of the current guess, wall solve under the fluid
       force, quasi-Newton update of the guess.

    The update is IQN-ILS (Degroote, Bathe and Vierendeels, Computers and
    Structures 87:793-801, 2009). With guess ``x``, wall response ``x~`` and
    residual ``r = x~ - x``, every iteration stores the differences of ``r``
    and of ``x~`` from the previous iteration as columns of ``V`` and ``W``.
    With ``c`` minimizing ``|V c + r|``, the guess moves by ``(W - V) c``,
    the quasi-Newton step for the part of ``r`` that the columns explain,
    plus ``relaxation`` times the rest, ``r + V c`` (all of ``r`` while there
    are no columns). With ``relaxation = 1`` after the first iteration this
    is IQN-ILS; a light wall needs less, because an unrelaxed step along a
    direction not yet sampled can invert it. This relaxed form is Anderson
    mixing (Walker and Ni, SIAM J. Numer. Anal. 49:1715-1735, 2011).
    Columns of the last ``reuse`` steps are kept, newest first; a column whose
    part orthogonal to the newer ones is below ``filter_tolerance`` of its
    norm is dropped (the QR2 filter of Haelterman et al., Computers and
    Structures 171:9-17, 2016).

    A step converges when the RMS interface residual is at most
    ``tolerance`` times the RMS interface motion of the step (guess minus
    the start-of-step displacement) plus ``absolute_tolerance``. The
    reported coupling residual is the relative part only, so a converged
    step can report more than ``tolerance`` when its motion is small.

    A step still unconverged after ``max_iterations`` is solved once more at
    its lowest-residual iterate, which it keeps (so the fluid and wall states
    belong to the kept interface), and the columns are restored to those of
    the earlier steps. Such a step usually stalls at the residual noise of
    the fluid and wall solves, and columns of that noise would mislead the
    next steps. It is reported, not rejected; its re-solve may also meet
    the convergence criterion.
    """

    def __init__(
        self,
        fluid: WarpNavierStokes,
        wall: WarpHyperelasticWall,
        boundary_regions: np.ndarray,
        *,
        reference_pressure: float,
        relaxation: float = 0.1,
        reuse: int = 2,
        filter_tolerance: float = 1e-2,
        tolerance: float = 1e-3,
        absolute_tolerance: float = 1e-9,
        max_iterations: int = 20,
    ) -> None:
        """Match the interface, prestress the wall and factor the mesh extension.

        Args:
            fluid: Flow solver whose wall (region 0) nodes are wall nodes.
            wall: Wall solver sharing the interface nodes.
            boundary_regions: Region of each fluid boundary face (0 = wall).
            reference_pressure: Uniform interface pressure (Pa) at which the
                given geometry is in equilibrium.
            relaxation: Relaxation of the residual part that the
                quasi-Newton columns do not explain (all of it without
                columns).
            reuse: Number of previous steps whose quasi-Newton columns are
                kept.
            filter_tolerance: Relative QR filter threshold of the columns.
            tolerance: Stop when the RMS interface residual is at most this
                fraction of the RMS interface motion over the step...
            absolute_tolerance: ...plus this many metres.
            max_iterations: Coupling iteration limit per step, after which
                the best iterate is solved once more and kept.
        """
        super().__init__(class_name=self.__class__.__name__)
        self.fluid, self.wall = fluid, wall
        self.relaxation, self.tolerance = float(relaxation), float(tolerance)
        self.reuse, self.filter_tolerance = int(reuse), float(filter_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.max_iterations = int(max_iterations)
        # (residual difference, response difference, step) columns, newest
        # first, kept on the GPU, where their QR factorization is cheap.
        self.columns: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        self.step_count = 0
        self.device = wp.device_to_torch(wp.get_device())
        distance, nodes = cKDTree(wall.dof_points).query(
            fluid.dof_points[fluid.wall_dofs]
        )
        if np.max(distance) > 1e-9:
            raise ValueError("Fluid wall nodes are not wall nodes")
        nodes = np.asarray(nodes, dtype=np.int64)
        self.interface = nodes
        vertex_count = len(fluid.points)
        self.wall_vertices = fluid.wall_dofs < vertex_count
        to_wall = np.full(vertex_count, -1)
        to_wall[fluid.wall_dofs[self.wall_vertices]] = nodes[self.wall_vertices]
        wall_faces = to_wall[fluid.face_dofs[np.asarray(boundary_regions) == 0, :3]]
        wall.prestress(wall.pressure_load(wall_faces, reference_pressure))

        # Harmonic mesh extension: wall vertices follow the wall, caps stay put.
        boundary = np.zeros(vertex_count, dtype=bool)
        boundary[fluid.face_dofs[:, :3].ravel()] = True
        self.boundary = np.flatnonzero(boundary)
        self.inner = np.flatnonzero(~boundary)
        matrix: Any = laplacian(fluid.points, fluid.tetrahedra)
        self.coupling = matrix[self.inner][:, self.boundary].tocsr()
        inner = matrix[self.inner][:, self.inner].tocsr()
        inner.sort_indices()
        self.extension = CuDssFactor(inner, "spd")
        self.boundary_values = np.zeros((vertex_count, 3))
        self.mesh_displacement = np.zeros((vertex_count, 3))
        self.previous = np.zeros((len(self.interface), 3))
        self.volume = self.fluid_volume()

    def extend(self, interface_displacement: np.ndarray) -> np.ndarray:
        """Extend the interface displacement harmonically to all fluid vertices."""
        values = self.boundary_values
        values[self.fluid.wall_dofs[self.wall_vertices]] = interface_displacement[
            self.wall_vertices
        ]
        result = values.copy()
        known = values[self.boundary]
        for axis in range(3):
            self.extension.rhs.assign(-(self.coupling @ known[:, axis]))
            self.extension.solve()
            result[self.inner, axis] = self.extension.solution.numpy()
        return result

    def fluid_volume(self) -> float:
        """Volume in m^3 of the current (moved) fluid mesh."""
        corners = (self.fluid.points + self.mesh_displacement)[self.fluid.tetrahedra]
        edges = corners[:, 1:] - corners[:, :1]
        return float(np.linalg.det(edges).sum() / 6)

    def step(self, inlet_flow: float) -> dict[str, Any]:
        """Advance the coupled system one fluid time step.

        Returns:
            The last fluid step result plus coupling iterations (fluid and
            wall solves; ``max_iterations + 1`` for a capped step, whose
            re-solve can still meet the criterion), whether the interface
            converged, the relative residual of the kept iterate, the summed
            GMRES and wall Newton iterations, the wall tangent
            refactorizations, the interface displacement extremes in m and
            the rate of change of the fluid volume in m^3/s.
        """
        fluid, wall, dt = self.fluid, self.wall, self.fluid.dt
        current = wall.displacement()[self.interface]
        guess = 2 * current - self.previous
        moved = self.extend(guess)
        fluid.move(moved, (moved - self.mesh_displacement) / dt)
        self.mesh_displacement = moved
        volume = self.fluid_volume()
        start_velocity = wp.clone(fluid.u_field.dof_values)
        start_pressure = {r: w.pressure for r, w in fluid.outlets.items()}
        wall_velocity = np.zeros_like(fluid.dof_points)
        load = np.zeros_like(wall.dof_points)
        self.step_count += 1
        self.columns = [c for c in self.columns if c[2] >= self.step_count - self.reuse]
        # The QR filter may drop earlier columns in favour of this step's.
        earlier = list(self.columns)
        last: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        gmres = newton = updates = 0
        best = (np.inf, guess)
        for iteration in range(1, self.max_iterations + 2):
            if iteration > self.max_iterations:
                self.columns, guess = earlier, best[1]
            if iteration > 1:
                fluid.u_field.dof_values.assign(start_velocity)
                for region, windkessel in fluid.outlets.items():
                    windkessel.pressure = start_pressure[region]
            wall_velocity[fluid.wall_dofs] = (guess - current) / dt
            result = fluid.step(inlet_flow, wall_velocity)
            load[self.interface] = fluid.wall_force()[fluid.wall_dofs]
            solid = wall.solve(load)
            gmres += result["iterations"]
            newton += solid["iterations"]
            updates += solid["tangent_updates"]
            response = wall.displacement()[self.interface]
            residual = response - guess
            size = np.sqrt(np.mean(residual**2))
            change = np.sqrt(np.mean((guess - current) ** 2))
            relative = float(size / max(change, 1e-300))
            converged = size <= self.tolerance * change + self.absolute_tolerance
            if (
                converged
                or iteration > self.max_iterations
                or not np.all(np.isfinite(residual))
            ):
                break
            if size < best[0]:
                best = (size, guess)
            pair = (
                torch.from_numpy(residual.ravel()).to(self.device),
                torch.from_numpy(response.ravel()).to(self.device),
            )
            if last is not None:
                self.columns.insert(
                    0, (pair[0] - last[0], pair[1] - last[1], self.step_count)
                )
            last = pair
            update = quasi_newton_update(
                pair[0], self.columns, self.relaxation, self.filter_tolerance
            )
            guess = guess + update.cpu().numpy().reshape(guess.shape)
        if not np.all(np.isfinite(residual)):
            raise RuntimeError("Coupled step diverged")
        self.previous = current
        displacement = wall.displacement()[self.interface]
        magnitude = np.linalg.norm(displacement, axis=1)
        volume_rate = (volume - self.volume) / dt
        self.volume = volume
        return {
            **result,
            "coupling_iterations": iteration,
            "converged": bool(converged),
            "coupling_residual": relative,
            "gmres_iterations": gmres,
            "wall_iterations": newton,
            "wall_tangent_updates": updates,
            "max_displacement_m": float(magnitude.max()),
            "mean_displacement_m": float(magnitude.mean()),
            "volume_rate_m3_s": volume_rate,
        }

    def close(self) -> None:
        """Release the factorizations of the fluid, the wall and the extension."""
        self.fluid.close()
        self.wall.close()
        self.extension.close()
