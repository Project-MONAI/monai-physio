"""Quasi-Newton update and analytic straight-tube checks of the Warp FSI classes."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial import Delaunay
from scipy.special import jv

from monai_physio.warp_fluid_structure import (
    WarpFluidStructure,
    WarpHyperelasticWall,
    quasi_newton_update,
)
from monai_physio.warp_navier_stokes import WarpNavierStokes, Windkessel
from tests.test_warp_navier_stokes import disk, extrude, outer_faces, require_cudss

RADIUS, THICKNESS, LENGTH = 0.005, 0.001, 0.1
YOUNG, POISSON, DENSITY, VISCOSITY = 1e5, 0.45, 1060.0, 0.004


def solve_linear_fixed_point(
    matrix: torch.Tensor,
    offset: torch.Tensor,
    columns: list[tuple[torch.Tensor, torch.Tensor, int]],
    step: int,
) -> int:
    """Iterations to solve x = offset - matrix x as WarpFluidStructure.step does.

    The response ``offset - matrix x`` stands in for the fluid and wall
    solves. Returns the number of response evaluations needed to reach a
    relative residual of 1e-8.
    """
    guess = torch.zeros_like(offset)
    last = None
    for iteration in range(1, 201):
        response = offset - matrix @ guess
        residual = response - guess
        if torch.linalg.vector_norm(residual) <= 1e-8 * torch.linalg.vector_norm(
            offset
        ):
            return iteration
        if last is not None:
            columns.insert(0, (residual - last[0], response - last[1], step))
        last = residual, response
        guess = guess + quasi_newton_update(residual, columns, 0.1, 1e-2)
    return 200


def test_quasi_newton_linear_convergence_and_reuse() -> None:
    """A linear 6-dimensional map converges in n + 2 updates; reuse cuts that.

    Its eigenvalues 0.5-5 make plain relaxation by 0.1 need over 100 updates.
    A second solve with another offset that keeps the first solve's columns
    needs only a couple.
    """
    generator = torch.Generator().manual_seed(0)
    size = 6
    basis = torch.linalg.qr(
        torch.randn(size, size, generator=generator, dtype=torch.float64)
    )[0]
    eigenvalues = torch.linspace(0.5, 5.0, size, dtype=torch.float64)
    matrix = basis @ torch.diag(eigenvalues) @ basis.T
    first, second = torch.randn(2, size, generator=generator, dtype=torch.float64)
    columns: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    assert solve_linear_fixed_point(matrix, first, columns, 0) - 1 <= size + 2
    fresh = solve_linear_fixed_point(matrix, second, [], 1)
    reused = solve_linear_fixed_point(matrix, second, columns, 1)
    assert reused <= 3 < fresh


def test_quasi_newton_filters_dependent_columns() -> None:
    """Duplicate and zero columns are dropped; one column gives its exact step."""
    generator = torch.Generator().manual_seed(1)
    v, w, residual = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    columns = [(v, w, 0), (v.clone(), w.clone(), 0), (torch.zeros_like(v), w, 0)]
    update = quasi_newton_update(residual, columns, 0.1, 1e-2)
    assert len(columns) == 1
    c = -(v @ residual) / (v @ v)
    expected = 0.1 * residual + (w - 0.9 * v) * c
    torch.testing.assert_close(update, expected)


def test_quasi_newton_keeps_at_most_residual_length_columns() -> None:
    """Three columns for a two-component residual: the two newest are used."""
    generator = torch.Generator().manual_seed(2)
    v, w = torch.randn(2, 2, 3, generator=generator, dtype=torch.float64)
    residual = torch.randn(2, generator=generator, dtype=torch.float64)
    columns = [(v[:, k], w[:, k], 0) for k in range(3)]
    update = quasi_newton_update(residual, columns, 0.1, 1e-2)
    assert len(columns) == 2
    c = torch.linalg.solve(v[:, :2], -residual)
    torch.testing.assert_close(update, 0.1 * residual + (w - 0.9 * v)[:, :2] @ c)


class ScriptedSolvers:
    """Fluid and wall stand-in whose interface residual follows a script.

    Starting from rest, each wall solve sets the displacement of the 4
    interface nodes to the guess (the wall velocity times dt) plus the next
    scripted residual in every component, and records that guess. Each fluid
    step raises the outlet pressure by 1 Pa, as a stand-in for its state.
    """

    def __init__(self, residuals: list[float]) -> None:
        self.residuals = list(residuals)
        self.guesses: list[np.ndarray] = []
        self.dt = 1.0
        self.outlets = {1: Windkessel(0.0, 1.0, 1.0, 100.0)}
        self.points = np.eye(4, 3)
        self.tetrahedra = np.array([[3, 0, 1, 2]])
        self.dof_points = self.points
        self.wall_dofs = np.arange(4)
        self.u_field = SimpleNamespace(dof_values=wp.zeros(4, device="cpu"))
        self.value = np.zeros((4, 3))
        self.velocity = np.zeros((4, 3))

    def move(self, displacement: np.ndarray, velocity: np.ndarray) -> None:
        """Ignore the mesh motion."""

    def step(self, inlet_flow: float, wall_velocity: np.ndarray) -> dict[str, Any]:
        """Record the wall velocity of the guess and advance the outlet."""
        self.velocity = wall_velocity.copy()
        self.outlets[1].pressure += 1.0
        return {"iterations": 1}

    def wall_force(self) -> np.ndarray:
        """No force."""
        return np.zeros((4, 3))

    def solve(self, load: np.ndarray) -> dict[str, Any]:
        """Respond with the guess plus the next scripted residual."""
        guess = self.velocity * self.dt
        self.guesses.append(guess)
        self.value = guess + self.residuals.pop(0)
        return {"iterations": 1, "tangent_updates": 0}

    def displacement(self) -> np.ndarray:
        """Current interface displacement."""
        return self.value


def test_capped_step_keeps_best_iterate_and_drops_all_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the iteration cap the best iterate is solved again and kept.

    The re-solve starts from the step-start fluid state. All quasi-Newton
    columns are dropped, the capped step's and the reused earlier step's,
    so the next step starts without reused columns.
    """
    solvers = ScriptedSolvers([1e-3, 1e-5, 1e-2, 1e-4])
    coupled = WarpFluidStructure.__new__(WarpFluidStructure)
    coupled.fluid = coupled.wall = solvers  # type: ignore[assignment]
    coupled.relaxation, coupled.filter_tolerance, coupled.reuse = 0.1, 1e-2, 2
    coupled.tolerance = coupled.absolute_tolerance = 0.0
    coupled.max_iterations = 3
    earlier = (
        torch.full((12,), 0.5, dtype=torch.float64),
        torch.linspace(0.0, 1.0, 12, dtype=torch.float64),
        0,
    )
    coupled.columns, coupled.step_count = [earlier], 0
    coupled.device = "cpu"
    coupled.interface = np.arange(4)
    coupled.mesh_displacement = coupled.previous = np.zeros((4, 3))
    coupled.volume = 0.0
    monkeypatch.setattr(coupled, "extend", lambda displacement: np.zeros((4, 3)))
    result = coupled.step(0.0)
    assert result["coupling_iterations"] == 4 and not result["converged"]
    np.testing.assert_array_equal(solvers.guesses[3], solvers.guesses[1])
    np.testing.assert_allclose(solvers.value, solvers.guesses[1] + 1e-4)
    motion = np.sqrt(np.mean(solvers.guesses[1] ** 2))
    assert result["coupling_residual"] == pytest.approx(1e-4 / motion)
    assert solvers.outlets[1].pressure == 101.0
    assert coupled.columns == []


def compact(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the used vertex ids and the tetrahedra renumbered over them."""
    ids, inverse = np.unique(tets, return_inverse=True)
    return ids, inverse.reshape(tets.shape)


def tube_mesh(
    rings: int = 4, wall_rings: int = 2, layers: int = 25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structured blood core and wall annulus extruded along z.

    Returns the shared points and the fluid and wall tetrahedra (about 10k
    each with the defaults).
    """
    core = disk(RADIUS, rings)
    rim = core[-8 * rings :]
    xy = np.concatenate(
        [core]
        + [
            rim * (1 + THICKNESS / RADIUS * k / wall_rings)
            for k in range(1, wall_rings + 1)
        ]
    )
    triangles = np.sort(Delaunay(xy).simplices, axis=1)
    radial = np.linalg.norm(xy, axis=1)[triangles]
    fluid = np.all(radial < RADIUS * (1 + 1e-9), axis=1)
    assert np.all(fluid ^ np.all(radial > RADIUS * (1 - 1e-9), axis=1))
    points, fluid_tets = extrude(xy, triangles[fluid], LENGTH, layers)
    return points, fluid_tets, extrude(xy, triangles[~fluid], LENGTH, layers)[1]


def tube_wall(points: np.ndarray, tets: np.ndarray) -> WarpHyperelasticWall:
    """Wall with clamped end annuli and no tether."""
    ids, local = compact(tets)
    z = points[ids, 2]
    return WarpHyperelasticWall(
        points[ids],
        local,
        (z < 1e-12) | (z > LENGTH - 1e-12),
        young_modulus=YOUNG,
        poisson_ratio=POISSON,
    )


def lame_compliance() -> float:
    """Plane-strain thick-wall inner radius change per pressure (m/Pa)."""
    a, b = RADIUS, RADIUS + THICKNESS
    return (
        a * (1 + POISSON) * ((1 - 2 * POISSON) * a**2 + b**2) / (YOUNG * (b**2 - a**2))
    )


def expected_wave_speed(steps_per_period: int) -> tuple[float, float]:
    """Bramwell-Hill speed, and that speed as this discrete tube should show it.

    At the quarter-wave frequency, viscosity slows the wave by the Womersley
    factor Re(1 / sqrt(1 - F10)), and backward Euler with ``steps_per_period``
    steps lowers the ringing frequency by atan(w dt) / (w dt).
    """
    speed = np.sqrt(RADIUS / (2 * DENSITY * lame_compliance()))
    omega = 2 * np.pi * speed / (4 * LENGTH)
    z = RADIUS * np.sqrt(omega * DENSITY / VISCOSITY) * 1j**1.5
    womersley = 2 * jv(1, z) / (z * jv(0, z))
    step = 2 * np.pi / steps_per_period
    viscous = speed / np.real(1 / np.sqrt(1 - womersley))
    return float(speed), float(viscous * np.arctan(step) / step)


@pytest.mark.requires_gpu
def test_wall_matches_lame_thick_wall() -> None:
    """Inner radius change under 100 Pa matches the Lame solution within 5%.

    Prestressed by the same load, the undeformed tube is then in equilibrium.
    """
    require_cudss()
    points, _, solid = tube_mesh()
    wall = tube_wall(points, solid)
    try:
        faces = outer_faces(wall.tetrahedra)
        radial = np.linalg.norm(wall.points[:, :2], axis=1)
        inner = faces[np.all(radial[faces] < RADIUS * (1 + 1e-9), axis=1)]
        # Solid faces point into the lumen; the blood pressure pushes outward.
        load = wall.pressure_load(inner[:, [0, 2, 1]], 100.0)
        result = wall.solve(load)
        assert result["residual"] <= 1e-6
        nodes = np.flatnonzero(
            (radial < RADIUS * (1 + 1e-9))
            & (np.abs(wall.points[:, 2] - LENGTH / 2) < LENGTH / 6)
        )
        displacement = wall.displacement()[nodes]
        outward = np.sum(displacement[:, :2] * wall.points[nodes, :2], axis=1)
        measured = float(np.mean(outward / radial[nodes]))
        expected = 100.0 * lame_compliance()
        print(f"Lame: measured {measured:.4e} m, expected {expected:.4e} m")
        assert abs(measured / expected - 1) < 0.05
        wall.prestress(load)
        wall.solve(load)
        assert np.abs(wall.displacement()).max() < 1e-3 * measured
    finally:
        wall.close()


@pytest.mark.requires_gpu
def test_pulse_wave_speed_in_closed_tube() -> None:
    """Coupled quarter-wave ringing gives the expected wave speed within 5%.

    The tube is closed at z = 0 and its outlet pressure steps from 0 to
    100 Pa. The closed-end pressure then rings with period 4 L / c. Twenty
    steps per period put the added-mass ratio of the Dirichlet-Neumann
    iteration near (20 / 2 pi)^2 = 10, within reach of the default
    relaxation 0.1; every step must converge, in at most 6 quasi-Newton
    iterations on average (about 4 to 5 are typical).
    """
    require_cudss()
    points, fluid_tets, solid_tets = tube_mesh()
    ids, local = compact(fluid_tets)
    faces = outer_faces(local)
    z = points[ids][faces, 2]
    regions = np.zeros(len(faces), dtype=np.int64)
    regions[np.all(z < 1e-12, axis=1)] = 1
    regions[np.all(z > LENGTH - 1e-12, axis=1)] = 2
    bramwell_hill, expected = expected_wave_speed(20)
    dt = 4 * LENGTH / bramwell_hill / 20
    fluid = WarpNavierStokes(
        points[ids],
        local,
        faces,
        regions,
        inlet_region=1,
        outlets={2: Windkessel(0.0, 1e30, 1e30, 100.0)},
        dt=dt,
        density=DENSITY,
        viscosity=VISCOSITY,
    )
    coupled = WarpFluidStructure(
        fluid, tube_wall(points, solid_tets), regions, reference_pressure=0.0
    )
    closed = np.flatnonzero(fluid.points[:, 2] < 1e-12)
    times, pressures, iterations, converged = [], [], [], []
    try:
        for index in range(44):
            result = coupled.step(0.0)
            times.append((index + 1) * dt)
            pressures.append(fluid.pressure()[closed].mean())
            iterations.append(result["coupling_iterations"])
            converged.append(result["converged"])
    finally:
        coupled.close()
    signal = np.asarray(pressures) - 100.0
    t = np.asarray(times)
    rising = np.flatnonzero(np.diff(np.sign(signal)) != 0)
    crossings = t[rising] - signal[rising] * dt / (signal[rising + 1] - signal[rising])
    measured = 2 * LENGTH / np.mean(np.diff(crossings))
    print(
        f"Wave speed: measured {measured:.3f} m/s, expected {expected:.3f} m/s, "
        f"Bramwell-Hill {bramwell_hill:.3f} m/s; crossings "
        f"{np.round(crossings, 4).tolist()}; coupling iterations "
        f"mean {np.mean(iterations):.1f} max {max(iterations)}"
    )
    assert len(crossings) >= 3 and all(converged)
    assert np.mean(iterations) <= 6
    assert abs(measured / expected - 1) < 0.05
