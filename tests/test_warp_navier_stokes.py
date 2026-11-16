"""Windkessel, inlet-flow and analytic pipe checks of the Warp Navier-Stokes solver."""

import inspect
from typing import Any

import numpy as np
import pytest
from scipy.spatial import Delaunay
from scipy.special import jv

from monai_physio.warp_navier_stokes import (
    WarpNavierStokes,
    Windkessel,
    find_cudss_library,
    inlet_flow,
)

RADIUS, LENGTH, VISCOSITY = 0.002, 0.024, 0.04
# This viscous pipe's float32 residual stagnates near 2e-5, above the default.
TOLERANCES = {"float64": 1e-6, "float32": 1e-4}


def require_cudss() -> None:
    """Skip when the external cuDSS library is not installed."""
    try:
        find_cudss_library()
    except RuntimeError as error:
        pytest.skip(str(error))


def disk(radius: float, rings: int) -> np.ndarray:
    """The centre plus rings k = 1..rings of 8 k points out to ``radius``."""
    points = [np.zeros((1, 2))]
    for k in range(1, rings + 1):
        angle = np.arange(8 * k) * 2 * np.pi / (8 * k)
        points.append(np.c_[np.cos(angle), np.sin(angle)] * radius * k / rings)
    return np.concatenate(points)


def extrude(
    xy: np.ndarray, triangles: np.ndarray, length: float, layers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude sorted triangles of ``xy`` along z into positive tetrahedra."""
    count = len(xy)
    points = np.concatenate(
        [np.c_[xy, np.full(count, z)] for z in np.linspace(0, length, layers + 1)]
    )
    cells = []
    for layer in range(layers):
        a, b, c = (triangles + layer * count).T
        # Splitting by sorted vertex index keeps neighboring prisms conforming.
        cells += [
            np.c_[a, b, c, c + count],
            np.c_[a, b, b + count, c + count],
            np.c_[a, a + count, b + count, c + count],
        ]
    tets = np.concatenate(cells)
    q = points[tets]
    det = np.einsum(
        "ij,ij->i", q[:, 1] - q[:, 0], np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0])
    )
    tets[det < 0] = tets[det < 0][:, [1, 0, 2, 3]]
    return points, tets


def outer_faces(tets: np.ndarray) -> np.ndarray:
    """Return the outward oriented boundary triangles of positive tetrahedra."""
    faces = np.concatenate(
        [tets[:, p] for p in ([1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1])]
    )
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    return np.asarray(faces[counts[inverse] == 1])


def pipe_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Structured tetrahedral pipe: a 121-point disk extruded into 24 layers.

    Regions are 0 wall, 1 inlet (z = 0) and 2 outlet (z = LENGTH).
    """
    xy = disk(RADIUS, 5)
    points, tets = extrude(xy, np.sort(Delaunay(xy).simplices, axis=1), LENGTH, 24)
    faces = outer_faces(tets)
    z = points[faces, 2]
    regions = np.zeros(len(faces), dtype=np.int64)
    regions[np.all(z < 1e-12, axis=1)] = 1
    regions[np.all(z > LENGTH - 1e-12, axis=1)] = 2
    return points, tets, faces, regions


def steady_pipe(
    precision: str = "float64", moving: bool = False
) -> tuple[WarpNavierStokes, float, list[dict[str, Any]]]:
    """Run 25 steps of viscous pipe flow (Re about 1) against a resistance."""
    points, tets, faces, regions = pipe_mesh()
    flow = 0.01 * np.pi * RADIUS**2
    solver = WarpNavierStokes(
        points,
        tets,
        faces,
        regions,
        inlet_region=1,
        outlets={2: Windkessel(1e9, 0.0, 1.0, 0.0)},
        dt=0.05,
        viscosity=VISCOSITY,
        tolerance=TOLERANCES[precision],
        precision=precision,
    )
    if moving:
        shift = np.tile([0.0005, -0.0002, 0.0], (len(points), 1))
        solver.move(shift, np.zeros_like(shift))
    return solver, flow, [solver.step(flow) for _ in range(25)]


def poiseuille_errors(solver: WarpNavierStokes, flow: float) -> tuple[float, float]:
    """Relative errors of the mid-pipe velocity profile and the pressure gradient."""
    velocity, pressure, points = solver.velocity(), solver.pressure(), solver.points
    area = -solver.port_flux[1][:, 2].sum()
    middle = np.abs(solver.dof_points[:, 2] - LENGTH / 2) < 1e-9
    radial = np.sum(solver.dof_points[middle, :2] ** 2, axis=1)
    expected = 2 * flow / area * (1 - radial / RADIUS**2)
    profile = np.linalg.norm(velocity[middle, 2] - expected) / np.linalg.norm(expected)
    bulk = (points[:, 2] > LENGTH / 4) & (points[:, 2] < 3 * LENGTH / 4)
    gradient = -np.polyfit(points[bulk, 2], pressure[bulk], 1)[0]
    poiseuille = 8 * VISCOSITY * flow / (np.pi * RADIUS**4)
    return float(profile), float(abs(gradient / poiseuille - 1))


def test_default_precision_is_float32() -> None:
    """The solver runs in float32 unless float64 is requested."""
    parameters = inspect.signature(WarpNavierStokes.__init__).parameters
    assert parameters["precision"].default == "float32"


def test_windkessel_steady_state_and_split() -> None:
    """Constant flow relaxes to (R_p + R_d) Q; split outlets recover the totals."""
    windkessel = Windkessel(1e7, 9e7, 1e-8, 0.0)
    flow, dt, tau = 5e-5, 1e-3, 9e7 * 1e-8
    pressures = [windkessel.advance(flow, dt) for _ in range(20000)]
    assert abs(pressures[-1] / (1e8 * flow) - 1) < 1e-6
    expected = 1e7 * flow + 9e7 * flow * (1 - np.exp(-0.5))
    assert abs(pressures[round(0.5 * tau / dt) - 1] / expected - 1) < 2e-3
    outlets = [
        Windkessel.from_total(f, 3.33e8, 6.8e-9, 0.06, 1e4)
        for f in (0.70, 0.17, 0.08, 0.05)
    ]
    resistances = [w.proximal_resistance + w.distal_resistance for w in outlets]
    assert abs(1 / sum(1 / r for r in resistances) / 3.33e8 - 1) < 1e-12
    assert abs(sum(w.compliance for w in outlets) / 6.8e-9 - 1) < 1e-12
    for windkessel, resistance in zip(outlets, resistances):
        assert abs(windkessel.proximal_resistance / resistance - 0.06) < 1e-12


def test_inlet_flow_preserves_volume_and_sign() -> None:
    """Gaussian smoothing keeps Q >= 0, every beat volume and a consistent dQ/dt."""
    flux = np.zeros(100)
    for start in (0, 34, 67):
        flux[start : start + 12] = np.repeat([20e-6, 120e-6, 180e-6, 60e-6], 3)
    times = (np.arange(30000) + 0.5) * 1e-4
    flow, derivative = inlet_flow(times, flux, 0.03, 0.03)
    assert np.all(flow >= 0)
    assert flow.max() < flux.max()
    for start, low, high in ((0, -0.2, 0.82), (34, 0.82, 1.81), (67, 1.81, 2.8)):
        window = np.mod(times - low, 3.0) < high - low
        beat = flow[window].sum() * 1e-4
        assert abs(beat / (flux[start : start + 12].sum() * 0.03) - 1) <= 1e-6
    numerical = np.gradient(flow, times)
    scale = np.abs(derivative).max()
    np.testing.assert_allclose(derivative[1:-1], numerical[1:-1], atol=1e-5 * scale)


@pytest.mark.requires_gpu
def test_solver_poiseuille_pipe() -> None:
    """Steady flow matches Poiseuille; outlet pressure and wall shear are exact."""
    require_cudss()
    solver, flow, results = steady_pipe()
    try:
        assert max(r["residual"] for r in results) <= TOLERANCES["float64"]
        profile, gradient = poiseuille_errors(solver, flow)
        assert profile < 0.02 and gradient < 0.03
        outlet = solver.points[:, 2] > LENGTH - 1e-12
        assert abs(solver.pressure()[outlet].mean() / (1e9 * flow) - 1) < 0.01
        z = solver.dof_points[:, 2]
        third = (z > LENGTH / 3) & (z < 2 * LENGTH / 3)
        shear = 4 * flow / (np.pi * RADIUS**3) * VISCOSITY * 2 * np.pi * RADIUS
        force = solver.wall_force()[third, 2].sum()
        assert abs(force / (shear * LENGTH / 3) - 1) < 0.06
    finally:
        solver.close()


@pytest.mark.requires_gpu
def test_precisions_match_poiseuille() -> None:
    """Float32 reproduces Poiseuille flow and agrees with float64."""
    require_cudss()
    fields = {}
    for precision in ("float64", "float32"):
        solver, flow, results = steady_pipe(precision)
        try:
            profile, gradient = poiseuille_errors(solver, flow)
            fields[precision] = (solver.velocity(), solver.pressure())
        finally:
            solver.close()
        iterations = [r["iterations"] for r in results]
        print(
            f"{precision}: profile error {profile:.2e}, gradient error "
            f"{gradient:.2e}, GMRES {np.mean(iterations):.0f} (max {max(iterations)})"
        )
        assert all(field.dtype == np.float64 for field in fields[precision])
        assert max(r["residual"] for r in results) <= TOLERANCES[precision]
        assert profile < 0.02 and gradient < 0.03
        net = max(abs(sum(r["flows_m3_s"].values())) for r in results)
        assert net < 1e-5 * flow
    for single, double in zip(fields["float32"], fields["float64"]):
        assert np.linalg.norm(single - double) / np.linalg.norm(double) < 1e-4


@pytest.mark.requires_gpu
def test_solver_moving_mesh_translation() -> None:
    """A rigidly translated deformable mesh reproduces the fixed-mesh solution."""
    require_cudss()
    fixed = steady_pipe()[0]
    moved = steady_pipe(moving=True)[0]
    try:
        scale = np.abs(fixed.velocity()).max()
        np.testing.assert_allclose(
            moved.velocity(), fixed.velocity(), rtol=0, atol=1e-6 * scale
        )
        np.testing.assert_allclose(moved.pressure(), fixed.pressure(), rtol=1e-6)
    finally:
        fixed.close()
        moved.close()


@pytest.mark.requires_gpu
def test_solver_womersley_pipe() -> None:
    """Oscillatory flow at Womersley number 4 matches the analytic profile."""
    require_cudss()
    points, tets, faces, regions = pipe_mesh()
    density, viscosity, alpha, gradient, steps = 1060.0, 0.004, 4.0, 50.0, 40
    omega = alpha**2 * viscosity / (density * RADIUS**2)
    period = 2 * np.pi / omega
    argument = np.asarray(1j**1.5 * alpha)
    amplitude = (
        np.pi
        * RADIUS**2
        * gradient
        / (1j * omega * density)
        * (1 - 2 * jv(1, argument) / (argument * jv(0, argument)))
    )
    solver = WarpNavierStokes(
        points,
        tets,
        faces,
        regions,
        inlet_region=1,
        outlets={2: Windkessel(0.0, 0.0, 1.0, 0.0)},
        dt=period / steps,
        density=density,
        viscosity=viscosity,
    )
    middle = np.abs(solver.dof_points[:, 2] - LENGTH / 2) < 1e-9
    samples, times = [], []
    try:
        for index in range(3 * steps):
            time = (index + 1) * period / steps
            solver.step(float(np.real(amplitude * np.exp(1j * omega * time))))
            if index >= 2 * steps:
                samples.append(solver.velocity()[middle, 2])
                times.append(time)
    finally:
        solver.close()
    phase = omega * np.asarray(times)
    coefficients = np.linalg.lstsq(
        np.c_[np.cos(phase), np.sin(phase)], np.asarray(samples), rcond=None
    )[0]
    observed = coefficients[0] - 1j * coefficients[1]
    radial = np.linalg.norm(solver.dof_points[middle, :2], axis=1)
    reference = (
        gradient
        / (1j * omega * density)
        * (1 - jv(0, argument * radial / RADIUS) / jv(0, argument))
    )
    active = np.abs(reference) > 0.1 * np.abs(reference).max()
    difference = observed[active] - reference[active]
    assert np.linalg.norm(difference) / np.linalg.norm(reference[active]) < 0.05
    angles = np.angle(observed[active] / reference[active], deg=True)
    assert np.abs(angles).max() < 5
