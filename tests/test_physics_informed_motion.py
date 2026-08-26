"""Synthetic tests for the neo-Hookean residual behind physics-informed motion.

Every case here is a deformation whose answer is known in closed form, applied
to a tetrahedron small enough to check by hand, so these run in the default fast
suite with no data and no training.

Two of them matter more than the rest.  The cross-check pins the symbolic energy
PhysicsNeMo Sym evaluates during training against the tensor energy used to
derive stress for export: the two are written independently, and nothing else
would catch them drifting apart.  The gradient-flow test asserts the physics term
is actually trainable -- a residual that no gradient reaches would leave training
silently unchanged rather than visibly broken.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import pyvista as pv

from physiotwin4d.train_physicsnemo_physics_informed_motion import (
    NeoHookeanResidual,
    compute_deformation_gradient,
    tet_edges,
    tet_volumes,
)

if TYPE_CHECKING:  # typed for mypy; imported lazily inside the tests
    import torch

_MU_KPA = 10.0
_LAMBDA_KPA = 100.0

#: The reference tetrahedron: a corner of the unit cube, positively oriented.
_REFERENCE_TET_POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_REFERENCE_TET = np.array([[0, 1, 2, 3]])


def _rotation(angle: float) -> np.ndarray:
    """Return a rotation about z, which is a deformation tissue does not feel."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _deformation_gradient_of(
    linear_map: np.ndarray,
    translation: np.ndarray | None = None,
    points: np.ndarray = _REFERENCE_TET_POINTS,
    tets: np.ndarray = _REFERENCE_TET,
) -> "torch.Tensor":
    """Return F for the affine motion ``x -> linear_map @ x + translation``.

    A tetrahedron carries linear shape functions, so F comes back as exactly
    *linear_map* and every assertion below can be written in closed form.
    """
    import torch

    shift = np.zeros(3) if translation is None else translation
    displacement = points @ linear_map.T + shift - points
    return compute_deformation_gradient(
        torch.tensor(points, dtype=torch.float64),
        torch.tensor(displacement, dtype=torch.float64),
        torch.tensor(tets, dtype=torch.int64),
    )


def _oriented(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return *tets* with every element positively oriented.

    A real template arrives pre-oriented from
    ``ContourTools.trim_tetrahedra_to_surface``; a mesh built ad hoc for a test
    does not, so this stands in for that guarantee.
    """
    corners = points[tets]
    negative = np.linalg.det(corners[:, 1:, :] - corners[:, 0:1, :]) < 0.0
    fixed = tets.copy()
    fixed[negative, 2], fixed[negative, 3] = tets[negative, 3], tets[negative, 2]
    return fixed


def _grid_mesh(size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return points and oriented tets of a filled cubic grid.

    The least-squares gradient reconstruction fits over each node's edge
    neighborhood, so it needs a mesh with interior nodes rather than one element.
    """
    grid = np.mgrid[0:size, 0:size, 0:size].reshape(3, -1).T.astype(np.float64)
    volume = pv.PolyData(grid).delaunay_3d()
    points = np.asarray(volume.points)
    tets = volume.cells_dict[np.uint8(pv.CellType.TETRA)]
    return points, _oriented(points, tets)


def test_a_translated_tetrahedron_stores_no_energy() -> None:
    """Rigid translation is not deformation, so it costs nothing."""
    residual = NeoHookeanResidual(_MU_KPA, _LAMBDA_KPA)
    gradient = _deformation_gradient_of(np.eye(3), np.array([3.0, -2.0, 7.0]))

    assert float(residual.jacobian(gradient)[0]) == pytest.approx(1.0)
    assert float(residual.strain_energy(gradient)[0]) == pytest.approx(0.0, abs=1e-9)


def test_a_rotated_tetrahedron_stores_no_energy() -> None:
    """Rigid rotation is not deformation either, which is frame indifference."""
    residual = NeoHookeanResidual(_MU_KPA, _LAMBDA_KPA)
    gradient = _deformation_gradient_of(_rotation(0.7))

    assert float(residual.jacobian(gradient)[0]) == pytest.approx(1.0)
    assert float(residual.strain_energy(gradient)[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(residual.incompressibility(gradient)[0]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_a_uniformly_dilated_tetrahedron_matches_the_closed_form() -> None:
    """Scaling by s gives J = s^3 and the energy the constitutive law predicts."""
    scale = 1.1
    residual = NeoHookeanResidual(_MU_KPA, _LAMBDA_KPA)
    gradient = _deformation_gradient_of(scale * np.eye(3))

    log_jacobian = 3.0 * np.log(scale)
    expected = (
        0.5 * _MU_KPA * (3.0 * scale**2 - 3.0)
        - _MU_KPA * log_jacobian
        + 0.5 * _LAMBDA_KPA * log_jacobian**2
    )
    assert float(residual.jacobian(gradient)[0]) == pytest.approx(scale**3)
    assert float(residual.strain_energy(gradient)[0]) == pytest.approx(expected)
    assert float(residual.incompressibility(gradient)[0]) == pytest.approx(
        (scale**3 - 1.0) ** 2
    )


def test_an_inverted_element_is_reported_rather_than_returning_nan() -> None:
    """A reflection inverts the element; the energy stays finite and is counted."""
    residual = NeoHookeanResidual(_MU_KPA, _LAMBDA_KPA)
    gradient = _deformation_gradient_of(np.diag([1.0, 1.0, -1.0]))

    energy = residual.strain_energy(gradient)
    assert residual.inverted_element_count > 0
    assert bool(np.isfinite(float(energy[0])))


def test_stress_vanishes_under_rotation_and_stays_symmetric_under_stretch() -> None:
    """Cauchy stress answers to strain alone, and is symmetric by construction."""
    import torch

    residual = NeoHookeanResidual(_MU_KPA, _LAMBDA_KPA)

    rotated = residual.cauchy_stress(_deformation_gradient_of(_rotation(0.4)))
    assert torch.allclose(rotated, torch.zeros_like(rotated), atol=1e-9)

    stretched = residual.cauchy_stress(
        _deformation_gradient_of(np.diag([1.2, 1.0, 1.0]))
    )
    assert torch.allclose(stretched, stretched.transpose(-1, -2), atol=1e-12)
    assert float(stretched[0, 0, 0]) > 0.0


def test_the_volumes_of_a_filled_cube_sum_to_the_cube() -> None:
    """Element volumes partition the mesh, and nodal volumes redistribute them."""
    points, tets = _grid_mesh(size=2)
    volumes, nodal = tet_volumes(points, tets)

    assert volumes.sum() == pytest.approx(1.0)
    assert nodal.sum() == pytest.approx(volumes.sum())
    assert np.all(nodal > 0.0)


def test_an_inverted_template_is_refused() -> None:
    """A template with a flipped element cannot be used as physics elements."""
    flipped = _REFERENCE_TET[:, [0, 1, 3, 2]]
    with pytest.raises(ValueError, match="inverted or degenerate"):
        tet_volumes(_REFERENCE_TET_POINTS, flipped)


def test_every_undirected_edge_is_listed_once() -> None:
    """A tetrahedron has six edges, and a shared edge is not counted twice."""
    assert tet_edges(_REFERENCE_TET).shape == (6, 2)

    two_tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    edges = tet_edges(two_tets)
    assert len(edges) == 9  # 6 + 6, less the 3 shared by the common face
    assert len({tuple(edge) for edge in edges.tolist()}) == len(edges)


def test_the_symbolic_and_tensor_energies_agree() -> None:
    """The trained-against energy and the exported-from energy are one law.

    They are written independently -- one in sympy for PhysicsNeMo Sym, one in
    torch for stress export -- so this is what keeps them from drifting.
    """
    pytest.importorskip("physicsnemo.sym")
    import torch

    from physiotwin4d.train_physicsnemo_physics_informed_motion import (
        PhysicsInformedMotion,
    )

    points, tets = _grid_mesh(size=4)
    linear_map = np.array(
        [[1.08, 0.02, -0.03], [0.00, 1.05, 0.01], [0.04, -0.01, 1.07]]
    )
    displacement = points @ linear_map.T - points

    motion = PhysicsInformedMotion(
        tets=tets, n_points=len(points), mu_kpa=_MU_KPA, lambda_lame_kpa=_LAMBDA_KPA
    )
    energy, incompressibility = motion(
        torch.tensor(points, dtype=torch.float64),
        torch.tensor(displacement, dtype=torch.float64),
        torch.tensor(tet_volumes(points, tets)[1], dtype=torch.float64),
    )

    residual = NeoHookeanResidual(_MU_KPA, _LAMBDA_KPA)
    gradient = _deformation_gradient_of(linear_map, points=points, tets=tets)
    assert float(energy) == pytest.approx(
        float(residual.strain_energy(gradient).mean()), rel=1e-4
    )
    assert float(incompressibility) == pytest.approx(
        float(residual.incompressibility(gradient).mean()), rel=1e-4
    )


def test_the_physics_residual_is_trainable() -> None:
    """Gradients reach the displacement, or the physics term changes nothing."""
    pytest.importorskip("physicsnemo.sym")
    import torch

    from physiotwin4d.train_physicsnemo_physics_informed_motion import (
        PhysicsInformedMotion,
    )

    points, tets = _grid_mesh(size=4)
    motion = PhysicsInformedMotion(
        tets=tets, n_points=len(points), mu_kpa=_MU_KPA, lambda_lame_kpa=_LAMBDA_KPA
    )
    displacement = torch.zeros(
        (len(points), 3), dtype=torch.float64, requires_grad=True
    )

    energy, incompressibility = motion(
        torch.tensor(points, dtype=torch.float64),
        displacement,
        torch.tensor(tet_volumes(points, tets)[1], dtype=torch.float64),
    )
    (energy + incompressibility).backward()

    assert displacement.grad is not None
    assert bool(torch.isfinite(displacement.grad).all())


def test_a_batch_reports_which_samples_it_drew() -> None:
    """The loss needs each row's subject, so batches carry their sample indices."""
    from physiotwin4d import TrainPhysicsNeMoMGN

    class _IndexDataset:
        """Stands in for PhaseSampleDataset, returning its own index as data."""

        def __init__(self, n: int) -> None:
            self._n = n

        def __len__(self) -> int:
            return self._n

        def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.full((2, 1), index, dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
            )

    method = TrainPhysicsNeMoMGN()
    method.set_batch_size(2)
    batches = list(
        method._iter_batches(
            cast(Any, _IndexDataset(4)), np.random.default_rng(0), shuffle=False
        )
    )

    for node_feats, _, batch_len, indices in batches:
        assert len(indices) == batch_len
        # Rows are stacked sample by sample, so the indices name them in order.
        assert [int(value) for value in node_feats[::2, 0]] == list(indices)
