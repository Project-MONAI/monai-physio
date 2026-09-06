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

from monai_physio.train_physicsnemo_physics_informed_motion import (
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

    from monai_physio.train_physicsnemo_physics_informed_motion import (
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
        torch.tensor(points, dtype=torch.float64, device=motion.device),
        torch.tensor(displacement, dtype=torch.float64, device=motion.device),
        torch.tensor(
            tet_volumes(points, tets)[1], dtype=torch.float64, device=motion.device
        ),
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

    from monai_physio.train_physicsnemo_physics_informed_motion import (
        PhysicsInformedMotion,
    )

    points, tets = _grid_mesh(size=4)
    motion = PhysicsInformedMotion(
        tets=tets, n_points=len(points), mu_kpa=_MU_KPA, lambda_lame_kpa=_LAMBDA_KPA
    )
    displacement = torch.zeros(
        (len(points), 3),
        dtype=torch.float64,
        device=motion.device,
        requires_grad=True,
    )

    energy, incompressibility = motion(
        torch.tensor(points, dtype=torch.float64, device=motion.device),
        displacement,
        torch.tensor(
            tet_volumes(points, tets)[1], dtype=torch.float64, device=motion.device
        ),
    )
    (energy + incompressibility).backward()

    assert displacement.grad is not None
    assert bool(torch.isfinite(displacement.grad).all())


def test_a_batch_reports_which_samples_it_drew() -> None:
    """The loss needs each row's subject, so batches carry their sample indices."""
    from monai_physio import TrainPhysicsNeMoMGN

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


def test_inverted_elements_are_counted_during_training() -> None:
    """The residual notices motion that turns tissue inside out.

    The energy clamps ``J`` so an inversion stays finite and trainable, which is
    exactly what would let one pass unnoticed. This is the only signal that it
    happened, so a counter that cannot move is worse than no counter at all.
    """
    pytest.importorskip("physicsnemo.sym")
    import torch

    from monai_physio.train_physicsnemo_physics_informed_motion import (
        PhysicsInformedMotion,
    )

    points, tets = _grid_mesh(size=4)
    motion = PhysicsInformedMotion(
        tets=tets, n_points=len(points), mu_kpa=_MU_KPA, lambda_lame_kpa=_LAMBDA_KPA
    )
    reference = torch.tensor(points, dtype=torch.float64, device=motion.device)
    volumes = torch.tensor(
        tet_volumes(points, tets)[1], dtype=torch.float64, device=motion.device
    )

    # A mild stretch: nothing inverts.
    motion(reference, 0.05 * reference, volumes)
    assert motion.inverted_element_count == 0

    # u = -2x turns the x axis inside out, so J goes negative everywhere.
    reflection = torch.zeros_like(reference)
    reflection[:, 0] = -2.0 * reference[:, 0]
    motion(reference, reflection, volumes)
    assert motion.inverted_element_count > 0, (
        "A reflected field inverts every element; the counter must move"
    )


def test_a_residual_on_the_wrong_device_is_refused() -> None:
    """A CPU residual cannot serve GPU training, and says so before it starts.

    The residual's connectivity and compiled symbolic graph are bound at
    construction and cannot be moved, so a residual built without an explicit
    device defaults to the CPU and will not meet predictions made on a GPU.
    Caught here rather than as an opaque tensor error inside the gradient
    reconstruction. Devices are compared, never allocated on, so this runs
    wherever the fast suite does -- including a machine with no NVIDIA driver,
    which ``test_the_device_check_runs_without_a_driver`` pins down.
    """
    import torch

    from monai_physio.physicsnemo_tools import DistributedContext
    from monai_physio.train_physicsnemo_physics_informed_motion import (
        TrainPhysicsNeMoPhysicsInformedMotion,
    )

    class _CpuResidual:
        """Stands in for a residual left on its default device."""

        device = torch.device("cpu")

    method = TrainPhysicsNeMoPhysicsInformedMotion()
    method._residual = cast(Any, _CpuResidual())

    cuda_context = DistributedContext(
        device=torch.device("cuda"), rank=0, local_rank=0, world_size=1
    )
    with pytest.raises(ValueError, match="cannot meet the predictions"):
        method._require_matching_device(cuda_context)

    cpu_context = DistributedContext(
        device=torch.device("cpu"), rank=0, local_rank=0, world_size=1
    )
    method._require_matching_device(cpu_context)


def test_the_device_check_runs_without_a_driver() -> None:
    """The device check must not need a GPU to say a GPU is missing.

    Resolving an index-less ``torch.device("cuda")`` to a concrete index asks
    torch for the current device, and that raises outright on a machine with no
    NVIDIA driver rather than reporting none. Guarding it matters because this
    check runs in the fast suite, which is expected to pass anywhere.

    Simulated rather than skipped, so the path is exercised on a machine that
    does have a GPU -- which is where the regression was written.
    """
    from unittest import mock

    import torch

    from monai_physio.physicsnemo_tools import DistributedContext
    from monai_physio.train_physicsnemo_physics_informed_motion import (
        TrainPhysicsNeMoPhysicsInformedMotion,
        _resolved_device,
    )

    def no_driver(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Found no NVIDIA driver on your system.")

    class _CpuResidual:
        device = torch.device("cpu")

    method = TrainPhysicsNeMoPhysicsInformedMotion()
    method._residual = cast(Any, _CpuResidual())
    cuda_context = DistributedContext(
        device=torch.device("cuda"), rank=0, local_rank=0, world_size=1
    )

    with (
        mock.patch("torch.cuda.is_available", lambda: False),
        mock.patch("torch.cuda.current_device", no_driver),
    ):
        # Unresolvable, so left as-is rather than raising.
        assert _resolved_device(torch.device("cuda")) == torch.device("cuda")
        # And an unresolved cuda still does not match a CPU residual.
        with pytest.raises(ValueError, match="cannot meet the predictions"):
            method._require_matching_device(cuda_context)


def test_a_residual_on_another_gpu_is_refused() -> None:
    """Two GPUs are as incompatible as a GPU and a CPU, and must be caught too.

    Under distributed training each rank owns one GPU, so a residual built on
    cuda:0 cannot serve a rank predicting on cuda:1. Comparing only the device
    *type* would wave that through and leave it to fail deep inside the gradient
    reconstruction.

    The converse also has to hold: ``torch.device("cuda")`` carries no index and
    compares unequal to ``cuda:0``, so a check that ignored that would reject a
    perfectly matched pair. No CUDA is needed -- devices are compared, never
    allocated on -- beyond resolving the index-less form, which is skipped when
    no GPU is present.
    """
    import torch

    from monai_physio.physicsnemo_tools import DistributedContext
    from monai_physio.train_physicsnemo_physics_informed_motion import (
        TrainPhysicsNeMoPhysicsInformedMotion,
    )

    class _ResidualOn:
        """Stands in for a residual whose tensors live on one specific GPU."""

        def __init__(self, device: "torch.device") -> None:
            self.device = device

    method = TrainPhysicsNeMoPhysicsInformedMotion()

    def context_on(device: "torch.device") -> DistributedContext:
        return DistributedContext(device=device, rank=0, local_rank=0, world_size=1)

    method._residual = cast(Any, _ResidualOn(torch.device("cuda", 0)))
    with pytest.raises(ValueError, match="cannot meet the predictions"):
        method._require_matching_device(context_on(torch.device("cuda", 1)))

    # Same GPU named two ways must still be accepted.
    method._require_matching_device(context_on(torch.device("cuda", 0)))
    if torch.cuda.is_available() and torch.cuda.current_device() == 0:
        method._require_matching_device(context_on(torch.device("cuda")))


def test_the_epoch_log_separates_the_two_loss_terms() -> None:
    """The data and physics terms are reported apart, then reset.

    They are in different units -- normalized displacement against kilopascals --
    so a total alone cannot say how they balance, and ``lambda_physics`` is not
    choosable without seeing them separately.
    """
    import torch

    from monai_physio.physicsnemo_tools import DistributedContext
    from monai_physio.train_physicsnemo_physics_informed_motion import (
        TrainPhysicsNeMoPhysicsInformedMotion,
    )

    method = TrainPhysicsNeMoPhysicsInformedMotion()
    method.lambda_physics = 0.25
    method._epoch_data_loss = torch.tensor(2.0)
    method._epoch_physics_loss = torch.tensor(8.0)
    method._epoch_batches = 2

    messages: list[str] = []
    method.log_info = lambda *args: messages.append(str(args[0]) % args[1:])  # type: ignore[method-assign]

    context = DistributedContext(
        device=torch.device("cpu"), rank=0, local_rank=0, world_size=1
    )
    method._log_epoch(context, epoch=0, epochs=1)

    assert messages, "The epoch hook should report something"
    reported = messages[-1]
    assert "data=1.000000" in reported, f"Data term should be the mean: {reported}"
    assert "physics=4.000000" in reported, (
        f"Physics term should be the mean: {reported}"
    )
    assert "1.000000)" in reported, f"Weighted physics term should appear: {reported}"

    # The accumulators reset, or every epoch would report the previous ones too.
    method._log_epoch(context, epoch=1, epochs=2)
    assert "data=0.000000" in messages[-1], (
        f"An epoch that accumulated nothing should report zero: {messages[-1]}"
    )
    assert "physics=0.000000" in messages[-1], (
        f"An epoch that accumulated nothing should report zero: {messages[-1]}"
    )


def test_bind_reference_meshes_repairs_against_template_elements(tmp_path: Any) -> None:
    """Repair must use ``self._tets``, not whatever cells the file stores.

    A fitted reference file's own connectivity is never read back by
    ``tet_volumes`` -- only ``self._tets`` is -- so repairing against the
    file's cells instead would validate the wrong topology. Here the file
    stores one degenerate, unrecoverable cell that touches only a single
    node; repairing against it would raise, but ``self._tets`` names a
    perfectly valid mesh, so binding must succeed unchanged.
    """
    import torch

    from monai_physio.physicsnemo_tools import DistributedContext
    from monai_physio.train_physicsnemo_physics_informed_motion import (
        TrainPhysicsNeMoPhysicsInformedMotion,
    )

    points, tets = _grid_mesh(size=3)
    mismatched = pv.UnstructuredGrid(
        {pv.CellType.TETRA: np.array([[0, 0, 0, 0]])}, points
    )
    mesh_path = tmp_path / "reference.vtu"
    mismatched.save(mesh_path)

    method = TrainPhysicsNeMoPhysicsInformedMotion()
    method._tets = tets
    method._sample_subjects = ["subj0"]
    method._reference_meshes = {"subj0": mesh_path}
    context = DistributedContext(
        device=torch.device("cpu"), rank=0, local_rank=0, world_size=1
    )

    method._bind_reference_meshes(context, n_points=len(points))

    reference, volumes = method._reference_cache["subj0"]
    assert np.allclose(reference.numpy(), points, atol=1e-5)
    assert torch.all(volumes > 0)


def test_bind_reference_meshes_tolerates_a_file_with_no_cells(tmp_path: Any) -> None:
    """A reference file need not carry any cells at all; only its points do."""
    import torch

    from monai_physio.physicsnemo_tools import DistributedContext
    from monai_physio.train_physicsnemo_physics_informed_motion import (
        TrainPhysicsNeMoPhysicsInformedMotion,
    )

    points, tets = _grid_mesh(size=3)
    mesh_path = tmp_path / "reference.vtp"
    pv.PolyData(points).save(mesh_path)

    method = TrainPhysicsNeMoPhysicsInformedMotion()
    method._tets = tets
    method._sample_subjects = ["subj0"]
    method._reference_meshes = {"subj0": mesh_path}
    context = DistributedContext(
        device=torch.device("cpu"), rank=0, local_rank=0, world_size=1
    )

    method._bind_reference_meshes(context, n_points=len(points))

    reference, volumes = method._reference_cache["subj0"]
    assert np.allclose(reference.numpy(), points, atol=1e-5)
    assert torch.all(volumes > 0)
