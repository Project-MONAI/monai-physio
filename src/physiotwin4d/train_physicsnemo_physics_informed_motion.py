"""Physics-informed motion training for PhysicsNeMo mesh-stage models.

:class:`physiotwin4d.TrainPhysicsNeMoMGN` scores predicted motion against
measured displacement alone, so nothing in its loss rules out motion no
myocardium could undergo: locally inverted elements, non-physical dilation, a
wall that thins past what tissue allows.  This module adds a neo-Hookean strain
energy to that loss, which prices those deformations, and exposes the Cauchy
stress the same constitutive law implies so predicted motion can be rendered as
stress.

The energy needs volume elements, so the shape model must be a tetrahedral one:
the template's own cells become the elements, which is what makes one set of
element node ids valid for every subject and phase at once.

**Reference configuration.**  The residual is measured against each subject's
*fitted* reference geometry, never the shared template.  The stored targets are
``phase.points - fitted_reference.points``, so the fitted reference is the
undeformed state; measuring against the population mean instead would charge
every subject a strain energy for merely being shaped unlike the mean, which
confuses variation between subjects with deformation within one.

Two formulations of the same energy live here on purpose.  The symbolic one
(:func:`neo_hookean_pde`) is what PhysicsNeMo Sym differentiates and evaluates
during training; the tensor one (:class:`NeoHookeanResidual`) is what computes
the Cauchy stress for export, which the symbolic path does not hand back.  The
tests cross-check them against each other.

PhysicsNeMo Sym is an optional dependency imported lazily, so ``import
physiotwin4d`` works without it.  It ships inside ``nvidia-physicsnemo``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pyvista as pv

from .physicsnemo_tools import DistributedContext, PhaseSampleDataset
from .physiotwin4d_base import PhysioTwin4DBase
from .train_physicsnemo_mgn import TrainPhysicsNeMoMGN

if TYPE_CHECKING:  # typed for mypy; imported lazily at runtime
    import torch

#: Floor applied to the Jacobian determinant before taking its logarithm.  An
#: inverted element makes ``J`` non-positive, which would poison the whole loss
#: with a NaN; clamping keeps the step finite so the inversion is reported and
#: trainable rather than fatal.
_MIN_JACOBIAN = 1.0e-6


def _resolved_device(device: "torch.device") -> "torch.device":
    """Return *device* with its CUDA index filled in.

    ``torch.device("cuda")`` carries no index and compares unequal to
    ``cuda:0``, even though a tensor allocated on it lands there.  Comparing
    devices without resolving that would reject a matched pair, while comparing
    only their types would accept ``cuda:0`` against ``cuda:1`` -- a real
    mismatch under distributed training, where each rank owns one GPU.

    Left unchanged when no CUDA device is visible: there is no current device to
    resolve to, and asking for one raises rather than returning nothing.  An
    unresolved ``cuda`` still compares unequal to ``cpu``, which is the answer
    the caller wants on such a machine anyway.
    """
    import torch

    # torch's stubs declare ``index`` as ``int``, so mypy reads the branch below
    # as dead; at runtime ``torch.device("cuda").index`` really is None.
    index: Optional[int] = device.index
    if device.type == "cuda" and index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


#: The six node pairs spanning a tetrahedron's edges.
_TET_EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def tet_volumes(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each tetrahedron's volume and the volume lumped onto each node.

    The nodal volumes are the quadrature weights the residual is averaged with,
    so a region contributes in proportion to the tissue it holds rather than to
    how finely it happens to be meshed.

    Args:
        points: ``(n_points, 3)`` node positions.
        tets: ``(n_tet, 4)`` node ids per tetrahedron.

    Returns:
        ``(element_volumes, nodal_volumes)``, shaped ``(n_tet,)`` and
        ``(n_points,)``.

    Raises:
        ValueError: If any element is inverted or degenerate.  Templates come
            from :meth:`physiotwin4d.ContourTools.trim_tetrahedra_to_surface`,
            which holds every cell above a scaled Jacobian of 0.1, so a
            violation here means the template is broken rather than merely
            tight.
    """
    corners = points[tets]
    edges = corners[:, 1:, :] - corners[:, 0:1, :]
    volumes = np.linalg.det(edges) / 6.0
    if not np.all(volumes > 0.0):
        n_bad = int(np.sum(volumes <= 0.0))
        raise ValueError(
            f"{n_bad} of {len(volumes)} tetrahedra are inverted or degenerate; "
            "the template mesh is unusable as a set of physics elements."
        )

    nodal = np.zeros(len(points), dtype=np.float64)
    np.add.at(nodal, tets.ravel(), np.repeat(volumes / 4.0, 4))
    return volumes, nodal


def tet_edges(tets: np.ndarray) -> np.ndarray:
    """Return the unique undirected ``(n_edge, 2)`` edges of a tetrahedral mesh.

    This is the stencil the least-squares gradient reconstruction fits over.
    """
    pairs = np.vstack([tets[:, pair] for pair in _TET_EDGE_PAIRS])
    return np.unique(np.sort(pairs, axis=1), axis=0)


def edge_matrix(points: "torch.Tensor", tets: "torch.Tensor") -> "torch.Tensor":
    """Return the ``(n_tet, 3, 3)`` matrix whose columns are an element's edges."""
    corners = points[tets]
    return (corners[:, 1:, :] - corners[:, 0:1, :]).transpose(-1, -2)


def compute_deformation_gradient(
    reference_points: "torch.Tensor",
    displacement: "torch.Tensor",
    tets: "torch.Tensor",
    reference_inverse: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Return the per-element deformation gradient ``F``.

    ``F = Ds @ Dm^-1``, with the columns of ``Dm`` the reference edge vectors of
    an element and the columns of ``Ds`` its deformed ones -- exact for the
    linear shape functions a tetrahedron carries.

    Args:
        reference_points: ``(n_points, 3)`` undeformed node positions.
        displacement: ``(n_points, 3)`` predicted displacement, same units.
        tets: ``(n_tet, 4)`` node ids.
        reference_inverse: Cached ``(n_tet, 3, 3)`` inverse of ``Dm``.  It
            depends only on the reference configuration, so passing it back in
            avoids re-inverting it once per phase of the same subject.

    Returns:
        ``(n_tet, 3, 3)`` deformation gradients.
    """
    import torch

    if reference_inverse is None:
        reference_inverse = torch.linalg.inv(edge_matrix(reference_points, tets))
    deformed = reference_points + displacement
    return torch.matmul(edge_matrix(deformed, tets), reference_inverse)


class NeoHookeanResidual(PhysioTwin4DBase):
    """Compressible neo-Hookean constitutive law, evaluated on tensors.

    The strain energy density is

    ``W = (mu / 2) (I1 - 3) - mu ln(J) + (lambda / 2) ln(J)^2``

    with ``I1 = tr(F^T F)`` and ``J = det(F)``.  Its Cauchy stress is
    ``sigma = (mu / J)(B - I) + (lambda ln(J) / J) I`` for ``B = F F^T``.

    Args:
        mu_kpa: Shear modulus, in kilopascals.
        lambda_lame_kpa: First Lame parameter, in kilopascals.
        log_level: Logging level.  Default: ``logging.INFO``.
    """

    def __init__(
        self,
        mu_kpa: float = 10.0,
        lambda_lame_kpa: float = 100.0,
        log_level: int | str = logging.INFO,
    ) -> None:
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        if mu_kpa <= 0.0:
            raise ValueError(f"mu_kpa must be > 0, got {mu_kpa}")
        if lambda_lame_kpa <= 0.0:
            raise ValueError(f"lambda_lame_kpa must be > 0, got {lambda_lame_kpa}")
        self.mu_kpa = mu_kpa
        self.lambda_lame_kpa = lambda_lame_kpa
        # Accumulated on whatever device the gradients arrive on, so counting
        # costs no host synchronization; only the property below pays one.
        self._inverted: Optional["torch.Tensor"] = None

    @property
    def inverted_element_count(self) -> int:
        """Elements whose Jacobian went non-positive since this was constructed.

        Non-zero means the deformation turns tissue inside out somewhere.  The
        clamp in :meth:`jacobian` keeps the energy finite so training can
        continue, which is also what would let an inversion pass unnoticed.
        """
        if self._inverted is None:
            return 0
        return int(self._inverted.item())

    def jacobian(self, deformation_gradient: "torch.Tensor") -> "torch.Tensor":
        """Return ``det(F)`` clamped away from zero, counting any inversion."""
        import torch

        jacobian = torch.linalg.det(deformation_gradient)
        inverted = (jacobian <= 0.0).sum().detach()
        self._inverted = (
            inverted if self._inverted is None else self._inverted + inverted
        )
        return torch.clamp(jacobian, min=_MIN_JACOBIAN)

    def strain_energy(self, deformation_gradient: "torch.Tensor") -> "torch.Tensor":
        """Return the per-element strain energy density, in kilopascals."""
        import torch

        first_invariant = torch.einsum(
            "...ij,...ij->...", deformation_gradient, deformation_gradient
        )
        log_jacobian = torch.log(self.jacobian(deformation_gradient))
        return (
            0.5 * self.mu_kpa * (first_invariant - 3.0)
            - self.mu_kpa * log_jacobian
            + 0.5 * self.lambda_lame_kpa * log_jacobian**2
        )

    def incompressibility(self, deformation_gradient: "torch.Tensor") -> "torch.Tensor":
        """Return ``(J - 1)^2``, the soft penalty on volume change."""
        import torch

        return cast("torch.Tensor", (torch.linalg.det(deformation_gradient) - 1.0) ** 2)

    def cauchy_stress(self, deformation_gradient: "torch.Tensor") -> "torch.Tensor":
        """Return the ``(..., 3, 3)`` Cauchy stress tensor, in kilopascals."""
        import torch

        left_cauchy_green = torch.matmul(
            deformation_gradient, deformation_gradient.transpose(-1, -2)
        )
        identity = torch.eye(
            3, dtype=deformation_gradient.dtype, device=deformation_gradient.device
        ).expand_as(left_cauchy_green)
        jacobian = self.jacobian(deformation_gradient).unsqueeze(-1).unsqueeze(-1)
        log_jacobian = torch.log(jacobian)
        return (self.mu_kpa / jacobian) * (left_cauchy_green - identity) + (
            self.lambda_lame_kpa * log_jacobian / jacobian
        ) * identity


def neo_hookean_pde(mu_kpa: float, lambda_lame_kpa: float) -> Any:
    """Return the neo-Hookean energy as a PhysicsNeMo Sym ``PDE``.

    The symbolic form is what :class:`PhysicsInformedMotion` differentiates: it
    is written in terms of the displacement fields ``u``, ``v`` and ``w``, so
    PhysicsNeMo Sym supplies their spatial derivatives and assembles
    ``F = I + grad(u)`` itself.

    Args:
        mu_kpa: Shear modulus, in kilopascals.
        lambda_lame_kpa: First Lame parameter, in kilopascals.

    Returns:
        A ``PDE`` exposing ``neo_hookean_energy`` and ``incompressibility``.
    """
    from physicsnemo.sym.eq.pde import PDE
    from sympy import Function, Matrix, Max, Number, Symbol, log

    class NeoHookeanEnergy(PDE):
        """Strain energy and incompressibility penalty of a neo-Hookean solid."""

        name = "NeoHookeanEnergy"

        def __init__(self, mu: float, lambda_lame: float) -> None:
            self.dim = 3
            x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
            u = Function("u")(x, y, z)
            v = Function("v")(x, y, z)
            w = Function("w")(x, y, z)
            deformation_gradient = Matrix(
                [
                    [1 + u.diff(x), u.diff(y), u.diff(z)],
                    [v.diff(x), 1 + v.diff(y), v.diff(z)],
                    [w.diff(x), w.diff(y), 1 + w.diff(z)],
                ]
            )
            jacobian = deformation_gradient.det()
            first_invariant = (deformation_gradient.T * deformation_gradient).trace()
            # Clamped exactly as NeoHookeanResidual clamps it, so an inverted
            # element costs a large finite penalty instead of a NaN.
            safe_jacobian = Max(jacobian, Number(_MIN_JACOBIAN))
            self.equations = {
                "neo_hookean_energy": mu / 2 * (first_invariant - 3)
                - mu * log(safe_jacobian)
                + lambda_lame / 2 * log(safe_jacobian) ** 2,
                "incompressibility": (jacobian - Number(1)) ** 2,
                # The unclamped determinant, exposed so a caller can count the
                # elements that inverted.  The clamp above keeps the energy
                # finite but hides them, and an inversion is the one failure
                # that says the predicted motion is not motion tissue can do.
                "jacobian": jacobian,
            }

    return NeoHookeanEnergy(mu_kpa, lambda_lame_kpa)


class PhysicsInformedMotion(PhysioTwin4DBase):
    """Evaluate the neo-Hookean residual of a predicted displacement field.

    Spatial derivatives come from PhysicsNeMo Sym's least-squares gradient
    reconstruction, the method built for unstructured meshes: it fits a gradient
    at every node over that node's edge neighborhood.  The neighborhood is fixed
    by the template's topology, so the connectivity is built once here rather
    than per batch.

    Args:
        tets: ``(n_tet, 4)`` template element node ids.
        n_points: Node count of the template.
        mu_kpa: Shear modulus, in kilopascals.
        lambda_lame_kpa: First Lame parameter, in kilopascals.
        device: Device the residual is evaluated on.  Defaults to the GPU when
            there is one, since this is the most expensive part of the loss and
            has to sit where the predictions are.  Pass it explicitly when
            training somewhere other than the default device.
        log_level: Logging level.  Default: ``logging.INFO``.
    """

    def __init__(
        self,
        tets: np.ndarray,
        n_points: int,
        mu_kpa: float = 10.0,
        lambda_lame_kpa: float = 100.0,
        device: Optional["torch.device"] = None,
        log_level: int | str = logging.INFO,
    ) -> None:
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        import torch

        from physicsnemo.sym.eq.gradients import compute_connectivity_tensor
        from physicsnemo.sym.eq.phy_informer import PhysicsInformer

        self.mu_kpa = mu_kpa
        self.lambda_lame_kpa = lambda_lame_kpa
        self.n_points = n_points
        # A GPU is assumed, so the residual goes there unless told otherwise:
        # it is evaluated once per sample per step and is the most expensive
        # part of the loss.
        self._device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        edges = torch.from_numpy(tet_edges(tets).astype(np.int64))
        node_ids = torch.arange(n_points).reshape(-1, 1)
        self._connectivity = tuple(
            tensor.to(self._device)
            for tensor in compute_connectivity_tensor(node_ids, edges)
        )
        self._informer = PhysicsInformer(
            required_outputs=[
                "neo_hookean_energy",
                "incompressibility",
                "jacobian",
            ],
            equations=neo_hookean_pde(mu_kpa, lambda_lame_kpa),
            grad_method="least_squares",
            compute_connectivity=False,
            device=str(self._device),
        )
        # Kept on the device and summed there, so counting inversions costs no
        # host synchronization in the training loop.
        self._inverted = torch.zeros((), dtype=torch.long, device=self._device)
        self.log_info(
            "Neo-Hookean residual over %d elements on %s "
            "(mu=%.3g kPa, lambda=%.3g kPa)",
            len(tets),
            self._device,
            mu_kpa,
            lambda_lame_kpa,
        )

    @property
    def device(self) -> "torch.device":
        """Device this residual's connectivity and symbolic graph were built on.

        Fixed at construction: ``PhysicsInformer`` is given the device when its
        graph is compiled, so the residual cannot be moved afterwards. The
        trainer checks this against the device it predicts on.

        Read from a tensor the residual actually owns, so the CUDA index is
        concrete even when the caller passed an index-less
        ``torch.device("cuda")``.
        """
        return self._inverted.device

    @property
    def inverted_element_count(self) -> int:
        """Nodes whose Jacobian went non-positive since this was built.

        Non-zero means the network predicted motion that turns tissue inside out
        somewhere.  The energy clamps ``J`` to stay finite and trainable, so
        without this count an inversion would leave no trace.
        """
        return int(self._inverted.item())

    def __call__(
        self,
        reference_points: "torch.Tensor",
        displacement_mm: "torch.Tensor",
        nodal_volumes: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return the volume-weighted ``(strain energy, incompressibility)``.

        Args:
            reference_points: ``(n_points, 3)`` undeformed positions of this
                subject, in millimeters.
            displacement_mm: ``(n_points, 3)`` predicted displacement, in
                millimeters rather than in the normalized units the data loss
                is scored on.
            nodal_volumes: ``(n_points,)`` quadrature weights from
                :func:`tet_volumes`.

        Returns:
            Two scalars, each a nodal-volume-weighted mean over the mesh.
        """
        residuals = self._informer.forward(
            {
                "coordinates": reference_points,
                "connectivity_tensor": self._connectivity,
                "u": displacement_mm[:, 0:1],
                "v": displacement_mm[:, 1:2],
                "w": displacement_mm[:, 2:3],
            }
        )
        # Accumulated on the device; only the property pays a synchronization.
        self._inverted += (residuals["jacobian"] <= 0.0).sum().detach()

        weights = nodal_volumes / nodal_volumes.sum()
        energy = (residuals["neo_hookean_energy"].squeeze(-1) * weights).sum()
        incompressibility = (residuals["incompressibility"].squeeze(-1) * weights).sum()
        return energy, incompressibility


class TrainPhysicsNeMoPhysicsInformedMotion(TrainPhysicsNeMoMGN):
    """Train a MeshGraphNet whose loss also prices the tissue's strain energy.

    Identical to :class:`physiotwin4d.TrainPhysicsNeMoMGN` apart from the loss,
    which becomes ``data + lambda_physics * (energy + incompressibility)``.  The
    data term is scored on normalized targets, as before, while the physics term
    is scored in millimeters and kilopascals, so predictions are returned to
    physical units before the residual sees them.

    Call :meth:`set_mechanics`, :meth:`set_elements` and
    :meth:`set_reference_meshes` before training, unless ``lambda_physics`` is
    zero -- which reproduces the data-only MeshGraphNet exactly and is the
    ablation a physics-informed run is measured against.
    """

    model_tag = "physics_informed_motion"

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the physics-informed MeshGraphNet training method.

        Args:
            log_level: Logging level. Default: ``logging.INFO``.
        """
        super().__init__(log_level=log_level)
        self.lambda_physics: float = 0.1
        self._residual: Optional[PhysicsInformedMotion] = None
        self._reference_meshes: dict[str, Path] = {}
        self._tets: Optional[np.ndarray] = None
        # Per-subject reference geometry, resolved once at the start of train().
        self._reference_cache: dict[str, tuple[Any, Any]] = {}
        self._sample_subjects: list[str] = []
        # Epoch bookkeeping, so the two loss terms can be reported apart.
        # Summed on the device and read once per logged epoch, so separating the
        # terms costs no per-batch synchronization.
        self._epoch_data_loss: Optional["torch.Tensor"] = None
        self._epoch_physics_loss: Optional["torch.Tensor"] = None
        self._epoch_batches = 0

    def set_mechanics(
        self,
        residual: Optional[PhysicsInformedMotion],
        lambda_physics: float = 0.1,
    ) -> None:
        """Set the constitutive residual and how heavily it is weighted.

        Args:
            residual: Configured :class:`PhysicsInformedMotion`, or ``None`` to
                train on displacement alone.
            lambda_physics: Weight of the physics term.  ``0.0`` skips it
                entirely, reproducing the data-only MeshGraphNet, which is the
                ablation a physics-informed run is measured against.

        Raises:
            ValueError: If *lambda_physics* is negative, or is positive without
                a residual to evaluate.
        """
        if lambda_physics < 0.0:
            raise ValueError(f"lambda_physics must be >= 0, got {lambda_physics}")
        if lambda_physics > 0.0 and residual is None:
            raise ValueError(
                "lambda_physics is positive but no residual was given; pass a "
                "PhysicsInformedMotion, or set lambda_physics to 0."
            )
        self._residual = residual
        self.lambda_physics = lambda_physics

    @property
    def inverted_element_count(self) -> int:
        """Elements whose Jacobian went non-positive during training.

        Non-zero means the network predicted motion that turns tissue inside
        out somewhere, which the clamp keeps trainable but does not make
        physical.
        """
        if self._residual is None:
            return 0
        return self._residual.inverted_element_count

    def set_reference_meshes(self, reference_meshes: dict[str, Path]) -> None:
        """Set each subject's fitted reference mesh, keyed by subject id.

        These are the undeformed configurations the strain energy is measured
        against -- the same meshes the manifests name as
        ``fitted_reference_mesh``, whose points the targets are defined at.
        """
        self._reference_meshes = dict(reference_meshes)

    def set_elements(self, tets: np.ndarray) -> None:
        """Set the ``(n_tet, 4)`` template elements the residual is summed over."""
        self._tets = np.asarray(tets, dtype=np.int64)

    def train(
        self,
        train_dataset: PhaseSampleDataset,
        val_dataset: PhaseSampleDataset,
        stats: dict,
        context: DistributedContext,
        epochs: int,
        output_dir: Path,
        template_mesh: pv.DataSet,
        template_coords: np.ndarray,
        resume_from: Optional[Path] = None,
    ) -> tuple["torch.nn.Module", list[float], list[dict]]:
        """Bind each sample to its subject's reference geometry, then train."""
        assert not self._shuffle_points_within_batch, (
            "The physics residual indexes the template's elements, so it needs "
            "each sample's vertex order left intact."
        )
        if self.lambda_physics > 0.0:
            if self._residual is None:
                raise ValueError("Call set_mechanics() before training.")
            if self._tets is None:
                raise ValueError("Call set_elements() before training.")
            self._require_matching_device(context)
            self._sample_subjects = train_dataset.subject_ids
            self._bind_reference_meshes(context, len(template_coords))
        return super().train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            stats=stats,
            context=context,
            epochs=epochs,
            output_dir=output_dir,
            template_mesh=template_mesh,
            template_coords=template_coords,
            resume_from=resume_from,
        )

    def checkpoint_fields(self) -> dict:
        fields = super().checkpoint_fields()
        residual = self._residual
        fields.update(
            {
                "lambda_physics": self.lambda_physics,
                "mu_kpa": residual.mu_kpa if residual is not None else None,
                "lambda_lame_kpa": (
                    residual.lambda_lame_kpa if residual is not None else None
                ),
            }
        )
        return fields

    def _require_matching_device(self, context: DistributedContext) -> None:
        """Refuse a residual built on a device other than the one training uses.

        The residual's connectivity and compiled symbolic graph are bound when
        it is constructed and cannot be moved afterwards, while the reference
        geometry and the predictions live on ``context.device``. A mismatch
        surfaces deep inside the gradient reconstruction as an opaque tensor
        error, so it is caught here where it can name both devices.

        Raises:
            ValueError: If the residual's device is not the training device.
        """
        if self._residual is None:
            return
        # Full identity, not just the type: under distributed training each rank
        # owns one GPU, so a residual on cuda:0 is as unusable to a rank
        # training on cuda:1 as one left on the CPU would be.
        training_device = _resolved_device(context.device)
        if self._residual.device != training_device:
            raise ValueError(
                f"The physics residual was built on {self._residual.device} but "
                f"training runs on {training_device}; its connectivity and "
                "symbolic graph cannot meet the predictions. Pass "
                "device=<training device> when constructing "
                "PhysicsInformedMotion."
            )

    def _bind_reference_meshes(
        self, context: DistributedContext, n_points: int
    ) -> None:
        """Load every training subject's reference geometry once."""
        import torch

        missing = sorted(set(self._sample_subjects) - set(self._reference_meshes))
        if missing:
            raise ValueError(
                "No fitted reference mesh for subject(s) "
                f"{', '.join(missing)}; call set_reference_meshes() with one "
                "entry per training subject."
            )

        assert self._tets is not None
        self._reference_cache = {}
        device = context.device
        for subject_id in sorted(set(self._sample_subjects)):
            mesh = pv.read(str(self._reference_meshes[subject_id]))
            points = np.asarray(mesh.points, dtype=np.float64)
            if len(points) != n_points:
                raise ValueError(
                    f"{self._reference_meshes[subject_id]} has {len(points)} "
                    f"points, but the template has {n_points}."
                )
            _, nodal = tet_volumes(points, self._tets)
            reference = torch.from_numpy(points).to(device=device, dtype=torch.float32)
            volumes = torch.from_numpy(nodal).to(device=device, dtype=torch.float32)
            self._reference_cache[subject_id] = (reference, volumes)
        self._log_main(
            context,
            "Bound reference geometry for %d subjects.",
            len(self._reference_cache),
        )

    def _compute_loss(
        self,
        pred: "torch.Tensor",
        tgt: "torch.Tensor",
        batch_len: int,
        target_scale: float,
        indices: np.ndarray,
    ) -> "torch.Tensor":
        """Return the data loss plus the weighted neo-Hookean residual."""
        data_loss = super()._compute_loss(pred, tgt, batch_len, target_scale, indices)
        self._accumulate("_epoch_data_loss", data_loss)
        self._epoch_batches += 1
        if self.lambda_physics <= 0.0 or self._residual is None:
            return data_loss

        import torch

        # The residual is a physical quantity and has to be computed in float32.
        # Upcasting the input is not enough: this runs inside the training
        # loop's bf16 autocast, which intercepts the *operations*, so the
        # matmuls, det(F) and the least-squares solve would all be cast back
        # down whatever dtype they were handed. bf16 carries about three decimal
        # digits, which cannot tell an almost-inverted element from an inverted
        # one -- the distinction this loss exists to price.
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            displacement = pred.float() * target_scale
            n_points = displacement.shape[0] // batch_len
            energy = displacement.new_zeros(())
            incompressibility = displacement.new_zeros(())
            for position, index in enumerate(indices):
                subject_id = self._sample_subjects[int(index)]
                reference, volumes = self._reference_cache[subject_id]
                rows = displacement[position * n_points : (position + 1) * n_points]
                sample_energy, sample_incompressibility = self._residual(
                    reference, rows, volumes
                )
                energy = energy + sample_energy
                incompressibility = incompressibility + sample_incompressibility

            physics_loss = (energy + incompressibility) / max(batch_len, 1)
        self._accumulate("_epoch_physics_loss", physics_loss)
        return data_loss + self.lambda_physics * physics_loss

    def _accumulate(self, name: str, value: "torch.Tensor") -> None:
        """Add *value* to the named epoch accumulator, on its own device."""
        running = getattr(self, name)
        detached = value.detach()
        setattr(self, name, detached if running is None else running + detached)

    def _log_epoch(self, context: DistributedContext, epoch: int, epochs: int) -> None:
        """Report the data and physics terms apart, then start the next epoch.

        The total alone cannot say how the two balance: the data term is scored
        on normalized displacement and the physics term in millimeters and
        kilopascals, so ``lambda_physics`` is only choosable by watching them
        separately.

        Every rank saw a disjoint slice, so the sums are pooled before they are
        divided, exactly as the epoch total above them is.  Reporting one rank's
        slice beside a total covering all of them would make the two disagree
        for no visible reason.  This runs on every rank because the reduction is
        collective; only rank 0 prints.
        """
        data = self._epoch_data_loss
        physics = self._epoch_physics_loss
        data_sum, batches = self._reduce_sums(
            context,
            float(data.item()) if data is not None else 0.0,
            self._epoch_batches,
        )
        physics_sum, inverted = self._reduce_sums(
            context,
            float(physics.item()) if physics is not None else 0.0,
            self.inverted_element_count,
        )

        divisor = max(batches, 1)
        physics_mean = physics_sum / divisor
        self._log_main(
            context,
            "    data=%.6f  physics=%.6f  (weighted %.6f)  inverted=%d",
            data_sum / divisor,
            physics_mean,
            self.lambda_physics * physics_mean,
            inverted,
        )
        self._epoch_data_loss = None
        self._epoch_physics_loss = None
        self._epoch_batches = 0
