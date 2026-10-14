"""FireANTs-based image registration implementation.

This module provides the RegisterImagesFireants class, a concrete implementation
of RegisterImagesBase that uses the GPU-accelerated FireANTs library for rigid,
affine, and deformable image registration.

See https://fireants.readthedocs.io/ and https://pypi.org/project/fireants/.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from typing import Any, ClassVar, Optional, Union

import itk
import numpy as np
import torch
from numpy.typing import NDArray

from .process_images import ProcessImages
from .process_transforms import ProcessTransforms
from .register_images_base import RegisterImagesBase


def _try_import_fireants(module: str) -> Any:
    """Import a ``fireants`` submodule; raise with a helpful message if missing.

    ``fireants.io.image`` imports the Unix-only ``fcntl`` module at import time
    and only uses it when writing images. When ``import fcntl`` fails (Windows),
    an empty stub module is registered so the package can be imported; the
    installed package is not modified.
    """
    try:
        import fcntl  # noqa: F401
    except ImportError:
        sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))
    try:
        return importlib.import_module(f"fireants.{module}")
    except ImportError as e:
        raise ImportError(
            "fireants is required for RegisterImagesFireants. "
            "Install with: pip install fireants --no-deps"
        ) from e


class RegisterImagesFireants(RegisterImagesBase):
    """FireANTs-based image registration implementation.

    This class extends RegisterImagesBase to provide rigid, affine, and deformable
    image registration on the GPU using FireANTs. FireANTs transforms are expressed
    in physical (LPS) coordinates as ``y = A x + t`` from the fixed to the moving
    image, which is the ITK ``fixed_to_moving_transform`` directly.

    Attributes:
        number_of_iterations: List of iterations per level (e.g. [40, 20, 10]).
            Level ``i`` of ``n`` runs at downsampling factor ``2 ** (n - 1 - i)``.
        transform_type: 'Deformable', 'Affine', or 'Rigid'
        metric: 'CC' (-> cc), 'Mattes' (-> mi), or 'MeanSquares' (-> mse)
        scales: Optional explicit downsampling factor per level (coarse to
            fine); when None they follow from ``number_of_iterations``.
        init_affine: Source of the deformable stage's initial transform:
            'Affine' (FireANTs affine registration) or 'Identity' (none).
        deformable_options: Keyword arguments forwarded to FireANTs'
            ``GreedyRegistration`` (see ``set_deformable_options``).
    """

    _DEFORMABLE_OPTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "cc_kernel_size",
            "cc_kernel_type",
            "smooth_grad_sigma",
            "smooth_warp_sigma",
            "optimizer",
            "optimizer_lr",
            "deformation_type",
            "integrator_n",
            "blur",
            "max_tolerance_iters",
            "tolerance",
        }
    )

    _FIREANTS_LOSS: ClassVar[dict[str, str]] = {
        "CC": "cc",
        "Mattes": "mi",
        "MeanSquares": "mse",
    }

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the FireANTs image registration class.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.number_of_iterations: list[int] = [40, 20, 10]
        self.transform_type = "Deformable"
        self.metric = "CC"
        self.scales: Optional[list[float]] = None
        self.deformable_options: dict[str, Any] = {}
        self.init_affine = "Affine"

    def set_number_of_iterations(self, number_of_iterations: list[int]) -> None:
        """Set the number of iterations per resolution level.

        Args:
            number_of_iterations: List of iterations (e.g. [40, 20, 10]).
        """
        self.number_of_iterations = number_of_iterations

    def set_transform_type(self, transform_type: str) -> None:
        """Set the type of transform: Deformable, Affine, or Rigid.

        Args:
            transform_type: 'Deformable', 'Affine', or 'Rigid'.
        """
        if transform_type not in ("Deformable", "Affine", "Rigid"):
            self.log_error("Invalid transform type: %s", transform_type)
            raise ValueError(f"Invalid transform type: {transform_type}")
        self.transform_type = transform_type

    def set_metric(self, metric: str) -> None:
        """Set the similarity metric (CC, Mattes->mi, MeanSquares->mse).

        Args:
            metric: 'CC', 'Mattes', or 'MeanSquares'.
        """
        if metric not in self._FIREANTS_LOSS:
            self.log_error("Invalid metric: %s", metric)
            raise ValueError(f"Invalid metric: {metric}")
        self.metric = metric

    def set_scales(self, scales: Optional[list[float]]) -> None:
        """Set the downsampling factor per level (one per iteration entry).

        Args:
            scales: Factors in descending order (e.g. [4, 2, 1]), or None to
                derive them from the number of iterations.
        """
        if scales is not None and len(scales) != len(self.number_of_iterations):
            raise ValueError("scales and number_of_iterations must have equal length")
        self.scales = scales

    def set_init_affine(self, init_affine: str) -> None:
        """Set the initial transform of the deformable stage.

        Args:
            init_affine: 'Affine' to register an affine first, or 'Identity'.
        """
        if init_affine not in ("Affine", "Identity"):
            self.log_error("Invalid init_affine: %s", init_affine)
            raise ValueError(f"Invalid init_affine: {init_affine}")
        self.init_affine = init_affine

    def set_deformable_options(self, **options: Any) -> None:
        """Set ``GreedyRegistration`` options for the deformable stage.

        Args:
            **options: Any of ``cc_kernel_size``, ``cc_kernel_type``,
                ``smooth_grad_sigma``, ``smooth_warp_sigma``, ``optimizer``,
                ``optimizer_lr``, ``deformation_type``, ``integrator_n``,
                ``blur``, ``max_tolerance_iters``, ``tolerance``.
        """
        unknown = set(options) - self._DEFORMABLE_OPTIONS
        if unknown:
            raise ValueError(f"Invalid deformable options: {sorted(unknown)}")
        self.deformable_options.update(options)

    def configure_for_heart_ct(self) -> None:
        """Apply the tuned deformable settings for cardiac 4D CT.

        Values from experiments/Registration-Fireants_vs_Greedy/results/REPORT.md:
        deformable-only (identity initial transform) CC registration.
        """
        self.set_transform_type("Deformable")
        self.set_metric("CC")
        self.set_number_of_iterations([80, 40, 20])
        self.set_scales([4.0, 2.0, 1.0])
        self.set_init_affine("Identity")
        self.set_deformable_options(
            optimizer_lr=1.0,
            smooth_warp_sigma=1.0,
            smooth_grad_sigma=1.0,
            cc_kernel_size=9,
        )

    def configure_for_lung_ct(self) -> None:
        """Apply the tuned deformable settings for lung 4D CT.

        Values from experiments/Registration-Fireants_vs_Greedy/results/REPORT.md:
        deformable-only (identity initial transform) CC registration at coarser
        scales than the heart.
        """
        self.set_transform_type("Deformable")
        self.set_metric("CC")
        self.set_number_of_iterations([20, 10, 5])
        self.set_scales([8.0, 4.0, 2.0])
        self.set_init_affine("Identity")
        self.set_deformable_options(
            optimizer_lr=1.0,
            smooth_warp_sigma=1.0,
            smooth_grad_sigma=1.0,
            cc_kernel_size=9,
        )

    def _scales(self) -> list[float]:
        """Downsampling factor per level, coarse to fine."""
        if self.scales is not None:
            return list(self.scales)
        n = len(self.number_of_iterations)
        return [float(2 ** (n - 1 - i)) for i in range(n)]

    def _to_fireants_image(
        self, image: itk.Image, mask: Optional[itk.Image] = None, masked: bool = False
    ) -> Any:
        """Wrap an ITK image as a batched FireANTs image on the GPU.

        When ``masked`` is True the mask is appended as the last channel, as the
        FireANTs masked losses expect; a missing mask becomes all ones.
        """
        fireants_image = _try_import_fireants("io.image")
        sitk_image = ProcessImages().convert_itk_image_to_sitk(image)
        wrapped = fireants_image.Image(sitk_image, device="cuda")
        if masked:
            imagemask = _try_import_fireants("io.imagemask")
            if mask is None:
                wrapped = imagemask.generate_and_apply_mask_allones(wrapped)
            else:
                sitk_mask = ProcessImages().convert_itk_image_to_sitk(mask)
                wrapped = imagemask.apply_mask_to_image(
                    wrapped, fireants_image.Image(sitk_mask, device="cuda")
                )
        return fireants_image.BatchedImages([wrapped])

    @staticmethod
    def _matrix_to_itk_affine(mat_4x4: NDArray[np.float64]) -> itk.Transform:
        """Convert a 4x4 physical-space ``y = A x + t`` matrix to an ITK affine."""
        mat_4x4 = np.asarray(mat_4x4, dtype=np.float64)
        if mat_4x4.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {mat_4x4.shape}")
        affine_tfm = itk.AffineTransform[itk.D, 3].New()
        affine_tfm.SetCenter(itk.Point[itk.D, 3]())
        affine_tfm.SetMatrix(itk.GetMatrixFromArray(mat_4x4[:3, :3].copy()))
        translation = itk.Vector[itk.D, 3]()
        for i in range(3):
            translation[i] = float(mat_4x4[i, 3])
        affine_tfm.SetTranslation(translation)
        return affine_tfm

    def _registration_method_affine_or_rigid(
        self,
        fixed: Any,
        moving: Any,
        loss_type: str,
        transform_type: str,
    ) -> tuple[NDArray[np.float64], float]:
        """Run FireANTs rigid or affine registration. Returns (4x4 matrix, loss)."""
        if transform_type == "Rigid":
            registration_class = _try_import_fireants(
                "registration.rigid"
            ).RigidRegistration
        else:
            registration_class = _try_import_fireants(
                "registration.affine"
            ).AffineRegistration
        registration = registration_class(
            self._scales(),
            list(self.number_of_iterations),
            fixed,
            moving,
            loss_type=loss_type,
            progress_bar=False,
        )
        registration.optimize()

        with torch.no_grad():
            matrix_fn = (
                registration.get_rigid_matrix
                if transform_type == "Rigid"
                else registration.get_affine_matrix
            )
            mat = matrix_fn(homogenous=True)[0].double().cpu().numpy()
            moved = registration.evaluate(fixed, moving)
            loss = float(registration.loss_fn(moved, fixed()))
        self.log_info("FireANTs %s registration loss: %s", transform_type, loss)
        return mat, loss

    def _registration_method_deformable(
        self,
        fixed: Any,
        moving: Any,
        loss_type: str,
        reference_image: itk.Image,
    ) -> tuple[NDArray[np.float64], itk.Transform, float]:
        """Run FireANTs affine then deformable registration.

        The deformable stage starts from the affine result (or identity). Returns the 4x4
        affine, the residual displacement transform on ``reference_image``'s
        grid (applied before the affine, so ``y = A (x + d(x))``), and the loss.
        """
        if self.init_affine == "Affine":
            mat, _ = self._registration_method_affine_or_rigid(
                fixed, moving, loss_type, "Affine"
            )
        else:
            mat = np.eye(4)
        registration = _try_import_fireants("registration.greedy").GreedyRegistration(
            self._scales(),
            list(self.number_of_iterations),
            fixed,
            moving,
            loss_type=loss_type,
            init_affine=torch.from_numpy(mat).float().cuda()[None],
            progress_bar=False,
            **self.deformable_options,
        )
        registration.optimize()

        with torch.no_grad():
            # The warp is a displacement in the moving image's normalized torch
            # coordinates, applied after the affine: y = A x + L d(x), with L the
            # linear part of moving's torch-to-physical map. Re-expressing it
            # before the affine gives d' = A^-1 L d, so the pair composes as an
            # ITK composite [affine, displacement].
            warp = registration.get_warp_parameters(fixed, moving)["grid"][0]
            torch_to_physical = moving.get_torch2phy()[0, :3, :3].float()
            affine_inverse = torch.from_numpy(np.linalg.inv(mat[:3, :3])).float()
            to_pre_affine = affine_inverse.cuda() @ torch_to_physical
            displacement = torch.einsum("ij,zyxj->zyxi", to_pre_affine, warp)
            displacement_array = displacement.double().cpu().numpy()
            moved = registration.evaluate(fixed, moving)
            loss = float(registration.loss_fn(moved, fixed()))
        self.log_info("FireANTs Deformable registration loss: %s", loss)

        displacement_image = ProcessImages().convert_array_to_image_of_vectors(
            displacement_array, reference_image, itk.D
        )
        displacement_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
        displacement_transform.SetDisplacementField(displacement_image)
        return mat, displacement_transform, loss

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Register moving image to fixed image using FireANTs.

        Returns a dict with "fixed_to_moving_transform", "moving_to_fixed_transform",
        and "loss". As with the other image-registration backends,
        fixed_to_moving_transform warps the moving image onto the fixed grid and
        moving_to_fixed_transform warps the fixed image onto the moving grid; point
        and landmark warps use the opposite transform from image warps (see
        docs/developer/transform_conventions).

        Raises:
            ValueError: If the fixed image is not set.
            NotImplementedError: For ``moving_labelmap`` or ``fixed_labelmap``,
                which FireANTs cannot use as a similarity term. Masks are used
                through the FireANTs masked loss; a mask on only one image is
                paired with an all-ones mask on the other.
        """
        if self.fixed_image is None or self.fixed_image_pre is None:
            raise ValueError("Fixed image must be set before registration.")
        for name, value in (
            ("moving_labelmap", moving_labelmap),
            ("fixed_labelmap", self.fixed_labelmap),
        ):
            if value is not None:
                raise NotImplementedError(
                    f"RegisterImagesFireants does not support {name}."
                )

        moving_pre = moving_image_pre if moving_image_pre is not None else moving_image
        masked = moving_mask is not None or self.fixed_mask is not None
        loss_type = ("masked_" if masked else "") + self._FIREANTS_LOSS[self.metric]
        fixed = self._to_fireants_image(self.fixed_image_pre, self.fixed_mask, masked)
        moving = self._to_fireants_image(moving_pre, moving_mask, masked)

        fixed_to_moving_transform: itk.Transform
        moving_to_fixed_transform: itk.Transform
        if self.transform_type == "Deformable":
            mat, displacement, loss_val = self._registration_method_deformable(
                fixed, moving, loss_type, self.fixed_image_pre
            )
            affine = self._matrix_to_itk_affine(mat)
            inverse_affine = itk.AffineTransform[itk.D, 3].New()
            affine.GetInverse(inverse_affine)
            forward = itk.CompositeTransform[itk.D, 3].New()
            forward.AddTransform(affine)
            forward.AddTransform(displacement)
            inverse = itk.CompositeTransform[itk.D, 3].New()
            inverse.AddTransform(
                ProcessTransforms().invert_displacement_field_transform(displacement)
            )
            inverse.AddTransform(inverse_affine)
            fixed_to_moving_transform = forward
            moving_to_fixed_transform = inverse
        else:
            mat, loss_val = self._registration_method_affine_or_rigid(
                fixed, moving, loss_type, self.transform_type
            )
            fixed_to_moving_transform = self._matrix_to_itk_affine(mat)
            moving_to_fixed_transform = itk.AffineTransform[itk.D, 3].New()
            fixed_to_moving_transform.GetInverse(moving_to_fixed_transform)

        return {
            "fixed_to_moving_transform": fixed_to_moving_transform,
            "moving_to_fixed_transform": moving_to_fixed_transform,
            "loss": loss_val,
        }
