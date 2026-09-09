"""
Tests for Greedy-based image registration.

Uses the same fixtures as test_register_images_ANTS (converted 3D CT images).
Requires the picsl-greedy package and test data.
"""

from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest

from monai_physio.register_images_greedy import RegisterImagesGreedy
from monai_physio.transform_tools import TransformTools

from .conftest import KnownAffineCase, KnownShiftCase


@pytest.mark.slow
class TestRegisterImagesGreedy:
    """Test suite for Greedy-based image registration."""

    def test_registrar_initialization(
        self, registrar_greedy: RegisterImagesGreedy
    ) -> None:
        """Test that RegisterImagesGreedy initializes correctly."""
        assert registrar_greedy is not None, "Registrar not initialized"
        assert hasattr(registrar_greedy, "fixed_image"), "Missing fixed_image attribute"
        assert hasattr(registrar_greedy, "fixed_mask"), "Missing fixed_mask attribute"

        print("\nGreedy registrar initialized successfully")

    def test_set_modality(self, registrar_greedy: RegisterImagesGreedy) -> None:
        """Test setting imaging modality."""
        registrar_greedy.set_modality("ct")
        assert registrar_greedy.modality == "ct", "Modality not set correctly"

        registrar_greedy.set_modality("mr")
        assert registrar_greedy.modality == "mr", "Modality change failed"

        print("\nModality setting works correctly")

    def test_set_transform_type_and_metric(
        self, registrar_greedy: RegisterImagesGreedy
    ) -> None:
        """Test setting transform type and metric."""
        registrar_greedy.set_transform_type("Rigid")
        assert registrar_greedy.transform_type == "Rigid"

        registrar_greedy.set_transform_type("Affine")
        assert registrar_greedy.transform_type == "Affine"

        registrar_greedy.set_transform_type("Deformable")
        assert registrar_greedy.transform_type == "Deformable"

        registrar_greedy.set_metric("CC")
        assert registrar_greedy.metric == "CC"
        registrar_greedy.set_metric("Mattes")
        assert registrar_greedy.metric == "Mattes"
        registrar_greedy.set_metric("MeanSquares")
        assert registrar_greedy.metric == "MeanSquares"

        with pytest.raises(ValueError, match="Invalid transform type"):
            registrar_greedy.set_transform_type("Invalid")
        with pytest.raises(ValueError, match="Invalid metric"):
            registrar_greedy.set_metric("Invalid")

        print("\nTransform type and metric setting work correctly")

    def test_set_fixed_image(
        self, registrar_greedy: RegisterImagesGreedy, test_images: list[Any]
    ) -> None:
        """Test setting fixed image."""
        fixed_image = test_images[0]
        registrar_greedy.set_fixed_image(fixed_image)
        assert registrar_greedy.fixed_image is not None, "Fixed image not set"

        print("\nFixed image set successfully")
        print(f"  Image size: {itk.size(registrar_greedy.fixed_image)}")

    def test_register_affine_without_mask(
        self,
        registrar_greedy: RegisterImagesGreedy,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test affine registration without masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_greedy"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        print("\nGreedy affine registration without mask...")

        registrar_greedy.set_modality("ct")
        registrar_greedy.set_transform_type("Affine")
        registrar_greedy.set_fixed_image(fixed_image)

        result = registrar_greedy.register(moving_image=moving_image)

        assert isinstance(result, dict), "Result should be a dictionary"
        assert "moving_to_fixed_transform" in result, (
            "Missing moving_to_fixed_transform in result"
        )
        assert "fixed_to_moving_transform" in result, (
            "Missing fixed_to_moving_transform in result"
        )

        moving_to_fixed_transform = result["moving_to_fixed_transform"]
        fixed_to_moving_transform = result["fixed_to_moving_transform"]

        assert moving_to_fixed_transform is not None, (
            "moving_to_fixed_transform is None"
        )
        assert fixed_to_moving_transform is not None, (
            "fixed_to_moving_transform is None"
        )

        print("Greedy affine registration complete without mask")

        itk.transformwrite(
            [moving_to_fixed_transform],
            str(reg_output_dir / "greedy_affine_inverse_no_mask.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [fixed_to_moving_transform],
            str(reg_output_dir / "greedy_affine_forward_no_mask.hdf"),
            compression=True,
        )

    def test_register_affine_with_mask(
        self,
        registrar_greedy: RegisterImagesGreedy,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test affine registration with binary masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_greedy"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        fixed_size_itk = itk.size(fixed_image)
        moving_size_itk = itk.size(moving_image)
        fixed_size = (
            int(fixed_size_itk[0]),
            int(fixed_size_itk[1]),
            int(fixed_size_itk[2]),
        )
        moving_size = (
            int(moving_size_itk[0]),
            int(moving_size_itk[1]),
            int(moving_size_itk[2]),
        )

        fixed_mask_arr = np.zeros(fixed_size[::-1], dtype=np.uint8)
        moving_mask_arr = np.zeros(moving_size[::-1], dtype=np.uint8)
        fixed_mask_arr[
            fixed_size[2] // 4 : 3 * fixed_size[2] // 4,
            fixed_size[1] // 4 : 3 * fixed_size[1] // 4,
            fixed_size[0] // 4 : 3 * fixed_size[0] // 4,
        ] = 1
        moving_mask_arr[
            moving_size[2] // 4 : 3 * moving_size[2] // 4,
            moving_size[1] // 4 : 3 * moving_size[1] // 4,
            moving_size[0] // 4 : 3 * moving_size[0] // 4,
        ] = 1

        fixed_mask = itk.image_from_array(fixed_mask_arr)
        fixed_mask.CopyInformation(fixed_image)
        moving_mask = itk.image_from_array(moving_mask_arr)
        moving_mask.CopyInformation(moving_image)

        registrar_greedy.set_modality("ct")
        registrar_greedy.set_transform_type("Affine")
        registrar_greedy.set_fixed_image(fixed_image)
        registrar_greedy.set_fixed_mask(fixed_mask)

        result = registrar_greedy.register(
            moving_image=moving_image, moving_mask=moving_mask
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["moving_to_fixed_transform"] is not None
        assert result["fixed_to_moving_transform"] is not None

        print("Greedy affine registration complete with masks")

    @pytest.mark.parametrize("transform_type", ["Rigid", "Affine", "Deformable"])
    def test_recovers_known_shift(
        self,
        known_shift_case: KnownShiftCase,
        transform_type: str,
    ) -> None:
        """Greedy must recover a known shift, in the right direction.

        Regression guard for the RAS/LPS conversion: Greedy reports its affine
        in RAS while ITK is LPS, so omitting the basis change negates x and y.
        Before the fix this recovered (+6, -4, -3) mm instead of (-6, +4, -3)
        and scored *below* the unregistered pair; the sign error passed every
        other test in this file, which only check that transforms exist.
        """
        registrar = RegisterImagesGreedy()
        registrar.set_modality("ct")
        registrar.set_transform_type(transform_type)
        registrar.set_number_of_iterations([60, 30, 10])
        registrar.set_fixed_image(known_shift_case.fixed)

        result = registrar.register(moving_image=known_shift_case.moving)
        fixed_to_moving_transform = result["fixed_to_moving_transform"]

        error_mm = known_shift_case.center_error_mm(fixed_to_moving_transform)
        ncc = known_shift_case.foreground_ncc(fixed_to_moving_transform)
        unregistered_ncc = known_shift_case.unregistered_ncc()

        print(f"\nGreedy {transform_type} known-shift recovery:")
        print(f"  expected displacement: {known_shift_case.expected_displacement}")
        print(f"  error: {error_mm:.2f} mm")
        print(f"  foreground NCC: {ncc:.4f} (unregistered {unregistered_ncc:.4f})")

        assert error_mm < 2.0, (
            f"Greedy {transform_type} recovered the shift {error_mm:.2f} mm off; "
            "a sign or axis error inverts the transform (see the RAS/LPS "
            "conversion in RegisterImagesGreedy._matrix_to_itk_affine)"
        )
        assert ncc > unregistered_ncc, (
            f"Greedy {transform_type} left the images less aligned than they "
            f"started ({ncc:.4f} vs {unregistered_ncc:.4f})"
        )

    def test_recovers_a_known_affine_at_any_distance_from_the_origin(
        self,
        known_affine_case_near_origin: KnownAffineCase,
        known_affine_case_far_from_origin: KnownAffineCase,
    ) -> None:
        """Recovering a known rotation must not depend on where the grid sits.

        ``test_recovers_known_shift`` moves content by a pure translation, so the
        affine's linear block is the identity and every reading of that block
        agrees. This rotates, and runs the same rotation twice: once near the
        world origin, once on a grid at ``z ~ 1800 mm`` -- the CT table
        coordinates the cardiac cohorts live in.

        The comparison between the two is the measurement, not either error on
        its own. ``RegisterImagesGreedy._matrix_to_itk_affine`` reads Greedy's
        4x4 as a world affine about the origin, ``y = Mx + t``, and encodes it
        with ``SetCenter(0, 0, 0)``. If that reading is right, distance from the
        origin is irrelevant and the two cases score alike. If it is wrong, the
        error scales with ``|p|``, so a few degrees at ``z ~ 1800`` becomes tens
        of millimeters while the near case stays clean.

        Absolute error on a single attempt is the wrong instrument. Greedy
        seeds nondeterministically and diverges outright every few runs --
        ``vnl_lbfgs`` reports a Netlib failure and the recovered affine is
        hundreds of millimeters out -- and it does so far more readily at
        ``z ~ 1800`` than near the origin, because an affine applied about the
        world origin is badly conditioned that far from it. That unreliability
        is real, and is the same divergence that strands ICON with a constant
        image, but it is a separate concern from the question here. So each grid
        gets a few attempts and is judged on its best: a misread convention
        would fail *every* attempt at ``z ~ 1800``, not one in three.

        The probes are spread across the volume rather than taken at its center,
        because a linear-block error is invisible at a single point -- any one
        displacement can be absorbed by the translation.
        """

        def best_of(case: KnownAffineCase, attempts: int = 3) -> float:
            errors = []
            for _ in range(attempts):
                registrar = RegisterImagesGreedy()
                registrar.set_modality("ct")
                registrar.set_transform_type("Affine")
                registrar.set_number_of_iterations([60, 30, 10])
                registrar.set_fixed_image(case.fixed)
                result = registrar.register(moving_image=case.moving)
                errors.append(
                    float(
                        case.probe_errors_mm(result["fixed_to_moving_transform"]).max()
                    )
                )
            diverged = sum(1 for value in errors if value > 10.0)
            print(
                f"  offset {case.origin_offset_mm}: worst-probe "
                f"{np.round(errors, 2).tolist()} mm, {diverged}/{attempts} diverged"
            )
            return min(errors)

        print("Greedy known-affine recovery:")
        near = best_of(known_affine_case_near_origin)
        far = best_of(known_affine_case_far_from_origin)
        print(f"  best near={near:.2f} mm  far={far:.2f} mm")

        assert far < 4.0, (
            f"Greedy never recovered the known affine at z ~ 1800 mm; its best "
            f"of three attempts was {far:.2f} mm off at the worst probe, against "
            f"{near:.2f} mm near the origin. Failing every attempt only far from "
            "the origin means the linear block is not the world-origin affine "
            "that RegisterImagesGreedy._matrix_to_itk_affine assumes when it "
            "pairs SetMatrix(M) with SetCenter(0, 0, 0)."
        )
        assert far - near < 3.0, (
            f"Recovery degraded by {far - near:.2f} mm when the same rotation "
            f"moved to z ~ 1800 mm (near {near:.2f} mm, far {far:.2f} mm), which "
            "is the signature of a linear block being applied about the wrong "
            "center."
        )

    def test_transform_application(
        self,
        registrar_greedy: RegisterImagesGreedy,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test applying registration transform to moving image."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_greedy"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        registrar_greedy.set_modality("ct")
        registrar_greedy.set_transform_type("Affine")
        registrar_greedy.set_fixed_image(fixed_image)
        result = registrar_greedy.register(moving_image=moving_image)

        fixed_to_moving_transform = result["fixed_to_moving_transform"]
        transform_tools = TransformTools()
        registered_image = transform_tools.transform_image(
            moving_image,
            fixed_to_moving_transform,
            fixed_image,
            interpolation_method="linear",
        )

        assert registered_image is not None, "Registered image is None"
        assert itk.size(registered_image) == itk.size(fixed_image), "Size mismatch"

        moving_arr = itk.array_from_image(moving_image)
        registered_arr = itk.array_from_image(registered_image)
        difference = np.sum(
            np.abs(moving_arr.astype(float) - registered_arr.astype(float))
        )

        print("Greedy transform applied successfully")
        print(f"  Registered image size: {itk.size(registered_image)}")
        print(f"  Total difference: {difference:.2f}")

        itk.imwrite(
            registered_image,
            str(reg_output_dir / "greedy_registered_image.mha"),
            compression=True,
        )


if __name__ == "__main__":
    pytest.main([__file__])
