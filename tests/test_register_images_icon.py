"""
Test for ICON-based image registration.

This test depends on test_convert_image_4d_to_3d and uses the converted
3D CT images to test ICON registration functionality.
Note: ICON requires CUDA-enabled GPU.
"""

from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pytest

from physiotwin4d.register_images_icon import RegisterImagesICON
from physiotwin4d.transform_tools import TransformTools

from .conftest import KnownShiftCase


@pytest.mark.requires_gpu
@pytest.mark.slow
class TestRegisterImagesICON:
    """Test suite for ICON-based image registration."""

    def test_recovers_known_shift(self, known_shift_case: KnownShiftCase) -> None:
        """ICON must recover a known shift, in the right direction.

        The companion of the Greedy check in test_register_images_greedy.py:
        ICON returns ITK transforms in the LPS frame throughout, so this asserts
        no equivalent RAS/LPS conversion is missing here.
        """
        registrar = RegisterImagesICON()
        registrar.set_modality("ct")
        registrar.set_number_of_iterations(5)
        registrar.set_fixed_image(known_shift_case.fixed)

        result = registrar.register(moving_image=known_shift_case.moving)
        forward_transform = result["forward_transform"]

        error_mm = known_shift_case.center_error_mm(forward_transform)
        ncc = known_shift_case.foreground_ncc(forward_transform)
        unregistered_ncc = known_shift_case.unregistered_ncc()

        print("\nICON known-shift recovery:")
        print(f"  error: {error_mm:.2f} mm")
        print(f"  foreground NCC: {ncc:.4f} (unregistered {unregistered_ncc:.4f})")

        assert error_mm < 2.0, f"ICON recovered the shift {error_mm:.2f} mm off"
        assert ncc > unregistered_ncc, (
            f"ICON left the images less aligned than they started "
            f"({ncc:.4f} vs {unregistered_ncc:.4f})"
        )

    def test_a_reused_network_matches_a_freshly_built_one(
        self, known_shift_case: KnownShiftCase
    ) -> None:
        """Reusing a network must not shift the answer more than noise does.

        Networks are shared across registrars, because building one costs four
        3D U-Nets on the CPU plus a second host copy of the checkpoint, and the
        workflows construct a registrar per frame -- hundreds of rebuilds in a
        cohort run, which is what walked into the OOM killer.

        Sharing is sound because ``finetune_execute`` restores the weights it
        started from before returning. That is a property of a third-party
        function, so it is measured here rather than assumed.

        Both distributions are measured in the same run, and the comparison is
        between them. An earlier version asserted a single reused-vs-fresh
        distance against a fixed 1e-3 tolerance and failed: that threshold sits
        *below* ICON's own run-to-run spread, so it could only ever report
        nondeterminism. Comparing like with like calibrates the test to whatever
        the machine's noise happens to be on the day.
        """

        def register() -> Any:
            registrar = RegisterImagesICON()
            registrar.set_modality("ct")
            registrar.set_number_of_iterations(5)
            registrar.set_fixed_image(known_shift_case.fixed)
            return registrar.register(moving_image=known_shift_case.moving)[
                "forward_transform"
            ]

        size = itk.size(known_shift_case.fixed)
        probes = [
            known_shift_case.fixed.TransformIndexToPhysicalPoint(
                [int(size[axis]) // divisor for axis in range(3)]
            )
            for divisor in (4, 2)
        ]

        def distance(first: Any, second: Any) -> float:
            return max(
                float(
                    np.linalg.norm(
                        np.array(list(first.TransformPoint(point)))
                        - np.array(list(second.TransformPoint(point)))
                    )
                )
                for point in probes
            )

        attempts = 3
        # Clearing between each is what makes this arm genuinely "fresh": a new
        # registrar alone would pick the cached network up.
        fresh = []
        for _ in range(attempts):
            RegisterImagesICON.clear_net_cache()
            fresh.append(register())
        # No clearing, so every one of these reuses the network the first built.
        reused = [register() for _ in range(attempts)]

        within_fresh = [
            distance(fresh[i], fresh[j])
            for i in range(attempts)
            for j in range(i + 1, attempts)
        ]
        across = [distance(a, b) for a in reused for b in fresh]
        noise = float(np.median(within_fresh))
        effect = float(np.median(across))
        print(f"fresh vs fresh median {noise:.5f} mm, reused vs fresh {effect:.5f} mm")

        assert effect < 4.0 * noise + 5.0e-3, (
            f"A reused network differed from a freshly built one by "
            f"{effect:.5f} mm at the median, against a {noise:.5f} mm median "
            "between two fresh ones. That is outside the method's own "
            "nondeterminism, so finetune_execute is no longer restoring the "
            "weights it started from and RegisterImagesICON._net_cache is "
            "unsound."
        )

    def test_registrar_initialization(self, registrar_ICON: RegisterImagesICON) -> None:
        """Test that RegisterImagesICON initializes correctly."""
        assert registrar_ICON is not None, "Registrar not initialized"
        assert hasattr(registrar_ICON, "fixed_image"), "Missing fixed_image attribute"
        assert hasattr(registrar_ICON, "fixed_mask"), "Missing fixed_mask attribute"
        assert hasattr(registrar_ICON, "number_of_iterations"), (
            "Missing number_of_iterations attribute"
        )
        assert hasattr(registrar_ICON, "net"), "Missing net attribute (ICON network)"

        print("\nICON registrar initialized successfully")
        print(f"  Default iterations: {registrar_ICON.number_of_iterations}")

    def test_set_modality(self, registrar_ICON: RegisterImagesICON) -> None:
        """Test setting imaging modality."""
        registrar_ICON.set_modality("ct")
        assert registrar_ICON.modality == "ct", "Modality not set correctly"

        registrar_ICON.set_modality("mr")
        assert registrar_ICON.modality == "mr", "Modality change failed"

        print("\nModality setting works correctly")

    def test_set_number_of_iterations(self, registrar_ICON: RegisterImagesICON) -> None:
        """Test setting number of iterations."""
        registrar_ICON.set_number_of_iterations(10)
        assert registrar_ICON.number_of_iterations == 10, "Number of iterations not set"

        registrar_ICON.set_number_of_iterations(5)
        assert registrar_ICON.number_of_iterations == 5, (
            "Number of iterations update failed"
        )

        print("\nNumber of iterations setting works correctly")

    def test_set_fixed_image(
        self, registrar_ICON: RegisterImagesICON, test_images: list[Any]
    ) -> None:
        """Test setting fixed image."""
        fixed_image = test_images[0]

        registrar_ICON.set_fixed_image(fixed_image)
        assert registrar_ICON.fixed_image is not None, "Fixed image not set"

        print("\nFixed image set successfully")
        print(f"  Image size: {itk.size(registrar_ICON.fixed_image)}")
        print(f"  Image spacing: {itk.spacing(registrar_ICON.fixed_image)}")

    def test_set_mass_preservation(self, registrar_ICON: RegisterImagesICON) -> None:
        """Test setting mass preservation flag."""
        registrar_ICON.set_mass_preservation(True)
        enabled = registrar_ICON.use_mass_preservation
        assert enabled, "Mass preservation not set"

        registrar_ICON.set_mass_preservation(False)
        disabled = registrar_ICON.use_mass_preservation
        assert not disabled, "Mass preservation update failed"

        print("\nMass preservation setting works correctly")

    def test_set_multi_modality(self, registrar_ICON: RegisterImagesICON) -> None:
        """Test setting multi-modality flag."""
        registrar_ICON.set_multi_modality(True)
        enabled = registrar_ICON.use_multi_modality
        assert enabled, "Multi-modality not set"

        registrar_ICON.set_multi_modality(False)
        disabled = registrar_ICON.use_multi_modality
        assert not disabled, "Multi-modality update failed"

        print("\nMulti-modality setting works correctly")

    def test_register_without_mask(
        self,
        registrar_ICON: RegisterImagesICON,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test basic ICON registration without masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ICON"
        reg_output_dir.mkdir(exist_ok=True)

        # Set up registration
        fixed_image = test_images[0]
        moving_image = test_images[1]

        print("\nRegistering images with ICON (no mask)...")
        print(f"  Fixed image: {itk.size(fixed_image)}")
        print(f"  Moving image: {itk.size(moving_image)}")

        registrar_ICON.set_modality("ct")
        registrar_ICON.set_fixed_image(fixed_image)
        registrar_ICON.set_number_of_iterations(2)  # Use fewer iterations for testing

        # Register
        result = registrar_ICON.register(moving_image=moving_image)

        # Verify result is a dictionary
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "inverse_transform" in result, "Missing inverse_transform in result"
        assert "forward_transform" in result, "Missing forward_transform in result"

        inverse_transform = result["inverse_transform"]
        forward_transform = result["forward_transform"]

        # Verify transforms are valid
        assert inverse_transform is not None, "inverse_transform is None"
        assert forward_transform is not None, "forward_transform is None"

        print("ICON registration complete without mask")
        print(f"  inverse_transform type: {type(inverse_transform).__name__}")
        print(f"  forward_transform type: {type(forward_transform).__name__}")

        # Save transforms
        itk.transformwrite(
            [inverse_transform],
            str(reg_output_dir / "icon_inverse_transform_no_mask.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [forward_transform],
            str(reg_output_dir / "icon_forward_transform_no_mask.hdf"),
            compression=True,
        )
        print(f"  Saved transforms to: {reg_output_dir}")

    def test_register_with_mask(
        self,
        registrar_ICON: RegisterImagesICON,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test ICON registration with binary masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ICON"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        # Create simple binary masks (central region)
        fixed_size_itk = itk.size(fixed_image)
        moving_size_itk = itk.size(moving_image)

        # Convert ITK Size to tuple for numpy indexing
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

        # Create mask arrays
        fixed_mask_arr = np.zeros(fixed_size[::-1], dtype=np.uint8)
        moving_mask_arr = np.zeros(moving_size[::-1], dtype=np.uint8)

        # Set central region to 1
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

        # Create ITK images
        fixed_mask = itk.image_from_array(fixed_mask_arr)
        fixed_mask.CopyInformation(fixed_image)

        moving_mask = itk.image_from_array(moving_mask_arr)
        moving_mask.CopyInformation(moving_image)

        print("\nRegistering images with ICON (with masks)...")
        print(f"  Fixed mask voxels: {np.sum(fixed_mask_arr)}")
        print(f"  Moving mask voxels: {np.sum(moving_mask_arr)}")

        # Set up registration with masks
        registrar_ICON.set_modality("ct")
        registrar_ICON.set_fixed_image(fixed_image)
        registrar_ICON.set_fixed_mask(fixed_mask)
        registrar_ICON.set_number_of_iterations(2)

        # Register
        result = registrar_ICON.register(
            moving_image=moving_image, moving_mask=moving_mask
        )

        # Verify result
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "inverse_transform" in result, "Missing inverse_transform in result"
        assert "forward_transform" in result, "Missing forward_transform in result"

        inverse_transform = result["inverse_transform"]
        forward_transform = result["forward_transform"]

        assert inverse_transform is not None, "inverse_transform is None"
        assert forward_transform is not None, "forward_transform is None"

        print("ICON registration complete with masks")

        # Save transforms
        itk.transformwrite(
            [inverse_transform],
            str(reg_output_dir / "icon_inverse_transform_with_mask.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [forward_transform],
            str(reg_output_dir / "icon_forward_transform_with_mask.hdf"),
            compression=True,
        )

    def test_transform_application(
        self,
        registrar_ICON: RegisterImagesICON,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test applying ICON registration transforms to images."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ICON"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        # Register
        registrar_ICON.set_modality("ct")
        registrar_ICON.set_fixed_image(fixed_image)
        registrar_ICON.set_number_of_iterations(2)
        result = registrar_ICON.register(moving_image=moving_image)

        forward_transform = result["forward_transform"]

        print("\nApplying ICON transform to moving image...")

        # Apply transform
        transform_tools = TransformTools()
        registered_image = transform_tools.transform_image(
            moving_image, forward_transform, fixed_image, interpolation_method="linear"
        )

        # Verify registered image
        assert registered_image is not None, "Registered image is None"
        assert itk.size(registered_image) == itk.size(fixed_image), "Size mismatch"

        # Check that image content changed
        moving_arr = itk.array_from_image(moving_image)
        registered_arr = itk.array_from_image(registered_image)

        difference = np.sum(
            np.abs(moving_arr.astype(float) - registered_arr.astype(float))
        )

        print("Transform applied successfully")
        print(f"  Registered image size: {itk.size(registered_image)}")
        print(f"  Total difference: {difference:.2f}")

        # Save registered image
        itk.imwrite(
            registered_image,
            str(reg_output_dir / "icon_registered_image.mha"),
            compression=True,
        )
        print(f"  Saved to: {reg_output_dir / 'icon_registered_image.mha'}")

    def test_inverse_consistency(
        self, registrar_ICON: RegisterImagesICON, test_images: list[Any]
    ) -> None:
        """Test ICON's inverse consistency property."""
        fixed_image = test_images[0]
        moving_image = test_images[1]

        print("\nTesting inverse consistency...")

        registrar_ICON.set_modality("ct")
        registrar_ICON.set_fixed_image(fixed_image)
        registrar_ICON.set_number_of_iterations(2)
        result = registrar_ICON.register(moving_image=moving_image)

        inverse_transform = cast(itk.Transform, result["inverse_transform"])
        forward_transform = cast(itk.Transform, result["forward_transform"])

        # Test point transformation
        test_point = itk.Point[itk.D, 3]()
        test_point[0] = float(itk.size(fixed_image)[0] / 2)
        test_point[1] = float(itk.size(fixed_image)[1] / 2)
        test_point[2] = float(itk.size(fixed_image)[2] / 2)

        # Forward then backward
        transformed_point = forward_transform.TransformPoint(test_point)
        back_transformed_point = inverse_transform.TransformPoint(transformed_point)

        # Calculate error
        error = np.sqrt(
            (test_point[0] - back_transformed_point[0]) ** 2
            + (test_point[1] - back_transformed_point[1]) ** 2
            + (test_point[2] - back_transformed_point[2]) ** 2
        )

        print("Inverse consistency tested")
        print(
            f"  Original point: [{test_point[0]:.2f}, {test_point[1]:.2f}, {test_point[2]:.2f}]"
        )
        print(
            f"  Round-trip point: [{back_transformed_point[0]:.2f}, {back_transformed_point[1]:.2f}, {back_transformed_point[2]:.2f}]"
        )
        print(f"  Round-trip error: {error:.4f} mm")

        # ICON should have small inverse consistency error
        assert error < 5.0, f"Inverse consistency error too large: {error:.2f} mm"

    def test_preprocess_images(
        self, registrar_ICON: RegisterImagesICON, test_images: list[Any]
    ) -> None:
        """Test image preprocessing for ICON."""
        test_image = test_images[0]

        print("\nTesting ICON image preprocessing...")
        print(f"  Original spacing: {itk.spacing(test_image)}")

        # Preprocess
        preprocessed = registrar_ICON.preprocess(test_image, modality="ct")

        assert preprocessed is not None, "Preprocessed image is None"

        preprocessed_spacing = itk.spacing(preprocessed)
        print("Image preprocessing complete")
        print(f"  Preprocessed spacing: {preprocessed_spacing}")

    def test_registration_with_initial_transform(
        self,
        registrar_ICON: RegisterImagesICON,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test ICON registration with initial transform."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ICON"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        # Create initial translation transform
        initial_tfm_forward = itk.TranslationTransform[itk.D, 3].New()
        initial_tfm_forward.SetOffset([-5.0, -5.0, -5.0])

        print("\nRegistering with initial transform...")
        print("  Initial offset: [-5.0, -5.0, -5.0]")

        registrar_ICON.set_modality("ct")
        registrar_ICON.set_fixed_image(fixed_image)
        registrar_ICON.set_number_of_iterations(2)

        result = registrar_ICON.register_from(
            initial_tfm_forward,
            moving_image,
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["inverse_transform"] is not None, "inverse_transform is None"
        assert result["forward_transform"] is not None, "forward_transform is None"

        print("Registration with initial transform complete")

    def test_transform_types(
        self, registrar_ICON: RegisterImagesICON, test_images: list[Any]
    ) -> None:
        """Test that ICON transforms are correct ITK types."""
        fixed_image = test_images[0]
        moving_image = test_images[1]

        registrar_ICON.set_modality("ct")
        registrar_ICON.set_fixed_image(fixed_image)
        registrar_ICON.set_number_of_iterations(2)
        result = registrar_ICON.register(moving_image=moving_image)

        inverse_transform = result["inverse_transform"]
        forward_transform = result["forward_transform"]

        print("\nVerifying ICON transform types...")

        # ICON returns transforms (either DisplacementFieldTransform or CompositeTransform wrapping it)
        # The important thing is that they are valid ITK transforms
        assert inverse_transform is not None, "inverse_transform is None"
        assert forward_transform is not None, "forward_transform is None"

        # Check if it's either a DisplacementFieldTransform or CompositeTransform
        valid_inverse = isinstance(
            inverse_transform, (itk.DisplacementFieldTransform, itk.CompositeTransform)
        )
        valid_forward = isinstance(
            forward_transform, (itk.DisplacementFieldTransform, itk.CompositeTransform)
        )

        assert valid_inverse, (
            f"inverse_transform should be DisplacementFieldTransform or CompositeTransform, got {type(inverse_transform)}"
        )
        assert valid_forward, (
            f"forward_transform should be DisplacementFieldTransform or CompositeTransform, got {type(forward_transform)}"
        )

        print("Transform types verified")
        print(f"  inverse_transform: {type(inverse_transform).__name__}")
        print(f"  forward_transform: {type(forward_transform).__name__}")

    def test_different_iteration_counts(
        self, registrar_ICON: RegisterImagesICON, test_images: list[Any]
    ) -> None:
        """Test ICON with different iteration counts."""
        fixed_image = test_images[0]
        moving_image = test_images[1]

        registrar_ICON.set_modality("ct")
        registrar_ICON.set_fixed_image(fixed_image)

        iteration_counts = [1, 2, 5]
        results = []

        print("\nTesting different iteration counts...")

        for num_iter in iteration_counts:
            print(f"  Running with {num_iter} iterations...")
            registrar_ICON.set_number_of_iterations(num_iter)
            result = registrar_ICON.register(moving_image=moving_image)
            results.append(result)

            assert isinstance(result, dict), "Result should be a dictionary"
            assert "inverse_transform" in result, "Missing inverse_transform"
            assert "forward_transform" in result, "Missing forward_transform"

        print(f"Tested {len(iteration_counts)} different iteration counts")


def test_set_weights_path_rejects_missing_file(tmp_path: Path) -> None:
    """A missing checkpoint must raise instead of downloading stock weights.

    uniGradICON treats an unknown ``weights_location`` as a download
    destination, so an unchecked path silently yields a stock-weights
    registration that looks like a finetuned one.
    """
    registrar = RegisterImagesICON()
    with pytest.raises(FileNotFoundError, match="weights not found"):
        registrar.set_weights_path(str(tmp_path / "network_weights_final.trch"))
    assert registrar.weights_path is None

    existing = tmp_path / "network_weights_final.trch"
    existing.touch()
    registrar.set_weights_path(str(existing))
    assert registrar.weights_path == str(existing)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
