"""
Test for ANTs-based image registration.

This test depends on test_convert_image_4d_to_3d and uses the converted
3D CT images to test ANTs registration functionality.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import ants
import itk
import numpy as np
import pytest

from monai_physio.register_images_ants import RegisterImagesANTS
from monai_physio.transform_tools import TransformTools

from .conftest import KnownShiftCase


def _foreground_ncc(
    reference_arr: np.ndarray, warped_arr: np.ndarray, foreground: np.ndarray
) -> float:
    """Normalized cross-correlation over a foreground mask (higher = better).

    Args:
        reference_arr: Reference image array (e.g. the fixed image), axes (Z, Y, X).
        warped_arr: Warped image array on the same grid/axes as ``reference_arr``.
        foreground: Boolean mask (same shape) selecting the voxels to score.

    Returns:
        NCC in [-1, 1] over the foreground voxels (nan if degenerate).
    """
    a = reference_arr[foreground].astype(np.float64)
    b = warped_arr[foreground].astype(np.float64)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.sqrt((a0 * a0).sum() * (b0 * b0).sum()))
    return float((a0 * b0).sum() / denom) if denom > 0 else float("nan")


@pytest.mark.slow
class TestRegisterImagesANTS:
    """Test suite for ANTs-based image registration."""

    def test_registrar_initialization(self, registrar_ANTS: RegisterImagesANTS) -> None:
        """Test that RegisterImagesANTS initializes correctly."""
        assert registrar_ANTS is not None, "Registrar not initialized"
        assert hasattr(registrar_ANTS, "fixed_image"), "Missing fixed_image attribute"
        assert hasattr(registrar_ANTS, "fixed_mask"), "Missing fixed_mask attribute"

        print("\nANTS registrar initialized successfully")

    def test_set_modality(self, registrar_ANTS: RegisterImagesANTS) -> None:
        """Test setting imaging modality."""
        registrar_ANTS.set_modality("ct")
        assert registrar_ANTS.modality == "ct", "Modality not set correctly"

        registrar_ANTS.set_modality("mr")
        assert registrar_ANTS.modality == "mr", "Modality change failed"

        print("\nModality setting works correctly")

    def test_set_fixed_image(
        self, registrar_ANTS: RegisterImagesANTS, test_images: list[Any]
    ) -> None:
        """Test setting fixed image."""
        fixed_image = test_images[0]

        registrar_ANTS.set_fixed_image(fixed_image)
        assert registrar_ANTS.fixed_image is not None, "Fixed image not set"

        print("\nFixed image set successfully")
        print(f"  Image size: {itk.size(registrar_ANTS.fixed_image)}")
        print(f"  Image spacing: {itk.spacing(registrar_ANTS.fixed_image)}")

    def test_register_without_mask(
        self,
        registrar_ANTS: RegisterImagesANTS,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test basic registration without masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ANTS"
        reg_output_dir.mkdir(exist_ok=True)

        # Set up registration
        fixed_image = test_images[0]
        moving_image = test_images[1]

        print("\nRegistering images without mask...")
        print(f"  Fixed image: {itk.size(fixed_image)}")
        print(f"  Moving image: {itk.size(moving_image)}")

        registrar_ANTS.set_modality("ct")
        registrar_ANTS.set_fixed_image(fixed_image)

        # Register
        result = registrar_ANTS.register(moving_image=moving_image)

        # Verify result is a dictionary
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "inverse_transform" in result, "Missing inverse_transform in result"
        assert "forward_transform" in result, "Missing forward_transform in result"

        inverse_transform = result["inverse_transform"]
        forward_transform = result["forward_transform"]

        # Verify transforms are valid
        assert inverse_transform is not None, "inverse_transform is None"
        assert forward_transform is not None, "forward_transform is None"

        print("Registration complete without mask")
        print(f"  inverse_transform type: {type(inverse_transform).__name__}")
        print(f"  forward_transform type: {type(forward_transform).__name__}")

        # Save transforms
        itk.transformwrite(
            [inverse_transform],
            str(reg_output_dir / "ants_inverse_transform_no_mask.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [forward_transform],
            str(reg_output_dir / "ants_forward_transform_no_mask.hdf"),
            compression=True,
        )
        print(f"  Saved transforms to: {reg_output_dir}")

    def test_register_with_mask(
        self,
        registrar_ANTS: RegisterImagesANTS,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test registration with binary masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ANTS"
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

        print("\nRegistering images with masks...")
        print(f"  Fixed mask voxels: {np.sum(fixed_mask_arr)}")
        print(f"  Moving mask voxels: {np.sum(moving_mask_arr)}")

        # Set up registration with masks
        registrar_ANTS.set_modality("ct")
        registrar_ANTS.set_fixed_image(fixed_image)
        registrar_ANTS.set_fixed_mask(fixed_mask)

        # Register
        result = registrar_ANTS.register(
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

        print("Registration complete with masks")

        # Save transforms
        itk.transformwrite(
            [inverse_transform],
            str(reg_output_dir / "ants_inverse_transform_with_mask.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [forward_transform],
            str(reg_output_dir / "ants_forward_transform_with_mask.hdf"),
            compression=True,
        )

    def test_transform_application(
        self,
        registrar_ANTS: RegisterImagesANTS,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test applying registration transforms to images."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ANTS"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        # Register
        registrar_ANTS.set_modality("ct")
        registrar_ANTS.set_fixed_image(fixed_image)
        result = registrar_ANTS.register(moving_image=moving_image)

        forward_transform = result["forward_transform"]

        print("\nApplying transform to moving image...")

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

        # They should be different (unless perfectly aligned)
        difference = np.sum(
            np.abs(moving_arr.astype(float) - registered_arr.astype(float))
        )

        print("Transform applied successfully")
        print(f"  Registered image size: {itk.size(registered_image)}")
        print(f"  Total difference: {difference:.2f}")

        # Save registered image
        itk.imwrite(
            registered_image,
            str(reg_output_dir / "ants_registered_image.mha"),
            compression=True,
        )
        print(f"  Saved to: {reg_output_dir / 'ants_registered_image.mha'}")

    def test_preprocess_images(
        self, registrar_ANTS: RegisterImagesANTS, test_images: list[Any]
    ) -> None:
        """Test image preprocessing."""
        test_image = test_images[0]

        print("\nTesting image preprocessing...")
        print(f"  Original spacing: {itk.spacing(test_image)}")

        # Preprocess
        preprocessed = registrar_ANTS.preprocess(test_image, modality="ct")

        assert preprocessed is not None, "Preprocessed image is None"

        preprocessed_spacing = itk.spacing(preprocessed)
        print("Image preprocessing complete")
        print(f"  Preprocessed spacing: {preprocessed_spacing}")

    def test_recovers_known_shift(self, known_shift_case: KnownShiftCase) -> None:
        """ANTs must recover a known shift, in the right direction.

        The companion of the Greedy check in test_register_images_greedy.py:
        ANTs works in ITK's LPS frame throughout, so this asserts no equivalent
        RAS/LPS conversion is missing here.
        """
        registrar = RegisterImagesANTS()
        registrar.set_modality("ct")
        registrar.set_transform_type("Affine")
        registrar.set_fixed_image(known_shift_case.fixed)

        result = registrar.register(moving_image=known_shift_case.moving)
        forward_transform = result["forward_transform"]

        error_mm = known_shift_case.center_error_mm(forward_transform)
        ncc = known_shift_case.foreground_ncc(forward_transform)
        unregistered_ncc = known_shift_case.unregistered_ncc()

        print("\nANTs known-shift recovery:")
        print(f"  error: {error_mm:.2f} mm")
        print(f"  foreground NCC: {ncc:.4f} (unregistered {unregistered_ncc:.4f})")

        assert error_mm < 2.0, f"ANTs recovered the shift {error_mm:.2f} mm off"
        assert ncc > unregistered_ncc, (
            f"ANTs left the images less aligned than they started "
            f"({ncc:.4f} vs {unregistered_ncc:.4f})"
        )

    def test_registration_with_initial_transform(
        self,
        registrar_ANTS: RegisterImagesANTS,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test registration with initial transform."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ANTS"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        # Create initial translation transform
        initial_tfm_forward = itk.TranslationTransform[itk.D, 3].New()
        initial_tfm_forward.SetOffset([-5.0, -5.0, -5.0])

        print("\nRegistering with initial transform...")
        print("  Initial offset: [-5.0, -5.0, -5.0]")

        registrar_ANTS.set_modality("ct")
        registrar_ANTS.set_fixed_image(fixed_image)

        result = registrar_ANTS.register_from(
            initial_tfm_forward,
            moving_image,
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["inverse_transform"] is not None, "inverse_transform is None"
        assert result["forward_transform"] is not None, "forward_transform is None"

        print("Registration with initial transform complete")

    def test_initial_transform_composition_metrics(
        self,
        registrar_ANTS: RegisterImagesANTS,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Verify the register_from() composition path with metrics.

        Exercises the two initial-transform inputs the platform actually uses
        (identity and a prior deformable forward_transform, as in prior-based
        time-series registration) and confirms the composed forward_transform
        warps the moving image onto the fixed grid. Scored with foreground NCC
        over the brightest 30% of the fixed image (tissue/blood pool). See
        docs/developer/transform_conventions.

        Asserted facts:
          * a plain registration improves on the unregistered pair,
          * an identity initial reproduces the baseline exactly (the
            composition machinery is a structurally correct no-op; note an
            identity AffineTransform is itself a matrix initial),
          * a prior-deformable initial reaches the no-initial baseline quality
            (the composition recovers the full transform).

        The initial transform is applied by pre-warping the moving image (as in
        RegisterImagesICON), which keeps the composition self-consistent for any
        initial transform type.
        """
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ANTS"
        reg_output_dir.mkdir(exist_ok=True)

        # Pick two phases that are far apart in the cycle so there is real motion.
        fixed_image = test_images[0]
        moving_image = test_images[min(10, len(test_images) - 1)]

        fixed_arr = itk.array_from_image(fixed_image)
        # Moving and fixed share the acquisition grid (split from one 4D image),
        # so the moving array is directly comparable for the unregistered score.
        moving_arr = itk.array_from_image(moving_image)
        threshold = float(np.percentile(fixed_arr, 70.0))
        foreground = fixed_arr > threshold

        transform_tools = TransformTools()

        def warp_score(forward_transform: Any) -> float:
            warped = transform_tools.transform_image(
                moving_image,
                forward_transform,
                fixed_image,
                interpolation_method="linear",
            )
            return _foreground_ncc(fixed_arr, itk.array_from_image(warped), foreground)

        ncc_unregistered = _foreground_ncc(fixed_arr, moving_arr, foreground)

        # Baseline: no initial transform.
        registrar_ANTS.set_modality("ct")
        registrar_ANTS.set_fixed_image(fixed_image)
        baseline = registrar_ANTS.register(moving_image=moving_image)
        ncc_baseline = warp_score(baseline["forward_transform"])

        # Identity initial: the composition machinery must be a no-op.
        identity = itk.AffineTransform[itk.D, 3].New()
        identity.SetIdentity()
        registrar_identity = RegisterImagesANTS()
        registrar_identity.set_modality("ct")
        registrar_identity.set_fixed_image(fixed_image)
        identity_result = registrar_identity.register_from(identity, moving_image)
        ncc_identity = warp_score(identity_result["forward_transform"])

        # Prior deformable initial: the realistic time-series prior use case.
        registrar_prior = RegisterImagesANTS()
        registrar_prior.set_modality("ct")
        registrar_prior.set_fixed_image(fixed_image)
        prior_result = registrar_prior.register_from(
            baseline["forward_transform"],
            moving_image,
        )
        ncc_prior = warp_score(prior_result["forward_transform"])

        print("\nANTS initial-transform composition metrics (foreground NCC):")
        print(f"  unregistered:          {ncc_unregistered:.4f}")
        print(f"  baseline (no initial): {ncc_baseline:.4f}")
        print(f"  identity initial:      {ncc_identity:.4f}")
        print(f"  prior-deformable init: {ncc_prior:.4f}")

        warped_prior = transform_tools.transform_image(
            moving_image,
            prior_result["forward_transform"],
            fixed_image,
            interpolation_method="linear",
        )
        itk.imwrite(
            warped_prior,
            str(reg_output_dir / "ants_warped_prior_initial.mha"),
            compression=True,
        )

        # Registration must improve alignment over the unregistered pair.
        assert ncc_baseline > ncc_unregistered, (
            f"Baseline registration did not improve alignment: "
            f"{ncc_baseline:.4f} <= {ncc_unregistered:.4f}"
        )
        # Identity initial must reproduce the baseline (composition is a no-op).
        assert abs(ncc_identity - ncc_baseline) < 0.03, (
            f"Identity initial transform changed the result: "
            f"identity={ncc_identity:.4f} vs baseline={ncc_baseline:.4f}"
        )
        # A prior-deformable initial must reach the no-initial baseline quality
        # (the composition recovers the full transform).
        assert ncc_prior >= ncc_baseline - 0.03, (
            f"Prior-initial composition did not reach baseline quality: "
            f"{ncc_prior:.4f} < {ncc_baseline:.4f} - 0.03"
        )

    def test_initial_transform_matrix_composition(
        self,
        registrar_ANTS: RegisterImagesANTS,
        test_images: list[Any],
    ) -> None:
        """A matrix (translation/affine) initial composes without corruption.

        Regression guard for the previously-broken matrix initial_transform
        path: feeding a translation initial used to corrupt the composition
        (foreground NCC far below the unregistered pair). With the moving image
        pre-warped by the initial, the composed forward_transform must align the
        moving image onto the fixed grid at least as well as the unregistered
        pair.
        """
        fixed_image = test_images[0]
        moving_image = test_images[min(10, len(test_images) - 1)]

        fixed_arr = itk.array_from_image(fixed_image)
        threshold = float(np.percentile(fixed_arr, 70.0))
        foreground = fixed_arr > threshold
        ncc_unregistered = _foreground_ncc(
            fixed_arr, itk.array_from_image(moving_image), foreground
        )

        translation = itk.TranslationTransform[itk.D, 3].New()
        translation.SetOffset([-5.0, -5.0, -5.0])

        registrar_ANTS.set_modality("ct")
        registrar_ANTS.set_fixed_image(fixed_image)
        result = registrar_ANTS.register_from(translation, moving_image)

        transform_tools = TransformTools()
        warped = transform_tools.transform_image(
            moving_image,
            result["forward_transform"],
            fixed_image,
            interpolation_method="linear",
        )
        ncc = _foreground_ncc(fixed_arr, itk.array_from_image(warped), foreground)
        print(
            f"\nMatrix-initial composed NCC={ncc:.4f} (unregistered={ncc_unregistered:.4f})"
        )
        assert ncc > ncc_unregistered, (
            f"Matrix-initial composition worse than unregistered: "
            f"{ncc:.4f} <= {ncc_unregistered:.4f}"
        )

    def test_affine_and_rigid_transform_types(
        self,
        registrar_ANTS: RegisterImagesANTS,
        test_images: list[Any],
    ) -> None:
        """Affine and Rigid transform types run and improve alignment.

        Regression guard for the ANTS preset names: ``set_transform_type``
        previously mapped Affine/Rigid to ``antsRegistration{Affine,Rigid}Quick``
        preset strings that do not exist in antspy, raising ValueError. Each
        type must now run and warp the moving image onto the fixed grid at least
        as well as the unregistered pair.
        """
        fixed_image = test_images[0]
        moving_image = test_images[min(10, len(test_images) - 1)]

        fixed_arr = itk.array_from_image(fixed_image)
        threshold = float(np.percentile(fixed_arr, 70.0))
        foreground = fixed_arr > threshold
        ncc_unregistered = _foreground_ncc(
            fixed_arr, itk.array_from_image(moving_image), foreground
        )

        transform_tools = TransformTools()
        for transform_type in ("Rigid", "Affine"):
            registrar = RegisterImagesANTS()
            registrar.set_modality("ct")
            registrar.set_transform_type(transform_type)
            registrar.set_fixed_image(fixed_image)
            result = registrar.register(moving_image=moving_image)
            warped = transform_tools.transform_image(
                moving_image,
                result["forward_transform"],
                fixed_image,
                interpolation_method="linear",
            )
            ncc = _foreground_ncc(fixed_arr, itk.array_from_image(warped), foreground)
            print(
                f"\n{transform_type} transform NCC={ncc:.4f} "
                f"(unregistered={ncc_unregistered:.4f})"
            )
            assert ncc > ncc_unregistered, (
                f"{transform_type} registration did not improve alignment: "
                f"{ncc:.4f} <= {ncc_unregistered:.4f}"
            )

    def test_multiple_registrations(
        self, registrar_ANTS: RegisterImagesANTS, test_images: list[Any]
    ) -> None:
        """Test running multiple registrations in sequence."""
        fixed_image = test_images[0]
        moving_image = test_images[1]

        print("\nRunning multiple registrations...")

        registrar_ANTS.set_modality("ct")
        registrar_ANTS.set_fixed_image(fixed_image)

        results = []
        for i in range(2):
            print(f"  Registration {i + 1}...")
            result = registrar_ANTS.register(moving_image=moving_image)
            results.append(result)

            assert isinstance(result, dict), f"Result {i + 1} should be a dictionary"
            assert "inverse_transform" in result, (
                f"Missing inverse_transform in result {i + 1}"
            )
            assert "forward_transform" in result, (
                f"Missing forward_transform in result {i + 1}"
            )

        print(f"Multiple registrations complete: {len(results)} runs")

    def test_transform_types(
        self, registrar_ANTS: RegisterImagesANTS, test_images: list[Any]
    ) -> None:
        """Test that transforms are correct ITK types."""
        fixed_image = test_images[0]
        moving_image = test_images[1]

        registrar_ANTS.set_modality("ct")
        registrar_ANTS.set_fixed_image(fixed_image)
        result = registrar_ANTS.register(moving_image=moving_image)

        inverse_transform = result["inverse_transform"]
        forward_transform = result["forward_transform"]

        print("\nVerifying transform types...")

        # Check that transforms are CompositeTransform (ANTs returns composite)
        assert isinstance(inverse_transform, itk.CompositeTransform), (
            f"inverse_transform should be CompositeTransform, got {type(inverse_transform)}"
        )
        assert isinstance(forward_transform, itk.CompositeTransform), (
            f"forward_transform should be CompositeTransform, got {type(forward_transform)}"
        )

        print("Transform types verified")
        print(f"  inverse_transform: {type(inverse_transform).__name__}")
        print(f"  forward_transform: {type(forward_transform).__name__}")

    def test_image_conversion_cycle_scalar(
        self, registrar_ANTS: RegisterImagesANTS, test_images: list[Any]
    ) -> None:
        """Test round-trip conversion: ITK image -> ANTs -> ITK for scalar images."""
        original_image = test_images[0]

        print("\nTesting scalar image conversion cycle...")
        print(f"  Original image size: {itk.size(original_image)}")
        print(f"  Original image spacing: {itk.spacing(original_image)}")
        print(f"  Original image origin: {itk.origin(original_image)}")

        # Get original data
        original_array = itk.array_from_image(original_image)
        original_spacing = tuple(itk.spacing(original_image))
        original_origin = tuple(itk.origin(original_image))
        original_direction = itk.array_from_matrix(original_image.GetDirection())

        # Convert ITK -> ANTs
        ants_image = registrar_ANTS._itk_to_ants_image(original_image, dtype="float")

        # Verify ANTs image
        assert ants_image is not None, "ANTs image is None"
        assert ants_image.dimension == 3, (
            f"ANTs dimension should be 3, got {ants_image.dimension}"
        )
        print(f"  ANTs image shape: {ants_image.shape}")

        # Convert ANTs -> ITK
        recovered_image = registrar_ANTS._ants_to_itk_image(ants_image)

        # Verify recovered image
        assert recovered_image is not None, "Recovered image is None"
        recovered_array = itk.array_from_image(recovered_image)
        recovered_spacing = tuple(itk.spacing(recovered_image))
        recovered_origin = tuple(itk.origin(recovered_image))
        recovered_direction = itk.array_from_matrix(recovered_image.GetDirection())

        print(f"  Recovered image size: {itk.size(recovered_image)}")
        print(f"  Recovered image spacing: {recovered_spacing}")
        print(f"  Recovered image origin: {recovered_origin}")

        # Compare arrays
        assert original_array.shape == recovered_array.shape, (
            f"Array shape mismatch: {original_array.shape} vs {recovered_array.shape}"
        )

        # Check pixel values (allowing for float conversion tolerance)
        max_diff = np.max(
            np.abs(
                original_array.astype(np.float32) - recovered_array.astype(np.float32)
            )
        )
        print(f"  Max pixel difference: {max_diff}")
        assert max_diff < 1e-5, f"Pixel values differ too much: {max_diff}"

        # Check spacing (with tolerance for floating point)
        spacing_diff = np.max(
            np.abs(np.array(original_spacing) - np.array(recovered_spacing))
        )
        print(f"  Max spacing difference: {spacing_diff}")
        assert spacing_diff < 1e-6, f"Spacing differs: {spacing_diff}"

        # Check origin
        origin_diff = np.max(
            np.abs(np.array(original_origin) - np.array(recovered_origin))
        )
        print(f"  Max origin difference: {origin_diff}")
        assert origin_diff < 1e-6, f"Origin differs: {origin_diff}"

        # Check direction
        direction_diff = np.max(np.abs(original_direction - recovered_direction))
        print(f"  Max direction difference: {direction_diff}")
        assert direction_diff < 1e-6, f"Direction differs: {direction_diff}"

        print("Scalar image conversion cycle successful")

    def test_image_conversion_cycle_different_dtypes(
        self, registrar_ANTS: RegisterImagesANTS, test_images: list[Any]
    ) -> None:
        """Test round-trip conversion with different data types."""
        original_image = test_images[0]

        print("\nTesting image conversion cycle with different dtypes...")

        dtypes = ["float", "double", "int", "uint", "uchar"]

        for dtype in dtypes:
            print(f"  Testing dtype: {dtype}")

            # Convert ITK -> ANTs with specified dtype
            ants_image = registrar_ANTS._itk_to_ants_image(original_image, dtype=dtype)
            assert ants_image is not None, f"ANTs image is None for dtype {dtype}"

            # Convert ANTs -> ITK
            recovered_image = registrar_ANTS._ants_to_itk_image(ants_image)
            assert recovered_image is not None, (
                f"Recovered image is None for dtype {dtype}"
            )

            # Verify dimensions match
            assert itk.size(original_image) == itk.size(recovered_image), (
                f"Size mismatch for dtype {dtype}"
            )

            print(f"    {dtype} conversion successful")

        print("All dtype conversions successful")

    def test_image_conversion_preserves_metadata(
        self, registrar_ANTS: RegisterImagesANTS
    ) -> None:
        """Test that image conversion preserves all metadata."""
        print("\nTesting metadata preservation in image conversion...")

        # Create a test image with specific metadata
        size = [50, 60, 40]
        spacing = [1.5, 2.0, 2.5]
        origin = [10.0, 20.0, 30.0]

        ImageType = itk.Image[itk.F, 3]
        test_image = ImageType.New()
        region = itk.ImageRegion[3]()
        region.SetSize(size)
        test_image.SetRegions(region)
        test_image.SetSpacing(spacing)
        test_image.SetOrigin(origin)
        test_image.Allocate()

        # Fill with test pattern
        arr = itk.array_from_image(test_image)
        arr[:] = np.random.randn(*arr.shape).astype(np.float32)

        print(f"  Test image size: {size}")
        print(f"  Test image spacing: {spacing}")
        print(f"  Test image origin: {origin}")

        # Convert ITK -> ANTs -> ITK
        ants_image = registrar_ANTS._itk_to_ants_image(test_image)
        recovered_image = registrar_ANTS._ants_to_itk_image(ants_image)

        # Verify all metadata
        recovered_size = [int(s) for s in itk.size(recovered_image)]
        recovered_spacing = [float(s) for s in itk.spacing(recovered_image)]
        recovered_origin = [float(o) for o in itk.origin(recovered_image)]

        assert recovered_size == size, f"Size mismatch: {recovered_size} vs {size}"
        assert np.allclose(recovered_spacing, spacing, atol=1e-6), (
            f"Spacing mismatch: {recovered_spacing} vs {spacing}"
        )
        assert np.allclose(recovered_origin, origin, atol=1e-6), (
            f"Origin mismatch: {recovered_origin} vs {origin}"
        )

        print("Metadata preservation verified")

    def test_transform_conversion_cycle_affine(
        self, registrar_ANTS: RegisterImagesANTS, test_images: list[Any]
    ) -> None:
        """Test round-trip conversion: ITK affine transform -> ANTs -> ITK."""
        reference_image = test_images[0]

        print("\nTesting affine transform conversion cycle...")

        # Create an ITK affine transform with known parameters
        affine_tfm = itk.AffineTransform[itk.D, 3].New()

        # Set center
        center = itk.Point[itk.D, 3]()
        center[0] = 100.0
        center[1] = 100.0
        center[2] = 50.0
        affine_tfm.SetCenter(center)

        # Set rotation (small rotation around Z axis)
        angle = np.pi / 12  # 15 degrees
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]
        )
        affine_tfm.SetMatrix(itk.GetMatrixFromArray(rotation_matrix))

        # Set translation
        translation = itk.Vector[itk.D, 3]()
        translation[0] = 5.0
        translation[1] = 10.0
        translation[2] = -3.0
        affine_tfm.SetTranslation(translation)

        print(f"  Original center: {[center[i] for i in range(3)]}")
        print(f"  Original translation: {[translation[i] for i in range(3)]}")

        # Convert ITK -> ANTs file
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_tfm_file = os.path.join(tmpdir, "temp_transform.mat")
            transform_files = registrar_ANTS.itk_transform_to_ANTSfile(
                affine_tfm, reference_image, temp_tfm_file
            )
            assert len(transform_files) == 1, "Should return one transform file"
            # Note: The returned filename may have extension added/modified
            assert os.path.exists(transform_files[0]), (
                f"Transform file not found: {transform_files[0]}"
            )
            print(f"  ANTs transform written to: {transform_files[0]}")

            # Read the transform to verify (use the actual written file)
            ants_tfm = ants.read_transform(transform_files[0])
            print(f"  ANTs transform type: {ants_tfm.transform_type}")

            # Convert back ANTs -> ITK
            # Affine transforms are stored as affine in ANTs, so read back as affine
            if ants_tfm.transform_type == "AffineTransform":
                recovered_tfm = registrar_ANTS._antsfile_to_itk_affine_transform(
                    transform_files[0]
                )
            else:
                # For displacement field transforms
                recovered_tfm = (
                    registrar_ANTS._antsfile_to_itk_displacement_field_transform(
                        transform_files[0], reference_image
                    )
                )

        assert recovered_tfm is not None, "Recovered transform is None"
        print(f"  Recovered transform type: {type(recovered_tfm).__name__}")

        # Test that transforms produce similar results on test points
        test_points = [
            [100.0, 100.0, 50.0],
            [120.0, 110.0, 60.0],
            [80.0, 90.0, 40.0],
        ]

        print("  Testing transform on sample points...")
        max_point_diff = 0.0

        for pt_coords in test_points:
            # Transform with original
            pt_itk = itk.Point[itk.D, 3]()
            for i in range(3):
                pt_itk[i] = pt_coords[i]

            pt_transformed_original = affine_tfm.TransformPoint(pt_itk)
            pt_transformed_recovered = recovered_tfm.TransformPoint(pt_itk)

            # Calculate difference
            diff = np.sqrt(
                sum(
                    (pt_transformed_original[i] - pt_transformed_recovered[i]) ** 2
                    for i in range(3)
                )
            )
            max_point_diff = max(max_point_diff, diff)

        print(f"  Max point transformation difference: {max_point_diff:.6f}")

        # Allow some tolerance due to displacement field discretization
        assert max_point_diff < 0.5, f"Transform difference too large: {max_point_diff}"

        print("Affine transform conversion cycle successful")

    def test_transform_conversion_cycle_displacement_field(
        self, registrar_ANTS: RegisterImagesANTS, test_images: list[Any]
    ) -> None:
        """Test round-trip conversion: ITK displacement field -> ANTs -> ITK."""
        reference_image = test_images[0]

        print("\nTesting displacement field transform conversion cycle...")

        # Create a simple displacement field with double precision
        # Use ImageTools to create the correct type
        from monai_physio.image_tools import ImageTools

        image_tools = ImageTools()

        # Create displacement array (small random displacements)
        ref_size = itk.size(reference_image)
        disp_array = (
            np.random.randn(
                int(ref_size[2]), int(ref_size[1]), int(ref_size[0]), 3
            ).astype(np.float64)
            * 0.5
        )

        # Convert to ITK image with correct vector type
        disp_field = image_tools.convert_array_to_image_of_vectors(
            disp_array, reference_image, itk.D
        )

        # Create displacement field transform
        disp_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
        disp_tfm.SetDisplacementField(disp_field)

        print(f"  Original displacement field size: {itk.size(disp_field)}")
        print(
            f"  Max displacement magnitude: {np.max(np.linalg.norm(disp_array, axis=-1)):.3f}"
        )

        # Convert ITK -> ANTs file
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_tfm_file = os.path.join(tmpdir, "temp_disp_transform.mat")
            transform_files = registrar_ANTS.itk_transform_to_ANTSfile(
                disp_tfm, reference_image, temp_tfm_file
            )
            assert len(transform_files) == 1, "Should return one transform file"
            # Note: The returned filename may have extension added/modified
            assert os.path.exists(transform_files[0]), (
                f"Transform file not found: {transform_files[0]}"
            )
            print(f"  ANTs transform written successfully to: {transform_files[0]}")

            # Convert back ANTs -> ITK

            recovered_tfm = (
                registrar_ANTS._antsfile_to_itk_displacement_field_transform(
                    transform_files[0], reference_image
                )
            )

        assert recovered_tfm is not None, "Recovered transform is None"
        print(f"  Recovered transform type: {type(recovered_tfm).__name__}")

        # Get recovered displacement field
        recovered_disp_field = recovered_tfm.GetDisplacementField()
        recovered_disp_array = itk.array_from_image(recovered_disp_field)

        print(f"  Recovered displacement field size: {itk.size(recovered_disp_field)}")
        print(
            f"  Recovered max displacement: {np.max(np.linalg.norm(recovered_disp_array, axis=-1)):.3f}"
        )

        # Compare displacement fields
        diff_array = disp_array - recovered_disp_array
        max_diff = np.max(np.linalg.norm(diff_array, axis=-1))
        mean_diff = np.mean(np.linalg.norm(diff_array, axis=-1))

        print(f"  Max displacement difference: {max_diff:.6f}")
        print(f"  Mean displacement difference: {mean_diff:.6f}")

        # Allow tolerance for conversion artifacts
        assert max_diff < 1.0, f"Displacement field difference too large: {max_diff}"
        assert mean_diff < 0.1, f"Mean displacement difference too large: {mean_diff}"

        print("Displacement field transform conversion cycle successful")

    def test_transform_conversion_with_composite(
        self, registrar_ANTS: RegisterImagesANTS, test_images: list[Any]
    ) -> None:
        """Test conversion of composite transforms."""
        reference_image = test_images[0]

        print("\nTesting composite transform conversion...")

        # Create a composite transform with translation + affine
        composite_tfm = itk.CompositeTransform[itk.D, 3].New()

        # Add translation
        translation_tfm = itk.TranslationTransform[itk.D, 3].New()
        translation_tfm.SetOffset([5.0, -3.0, 2.0])
        composite_tfm.AddTransform(translation_tfm)

        # Add affine (rotation)
        affine_tfm = itk.AffineTransform[itk.D, 3].New()
        center = itk.Point[itk.D, 3]()
        center.Fill(100.0)
        affine_tfm.SetCenter(center)

        angle = np.pi / 18  # 10 degrees
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        affine_tfm.SetMatrix(itk.GetMatrixFromArray(rotation_matrix))
        composite_tfm.AddTransform(affine_tfm)

        print(
            f"  Composite transform with {composite_tfm.GetNumberOfTransforms()} transforms"
        )

        # Convert to ANTs file
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_tfm_file = os.path.join(tmpdir, "temp_composite_transform.mat")
            transform_files = registrar_ANTS.itk_transform_to_ANTSfile(
                composite_tfm, reference_image, temp_tfm_file
            )
            assert len(transform_files) == 1, "Should return one transform file"
            # Note: The returned filename may have extension added/modified
            assert os.path.exists(transform_files[0]), (
                f"Transform file not found: {transform_files[0]}"
            )
            print(f"  ANTs transform written to: {transform_files[0]}")

        # Test on sample points
        test_points = [
            [100.0, 100.0, 50.0],
            [150.0, 120.0, 60.0],
        ]

        print("  Testing composite transform on sample points...")

        for pt_coords in test_points:
            pt_itk = itk.Point[itk.D, 3]()
            for i in range(3):
                pt_itk[i] = pt_coords[i]

            pt_transformed = composite_tfm.TransformPoint(pt_itk)
            print(f"    Point {pt_coords} -> {[pt_transformed[i] for i in range(3)]}")

        print("Composite transform conversion successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
