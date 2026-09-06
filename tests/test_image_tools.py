"""
Tests for ImageTools functionality.

Tests conversion between ITK and SimpleITK image formats for both
scalar and vector images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import itk
import numpy as np
import pytest
import SimpleITK as sitk

from monai_physio.image_tools import ImageTools


class TestImageTools:
    """Test suite for ImageTools conversions."""

    @pytest.fixture
    def image_tools(self) -> ImageTools:
        """Create ImageTools instance."""
        return ImageTools()

    def test_itk_to_sitk_scalar_image(self, image_tools: ImageTools) -> None:
        """Test conversion of scalar ITK image to SimpleITK."""
        # Create a simple 3D scalar ITK image
        size = [10, 20, 30]
        spacing = [1.0, 2.0, 3.0]
        origin = [0.0, 0.0, 0.0]

        # Create ITK image with known values
        ImageType = itk.Image[itk.F, 3]
        itk_image = ImageType.New()

        region = itk.ImageRegion[3]()
        region.SetSize(size)
        itk_image.SetRegions(region)
        itk_image.SetSpacing(spacing)
        itk_image.SetOrigin(origin)
        itk_image.Allocate()

        # Fill with test pattern
        itk_image.FillBuffer(42.0)

        # Convert to SimpleITK
        sitk_image = image_tools.convert_itk_image_to_sitk(itk_image)

        # Verify metadata
        assert sitk_image.GetSize() == tuple(size)
        assert sitk_image.GetSpacing() == tuple(spacing)
        assert sitk_image.GetOrigin() == tuple(origin)

        # Verify data
        array_sitk = sitk.GetArrayFromImage(sitk_image)
        assert np.allclose(array_sitk, 42.0)

        print("ITK to SimpleITK scalar conversion successful")

    def test_sitk_to_itk_scalar_image(self, image_tools: ImageTools) -> None:
        """Test conversion of scalar SimpleITK image to ITK."""
        # Create a simple 3D scalar SimpleITK image
        size = [10, 20, 30]
        spacing = [1.0, 2.0, 3.0]
        origin = [0.0, 0.0, 0.0]

        # Create SimpleITK image
        sitk_image = sitk.Image(size, sitk.sitkFloat32)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)

        # Fill with test pattern
        array = np.ones((size[2], size[1], size[0]), dtype=np.float32) * 99.0
        sitk_image = sitk.GetImageFromArray(array)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)

        # Convert to ITK
        itk_image = image_tools.convert_sitk_image_to_itk(sitk_image)

        # Verify metadata
        assert itk.size(itk_image) == tuple(size)
        assert itk.spacing(itk_image) == tuple(spacing)
        assert itk.origin(itk_image) == tuple(origin)

        # Verify data
        array_itk = itk.array_from_image(itk_image)
        assert np.allclose(array_itk, 99.0)

        print("SimpleITK to ITK scalar conversion successful")

    def test_roundtrip_scalar_image(self, image_tools: ImageTools) -> None:
        """Test roundtrip conversion: ITK -> SimpleITK -> ITK."""
        # Create ITK image
        size = [15, 25, 35]
        spacing = [0.5, 1.5, 2.5]
        origin = [10.0, 20.0, 30.0]

        ImageType = itk.Image[itk.F, 3]
        itk_image_original = ImageType.New()

        region = itk.ImageRegion[3]()
        region.SetSize(size)
        itk_image_original.SetRegions(region)
        itk_image_original.SetSpacing(spacing)
        itk_image_original.SetOrigin(origin)
        itk_image_original.Allocate()

        # Fill with test data
        array_original = np.random.rand(size[2], size[1], size[0]).astype(np.float32)
        itk.array_view_from_image(itk_image_original)[:] = array_original

        # Roundtrip conversion
        sitk_image = image_tools.convert_itk_image_to_sitk(itk_image_original)
        itk_image_final = image_tools.convert_sitk_image_to_itk(sitk_image)

        # Verify metadata preserved
        assert itk.size(itk_image_final) == tuple(size)
        assert np.allclose(itk.spacing(itk_image_final), spacing)
        assert np.allclose(itk.origin(itk_image_final), origin)

        # Verify data preserved
        array_final = itk.array_from_image(itk_image_final)
        assert np.allclose(array_original, array_final)

        print("Roundtrip scalar conversion successful")

    def test_itk_to_sitk_vector_image(self, image_tools: ImageTools) -> None:
        """Test conversion of vector ITK image to SimpleITK."""
        # Create a 3D vector ITK image (like a displacement field)
        size = [8, 12, 16]
        spacing = [1.0, 1.0, 1.0]
        origin = [0.0, 0.0, 0.0]

        # Create vector image with 3 components
        VectorImageType = itk.Image[itk.Vector[itk.F, 3], 3]
        itk_image = VectorImageType.New()

        region = itk.ImageRegion[3]()
        region.SetSize(size)
        itk_image.SetRegions(region)
        itk_image.SetSpacing(spacing)
        itk_image.SetOrigin(origin)
        itk_image.Allocate()

        # Fill with test vector data
        array = np.random.rand(size[2], size[1], size[0], 3).astype(np.float32)
        itk.array_view_from_image(itk_image)[:] = array

        # Convert to SimpleITK
        sitk_image = image_tools.convert_itk_image_to_sitk(itk_image)

        # Verify it's a vector image
        assert sitk_image.GetNumberOfComponentsPerPixel() == 3

        # Verify metadata
        assert sitk_image.GetSize() == tuple(size)
        assert sitk_image.GetSpacing() == tuple(spacing)

        # Verify data
        array_sitk = sitk.GetArrayFromImage(sitk_image)
        assert np.allclose(array, array_sitk)

        print("ITK to SimpleITK vector conversion successful")

    def test_sitk_to_itk_vector_image(self, image_tools: ImageTools) -> None:
        """Test conversion of vector SimpleITK image to ITK."""
        # Create a 3D vector SimpleITK image
        size = [8, 12, 16]
        spacing = [1.0, 1.0, 1.0]
        origin = [0.0, 0.0, 0.0]

        # Create vector data
        array = np.random.rand(size[2], size[1], size[0], 3).astype(np.float32)

        # Create SimpleITK vector image
        sitk_image = sitk.GetImageFromArray(array, isVector=True)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)

        # Convert to ITK
        itk_image = image_tools.convert_sitk_image_to_itk(sitk_image)

        # Verify it's a vector image
        assert itk_image.GetNumberOfComponentsPerPixel() == 3

        # Verify metadata
        assert itk.size(itk_image) == tuple(size)
        assert itk.spacing(itk_image) == tuple(spacing)

        # Verify data
        array_itk = itk.array_from_image(itk_image)
        assert np.allclose(array, array_itk)

        print("SimpleITK to ITK vector conversion successful")

    def test_roundtrip_vector_image(self, image_tools: ImageTools) -> None:
        """Test roundtrip conversion for vector images: ITK -> SimpleITK -> ITK."""
        # Create ITK vector image
        size = [10, 15, 20]
        spacing = [0.8, 1.2, 1.6]
        origin = [5.0, 10.0, 15.0]

        VectorImageType = itk.Image[itk.Vector[itk.F, 3], 3]
        itk_image_original = VectorImageType.New()

        region = itk.ImageRegion[3]()
        region.SetSize(size)
        itk_image_original.SetRegions(region)
        itk_image_original.SetSpacing(spacing)
        itk_image_original.SetOrigin(origin)
        itk_image_original.Allocate()

        # Fill with test vector data
        array_original = np.random.rand(size[2], size[1], size[0], 3).astype(np.float32)
        itk.array_view_from_image(itk_image_original)[:] = array_original

        # Roundtrip conversion
        sitk_image = image_tools.convert_itk_image_to_sitk(itk_image_original)
        itk_image_final = image_tools.convert_sitk_image_to_itk(sitk_image)

        # Verify metadata preserved
        assert itk.size(itk_image_final) == tuple(size)
        assert np.allclose(itk.spacing(itk_image_final), spacing)
        assert np.allclose(itk.origin(itk_image_final), origin)
        assert itk_image_final.GetNumberOfComponentsPerPixel() == 3

        # Verify data preserved
        array_final = itk.array_from_image(itk_image_final)
        assert np.allclose(array_original, array_final)

        print("Roundtrip vector conversion successful")

    @pytest.mark.slow
    def test_imwrite_imread_vd3(
        self,
        image_tools: ImageTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test reading and writing double precision vector images."""
        from monai_physio.transform_tools import TransformTools

        output_dir = test_directories["output"]
        img_output_dir = output_dir / "image_tools"
        img_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        fixed_to_moving_transform = test_transforms["fixed_to_moving_transform"]

        print("\nTesting imwriteVD3 and imreadVD3...")

        # Generate a deformation field using TransformTools
        transform_tools = TransformTools()
        deformation_field = transform_tools.convert_transform_to_displacement_field(
            fixed_to_moving_transform, fixed_image
        )

        # Verify it's double precision vector image
        field_type = str(type(deformation_field))
        print(f"  Original field type: {field_type}")
        assert "VD" in field_type, "Expected double precision vector image"

        # Get original data for comparison
        original_arr = itk.array_from_image(deformation_field)

        # Write using imwriteVD3
        output_path = str(img_output_dir / "test_vector_field_vd3.mha")
        image_tools.imwriteVD3(deformation_field, output_path, compression=True)

        print(f"  Wrote to: {output_path}")

        # Read back using imreadVD3
        field_read = image_tools.imreadVD3(output_path)

        # Verify read field
        assert field_read is not None, "Read field is None"
        assert itk.size(field_read) == itk.size(deformation_field), "Size mismatch"

        # Verify it's double precision
        read_type = str(type(field_read))
        print(f"  Read field type: {read_type}")
        assert "VD" in read_type, "Expected double precision vector image after reading"

        # Compare data
        read_arr = itk.array_from_image(field_read)
        assert read_arr.shape == original_arr.shape, "Array shape mismatch"

        # Check numerical accuracy (should be very close, small float precision loss)
        max_diff = np.max(np.abs(read_arr - original_arr))
        mean_diff = np.mean(np.abs(read_arr - original_arr))

        print("Vector field I/O test complete")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")

        # Differences should be very small (float precision conversion)
        assert max_diff < 1e-5, f"Max difference too large: {max_diff}"
        assert mean_diff < 1e-6, f"Mean difference too large: {mean_diff}"


def _make_synthetic_itk_image(
    shape_xyz: tuple[int, int, int],
    arr: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
) -> Any:
    """Create a small 3D ITK image. shape_xyz is (nx, ny, nz); array from ITK is (nz, ny, nx)."""
    # ITK uses (x, y, z) for size; array_from_image gives (z, y, x)
    if arr is None:
        arr = np.arange(np.prod(shape_xyz), dtype=np.float32).reshape(
            shape_xyz[2], shape_xyz[1], shape_xyz[0]
        )
    ImageType = itk.Image[itk.F, 3]
    itk_image = ImageType.New()
    region = itk.ImageRegion[3]()
    region.SetSize(shape_xyz)  # (nx, ny, nz)
    itk_image.SetRegions(region)
    itk_image.SetSpacing([1.0, 1.0, 1.0])
    itk_image.SetOrigin([0.0, 0.0, 0.0])
    if direction is not None:
        itk_image.SetDirection(itk.matrix_from_array(direction))
    itk_image.Allocate()
    itk.array_view_from_image(itk_image)[:] = arr
    return itk_image


class TestFlipImage:
    """Unit tests for ImageTools.flip_image (axis flips and direction reset)."""

    @pytest.fixture
    def image_tools(self) -> ImageTools:
        return ImageTools()

    def test_flip_x_flips_along_last_array_axis(self, image_tools: ImageTools) -> None:
        """flip_x flips the image along the x (last) array dimension."""
        # Small image: ITK size (nx, ny, nz) = (3, 2, 2) -> array shape (2, 2, 3)
        shape_xyz = (3, 2, 2)
        arr = np.arange(12, dtype=np.float32).reshape(2, 2, 3)  # (z, y, x)
        itk_image = _make_synthetic_itk_image(shape_xyz, arr=arr)
        out = image_tools.flip_image(itk_image, flip_x=True)
        out_arr = itk.array_from_image(out)
        expected = np.flip(arr, axis=2)
        assert np.allclose(out_arr, expected), (
            "flip_x should match np.flip(..., axis=2)"
        )

    def test_flip_y_flips_along_middle_array_axis(
        self, image_tools: ImageTools
    ) -> None:
        """flip_y flips the image along the y (middle) array dimension."""
        shape_xyz = (3, 2, 2)
        arr = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        itk_image = _make_synthetic_itk_image(shape_xyz, arr=arr)
        out = image_tools.flip_image(itk_image, flip_y=True)
        out_arr = itk.array_from_image(out)
        expected = np.flip(arr, axis=1)
        assert np.allclose(out_arr, expected), (
            "flip_y should match np.flip(..., axis=1)"
        )

    def test_flip_z_flips_along_first_array_axis(self, image_tools: ImageTools) -> None:
        """flip_z flips the image along the z (first) array dimension."""
        shape_xyz = (3, 2, 2)
        arr = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        itk_image = _make_synthetic_itk_image(shape_xyz, arr=arr)
        out = image_tools.flip_image(itk_image, flip_z=True)
        out_arr = itk.array_from_image(out)
        expected = np.flip(arr, axis=0)
        assert np.allclose(out_arr, expected), (
            "flip_z should match np.flip(..., axis=0)"
        )

    def test_flip_xy_combines_flips(self, image_tools: ImageTools) -> None:
        """flip_x and flip_y together flip both axes."""
        shape_xyz = (3, 2, 2)
        arr = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        itk_image = _make_synthetic_itk_image(shape_xyz, arr=arr)
        out = image_tools.flip_image(itk_image, flip_x=True, flip_y=True)
        out_arr = itk.array_from_image(out)
        expected = np.flip(np.flip(arr, axis=2), axis=1)
        assert np.allclose(out_arr, expected)

    def test_no_flip_returns_same_image(self, image_tools: ImageTools) -> None:
        """With no flip flags, image is returned unchanged."""
        shape_xyz = (2, 2, 2)
        arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        itk_image = _make_synthetic_itk_image(shape_xyz, arr=arr)
        out = image_tools.flip_image(
            itk_image, flip_x=False, flip_y=False, flip_z=False
        )
        out_arr = itk.array_from_image(out)
        assert np.allclose(out_arr, arr)

    def test_mask_flipped_in_lockstep_with_image(self, image_tools: ImageTools) -> None:
        """When a mask is provided, it is flipped with the same axes as the image."""
        shape_xyz = (3, 2, 2)
        arr = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        mask_arr = (arr % 2 == 0).astype(np.float32)  # 0/1 pattern in lockstep with arr
        itk_image = _make_synthetic_itk_image(shape_xyz, arr=arr)
        itk_mask = _make_synthetic_itk_image(shape_xyz, arr=mask_arr)
        out_image, out_mask = image_tools.flip_image(
            itk_image, in_mask=itk_mask, flip_x=True, flip_z=True
        )
        out_img_arr = itk.array_from_image(out_image)
        out_msk_arr = itk.array_from_image(out_mask)
        expected_img = np.flip(np.flip(arr, axis=2), axis=0)
        expected_msk = np.flip(np.flip(mask_arr, axis=2), axis=0)
        assert np.allclose(out_img_arr, expected_img), "Image should be flipped x and z"
        assert np.allclose(out_msk_arr, expected_msk), (
            "Mask should be flipped in lockstep"
        )
        # Consistency: mask value should still align with image (same pattern, flipped)
        assert np.allclose(out_msk_arr, (out_img_arr % 2 == 0).astype(np.float32))

    def test_flip_and_make_identity_sets_direction_to_identity(
        self, image_tools: ImageTools
    ) -> None:
        """flip_and_make_identity flips as needed and sets direction matrix to identity."""
        shape_xyz = (2, 2, 2)
        arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        # Direction with negative diagonal (e.g. flipped along one axis)
        direction = np.diag([-1.0, 1.0, 1.0])
        itk_image = _make_synthetic_itk_image(shape_xyz, arr=arr, direction=direction)
        out = image_tools.flip_image(itk_image, flip_and_make_identity=True)
        out_direction = itk.array_from_matrix(out.GetDirection())
        identity = np.eye(3)
        assert np.allclose(out_direction, identity), (
            "flip_and_make_identity should set direction to identity"
        )

    def test_flip_and_make_identity_with_mask_sets_both_directions_to_identity(
        self, image_tools: ImageTools
    ) -> None:
        """With mask and flip_and_make_identity, both image and mask get identity direction."""
        shape_xyz = (2, 2, 2)
        arr = np.ones((2, 2, 2), dtype=np.float32)
        mask_arr = np.ones((2, 2, 2), dtype=np.float32)
        direction = np.diag([1.0, -1.0, 1.0])
        itk_image = _make_synthetic_itk_image(shape_xyz, arr=arr, direction=direction)
        itk_mask = _make_synthetic_itk_image(
            shape_xyz, arr=mask_arr, direction=direction
        )
        out_image, out_mask = image_tools.flip_image(
            itk_image, in_mask=itk_mask, flip_and_make_identity=True
        )
        for im, name in [(out_image, "image"), (out_mask, "mask")]:
            dir_mat = itk.array_from_matrix(im.GetDirection())
            assert np.allclose(dir_mat, np.eye(3)), (
                f"flip_and_make_identity should set {name} direction to identity"
            )

    def test_keep_largest_connected_component_keeps_largest_blob(
        self, image_tools: ImageTools
    ) -> None:
        """Pure connected-component unit test on a synthetic 10x10x10 mask
        with two disjoint blobs (shape (X, Y, Z) = (10, 10, 10))."""
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 1  # small blob (8 voxels)
        arr[5:9, 5:9, 5:9] = 1  # large blob (64 voxels)
        image = itk.image_from_array(arr)

        result = image_tools.keep_largest_connected_component(image)
        result_arr = itk.array_from_image(result)

        assert result_arr[1:3, 1:3, 1:3].sum() == 0
        assert result_arr[5:9, 5:9, 5:9].sum() == arr[5:9, 5:9, 5:9].sum()

    def test_keep_largest_connected_component_no_foreground_returns_empty(
        self, image_tools: ImageTools
    ) -> None:
        """An all-background mask (shape (X, Y, Z) = (8, 8, 8)) returns
        an all-background result."""
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        image = itk.image_from_array(arr)

        result = image_tools.keep_largest_connected_component(image)
        result_arr = itk.array_from_image(result)

        assert result_arr.sum() == 0


class TestResampleImageByScale:
    """Unit tests for ImageTools.resample_image_by_scale."""

    @pytest.fixture
    def image_tools(self) -> ImageTools:
        return ImageTools()

    @pytest.mark.parametrize("scale", [0.5, 0.25, 2.0])
    def test_physical_extent_is_preserved(
        self, image_tools: ImageTools, scale: float
    ) -> None:
        """Voxel count scales while the physical extent stays put."""
        itk_image = _make_synthetic_itk_image((8, 6, 4))
        itk_image.SetSpacing([1.0, 2.0, 3.0])

        out = image_tools.resample_image_by_scale(itk_image, scale)

        size = np.asarray(itk_image.GetLargestPossibleRegion().GetSize())
        new_size = np.asarray(out.GetLargestPossibleRegion().GetSize())
        assert np.array_equal(new_size, np.ceil(size * scale).astype(int))
        assert np.allclose(
            new_size * np.asarray(out.GetSpacing()),
            size * np.asarray(itk_image.GetSpacing()),
        ), "resampling must not change the physical extent"

    def test_direction_is_unchanged(self, image_tools: ImageTools) -> None:
        """A left-handed direction survives resampling untouched."""
        direction = np.diag([1.0, 1.0, -1.0])
        itk_image = _make_synthetic_itk_image((8, 6, 4), direction=direction)

        out = image_tools.resample_image_by_scale(itk_image, 0.5)

        assert np.allclose(itk.array_from_matrix(out.GetDirection()), direction)

    def test_nearest_neighbor_keeps_input_values(self, image_tools: ImageTools) -> None:
        """Nearest-neighbor resampling introduces no new intensities."""
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        arr[1:3, 1:3, 1:3] = 7.0
        itk_image = _make_synthetic_itk_image((4, 4, 4), arr=arr)

        out = image_tools.resample_image_by_scale(itk_image, 0.5, interpolate=False)

        values = set(np.unique(itk.array_from_image(out)))
        assert values <= {0.0, 7.0}
        assert 7.0 in values, "coarsening must keep the block, not sample past it"

    @pytest.mark.parametrize("scale", [0.0, -1.0])
    def test_non_positive_scale_raises(
        self, image_tools: ImageTools, scale: float
    ) -> None:
        """A scale of zero or less is rejected."""
        itk_image = _make_synthetic_itk_image((4, 4, 4))
        with pytest.raises(ValueError, match="scale must be positive"):
            image_tools.resample_image_by_scale(itk_image, scale)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
