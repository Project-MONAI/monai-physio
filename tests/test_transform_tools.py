"""
Test for transform tools functionality.

This test depends on test_register_images_ANTS and uses registration
transforms to test transform manipulation and application.
"""

from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest
import pyvista as pv
import vtk

from physiotwin4d.image_tools import ImageTools
from physiotwin4d.transform_tools import TransformTools


def _sphere_shell_samples(
    radius_mm: float = 12.0, size: int = 40
) -> tuple[Any, Any, Any, Any, Any]:
    """A one-voxel sphere shell of displacement samples on a 1 mm grid.

    Returns the shell's unit normals, a purely radial (expanding) field, a
    purely tangential (rotating) field, the per-voxel sample weights, and a
    binary mask of the sphere's interior. The two fields are the extremes the
    normal/tangential split has to separate: the radial one survives it whole,
    the tangential one is entirely what it must stop at the boundary.
    """
    center = 0.5 * (size - 1)
    grid = np.arange(size, dtype=np.float64) - center
    # (z, y, x) index space; the vector components stay in (x, y, z) order.
    dz, dy, dx = np.meshgrid(grid, grid, grid, indexing="ij")
    offset = np.stack([dx, dy, dz], axis=3)
    distance = np.linalg.norm(offset, axis=3)

    shell = np.abs(distance - radius_mm) <= 0.5
    normals = np.zeros_like(offset)
    normals[shell] = offset[shell] / distance[shell][:, None]

    radial = normals * 3.0
    # A rotation about +z: perpendicular to the normal everywhere on a sphere.
    tangential = np.zeros_like(offset)
    tangential[shell] = (
        np.stack([-dy, dx, np.zeros_like(dx)], axis=3)[shell] * 3.0 / radius_mm
    )

    weights = shell.astype(np.float32)
    interior = (distance <= radius_mm).astype(np.float32)
    return normals, radial, tangential, weights, interior


def _as_field(array: Any) -> Any:
    """Wrap a ``(z, y, x, 3)`` array as the vector image type ITK rasterizes to."""
    return itk.image_from_array(
        np.ascontiguousarray(array.astype(np.float32)), is_vector=True
    )


def test_smooth_deformation_field_transform_stops_sliding_outside_the_mask() -> None:
    """Outside the mask only motion along the surface normal is propagated.

    A radial field is entirely normal, so restricting it changes nothing. A
    tangential field is entirely sliding, so outside the mask it must vanish
    while staying untouched inside.
    """
    radius_mm, sigma_mm = 12.0, 4.0
    normals, radial, tangential, weights, interior = _sphere_shell_samples(radius_mm)
    normal_image = _as_field(normals)
    weight_image = itk.image_from_array(weights)
    mask_image = itk.image_from_array(interior)

    tools = TransformTools()

    def spread(samples: Any, restrict: bool) -> Any:
        field = _as_field(samples)
        transform = tools.smooth_deformation_field_transform(
            field,
            sigma_mm,
            weight_image,
            normal_image if restrict else None,
            mask_image if restrict else None,
        )
        return itk.array_from_image(transform.GetDisplacementField())

    # Well outside the shell but still within reach of the smoothing, and well
    # inside it. Sampling on the shell itself would straddle the mask edge.
    distance = np.linalg.norm(
        np.stack(np.meshgrid(*(3 * [np.arange(40.0) - 19.5]), indexing="ij"), axis=3),
        axis=3,
    )
    outside = (distance > radius_mm + 2.0) & (distance < radius_mm + 5.0)
    inside = distance < radius_mm - 2.0

    # Tolerances are set by the float32 the samples are rasterized in: the
    # projection reconstructs a purely normal vector, and annihilates a purely
    # tangential one, to about 1e-7 of the 3 mm they carry.
    radial_free, radial_held = spread(radial, False), spread(radial, True)
    np.testing.assert_allclose(radial_held, radial_free, atol=1e-5)

    tangential_free, tangential_held = (
        spread(tangential, False),
        spread(tangential, True),
    )
    # The unrestricted spread really does drag the surroundings around, so the
    # assertion below is not passing on an already-zero field.
    assert np.abs(tangential_free[outside]).max() > 0.1
    assert np.abs(tangential_held[outside]).max() < 1e-4
    np.testing.assert_allclose(
        tangential_held[inside], tangential_free[inside], atol=1e-5
    )


def test_smooth_deformation_field_transform_rejects_a_lone_normal_or_mask() -> None:
    """The normals say what to project onto, the mask says where to."""
    normals, radial, _, weights, interior = _sphere_shell_samples()
    tools = TransformTools()

    with pytest.raises(ValueError, match="must be given together"):
        tools.smooth_deformation_field_transform(
            _as_field(radial), 4.0, itk.image_from_array(weights), _as_field(normals)
        )
    with pytest.raises(ValueError, match="must be given together"):
        tools.smooth_deformation_field_transform(
            _as_field(radial),
            4.0,
            itk.image_from_array(weights),
            interior_mask=itk.image_from_array(interior),
        )


def test_generate_grid_image_clamps_boundary_lines() -> None:
    """
    Grid image clamps boundary slices for an ITK image with axes (X, Y, Z).

    The synthetic ITK image has axes (X, Y, Z) = (7, 6, 5), created from
    NumPy array shape (Z, Y, X) = (5, 6, 7).
    """
    image_arr = np.zeros((5, 6, 7), dtype=np.float32)
    image_arr[-1, -1, -1] = 5.0
    image = itk.image_from_array(image_arr)

    grid_image = TransformTools().generate_grid_image(image, grid_size=2, line_width=3)
    grid_arr = itk.array_from_image(grid_image)

    assert grid_arr.shape == image_arr.shape
    assert grid_arr[0, 0, 0] == 5.0


def _small_reference_image() -> itk.Image:
    """Return a small grid to compose against."""
    image = itk.Image[itk.F, 3].New()
    region = itk.ImageRegion[3]()
    region.SetSize([40, 40, 40])
    image.SetRegions(region)
    image.SetSpacing([2.0, 2.0, 2.0])
    image.SetOrigin([-40.0, -40.0, -40.0])
    image.Allocate()
    image.FillBuffer(0)
    return image


def _affine_and_translation() -> tuple[Any, Any]:
    """Return two transforms whose composition is not the identity."""
    affine = itk.AffineTransform[itk.D, 3].New()
    affine.SetMatrix(
        itk.GetMatrixFromArray(
            np.array([[1.02, 0.01, 0.0], [0.0, 0.99, 0.02], [0.01, 0.0, 1.01]])
        )
    )
    offset = itk.Vector[itk.D, 3]()
    for index, value in enumerate((1.5, -2.0, 0.75)):
        offset[index] = value
    affine.SetTranslation(offset)

    translation = itk.TranslationTransform[itk.D, 3].New()
    translation.SetOffset([0.6, -0.4, 0.9])
    return affine, translation


def test_composing_at_unit_weight_chains_the_inputs_instead_of_rasterizing() -> None:
    """Composing with nothing to apply must not build a displacement field.

    A CompositeTransform chains its members lazily, so rasterizing them first
    only reproduces what it would compute anyway -- less accurately, since
    sampling onto a grid and interpolating back adds error that evaluating the
    originals does not.

    It is also what dominated memory. The distance-map caller composes on a grid
    padded to 2.5 * mask_dilation_mm per side; on the lung chest CT that is
    758 x 758 x 664, where one ``Vector<double, 3>`` field is 9.2 GB, and both
    directions together retained 36.6 GB to hold what is really an affine plus a
    175-cubed field.
    """
    transform_tools = TransformTools()
    affine, translation = _affine_and_translation()

    composed = transform_tools.combine_displacement_field_transforms(
        affine, translation, reference_image=_small_reference_image(), mode="compose"
    )

    # The members are the inputs themselves, not fields sampled from them.
    # Every ITK transform surfaces in Python as ``itkTransformD33``, so
    # ``isinstance`` cannot tell them apart and would pass vacuously here;
    # ``GetNameOfClass`` is what actually discriminates.
    assert composed.GetNumberOfTransforms() == 2
    members = [
        composed.GetNthTransform(index).GetNameOfClass()
        for index in range(composed.GetNumberOfTransforms())
    ]
    assert members == ["AffineTransform", "TranslationTransform"], (
        f"Composing at unit weight should chain the inputs unchanged, got {members}"
    )

    expected = itk.CompositeTransform[itk.D, 3].New()
    expected.AddTransform(affine)
    expected.AddTransform(translation)
    for point in (
        [-20.0, -10.0, 5.0],
        [0.0, 0.0, 0.0],
        [15.0, 20.0, -25.0],
        [30.0, -30.0, 30.0],
    ):
        got = np.array(list(composed.TransformPoint(point)))
        want = np.array(list(expected.TransformPoint(point)))
        assert np.allclose(got, want, atol=1e-9), (
            f"Chained composition disagreed with an explicit CompositeTransform "
            f"at {point}: {got} vs {want}"
        )


def test_composing_with_a_weight_or_a_blur_still_rasterizes() -> None:
    """Weighting and blurring need the field, so those keep the sampling path.

    This is the half of the branch the existing callers rely on: scaling or
    smoothing a transform cannot be expressed by chaining it unchanged.
    """
    transform_tools = TransformTools()
    affine, translation = _affine_and_translation()
    reference = _small_reference_image()

    for description, kwargs in (
        ("a non-unit weight", {"tfm2_weight": 0.5}),
        ("a blur", {"tfm2_blur_sigma": 0.5}),
    ):
        composed = transform_tools.combine_displacement_field_transforms(
            affine, translation, reference_image=reference, mode="compose", **kwargs
        )
        assert (
            composed.GetNthTransform(1).GetNameOfClass() == "DisplacementFieldTransform"
        ), f"Composing with {description} must still build a displacement field"


@pytest.mark.slow
class TestTransformTools:
    """Test suite for TransformTools functionality."""

    @pytest.fixture(scope="class")
    def test_contour(self, test_images: list[Any]) -> Any:
        """Create a simple test contour mesh."""
        # Create a sphere mesh for testing
        sphere = pv.Sphere(radius=50.0, center=(100, 100, 100))
        return sphere

    def test_transform_tools_initialization(
        self, transform_tools: TransformTools
    ) -> None:
        """Test that TransformTools initializes correctly."""
        assert transform_tools is not None, "TransformTools not initialized"
        print("\nTransformTools initialized successfully")

    def test_transform_image_linear(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test transforming image with linear interpolation."""
        output_dir = test_directories["output"]
        tfm_output_dir = output_dir / "transform_tools"
        tfm_output_dir.mkdir(exist_ok=True)

        moving_image = test_images[1]
        fixed_image = test_images[0]
        forward_transform = test_transforms["forward_transform"]

        print("\nTransforming image with linear interpolation...")

        transformed_image = transform_tools.transform_image(
            moving_image, forward_transform, fixed_image, interpolation_method="linear"
        )

        # Verify result
        assert transformed_image is not None, "Transformed image is None"
        assert itk.size(transformed_image) == itk.size(fixed_image), "Size mismatch"
        assert itk.spacing(transformed_image) == itk.spacing(fixed_image), (
            "Spacing mismatch"
        )

        print("Image transformed with linear interpolation")
        print(f"  Output size: {itk.size(transformed_image)}")
        print(f"  Output spacing: {itk.spacing(transformed_image)}")

        # Save transformed image
        itk.imwrite(
            transformed_image,
            str(tfm_output_dir / "transformed_linear.mha"),
            compression=True,
        )

    def test_transform_image_nearest(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test transforming image with nearest neighbor interpolation."""
        output_dir = test_directories["output"]
        tfm_output_dir = output_dir / "transform_tools"
        tfm_output_dir.mkdir(exist_ok=True)

        moving_image = test_images[1]
        fixed_image = test_images[0]
        forward_transform = test_transforms["forward_transform"]

        print("\nTransforming image with nearest neighbor interpolation...")

        transformed_image = transform_tools.transform_image(
            moving_image, forward_transform, fixed_image, interpolation_method="nearest"
        )

        assert transformed_image is not None, "Transformed image is None"
        assert itk.size(transformed_image) == itk.size(fixed_image), "Size mismatch"

        print("Image transformed with nearest neighbor interpolation")

        # Save transformed image
        itk.imwrite(
            transformed_image,
            str(tfm_output_dir / "transformed_nearest.mha"),
            compression=True,
        )

    def test_transform_image_sinc(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test transforming image with sinc interpolation."""
        output_dir = test_directories["output"]
        tfm_output_dir = output_dir / "transform_tools"
        tfm_output_dir.mkdir(exist_ok=True)

        moving_image = test_images[1]
        fixed_image = test_images[0]
        forward_transform = test_transforms["forward_transform"]

        print("\nTransforming image with sinc interpolation...")

        transformed_image = transform_tools.transform_image(
            moving_image, forward_transform, fixed_image, interpolation_method="sinc"
        )

        assert transformed_image is not None, "Transformed image is None"
        assert itk.size(transformed_image) == itk.size(fixed_image), "Size mismatch"

        print("Image transformed with sinc interpolation")

        # Save transformed image
        itk.imwrite(
            transformed_image,
            str(tfm_output_dir / "transformed_sinc.mha"),
            compression=True,
        )

    def test_transform_image_invalid_method(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
    ) -> None:
        """Test that invalid interpolation method raises error."""
        moving_image = test_images[1]
        fixed_image = test_images[0]
        forward_transform = test_transforms["forward_transform"]

        print("\nTesting invalid interpolation method...")

        with pytest.raises(ValueError):
            transform_tools.transform_image(
                moving_image,
                forward_transform,
                fixed_image,
                interpolation_method="invalid",
            )

        print("Invalid method correctly raises ValueError")

    def test_transform_pvcontour_without_deformation(
        self,
        transform_tools: TransformTools,
        test_contour: Any,
        test_transforms: dict[str, Any],
    ) -> None:
        """Test transforming PyVista contour without deformation magnitude."""
        forward_transform = test_transforms["forward_transform"]

        print("\nTransforming contour without deformation magnitude...")
        print(f"  Original contour points: {test_contour.n_points}")

        transformed_contour = transform_tools.transform_pvcontour(
            test_contour, forward_transform, with_deformation_magnitude=False
        )

        # Verify result
        assert transformed_contour is not None, "Transformed contour is None"
        assert transformed_contour.n_points == test_contour.n_points, (
            "Point count changed"
        )
        assert "DeformationMagnitude" not in transformed_contour.point_data, (
            "DeformationMagnitude should not be present"
        )

        # Check that points actually changed
        original_points = test_contour.points
        transformed_points = transformed_contour.points

        max_diff = np.max(np.abs(transformed_points - original_points))

        print("Contour transformed without deformation magnitude")
        print(f"  Transformed contour points: {transformed_contour.n_points}")
        print(f"  Max point displacement: {max_diff:.2f} mm")

    def test_transform_pvcontour_with_deformation(
        self,
        transform_tools: TransformTools,
        test_contour: Any,
        test_transforms: dict[str, Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test transforming PyVista contour with deformation magnitude."""
        output_dir = test_directories["output"]
        tfm_output_dir = output_dir / "transform_tools"
        tfm_output_dir.mkdir(exist_ok=True)

        forward_transform = test_transforms["forward_transform"]

        print("\nTransforming contour with deformation magnitude...")

        transformed_contour = transform_tools.transform_pvcontour(
            test_contour, forward_transform, with_deformation_magnitude=True
        )

        # Verify result
        assert transformed_contour is not None, "Transformed contour is None"
        assert "DeformationMagnitude" in transformed_contour.point_data, (
            "DeformationMagnitude not present"
        )

        # Check deformation magnitude values
        deformation = transformed_contour["DeformationMagnitude"]
        assert len(deformation) == transformed_contour.n_points, (
            "Deformation array size mismatch"
        )
        assert np.all(deformation >= 0), "Deformation magnitude should be non-negative"

        mean_def = np.mean(deformation)
        max_def = np.max(deformation)

        print("Contour transformed with deformation magnitude")
        print(f"  Mean deformation: {mean_def:.2f} mm")
        print(f"  Max deformation: {max_def:.2f} mm")

        # Save transformed contour
        transformed_contour.save(str(tfm_output_dir / "transformed_contour.vtp"))

    def test_transform_dataset_preserves_unstructured_grid_topology(
        self,
        transform_tools: TransformTools,
    ) -> None:
        """Transform UnstructuredGrid points with image shape (Z, Y, X) = (3, 3, 3)."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cells = np.array([4, 0, 1, 2, 3])
        celltypes = np.array([pv.CellType.TETRA])
        mesh = pv.UnstructuredGrid(cells, celltypes, points)
        mesh.cell_data["label"] = np.array([7], dtype=np.uint8)
        mesh.point_data["weights"] = np.arange(mesh.n_points, dtype=np.float64)

        transform = itk.AffineTransform[itk.D, 3].New()
        transform.SetIdentity()
        transform.SetTranslation((1.0, 2.0, 3.0))

        output = transform_tools.transform_dataset(mesh, transform)

        assert isinstance(output, pv.UnstructuredGrid)
        assert output.n_cells == mesh.n_cells
        np.testing.assert_array_equal(output.celltypes, mesh.celltypes)
        np.testing.assert_array_equal(
            output.cell_data["label"], mesh.cell_data["label"]
        )
        np.testing.assert_array_equal(
            output.point_data["weights"], mesh.point_data["weights"]
        )
        np.testing.assert_allclose(output.points, points + np.array([1.0, 2.0, 3.0]))

    def test_convert_transform_to_displacement_field(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test converting transform to deformation field image."""
        output_dir = test_directories["output"]
        tfm_output_dir = output_dir / "transform_tools"
        tfm_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        forward_transform = test_transforms["forward_transform"]

        print("\nConverting transform to deformation field...")

        deformation_field = transform_tools.convert_transform_to_displacement_field(
            forward_transform, fixed_image
        )

        # Verify deformation field
        assert deformation_field is not None, "Deformation field is None"
        assert itk.size(deformation_field) == itk.size(fixed_image), "Size mismatch"

        # Check that it's a vector image
        field_arr = itk.array_from_image(deformation_field)
        assert field_arr.shape[-1] == 3, "Should have 3 components (x, y, z)"

        print("Transform converted to deformation field")
        print(f"  Field size: {itk.size(deformation_field)}")
        print(f"  Field shape: {field_arr.shape}")

        # Save deformation field using imwriteVD3 (for double precision vector images)
        image_tools = ImageTools()
        image_tools.imwriteVD3(
            deformation_field,
            str(tfm_output_dir / "deformation_field.mha"),
            compression=True,
        )

    def test_convert_vtk_matrix_to_itk_transform(
        self, transform_tools: TransformTools
    ) -> None:
        """Test converting VTK matrix to ITK transform."""
        # Create a VTK matrix
        vtk_matrix = vtk.vtkMatrix4x4()
        vtk_matrix.Identity()

        # Set translation
        vtk_matrix.SetElement(0, 3, 10.0)
        vtk_matrix.SetElement(1, 3, 20.0)
        vtk_matrix.SetElement(2, 3, 30.0)

        print("\nConverting VTK matrix to ITK transform...")

        itk_transform = transform_tools.convert_vtk_matrix_to_itk_transform(vtk_matrix)

        # Verify transform
        assert itk_transform is not None, "ITK transform is None"
        assert isinstance(itk_transform, itk.AffineTransform), (
            "Should be an AffineTransform"
        )

        # Check translation
        offset = itk_transform.GetOffset()
        assert abs(offset[0] - 10.0) < 0.01, "X translation incorrect"
        assert abs(offset[1] - 20.0) < 0.01, "Y translation incorrect"
        assert abs(offset[2] - 30.0) < 0.01, "Z translation incorrect"

        print("VTK matrix converted to ITK transform")
        print(f"  Translation: [{offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f}]")

    def test_compute_jacobian_determinant_from_field(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test computing Jacobian determinant from deformation field."""
        output_dir = test_directories["output"]
        tfm_output_dir = output_dir / "transform_tools"
        tfm_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        forward_transform = test_transforms["forward_transform"]

        # First convert transform to field
        print("\nComputing Jacobian determinant from deformation field...")

        deformation_field = transform_tools.convert_transform_to_displacement_field(
            forward_transform, fixed_image
        )

        jacobian_det = transform_tools.compute_jacobian_determinant_from_field(
            deformation_field
        )

        # Verify Jacobian determinant
        assert jacobian_det is not None, "Jacobian determinant is None"
        assert itk.size(jacobian_det) == itk.size(fixed_image), "Size mismatch"

        # Check values
        jac_arr = itk.array_from_image(jacobian_det)
        mean_jac = np.mean(jac_arr)
        min_jac = np.min(jac_arr)
        max_jac = np.max(jac_arr)

        print("Jacobian determinant computed")
        print(f"  Mean: {mean_jac:.3f}")
        print(f"  Min: {min_jac:.3f}")
        print(f"  Max: {max_jac:.3f}")

        # Jacobian determinant should be around 1.0 for volume-preserving transforms
        assert mean_jac > 0, "Mean Jacobian should be positive"

        # Save Jacobian determinant
        itk.imwrite(
            jacobian_det,
            str(tfm_output_dir / "jacobian_determinant.mha"),
            compression=True,
        )

    def test_detect_folding_in_field(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
    ) -> None:
        """Test detecting spatial folding in deformation field."""
        fixed_image = test_images[0]
        forward_transform = test_transforms["forward_transform"]

        # Convert transform to field
        print("\nDetecting folding in deformation field...")

        deformation_field = transform_tools.convert_transform_to_displacement_field(
            forward_transform, fixed_image
        )

        # Compute jacobian determinant from field
        jacobian_det = transform_tools.compute_jacobian_determinant_from_field(
            deformation_field
        )

        has_folding = transform_tools.detect_folding_in_field(jacobian_det)

        # Verify result
        assert isinstance(has_folding, bool), "Result should be boolean"

        print("Folding detection complete")
        print(f"  Has folding: {has_folding}")

    def test_interpolate_transforms(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
    ) -> None:
        """Test temporal interpolation between transforms."""
        forward_transform = test_transforms["forward_transform"]

        # Create an identity transform as second transform
        identity_tfm = itk.AffineTransform[itk.D, 3].New()
        identity_tfm.SetIdentity()

        fixed_image = test_images[0]

        print("\nInterpolating between transforms...")

        # Interpolate at midpoint (portion=0.5)
        interpolated_tfm = transform_tools.combine_displacement_field_transforms(
            forward_transform,
            identity_tfm,
            fixed_image,
            tfm1_weight=0.5,
            tfm2_weight=0.5,
            mode="add",
        )

        # Verify result
        assert interpolated_tfm is not None, "Interpolated transform is None"
        assert isinstance(interpolated_tfm, itk.DisplacementFieldTransform), (
            "Should be a DisplacementFieldTransform"
        )

        print("Transform interpolation complete")
        print("  Interpolation alpha: 0.5")
        print(f"  Result type: {type(interpolated_tfm).__name__}")

    def test_combine_displacement_field_transforms(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
    ) -> None:
        """Test composing two transforms with various weights."""
        forward_transform = test_transforms["forward_transform"]
        fixed_image = test_images[0]

        # Create an identity transform as second transform
        identity_tfm = itk.AffineTransform[itk.D, 3].New()
        identity_tfm.SetIdentity()

        print("\nComposing transforms...")

        # Test 1: Equal weights (should be similar to interpolation at 0.5)
        print("  Test 1: Equal weights (0.5, 0.5)")
        composed_tfm1 = transform_tools.combine_displacement_field_transforms(
            forward_transform,
            identity_tfm,
            fixed_image,
            tfm1_weight=0.5,
            tfm2_weight=0.5,
            mode="add",
        )

        # Verify result
        assert composed_tfm1 is not None, "Composed transform is None"
        assert isinstance(composed_tfm1, itk.DisplacementFieldTransform), (
            "Should be a DisplacementFieldTransform"
        )

        # Test 2: First transform only (weight 1.0, 0.0)
        print("  Test 2: First transform only (1.0, 0.0)")
        composed_tfm2 = transform_tools.combine_displacement_field_transforms(
            forward_transform,
            identity_tfm,
            fixed_image,
            tfm1_weight=1.0,
            tfm2_weight=0.0,
            mode="add",
        )

        assert composed_tfm2 is not None, "Composed transform is None"

        # Test 3: Second transform only (weight 0.0, 1.0)
        print("  Test 3: Second transform only (0.0, 1.0)")
        composed_tfm3 = transform_tools.combine_displacement_field_transforms(
            forward_transform,
            identity_tfm,
            fixed_image,
            tfm1_weight=0.0,
            tfm2_weight=1.0,
            mode="add",
        )

        assert composed_tfm3 is not None, "Composed transform is None"

        # Test 4: Custom weights
        print("  Test 4: Custom weights (0.75, 0.25)")
        composed_tfm4 = transform_tools.combine_displacement_field_transforms(
            forward_transform,
            identity_tfm,
            fixed_image,
            tfm1_weight=0.75,
            tfm2_weight=0.25,
            mode="add",
        )

        assert composed_tfm4 is not None, "Composed transform is None"

        # Test 5: With blur sigma
        print("  Test 5: With blur sigma (1.0, 1.0)")
        composed_tfm5 = transform_tools.combine_displacement_field_transforms(
            forward_transform,
            identity_tfm,
            fixed_image,
            tfm1_weight=0.5,
            tfm2_weight=0.5,
            tfm1_blur_sigma=1.0,
            tfm2_blur_sigma=1.0,
            mode="add",
        )

        assert composed_tfm5 is not None, "Composed transform with blur is None"

        # Verify that different weights produce different results
        field1 = composed_tfm1.GetDisplacementField()
        field2 = composed_tfm2.GetDisplacementField()
        field3 = composed_tfm3.GetDisplacementField()

        arr1 = itk.array_from_image(field1)
        arr2 = itk.array_from_image(field2)
        arr3 = itk.array_from_image(field3)

        # field2 (1.0, 0.0) should be different from field3 (0.0, 1.0)
        diff_2_3 = np.mean(np.abs(arr2 - arr3))

        # field1 (0.5, 0.5) should be between field2 and field3
        # Check that field1 magnitude is between field2 and field3 magnitudes
        mag1 = np.mean(np.linalg.norm(arr1, axis=-1))
        mag2 = np.mean(np.linalg.norm(arr2, axis=-1))
        mag3 = np.mean(np.linalg.norm(arr3, axis=-1))

        print("Transform composition complete")
        print(f"  Field magnitude (0.5, 0.5): {mag1:.3f} mm")
        print(f"  Field magnitude (1.0, 0.0): {mag2:.3f} mm")
        print(f"  Field magnitude (0.0, 1.0): {mag3:.3f} mm")
        print(f"  Difference between (1.0,0.0) and (0.0,1.0): {diff_2_3:.3f} mm")

        # The difference should be non-zero since forward_transform is not identity
        assert diff_2_3 > 0, "Different weights should produce different results"

    def test_smooth_transform(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
    ) -> None:
        """Test smoothing a transform."""
        forward_transform = test_transforms["forward_transform"]
        fixed_image = test_images[0]

        print("\nSmoothing transform...")

        # Smooth the transform
        smoothed_tfm = transform_tools.smooth_transform(
            forward_transform, sigma=2.0, reference_image=fixed_image
        )

        # Verify result
        assert smoothed_tfm is not None, "Smoothed transform is None"
        assert isinstance(smoothed_tfm, itk.DisplacementFieldTransform), (
            "Should be a DisplacementFieldTransform"
        )

        print("Transform smoothing complete")
        print("  Smoothing sigma: 2.0")

    def test_combine_transforms_with_masks(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
    ) -> None:
        """Test combining transforms with spatial masks."""
        forward_transform = test_transforms["forward_transform"]
        fixed_image = test_images[0]

        # Create identity transform
        identity_tfm = itk.AffineTransform[itk.D, 3].New()
        identity_tfm.SetIdentity()

        # Create simple masks
        img_size = itk.size(fixed_image)
        img_size_tuple = (img_size[0], img_size[1], img_size[2])
        mask1_arr = np.zeros(img_size_tuple[::-1], dtype=np.uint8)
        mask2_arr = np.zeros(img_size_tuple[::-1], dtype=np.uint8)

        # Split image in half
        mask1_arr[:, :, : img_size_tuple[0] // 2] = 1
        mask2_arr[:, :, img_size_tuple[0] // 2 :] = 1

        mask1 = itk.image_from_array(mask1_arr)
        mask1.CopyInformation(fixed_image)

        mask2 = itk.image_from_array(mask2_arr)
        mask2.CopyInformation(fixed_image)

        print("\nCombining transforms with masks...")
        print(f"  Mask 1 voxels: {np.sum(mask1_arr)}")
        print(f"  Mask 2 voxels: {np.sum(mask2_arr)}")

        # Combine transforms
        combined_tfm = transform_tools.combine_transforms_with_masks(
            forward_transform, identity_tfm, mask1, mask2, fixed_image
        )

        # Verify result
        assert combined_tfm is not None, "Combined transform is None"
        assert isinstance(combined_tfm, itk.DisplacementFieldTransform), (
            "Should be a DisplacementFieldTransform"
        )

        print("Transforms combined with masks")

    def test_multiple_transform_applications(
        self,
        transform_tools: TransformTools,
        test_transforms: dict[str, Any],
        test_images: list[Any],
    ) -> None:
        """Test applying multiple transforms in sequence."""
        moving_image = test_images[1]
        fixed_image = test_images[0]
        forward_transform = test_transforms["forward_transform"]

        print("\nApplying transforms multiple times...")

        # Apply transform once
        result1 = transform_tools.transform_image(
            moving_image, forward_transform, fixed_image, interpolation_method="linear"
        )

        # Apply transform again (should work even though it's already transformed)
        result2 = transform_tools.transform_image(
            result1, forward_transform, fixed_image, interpolation_method="linear"
        )

        assert result1 is not None, "First transform result is None"
        assert result2 is not None, "Second transform result is None"

        print("Multiple sequential transforms applied")

    def test_identity_transform(
        self, transform_tools: TransformTools, test_images: list[Any]
    ) -> None:
        """Test that identity transform doesn't change the image."""
        moving_image = test_images[1]
        fixed_image = test_images[0]

        # Create identity transform
        identity_tfm = itk.AffineTransform[itk.D, 3].New()
        identity_tfm.SetIdentity()

        print("\nTesting identity transform...")

        transformed_image = transform_tools.transform_image(
            moving_image, identity_tfm, fixed_image, interpolation_method="linear"
        )

        # Images should be very similar (small differences due to interpolation)
        transformed_arr = itk.array_from_image(transformed_image)

        # Resample moving to fixed grid first for fair comparison
        resampler = itk.ResampleImageFilter.New(
            Input=moving_image, UseReferenceImage=True, ReferenceImage=fixed_image
        )
        resampler.Update()
        resampled_moving = resampler.GetOutput()
        resampled_arr = itk.array_from_image(resampled_moving)

        diff = np.abs(resampled_arr - transformed_arr)
        mean_diff = np.mean(diff)

        print("Identity transform tested")
        print(f"  Mean difference: {mean_diff:.4f}")

        # Should be very small (just interpolation error)
        assert mean_diff < 10.0, "Identity transform changed image too much"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
