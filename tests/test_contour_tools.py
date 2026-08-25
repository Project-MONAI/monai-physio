"""
Test for contour tools functionality.

This test depends on test_segment_chest_total_segmentator and uses the
segmentation results to test contour extraction and manipulation.
"""

from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pytest
import pyvista as pv

from physiotwin4d.contour_tools import ContourTools


@pytest.mark.slow
class TestContourTools:
    """Test suite for ContourTools functionality."""

    def test_contour_tools_initialization(self, contour_tools: ContourTools) -> None:
        """Test that ContourTools initializes correctly."""
        assert contour_tools is not None, "ContourTools not initialized"
        print("\nContourTools initialized successfully")

    def test_extract_contours_from_heart_mask(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> None:
        """Test extracting contours from heart mask."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"
        contour_output_dir.mkdir(exist_ok=True)

        # Extract contours from heart mask of first time point
        heart_mask = test_labelmaps[0]["heart"]

        print("\nExtracting contours from heart mask...")
        print(f"  Heart mask size: {itk.size(heart_mask)}")

        contours = contour_tools.extract_contours(heart_mask)

        # Verify contours
        assert contours is not None, "Contours not extracted"
        assert isinstance(contours, pv.PolyData), "Contours should be PyVista PolyData"
        assert contours.n_points > 0, "Contours should have points"

        print("Heart contours extracted successfully")
        print(f"  Number of points: {contours.n_points}")
        print(f"  Number of cells: {contours.n_cells}")

        # Save contours
        output_file = contour_output_dir / "heart_contours.vtp"
        contours.save(str(output_file))
        print(f"  Saved to: {output_file}")

    def test_extract_contours_from_lung_mask(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> None:
        """Test extracting contours from lung mask."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"
        contour_output_dir.mkdir(exist_ok=True)

        # Extract contours from lung mask
        lung_mask = test_labelmaps[0]["lung"]

        print("\nExtracting contours from lung mask...")
        contours = contour_tools.extract_contours(lung_mask)

        assert contours is not None, "Lung contours not extracted"
        assert contours.n_points > 0, "Lung contours should have points"

        print("Lung contours extracted successfully")
        print(f"  Number of points: {contours.n_points}")
        print(f"  Number of cells: {contours.n_cells}")

        # Save contours
        output_file = contour_output_dir / "lung_contours.vtp"
        contours.save(str(output_file))
        print(f"  Saved to: {output_file}")

    def test_extract_contours_multiple_anatomy(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> None:
        """Test extracting contours from multiple anatomical structures."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"
        contour_output_dir.mkdir(exist_ok=True)

        # Test on multiple anatomy groups
        anatomy_groups = ["lung", "heart", "bone"]
        contours_dict: dict[str, pv.PolyData] = {}

        for group in anatomy_groups:
            mask = test_labelmaps[0][group]

            # Check if mask has any foreground voxels
            mask_arr = itk.array_from_image(mask)
            if np.sum(mask_arr > 0) > 100:  # Only extract if sufficient voxels
                print(f"\nExtracting contours from {group} mask...")
                contours = contour_tools.extract_contours(mask)
                contours_dict[group] = contours

                print(
                    f"  {group}: {contours.n_points} points, {contours.n_cells} cells"
                )

                # Save contours
                output_file = contour_output_dir / f"{group}_contours_slice000.vtp"
                contours.save(str(output_file))

        assert len(contours_dict) > 0, (
            "Should extract contours from at least one anatomy group"
        )
        print(f"\nExtracted contours from {len(contours_dict)} anatomy groups")

    def test_create_mask_from_mesh(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test creating a mask from extracted mesh."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"
        contour_output_dir.mkdir(exist_ok=True)

        # Extract contours from heart mask
        heart_mask = test_labelmaps[0]["heart"]
        contours = contour_tools.extract_contours(heart_mask)

        # Create mask from the extracted mesh
        print("\nCreating mask from extracted heart contours...")
        reference_image = test_images[0]
        recreated_mask = contour_tools.create_mask_from_mesh(contours, reference_image)

        # Verify recreated mask
        assert recreated_mask is not None, "Mask not created from mesh"
        assert itk.size(recreated_mask) == itk.size(reference_image), (
            "Mask size should match reference"
        )

        # Check that mask has foreground voxels
        mask_arr = itk.array_from_image(recreated_mask)
        num_foreground = np.sum(mask_arr > 0)
        assert num_foreground > 0, "Recreated mask should have foreground voxels"

        print("Mask created from mesh successfully")
        print(f"  Mask size: {itk.size(recreated_mask)}")
        print(f"  Foreground voxels: {num_foreground}")

        # Save recreated mask
        output_file = contour_output_dir / "heart_mask_from_contours.mha"
        itk.imwrite(recreated_mask, str(output_file), compression=True)
        print(f"  Saved to: {output_file}")

    def test_merge_meshes(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> None:
        """Test merging multiple meshes."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"
        contour_output_dir.mkdir(exist_ok=True)

        # Extract contours from multiple structures
        meshes = []
        anatomy_groups = ["lung", "heart"]

        for group in anatomy_groups:
            mask = test_labelmaps[0][group]
            mask_arr = itk.array_from_image(mask)

            if np.sum(mask_arr > 0) > 100:
                contours = contour_tools.extract_contours(mask)
                meshes.append(contours)

        if len(meshes) >= 2:
            print(f"\nMerging {len(meshes)} meshes...")
            merged_mesh, individual_meshes = contour_tools.merge_meshes(meshes)

            # Verify merged mesh
            assert merged_mesh is not None, "Merged mesh is None"
            assert isinstance(merged_mesh, pv.PolyData), (
                "Merged mesh should be PyVista PolyData"
            )
            assert merged_mesh.n_points > 0, "Merged mesh should have points"

            # Verify individual meshes were also transformed
            assert len(individual_meshes) == len(meshes), (
                "Should return same number of individual meshes"
            )

            print("Meshes merged successfully")
            print(f"  Merged mesh points: {merged_mesh.n_points}")
            print(f"  Merged mesh cells: {merged_mesh.n_cells}")

            # Save merged mesh
            output_file = contour_output_dir / "merged_contours.vtp"
            merged_mesh.save(str(output_file))
            print(f"  Saved to: {output_file}")
        else:
            pytest.skip("Not enough meshes with sufficient voxels to test merging")

    def test_transform_contours_identity(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> None:
        """Test transforming contours with identity transform."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"
        contour_output_dir.mkdir(exist_ok=True)

        # Extract contours
        heart_mask = test_labelmaps[0]["heart"]
        contours = contour_tools.extract_contours(heart_mask)

        # Create identity transform
        identity_tfm = itk.AffineTransform[itk.D, 3].New()
        identity_tfm.SetIdentity()

        print("\nTransforming contours with identity transform...")
        transformed_contours = contour_tools.transform_contours(
            contours, identity_tfm, with_deformation_magnitude=False
        )

        # Verify transformed contours
        assert transformed_contours is not None, "Transformed contours is None"
        assert isinstance(transformed_contours, pv.PolyData), (
            "Should be PyVista PolyData"
        )
        assert transformed_contours.n_points == contours.n_points, (
            "Should have same number of points"
        )

        # Points should be nearly identical for identity transform
        original_points = contours.points
        transformed_points = transformed_contours.points
        max_diff = np.max(np.abs(original_points - transformed_points))

        print("Contours transformed successfully")
        print(f"  Number of points: {transformed_contours.n_points}")
        print(f"  Max point difference (identity): {max_diff:.6f} mm")

        assert max_diff < 0.01, (
            "Identity transform should produce nearly identical points"
        )

    def test_transform_contours_with_deformation(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> None:
        """Test transforming contours with deformation magnitude calculation."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"
        contour_output_dir.mkdir(exist_ok=True)

        # Extract contours
        heart_mask = test_labelmaps[0]["heart"]
        contours = contour_tools.extract_contours(heart_mask)

        # Create a translation transform (10mm in each direction)
        translation_tfm = itk.TranslationTransform[itk.D, 3].New()
        translation_tfm.SetOffset([10.0, 10.0, 10.0])

        print("\nTransforming contours with translation (10mm)...")
        transformed_contours = contour_tools.transform_contours(
            contours, translation_tfm, with_deformation_magnitude=True
        )

        # Verify transformed contours
        assert transformed_contours is not None, "Transformed contours is None"

        # Check for DeformationMagnitude in point data
        if "DeformationMagnitude" in transformed_contours.point_data:
            deformation = transformed_contours["DeformationMagnitude"]
            mean_deformation = np.mean(deformation)
            expected_deformation = np.sqrt(10**2 + 10**2 + 10**2)  # ~17.32 mm

            print("Deformation magnitude calculated")
            print(f"  Mean deformation: {mean_deformation:.2f} mm")
            print(f"  Expected: {expected_deformation:.2f} mm")

            # Should be close to expected (within 1mm tolerance)
            assert abs(mean_deformation - expected_deformation) < 1.0, (
                f"Mean deformation {mean_deformation:.2f} should be close to {expected_deformation:.2f}"
            )

            # Save transformed contours with deformation
            output_file = contour_output_dir / "heart_contours_transformed.vtp"
            transformed_contours.save(str(output_file))
            print(f"  Saved to: {output_file}")
        else:
            print("  Note: DeformationMagnitude not in point data")

    def test_contours_from_both_time_points(
        self,
        contour_tools: ContourTools,
        test_labelmaps: list[dict[str, Any]],
        test_directories: dict[str, Path],
    ) -> None:
        """Test extracting contours from both time points."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"
        contour_output_dir.mkdir(exist_ok=True)

        print("\nExtracting heart contours from both time points...")

        for i, result in enumerate(test_labelmaps):
            heart_mask = result["heart"]
            contours = contour_tools.extract_contours(heart_mask)

            print(f"  Time point {i}: {contours.n_points} points")

            # Save contours
            output_file = contour_output_dir / f"heart_contours_slice{i:03d}.vtp"
            contours.save(str(output_file))

        print(f"Extracted contours from {len(test_labelmaps)} time points")


class TestSaveCombinedSurfaces:
    """Structure identity must survive merging into one file."""

    @staticmethod
    def _annotated_sphere(
        center: tuple[float, float, float], label_id: int
    ) -> pv.PolyData:
        surface = pv.Sphere(radius=1.0, center=center)
        surface.field_data["SegmentationLabelIds"] = np.array(
            [label_id], dtype=np.int32
        )
        return surface

    def test_per_cell_label_ids_from_single_label_surfaces(
        self, tmp_path: Path
    ) -> None:
        """Each merged cell carries the label ID of the surface it came from."""
        surfaces = {
            "atrium_left": self._annotated_sphere((0.0, 0.0, 0.0), 141),
            "ventricle_left": self._annotated_sphere((3.0, 0.0, 0.0), 142),
        }

        output_file = tmp_path / "combined.vtp"
        ContourTools.save_combined_surfaces(surfaces, str(output_file))

        merged = pv.read(str(output_file))
        label_ids = merged.cell_data["SegmentationLabelIds"]
        assert set(np.unique(label_ids)) == {141, 142}
        for name, surface in surfaces.items():
            expected = int(surface.field_data["SegmentationLabelIds"][0])
            assert int(np.sum(label_ids == expected)) == surface.n_cells, (
                f"Cell count for {name} did not survive the merge"
            )
        # Tagging must not leak back into the caller's surfaces.
        for surface in surfaces.values():
            assert "SegmentationLabelIds" not in surface.cell_data

    def test_multi_label_surfaces_tagged_unknown(self, tmp_path: Path) -> None:
        """Per-group surfaces list several labels, so no cell can be attributed."""
        group_surface = pv.Sphere(radius=1.0)
        group_surface.field_data["SegmentationLabelIds"] = np.array(
            [141, 142], dtype=np.int32
        )

        output_file = tmp_path / "combined_group.vtp"
        ContourTools.save_combined_surfaces({"heart": group_surface}, str(output_file))

        merged = pv.read(str(output_file))
        assert set(np.unique(merged.cell_data["SegmentationLabelIds"])) == {0}


class TestRemeshCarriesCellLabels:
    """Remeshing rebuilds the topology, so the cell labels have to be carried."""

    @staticmethod
    def _two_label_sphere() -> pv.PolyData:
        """A sphere whose cells carry two structure ids and their label pairs.

        Split by the sign of the cell centre's x, so both labels own a large,
        contiguous patch that uniform re-tiling cannot absorb.
        """
        surface = cast(
            pv.PolyData,
            pv.Sphere(
                radius=10.0, theta_resolution=40, phi_resolution=40
            ).triangulate(),
        )
        left = np.asarray(surface.cell_centers().points)[:, 0] < 0.0
        surface.cell_data["SegmentationLabelIds"] = np.where(left, 141, 142).astype(
            np.int32
        )
        surface.cell_data["boundary_labels"] = np.column_stack(
            (np.where(left, 141, 142), np.zeros(surface.n_cells))
        ).astype(np.int32)
        return surface

    def test_remeshing_carries_both_cell_arrays(
        self, contour_tools: ContourTools
    ) -> None:
        """Anatomy splitting downstream reads these off the remeshed surface."""
        original = self._two_label_sphere()

        remeshed = contour_tools.remesh_and_smooth_surface(
            original, surface_reduction_rate=0.5
        )

        assert remeshed.n_cells < original.n_cells
        for name in ("SegmentationLabelIds", "boundary_labels"):
            assert name in remeshed.cell_data, f"{name} did not survive remeshing"
            carried = np.asarray(remeshed.cell_data[name])
            assert len(carried) == remeshed.n_cells
            # Nearest-cell transfer, so every value is one the source carried
            # rather than an interpolation between two of them.
            source = np.asarray(original.cell_data[name])
            assert set(np.unique(carried).tolist()) <= set(np.unique(source).tolist())

        # Both structures survive: a transfer that collapsed onto one label
        # would still satisfy the membership check above.
        assert set(np.unique(remeshed.cell_data["SegmentationLabelIds"])) == {141, 142}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
