# %% [markdown]
# # Registration-Based Correspondence
#
# This notebook aligns ICP-aligned models from step 2 to the average surface using **Greedy affine + ICON deformable** registration via mask-based registration.
#
# **Workflow:**
# 1. Load ICP-aligned models from `kcl-heart-model/surfaces_aligned/`
# 2. Load average surface (`average_surface.vtp`)
# 3. Use `RegisterModelsDistanceMaps` to perform Greedy affine + ICON deformable registration
# 4. Save corresponded models to `kcl-heart-model/surfaces_aligned_corresponded/`
# 5. Visualize before/after comparisons
# 6. Analyze deformation magnitude and registration statistics
#
# **Method:**
# - **Greedy** performs fast CPU-based affine pre-alignment
# - **ICON** provides deep learning deformable registration on the affine-pre-aligned masks
# - Mask-based approach focuses registration on the anatomical structures

# %%
import itk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

from pathlib import Path

from monai_physio.contour_tools import ContourTools
from monai_physio.register_models_distance_maps import RegisterModelsDistanceMaps

_HERE = Path(__file__).parent

# Initialize ContourTools
contour_tools = ContourTools()

# Setup paths
input_dir = _HERE / "kcl-heart-model/surfaces_aligned"
output_dir = _HERE / "kcl-heart-model/surfaces_aligned_corresponded"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"Input directory exists: {input_dir.exists()}")

# %%
# Load average surface (fixed model)
average_surface_path = _HERE / "kcl-heart-model/average_surface.vtp"
if not average_surface_path.exists():
    raise FileNotFoundError(f"Average surface not found: {average_surface_path}")

fixed_model = pv.read(average_surface_path)
print(f"Loaded average surface: {average_surface_path}")
print(f"  Points: {fixed_model.n_points}")
print(f"  Cells: {fixed_model.n_cells}")
print(f"  Bounds: {fixed_model.bounds}")

# %%
# Create reference image from average surface
# This provides the coordinate frame for mask generation
reference_image = contour_tools.create_reference_image(
    mesh=fixed_model,
    spatial_resolution=1.0,  # 1mm isotropic resolution
    buffer_factor=0.25,  # 25% buffer around mesh bounds
    ptype=itk.UC,  # Unsigned char for masks
)

# Display reference image properties
origin = reference_image.GetOrigin()
spacing = reference_image.GetSpacing()
size = reference_image.GetLargestPossibleRegion().GetSize()

print("Created reference image:")
print(f"  Origin: ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}) mm")
print(f"  Spacing: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm")
print(f"  Size: ({size[0]}, {size[1]}, {size[2]}) voxels")
print(
    f"  Physical size: ({size[0] * spacing[0]:.1f}, {size[1] * spacing[1]:.1f}, {size[2] * spacing[2]:.1f}) mm"
)

# %%
# Get list of ICP-aligned VTK files
vtk_files = sorted(input_dir.glob("??.vtp"))
print(f"Found {len(vtk_files)} ICP-aligned models:")
for f in vtk_files:
    print(f"  {f.name}")

# %%
# Process each model
registration_stats = []

for vtk_file in vtk_files:
    case_id = vtk_file.stem
    print(f"\n{'=' * 60}")
    print(f"Processing: {case_id}")
    print(f"{'=' * 60}")

    # Load moving model
    moving_model = pv.read(vtk_file)
    print(f"Loaded moving model: {vtk_file.name}")
    print(f"  Points: {moving_model.n_points}")

    # Initialize registrar with mask-based registration
    registrar = RegisterModelsDistanceMaps(
        moving_model=moving_model,
        fixed_model=fixed_model,
        reference_image=reference_image,
        mask_dilation_mm=20.0,  # Dilation for binary registration mask
    )

    # Perform Greedy affine + ICON deformable registration
    result = registrar.register(
        transform_type="Deformable",
    )

    forward_transform = result["forward_transform"]
    inverse_transform = result["inverse_transform"]

    # Get registered model
    registered_model = result["registered_model"]

    # Save registered model
    output_file = output_dir / f"{case_id}.vtp"
    registered_model.save(output_file)
    print(f"Saved: {output_file.name}")

    itk.transformwrite(
        forward_transform, output_dir / f"{case_id}.forward_transform.hdf"
    )
    itk.transformwrite(
        inverse_transform, output_dir / f"{case_id}.inverse_transform.hdf"
    )

    # Calculate registration statistics
    if "DeformationMagnitude" in registered_model.array_names:
        deformation = registered_model["DeformationMagnitude"]
        stats = {
            "Case ID": case_id,
            "Mean Deformation (mm)": np.mean(deformation),
            "Max Deformation (mm)": np.max(deformation),
            "Min Deformation (mm)": np.min(deformation),
            "Std Deformation (mm)": np.std(deformation),
        }
        registration_stats.append(stats)
        print(f"  Mean deformation: {stats['Mean Deformation (mm)']:.2f} mm")
        print(f"  Max deformation: {stats['Max Deformation (mm)']:.2f} mm")

print(f"\n{'=' * 60}")
print("Processing complete!")
print(f"{'=' * 60}")

# Store processed models for visualization
processed_models = {}
for vtk_file in vtk_files:
    case_id = vtk_file.stem  # e.g., "01", "02", etc.
    output_file = output_dir / f"{case_id}.vtp"
    if output_file.exists():
        processed_models[case_id] = {
            "before": pv.read(vtk_file),
            "after": pv.read(output_file),
        }

print(f"Loaded {len(processed_models)} models for visualization")

# %% [markdown]
# ## Visualize Results: Before and After Registration
#
# Compare ICP-aligned models (before) with distance map registered models (after)

# %%
# Select a few examples to visualize (e.g., cases 01, 05, 10, 15, 20)
example_ids = ["01", "05", "10", "15", "20"]

for case_id in example_ids:
    if case_id not in processed_models:
        print(f"Skipping Case {case_id} - not found in processed models")
        continue

    before_mesh = processed_models[case_id]["before"]
    after_mesh = processed_models[case_id]["after"]

    # Create side-by-side comparison
    plotter = pv.Plotter(shape=(1, 2))

    # Left: Before distance map registration (ICP-aligned only)
    plotter.subplot(0, 0)
    plotter.add_mesh(
        fixed_model, color="lightblue", opacity=1.0, label="Average Surface"
    )
    plotter.add_mesh(
        before_mesh, color="red", opacity=1.0, label=f"Case {case_id} (ICP-aligned)"
    )
    plotter.add_text(
        f"Before Distance Map Registration\nCase {case_id}",
        position="upper_left",
        font_size=10,
    )
    plotter.add_legend()
    plotter.show_axes()
    plotter.camera_position = "iso"

    # Right: After distance map registration (Greedy affine + ICON deformable)
    plotter.subplot(0, 1)
    plotter.add_mesh(
        fixed_model, color="lightblue", opacity=1.0, label="Average Surface"
    )
    plotter.add_mesh(
        after_mesh, color="green", opacity=1.0, label=f"Case {case_id} (Corresponded)"
    )
    plotter.add_text(
        f"After Distance Map Registration (Greedy + ICON)\nCase {case_id}",
        position="upper_left",
        font_size=10,
    )
    plotter.add_legend()
    plotter.show_axes()
    plotter.camera_position = "iso"

    # Link the camera views so they rotate together
    plotter.link_views()
    plotter.show()

# %% [markdown]
# ## Visualize Deformation Magnitude
#
# Show the amount of deformation applied during distance map registration

# %%
# Visualize deformation magnitude for selected examples
example_ids = ["01", "05", "10", "15", "20"]

for case_id in example_ids:
    if case_id not in processed_models:
        continue

    after_mesh = processed_models[case_id]["after"]

    # Check if deformation magnitude is available
    if "DeformationMagnitude" not in after_mesh.array_names:
        print(f"No deformation magnitude data for Case {case_id}")
        continue

    # Create plotter
    plotter = pv.Plotter()

    # Add mesh colored by deformation magnitude
    plotter.add_mesh(
        after_mesh,
        scalars="DeformationMagnitude",
        cmap="jet",
        clim=[0, 10],  # Adjust based on your data
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Deformation (mm)",
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.1,
        },
    )

    # Calculate statistics
    deformation = after_mesh["DeformationMagnitude"]
    mean_def = np.mean(deformation)
    max_def = np.max(deformation)

    plotter.add_text(
        f"Deformation Magnitude - Case {case_id}\n"
        f"Mean: {mean_def:.2f} mm, Max: {max_def:.2f} mm",
        position="upper_left",
        font_size=10,
    )

    plotter.show_axes()
    plotter.camera_position = "iso"
    plotter.show()

# %%
# Save registration statistics
if registration_stats:
    stats_df = pd.DataFrame(registration_stats)
    stats_file = output_dir / "registration_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\nSaved registration statistics: {stats_file}")
    print("\nSummary:")
    print(stats_df.to_string(index=False))
else:
    print("\nNo registration statistics available.")

# %%
# Visualize registration statistics
if registration_stats:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Mean deformation
    axes[0].bar(
        stats_df["Case ID"], stats_df["Mean Deformation (mm)"], color="steelblue"
    )
    axes[0].set_xlabel("Case ID")
    axes[0].set_ylabel("Mean Deformation (mm)")
    axes[0].set_title("Mean Deformation per Case (After Distance Map Registration)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)

    # Plot 2: Max deformation
    axes[1].bar(stats_df["Case ID"], stats_df["Max Deformation (mm)"], color="coral")
    axes[1].set_xlabel("Case ID")
    axes[1].set_ylabel("Max Deformation (mm)")
    axes[1].set_title("Maximum Deformation per Case")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / "registration_statistics.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_file}")
    plt.show()
else:
    print("\nNo statistics to plot.")

# %% [markdown]
# ## Summary
#
# This notebook performed mask-based deformable registration using **Greedy affine + ICON deformable** to establish correspondence between the ICP-aligned models and the average surface.
#
# **Next Steps:**
# - Proceed to step 4: `4-surfaces_aligned_correspond_to_pca_inputs.ipynb` to prepare data for PCA analysis
# - The corresponded models in `kcl-heart-model/surfaces_aligned_corresponded/` now have improved point-to-point correspondence
# - The registration statistics show the deformation applied to each model
#
# **Registration Details:**
# - The `RegisterModelsDistanceMaps` class uses a two-stage pipeline:
#   1. **Greedy affine** registration (fast CPU-based alignment)
#   2. **ICON deformable** registration on the affine-pre-aligned masks (deep learning)
# - The `mask_dilation_mm` parameter controls the dilation of the binary registration mask (default 20mm)
# - Composed Greedy + ICON transforms provide smooth, invertible deformation fields for anatomical correspondence
