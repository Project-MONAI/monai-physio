# %% [markdown]
# # ICP Affine Registration: Align Heart Models to Average
#
# This script performs ICP (Iterative Closest Point) affine registration to align each individual heart model to the average model.
#
# **Workflow:**
# 1. Load the average mesh (`input_meshes/average.vtk`)
# 2. Load each individual mesh (`input_meshes/01.vtk` through `20.vtk`)
# 3. Use ICP affine registration to align each mesh to the average
# 4. Save the aligned meshes to `icp_aligned_meshes/`
# 5. Visualize the results
#

# %%
import itk

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

from monai_physio.contour_tools import ContourTools
from monai_physio.register_models_icp import RegisterModelsICP

_HERE = Path(__file__).parent

# %% [markdown]
# ## 1. Load the Average Mesh (Target)
#

# %%
# Load the average mesh - this will be our fixed target
template_mesh_path = _HERE / "kcl-heart-model/average_surface.vtp"
template_mesh = pv.read(template_mesh_path)

print("Average mesh loaded:")
print(f"  Points: {template_mesh.n_points}")
print(f"  Cells: {template_mesh.n_cells}")
print(f"  Center: {template_mesh.center}")
print(f"  Bounds: {template_mesh.bounds}")

# %% [markdown]
# ## 2. Find All Individual Mesh Files
#

# %%
# Get all individual mesh files (excluding average.vtk)
input_meshes_dir = _HERE / "kcl-heart-model/surfaces"
mesh_files = sorted([f for f in input_meshes_dir.glob("??.vtp")])

if not mesh_files:
    raise FileNotFoundError(
        f"No mesh files matching '??.vtp' found in {input_meshes_dir}. "
        "Ensure the kcl-heart-model/surfaces/ directory contains the input meshes."
    )

print(f"Found {len(mesh_files)} individual mesh files:")
for mesh_file in mesh_files:
    print(f"  {mesh_file.name}")

# %% [markdown]
# ## 3. Perform ICP Registration for Each Mesh
#

# %%
# Create output directory for aligned meshes
output_dir = _HERE / "kcl-heart-model/surfaces_aligned"
output_dir.mkdir(exist_ok=True)

# Store results
aligned_meshes = {}
transforms_point_forward = {}  # Moving to Fixed point transforms (moving_to_fixed_transform)
transforms_point_inverse = {}  # Fixed to Moving point transforms (fixed_to_moving_transform)

contour_tools = ContourTools()

# Process each mesh
for mesh_file in mesh_files:
    print(f"\n{'=' * 60}")
    print(f"Processing: {mesh_file.name}")
    print(f"{'=' * 60}")

    # Load the moving mesh
    moving_mesh = pv.read(mesh_file)
    print(f"  Loaded mesh: {moving_mesh.n_points} points")

    # Extract surface if needed (in case it's a volume mesh)
    if isinstance(moving_mesh, pv.UnstructuredGrid):
        print("  Extracting surface from volume mesh...")
        moving_mesh = moving_mesh.extract_surface(algorithm="dataset_surface")
        print(f"  Surface mesh: {moving_mesh.n_points} points")

    registrar = RegisterModelsICP(fixed_model=template_mesh)

    # Perform affine ICP registration to align each mesh to the template
    result = registrar.register(
        moving_model=moving_mesh, transform_type="Affine", max_iterations=2000
    )

    # Store results
    mesh_id = mesh_file.stem
    aligned_meshes[mesh_id] = result["registered_model"]
    transforms_point_forward[mesh_id] = result["moving_to_fixed_transform"]
    transforms_point_inverse[mesh_id] = result["fixed_to_moving_transform"]

    # Save aligned mesh
    output_path = output_dir / f"{mesh_id}.vtp"
    result["registered_model"].save(output_path)
    itk.transformwrite(
        result["moving_to_fixed_transform"],
        output_dir / f"{mesh_id}_moving_to_fixed_transform.hdf",
    )
    itk.transformwrite(
        result["fixed_to_moving_transform"],
        output_dir / f"{mesh_id}_fixed_to_moving_transform.hdf",
    )
    print(f"\n  Saved aligned mesh: {output_path}")

print(f"\n{'=' * 60}")
print(f"ICP registration complete for all {len(mesh_files)} meshes!")
print(f"{'=' * 60}")

# %% [markdown]
# ## 4. Visualize Results: Before and After Registration
#

# %%
# Select a few examples to visualize (e.g., 01, 05, 10, 15, 20)
example_ids = ["01", "05", "10", "15", "20"]

for mesh_id in example_ids:
    if mesh_id not in aligned_meshes:
        continue

    # Load original mesh
    original_mesh = pv.read(_HERE / f"kcl-heart-model/surfaces/{mesh_id}.vtp")
    if isinstance(original_mesh, pv.UnstructuredGrid):
        original_mesh = original_mesh.extract_surface(algorithm="dataset_surface")

    # Create side-by-side comparison
    plotter = pv.Plotter(shape=(1, 2))

    # Left: Before registration
    plotter.subplot(0, 0)
    plotter.add_mesh(template_mesh, color="lightblue", opacity=1.0, label="Average")
    plotter.add_mesh(
        original_mesh, color="red", opacity=1.0, label=f"Original {mesh_id}"
    )
    plotter.add_text(
        f"Before ICP Registration - {mesh_id}", position="upper_left", font_size=10
    )
    plotter.add_legend()
    plotter.show_axes()

    # Right: After registration
    plotter.subplot(0, 1)
    plotter.add_mesh(template_mesh, color="lightblue", opacity=1.0, label="Average")
    plotter.add_mesh(
        aligned_meshes[mesh_id], color="green", opacity=1.0, label=f"Aligned {mesh_id}"
    )
    plotter.add_text(
        f"After ICP Registration - {mesh_id}", position="upper_left", font_size=10
    )
    plotter.add_legend()
    plotter.show_axes()

    plotter.link_views()
    plotter.show()

# %% [markdown]
# ## 6. Calculate Registration Statistics
#

# %%
# Calculate statistics for each registration
stats_data = []

for mesh_id, aligned_mesh in aligned_meshes.items():
    # Calculate distance from aligned mesh to average mesh
    # Using point-to-point distance as a metric

    # Get closest points on average mesh for each point in aligned mesh
    closest_points = template_mesh.find_closest_cell(
        aligned_mesh.points, return_closest_point=True
    )[1]

    # Calculate distances
    distances = np.linalg.norm(aligned_mesh.points - closest_points, axis=1)

    stats_data.append(
        {
            "Mesh ID": mesh_id,
            "Mean Distance (mm)": np.mean(distances),
            "Median Distance (mm)": np.median(distances),
            "Std Distance (mm)": np.std(distances),
            "Max Distance (mm)": np.max(distances),
            "Min Distance (mm)": np.min(distances),
        }
    )

# Create DataFrame and display
stats_df = pd.DataFrame(stats_data)
stats_df = stats_df.sort_values("Mesh ID")

print("\nRegistration Statistics (Distance from aligned mesh to average mesh):")
print("=" * 80)
print(stats_df.to_string(index=False))
print("=" * 80)

# Summary statistics
print("\nOverall Summary:")
print(f"  Average mean distance: {stats_df['Mean Distance (mm)'].mean():.3f} mm")
print(f"  Average median distance: {stats_df['Median Distance (mm)'].mean():.3f} mm")
print(
    f"  Range of mean distances: {stats_df['Mean Distance (mm)'].min():.3f} - {stats_df['Mean Distance (mm)'].max():.3f} mm"
)

# %% [markdown]
# ## 7. Save Registration Statistics
#

# %%
# Save statistics to CSV
stats_csv_path = output_dir / "registration_statistics.csv"
stats_df.to_csv(stats_csv_path, index=False)
print(f"\nStatistics saved to: {stats_csv_path}")

# %% [markdown]
# ## 8. Visualize Distance Distributions
#

# %%
# Create bar plot of mean distances
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Mean distances
axes[0].bar(stats_df["Mesh ID"], stats_df["Mean Distance (mm)"], color="steelblue")
axes[0].set_xlabel("Mesh ID")
axes[0].set_ylabel("Mean Distance (mm)")
axes[0].set_title(
    "Mean Distance from Aligned Mesh to Average Mesh (After ICP Registration)"
)
axes[0].grid(axis="y", alpha=0.3)

# Plot 2: Box plot style visualization
axes[1].errorbar(
    stats_df["Mesh ID"],
    stats_df["Median Distance (mm)"],
    yerr=stats_df["Std Distance (mm)"],
    fmt="o",
    capsize=5,
    capthick=2,
    color="coral",
    ecolor="gray",
    label="Median +/- Std",
)
axes[1].set_xlabel("Mesh ID")
axes[1].set_ylabel("Distance (mm)")
axes[1].set_title("Median Distance +/- Standard Deviation")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "registration_statistics.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nPlot saved to: {output_dir / 'registration_statistics.png'}")
