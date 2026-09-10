# %% [markdown]
# # Convert VTK to VTP Surface Files
#
# This notebook reads VTK files from the current directory, extracts their surfaces using PyVista, and saves them as VTP files with "_surface.vtp" appended to the filename.
#

# %%
import os
from glob import glob
from pathlib import Path

import pyvista as pv

_HERE = Path(__file__).parent

# %%
# Get the current directory and find all .vtk files
input_dir = _HERE / "../../data/KCL-Heart-Model/input_meshes"
vtk_files = sorted(glob("??.vtk", root_dir=input_dir))

print(f"Found {len(vtk_files)} VTK files:")
for vtk_file in vtk_files:
    print(f"  - {vtk_file}")

# %%
output_dir = _HERE / "kcl-heart-model/surfaces"
os.makedirs(output_dir, exist_ok=True)
# Process each VTK file
for vtk_file in vtk_files:
    try:
        # Read the VTK file
        print(f"\nProcessing {vtk_file}...")
        mesh = pv.read(input_dir / vtk_file)

        # Extract the surface
        surface = mesh.extract_surface(algorithm="dataset_surface")

        # Generate the output filename
        base_name = Path(vtk_file).stem
        output_file = f"{base_name}.vtp"

        # Save the surface as VTP
        surface.save(output_dir / output_file)

        print(f"  Saved {output_file}")
        print(f"    Original mesh: {mesh.n_cells} cells, {mesh.n_points} points")
        print(f"    Surface mesh: {surface.n_cells} cells, {surface.n_points} points")

    except Exception as e:
        print(f"  Error processing {vtk_file}: {e!s}")

print("\n" + "=" * 50)
print("Processing complete!")

# %%
# Process average_mesh.vtk provided from the KCL data collection.

mesh = pv.read(_HERE / "../../data/KCL-Heart-Model/average_mesh.vtk")

# Extract the surface
surface = mesh.extract_surface(algorithm="dataset_surface")

# Save the surface as VTP
surface.save(f"{output_dir}/../average_surface.vtp")

# %%
# Optional: Visualize one of the surfaces to verify the result
if len(vtk_files) > 0:
    # Load the first surface file
    first_surface = pv.read(f"{output_dir}/{Path(vtk_files[0]).stem}.vtp")

    # Create a plotter and display the surface
    plotter = pv.Plotter()
    plotter.add_mesh(first_surface, color="lightblue", show_edges=True)
    plotter.add_axes()
    plotter.show()
