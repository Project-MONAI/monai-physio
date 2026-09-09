# %%
from pathlib import Path

import itk
import numpy as np
import pyvista as pv

from monai_physio.contour_tools import ContourTools

_HERE = Path(__file__).parent

# %%
# Define directories
correspond_dir = _HERE / "kcl-heart-model/surfaces_aligned_corresponded"
surfaces_dir = _HERE / "kcl-heart-model/surfaces_aligned"
output_dir = _HERE / "kcl-heart-model/pca_inputs"

# Create output directory
output_dir.mkdir(exist_ok=True)

contour_tools = ContourTools()

template_mesh = pv.read(_HERE / "kcl-heart-model/average_surface.vtp")

# Find all VTK/VTP files in correspondence directory
tfm_filenames = sorted(correspond_dir.glob("??.fixed_to_moving_transform.hdf"))
# Since we are transforming points, we use the forward transform to effect an inverse transform

print(f"Found {len(tfm_filenames)} files in {correspond_dir}/")
for f in tfm_filenames:
    print(f"  {f.name}")

# %%
for tfm_fname in tfm_filenames:
    tfm = itk.transformread(tfm_fname)[0]
    correspond_mesh = contour_tools.transform_contours(
        template_mesh,
        tfm,
        with_deformation_magnitude=True,
    )
    correspond_mesh.save(output_dir / f"{tfm_fname.stem[:2]}.vtp")

# %% [markdown]
# ## Visualize Results: Before and After Correspondence
#
# Compare original surfaces with corresponded surfaces (after applying inverse transforms)

# %%
# Load processed models for visualization
processed_models = {}

# Get list of case IDs from output files
output_files = sorted(output_dir.glob("*.vtp"))

for output_file in output_files:
    # Extract case ID (e.g., "01" from "01.vtp")
    case_id = output_file.stem

    # Find corresponding original surface file
    original_file = surfaces_dir / f"{case_id}.vtp"

    if original_file.exists():
        processed_models[case_id] = {
            "before": pv.read(original_file),
            "after": pv.read(output_file),
        }

print(f"Loaded {len(processed_models)} models for visualization")

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

    # Left: Before correspondence (original surface)
    plotter.subplot(0, 0)
    plotter.add_mesh(
        before_mesh, color="lightblue", opacity=0.5, label=f"Case {case_id} (Original)"
    )
    plotter.add_mesh(template_mesh, color="red", opacity=1.0, label="Template")
    plotter.add_text(
        f"Before Correspondence\nCase {case_id}", position="upper_left", font_size=10
    )
    plotter.add_legend()
    plotter.show_axes()
    plotter.camera_position = "iso"

    # Right: After correspondence (with inverse transform applied)
    plotter.subplot(0, 1)
    plotter.add_mesh(before_mesh, color="lightblue", opacity=0.5, label="Template")
    plotter.add_mesh(
        after_mesh, color="green", opacity=1.0, label=f"Case {case_id} (Corresponded)"
    )
    plotter.add_text(
        f"After Correspondence\nCase {case_id}", position="upper_left", font_size=10
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
# Show the amount of deformation applied during correspondence

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
