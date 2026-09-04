# %% [markdown]
# # Heart Segmentation using Simpleware Medical ASCardio
#
# This notebook demonstrates the use of the `SegmentHeartSimpleware` class to perform automated cardiac segmentation using Synopsys Simpleware Medical's ASCardio module.
#
# ## Requirements
#
# - Synopsys Simpleware Medical X-2025.06 or later installed
# - ASCardio module license
# - Cardiac CT image (gated or high-resolution)
#
# ## Overview
#
# The `SegmentHeartSimpleware` class provides:
# - Automated heart chamber segmentation (LV, RV, LA, RA)
# - Myocardium segmentation
# - Major vessel segmentation (aorta, pulmonary artery, coronary arteries)
# - Integration with MONAI Physio workflows

# %% [markdown]
# ## 1. Setup and Imports

# %%
import logging
import os
import sys
import tkinter as tk
from tkinter import filedialog

import itk
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from monai_physio.landmark_tools import LandmarkTools
from monai_physio.segment_heart_simpleware import SegmentHeartSimpleware

_HERE = os.path.dirname(os.path.abspath(__file__))

# %% [markdown]
# ## 2. Configuration

# %%
# Directory setup
output_dir = os.path.join(_HERE, "results")
os.makedirs(output_dir, exist_ok=True)

# Optional: Set custom Simpleware path if not in default location
custom_simpleware_path = None
# Example:
# custom_simpleware_path = "D:/Synopsys/Simpleware/ConsoleSimplewareMedical.exe"

# Enable detailed logging
log_level = logging.INFO  # Change to logging.DEBUG for more detail

# %% [markdown]
# ## 3. Load Input CT Image
#
# Load a cardiac CT image for segmentation. This should be a 3D volume containing the heart.

# %%
root = tk.Tk()
root.withdraw()
input_image_path = filedialog.askopenfilename(
    title="Select a cardiac CT image",
    filetypes=[("NIfTI", "*.nii.gz"), ("MetaIO", "*.mhd"), ("All files", "*.*")],
)
root.destroy()

if not input_image_path:
    print("No image selected; nothing to segment.")
    sys.exit(0)

# Load the image
try:
    input_image = itk.imread(input_image_path)
    print(f"Successfully loaded image from: {input_image_path}")
    print(f"Image size: {itk.size(input_image)}")
    print(f"Image spacing: {input_image.GetSpacing()}")
    print(f"Image origin: {input_image.GetOrigin()}")
except (FileNotFoundError, OSError) as e:
    print(f"Error loading image: {e}")
    print("Please update the input_image_path to point to a valid cardiac CT image.")
    input_image = None

# %% [markdown]
# ## 4. Visualize Input Image
#
# Display a few slices of the input image to verify it loaded correctly.

# %%
image_array = None
if input_image is not None:
    # Get numpy array from ITK image
    image_array = itk.array_from_image(input_image)

    # Display axial, sagittal, and coronal slices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial slice (middle)
    dirs = input_image.GetDirection()
    print(f"Direction: {dirs}")
    axial_slice = image_array[image_array.shape[0] // 2, :, :]
    axes[0].imshow(axial_slice, cmap="gray", vmin=-200, vmax=400)
    axes[0].set_title("Axial View")
    axes[0].axis("off")

    # Sagittal slice (middle)
    sagittal_slice = image_array[::-1, :, image_array.shape[2] // 2]
    axes[1].imshow(sagittal_slice, cmap="gray", vmin=-200, vmax=400)
    axes[1].set_title("Sagittal View")
    axes[1].axis("off")

    # Coronal slice (middle)
    coronal_slice = image_array[::-1, image_array.shape[1] // 2, :]
    axes[2].imshow(coronal_slice, cmap="gray", vmin=-200, vmax=400)
    axes[2].set_title("Coronal View")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Image intensity range: [{image_array.min():.1f}, {image_array.max():.1f}]")
else:
    print("No input image available for visualization.")

# %% [markdown]
# ## 5. Initialize Simpleware Segmentation
#
# Create an instance of the `SegmentHeartSimpleware` class.

# %%
# Create segmenter instance with logging
segmenter = SegmentHeartSimpleware(log_level=log_level)

# Set custom Simpleware path if specified
if custom_simpleware_path is not None:
    print(f"Setting custom Simpleware path: {custom_simpleware_path}")
    segmenter.set_simpleware_executable_path(custom_simpleware_path)
else:
    print(f"Using default Simpleware path: {segmenter.simpleware_exe_path}")

# Display segmentation configuration
print("\nSegmentation Configuration:")
print(f"  Target spacing: {segmenter.target_spacing} mm")
heart_labels = segmenter.taxonomy.labels_in_group("heart")
vessel_labels = segmenter.taxonomy.labels_in_group("major_vessels")
print(f"  Heart structures: {len(heart_labels)} labels")
print(f"  Vessel structures: {len(vessel_labels)} labels")

print("\nHeart Structure IDs:")
for id, name in heart_labels.items():
    print(f"  {id}: {name}")

print("\nMajor Vessel IDs:")
for id, name in vessel_labels.items():
    print(f"  {id}: {name}")

# %% [markdown]
# ## 6. Run Segmentation
#
# Perform the heart segmentation using Simpleware Medical ASCardio.
#
# **Note**: This step calls Simpleware Medical as an external process and may take several minutes depending on image size and system performance.

# %%
if input_image is not None:
    print("Starting heart segmentation with Simpleware Medical ASCardio...")
    print("This may take several minutes. Please wait...\n")

    try:
        # Perform segmentation
        result = segmenter.segment(input_image)

        print("\nSegmentation completed successfully!")

        # Extract individual results
        labelmap_image = result["labelmap"]
        heart_mask = result["heart"]
        major_vessels_mask = result["major_vessels"]
        contrast_mask = result["contrast"]

        # Save results
        print("\nSaving segmentation results...")
        itk.imwrite(
            labelmap_image,
            os.path.join(output_dir, "heart_labelmap_simpleware.nii.gz"),
            compression=True,
        )
        itk.imwrite(
            heart_mask,
            os.path.join(output_dir, "heart_mask_simpleware.nii.gz"),
            compression=True,
        )
        itk.imwrite(
            major_vessels_mask,
            os.path.join(output_dir, "vessels_mask_simpleware.nii.gz"),
            compression=True,
        )
        itk.imwrite(
            contrast_mask,
            os.path.join(output_dir, "contrast_mask_simpleware.nii.gz"),
            compression=True,
        )
        itk.imwrite(
            input_image,
            os.path.join(output_dir, "input_image_simpleware.nii.gz"),
            compression=True,
        )

        # Save landmarks
        print("\nSaving landmarks...")
        landmarks = segmenter.get_landmarks()
        LandmarkTools().write_landmarks_3dslicer(
            landmarks=landmarks, path=os.path.join(output_dir, "landmarks.mrk.json")
        )

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure Simpleware Medical is installed at the correct path.")
        result = None

    except RuntimeError as e:
        print(f"\nSegmentation failed: {e}")
        print("Please check the error messages above for details.")
        result = None

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        result = None

else:
    print("No input image available. Skipping segmentation.")
    result = None

# %% [markdown]
# ## 7. Analyze Segmentation Results
#
# Display statistics about the segmented structures.

# %%
if result is not None:
    labelmap_array = itk.array_from_image(result["labelmap"])

    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(labelmap_array, return_counts=True)

    # Calculate voxel volume
    spacing = result["labelmap"].GetSpacing()
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    print("\n=== Segmentation Statistics ===")
    print(f"\nTotal unique labels found: {len(unique_labels) - 1}")  # -1 for background
    print(f"Voxel volume: {voxel_volume_mm3:.3f} mm³")
    print("\nStructure Volumes:")

    # Combine all mask dictionaries for label lookup
    all_labels = {
        **segmenter.taxonomy.labels_in_group("heart"),
        **segmenter.taxonomy.labels_in_group("major_vessels"),
        **segmenter.taxonomy.labels_in_group("contrast"),
    }

    for label, count in zip(unique_labels, label_counts):
        if label == 0:  # Skip background
            continue

        volume_mm3 = count * voxel_volume_mm3
        volume_ml = volume_mm3 / 1000

        label_name = all_labels.get(label, f"unknown_{label}")
        print(f"  {label_name} (ID {label}): {volume_ml:.2f} mL ({count:,} voxels)")

    # Calculate combined volumes
    heart_array = itk.array_from_image(result["heart"])
    vessels_array = itk.array_from_image(result["major_vessels"])

    heart_volume_ml = (np.sum(heart_array > 0) * voxel_volume_mm3) / 1000
    vessels_volume_ml = (np.sum(vessels_array > 0) * voxel_volume_mm3) / 1000

    print("\nCombined Volumes:")
    print(f"  Total heart structures: {heart_volume_ml:.2f} mL")
    print(f"  Total major vessels: {vessels_volume_ml:.2f} mL")
else:
    print("No segmentation results available for analysis.")

# %% [markdown]
# ## 8. Visualize Segmentation Results (2D)
#
# Display the segmentation overlaid on the original image.

# %%
if result is not None and input_image is not None:
    # Get arrays
    labelmap_array = itk.array_from_image(result["labelmap"])
    heart_array = itk.array_from_image(result["heart"])
    vessels_array = itk.array_from_image(result["major_vessels"])

    labelmap_essentials = segmenter.trim_branches(result["labelmap"])
    labelmap_essentials_array = itk.array_from_image(labelmap_essentials)

    # Select middle slice
    mid_slice = image_array.shape[0] // 2

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Original, labelmap, heart mask
    axes[0, 0].imshow(image_array[mid_slice, :, :], cmap="gray", vmin=-200, vmax=400)
    axes[0, 0].set_title("Original Image (Axial)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image_array[mid_slice, :, :], cmap="gray", vmin=-200, vmax=400)
    labelmap_overlay = np.ma.masked_where(
        labelmap_essentials_array[mid_slice, :, :] == 0,
        labelmap_essentials_array[mid_slice, :, :],
    )
    axes[0, 1].imshow(labelmap_overlay, cmap="jet", alpha=0.5, vmin=1, vmax=10)
    axes[0, 1].set_title("Labelmap Overlay")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(image_array[mid_slice, :, :], cmap="gray", vmin=-200, vmax=400)
    heart_overlay = np.ma.masked_where(
        heart_array[mid_slice, :, :] == 0, heart_array[mid_slice, :, :]
    )
    axes[0, 2].imshow(heart_overlay, cmap="Reds", alpha=0.5)
    axes[0, 2].set_title("Heart Mask")
    axes[0, 2].axis("off")

    # Row 2: Vessels, sagittal view, coronal view
    axes[1, 0].imshow(image_array[mid_slice, :, :], cmap="gray", vmin=-200, vmax=400)
    vessels_overlay = np.ma.masked_where(
        vessels_array[mid_slice, :, :] == 0, vessels_array[mid_slice, :, :]
    )
    axes[1, 0].imshow(vessels_overlay, cmap="Blues", alpha=0.5)
    axes[1, 0].set_title("Vessels Mask")
    axes[1, 0].axis("off")

    # Sagittal view
    mid_sagittal = image_array.shape[2] // 2
    axes[1, 1].imshow(image_array[:, :, mid_sagittal], cmap="gray", vmin=-200, vmax=400)
    sagittal_overlay = np.ma.masked_where(
        labelmap_essentials_array[:, :, mid_sagittal] == 0,
        labelmap_essentials_array[:, :, mid_sagittal],
    )
    axes[1, 1].imshow(sagittal_overlay, cmap="jet", alpha=0.5, vmin=1, vmax=10)
    axes[1, 1].set_title("Sagittal View")
    axes[1, 1].axis("off")

    # Coronal view
    mid_coronal = image_array.shape[1] // 2
    axes[1, 2].imshow(image_array[:, mid_coronal, :], cmap="gray", vmin=-200, vmax=400)
    coronal_overlay = np.ma.masked_where(
        labelmap_essentials_array[:, mid_coronal, :] == 0,
        labelmap_essentials_array[:, mid_coronal, :],
    )
    axes[1, 2].imshow(coronal_overlay, cmap="jet", alpha=0.5, vmin=1, vmax=10)
    axes[1, 2].set_title("Coronal View")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segmentation_visualization.png"), dpi=150)
    plt.show()

    print(
        f"Visualization saved to: {os.path.join(output_dir, 'segmentation_visualization.png')}"
    )
else:
    print("No results available for visualization.")

# %% [markdown]
# ## 9. 3D Visualization (Optional)
#
# Create 3D surface meshes of the segmented structures using PyVista.

# %%
if result is not None:
    print("Creating 3D visualization...")

    # Create VTK images from ITK images for PyVista

    # Convert heart mask to VTK
    heart_vtk = itk.vtk_image_from_image(result["heart"])
    heart_essentials_vtk = itk.vtk_image_from_image(labelmap_essentials)
    vessels_vtk = itk.vtk_image_from_image(result["major_vessels"])

    # Create PyVista plotter
    plotter = pv.Plotter()

    # Extract heart surface
    heart_grid = pv.wrap(heart_vtk)
    heart_surface = heart_grid.contour([0.5])
    if heart_surface.n_points > 0:
        plotter.add_mesh(heart_surface, color="red", opacity=0.5, label="Heart")

    # Extract heart surface
    heart_essentials_grid = pv.wrap(heart_essentials_vtk)
    heart_essentials_surface = heart_essentials_grid.contour([0.5])
    if heart_essentials_surface.n_points > 0:
        plotter.add_mesh(
            heart_essentials_surface, color="grey", opacity=1.0, label="Heart Essential"
        )

    # Extract vessels surface
    vessels_grid = pv.wrap(vessels_vtk)
    vessels_surface = vessels_grid.contour([0.5])
    if vessels_surface.n_points > 0:
        plotter.add_mesh(vessels_surface, color="blue", opacity=1.0, label="Vessels")

    # Configure plotter
    plotter.add_legend()
    plotter.set_background("white")
    plotter.add_axes()

    # Save screenshot
    screenshot_path = os.path.join(output_dir, "3d_visualization.png")
    plotter.show(screenshot=screenshot_path)

    print(f"3D visualization saved to: {screenshot_path}")
else:
    print("No results available for 3D visualization.")

# %%

# %% [markdown]
# ## 10. Summary
#
# This notebook demonstrated:
# 1. Loading a cardiac CT image
# 2. Initializing the `SegmentHeartSimpleware` class
# 3. Running ASCardio heart segmentation through Simpleware Medical
# 4. Analyzing segmentation results
# 5. Visualizing the segmented structures in 2D and 3D
#
# The segmentation results can be used for:
# - Cardiac motion analysis
# - 4D heart visualization in USD format
# - Registration with statistical heart models
# - Clinical measurements and analysis
#
# ## Next Steps
#
# - Export segmentation to USD format for Omniverse visualization
# - Register with statistical heart model (see `Heart-Statistical_Model_To_Patient`)
# - Create 4D heart animation (see `Heart-GatedCT_To_USD`)
# - Perform cardiac function analysis
