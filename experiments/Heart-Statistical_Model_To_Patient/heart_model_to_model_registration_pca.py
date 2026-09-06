# %% [markdown]
# # PCA-based Heart Model to Image Registration Experiment
#
# This notebook demonstrates using the `RegisterModelToImagePCA` class to register
# a statistical shape model to patient CT images using PCA-based shape variation.
#
# ## Overview
# - Uses the KCL Heart Model PCA statistical shape model
# - Registers to the same Duke Heart CT data as the original notebook
# - Two-stage optimization: rigid alignment + PCA shape fitting
# - Converts segmentation mask to intensity image for registration

# %% [markdown]
# ## Setup and Imports

# %%
# PCA-based Heart Model to Image Registration Experiment

import json
import os
from pathlib import Path

import itk
import numpy as np
import pyvista as pv

# Import from MONAI Physio package
from monai_physio import (
    ContourTools,
    RegisterModelsICP,
    RegisterModelsPCA,
    TransformTools,
)
from monai_physio.image_tools import ImageTools

# %% [markdown]
# ## Define File Paths

# %%
# Patient CT image (defines coordinate frame)
patient_data_dir = Path.cwd().parent.parent / "data" / "Slicer-Heart-CT"
patient_ct_path = patient_data_dir / "patient_img.mha"
patient_ct_heart_mask_path = patient_data_dir / "patient_heart_wall_mask.nii.gz"

# PCA heart model data
heart_model_data_dir = Path.cwd().parent.parent / "data" / "KCL-Heart-Model"
heart_model_path = heart_model_data_dir / "average_mesh.vtk"

# PCA statistical model (from Heart-Create_Statistical_Model workflow)
template_model_data_dir = (
    Path.cwd().parent / "Heart-Create_Statistical_Model" / "kcl-heart-model"
)
template_model_surface_path = template_model_data_dir / "pca_mean.vtp"
pca_json_path = template_model_data_dir / "pca_model.json"

# Output directory
output_dir = Path.cwd() / "results_pca"
os.makedirs(output_dir, exist_ok=True)

print(f"Patient data: {patient_data_dir}")
print(f"PCA Model data: {template_model_data_dir}")
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## Load and Preprocess Patient Image

# %%
# Load patient CT image
print("Loading patient CT image...")
patient_image = itk.imread(str(patient_ct_path))
print(f"  Original size: {itk.size(patient_image)}")
print(f"  Original spacing: {itk.spacing(patient_image)}")

# Resample to 1mm isotropic spacing
print("Resampling to isotropic...")
patient_image = ImageTools().make_isotropic_image(patient_image)

print(f"  Resampled size: {itk.size(patient_image)}")
print(f"  Resampled spacing: {itk.spacing(patient_image)}")

# Save preprocessed image
itk.imwrite(patient_image, str(output_dir / "patient_image.mha"), compression=True)
print("Saved preprocessed image")

# %% [markdown]
# ## Load and Process Heart Segmentation Mask

# %%
# Load heart segmentation mask
print("Loading heart segmentation mask...")
patient_heart_mask_image = itk.imread(str(patient_ct_heart_mask_path))

print(f"  Mask size: {itk.size(patient_heart_mask_image)}")
print(f"  Mask spacing: {itk.spacing(patient_heart_mask_image)}")

# %%
# Handle image orientation (flip if needed)
flip0 = np.array(patient_heart_mask_image.GetDirection())[0, 0] < 0
flip1 = np.array(patient_heart_mask_image.GetDirection())[1, 1] < 0
flip2 = np.array(patient_heart_mask_image.GetDirection())[2, 2] < 0

if flip0 or flip1 or flip2:
    print(f"Flipping image axes: {flip0}, {flip1}, {flip2}")

    # Flip CT image
    flip_filter = itk.FlipImageFilter.New(Input=patient_image)
    flip_filter.SetFlipAxes([int(flip0), int(flip1), int(flip2)])
    flip_filter.SetFlipAboutOrigin(True)
    flip_filter.Update()
    patient_image = flip_filter.GetOutput()
    id_mat = itk.Matrix[itk.D, 3, 3]()
    id_mat.SetIdentity()
    patient_image.SetDirection(id_mat)

    # Flip mask image
    flip_filter = itk.FlipImageFilter.New(Input=patient_heart_mask_image)
    flip_filter.SetFlipAxes([int(flip0), int(flip1), int(flip2)])
    flip_filter.SetFlipAboutOrigin(True)
    flip_filter.Update()
    patient_heart_mask_image = flip_filter.GetOutput()
    patient_heart_mask_image.SetDirection(id_mat)

    print("Images flipped to standard orientation")

# Save oriented images
itk.imwrite(
    patient_image, str(output_dir / "patient_image_oriented.mha"), compression=True
)
itk.imwrite(
    patient_heart_mask_image,
    str(output_dir / "patient_heart_mask_oriented.mha"),
    compression=True,
)

# %% [markdown]
# ## Convert Segmentation Mask to a Surface

# %%
contour_tools = ContourTools()
patient_surface = contour_tools.extract_contours(patient_heart_mask_image)

# %% [markdown]
# ## Perform Initial ICP Affine Registration
#
# Use ICP (Iterative Closest Point) with affine mode to align the model surface to the patient surface extracted from the segmentation mask. This provides a good initial alignment for the PCA-based registration.
#
# The ICP registration pipeline:
# 1. Centroid alignment (automatic)
# 2. Rigid ICP alignment
#
# The PCA registration will then refine this initial alignment with shape model constraints.

# %%
# Load the pca model
print("Loading PCA heart model...")
template_model = pv.read(str(heart_model_path))

template_model_surface = pv.read(template_model_surface_path)
print(f"  Template surface: {template_model_surface.n_points} points")

icp_registrar = RegisterModelsICP(fixed_model=patient_surface)

icp_result = icp_registrar.register(
    transform_type="Affine",
    moving_model=template_model_surface,
    max_iterations=2000,
)

# Get the aligned mesh and transform
icp_registered_model_surface = icp_result["registered_model"]
icp_moving_to_fixed_transform = icp_result["moving_to_fixed_transform"]

print("\nICP affine registration complete")
print("   Transform =", icp_result["moving_to_fixed_transform"])

# Save aligned model
icp_registered_model_surface.save(str(output_dir / "icp_registered_model_surface.vtp"))
print("  Saved ICP-aligned model surface")

itk.transformwrite(
    [icp_result["moving_to_fixed_transform"]],
    str(output_dir / "icp_transform.hdf"),
    compression=True,
)
print("  Saved ICP transform")

# %%
# Apply ICP transform to the full average mesh (not just surface)
# This gets the volumetric mesh into patient space for PCA registration
transform_tools = TransformTools()
icp_registered_model = transform_tools.transform_pvcontour(
    template_model, icp_moving_to_fixed_transform
)
icp_registered_model.save(str(output_dir / "icp_registered_model.vtk"))
print("\nApplied ICP transform to full model mesh")

# %% [markdown]
# ## Initialize PCA Registration

# %%
## Initialize PCA Registration
print("=" * 70)

# Use the mean PCA template and apply the ICP alignment after PCA deformation.
with open(pca_json_path, encoding="utf-8") as f:
    pca_model = json.load(f)
pca_registrar = RegisterModelsPCA.from_pca_model(
    pca_template_model=template_model_surface,
    pca_model=pca_model,
    pca_number_of_modes=10,
    post_pca_transform=icp_moving_to_fixed_transform,
    fixed_model=patient_surface,
    reference_image=patient_image,
)

itk.imwrite(pca_registrar.fixed_distance_map, str(output_dir / "distance_map.mha"))

print("PCA registrar initialized")
print("  Applying ICP alignment as the post-PCA transform")
print(f"  Number of points: {len(pca_registrar.pca_template_model.points)}")
print(f"  Number of PCA modes: {pca_registrar.pca_number_of_modes}")

# %% [markdown]
# ## Run PCA-Based Shape Optimization
#
# Now that we have a good initial alignment from ICP affine registration, we run the PCA-based registration to optimize the shape parameters.

# %%
print("\n" + "=" * 70)
print("PCA-BASED SHAPE OPTIMIZATION")
print("=" * 70)
print("\nRunning complete PCA registration pipeline...")
print("  (Applying PCA deformation, then ICP alignment)")

result = pca_registrar.register(
    pca_number_of_modes=10,  # Use first 10 PCA modes
)

dm = pca_registrar.fixed_distance_map
itk.imwrite(dm, str(output_dir / "target_distance_map.mha"))

pca_registered_model_surface = result["registered_model"]

print("\nPCA registration complete")

# %% [markdown]
# ### Display Registration Results
#
# Review the optimization results from the PCA registration pipeline.
#

# %%
print("\n" + "=" * 70)
print("REGISTRATION RESULTS")
print("=" * 70)

# Display results
print("\nFinal Registration Metrics:")
print(f"  Final mean intensity: {result['mean_distance']:.4f}")

print("\nOptimized PCA Coefficients (in units of std deviations):")
for i, coef in enumerate(result["pca_coefficients"]):
    print(f"  Mode {i + 1:2d}: {coef:7.4f}")

print("\nRegistration pipeline complete!")

# %% [markdown]
# ## Save Registration Results
#

# %%
print("\nSaving results...")

# Save final PCA-registered mesh
pca_registered_model_surface.save(str(output_dir / "pca_registered_model_surface.vtk"))
print("  Saved final PCA-registered mesh")

ref_image = contour_tools.create_reference_image(pca_registered_model_surface)

distance_map = contour_tools.create_distance_map(
    pca_registered_model_surface,
    ref_image,
    squared_distance=True,
    negative_inside=False,
    zero_inside=True,
    norm_to_max_distance=200.0,
)

itk.imwrite(distance_map, str(output_dir / "pca_distance_map.mha"))

# Save PCA coefficients
np.savetxt(
    str(output_dir / "pca_coefficients.txt"),
    result["pca_coefficients"],
    header=f"PCA coefficients for {len(result['pca_coefficients'])} modes",
)
print("  Saved PCA coefficients")

# %% [markdown]
# ## Visualize Results
#

# %%
# Create side-by-side comparison
plotter = pv.Plotter(shape=(1, 2), window_size=[1000, 600])

plotter.subplot(0, 0)
plotter.add_mesh(patient_surface, color="red", opacity=1.0, label="Patient")
plotter.add_mesh(
    icp_registered_model_surface, color="green", opacity=1.0, label="ICP Registered"
)
plotter.add_title("ICP Shape Fitting")
plotter.add_axes()

# After PCA shape fitting
plotter.subplot(0, 1)
plotter.add_mesh(patient_surface, color="red", opacity=1.0, label="Patient")
plotter.add_mesh(
    pca_registered_model_surface, color="green", opacity=1.0, label="PCA Registered"
)
plotter.add_title("PCA Shape Fitting")
plotter.add_axes()

plotter.link_views()
plotter.show()

# %% [markdown]
# ## Visualize PCA Displacement Magnitude
#
# Compute and display the displacement magnitude caused by PCA optimization. This shows how much each point moved from the ICP-aligned mean shape to the final PCA-registered shape.

# %%
# Compute displacement from ICP-aligned (mean shape) to PCA-registered shape
icp_points = icp_registered_model_surface.points
pca_points = pca_registered_model_surface.points

# Calculate displacement vectors
displacement_vectors = pca_points - icp_points

# Compute surface normals for the ICP-aligned mesh
icp_registered_model_with_normals = icp_registered_model_surface.compute_normals(
    point_normals=True, cell_normals=False
)
normals = icp_registered_model_with_normals.point_data["Normals"]

# Calculate signed displacement along the normal direction
# Positive = outward displacement, Negative = inward displacement
signed_displacement = np.sum(displacement_vectors * normals, axis=1)

# Add displacement as scalar data to the mesh
pca_registered_model_with_displacement = pca_registered_model_surface.copy()
pca_registered_model_with_displacement["PCA Signed Displacement (mm)"] = (
    signed_displacement
)

# Print statistics
print("PCA Signed Displacement Statistics:")
print(f"  Mean displacement: {np.mean(signed_displacement):.2f} mm")
print(f"  Max displacement (outward): {np.max(signed_displacement):.2f} mm")
print(f"  Min displacement (inward): {np.min(signed_displacement):.2f} mm")
print(f"  Std displacement: {np.std(signed_displacement):.2f} mm")

# Visualize the signed displacement with diverging colormap
# Blue = inward displacement, Red = outward displacement
plotter = pv.Plotter(window_size=[800, 600])
plotter.add_mesh(
    pca_registered_model_with_displacement,
    scalars="PCA Signed Displacement (mm)",
    cmap="RdBu_r",  # Red for positive (outward), Blue for negative (inward)
    clim=[
        -np.max(np.abs(signed_displacement)),
        np.max(np.abs(signed_displacement)),
    ],  # Symmetric color scale
    show_scalar_bar=True,
    scalar_bar_args={
        "title": "PCA Signed Displacement (mm)\n(Red=Outward, Blue=Inward)",
        "vertical": True,
        "position_x": 0.82,
        "position_y": 0.1,
    },
)
plotter.add_title("PCA Signed Displacement on Registered Model")
plotter.add_axes()
plotter.show()

# Save the mesh with displacement data
pca_registered_model_with_displacement.save(
    str(output_dir / "pca_registered_model_with_signed_displacement.vtp")
)
print("\nSaved model with signed displacement data")
