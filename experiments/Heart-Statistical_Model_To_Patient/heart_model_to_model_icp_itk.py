# %% [markdown]
# # ICP via ITK Heart Model to Image Registration Experiment
#
# This notebook demonstrates using the `RegisterModelToImageICPITK` class to register
# a statistical shape model to patient CT images using PCA-based shape variation.
#
# ## Overview
# - Uses the KCL Heart Model PCA statistical shape model
# - Registers to the same Duke Heart CT data as the original notebook
# - Converts segmentation mask to intensity image for registration

# %% [markdown]
# ## Setup and Imports

# %%
# PCA-based Heart Model to Image Registration Experiment

import os
from pathlib import Path

import itk
import numpy as np
import pyvista as pv

# Import from MONAI Physio package
from monai_physio import (
    ContourTools,
    RegisterModelsICPITK,
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

# heart model data
heart_model_data_dir = Path.cwd().parent.parent / "data" / "KCL-Heart-Model"
heart_model_path = heart_model_data_dir / "average_mesh.vtk"

# Output directory
output_dir = Path.cwd() / "results_icp_itk"
os.makedirs(output_dir, exist_ok=True)

print(f"Patient data: {patient_data_dir}")
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
patient_heart_mask = itk.imread(str(patient_ct_heart_mask_path))

print(f"  Mask size: {itk.size(patient_heart_mask)}")
print(f"  Mask spacing: {itk.spacing(patient_heart_mask)}")

# %%
# Handle image orientation (flip if needed)
flip0 = np.array(patient_image.GetDirection())[0, 0] < 0
flip1 = np.array(patient_image.GetDirection())[1, 1] < 0
flip2 = np.array(patient_image.GetDirection())[2, 2] < 0

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
    flip_filter = itk.FlipImageFilter.New(Input=patient_heart_mask)
    flip_filter.SetFlipAxes([int(flip0), int(flip1), int(flip2)])
    flip_filter.SetFlipAboutOrigin(True)
    flip_filter.Update()
    patient_heart_mask = flip_filter.GetOutput()
    patient_heart_mask.SetDirection(id_mat)

    print("Images flipped to standard orientation")

# Save oriented images
itk.imwrite(
    patient_image, str(output_dir / "patient_image_oriented.mha"), compression=True
)
itk.imwrite(
    patient_heart_mask,
    str(output_dir / "patient_heart_mask_oriented.mha"),
    compression=True,
)

# %% [markdown]
# ## Convert Segmentation Mask to a Surface

# %%
contour_tools = ContourTools()
patient_heart_surface = contour_tools.extract_contours(patient_heart_mask)

# %% [markdown]
# ## Perform Initial ICP Rigid Registration
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
print("Loading template heart model...")
template_model = pv.read(str(heart_model_path))
template_model_surface = template_model.extract_surface(algorithm="dataset_surface")

icp_registrar = RegisterModelsICPITK(
    fixed_model=patient_heart_surface, reference_image=patient_image
)

icp_result = icp_registrar.register(
    moving_model=template_model_surface,
    transform_type="Affine",
    max_iterations=100,
    method="L-BFGS-B",
)

# Get the aligned mesh and transform
icp_registered_model_surface = icp_result["registered_model"]
icp_forward_point_transform = icp_result["forward_point_transform"]

print("\nICP affine registration complete")
print("   Transform =", icp_result["forward_point_transform"])

# Save aligned model
icp_registered_model_surface.save(str(output_dir / "icp_registered_model_surface.vtp"))
print("  Saved ICP-aligned model surface")

itk.transformwrite(
    [icp_result["forward_point_transform"]],
    str(output_dir / "icp_transform.hdf"),
    compression=True,
)
print("  Saved ICP transform")

# %%
# Apply ICP transform to the full average mesh (not just surface)
# This gets the volumetric mesh into patient space for PCA registration
transform_tools = TransformTools()
icp_registered_model = transform_tools.transform_pvcontour(
    template_model, icp_forward_point_transform
)
icp_registered_model.save(str(output_dir / "icp_registered_model.vtk"))
print("\nApplied ICP transform to full model mesh")

# %% [markdown]
# ## Visualize Results
#

# %%
# Create side-by-side comparison
plotter = pv.Plotter(window_size=[600, 600])

plotter.add_mesh(patient_heart_surface, color="red", opacity=1.0, label="Patient")
plotter.add_mesh(
    icp_registered_model_surface, color="green", opacity=1.0, label="ICP Registered"
)
plotter.add_title("ICP Shape Fitting")
plotter.add_axes()

plotter.show()
