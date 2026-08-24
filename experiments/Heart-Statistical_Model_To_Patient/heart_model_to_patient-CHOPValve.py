# %% [markdown]
# ## Setup and Imports

# %%
import json
import os
from pathlib import Path

import itk
import pyvista as pv

# Import from PhysioTwin4D package
from physiotwin4d import (
    SegmentHeartSimplewareTrimmedBranches,
    WorkflowFitStatisticalModelToPatient,
)

# %% [markdown]
# ## Define File Paths

# %%
# Patient CT image (defines coordinate frame)
patient_data_dir = Path(__file__).parent.parent.parent / "data" / "CHOP-Valve4D" / "CT"
patient_ct_path = patient_data_dir / "RVOT28-Dias.mha"

# Template model (moving)
model_data_dir = Path(__file__).parent.parent.parent / "data" / "KCL-Heart-Model"
model_labelmap_path = model_data_dir / "labelmap" / "average_labelmap_with_bkg.mha"
model_pca_data_dir = (
    Path(__file__).parent.parent / "Heart-Create_Statistical_Model" / "kcl-heart-model"
)
model_pca_json_path = model_pca_data_dir / "pca_model.json"
model_mesh_path = model_pca_data_dir / "pca_mean.vtp"
model_pca_n_modes = 10

# Output directory
output_dir = Path(__file__).parent / "results-chop"

# %%
patient_image = itk.imread(str(patient_ct_path))

template_model = pv.read(str(model_mesh_path))

with open(model_pca_json_path, encoding="utf-8") as f:
    model_pca_data = json.load(f)

os.makedirs(output_dir, exist_ok=True)

# %%
registrar = WorkflowFitStatisticalModelToPatient(
    template_model=template_model,
    patient_image=patient_image,
    segmentation_method=SegmentHeartSimplewareTrimmedBranches(),
)

registrar.set_use_pca_registration(
    True, pca_model=model_pca_data, number_of_pca_components=model_pca_n_modes
)

registrar.set_use_labelmap_to_labelmap_registration(True)

# %%
patient_image = registrar.patient_image
itk.imwrite(
    patient_image, str(output_dir / "patient_image_preprocessed.mha"), compression=True
)

# %%
results = registrar.process()

# %%
registered_model = results["fitted_reference_model"]
registered_model_surface = results["fitted_reference_mesh"]

registered_model.save(str(output_dir / "registered_model.vtp"))
registered_model_surface.save(str(output_dir / "registered_model_surface.vtp"))

# %%
pca_model = registrar.pca_template_model
pca_model_surface = registrar.pca_template_model_surface
pca_labelmap = registrar.pca_template_labelmap

pca_model.save(str(output_dir / "pca_model.vtu"))
pca_model_surface.save(str(output_dir / "pca_model_surface.vtp"))
itk.imwrite(pca_labelmap, str(output_dir / "pca_labelmap.mha"), compression=True)

# %% [markdown]
# ## Visualize Final Results

# %%
# Load meshes from registrar member variables
patient_surface = registrar.patient_model_surface

# Create side-by-side comparison
plotter = pv.Plotter(shape=(1, 2))

# After rough alignment
plotter.subplot(0, 0)
plotter.add_mesh(patient_surface, color="red", opacity=0.5, label="Patient")
plotter.add_mesh(pca_model_surface, color="green", opacity=0.8, label="After PCA")
plotter.add_title("PCA Alignment")

# After deformable registration
plotter.subplot(0, 1)
plotter.add_mesh(patient_surface, color="red", opacity=0.5, label="Patient")
plotter.add_mesh(
    registered_model_surface, color="green", opacity=0.8, label="Registered"
)
plotter.add_title("Final Registration")

plotter.link_views()
plotter.show()

# %% [markdown]
# ## Visualize Deformation Magnitude

# %%
# The transformed mesh has deformation magnitude stored as point data
if "DeformationMagnitude" in registered_model_surface.point_data:
    plotter = pv.Plotter()
    plotter.add_mesh(
        registered_model_surface,
        scalars="DeformationMagnitude",
        cmap="jet",
        show_scalar_bar=True,
        scalar_bar_args={"title": "Deformation (mm)"},
    )
    plotter.add_title("Deformation Magnitude")
    plotter.show()

    # Print statistics
    deformation = registered_model_surface["DeformationMagnitude"]
    print("Deformation statistics:")
    print(f"  Min: {deformation.min():.2f} mm")
    print(f"  Max: {deformation.max():.2f} mm")
    print(f"  Mean: {deformation.mean():.2f} mm")
    print(f"  Std: {deformation.std():.2f} mm")
else:
    print("DeformationMagnitude not found in mesh point data")
