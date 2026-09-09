# %%
import os

import itk
import pyvista as pv

from monai_physio.transform_tools import TransformTools

# %%
os.makedirs("results_CombineModels", exist_ok=True)

# %%
all_mask_t00 = itk.imread("results/Case1Pack_T00_all_mask_org.mha")
lung_mask_t30 = itk.imread("results/Case1Pack_T30_dynamic_anatomy_mask_org.mha")
other_mask_t30 = itk.imread("results/Case1Pack_T30_static_anatomy_mask_org.mha")

img_tfm_lung_t00 = itk.transformread(
    "results/Case1Pack_T00_dynamic_anatomy_forward.hdf"
)[0]
img_tfm_other_t00 = itk.transformread(
    "results/Case1Pack_T00_static_anatomy_forward.hdf"
)[0]

# %%
tfm_tools = TransformTools()

lung_mask_t00 = tfm_tools.transform_image(lung_mask_t30, img_tfm_lung_t00, all_mask_t00)
other_mask_t00 = tfm_tools.transform_image(
    other_mask_t30, img_tfm_other_t00, all_mask_t00
)

itk.imwrite(lung_mask_t00, "results_CombineModels/lung_mask_t00.mha", compression=True)
itk.imwrite(
    other_mask_t00, "results_CombineModels/other_mask_t00.mha", compression=True
)

# %%
lung_model_t30 = pv.read("results/Case1Pack_T30_dynamic_anatomy_lungGatedBase.vtp")
other_model_t30 = pv.read("results/Case1Pack_T30_static_anatomy_lungGatedBase.vtp")

model_tfm_lung_t00 = itk.transformread(
    "results/Case1Pack_T00_dynamic_anatomy_inverse.hdf"
)[0]
model_tfm_other_t00 = itk.transformread(
    "results/Case1Pack_T00_static_anatomy_inverse.hdf"
)[0]

lung_model_t00 = tfm_tools.transform_pvcontour(lung_model_t30, model_tfm_lung_t00)
other_model_t00 = tfm_tools.transform_pvcontour(other_model_t30, model_tfm_other_t00)

lung_model_t00.save("results_CombineModels/lung_model_t00.vtp")
other_model_t00.save("results_CombineModels/other_model_t00.vtp")
