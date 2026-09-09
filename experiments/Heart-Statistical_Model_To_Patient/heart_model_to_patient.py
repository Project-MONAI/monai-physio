# %% [markdown]
# ## Setup and Imports

# %%
import json
import os
import time
from pathlib import Path

import itk
import numpy as np
import pyvista as pv

# Import from MONAI Physio package
from monai_physio import (
    ContourTools,
    SegmentChestTotalSegmentator,
    WorkflowFitStatisticalModelToPatient,
)

# nnUNetv2 (used by TotalSegmentator) spawns a multiprocessing.Pool. On Windows
# the spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around the top-level work, that re-import fires
# segment() again and Python's spawn-cascade detector raises RuntimeError. The
# segmenter call below is currently gated by `if False:` but the guard is kept
# in case that branch is re-enabled, and to be consistent with sibling scripts.
if __name__ == "__main__":
    # %% [markdown]
    # ## Define File Paths

    # %%
    # Patient CT image (defines coordinate frame)
    patient_data_dir = Path.cwd().parent / ".." / "data" / "Slicer-Heart-CT"
    patient_ct_path = patient_data_dir / "patient_img.mha"
    patient_ct_heart_mask_path = patient_data_dir / "patient_heart_wall_mask.nii.gz"

    # Atlas template model (moving)
    atlas_data_dir = Path.cwd().parent / ".." / "data" / "KCL-Heart-Model"
    atlas_vtu_path = atlas_data_dir / "average_mesh.vtk"
    atlas_labelmap_path = atlas_data_dir / "labelmap" / "average_labelmap_with_bkg.mha"

    pca_data_dir = (
        Path.cwd().parent / "Heart-Create_Statistical_Model" / "kcl-heart-model"
    )
    pca_json_path = pca_data_dir / "pca_model.json"
    pca_n_modes = 10

    # Output directory
    output_dir = Path.cwd() / "results"

    os.makedirs(output_dir, exist_ok=True)

    # %%
    patient_image = itk.imread(str(patient_ct_path))
    itk.imwrite(patient_image, str(output_dir / "patient_image.mha"), compression=True)

    # %%
    if False:
        segmentator = SegmentChestTotalSegmentator()
        patient_segmentation_data = segmentator.segment(patient_image)
        labelmap = patient_segmentation_data["labelmap"]
        lung_mask = patient_segmentation_data["lung"]
        heart_mask = patient_segmentation_data["heart"]
        major_vessels_mask = patient_segmentation_data["major_vessels"]
        bone_mask = patient_segmentation_data["bone"]
        soft_tissue_mask = patient_segmentation_data["soft_tissue"]
        other_mask = patient_segmentation_data["other"]

        itk.imwrite(
            labelmap, str(output_dir / "patient_labelmap.mha"), compression=True
        )

        heart_arr = itk.GetArrayFromImage(heart_mask)
        mask_arr = (heart_arr > 0).astype(
            np.uint8
        )  # ((heart_arr + contrast_arr) > 0).astype(np.uint8)
        patient_mask = itk.GetImageFromArray(mask_arr)
        patient_mask.CopyInformation(patient_image)

        itk.imwrite(
            patient_mask,
            str(output_dir / "patient_heart_mask_draft.mha"),
            compression=True,
        )

        # hand edit fixed_mask to make patient_heart_wall_mask.nii.gz that is saved in patient_data_dir
    else:
        patient_mask = itk.imread(str(patient_ct_heart_mask_path))

    # %%
    flip0 = np.array(patient_mask.GetDirection())[0, 0] < 0
    flip1 = np.array(patient_mask.GetDirection())[1, 1] < 0
    flip2 = np.array(patient_mask.GetDirection())[2, 2] < 0
    if flip0 or flip1 or flip2:
        print("Flipping patient image...")
        print(flip0, flip1, flip2)
        flip_filter = itk.FlipImageFilter.New(Input=patient_image)
        flip_filter.SetFlipAxes([int(flip0), int(flip1), int(flip2)])
        flip_filter.SetFlipAboutOrigin(True)
        flip_filter.Update()
        patient_image = flip_filter.GetOutput()
        id_mat = itk.Matrix[itk.D, 3, 3]()
        id_mat.SetIdentity()
        patient_image.SetDirection(id_mat)
        itk.imwrite(
            patient_image, str(output_dir / "patient_image.mha"), compression=True
        )
        print("Flipping patient mask image...")
        flip_filter = itk.FlipImageFilter.New(Input=patient_mask)
        flip_filter.SetFlipAxes([int(flip0), int(flip1), int(flip2)])
        flip_filter.SetFlipAboutOrigin(True)
        flip_filter.Update()
        patient_mask = flip_filter.GetOutput()
        patient_mask.SetDirection(id_mat)
        itk.imwrite(
            patient_mask, str(output_dir / "patient_mask.mha"), compression=True
        )

    # %%
    patient_model = ContourTools().extract_contours(patient_mask)
    patient_model.save(str(output_dir / "patient_mesh.vtp"))
    patient_model = pv.read(str(output_dir / "patient_mesh.vtp"))

    template_model = pv.read(str(atlas_vtu_path))
    template_model_surface = template_model.extract_surface(algorithm="dataset_surface")
    template_model_surface.save(str(output_dir / "model_surface.vtp"))
    template_model_surface = pv.read(str(output_dir / "model_surface.vtp"))
    template_labelmap = itk.imread(str(atlas_labelmap_path))

    # %%
    with open(pca_json_path, encoding="utf-8") as f:
        pca_model = json.load(f)
    registrar = WorkflowFitStatisticalModelToPatient(
        template_model=template_model,
        patient_models=[patient_model],
        patient_image=patient_image,
    )
    registrar.set_use_pca_registration(
        True, pca_model=pca_model, number_of_pca_components=pca_n_modes
    )
    registrar.set_use_labelmap_to_image_registration(
        True,
        template_labelmap=template_labelmap,
        template_labelmap_organ_mesh_ids=[1],
        template_labelmap_organ_extra_ids=[2, 3, 4, 5],
        template_labelmap_background_ids=[6],
    )

    registrar.set_labelmap_dilation_mm(0)
    registrar.set_mask_dilation_mm(25)

    patient_image = registrar.patient_image
    itk.imwrite(
        patient_image,
        str(output_dir / "patient_image_preprocessed.mha"),
        compression=True,
    )

    # %%
    # Rough alignment using ICP
    icp_results = registrar.register_model_to_model_icp()
    icp_fixed_to_moving_transform = icp_results["fixed_to_moving_transform"]
    icp_moving_to_fixed_transform = icp_results["moving_to_fixed_transform"]
    icp_model_surface = icp_results["fitted_reference_mesh"]
    icp_labelmap = icp_results["registered_template_labelmap"]

    icp_model_surface.save(str(output_dir / "icp_model_surface.vtp"))
    itk.imwrite(icp_labelmap, str(output_dir / "icp_labelmap.mha"), compression=True)

    # %%
    pca_results = registrar.register_model_to_model_pca()
    pca_coefficients = pca_results["pca_coefficients"]
    pca_model_surface = pca_results["fitted_reference_mesh"]
    pca_labelmap = pca_results["registered_template_labelmap"]

    pca_model_surface.save(str(output_dir / "pca_model_surface.vtp"))
    itk.imwrite(pca_labelmap, str(output_dir / "pca_labelmap.mha"), compression=True)

    # %% [markdown]
    # ## Labelmap Alignment

    # %%
    # Perform deformable registration
    print("Starting deformable labelmap-to-labelmap registration...")

    l2l_results = registrar.register_labelmap_to_labelmap()
    l2l_moving_to_fixed_transform = l2l_results["moving_to_fixed_transform"]
    l2l_fixed_to_moving_transform = l2l_results["fixed_to_moving_transform"]
    l2l_model_surface = l2l_results["fitted_reference_mesh"]
    l2l_labelmap = l2l_results["registered_template_labelmap"]

    print("Registration complete!")

    l2l_model_surface.save(str(output_dir / "l2l_model_surface.vtp"))
    itk.imwrite(l2l_labelmap, str(output_dir / "l2l_labelmap.mha"), compression=True)

    # %%
    print("Starting deformable registration...")
    print("This may take several minutes depending on GPU availability.")

    l2i_results = registrar.register_labelmap_to_image()
    l2i_moving_to_fixed_transform = l2i_results["moving_to_fixed_transform"]
    l2i_fixed_to_moving_transform = l2i_results["fixed_to_moving_transform"]
    l2i_surface = l2i_results["fitted_reference_mesh"]
    l2i_labelmap = l2i_results["registered_template_labelmap"]
    print("\nRegistration complete!")

    # Save registration results to output folder
    l2i_surface.save(str(output_dir / "l2i_model_surface.vtp"))
    itk.imwrite(l2i_labelmap, str(output_dir / "l2i_labelmap.mha"), compression=True)

    # %%
    tmp_p = itk.Point[itk.D, 3]()
    point = registrar.template_model.points[0]
    tmp_p[0] = float(point[0])
    tmp_p[1] = float(point[1])
    tmp_p[2] = float(point[2])

    start_time = time.time()
    # Timing only; the workflow stores the ICP-aligned model before PCA registration.
    icp_p = registrar.icp_registrar.moving_to_fixed_transform.TransformPoint(tmp_p)
    print(
        f"--- ICP forward transform time: {time.time() - start_time} seconds",
        flush=True,
    )

    start_time = time.time()
    # Timing only; PCA registration operates in the ICP-aligned coordinate frame here.
    _ = registrar.pca_registrar.transform_point(icp_p)
    print(f"--- PCA setup time: {time.time() - start_time} seconds", flush=True)
    start_time = time.time()
    # Apply the PCA transform in the workflow's ICP-aligned coordinate frame.
    tmp_p = registrar.pca_registrar.transform_point(icp_p)
    print(f"PCA transform time: {time.time() - start_time} seconds", flush=True)

    start_time = time.time()
    tmp_p = registrar.l2l_moving_to_fixed_transform.TransformPoint(tmp_p)
    print(f"L2L inverse transform time: {time.time() - start_time} seconds", flush=True)

    start_time = time.time()
    tmp_p = registrar.l2i_moving_to_fixed_transform.TransformPoint(tmp_p)
    print(f"L2I inverse transform time: {time.time() - start_time} seconds", flush=True)

    # %%
    # Verify registration using the transform member function
    surface_transformed = registrar.l2i_template_model_surface
    surface_transformed.save(str(output_dir / "registered_template_surface.vtp"))

    model_transformed = registrar.transform_model()
    model_transformed.save(str(output_dir / "registered_template.vtu"))

    # %% [markdown]
    # ## Visualize Final Results

    # %%
    # Load meshes from registrar member variables
    patient_surface = registrar.patient_model_surface
    registered_surface = registrar.fitted_reference_mesh
    icp_surface = registrar.icp_template_model_surface
    pca_surface = registrar.pca_template_model_surface
    l2l_surface = registrar.l2l_template_model_surface
    l2i_surface = registrar.l2i_template_model_surface

    # Create side-by-side comparison
    plotter = pv.Plotter(shape=(1, 2))

    # After rough alignment
    plotter.subplot(0, 0)
    plotter.add_mesh(patient_surface, color="red", opacity=0.5, label="Patient")
    plotter.add_mesh(pca_surface, color="green", opacity=1.0, label="After ICP")
    plotter.add_title("PCA Alignment")

    # After deformable registration
    plotter.subplot(0, 1)
    plotter.add_mesh(patient_surface, color="red", opacity=0.5, label="Patient")
    plotter.add_mesh(l2i_surface, color="blue", opacity=1.0, label="Registered")
    plotter.add_title("Final Registration")

    plotter.link_views()
    plotter.show()

    # %% [markdown]
    # ## Visualize Deformation Magnitude

    # %%
    # The transformed mesh has deformation magnitude stored as point data
    if "DeformationMagnitude" in registered_surface.point_data:
        plotter = pv.Plotter()
        plotter.add_mesh(
            registered_surface,
            scalars="DeformationMagnitude",
            cmap="jet",
            show_scalar_bar=True,
            scalar_bar_args={"title": "Deformation (mm)"},
        )
        plotter.add_title("Deformation Magnitude")
        plotter.show()

        # Print statistics
        deformation = registered_surface["DeformationMagnitude"]
        print("Deformation statistics:")
        print(f"  Min: {deformation.min():.2f} mm")
        print(f"  Max: {deformation.max():.2f} mm")
        print(f"  Mean: {deformation.mean():.2f} mm")
        print(f"  Std: {deformation.std():.2f} mm")
    else:
        print("DeformationMagnitude not found in mesh point data")
