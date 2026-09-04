# %%
import os

import itk
import numpy as np
import pyvista as pv

from monai_physio.contour_tools import ContourTools
from monai_physio.segment_chest_total_segmentator_with_contrast import (
    SegmentChestTotalSegmentatorWithContrast,
)

# nnUNetv2 (used by TotalSegmentator) spawns a multiprocessing.Pool. On Windows
# the spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around the top-level work, that re-import fires
# segment() again and Python's spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))

    # %%
    # When re-running, you can bypass certain long-running steps
    re_run_image_max = True
    use_fixed_image = True
    re_run_image_segmentation = True

    # %%
    output_dir = os.path.join(_HERE, "results")
    max_image = None
    print("Computing max image...")
    if re_run_image_max and not use_fixed_image:
        # Compute max of all images
        image = None
        try:
            image = itk.imread(
                os.path.join(output_dir, "slice_000.reg_dynamic_anatomy.mha")
            )
        except (FileNotFoundError, OSError):
            print("No image found. Aborting. Please run 1-generate_images.ipynb first.")
            exit(1)
        arr = itk.array_from_image(image)
        print(arr.shape)
        arr = np.where(arr == 0, -1000, arr)
        for i in range(0, 21):
            print(f"Processing slice {i:03d}...")
            tmp_arr = itk.array_from_image(
                itk.imread(
                    os.path.join(output_dir, f"slice_{i:03d}.reg_dynamic_anatomy.mha")
                )
            )
            tmp_arr = np.where(tmp_arr == 0, -1000, tmp_arr)
            arr = np.maximum(arr, tmp_arr)
        print("Max image computed.")
        max_image = itk.image_from_array(arr)
        max_image.CopyInformation(image)
        itk.imwrite(
            max_image,
            os.path.join(output_dir, "slice_max.reg_dynamic_anatomy.mha"),
            compression=True,
        )

    # %%
    if use_fixed_image:
        max_image = itk.imread(os.path.join(output_dir, "slice_fixed.mha"))
        outname = "slice_fixed"
    else:
        max_image = itk.imread(
            os.path.join(output_dir, "slice_max.reg_dynamic_anatomy.mha")
        )
        outname = "slice_max"

    seg = SegmentChestTotalSegmentatorWithContrast()
    if re_run_image_segmentation:
        result = seg.segment(max_image)
        labelmap_image = result["labelmap"]
        itk.imwrite(
            labelmap_image,
            os.path.join(output_dir, f"{outname}.all_mask.mha"),
            compression=True,
        )
    else:
        labelmap_image = itk.imread(os.path.join(output_dir, f"{outname}.all_mask.mha"))
        result = seg.create_anatomy_group_labelmaps(labelmap_image)

    lung_mask = result["lung"]
    heart_mask = result["heart"]
    major_vessels_mask = result["major_vessels"]
    bone_mask = result["bone"]
    soft_tissue_mask = result["soft_tissue"]
    other_mask = result["other"]
    contrast_mask = result["contrast"]

    # %%
    con = ContourTools()
    all_contours = con.extract_contours(labelmap_image)
    all_contours.save(os.path.join(output_dir, f"{outname}.all_mask.vtp"))

    # %%
    label_arr = itk.array_from_image(labelmap_image)
    lung_arr = itk.array_from_image(lung_mask)
    heart_arr = itk.array_from_image(heart_mask)
    major_vessels_arr = itk.array_from_image(major_vessels_mask)
    bone_arr = itk.array_from_image(bone_mask)
    soft_tissue_arr = itk.array_from_image(soft_tissue_mask)
    other_arr = itk.array_from_image(other_mask)
    contrast_arr = itk.array_from_image(contrast_mask)

    # %%
    dynamic_anatomy_arr = np.maximum(heart_arr, contrast_arr)
    dynamic_anatomy_arr = np.maximum(dynamic_anatomy_arr, major_vessels_arr)
    dynamic_anatomy_arr = np.where(dynamic_anatomy_arr, label_arr, 0)
    dynamic_anatomy_image = itk.image_from_array(dynamic_anatomy_arr.astype(np.int16))
    dynamic_anatomy_image.CopyInformation(labelmap_image)
    itk.imwrite(
        dynamic_anatomy_image,
        os.path.join(output_dir, f"{outname}.dynamic_anatomy_mask.mha"),
        compression=True,
    )

    contours = con.extract_contours(dynamic_anatomy_image)
    contours.save(os.path.join(output_dir, f"{outname}.dynamic_anatomy_mask.vtp"))

    # %%
    static_anatomy_arr = lung_arr + bone_arr + soft_tissue_arr + other_arr
    static_anatomy_arr = np.where(static_anatomy_arr, label_arr, 0)
    static_anatomy_image = itk.image_from_array(static_anatomy_arr.astype(np.int16))
    static_anatomy_image.CopyInformation(labelmap_image)
    itk.imwrite(
        static_anatomy_image,
        os.path.join(output_dir, f"{outname}.static_anatomy_mask.mha"),
        compression=True,
    )

    contours = con.extract_contours(static_anatomy_image)
    contours.save(os.path.join(output_dir, f"{outname}.static_anatomy_mask.vtp"))

    # %%
    input_image = None
    if use_fixed_image:
        input_image = itk.imread(os.path.join(output_dir, "slice_fixed.mha"), itk.SS)
    else:
        input_image = itk.imread(
            os.path.join(output_dir, "slice_max.reg_dynamic_anatomy.mha"), itk.SS
        )
    arr = itk.array_from_image(input_image)
    flipped_input_image = itk.image_from_array(arr)
    flipped_input_image.CopyInformation(input_image)

    image = pv.wrap(itk.vtk_image_from_image(flipped_input_image))

    pl = pv.Plotter()
    pl.add_mesh(
        image.slice(normal="z"), cmap="bone", show_scalar_bar=False, opacity=0.5
    )
    pl.add_mesh(
        contours.slice(normal="z"),
        cmap="pink",
        clim=[50, 800],
        show_scalar_bar=False,
        opacity=1.0,
    )
    pl.set_background("black")
    pl.camera_position = "xy"
    pl.show()
