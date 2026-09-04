# %%
import os
from typing import Optional

import itk
import numpy as np
from data_dirlab_4d_ct import DataDirLab4DCT

from monai_physio.image_tools import ImageTools
from monai_physio.register_images_icon import RegisterImagesICON
from monai_physio.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from monai_physio.transform_tools import TransformTools

# nnUNetv2 (used by TotalSegmentator) spawns a multiprocessing.Pool. On Windows
# the spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around the top-level work, that re-import fires
# segment() again and Python's spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))

    fixed_image_num = 3
    heart_mask_dilation = 5

    case_names = DataDirLab4DCT().case_names
    case_names = [case_names[4]]
    images = range(10)
    # images = [1]

    input_dir = os.path.join(_HERE, "..", "..", "data", "DirLab-4DCT")
    output_dir = os.path.join(_HERE, "results")

    # %%
    def dilate_mask(mask: Optional[itk.image], dilation: int) -> Optional[itk.image]:
        if mask is not None:
            return ImageTools().binary_dilate_image(mask, dilation, 1, 0)
        return None

    def register_image(
        fixed_image: itk.image,
        fixed_mask: itk.image,
        moving_image: itk.image,
        moving_mask: itk.image,
        case_name: str,
        image_num: int,
        mask_name: str,
        output_dir: str,
    ) -> None:
        """
        Register a moving image to a fixed image using a mask.
        """

        reg_images = RegisterImagesICON()
        reg_images.set_modality("ct")
        reg_images.set_number_of_iterations(20)

        if moving_mask is not None:
            itk.imwrite(
                moving_mask,
                f"{output_dir}/{case_name}_T{image_num * 10:02d}_{mask_name}_mask_org.mha",
                compression=True,
            )

        print("Registering image...")
        reg_images.set_fixed_image(fixed_image)
        moving_mask_d = None
        if fixed_mask is not None:
            fixed_mask_d = dilate_mask(fixed_mask, heart_mask_dilation)
            moving_mask_d = dilate_mask(moving_mask, heart_mask_dilation)
            reg_images.set_fixed_mask(fixed_mask_d)
        results = reg_images.register(moving_image, moving_mask_d)
        inverse_transform = results["inverse_transform"]
        forward_transform = results["forward_transform"]
        print("Registering image...Done!")
        moving_image_reg = TransformTools().transform_image(
            moving_image, forward_transform, fixed_image, "sinc"
        )  # Final resampling with sinc
        itk.imwrite(
            moving_image_reg,
            f"{output_dir}/{case_name}_T{image_num * 10:02d}_{mask_name}_reg.mha",
            compression=True,
        )

        itk.transformwrite(
            [forward_transform],
            f"{output_dir}/{case_name}_T{image_num * 10:02d}_{mask_name}_forward.hdf",
            compression=True,
        )

        itk.transformwrite(
            [inverse_transform],
            f"{output_dir}/{case_name}_T{image_num * 10:02d}_{mask_name}_inverse.hdf",
            compression=True,
        )

    # %%
    seg_image = SegmentChestTotalSegmentator()

    os.makedirs(output_dir, exist_ok=True)

    for case_name in case_names:
        # .mha files are DirLab-4DCT data already converted to HU by
        # data/DirLab-4DCT/fix_downloaded_data.py.
        fixed_image_filename = (
            f"{input_dir}/{case_name}_T{fixed_image_num * 10:02d}.mha"
        )
        fixed_image = itk.imread(fixed_image_filename)

        print("Segmenting fixed image...")
        fixed_result = seg_image.segment(fixed_image)
        fixed_image_mask = fixed_result["labelmap"]
        fixed_image_lung_mask = fixed_result["lung"]
        fixed_image_heart_mask = fixed_result["heart"]
        fixed_image_major_vessels_mask = fixed_result["major_vessels"]
        fixed_image_bone_mask = fixed_result["bone"]
        fixed_image_soft_tissue_mask = fixed_result["soft_tissue"]
        fixed_image_other_mask = fixed_result["other"]
        fixed_image_contrast_mask = fixed_result["contrast"]

        itk.imwrite(
            fixed_image_mask,
            f"{output_dir}/{case_name}_T{fixed_image_num * 10:02d}_mask_org.mha",
            compression=True,
        )

        # Dynamic anatomy = lung (the structure that moves with respiration)
        lung_mask_arr = itk.array_from_image(fixed_image_lung_mask)
        fixed_image_dynamic_anatomy_mask_arr = lung_mask_arr
        fixed_image_dynamic_anatomy_mask = itk.image_from_array(
            fixed_image_dynamic_anatomy_mask_arr.astype(np.uint16)
        )
        fixed_image_dynamic_anatomy_mask.CopyInformation(fixed_image_mask)

        # Static anatomy = heart, major vessels, contrast, bone, other (all non-lung)
        heart_mask_arr = itk.array_from_image(fixed_image_heart_mask)
        major_vessels_mask_arr = itk.array_from_image(fixed_image_major_vessels_mask)
        contrast_mask_arr = itk.array_from_image(fixed_image_contrast_mask)
        bone_mask_arr = itk.array_from_image(fixed_image_bone_mask)
        other_mask_arr = itk.array_from_image(fixed_image_other_mask)
        fixed_image_static_anatomy_mask_arr = (
            heart_mask_arr
            + major_vessels_mask_arr
            + contrast_mask_arr
            + bone_mask_arr
            + other_mask_arr
        )
        fixed_image_static_anatomy_mask = itk.image_from_array(
            fixed_image_static_anatomy_mask_arr.astype(np.uint16)
        )
        fixed_image_static_anatomy_mask.CopyInformation(fixed_image_mask)
        print("Segmenting fixed image...Done!")

        for image_num in images:
            if image_num != fixed_image_num:
                moving_image = itk.imread(
                    os.path.join(input_dir, f"{case_name}_T{image_num * 10:02d}.mha")
                )

                print("***")
                print(
                    "*** Processing case:", case_name, "Image number:", image_num, "***"
                )
                print("***")

                print("Segmenting moving image...")
                moving_result = seg_image.segment(moving_image)
                moving_image_mask = moving_result["labelmap"]
                moving_image_lung_mask = moving_result["lung"]
                moving_image_heart_mask = moving_result["heart"]
                moving_image_major_vessels_mask = moving_result["major_vessels"]
                moving_image_bone_mask = moving_result["bone"]
                moving_image_soft_tissue_mask = moving_result["soft_tissue"]
                moving_image_other_mask = moving_result["other"]
                moving_image_contrast_mask = moving_result["contrast"]

                # Create heart mask by including major vessels and contrast masks
                lung_mask_arr = itk.array_from_image(moving_image_lung_mask)
                moving_image_dynamic_anatomy_mask_arr = lung_mask_arr
                moving_image_dynamic_anatomy_mask = itk.image_from_array(
                    moving_image_dynamic_anatomy_mask_arr.astype(np.uint16)
                )
                moving_image_dynamic_anatomy_mask.CopyInformation(moving_image_mask)

                # Create other mask by including lung, bone and soft tissue masks
                heart_mask_arr = itk.array_from_image(moving_image_heart_mask)
                major_vessels_mask_arr = itk.array_from_image(
                    moving_image_major_vessels_mask
                )
                contrast_mask_arr = itk.array_from_image(moving_image_contrast_mask)
                bone_mask_arr = itk.array_from_image(moving_image_bone_mask)
                other_mask_arr = itk.array_from_image(moving_image_other_mask)
                moving_image_static_anatomy_mask_arr = (
                    heart_mask_arr
                    + major_vessels_mask_arr
                    + contrast_mask_arr
                    + bone_mask_arr
                    + other_mask_arr
                )
                moving_image_static_anatomy_mask = itk.image_from_array(
                    moving_image_static_anatomy_mask_arr.astype(np.uint16)
                )
                moving_image_static_anatomy_mask.CopyInformation(moving_image_mask)

                print("Segmenting moving image...Done!")

                itk.imwrite(
                    moving_image_mask,
                    f"{output_dir}/{case_name}_T{image_num * 10:02d}_all_mask_org.mha",
                    compression=True,
                )

                print("Registering with All mask...")
                # all
                register_image(
                    fixed_image,
                    None,
                    moving_image,
                    None,
                    case_name,
                    image_num,
                    "all",
                    output_dir,
                )
                print("Registering with All mask...Done!")

                print("Registering with Dynamic Anatomy mask...")
                # Lungs
                register_image(
                    fixed_image,
                    fixed_image_dynamic_anatomy_mask,
                    moving_image,
                    moving_image_dynamic_anatomy_mask,
                    case_name,
                    image_num,
                    "dynamic_anatomy",
                    output_dir,
                )
                print("Registering with Dynamic Anatomy mask...Done!")

                print("Registering with Static Anatomy mask...")
                # Bone
                register_image(
                    fixed_image,
                    fixed_image_static_anatomy_mask,
                    moving_image,
                    moving_image_static_anatomy_mask,
                    case_name,
                    image_num,
                    "static_anatomy",
                    output_dir,
                )
                print("Registering with Static Anatomy mask...Done!")

            else:
                print("Baseline image: no segmentation or registration...")
                identity_transform = itk.CenteredAffineTransform[itk.D, 3].New()
                composite_transform = itk.CompositeTransform[itk.D, 3].New()
                composite_transform.AddTransform(identity_transform)

                for mask, mask_name in [
                    (fixed_image_mask, "all"),
                    (fixed_image_static_anatomy_mask, "static_anatomy"),
                    (fixed_image_dynamic_anatomy_mask, "dynamic_anatomy"),
                ]:
                    itk.imwrite(
                        mask,
                        f"{output_dir}/{case_name}_T{image_num * 10:02d}_{mask_name}_mask_org.mha",
                        compression=True,
                    )

                    itk.imwrite(
                        fixed_image,
                        f"{output_dir}/{case_name}_T{image_num * 10:02d}_{mask_name}_reg.mha",
                        compression=True,
                    )

                    itk.transformwrite(
                        [composite_transform],
                        f"{output_dir}/{case_name}_T{image_num * 10:02d}_{mask_name}_forward.hdf",
                        compression=True,
                    )

                    itk.transformwrite(
                        [composite_transform],
                        f"{output_dir}/{case_name}_T{image_num * 10:02d}_{mask_name}_inverse.hdf",
                        compression=True,
                    )

                print("Baseline image: no segmentation or registration...Done!")
