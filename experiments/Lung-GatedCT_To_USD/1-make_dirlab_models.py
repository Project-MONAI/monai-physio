# %%
from pathlib import Path

import itk
import numpy as np
import pyvista as pv
from data_dirlab_4d_ct import DataDirLab4DCT

from monai_physio.contour_tools import ContourTools
from monai_physio import ConvertVTKToUSD
from monai_physio.segment_chest_total_segmentator import SegmentChestTotalSegmentator

# Defensive: today this script only reads `seg.all_mask_ids`, but if anyone
# adds a `seg.segment(...)` call it would trigger the nnUNet
# multiprocessing.Pool which re-imports the script on Windows (spawn start
# method) and crashes with a spawn-cascade RuntimeError. Guard pre-emptively.
if __name__ == "__main__":
    case_names = DataDirLab4DCT().case_names
    case_names = [case_names[4]]

    base_timepoint = 30

    output_dir = Path(__file__).parent / "results"

    # %%
    def transform_contours_list(
        contours: pv.PolyData, case_name: str, mask_name: str, output_dir: str
    ):
        """
        Transform a list of contours to a list of transformed contours.
        """
        con_tools = ContourTools()
        new_contours = []
        for i in range(10):
            moving_to_fixed_transform = itk.transformread(
                f"{output_dir}/{case_name}_T{i * 10:02d}_{mask_name}_inverse.hdf"
            )[0]

            print(f"Transforming {case_name} - {mask_name} - T{i * 10:02d}")
            new_contours.append(
                con_tools.transform_contours(contours, moving_to_fixed_transform)
            )

        return new_contours

    # %%
    def make_dirlab_models(
        output_dir,
        label,
        case_name,
        base_timepoint,
        all_labelmap_arr,
        all_mask_ids,
        con_tools,
        seg,
    ):
        """
        Make DirLab models for a list of cases.
        """
        labelmap_image = itk.imread(
            f"{output_dir}/{case_name}_T{base_timepoint}_{label}_mask_org.mha",
            pixel_type=itk.UC,
        )
        labelmap_arr = itk.array_view_from_image(labelmap_image)

        print(f"Extracting contours from {case_name} - {label} Contours")
        label_labelmap_arr = np.where(labelmap_arr > 0, all_labelmap_arr, 0).astype(
            np.uint8
        )
        label_labelmap_image = itk.image_from_array(label_labelmap_arr)
        label_labelmap_image.CopyInformation(labelmap_image)

        contours = con_tools.extract_contours(label_labelmap_image)
        contours.save(
            f"{output_dir}/{case_name}_T{base_timepoint}_{label}_lungGatedBase.vtp",
            binary=True,
        )

        print(f"Applying transforms to vtp models from {case_name}")
        transformed_contours = transform_contours_list(
            contours, case_name, label, output_dir
        )

        print(f"Converting vtp models to USD for {case_name}")
        # Forwarding `seg` groups labels by anatomy type under
        # /World/DirLab4DCT/{type}/{label_name}.
        converter = ConvertVTKToUSD(
            "DirLab4DCT",
            transformed_contours,
            mask_ids=all_mask_ids,
            segmenter=seg,
        )
        converter.convert(
            f"{output_dir}/{case_name}_{label}_lungGated.usd",
            convert_to_surface=True,
        )

    # %%
    con_tools = ContourTools()

    seg = SegmentChestTotalSegmentator()
    for case_name in case_names:
        # all labelmap
        all_labelmap = itk.imread(
            f"{output_dir}/{case_name}_T{base_timepoint}_all_mask_org.mha",
            pixel_type=itk.UC,
        )
        all_labelmap_arr = itk.array_view_from_image(all_labelmap)
        all_mask_ids = seg.taxonomy.all_labels()

        for label in ["all", "static_anatomy", "dynamic_anatomy"]:
            make_dirlab_models(
                output_dir,
                label,
                case_name,
                base_timepoint,
                all_labelmap_arr,
                all_mask_ids,
                con_tools,
                seg,
            )
