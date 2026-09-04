# %%
import os

import itk
import pyvista as pv

from monai_physio import ConvertVTKToUSD
from monai_physio.contour_tools import ContourTools
from monai_physio.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from monai_physio.usd_anatomy_tools import USDAnatomyTools

# Defensive: this script only reads `seg.all_mask_ids` today, but if anyone
# ever adds a `seg.segment(...)` call it would trigger the nnUNet
# multiprocessing.Pool which re-imports the script on Windows (spawn start
# method) and crashes with a spawn-cascade RuntimeError. Guard pre-emptively.
if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))

    # Must match N_FRAMES / FRAME_STEP in 1-register_images.py
    N_FRAMES = 21
    FRAME_STEP = 1

    # %%
    output_dir = os.path.join(_HERE, "results")

    base_name = "slice_fixed"
    # base_name = "slice_max.reg_dynamic_anatomy"

    project_name = "Slicer_CardiacGatedCT"

    do_transform_contours = True

    # %%
    def transform_contours(
        contours, transform_filenames, frame_indices, base_name, output_dir
    ):
        con = ContourTools()
        for i, transform_filename in zip(frame_indices, transform_filenames):
            forward_transform = itk.transformread(transform_filename)[0]
            print(f"Applying transform {transform_filename} to {base_name}")

            new_contours = con.transform_contours(
                contours, forward_transform, with_deformation_magnitude=True
            )
            new_contours.save(
                os.path.join(
                    output_dir,
                    f"slice_{i:03d}.reg_{base_name}_inv.{base_name}_mask.vtp",
                )
            )

    def convert_contours(base_name, output_dir, project_name, compute_normals=False):
        files = [
            f"{output_dir}/slice_{i:03d}.reg_{base_name}_inv.{base_name}_mask.vtp"
            for i in range(0, N_FRAMES, FRAME_STEP)
        ]
        seg = SegmentChestTotalSegmentator()
        all_mask_ids = seg.taxonomy.all_labels()

        polydata = [pv.read(f) for f in files]

        # For cardiac gated CT data with 21 time points (0-20), each frame = 1 second
        # so we use frames_per_second=1.0 instead of the default 24.0.
        # Forwarding `seg` groups labels by anatomy type under
        # /World/{project_name}/{type}/{label_name}.
        converter = ConvertVTKToUSD(
            project_name,
            polydata,
            all_mask_ids,
            segmenter=seg,
            compute_normals=compute_normals,
            frames_per_second=1.0,
        )
        stage = converter.convert(
            os.path.join(output_dir, f"{project_name}.{base_name}.usd"),
        )

        painter = USDAnatomyTools(stage)
        painter.enhance_meshes(seg)
        if os.path.exists(
            os.path.join(output_dir, f"{project_name}.{base_name}_painted.usd")
        ):
            os.remove(
                os.path.join(output_dir, f"{project_name}.{base_name}_painted.usd")
            )
        stage.Export(
            os.path.join(output_dir, f"{project_name}.{base_name}_painted.usd")
        )

    # %%
    dynamic_transform_filenames = [
        os.path.join(output_dir, f"slice_{i:03d}.reg_dynamic_anatomy.forward.hdf")
        for i in range(0, N_FRAMES, FRAME_STEP)
    ]
    static_transform_filenames = [
        os.path.join(output_dir, f"slice_{i:03d}.reg_static_anatomy.forward.hdf")
        for i in range(0, N_FRAMES, FRAME_STEP)
    ]
    all_transform_filenames = [
        os.path.join(output_dir, f"slice_{i:03d}.reg_all.forward.hdf")
        for i in range(0, N_FRAMES, FRAME_STEP)
    ]

    dynamic_anatomy_contours = pv.read(
        os.path.join(output_dir, f"{base_name}.dynamic_anatomy_mask.vtp")
    )
    static_anatomy_contours = pv.read(
        os.path.join(output_dir, f"{base_name}.static_anatomy_mask.vtp")
    )
    all_contours = pv.read(os.path.join(output_dir, f"{base_name}.all_mask.vtp"))

    frame_indices = list(range(0, N_FRAMES, FRAME_STEP))

    # %%
    transform_contours(
        all_contours, all_transform_filenames, frame_indices, "all", output_dir
    )
    transform_contours(
        dynamic_anatomy_contours,
        dynamic_transform_filenames,
        frame_indices,
        "dynamic_anatomy",
        output_dir,
    )
    transform_contours(
        static_anatomy_contours,
        static_transform_filenames,
        frame_indices,
        "static_anatomy",
        output_dir,
    )

    # %%
    convert_contours("all", output_dir, project_name, compute_normals=True)
    convert_contours("dynamic_anatomy", output_dir, project_name, compute_normals=True)
    convert_contours("static_anatomy", output_dir, project_name, compute_normals=True)
