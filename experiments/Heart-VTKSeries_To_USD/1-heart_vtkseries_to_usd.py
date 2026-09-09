# %%
import glob
import os

import pyvista as pv

from monai_physio import ConvertVTKToUSD

# nnUNetv2 (used by TotalSegmentator) spawns a multiprocessing.Pool. On Windows
# the spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around the top-level work, that re-import fires
# segment() again and Python's spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = os.path.join(_HERE, "..", "..", "data", "Slicer-Heart-CT")

    # %%
    if not os.path.exists(os.path.join(_DATA_DIR, "slice_000.vtp")):
        # Segment chest from CT images to generate vtk files
        import itk

        from monai_physio.contour_tools import ContourTools
        from monai_physio.segment_chest_total_segmentator_with_contrast import (
            SegmentChestTotalSegmentatorWithContrast,
        )

        input_images = sorted(glob.glob(os.path.join(_DATA_DIR, "slice_*.mha")))
        seg = SegmentChestTotalSegmentatorWithContrast()
        con = ContourTools()
        for i, img_path in enumerate(input_images):
            print(f"Segmenting {img_path}...")
            img = itk.imread(img_path)
            result = seg.segment(img)
            labelmap_mask = result["labelmap"]
            img_con = con.extract_contours(labelmap_mask)
            img_con.save(os.path.join(_DATA_DIR, f"slice_{i:03d}.vtp"))

    # %%
    project_name = "Heart_VTKSeries_To_USD_all"

    input_files = sorted(glob.glob(os.path.join(_DATA_DIR, "slice_*.vtp")))

    output_dir = os.path.join(_HERE, "results")
    os.makedirs(output_dir, exist_ok=True)

    input_polydata = []
    for file in input_files:
        print(f"Processing file: {file}")
        pd = pv.read(file)
        print(f"  Number of points: {pd.n_points}")
        # Print available data arrays
        if len(input_polydata) == 0:  # Only print for first file
            print(f"  Point data arrays: {list(pd.point_data.keys())}")
        input_polydata.append(pd)

    # Convert with transmembrane potential coloring
    converter = ConvertVTKToUSD(
        project_name,
        input_polydata,
    )
    # converter.set_colormap(
    # color_by_array='transmembrane_potential',
    # colormap='viridis',
    # intensity_range=(-80.0, 40.0),
    # )
    stage = converter.convert(
        os.path.join(output_dir, f"{project_name}.usd"), convert_to_surface=True
    )

    print("\nUSD files created!")
    print(f"  - {os.path.join(output_dir, f'{project_name}.usd')}")
