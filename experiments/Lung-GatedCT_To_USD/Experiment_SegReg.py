# %%
import os

import itk

from monai_physio import RegisterImagesICON
from monai_physio import SegmentChestTotalSegmentator

# nnUNetv2 (used by TotalSegmentator) spawns a multiprocessing.Pool. On Windows
# the spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around the top-level work, that re-import fires
# segment() again and Python's spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = os.path.join(_HERE, "..", "..", "data", "DirLab-4DCT")
    _RESULTS_DIR = os.path.join(_HERE, "results_SegReg")

    # %%
    # .mha files are DirLab-4DCT data already converted to HU by
    # data/DirLab-4DCT/fix_downloaded_data.py.
    fixed_image = itk.imread(os.path.join(_DATA_DIR, "Case1Pack_T30.mha"))
    moving_image = itk.imread(os.path.join(_DATA_DIR, "Case1Pack_T00.mha"))

    # %%
    # Register images
    reg_images = RegisterImagesICON()
    reg_images.set_fixed_image(fixed_image)
    _ = reg_images.register(moving_image)
    moving_image_registered = reg_images.get_registered_image()
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    itk.imwrite(
        moving_image_registered,
        os.path.join(_RESULTS_DIR, "Experiment_reg.mha"),
        compression=True,
    )

    # %%
    img = itk.imread(os.path.join(_RESULTS_DIR, "Experiment_reg.mha"))
    tot_seg = SegmentChestTotalSegmentator()
    seg_results = tot_seg.segment(img)
    itk.imwrite(
        seg_results["labelmap"],
        os.path.join(_RESULTS_DIR, "Experiment_totseg.mha"),
        compression=True,
    )
