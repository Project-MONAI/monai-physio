# %%
from pathlib import Path

from data_dirlab_4d_ct import DataDirLab4DCT
from pxr import Usd

from monai_physio.process_usd_anatomy import ProcessUSDAnatomy
from monai_physio.segment_chest_total_segmentator import SegmentChestTotalSegmentator

# Defensive: today this script only instantiates SegmentChestTotalSegmentator
# to read its anatomy labels for ProcessUSDAnatomy, but if anyone adds a
# `seg.segment(...)` call it would trigger the nnUNet multiprocessing.Pool
# which re-imports the script on Windows (spawn start method) and crashes
# with a spawn-cascade RuntimeError. Guard pre-emptively.
if __name__ == "__main__":
    case_names = DataDirLab4DCT().case_names

    case_names = [case_names[4]]

    output_dir = Path(__file__).parent / "results"

    # %%
    seg = SegmentChestTotalSegmentator()

    for anatomy in ["all", "static_anatomy", "dynamic_anatomy"]:
        for case_name in case_names:
            stage = Usd.Stage.Open(f"{output_dir}/{case_name}_{anatomy}_lungGated.usd")
            painter = ProcessUSDAnatomy(stage)
            painter.enhance_meshes(seg)
            stage.Export(f"{output_dir}/{case_name}_{anatomy}_lungGated_painted.usd")
