# %%
from pathlib import Path

from data_dirlab_4d_ct import DataDirLab4DCT
from pxr import Usd

from monai_physio.segment_nv_segment_ct_mri import SegmentNVSegmentCTMRI
from monai_physio.usd_anatomy_tools import USDAnatomyTools

# Defensive: today this script only instantiates SegmentNVSegmentCTMRI to read
# its anatomy labels for USDAnatomyTools, but if anyone adds a
# `seg.segment(...)` call the model pipeline's MONAI DataLoader may spawn
# worker processes, which re-import the script on Windows (spawn start method)
# and crash with a spawn-cascade RuntimeError. Guard pre-emptively.
if __name__ == "__main__":
    case_names = DataDirLab4DCT().case_names

    case_names = [case_names[4]]

    output_dir = Path(__file__).parent / "results"

    # %%
    seg = SegmentNVSegmentCTMRI()

    for anatomy in ["all", "static_anatomy", "dynamic_anatomy"]:
        for case_name in case_names:
            stage = Usd.Stage.Open(f"{output_dir}/{case_name}_{anatomy}_lungGated.usd")
            painter = USDAnatomyTools(stage)
            painter.enhance_meshes(seg)
            stage.Export(f"{output_dir}/{case_name}_{anatomy}_lungGated_painted.usd")
