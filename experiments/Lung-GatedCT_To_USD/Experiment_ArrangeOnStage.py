# %%
import os

from data_dirlab_4d_ct import DataDirLab4DCT

from monai_physio.tools_for_usd import ToolsForUSD

# %%
os.makedirs("Results_ArrangeOnStage", exist_ok=True)

# %%
case_names = [
    DataDirLab4DCT().get_case_names()[0],
    DataDirLab4DCT().get_case_names()[1],
]

usd_tools = ToolsForUSD()

for label in ["dynamic_anatomy", "static_anatomy"]:
    usd_file_names = [
        f"results/{case_name}_{label}_lungGated_painted.usd" for case_name in case_names
    ]
    usd_tools.save_usd_file_arrangement(
        f"Results_ArrangeOnStage/stage-{label}.usd", usd_file_names
    )
