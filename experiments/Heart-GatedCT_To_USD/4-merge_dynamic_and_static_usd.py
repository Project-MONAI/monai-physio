# %%
import os

from monai_physio.usd_tools import USDTools

_HERE = os.path.dirname(os.path.abspath(__file__))

# %%
usd_tools = USDTools()

_merged = os.path.join(_HERE, "results", "Slicer_CardiacGatedCT.merged_painted.usd")
_dynamic = os.path.join(
    _HERE, "results", "Slicer_CardiacGatedCT.dynamic_anatomy_painted.usd"
)
_static = os.path.join(
    _HERE, "results", "Slicer_CardiacGatedCT.static_anatomy_painted.usd"
)

if os.path.exists(_merged):
    os.remove(_merged)

usd_tools.merge_usd_files(
    _merged,
    [_dynamic, _static],
)

_flattened = os.path.join(
    _HERE, "results", "Slicer_CardiacGatedCT.flattened_merged_painted.usd"
)

if os.path.exists(_flattened):
    os.remove(_flattened)

usd_tools.merge_usd_files_flattened(
    _flattened,
    [_dynamic, _static],
)
