# %%
from pathlib import Path

from monai_physio.data_download_tools import DataDownloadTools

_HERE = Path(__file__).resolve().parent

# %%
input_dir = _HERE / "downloaded_data"
output_dir = _HERE

# Converts raw DirLab-4DCT intensities (.mhd) to clipped Hounsfield units and
# writes each result as a compressed .mha file in output_dir.
DataDownloadTools.FixDirLab4DCTData(input_dir, output_dir)
