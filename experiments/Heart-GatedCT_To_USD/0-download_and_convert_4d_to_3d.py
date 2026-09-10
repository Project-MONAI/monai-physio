# %%
import shutil
from pathlib import Path

from monai_physio.download_data import DownloadData

_HERE = Path(__file__).resolve().parent

# %%
data_dir = _HERE.parent.parent / "data" / "Slicer-Heart-CT"
output_dir = _HERE / "results"

output_dir.mkdir(parents=True, exist_ok=True)

# Downloads TruncalValve_4DCT.seq.nrrd and splits it into slice_???.mha.
DownloadData.DownloadSlicerHeartCTData(data_dir)

# %%
# Save the mid-stroke slice as the fixed/reference image
shutil.copyfile(data_dir / "slice_007.mha", output_dir / "slice_fixed.mha")
