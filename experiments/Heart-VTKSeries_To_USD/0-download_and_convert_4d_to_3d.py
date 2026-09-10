# %%
import os

from monai_physio.download_data import DownloadData

_HERE = os.path.dirname(os.path.abspath(__file__))

# %%
data_dir = os.path.join(_HERE, "..", "..", "data", "Slicer-Heart-CT")
output_dir = os.path.join(_HERE, "results")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Downloads TruncalValve_4DCT.seq.nrrd and splits it into slice_???.mha.
DownloadData.DownloadSlicerHeartCTData(data_dir)
