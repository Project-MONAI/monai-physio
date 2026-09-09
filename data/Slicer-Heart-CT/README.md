# Slicer-Heart-CT

## Download

Download this dataset automatically with:

```bash
monai-physio-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT
```

or from Python:

```python
from monai_physio import DataDownloadTools

data_file = DataDownloadTools.DownloadSlicerHeartCTData("data/Slicer-Heart-CT")
assert DataDownloadTools.VerifySlicerHeartCTData("data/Slicer-Heart-CT")
```

This fetches a single ~1.2 GB file from
[the MONAI Physio release assets](https://github.com/Project-MONAI/monai-physio/releases/download/2026.07.1/TruncalValve_4DCT.seq.nrrd),
then splits it into per-phase 3D `slice_???.mha` volumes in the same
directory via `ConvertImage4DTo3D`. An existing non-empty `.seq.nrrd` is
reused, and the split is skipped once the `slice_???.mha` files are present -
so re-running the command resumes an interrupted download or conversion.

**Directory structure after download:**
```text
data/Slicer-Heart-CT/
├── TruncalValve_4DCT.seq.nrrd
├── slice_000.mha ... slice_020.mha
└── README.md (this file)
```

## Overview

4D cardiac CT dataset with temporal gating showing a complete cardiac
cycle. Pediatric cardiac CT with truncal valve visualization.

### Dataset Details

- **Format**: `.seq.nrrd` (4D NRRD sequence file)
- **Phases**: 21 temporal cardiac phases
- **Size**: ~1.2 GB
- **Content**: Contrast-enhanced cardiac CT
- **Anatomy**: Heart, great vessels, thoracic structures

### Acknowledgement

Data provided by the Jolley Lab at CHOP (Children's Hospital of
Philadelphia):
- https://www.linkedin.com/company/jolleylab
- https://github.com/SlicerHeart/SlicerHeart

## Using This Dataset

- Primary dataset for tutorials
- Primary dataset for `experiments/Heart-GatedCT_To_USD/` and
  `experiments/Heart-VTKSeries_To_USD/`, whose
  `0-download_and_convert_4d_to_3d.py` scripts call
  `DataDownloadTools.DownloadSlicerHeartCTData`, which downloads this
  sequence and splits it into per-phase 3D `.mha` slices
- Used in the test suite (`tests/test_download_heart_data.py`)
- Example data for cardiac motion visualization in NVIDIA Omniverse

### Files in This Directory

- `TruncalValve_4DCT.seq.nrrd` - the downloaded 4D CT sequence
- `slice_000.mha` ... `slice_020.mha` - the 21 cardiac phases split out of
  the sequence above
