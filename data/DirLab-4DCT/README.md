# DirLab-4DCT

## Download

**Manual download required** — this dataset is not fetched by
`monai-physio-download-data`; there is no automatic downloader because
DIR-Lab distributes each case individually and may require registration.

1. Visit the DIR-Lab 4D-CT page and request/download the case archives:
   https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html
2. Extract each case's raw images into `data/DirLab-4DCT/downloaded_data/`
   (see layout below). The `.mhd` headers already committed in that
   directory point at those raw files, so no header-writing step is
   required — see "About the Committed `.mhd` Files" below.
3. Run `data/DirLab-4DCT/fix_downloaded_data.py`. DIR-Lab's raw volumes are
   not in Hounsfield units (see "Fixing Raw Intensities" below); this
   script reads every `.mhd` header in `downloaded_data/` with its backing
   pixel data present, corrects the intensities, and writes one compressed
   `.mha` volume per phase directly into `data/DirLab-4DCT/`. Tutorials and
   experiments read these top-level `.mha` files, not the raw `.mhd` ones.

Once populated, check the layout with:

```python
from monai_physio import DataDownloadTools

assert DataDownloadTools.VerifyDirLab4DCTData("data/DirLab-4DCT")
```

**Directory structure after download and fixing:**
```
data/DirLab-4DCT/
├── downloaded_data/
│   ├── Case1Pack/
│   │   ├── Images/              # T00-T90 phase images
│   │   ├── ExtremePhases/       # T00 and T50 (max inhale/exhale)
│   │   └── Sampled4D/           # Sampled time points
│   ├── Case1Pack_T00.mhd        # Already-committed headers, per phase
│   ├── Case1Pack_T10.mhd
│   ...
│   ├── Case10Pack/
│   └── Convert4DCTToMHD.py      # Documents how the .mhd headers were generated
├── Case1Pack_T00.mha             # Written by fix_downloaded_data.py
├── Case1Pack_T10.mha
...
├── fix_downloaded_data.py
└── README.md (this file)
```

### About the Committed `.mhd` Files

The `Case*Pack_T*.mhd` files in `downloaded_data/` are already committed to
the repository, but they are only MetaImage **headers** (a few hundred
bytes each) — for example:

```
ObjectType = Image
NDims = 3
DimSize = 256 256 94
...
ElementDataFile = Case1Pack/Images/case1_T00_s.img
```

Each header's `ElementDataFile` points at the raw image data inside the
corresponding `Case*Pack/`/`Case*Deploy/` subdirectory (`.gitignore`d
because those raw volumes are large). These `.mhd` files will not load
until you complete the manual download above and the referenced
`Case*Pack/Images/*.img` files exist alongside them.

`Convert4DCTToMHD.py` is what originally generated these headers from the
raw DIR-Lab archives. It is included for provenance/documentation only —
you do not need to run it; the `.mhd` files are already committed.

### Fixing Raw Intensities

DIR-Lab's raw `.mhd`/`.img` volumes store intensities offset by +1024 from
Hounsfield units, not real HU. `fix_downloaded_data.py` calls
`DataDownloadTools.FixDirLab4DCTData`, which subtracts 1024 and clips the
result to `[-1024, 1024]` for every downloaded phase, writing one
compressed `.mha` file per phase into `data/DirLab-4DCT/` (`.gitignore`d,
like the raw data). Run it once after downloading; tutorials and
experiments assume the `.mha` files it produces already exist.

## Overview

Benchmark dataset for 4D CT respiratory motion analysis. Contains 10 cases
of lung CT scans at different respiratory phases with annotated landmark
points for registration validation.

### Dataset Details

- **Format**: `.mhd` headers + `.img` raw volumes (MetaImage format)
- **Cases**: 10 patient cases (Case 1-10)
- **Phases**: 10 respiratory phases per case (T00-T90)
- **Content**: Non-contrast lung CT
- **Anatomy**: Lungs, airways, thoracic structures
- **Landmarks**: 300+ annotated points per case for registration validation

### Acknowledgement

Data provided by the DIR-Lab at MD Anderson Cancer Center / Emory
University:
https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/

### Citation

Dataset: https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html

If you use this dataset, please cite:

- Case numbers 4DCT1-4DCT5: Castillo R, Castillo E, Guerra R, Johnson VE,
  McPhail T, Garg AK, Guerrero T. 2009. "A framework for evaluation of
  deformable image registration spatial accuracy using large landmark
  point sets." *Phys Med Biol* 54:1849-1870.
- Case numbers 4DCT6-4DCT10: Castillo E, Castillo R, Martinez J, Shenoy M,
  Guerrero T. 2009. "Four-dimensional deformable image registration using
  trajectory modeling." *Phys Med Biol* 55:305-327.

## Using This Dataset

- Primary dataset for `experiments/Lung-GatedCT_To_USD/`
- Registration algorithm validation
- Respiratory motion analysis
- Benchmark for deformable registration accuracy

### Files in This Directory

- `fix_downloaded_data.py` — converts `downloaded_data/*.mhd` to
  HU-corrected `Case*Pack_T*.mha` files in this directory (see "Fixing Raw
  Intensities" above)
- `downloaded_data/Convert4DCTToMHD.py` — documents how the committed
  `.mhd` headers were generated from the raw DIR-Lab archives; not needed
  to run
- `downloaded_data/Case*Pack_T*.mhd` — MetaImage headers for each
  case/phase (see above)
- `Case*Pack_T*.mha` — HU-corrected volumes written by
  `fix_downloaded_data.py`; what tutorials and experiments actually read

### Additional Resources

- Reference implementation for reading DIR-Lab's raw `.img` case files:
  https://github.com/hsokooti/RegNet/blob/master/functions/preprocessing/dirlab.py
