# TCIA-4DLung

## Download

**Automatic (tutorial subset)** - a converted subset of the collection,
already split into per-case gated-phase volumes, is fetched by
`monai-physio-download-data`:

```
monai-physio-download-data TCIA-4DLung --directory data/TCIA-4DLung
```

This downloads and extracts the two zip archives attached to the MONAI
Physio `2026.07.1` GitHub release
(https://github.com/Project-MONAI/monai-physio/releases/tag/2026.07.1),
giving each case's `<case>_HM10395/<case>_HM10395_g0??.nii.gz` phase
volumes directly under `data/TCIA-4DLung/`. Re-running is a no-op once any
case's phase volumes are present. Check the layout with:

```python
from monai_physio import DownloadData

assert DownloadData.VerifyTCIA4DLungData("data/TCIA-4DLung")
```

**Manual (full collection)** - the automatic download is a tutorial
subset, not the full TCIA 4D-Lung collection. To obtain more cases, or the
raw DICOM (including RTSTRUCT contours) instead of pre-converted NIfTI:

1. Visit the TCIA 4D-Lung collection page and download the DICOM series:
   https://www.cancerimagingarchive.net/collection/4d-lung/
2. Place each patient's DICOM series under
   `data/TCIA-4DLung/download/<case>_HM10395/` (see "Structure" below for
   the expected DICOM layout).
3. Run `data/TCIA-4DLung/convert.py`. It reads every `download/*_HM10395`
   patient directory, converts each `CT_*` series with `dcm2niix`, and
   writes `<case>_HM10395/<case>_HM10395_g0??.nii.gz` alongside the
   download directory - the same layout the automatic download produces.
   `dcm2niix` is expected at the hardcoded path in `convert.py`; update it
   for your machine before running.

## Overview

Longitudinal 4D (respiratory-gated, phase-resolved) fan-beam CT of
locally-advanced non-small-cell lung cancer (NSCLC) patients, with expert
manual RTSTRUCT contours where present, from *Data from 4D Lung Imaging of
NSCLC Patients (4D-Lung)* on The Cancer Imaging Archive (Hugo et al., VCU).
See `TCIA-README.md` in this directory for the full dataset card (fields,
segmentation labels, DICOM structure) describing the source mirror this
subset and the manual download are drawn from.

### Dataset Details

- **Format**: DICOM (CT + RTSTRUCT) upstream; `.nii.gz` per gated phase
  once converted
- **Cases**: patient directories named `<n>_HM10395`
- **Phases**: gated respiratory phases per case, named `g000`-`g090`
  (0%-90% of the breathing cycle)
- **Content**: respiratory-gated lung CT
- **Anatomy**: lungs, airways, thoracic structures; RTSTRUCT contours
  (tumor and involved-node targets, partial organs-at-risk) on the full
  collection's DICOM series

### Citation

Dataset: https://www.cancerimagingarchive.net/collection/4d-lung/,
DOI `10.7937/K9/TCIA.2016.ELN8YGLE`

If you use this dataset, please cite:

- Hugo GD, Weiss E, Sleeman WC, Balik S, Keall PJ, Lu J, Williamson JF.
  2017. "A longitudinal four-dimensional computed tomography and cone beam
  computed tomography dataset for image-guided radiation therapy research
  in lung cancer." *Medical Physics* 44(2):762-771.
- Hugo GD, Weiss E, Sleeman WC, Balik S, Keall PJ, Lu J, Williamson JF.
  2016. "Data from 4D Lung Imaging of NSCLC Patients (4D-Lung) [Data
  set]." The Cancer Imaging Archive.
- Clark K, Vendt B, Smith K, et al. 2013. "The Cancer Imaging Archive
  (TCIA): Maintaining and Operating a Public Information Repository."
  *Journal of Digital Imaging* 26(6):1045-1057.

## Using This Dataset

- Primary dataset for the lung tutorials (Lung Tutorials 1, 2, 3, 4, 6, 8,
  and 10-12) and the lung 4D-CT reconstruction/registration workflows
- Statistical shape model fitting and PhysicsNeMo surrogate training

### Files in This Directory

- `<case>_HM10395/<case>_HM10395_g0??.nii.gz` - per-case gated-phase
  volumes, what tutorials and workflows actually read
- `convert.py` - converts `download/*_HM10395` DICOM series to the
  `.nii.gz` layout above (manual/full-collection path)
- `TCIA-README.md` - full dataset card for the source mirror (fields,
  segmentation labels, DICOM structure, citations)
- `LICENSE.txt` - dataset license (CC BY 3.0)

### Structure

```
data/TCIA-4DLung/
├── 100_HM10395/
│   ├── 100_HM10395_g000.nii.gz
│   ├── 100_HM10395_g010.nii.gz
│   ...
├── 101_HM10395/
...
├── download/                      # only present for the manual/full path
│   └── <case>_HM10395/<StudyInstanceUID>/CT_<SeriesInstanceUID>/*.dcm
├── convert.py
├── TCIA-README.md
├── LICENSE.txt
└── README.md (this file)
```
