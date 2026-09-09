# Chest-CT

## Download

Download this dataset automatically with:

```bash
monai-physio-download-data Chest-CT --directory data/Chest-CT
```

or from Python:

```python
from monai_physio import DataDownloadTools

data_file = DataDownloadTools.DownloadChestCTData("data/Chest-CT")
assert DataDownloadTools.VerifyChestCTData("data/Chest-CT")
```

This fetches a single ~200 MB file from the MONAI Physio GitHub release
[2026.07.1](https://github.com/Project-MONAI/monai-physio/releases/download/2026.07.1/Chest-CT.mha).
An existing non-empty `Chest-CT.mha` is reused, so re-running the command
resumes an interrupted download.

**Directory structure after download:**
```text
data/Chest-CT/
├── Chest-CT.mha
└── README.md (this file)
```

## Overview

A single-acquisition 3D chest CT from the AREN0534 pediatric Wilms tumor
trial. Unlike the gated 4D datasets in this directory, it is a single static
volume - one acquisition, no temporal phases - so it stands in for the ungated
clinical scan a patient-specific model is fitted to.

### Dataset Details

- **Format**: `.mha` (compressed MetaImage)
- **Dimensionality**: 3D, single time point
- **Size**: ~200 MB
- **Content**: Ungated chest CT
- **Anatomy**: Lungs, heart, mediastinum, thoracic skeleton

### Acknowledgement

Data provided by The Cancer Imaging Archive (TCIA):
https://www.cancerimagingarchive.net/

Released under the NCTN Data Archive License; see the collection page linked
from the DOI below for the terms.

### Citation

Dataset: https://doi.org/10.7937/TCIA.5M9S-6Y97

If you use this dataset, please cite:

> Ehrlich, P., Chi, Y. Y., Chintagumpala, M. M., Hoffer, F. A., Perlman, E. J., Kalapurakal, J. A., Warwick, A., Shamberger, R. C., Khanna, G., Hamilton, T. E., Gow, K. W., Paulino, A. C., Gratias, E. J., Mullen, E. A., Geller, J. I., Grundy, P. E., Fernandez, C. V., Ritchey, M. L., & Dome, J. S. (2021). Combination Chemotherapy and Surgery in Treating Young Patients With Wilms Tumor (AREN0534) [Data set]. The Cancer Imaging Archive. DOI: [10.7937/TCIA.5M9S-6Y97](https://doi.org/10.7937/TCIA.5M9S-6Y97)

TCIA's data usage policy also asks that the archive itself be cited:

> Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. *Journal of Digital Imaging*, 26(6), 1045-1057. DOI: [10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)

## Using This Dataset

- Patient image for
  `tutorials/tutorial_07_lung_fit_statistical_model_to_patient.py`, which
  segments the lungs from this scan and fits the lung PCA shape model built
  by `tutorials/tutorial_06_lung_create_statistical_model.py` to them

### Files in This Directory

- `Chest-CT.mha` - the downloaded chest CT volume
