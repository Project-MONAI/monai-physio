---
license: cc-by-3.0
task_categories:
- image-segmentation
tags:
- medical
- ct
- 4dct
- lung
- nsclc
- lung-cancer
- radiotherapy
- gtv
- tumor-segmentation
- respiratory-gated
- dicom
- rtstruct
- tcia
pretty_name: 4D-Lung (segmentation subset)
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: preview
    path: data/preview-*
dataset_info:
  features:
  - name: patient_id
    dtype: string
  - name: study_uid
    dtype: string
  - name: ct_series_uid
    dtype: string
  - name: rt_series_uid
    dtype: string
  - name: phase
    dtype: int32
  - name: num_ct_slices
    dtype: int32
  - name: slice_index
    dtype: int32
  - name: gtv_voxels
    dtype: int64
  - name: node_voxels
    dtype: int64
  - name: image
    dtype: image
  - name: mask
    dtype: image
  - name: overlay
    dtype: image
  splits:
  - name: preview
    num_bytes: 191761328
    num_examples: 800
  download_size: 191647526
  dataset_size: 191761328
---

# 4D-Lung (segmentation subset)

Longitudinal 4D (respiratory-gated, phase-resolved) fan-beam CT of 20
locally-advanced non-small-cell lung cancer (NSCLC) patients, with expert
manual RTSTRUCT contours, from *Data from 4D Lung Imaging of NSCLC Patients
(4D-Lung)* on The Cancer Imaging Archive (Hugo et al., VCU).

> **This is the segmentable subset of the full collection — read carefully.**
> The full TCIA 4D-Lung collection is ~183 GB and contains both 4D fan-beam CT
> (4D-FBCT, "4DCT") and 4D cone-beam CT (4D-CBCT). **Only the 4DCT series carry
> public RTSTRUCT segmentations**; the 4DCBCT series (~134 GB) have **no public
> masks**. This mirror therefore contains **only the 820 4DCT phase series + the
> 800 RTSTRUCT series (~46 GB)** that form usable image/mask pairs. If you need
> the un-annotated 4DCBCT, download it from TCIA directly.

## Dataset Details

| Field | Value |
|---|---|
| Modality | CT (4D fan-beam, respiratory-gated; 10 breathing phases 0–90%) + RTSTRUCT contours |
| Body part | Lung / thorax — locally-advanced NSCLC |
| Task | 3D tumor-target segmentation (GTV + involved nodes) |
| Patients | 20 (`100_HM10395` … `119_HM10395`) |
| CT series (phase volumes) | 820 |
| RTSTRUCT series | 800 |
| Image/mask pairs | 800 (each RTSTRUCT references exactly one CT phase volume) |
| DICOM files | 93,830 |
| Acquisition | longitudinal — each patient imaged repeatedly across the chemo-radiotherapy course |
| Format | DICOM (CT + RTSTRUCT) |
| License | CC BY 3.0 |

## Segmentation Labels & Recommended Ground Truth

RTSTRUCT ROI names follow the convention `{Structure}_c{PP}`, where `PP` is the
breathing-phase percentage (`c00` = 0% … `c90` = 90%). Each RTSTRUCT is
phase-specific and references the matching phase CT.

**Recommended gold-standard ground truth: the radiotherapy target volume —**

| Label | ROI names | Coverage |
|---|---|---|
| **GTV (gross primary tumor)** | `Tumor_cPP` | all 800 pairs (100%) |
| **Involved lymph nodes** | `LN_cPP`, `LN2_cPP` | 720 / 800 pairs (90%) |

All contours were delineated by a single experienced radiation oncologist
(E. Weiss) under physician supervision — there is **one annotation tier** (no
multi-rater / partial-vs-full split).

The RTSTRUCT files also contain, with **partial** coverage, organs-at-risk
(`RLung`, `LLung`, `Esophagus`, `Heart`, `Cord`, `Trachea` — present on only
~101 RTSTRUCT) and landmarks/fiducials (`Carina`, `Vertebra`, `MarkerA`–`D`;
markers in 7 patients only). **These are NOT the gold target** but are preserved
intact in the raw RTSTRUCT for downstream use. A handful of ROI names have minor
typos (`Tumor_ c00`, `LN2_C10`).

## Cross-dataset Overlap

**None known.** 4D-Lung is a single-institution VCU cohort (all subjects share
the `HM10395` site suffix). It is disjoint from NSCLC-Radiomics (Maastricht
Lung1), LIDC-IDRI, and the Medical Segmentation Decathlon Lung task — no shared
patients or source archives are documented. The only identifier is the TCIA
Subject ID (`NNN_HM10395`).

## Structure

```
<PatientID>/<StudyInstanceUID>/CT_<SeriesInstanceUID>/*.dcm        # CT phase volume
<PatientID>/<StudyInstanceUID>/RTSTRUCT_<SeriesInstanceUID>/*.dcm  # contours
series_index.json                                                  # index + RT→CT pairing
LICENSE.txt
```

Each RTSTRUCT references its source CT phase series via
`ReferencedFrameOfReferenceSequence / RTReferencedSeriesSequence`. The flat
layout encodes the DICOM `Modality` as the series-folder prefix (`CT_…` /
`RTSTRUCT_…`). `series_index.json` provides, for every series, its patient /
study / modality / relative path / breathing phase, and for every RTSTRUCT its
resolved `ref_ct_uid`, ROI names, and GTV/node/OAR coverage flags — plus a
ready-made `pairs` list of the 800 RTSTRUCT→CT pairings.

## Source & Citation

- TCIA collection: https://www.cancerimagingarchive.net/collection/4d-lung/
- Data DOI: `10.7937/K9/TCIA.2016.ELN8YGLE`
- Official, author-deposited (VCU); fully public, CC BY 3.0, no registration.

```bibtex
@article{hugo2017longitudinal4dlung,
  author  = {Hugo, Geoffrey D. and Weiss, Elisabeth and Sleeman, William C. and
             Balik, Salim and Keall, Paul J. and Lu, Jun and Williamson, Jeffrey F.},
  title   = {A longitudinal four-dimensional computed tomography and cone beam
             computed tomography dataset for image-guided radiation therapy
             research in lung cancer},
  journal = {Medical Physics},
  volume  = {44},
  number  = {2},
  pages   = {762--771},
  year    = {2017},
  doi     = {10.1002/mp.12059}
}

@misc{hugo20164dlungtcia,
  author    = {Hugo, G. D. and Weiss, E. and Sleeman, W. C. and Balik, S. and
               Keall, P. J. and Lu, J. and Williamson, J. F.},
  title     = {Data from 4D Lung Imaging of NSCLC Patients (4D-Lung) [Data set]},
  year      = {2016},
  publisher = {The Cancer Imaging Archive},
  doi       = {10.7937/K9/TCIA.2016.ELN8YGLE}
}

@article{clark2013tcia,
  author  = {Clark, Kenneth and Vendt, Bruce and Smith, Kirk and others},
  title   = {The Cancer Imaging Archive (TCIA): Maintaining and Operating a
             Public Information Repository},
  journal = {Journal of Digital Imaging},
  volume  = {26},
  number  = {6},
  pages   = {1045--1057},
  year    = {2013},
  doi     = {10.1007/s10278-013-9622-7}
}
```
