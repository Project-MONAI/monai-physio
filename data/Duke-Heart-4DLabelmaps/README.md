# Duke-Heart-4DLabelmaps

Gated 4D cardiac labelmaps and landmark files derived from the PROMISE study
CT data acquired at Duke University, Dr. Paul Segars' lab.

## Provenance

The labelmaps were created from the original PROMISE study CT data using the
AS Cardio (ASCardio) segmentation module in Synopsys' Simpleware application.
MONAI Physio calls that module through the `SegmentHeartSimplewareTrimmedBranches`
class.

**Important:** any AI physiological simulation model trained using these
labelmaps must be applied to labelmaps that were also generated with the
ASCardio module in Simpleware (for example via
`SegmentHeartSimplewareTrimmedBranches`). Applying such a model to labelmaps
from a different segmenter will degrade the results.

## Availability

The labelmaps (`*_labelmap.nii.gz`) and landmark files (`*_landmark.mrk.json`)
can be downloaded automatically:

```bash
monai-physio-download-data Duke-Heart-4DLabelmaps --directory data/Duke-Heart-4DLabelmaps
```

or from Python with `DownloadData.DownloadDukeHeart4DLabelmapsData`. The archive
is fetched from
<https://github.com/Project-MONAI/monai-physio/releases/download/2026.07.1/Duke-Heart-4DLabelmaps.zip>
and extracted into this directory, creating the `pm00*/` case subdirectories.
Re-running is a no-op once the data is present.

The original PROMISE CT images are not distributed with this repository.

## License

The labelmaps and landmark files are distributed under the Creative Commons
Attribution (CC BY) license. Attribution: Dr. W. Paul Segars, Center for
Virtual Imaging Trials, Duke University.

If you use this data, please cite:

> Malin EJ, Ceritoglu C, Abadi E, Ratnanather JT, Samei E, Segars WP.
> Library of realistic 4D digital beating heart models based on patient CT
> data. *Med Phys*. 2025;52(7):e17945. doi:10.1002/mp.17945.
> PMID: 40660846. PMCID: [PMC12409789](https://pmc.ncbi.nlm.nih.gov/articles/PMC12409789/).

## Effect on the tutorials

Tutorials that depend on this dataset are named with a `duke_heart` prefix in
their organ field, for example:

- `tutorials/tutorial_02_finetune_icon_distancemap_duke_heart_4d_labelmaps.py`

There are fifteen of them. Fourteen form their own chain, from Tutorial 4
(duke heart) through Tutorial 18; Tutorials 16 to 18 branch off it after
Tutorial 4 to build a volumetric shape model. The fifteenth, Tutorial 2 (duke
heart), is a separate optional ICON finetuning variant that the chain does not
require. Download the data before running them. Two Tutorial 00 notebooks,
`tutorials/tutorial_00_infer_cardiac_motion_duke_heart_4d_labelmaps.ipynb` and
`tutorials/tutorial_00_infer_physics_informed_motion_duke_heart_4d_labelmaps.ipynb`,
download this data and pretrained networks themselves and need no other
tutorial run first. The other 22 tutorial scripts use other datasets and are
unaffected - see [../README.md](../README.md) for download instructions.

## Expected layout

The data is placed under `data/Duke-Heart-4DLabelmaps/` as one directory per
case (`pm0002/`, `pm0003/`, ...), each holding one labelmap and one landmark
file per gated frame.
