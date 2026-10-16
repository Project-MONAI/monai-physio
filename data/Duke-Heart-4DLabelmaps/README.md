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

## Effect on the tutorials

Tutorials that depend on this dataset are named with a `duke_heart` prefix in
their organ field, for example:

- `tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py`

There are ten of them. Nine form their own chain: Tutorial 4 (duke heart) -> 5
-> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12. The tenth, Tutorial 2 (duke heart), is a
separate optional ICON finetuning variant that the chain does not require.
Download the data before running them. The other 19 tutorial scripts use
publicly available datasets and are unaffected - see [../README.md](../README.md)
for download instructions.

## Expected layout

The data is placed under `data/Duke-Heart-4DLabelmaps/` as one directory per
case (`pm0002/`, `pm0003/`, ...), each holding one labelmap and one landmark
file per gated frame.
