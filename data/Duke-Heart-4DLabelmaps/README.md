# Duke-Heart-4DLabelmaps

Gated 4D cardiac labelmaps acquired at Duke University by Dr. Paul Segars.

## Availability

This dataset is **scheduled for public release soon**. It is not distributed
with this repository and has no automatic downloader yet.

In the meantime, contact Stephen Aylward (<saylward@nvidia.com>) to request
access.

## Effect on the tutorials

Tutorials that depend on this dataset are named with a `duke_heart` prefix in
their organ field, for example:

- `tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py`

There are ten of them. Nine form their own chain: Tutorial 4 (duke heart) -> 5
-> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12. The tenth, Tutorial 2 (duke heart), is a
separate optional ICON finetuning variant that the chain does not require.
They will not run until the data is
available. The other 19 tutorial scripts use publicly available datasets and are
unaffected - see [../README.md](../README.md) for download instructions.

Downstream tutorials that consume `duke_heart` outputs (such as the finetuned
distance-map ICON weights used by
`tutorials/tutorial_07_heart_fit_statistical_model_to_patient.py`) fall back to
stock weights and still run, with reduced accuracy.

## Expected layout

When available, the data is expected under `data/Duke-Heart-4DLabelmaps/` as
one directory per case (`pm0002/`, `pm0003/`, ...), each holding one labelmap
per gated frame.
