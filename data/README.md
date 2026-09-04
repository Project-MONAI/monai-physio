# MONAI Physio Data Directory

This directory holds the sample datasets used for experiments, testing, and
development of the MONAI Physio library. Each subdirectory contains one
dataset and its own `README.md` with download instructions, specifications,
and citation — this file is just an index; treat the per-dataset READMEs as
the source of truth.

## Datasets

| Directory | Description | Provided By | Download | README |
| --- | --- | --- | --- | --- |
| `Slicer-Heart-CT/` | 4D cardiac CT with gated cardiac phases | Jolley Lab, Children's Hospital of Philadelphia (CHOP) | Automatic | [Slicer-Heart-CT/README.md](Slicer-Heart-CT/README.md) |
| `DirLab-4DCT/` | 4D lung CT respiratory motion benchmark | DIR-Lab, MD Anderson Cancer Center / Emory University | Manual | [DirLab-4DCT/README.md](DirLab-4DCT/README.md) |
| `KCL-Heart-Model/` | Statistical shape model of the heart | King's College London (KCL) | Automatic | [KCL-Heart-Model/README.md](KCL-Heart-Model/README.md) |
| `CHOP-Valve4D/` | 4D valve reconstruction models | Jolley Lab, CHOP (original FEBio model) | Automatic | [CHOP-Valve4D/README.md](CHOP-Valve4D/README.md) |
| `Chest-CT/` | Ungated 3D chest CT, single static volume | AREN0534 trial, The Cancer Imaging Archive (TCIA) | Automatic | [Chest-CT/README.md](Chest-CT/README.md) |
| `test/` | pytest-managed cache; not a downloadable dataset | — | N/A | [test/README.md](test/README.md) |

## Automatic Download

`Slicer-Heart-CT`, `KCL-Heart-Model`, `CHOP-Valve4D`, and `Chest-CT` can be
fetched with the `monai-physio-download-data` CLI or `DataDownloadTools`; see each
dataset's README for the exact command. `DirLab-4DCT` has no automatic
downloader — DIR-Lab distributes each case individually and may require
registration, so it must be obtained manually; see
[DirLab-4DCT/README.md](DirLab-4DCT/README.md).

## Keeping the Data Outside the Clone

The tutorials resolve three roots through `ParametersBase` in
[`tutorials/parameters_base.py`](../tutorials/parameters_base.py), each
defaulting to its location in the clone and each overridable by environment
variable. Point them elsewhere when the data should outlive the checkout — on a
CI runner, for instance, where every run starts from a fresh working tree:

| Variable | Default | Holds |
| --- | --- | --- |
| `MONAI_PHYSIO_INPUT_DATA_DIR` | `<repo>/data` | The datasets in this directory |
| `MONAI_PHYSIO_OUTPUT_DATA_DIR` | `<repo>/tutorials/output` | What the tutorials write |
| `MONAI_PHYSIO_WEIGHTS_DIR` | `<repo>/tutorials/network_weights` | Networks the tutorials train |

Each root has a `test` subdirectory used when a tutorial runs under
`MONAI_PHYSIO_RUNNING_AS_TEST`, so a test run reads the small downsampled subsets
and writes beside them rather than touching a full run's datasets, results, or
checkpoints. The subsets under `<input root>/test` are built on demand by the
fixtures in `tests/conftest.py`; putting that root outside the clone means they
survive a checkout and are built once rather than every run.

The layout under an overridden input root is the same as here:

```text
<MONAI_PHYSIO_INPUT_DATA_DIR>/
  DirLab-4DCT/             Case1Pack_T00.mha, ...
  Duke-Heart-4DLabelmaps/  pm0027/*_labelmap.nii.gz, *_landmark.mrk.json
  Chest-CT/                Chest-CT.mha
  KCL-Heart-Model/         average_mesh.vtk, input_meshes/
  Slicer-Heart-CT/         slice_000.mha, ...
  test/                    built by the pytest fixtures
```

## Notes

- Always cite the original data source in publications — see each dataset's
  README for the required citation.
- The full set of datasets is ~10-20 GB; ensure adequate disk space before
  downloading everything.
