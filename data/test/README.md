# data/test

This directory is **automatically managed by the pytest infrastructure**
(`tests/conftest.py`) — it is a cache, not a dataset you download or
maintain by hand. It holds data used to run the unit test suite.

It is the `test` subdirectory of the input data root, so setting
`PHYSIOTWIN_INPUT_DATA_DIR` moves it too; see
[`data/README.md`](../README.md#keeping-the-data-outside-the-clone). The paths
below are written relative to that root, which defaults to `data/`.

The workflows, tutorials, and CLIs of the PhysioTwin4D library consume the
full datasets documented in [`data/README.md`](../README.md). They read the
subsets here only when run as tests, under `PHYSIOTWIN_RUNNING_AS_TEST`.

## What Lives Here

- `slicer_heart/` — a cached copy of the `Slicer-Heart-CT` 4D CT sequence
  (see the `download_test_data` fixture), split into per-phase `.mha`
  slices by `test_download_heart_data.py`.
- `slicer_heart_small/` — the same phases downsampled to 1.5x1.5x1.5 mm,
  used by tests that need a smaller/faster image (labelmaps and
  transforms computed from this data are cached here too).
- `KCL-Heart-Model/` — downloaded by the `download_kcl_heart_model` fixture.
- `DirLab-4DCT/` — a few cases from `<input root>/DirLab-4DCT`, downsampled to
  3 mm by the `dirlab_test_data` fixture.
- `Duke-Heart-4DLabelmaps/` — a few cases from
  `<input root>/Duke-Heart-4DLabelmaps`, their labelmaps downsampled to 2 mm
  nearest-neighbour by the `duke_heart_test_data` fixture.
- `Chest-CT/` — `<input root>/Chest-CT` downsampled to 3 mm by the
  `chest_ct_test_data` fixture.

Here `<input root>` is whatever `PHYSIOTWIN_INPUT_DATA_DIR` names, defaulting to
the `data/` directory of the clone — so each subset is built from the full
dataset alongside it, wherever that root has been pointed.

Every subdirectory is created on demand by `tests/conftest.py` fixtures
the first time a test needs them, and is `.gitignore`d — do not commit
their contents. The subsets derived from another dataset are only built
when that source dataset is present under the input root; otherwise the tests
that need them skip, or fail if `--require-tutorial-data` was passed.

The tutorials read these directories rather than the full datasets whenever
`PHYSIOTWIN_RUNNING_AS_TEST` is set, and write to the matching `test` subtree of
the output and weights roots — `tutorials/output/test/` and
`tutorials/network_weights/test/` by default. A test run therefore never reads
or overwrites the datasets, results, or trained checkpoints of a full run.

## Regenerating

If this directory is deleted or corrupted, simply re-run the test suite;
the fixtures in `tests/conftest.py` will re-download and rebuild everything
here automatically:

```bash
py -m pytest tests/ -v
```
