# MONAI Physio Tests

This directory contains comprehensive test suites for the MONAI Physio package, validating the complete medical imaging to Omniverse pipeline.

## Documentation

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive testing guide with setup, troubleshooting, and best practices
- **[GITHUB_WORKFLOWS.md](GITHUB_WORKFLOWS.md)** - CI/CD documentation and GitHub Actions workflow details
- **[PARALLEL_EXECUTION_GUIDE.md](PARALLEL_EXECUTION_GUIDE.md)** - How parallel execution works
- **[TEST_FIXES_SUMMARY.md](TEST_FIXES_SUMMARY.md)** - Recent bug fixes and known issues

## Test Categories

### Data Pipeline Tests
- **`test_download_heart_data.py`** - Automatic data download with fallback logic
- **`test_convert_image_4d_to_3d.py`** - 4D image to 3D time series conversion (ITK)

### Segmentation Tests (GPU Required)
- **`test_segment_chest_total_segmentator.py`** - TotalSegmentator chest CT segmentation

### Registration Tests (Slow ~5-10 min)
- **`test_register_images_ants.py`** - ANTs deformable registration
- **`test_register_images_icon.py`** - ICON deep learning registration

### Geometry & Visualization Tests
- **`test_contour_tools.py`** - PyVista mesh extraction and manipulation
- **`test_transform_tools.py`** - ITK transform operations and visualization
- **`test_convert_vtk_to_usd.py`** - VTK to USD conversion

### USD Utility Tests
- **`test_usd_merge.py`** - USD file merging with material preservation
- **`test_usd_time_preservation.py`** - Time-varying data validation

### Workflow Tests
- **`test_workflow_evaluate_movement.py`** - `WorkflowEvaluateMovement` metrics, CSV, and Markdown report
- **`test_workflow_train_physicsnemo.py`** - `WorkflowTrainPhysicsNeMo` checkpoint plus the assets inference reads beside it (needs `[physicsnemo]`)

### Tutorial Tests (SLOW - Opt-in)
- **`test_tutorials.py`** - End-to-end execution of each `tutorials/*.py` script,
  comparing the screenshots it writes against stored baselines
  - **Opt-in only** - Requires `--run-tutorials` flag to run
  - Requires GPU, CUDA, all dependencies, and the tutorial datasets
  - Scripts in `experiments/` are exploratory and are **not** run as tests



## Directory Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── data/                    # Downloaded/input test data
│   └── Slicer-Heart-CT/    # 4D cardiac CT dataset
├── results/                 # Test outputs (organized by module)
├── baselines/               # Regression test baselines (.hdf, .mha via Git LFS)
└── *.py                     # Test modules
```

## Quick Start

### Git LFS (required for tests)

Baseline files in `tests/baselines/` (`.hdf` and `.mha`) are stored with **Git LFS**. You must install and use Git LFS so these files are pulled correctly; otherwise tests that compare against baselines will fail.

```bash
# Install Git LFS (one-time per machine)
# macOS (Homebrew): brew install git-lfs
# Ubuntu/Debian:    sudo apt-get install git-lfs
# Windows:          winget install GitHub.GitLFS  (or download from https://git-lfs.github.com)

git lfs install          # one-time per user
git lfs pull            # fetch LFS files if you already cloned
```

If you clone after running `git lfs install`, LFS files are pulled automatically with `git clone`.

### Install Dependencies
```bash
uv pip install -e ".[test]"
```

### Run Tests

The fast path is the default. Heavy buckets (slow tests, GPU tests, Simpleware
tests, tutorial scripts) are **auto-skipped** unless you
pass their `--run-<bucket>` flag. Tests that need downloadable data fetch it
through the session fixtures and run by default — there is no `requires_data`
marker any more.

```bash
# Fast tests only (recommended for development)
pytest tests/ -v

# Specific test module
pytest tests/test_usd_merge.py -v
```

### Opt-in Buckets (`--run-*` flags)

Each flag enables one marker family. Flags compose, so you can stack them.

| Flag | Enables tests marked | Typical use |
|---|---|---|
| `--run-slow` | `slow` | Slow registration / segmentation tests (>30 s) |
| `--run-gpu` | `requires_gpu` | CUDA-dependent tests (ICON, Simpleware, etc.) |
| `--run-simpleware` | `requires_simpleware` | Need a licensed Synopsys Simpleware Medical install (also marked `requires_gpu`) |
| `--run-physicsnemo` | `requires_physicsnemo` | Need the optional `[physicsnemo]` extra installed |
| `--run-tutorials` | `tutorial` | Tutorial scripts run end-to-end (hours to run) |
| `--run-all` | every bucket above | Equivalent to passing all `--run-*` flags at once |

```bash
# Slow tests
pytest tests/ -v --run-slow

# Typical local GPU profile. The self-hosted CI GPU runner uses --run-all.
pytest tests/ -v --run-gpu --run-slow

# Enable every bucket at once
pytest tests/ -v --run-all

# Full Simpleware coverage (requires Simpleware Medical installed locally)
pytest tests/ -v --run-simpleware --run-gpu --run-slow

# Tutorial tests (SLOW — hours to complete)
pytest tests/test_tutorials.py -v --run-tutorials

# A single tutorial by name
pytest tests/test_tutorials.py::TestTutorial01HeartGatedCTToUSD -v -s --run-tutorials
```

### Common Test Commands
```bash
# Run with coverage
pytest tests/ --cov=src/monai_physio --cov-report=html

# Run a specific test class or method
pytest tests/test_usd_merge.py::TestUSDMerge::test_merge_usd_files_copy_method -v

# Create missing regression baselines instead of failing
pytest tests/ --create-baselines
```

**For detailed instructions**, see [TESTING_GUIDE.md](TESTING_GUIDE.md)

## Test Timing Reports

All test runs automatically generate a comprehensive timing report at the end
showing individual test durations, session time, and pass/fail/skip counts. The
report separates regular tests from tutorial tests and highlights the slowest
tests.

## Test Configuration

### Global Settings
- **Timeout**: 900 seconds (15 minutes) per test
- **Data Management**: Automatic download with intelligent fallback
- **Output Organization**: Results saved to `tests/results/` by module

### Test Markers (all opt-in via a matching `--run-*` flag)
- `@pytest.mark.slow` — Tests taking >30 seconds. Opt in: `--run-slow`.
- `@pytest.mark.requires_gpu` — Tests needing CUDA. Opt in: `--run-gpu`.
- `@pytest.mark.requires_simpleware` — Tests needing a licensed Synopsys
  Simpleware Medical install. Opt in: `--run-simpleware`. (Combine with
  `--run-gpu` and `--run-slow`.)
- `@pytest.mark.requires_physicsnemo` — Tests needing the optional
  `[physicsnemo]` extra (`pip install "monai-physio[physicsnemo]"`, requires
  Python >= 3.11). Opt in: `--run-physicsnemo`.
- `@pytest.mark.tutorial` — Tutorial scripts run end-to-end (SLOW, never in
  CI). Opt in: `--run-tutorials`.

`--run-all` is a convenience flag that turns on every `--run-*` bucket at once.
- `@pytest.mark.integration` — Integration tests vs unit tests (filter-only).
- `@pytest.mark.timeout(seconds)` — Per-test timeout override.

The previous `requires_data` marker has been retired: tests that need
downloadable data request the session fixtures (`test_directories`,
`download_test_data`, `test_images`) and the data is fetched on first use.

### Test Dependencies

Tests are organized hierarchically - some tests depend on outputs from earlier tests:

```
test_download_heart_data
    ↓
test_convert_image_4d_to_3d
    ↓                    ↓
    ↓                    ├─→ test_register_images_ANTS ──→ test_transform_tools
    ↓                    ├─→ test_register_images_ICON
    ↓                    ↓
test_segment_chest_total_segmentator ────→ test_contour_tools
                                                ↓
                                           test_convert_vtk_to_usd
```

Fixtures in `conftest.py` automatically manage these dependencies.

## Continuous Integration

Tests automatically run on pull requests via GitHub Actions. The CI workflow:

- **Runs fast tests** - USD utilities, data conversion, basic validation
- **Skips slow tests** - Registration and segmentation (too slow for CI)
- **Automatically skips tutorial tests** - Protected by `--run-tutorials` flag requirement
- **Caches test data** - Speeds up subsequent runs
- **Generates coverage** - Reports uploaded to Codecov

**CI Configuration:**
- Platforms: Ubuntu, Windows, macOS
- Python versions: 3.11, 3.12
- Target coverage: >70%
- Protection: Tutorial tests require `--run-tutorials` flag (never used in CI/CD)

**For detailed CI/CD information**, see [GITHUB_WORKFLOWS.md](GITHUB_WORKFLOWS.md)

## Troubleshooting

### Common Issues

**Problem: HTTP 404 when downloading data**
- Tests now check `data/Slicer-Heart-CT/` first
- Place `TruncalValve_4DCT.seq.nrrd` there to avoid download

**Problem: Out of memory errors**
- Some segmentation models require significant RAM
- Skip segmentation tests with: `pytest tests/ --ignore=tests/test_segment_chest_total_segmentator.py`

**Problem: Test timeout**
- Global timeout: 900 seconds (15 minutes)
- Registration tests need GPU for reasonable speed
- Override with: `@pytest.mark.timeout(1800)` decorator or use `-o timeout=1800`

**Problem: Fixture naming errors**
- Use correct fixture names from `conftest.py`
- Don't import from other test files

**For complete troubleshooting guide**, see [TESTING_GUIDE.md](TESTING_GUIDE.md#troubleshooting)

## Adding New Tests

When creating new tests:

1. **Name properly**: Use `test_` prefix for files and methods
2. **Add markers**: `@pytest.mark.slow`, `@pytest.mark.requires_gpu`,
   `@pytest.mark.requires_simpleware`, etc. — each one places the test in
   its corresponding `--run-*` opt-in bucket.
3. **Use fixtures**: Define shared fixtures in `conftest.py`
4. **Document well**: Clear docstrings explaining what's validated
5. **Organize outputs**: Save results to `tests/results/<module_name>/`
6. **Update docs**: Add test description to this README

**For best practices**, see [TESTING_GUIDE.md](TESTING_GUIDE.md#best-practices)

## Test Data

### Slicer-Heart-CT Dataset
- **Source**: [SlicerHeart GitHub](https://github.com/SlicerHeart/SlicerHeart)
- **File**: `TruncalValve_4DCT.seq.nrrd` (~1.2 GB)
- **Content**: 21-phase pediatric cardiac CT
- **Usage**: Tests use first 2 time points for speed

### Data Management
Tests automatically:
1. Check `tests/data/Slicer-Heart-CT/`
2. Check `data/Slicer-Heart-CT/` (copies if found)
3. Download from GitHub if needed
4. Skip gracefully if unavailable

## Additional Resources

- **Detailed Testing Guide**: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Parallel Execution Guide**: [PARALLEL_EXECUTION_GUIDE.md](PARALLEL_EXECUTION_GUIDE.md)
- **CI/CD Documentation**: [GITHUB_WORKFLOWS.md](GITHUB_WORKFLOWS.md)
- **Recent Fixes**: [TEST_FIXES_SUMMARY.md](TEST_FIXES_SUMMARY.md)
- **Main Project**: [../README.md](../README.md)
- **Example Workflows**: [../experiments/Heart-GatedCT_To_USD/](../experiments/Heart-GatedCT_To_USD/)

---

**Need help?** Check the [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive documentation and FAQ.
