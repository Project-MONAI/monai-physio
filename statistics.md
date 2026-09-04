# MONAI Physio - Software Development Statistics

**Report Generated:** August 14, 2026
**Project Version:** 2026.08.0
**Status:** Beta (Development Status: 4 - Beta)

Line counts below are total lines per file (`wc -l`), including blanks and
comments, over Git-tracked files only (`git ls-files`).

---

## Executive Summary

MONAI Physio is a collection of methods, workflows, tutorials, and CLI tools
for creating personalized physiological digital twins from 3D/4D medical images.
This report summarizes development effort, code quality, and project maturity.

### Key Metrics at a Glance

| Metric                         | Value                                          |
| ------------------------------ | ---------------------------------------------- |
| **Total Lines of Code**        | ~74,800                                        |
| **Development Period**         | December 5, 2025 - August 14, 2026 (~8 months) |
| **Total Commits**              | 123                                            |
| **Primary Developer**          | 1 (Stephen Aylward), plus 1 outside contributor |

---

## Detailed Code Statistics

### Lines of Code Breakdown

| Category                                | Files          | Lines of Code | Percentage |
| ---------------------------------------- | -------------- | -------------- | ---------- |
| **Core Python Source (`src/`)**          | 74 files       | 29,345         | 39.2%      |
| **Test Suite (`tests/`)**                | 40 files       | 12,323         | 16.5%      |
| **Experiment Scripts (`experiments/`)**  | 46 files       | 9,219          | 12.3%      |
| **Tutorial Scripts (`tutorials/`)**      | 32 files       | 9,207          | 12.3%      |
| **Utility Scripts (`utils/`)**           | 3 files        | 1,739          | 2.3%       |
| **Documentation (`docs/*.rst`)**         | 85 files       | 8,892          | 11.9%      |
| **Markdown (repo-wide READMEs, guides)** | 35 files       | 4,051          | 5.4%       |
| **TOTAL**                                | **315 files**  | **~74,800**    | **100%**   |

The 32 files under `tutorials/` are 29 numbered tutorial scripts plus 3
per-organ parameter modules (`parameters_heart_ct_kcl.py`,
`parameters_lung_ct_dirlab.py`, `parameters_duke_heart_labelmaps.py`) that
carry the constants the tutorials share.

All experiment and tutorial sources are plain `.py` files run with
`python <script>.py`. Experiment scripts additionally carry `# %%` percent-cell
markers, so they can be stepped through cell-by-cell in VS Code / Cursor;
tutorials are straightforward top-to-bottom scripts.

### Core Module Highlights (Python Source)

| Module                                          | Lines | Purpose                                         |
| ------------------------------------------------ | ----- | ----------------------------------------------- |
| `usd_tools.py`                                   | 1,523 | USD file manipulation and inspection            |
| `contour_tools.py`                               | 1,415 | Mesh extraction and contour manipulation        |
| `register_models_pca.py`                         | 1,117 | PCA-based shape model registration              |
| `convert_vtk_to_usd.py`                          | 1,071 | High-level VTK -> USD converter                 |
| `transform_tools.py`                             | 1,065 | ITK transform utilities                         |
| `usd_anatomy_tools.py`                           | 1,053 | OmniSurface materials for labeled anatomy       |
| `workflow_fit_statistical_model_to_patient.py`   | 1,049 | Model-to-patient registration workflow          |
| `segment_nv_segment_ct_mri.py`                   | 695   | NVIDIA CT/MRI segmentation bundle bridge        |
| `register_images_ants.py`                        | 691   | ANTs-based image registration                   |
| `image_tools.py`                                 | 685   | Image I/O, resampling, preprocessing            |
| `register_images_base.py`                        | 685   | Shared registration base class                  |
| `workflow_infer_movement.py`                     | 625   | Predicted displacements back into geometry      |
| `register_images_greedy.py`                      | 593   | Greedy classical deformable registration        |
| `workflow_evaluate_movement.py`                  | 565   | Per-structure scoring against acquired frames   |
| `vtk_to_usd/` subpackage                         | 2,717 | Low-level VTK -> USD building blocks (9 files)  |
| `cli/` subpackage                                | 2,454 | CLI entry-point scripts (11 commands, 13 files) |

---

## Project Maturity Indicators

| Indicator                  | Status                                              |
| --------------------------- | ---------------------------------------------------- |
| **Documentation Coverage**  | Sphinx site + per-package READMEs                    |
| **Test Suite Present**      | Yes (`tests/` with baselines via Git LFS)             |
| **CI/CD Pipeline**          | GitHub Actions (Ubuntu + Windows; Python 3.11/3.12), plus a self-hosted Windows GPU runner |
| **Dependency Management**   | `pyproject.toml`, `uv`-friendly                       |
| **Code Quality Tools**      | Ruff (lint + format), mypy                            |
| **Example Scripts**         | 46 experiment scripts + 29 tutorial scripts           |
| **Version Management**      | Calendar versioning via bumpver                       |
| **API Reference**           | Google-style docstrings + Sphinx API docs under `docs/api/` |
| **Package Distribution**    | PyPI-ready                                            |

---

## Technical Complexity Assessment

### Domain Complexity

MONAI Physio operates across several technically demanding domains:

| Domain                   | Complexity Level | Key Technologies                       |
| ------------------------- | ----------------- | ---------------------------------------- |
| **Medical Imaging**      | Very High         | ITK, MONAI, nibabel, pydicom, pynrrd     |
| **Deep Learning**        | High               | PyTorch, CUDA 13, transformers            |
| **3D Graphics / USD**    | High               | VTK, PyVista, OpenUSD, trimesh            |
| **Image Registration**   | Very High          | ANTs, Greedy, Icon, UniGradICON           |
| **AI Segmentation**      | High               | TotalSegmentator, Simpleware bridge       |
| **Geometric Processing** | High               | ICP, PCA, distance maps, statistical shape models |
| **AI Surrogates**        | Very High          | PhysicsNeMo, MeshGraphNet, torch-geometric |

### Architectural Sophistication

- Class hierarchy depth: 3-4 levels (well-structured inheritance from
  `MONAIPhysioBase`)
- Module coupling: medium (clear separation between segmentation,
  registration, USD conversion, and workflow layers)
- Public API surface documented via Sphinx API docs under `docs/api/`
- 24 required external dependencies (medical imaging, AI/ML, USD, registration),
  plus six optional extras: `cuda13`, `physicsnemo`, `dev`, `docs`, `test`, and
  `all`

---

## Dependencies & Infrastructure

### Core Dependencies (selected)

| Category              | Key Packages                                        |
| ---------------------- | ----------------------------------------------------- |
| **Medical Imaging**    | ITK, MONAI, nibabel, pydicom, pynrrd                 |
| **Deep Learning**      | PyTorch, CuPy (CUDA 13), transformers                |
| **AI Surrogates**      | PhysicsNeMo, torch-geometric, torch-scatter (optional `[physicsnemo]` extra) |
| **Registration**       | ANTs (antspyx), picsl-greedy, icon-registration, UniGradICON |
| **3D Graphics / USD**  | VTK, PyVista, USD-core, trimesh                       |
| **AI Segmentation**    | TotalSegmentator                                      |
| **Development Tools**  | pytest, pytest-cov, pytest-xdist, ruff, mypy, sphinx, uv |

### Infrastructure Files

| File             | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `pyproject.toml` | Modern Python packaging, dependencies, tool configs |
| `README.md`      | Repository highlights and quick start               |
| `LICENSE`        | Apache 2.0 license                                  |
| `CLAUDE.md`      | Per-repo guidance for Claude Code                   |
| `AGENTS.md`      | Per-repo guidance for AI coding agents              |

---

## Quality Metrics

### Code Quality Configuration

- **Ruff** - Formatting and linting (line length: 88)
- **mypy** - Strict type checking (`disallow_untyped_defs = true`)
- **pre-commit** - Hooks for ruff + mypy + fast tests on push

### Testing Framework

- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **pytest-xdist** - Parallel test execution
- **pytest-timeout** - Per-test timeout (15 min default)

**Test Categories** (opt-in buckets via marker flags):
- Unit and integration tests (fast, run by default)
- `slow` - slower tests (opt-in via `--run-slow`)
- `requires_gpu` - GPU/CUDA-dependent tests (opt-in via `--run-gpu`)
- `requires_simpleware` - tests needing a local Synopsys Simpleware Medical install (opt-in via `--run-simpleware`)
- `requires_physicsnemo` - tests needing the optional `[physicsnemo]` extra (opt-in via `--run-physicsnemo`)
- `tutorial` - runs tutorial scripts end-to-end
  (opt-in via `--run-tutorials`; multi-hour)

---

## Documentation Statistics

| Type                  | Count                   | Lines |
| ---------------------- | ------------------------ | ----- |
| **Markdown files**    | 35 (repo-wide READMEs, guides) | 4,051 |
| **reStructuredText**  | 85 files under `docs/`   | 8,892 |
| **Python docstrings** | All public modules       | embedded |
| **Knowledge graph**   | `graphify-out/`, refreshed via `graphify update .` | n/a (not checked in) |

### Documentation Highlights

- Sphinx site (published to GitHub Pages) covering getting started,
  tutorials, CLI & scripts, API reference, developer guides, contributing,
  testing, FAQ, and troubleshooting
- Per-subpackage READMEs and `CLAUDE.md` files (e.g.
  `src/monai_physio/vtk_to_usd/CLAUDE.md`)
- Shared `.agents/` configuration: 4 role-specific subagents
  (`.agents/agents/`) and 8 slash-command skills (`.agents/skills/`) for
  Claude Code and other AI coding agents

---

## Summary

MONAI Physio is a beta-quality scientific toolkit for creating personalized
physiological digital twins: it extracts anatomic models from 3D/4D medical
images and uses AI surrogates - together with statistical shape models for
subject-specific characterization and cross-subject correspondence - to
estimate a subject's physiological processes, currently cardiac and
respiratory motion. It is built on top of established medical imaging, AI/ML,
and 3D graphics libraries with a small, focused public API and a
plain-Python-script example/tutorial layout that runs both interactively and
unattended.

---

**Last Updated:** August 14, 2026
