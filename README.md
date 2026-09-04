# PhysioTwin4D

[![CI](https://github.com/Project-MONAI/physiotwin4d/actions/workflows/ci.yml/badge.svg)](https://github.com/Project-MONAI/physiotwin4d/actions/workflows/ci.yml)
[![Documentation](https://github.com/Project-MONAI/physiotwin4d/actions/workflows/docs.yml/badge.svg)](https://github.com/Project-MONAI/physiotwin4d/actions/workflows/docs.yml)
[![Nightly Health](https://img.shields.io/endpoint?url=https://project-monai.github.io/physiotwin4d/status.json)](https://github.com/Project-MONAI/physiotwin4d/actions/workflows/nightly-health.yml)
[![codecov](https://codecov.io/gh/Project-MONAI/physiotwin4d/branch/main/graph/badge.svg)](https://codecov.io/gh/Project-MONAI/physiotwin4d)

[![PyPI version](https://img.shields.io/pypi/v/physiotwin4d.svg)](https://pypi.org/project/physiotwin4d/)
[![Python versions](https://img.shields.io/pypi/pyversions/physiotwin4d.svg)](https://pypi.org/project/physiotwin4d/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**A collection of methods, workflows, tutorials, and CLI tools for creating personalized physiological digital twins.**

PhysioTwin4D typically begins with a 3D medical image of a subject, extracts anatomic models from that image, and then uses AI surrogates to estimate the subject's physiological processes — initially focusing on cardiac and respiratory motion, and expanding to electrophysiology, blood flow, and organ perfusion. The package provides methods for forming these physiological AI surrogates and for finetuning the segmentation and registration AI methods that power them, with special emphasis on statistical shape models: they capture subject-specific characteristics that help determine subject-specific physiological function, and establish correspondence across subjects to aid AI surrogate generalization and simplify the application of traditional solvers.

PhysioTwin4D is not validated for clinical use. It is a research and
visualization toolkit, not a medical device, and must not be used for
diagnosis, treatment planning, or clinical decision-making.

## Documentation

**https://project-monai.github.io/physiotwin4d/** is the primary entry point
for users and contributors. Key sections:

- [Installation](https://project-monai.github.io/physiotwin4d/installation.html) and [Quickstart](https://project-monai.github.io/physiotwin4d/quickstart.html)
- [Tutorials](https://project-monai.github.io/physiotwin4d/tutorials.html) — runnable end-to-end workflows and their datasets
- [CLI & Scripts Guide](https://project-monai.github.io/physiotwin4d/cli_scripts/overview.html) — command-line tools for conversion, segmentation, registration, and USD workflows
- [API Reference](https://project-monai.github.io/physiotwin4d/api/index.html) — workflow, registration, segmentation, and USD classes
- [Developer Guides](https://project-monai.github.io/physiotwin4d/developer/architecture.html) — architecture, extension points, and implementation conventions
- [Contributing](CONTRIBUTING.md) and [Testing](https://project-monai.github.io/physiotwin4d/testing.html)
- [FAQ](https://project-monai.github.io/physiotwin4d/faq.html) and [Troubleshooting](https://project-monai.github.io/physiotwin4d/troubleshooting.html)
- [Issues](https://github.com/Project-MONAI/physiotwin4d/issues) and [Discussions](https://github.com/Project-MONAI/physiotwin4d/discussions)

## Highlights

- **Personalized digital twins**: build subject-specific anatomic models and physiological AI surrogates from 3D/4D medical images
- **Statistical shape models**: capture subject-specific anatomy and establish cross-subject correspondence, aiding AI surrogate generalization and simplifying traditional solver setup
- **Simplified workflows on industry-leading open-source tools**: ICON and Greedy for registration; MONAI with TotalSegmentator and Simpleware for segmentation; scikit-learn for statistical shape modeling; ITK for image processing; PyVista and OpenUSD/Omniverse for geometry manipulation; CuPy for accelerated computing; and PhysicsNeMo for AI surrogates
- **Extensible class hierarchy**: add new segmentation and registration methods, and extend to new data types, organs, and physiological processes, without reworking the workflow layer
- **Physiological motion**: cardiac and respiratory motion today, expanding to electrophysiology, blood flow, and organ perfusion
- **NVIDIA Omniverse as the simulation hub**: the end goal for simulation — a simulation-information hub and gateway to other engines (e.g., Ansys solvers), interactive simulations for treatment planning (e.g., Isaac Sim, Newton), visualization systems (e.g., AR/VR devices), and physical systems (e.g., robots via ROS)
- **CLI and Python API**: installed command-line tools and workflow classes for repeatable, scriptable pipelines

## Quick Start

### Install

```
uv pip install "physiotwin4d[all]"
```

See the [installation guide](https://project-monai.github.io/physiotwin4d/installation.html) for GPU setup, source installs, and optional extras (PhysicsNeMo). 

### Download Tutorials

The tutorials are not installed by pip. They live in this repository.
Clone it to run them:

```
git clone https://github.com/Project-MONAI/physiotwin4d.git
```

### Download Tutorial Data

Tutorial 1 (heart) runs on the public Slicer-Heart 4D CT sample.  We provide
automated download for multiple datasets via a CLI.  However, one key dataset
from DirLab requires manual download, see [data/DirLab-4DCT/README.md](data/DirLab-4DCT/README.md).

**IMPORTANT:** Run the download from the top level of the clone. The tutorials
resolve their inputs against the repository root, so downloading
elsewhere puts the data where they will not find it.

```
cd physiotwin4d
physiotwin4d-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT
```

### Run Tutorial 01: Gated CT to USD

```
python tutorials/tutorial_01_heart_gated_ct_to_usd.py
```

### Explore the Code

That tutorial builds the same workflow the Python API exposes:

```python
import itk
from pathlib import Path

from physiotwin4d import (
    RegisterImagesICON,
    SegmentChestTotalSegmentatorWithContrast,
    WorkflowConvertImageToUSD,
)

frame_files = sorted(Path("data/Slicer-Heart-CT").glob("slice_???.mha"))
time_series_images = [itk.imread(str(path)) for path in frame_files]

workflow = WorkflowConvertImageToUSD(
    time_series_images=time_series_images,
    reference_image=time_series_images[int(0.7 * len(time_series_images))],
    output_directory="./results",
    usd_project_name="cardiac_model",
    registration_method=RegisterImagesICON(),  # or RegisterImagesGreedy()
    segmentation_method=SegmentChestTotalSegmentatorWithContrast(),
)
results = workflow.process()
```

### Explore the CLIs

The Tutorial 01 workflow and many of the workflows in this toolkit are also
available as command-line tools but CLIs provide fewer options for
customization:

```bash
physiotwin4d-convert-image-to-usd cardiac_4d.nrrd --contrast --output-dir ./results
```

# Next Steps

See the [quickstart](https://project-monai.github.io/physiotwin4d/quickstart.html) and [tutorials](https://project-monai.github.io/physiotwin4d/tutorials.html) for full walkthroughs covering segmentation, registration, statistical shape modeling, and USD export.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (also published at [project-monai.github.io/physiotwin4d/contributing.html](https://project-monai.github.io/physiotwin4d/contributing.html)) for code style, testing, IDE setup, and pull request conventions.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

Additionally, NVIDIA Omniverse is distributed under its own custom license, which makes it
free for academic and commercial use.  https://docs.omniverse.nvidia.com/ov/latest/common/NVIDIA_Omniverse_License_Agreement.html

### Non-commercial Licenses (optional)
* NVIDIA Segment CT MRI AI weights (used in the SegmentNVSegmentCTMRI class,
are restricted from commercial use. https://github.com/NVIDIA-Medtech/NV-Segment-CTMR
* TotalSegmentator includes the optional use of some of their research-only models. Using those models assumes that you have
the appropriate license key install, otherwise an error occurs.   Those models can be disabled by calling ```set_has_academic_license(False)``` member function of the ```SegmentChestTotalSegmentator``` class.
