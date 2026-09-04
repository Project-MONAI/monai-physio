============
Installation
============

This guide covers the installation of MONAI Physio and its dependencies.

Prerequisites
=============

System Requirements
-------------------

* **Python**: 3.11, 3.12, or 3.13
* **GPU**: NVIDIA GPU with CUDA 13 — required for full capability and best performance; a CPU-only PyPI installation is a supported fallback, but it is slow, emits a runtime warning, and cannot run the AI-surrogate workflows
* **RAM**: 16GB minimum (32GB+ recommended for large datasets)
* **Storage**: 10GB+ for package and model weights
* **Visualization**: NVIDIA Omniverse (optional, for USD visualization)

Software Dependencies
---------------------

MONAI Physio relies on several key packages:

* **Medical Imaging**: ITK, MONAI, nibabel, PyVista
* **AI/ML**: PyTorch, CuPy (CUDA 13), transformers, MONAI
* **Registration**: icon-registration, unigradicon
* **Visualization**: USD-core, PyVista
* **Segmentation**: TotalSegmentator
* **AI surrogates**: PhysicsNeMo (``nvidia-physicsnemo``), torch-geometric,
  torch-scatter - optional, installed with the ``[physicsnemo]`` extra

Installation Methods
====================

Method 1: Install from PyPI (Recommended)
------------------------------------------

Install the ``[all]`` extra. It enables every feature and gives the best
performance:

.. code-block:: bash

   uv pip install "monai-physio[all]"

The ``[all]`` extra installs PhysicsNeMo, CuPy, and dependencies for 
development, testing, and documenting. In uv-managed source environments,
PyTorch, torchvision, and torchaudio should resolve from the CUDA 13.0 PyTorch wheel
index. However, that automation is not garaunteed for every platform, and it is recommended
to pre-install torch with CUDA acceleration, e.g., as described at 
https://pytorch.org/get-started/locally/.

CPU-only fallback (evaluation, or no NVIDIA GPU available):

.. code-block:: bash

   pip install monai-physio

This works immediately but is a limited configuration: GPU acceleration is
unavailable, segmentation and registration run slowly enough that the larger
tutorials become impractical, and the AI-surrogate workflows behind the
``[physicsnemo]`` extra need CUDA and cannot run at all. CuPy is absent, so a
``UserWarning`` is emitted at import time (visible by default in all standard
Python runs):

.. code-block:: text

   CuPy is not installed — GPU acceleration is unavailable and processing will be
   slow. Re-install with uv to get CuPy and CUDA-enabled PyTorch in one step
   (pip alone will not select the correct CUDA wheel):
     uv pip install 'monai-physio[cuda13]'  # CUDA 13

Method 2: Install from Source
------------------------------

For development or to get the latest features:

**Step 1: Clone the repository**

.. code-block:: bash

   git clone https://github.com/Project-MONAI/monai-physio.git
   cd monai_physio

**Step 2: Create virtual environment**

.. tabs::

   .. tab:: Linux/macOS

      .. code-block:: bash

         python -m venv venv
         source venv/bin/activate

   .. tab:: Windows

      .. code-block:: bash

         python -m venv venv
         venv\Scripts\activate

**Step 3: Install uv package manager** (optional but recommended)

.. code-block:: bash

   pip install uv

**Step 4: Install MONAI Physio**

Install the ``[cuda13]`` extra for the full-capability source install:

.. code-block:: bash

   uv pip install -e ".[cuda13]"

Without the extra:

.. code-block:: bash

   uv pip install -e "."

still uses the CUDA 13.0 PyTorch wheel index by default, but leaves out CuPy
and the GPU acceleration that depends on it.

Optional Dependencies
=====================

Everything at Once
------------------

The ``[all]`` extra pulls in every optional component — ``[cuda13]``,
``[physicsnemo]``, ``[dev]``, ``[docs]`` and ``[test]`` — so every feature is
enabled and every use of the platform is supported, from the AI-surrogate
workflows to building the docs and running the full test suite:

.. code-block:: bash

   uv pip install "monai-physio[all]"

It inherits the ``[physicsnemo]`` caveats: PyTorch and setuptools must already
be installed, because ``torch-scatter`` compiles against torch when no matching
wheel exists, and ``nvidia-physicsnemo`` requires Python >= 3.11. uv handles the
build isolation automatically; with pip, install in two steps:

.. code-block:: bash

   pip install "monai-physio[cuda13]" setuptools
   pip install "monai-physio[all]" --no-build-isolation

Development Tools
-----------------

To install development dependencies (testing, linting, formatting):

.. code-block:: bash

   pip install monai-physio[dev]

This includes:

* **ruff** (fast linting and formatting)
* **mypy** (type checking)
* **pytest, pytest-cov** (testing)
* **pre-commit** (git hooks for automatic checks)

.. note::
   As of 2026, MONAI Physio uses Ruff as the primary linter and formatter,
   replacing the previous black, isort, flake8, and pylint tools for improved
   speed and simplicity.

Documentation Tools
-------------------

To build documentation locally:

.. code-block:: bash

   pip install monai-physio[docs]

Testing Dependencies
--------------------

To run tests:

.. code-block:: bash

   pip install monai-physio[test]

Verify Installation
===================

After installation, verify that MONAI Physio is correctly installed:

.. code-block:: python

   import monai_physio
   from monai_physio import WorkflowConvertImageToUSD
   
   print(f"MONAI Physio version: {monai_physio.__version__}")
   print(WorkflowConvertImageToUSD.__name__)

Expected output:

.. code-block:: text

   MONAI Physio version: {{ mphysio_project_version }}
   WorkflowConvertImageToUSD

Command-Line Tools
==================

MONAI Physio installs eleven command-line tools, each prefixed
``monai-physio-``. There is no bare ``monai_physio`` command; check the install
with any one of them:

.. code-block:: bash

   # Check CLI is available
   monai-physio-download-data --help
   monai-physio-convert-image-to-usd --help

See :doc:`cli_scripts/overview` for the full list.

GPU Setup
=========

CUDA Installation
-----------------

An NVIDIA GPU is strongly recommended. CUDA 13 is supported via the optional
extra:

* **CUDA 13** — installed when you use the ``[cuda13]`` extra (recommended)

A plain ``pip install monai-physio`` installs a CPU-only build. It runs
without error but emits a ``UserWarning`` at import time and will be
significantly slower than a GPU-enabled install.

Optional External Software
--------------------------

One segmentation backend is not a Python dependency and cannot be installed
with pip:

* **Synopsys Simpleware Medical** — required by
  :class:`~monai_physio.SegmentHeartSimpleware` and
  :class:`~monai_physio.SegmentHeartSimplewareTrimmedBranches`, and therefore by
  Tutorial 13, which uses Simpleware to segment the heart. It needs a local
  licensed installation; see :doc:`api/segmentation/simpleware`. Everything
  else in the toolkit runs without it, and the ``requires_simpleware`` tests
  skip cleanly when it is absent.

If CUDA is not yet installed, download the CUDA Toolkit from
`NVIDIA's website <https://developer.nvidia.com/cuda-downloads>`_, then verify:

.. code-block:: bash

   nvcc --version
   nvidia-smi

PyTorch with CUDA
-----------------

uv-managed source environments source PyTorch, torchvision, and torchaudio from
the ``https://download.pytorch.org/whl/cu130`` index by default. To verify the
active version:

.. code-block:: python

   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")

Troubleshooting
===============

Common Issues
-------------

**Issue: CUDA out of memory**

Solution: Reduce batch sizes or process smaller images. Most MONAI Physio functions work with limited GPU memory.

**Issue: Import errors for ITK or VTK**

Solution: These packages sometimes require system dependencies. On Ubuntu:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx libglib2.0-0

**Issue: TotalSegmentator download fails**

Solution: TotalSegmentator downloads models on first use. Ensure you have:

* Stable internet connection
* Sufficient disk space (~2GB for models)
* Write permissions in the cache directory

**Issue: USD files not rendering in Omniverse**

Solution:

1. Ensure NVIDIA Omniverse is installed
2. Set the viewport renderer to RTX and switch to the scene's
   ``/World/Camera``; see :doc:`viewing_usd`
3. Verify file paths are accessible to Omniverse

Getting Help
------------

If you encounter issues:

1. Check the :doc:`troubleshooting` guide
2. Search `GitHub Issues <https://github.com/Project-MONAI/monai-physio/issues>`_
3. Open a new issue with:

   * Python version
   * CUDA version
   * Error messages
   * Minimal code to reproduce

Next Steps
==========

* Continue to :doc:`quickstart` for your first MONAI Physio workflow
* Explore :doc:`tutorials` for runnable, workflow-by-workflow examples
* Read :doc:`cli_scripts/overview` for detailed command-line workflows

