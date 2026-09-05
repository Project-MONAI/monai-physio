============
Installation
============

This guide covers the installation of MONAI Physio and its dependencies.

Prerequisites
=============

System Requirements
-------------------

* **Python**: 3.11, 3.12, or 3.13
* **GPU**: NVIDIA GPU with CUDA 12.6 or CUDA 13 - needed for full
  performance; CPU-only runs every workflow, significantly slower
* **RAM**: 16GB minimum (32GB+ recommended for large datasets)
* **Storage**: 10GB+ for package and model weights
* **Visualization**: NVIDIA Omniverse (optional, for USD visualization)

Software Dependencies
---------------------

Installed by default:

* **Medical Imaging**: ITK, MONAI, nibabel, PyVista
* **AI/ML**: PyTorch, transformers, MONAI
* **AI surrogates**: PhysicsNeMo (``nvidia-physicsnemo``), torch-geometric,
  torch-scatter
* **Registration**: icon-registration, unigradicon
* **Visualization**: USD-core, PyVista
* **Segmentation**: TotalSegmentator

CuPy is optional, installed with the ``[cuda12]`` or ``[cuda13]`` extra.

Installing
==========

Use ``uv``. It selects the CUDA wheel and applies this project's
``torch-scatter`` build configuration from ``pyproject.toml``.

**CUDA 12.6 - recommended, installs entirely from prebuilt wheels:**

.. code-block:: bash

   uv pip install "monai-physio[cuda12]"     # runtime
   uv pip install -e ".[dev_cuda12]"         # plus dev/test/docs tooling

**CUDA 13 - same, plus a** ``torch-scatter`` **source build:**

.. code-block:: bash

   uv pip install "monai-physio[cuda13]"
   uv pip install -e ".[dev_cuda13]"

That build needs a CUDA toolkit with ``nvcc`` on ``PATH`` matching the
installed torch, and a C++ toolchain (MSVC Build Tools on Windows, gcc/g++ on
Linux). See :ref:`torch-scatter-wheels`.

**Auto-detected PyTorch, without CuPy:**

.. code-block:: bash

   uv pip install --torch-backend=auto monai-physio
   uv pip install --torch-backend=auto -e ".[dev]"

``--torch-backend=auto`` picks the PyTorch build matching your driver, so no
CUDA version is named. CuPy is not covered - it is not a PyTorch package -
and the selected torch may need a ``torch-scatter`` source build. Export
``UV_TORCH_BACKEND=auto`` to apply the flag to every command in the shell
session.

``--torch-backend`` needs uv 0.6.9 or newer (``uv self update`` on an older
install). On an older uv, use the ``cuda12``/``cuda13`` extras instead, which
route to the matching PyTorch index without the flag.

**CPU-only:**

.. code-block:: bash

   uv pip install monai-physio

Every workflow runs, including the AI-surrogate ones, but significantly
slower: segmentation, registration, and AI-surrogate training and inference
all lose GPU acceleration. CuPy is absent, so importing ``monai_physio``
emits a ``UserWarning`` naming the extras that provide it.

.. _torch-scatter-wheels:

``torch-scatter`` wheel availability
-------------------------------------

Every dependency installs from a prebuilt wheel except ``torch-scatter``,
the compiled CUDA extension behind PhysicsNeMo's MeshGraphNet. Its wheels
(https://data.pyg.org/whl/) cover only specific
``(torch, CUDA, Python, platform)`` combinations:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Install
     - Linux
     - Windows
   * - ``[cuda12]`` (CUDA 12.6, torch < 2.13)
     - Prebuilt wheel
     - Prebuilt wheel
   * - ``[cuda13]`` (CUDA 13)
     - Source build
     - Source build
   * - ``--torch-backend=auto``
     - Depends on the selected torch
     - Depends on the selected torch

No wheel covers torch 2.13 or newer on any platform, and the CUDA 13.0 wheels
are Linux-only. ``[cuda12]`` pins torch below 2.13 to stay inside the
prebuilt-wheel range, which is why it is the recommended install.

Installing with pip
--------------------

``pip`` does not read the PyTorch index from ``pyproject.toml``, so install
PyTorch first, then the package:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install "monai-physio[cuda12]"

See https://pytorch.org/get-started/locally/ for the PyTorch selector.
Importing ``monai_physio`` from a pip install emits a ``UserWarning``
pointing at the ``uv`` command, which needs neither step.

For CUDA 13, ``torch-scatter`` is built from source, which additionally needs
``setuptools`` present and build isolation disabled. ``pip`` disables it for
the whole install, where ``uv`` scopes it to the one package:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   pip install setuptools
   pip install "monai-physio[cuda13]" --no-build-isolation

Installing from Source
-----------------------

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

A source install implies running tests and building docs, so use a ``dev``
extra:

.. code-block:: bash

   uv pip install -e ".[dev_cuda12]"                   # CUDA 12.6
   uv pip install -e ".[dev_cuda13]"                   # CUDA 13
   uv pip install --torch-backend=auto -e ".[dev]"     # no CuPy

Development Tools
==================

The ``dev`` extra - included in ``dev_cuda12`` and ``dev_cuda13`` - provides:

* **ruff** (linting and formatting)
* **mypy** (type checking)
* **pytest, pytest-cov, pytest-xdist** (testing)
* **pre-commit** (git hooks for automatic checks)
* **sphinx** and extensions (documentation)

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

Download the CUDA Toolkit from
`NVIDIA's website <https://developer.nvidia.com/cuda-downloads>`_, then
verify:

.. code-block:: bash

   nvcc --version
   nvidia-smi

Optional External Software
--------------------------

One segmentation backend is not a Python dependency and cannot be installed
with pip:

* **Synopsys Simpleware Medical** - required by
  :class:`~monai_physio.SegmentHeartSimpleware` and
  :class:`~monai_physio.SegmentHeartSimplewareTrimmedBranches`, and therefore by
  Tutorial 13, which uses Simpleware to segment the heart. It needs a local
  licensed installation; see :doc:`api/segmentation/simpleware`. Everything
  else in the toolkit runs without it, and the ``requires_simpleware`` tests
  skip cleanly when it is absent.

PyTorch with CUDA
-----------------

The ``[cuda12]`` and ``[cuda13]`` extras pin PyTorch, torchvision, and
torchaudio to the ``cu126`` and ``cu130`` wheel indexes respectively. To
verify the active build:

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
