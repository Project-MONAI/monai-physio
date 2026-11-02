============
Installation
============

This guide covers the installation of MONAI Physio and its dependencies.

Prerequisites
=============

System Requirements
-------------------

* **Python**: 3.11, 3.12, or 3.13
* **GPU**: NVIDIA GPU of compute capability 7.5 (Turing) or newer, with a
  driver for CUDA 12.8 or newer, required.
  Surface extraction, segmentation, and the AI surrogates run on the GPU;
  there is no CPU-only mode.
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

CuPy is installed with the ``[cuda12]`` or ``[cuda13]`` extra.

Installing
==========

Create and activate a virtual environment first (``python -m venv .venv``,
then ``source .venv/bin/activate`` on Linux or ``.venv\Scriptsctivate`` on
Windows), and install ``uv`` into it with ``pip install uv``.

**CUDA 12.8 - recommended, installs entirely from prebuilt wheels:**

.. code-block:: bash

   uv pip install "monai-physio[cuda12]" --torch-backend=cu128        -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

Both options are required:

* ``--torch-backend=cu128`` installs PyTorch's CUDA 12.8 build. Without it uv
  takes torch from PyPI, which is a CPU-only build on Windows and a CUDA 13
  build on Linux, neither of which matches the prebuilt ``torch-scatter``.
* ``-f`` points uv at the PyG wheel index, where the prebuilt
  ``torch-scatter`` lives. PyPI hosts only its source distribution, whose build
  fails under ``uv pip`` (``No module named 'torch'``).

CUDA 12.8 is the newest CUDA 12 build with kernels for Blackwell GPUs, and the
``[cuda12]`` extra pins torch below 2.12 because the CUDA 12.8 PyTorch index
stops at 2.11.

**CUDA 13 - builds** ``torch-scatter`` **from source:**

PyG publishes CUDA 13 ``torch-scatter`` wheels for Linux only
(``https://data.pyg.org/whl/torch-2.12.0+cu130.html``). On Windows, and for
any other torch version, ``torch-scatter`` is compiled during the install,
which needs a CUDA toolkit with ``nvcc`` on ``PATH`` matching the installed
torch and a C++ toolchain (MSVC Build Tools on Windows, gcc/g++ on Linux).
Use pip, with build isolation disabled so the build sees the installed torch:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   pip install setuptools
   pip install "monai-physio[cuda13]" --no-build-isolation

On Windows, run this from a plain shell rather than an MSVC developer prompt
(or set ``DISTUTILS_USE_SDK=1`` in one), and first set
``CL=/std:c++20 /Zc:preprocessor`` (PowerShell:
``$env:CL = "/std:c++20 /Zc:preprocessor"``); the ``torch-scatter`` build
takes about ten minutes.

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
   * - ``[cuda12]`` (CUDA 12.8, torch < 2.12), with ``--torch-backend=cu128``
       and ``-f``
     - Prebuilt wheel
     - Prebuilt wheel
   * - ``[cuda13]`` (CUDA 13)
     - Source build, or a prebuilt wheel for torch 2.12 with ``-f``
     - Source build

No wheel covers torch 2.13 or newer on any platform, and the CUDA 13.0 wheels
are Linux-only.

Installing with pip
--------------------

``pip`` cannot select a PyTorch build, so install a PyTorch that the
``[cuda12]`` extra accepts first, then the package:

.. code-block:: bash

   pip install "torch>=2.10,<2.12" torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install "monai-physio[cuda12]" -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

The version bound in the first command matters: an unbounded install takes
the newest torch on that index, which the extra rejects, and pip then replaces
it with the CPU-only PyPI build. See https://pytorch.org/get-started/locally/
for the PyTorch selector.

Installing from Source
-----------------------

**Step 1: Clone the repository**

.. code-block:: bash

   git clone https://github.com/Project-MONAI/monai-physio.git
   cd monai-physio

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

   uv pip install -e ".[dev_cuda12]" --torch-backend=cu128        -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

For CUDA 13, follow the pip sequence under *CUDA 13* above with
``-e ".[dev_cuda13]"`` in place of ``"monai-physio[cuda13]"``.

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

To verify that the installed PyTorch is a CUDA build that runs on your GPU:

.. code-block:: python

   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(torch.ones(1, device="cuda") + 1)  # fails if the build lacks kernels for this GPU
   import torch_scatter  # fails if torch-scatter does not match torch

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
