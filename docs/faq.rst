===
FAQ
===

Frequently Asked Questions about MONAI Physio.

General Questions
=================

What is MONAI Physio?
-----------------------

MONAI Physio is a collection of methods, workflows, tutorials, and CLI tools for
creating personalized physiological digital twins: starting from a 3D medical
image of a subject, extracting anatomic models, and then using AI surrogates to
estimate the subject's physiological processes (initially cardiac and
respiratory motion, expanding to electrophysiology, blood flow, and organ
perfusion).

What data formats are supported?
---------------------------------

* **Input**: NRRD, MHA, NIfTI, DICOM
* **Output**: USD (Universal Scene Description), VTK

Do I need NVIDIA Omniverse?
----------------------------

Omniverse is the recommended way to view the USD scenes: its RTX renderer is
what evaluates the material properties assigned to each tissue. See
:doc:`viewing_usd`. For the intermediate results you can also use:

* PyVista, for the intermediate ``.vtp`` / ``.vtu`` meshes
* ParaView, likewise for the VTK files

Installation Questions
======================

Do I need a GPU?
----------------

No. A plain ``pip install monai-physio`` works without a GPU, and runs every
workflow including the AI-surrogate ones - just significantly slower than a
GPU-enabled install. At import time a ``UserWarning`` is emitted (visible by
default in all standard Python runs):

.. code-block:: text

   CuPy is not installed - GPU-accelerated mesh operations will fall back to
   NumPy and run significantly slower. Every workflow still runs. Re-install
   with uv to get CuPy and CUDA-enabled PyTorch in one step (pip alone will
   not select the correct CUDA wheel):
     uv pip install 'monai-physio[cuda12]'  # CUDA 12.6
     uv pip install 'monai-physio[cuda13]'  # CUDA 13

CPU-only mode is suitable for evaluation and small datasets. For production
workloads an NVIDIA GPU is strongly recommended.

Which CUDA version is required?
--------------------------------

Both CUDA 13 and CUDA 12.6 are supported. CUDA 12.6 is the smoothest
install, because it is the newest combination with prebuilt ``torch-scatter``
wheels on every supported platform - nothing compiles:

.. code-block:: bash

   uv pip install "monai-physio[cuda12]"   # CUDA 12.6
   uv pip install "monai-physio[cuda13]"   # CUDA 13, builds torch-scatter

Each extra installs CuPy and pins PyTorch, torchvision, and torchaudio to the
matching wheel index (``https://download.pytorch.org/whl/cu126`` or
``cu130``). On CUDA 13 there is no prebuilt ``torch-scatter`` wheel, so it
compiles from source and needs a CUDA toolkit plus a C++ toolchain; see
:doc:`installation`.

To let uv pick the PyTorch build from your driver without naming a CUDA
version - at the cost of CuPy, which has no auto-detect option:

.. code-block:: bash

   uv pip install --torch-backend=auto monai-physio

What Python version is required?
---------------------------------

Python 3.11, 3.12 and 3.13 are supported.

The floor is set by ``nvidia-physicsnemo`` (a base dependency): it supports
>= 3.11, < 3.14, and the AI-surrogate tutorials need it. The rest of the
library would run on 3.10, but declaring 3.10 would promise an install that
cannot resolve.

Usage Questions
===============

How long does processing take?
-------------------------------

Typical processing time for 10-frame cardiac CT (with GPU):

* 4D to 3D conversion: ~1 minute
* Registration: ~5-10 minutes
* Segmentation: ~1-2 minutes
* USD creation: ~1 minute
* **Total**: ~10-15 minutes

Which segmentation method should I use?
----------------------------------------

* **TotalSegmentator**: Fast, good quality, general purpose
* **Simpleware**: Best quality for cardiac imaging, requires Simpleware Medical
* **NV-Segment-CTMR**: CT *and* MRI, 345 classes; weights are licensed for
  non-commercial academic research only

See :doc:`api/segmentation/index` for comparison.

Which registration method should I use?
----------------------------------------

* **Greedy**: CPU-capable classical deformable registration; what Tutorials 1
  and 3 use by default
* **ICON**: Recommended for cardiac/lung (fast, GPU), and finetunable on your
  own cohort - see Tutorial 2
* **ANTs**: Best for brain imaging and general purpose
* **Greedy+ICON** (``RegisterImagesGreedyICON``, a ``RegisterImagesChain``
  preset): Greedy for the coarse alignment, ICON for the refinement

See :doc:`api/registration/index` for comparison.

Troubleshooting
===============

See :doc:`troubleshooting` for common issues and solutions.

More Questions?
===============

* Check the :doc:`cli_scripts/heart_gated_ct`
* Browse :doc:`tutorials`
* Open an issue on `GitHub <https://github.com/Project-MONAI/monai-physio/issues>`_

