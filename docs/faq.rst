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

Yes. Surface extraction, segmentation, and the AI surrogates run on an NVIDIA
GPU, and there is no CPU-only mode.

Which CUDA version is required?
--------------------------------

A Turing (compute capability 7.5) or newer GPU with a driver for CUDA 12.8 or
newer. The recommended install uses PyTorch's CUDA
12.8 build, the newest CUDA 12 build with Blackwell kernels and prebuilt
``torch-scatter`` wheels on every supported platform, so nothing compiles:

.. code-block:: bash

   uv pip install "monai-physio[cuda12]" --torch-backend=cu128 \
       -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

CUDA 13 is also supported, but ``torch-scatter`` then compiles from source on
Windows (and on Linux for any torch other than 2.12), which needs a CUDA
toolkit and a C++ toolchain; see :doc:`installation`.

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

Why does inference fail with "Fitted reference mesh has N points, expected M"?
-------------------------------------------------------------------------------

An error such as:

.. code-block:: text

   File ".../src/monai_physio/workflow_infer_movement.py", line 79, in _fitted_reference_points
     raise ValueError(
   ValueError: Fitted reference mesh has 20000 points, expected 19997 (template topology).

means that the fitted mesh was produced by a different PCA shape model than the
one the AI model in ``tutorials/network_weights`` was trained with.

Most often the fitted mesh was cached in a tutorial's output directory by an
earlier run, before the shape model changed. Delete that cached fit so the
tutorial fits the current shape model again. Tutorial 13 does this on its own
when ``pca_model.json`` is newer than its saved heart fit.

If instead the shape model in ``tutorials/network_weights`` is not the one the
AI model there was trained with, either retrain the AI model using that shape
model, or, if you are running a tutorial, download the pre-trained PCA shape
model and AI model with the matching ``monai-physio-download-data`` command, for
example:

.. code-block:: bash

   monai-physio-download-data PhysicsNeMo-MGN-Lung-Motion

The other pre-trained models are ``PhysicsNeMo-MGN-Duke-Heart-Motion`` and
``PhysicsNeMo-PhysicsInformed-Motion-Duke-Heart``.

Troubleshooting
===============

See :doc:`troubleshooting` for common issues and solutions.

More Questions?
===============

* Check the :doc:`cli_scripts/heart_gated_ct`
* Browse :doc:`tutorials`
* Open an issue on `GitHub <https://github.com/Project-MONAI/monai-physio/issues>`_

