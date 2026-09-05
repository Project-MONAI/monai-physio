===============
Troubleshooting
===============

Common issues and solutions for MONAI Physio.

Installation Issues
===================

CUDA Out of Memory
------------------

**Problem**: ``RuntimeError: CUDA out of memory``

**Solutions**:

1. Resample or crop the input image before running the workflow.
2. Use ``--registration-method Greedy`` when CUDA is unavailable.
3. Process fewer frames per run.

Process Killed During Registration (Host Out of Memory)
-------------------------------------------------------

**Problem**: a run stops with a bare ``Killed`` and no traceback, typically just
after a line of ICON finetuning output:

.. code-block:: text

   ICONLoss(all_loss=tensor(1.0676, device='cuda:0', ...), ...)
   Killed

**Cause**: the Linux OOM killer, not CUDA. A GPU shortage raises a catchable
``RuntimeError: CUDA out of memory`` with a Python traceback; ``Killed`` is the
shell reporting that the kernel sent ``SIGKILL`` because the machine ran out of
*host* RAM.

Confirm it, and see how much was in use at the time:

.. code-block:: bash

   dmesg | grep -i "killed process"

Read ``anon-rss`` in that line. It is the process's own heap; if ``file-rss`` is
near zero the memory was genuinely allocated, rather than reclaimable file cache.

**Cause on WSL2**: WSL2 caps its virtual machine at **half the host's RAM** by
default, so a 128 GB machine gives Linux only about 64 GB and the OOM killer
fires at that ceiling rather than at the physical limit. Check what Linux
actually sees:

.. code-block:: bash

   grep MemTotal /proc/meminfo

**Solutions**:

1. On WSL2, raise the ceiling in the Windows-side ``.wslconfig`` (in your user
   profile directory), then run ``wsl --shutdown`` and restart the
   distribution:

   .. code-block:: ini

      [wsl2]
      memory=112GB
      swap=32GB

2. Process fewer cases or frames per run. The cohort workflows cache every
   artifact they write, so a re-run resumes where it stopped rather than
   starting over.
3. Coarsen the registration grid, which sets the size of the distance maps and
   displacement fields held during a registration.

CUDA Version Mismatch
---------------------

**Problem**: Errors such as ``cupy`` failing to import, ``torch.cuda.is_available()``
returning ``False``, or runtime messages indicating a CUDA library version conflict.

**Cause**: The installed ``cupy`` or PyTorch wheel was built for a different CUDA
version than the one present on the system.

**Solution**: Install the extra matching your CUDA version:

.. code-block:: bash

   uv pip install "monai-physio[cuda12]"   # CUDA 12.6, nothing compiles
   uv pip install "monai-physio[cuda13]"   # CUDA 13, builds torch-scatter

Each extra installs CuPy and pins PyTorch to the matching wheel index.
``[cuda12]`` is the smoothest: it is the newest combination with prebuilt
``torch-scatter`` wheels, while CUDA 13 has none and compiles it from source.
If you're not sure which CUDA version to pick, ``uv pip install
--torch-backend=auto monai-physio`` auto-detects the driver and installs a
matching PyTorch build without CuPy.

Verify the active CUDA version before reinstalling:

.. code-block:: bash

   nvidia-smi   # shows driver and CUDA version

.. note::
   If you have no NVIDIA GPU, a plain ``pip install monai-physio`` installs a
   CPU-only build. CuPy is absent and a ``UserWarning`` is emitted at import time.
   CPU execution of all operations is supported but will be significantly slower
   than a GPU-enabled install.

Import Errors
-------------

**Problem**: ``ImportError: No module named 'itk'``

**Solution**: Reinstall with all dependencies:

.. code-block:: bash

   pip install --upgrade monai_physio

Processing Issues
=================

Poor Segmentation Quality
-------------------------

**Problem**: Segmentation masks are inaccurate

**Solutions**:

1. Check if image is contrast-enhanced. Use
   :class:`SegmentChestTotalSegmentatorWithContrast` instead of
   :class:`SegmentChestTotalSegmentator` for contrast-enhanced studies:

   .. code-block:: python

      from monai_physio import (
          SegmentChestTotalSegmentatorWithContrast,
          WorkflowConvertImageToUSD,
      )

      workflow = WorkflowConvertImageToUSD(
          ...,
          segmentation_method=SegmentChestTotalSegmentatorWithContrast(),
      )

2. Preprocess intensity, spacing, and field of view before invoking the workflow.

Registration Not Converging
---------------------------

**Problem**: Registration produces poor results

**Solutions**:

1. Increase ``--registration-iterations`` for the heart-gated CT CLI.

2. Try different method:

   .. code-block:: bash

      monai-physio-convert-image-to-usd cardiac_4d.nrrd --registration-method Greedy

3. Check image orientation and spacing

USD Issues
==========

USD Not Animating
-----------------

**Problem**: USD file loads but doesn't animate

**Solutions**:

1. Validate USD file:

   .. code-block:: bash

      usdchecker model.usd

   ``usdchecker`` is not part of the ``usd-core`` package installed with
   MONAI Physio; it ships with the OpenUSD toolset, available pre-built from
   https://developer.nvidia.com/usd.

2. Open the scene in an Omniverse Kit application, switch the viewport to the
   scene's ``/World/Camera``, and press Play; see :doc:`viewing_usd`.

3. Verify that the generated USD contains time samples.

USD File Too Large
------------------

**Problem**: USD files are very large

**Solutions**:

1. Reduce mesh complexity before USD export.
2. Export fewer anatomy groups or fewer time points.

Performance Issues
==================

Slow Processing
---------------

**Problem**: Processing takes too long

**Solutions**:

1. Install ``monai-physio[cuda12]`` (or ``[cuda13]``) for GPU-accelerated
   PyTorch and CuPy, or ``uv pip install --torch-backend=auto monai-physio``
   for auto-detected PyTorch without CuPy.
2. Reduce ``--registration-iterations`` during exploratory runs.
3. Run tutorial workflows with reduced frame counts where supported.

Getting Help
============

If you still have issues:

1. Check :doc:`faq`
2. Search `GitHub Issues <https://github.com/Project-MONAI/monai-physio/issues>`_
3. Open a new issue with:

   * Python version
   * CUDA version
   * Complete error message
   * Minimal code to reproduce

