====================================
High-Resolution 4D CT Reconstruction
====================================

The ``monai-physio-reconstruct-highres-4d-ct`` command reconstructs a
high-resolution 4D CT time series from ordered phase images and a
high-resolution reference image.

Input Requirements
==================

* Ordered 3D phase images, such as ``.mha``, ``.mhd``, ``.nrrd``, or
  ``.nii.gz`` files.
* A fixed high-resolution reference image.
* Optional fixed and moving masks for registration focus.

DirLab-4DCT data cannot be downloaded automatically by MONAI Physio. Prepare
it manually before using the DirLab tutorial or examples — see
``data/DirLab-4DCT/README.md`` for the download and layout instructions.

Basic Usage
===========

.. code-block:: bash

   monai-physio-reconstruct-highres-4d-ct \
       --time-series-images frame_*.mha \
       --fixed-image highres_reference.mha \
       --output-dir ./results

Choosing a Reference Frame
===========================

.. code-block:: bash

   monai-physio-reconstruct-highres-4d-ct \
       --time-series-images frame_000.mha frame_001.mha frame_002.mha \
       --fixed-image highres_reference.mha \
       --reference-frame 0 \
       --output-dir ./results

Registration Options
====================

.. code-block:: bash

   monai-physio-reconstruct-highres-4d-ct \
       --time-series-images frame_*.mha \
       --fixed-image highres_reference.mha \
       --registration-method Greedy_ICON \
       --Greedy-iterations 30 15 7 3 \
       --prior-weight 0.5 \
       --output-dir ./results

Composite Mode
===============

By default, the high-resolution reference image is warped back to each time
point. ``--composite-mode mean`` or ``--composite-mode max`` instead build a
pixel-by-pixel mean or max composite of the reference image and every
registered time-series image first, then warp that composite back to each
time point — useful when anatomy or contrast is only visible in some frames.

.. code-block:: bash

   monai-physio-reconstruct-highres-4d-ct \
       --time-series-images frame_*.mha \
       --fixed-image highres_reference.mha \
       --composite-mode mean \
       --output-dir ./results

Outputs
=======

The command writes reconstructed images to ``--output-dir`` using
``--output-prefix`` as the filename prefix. Use ``--save-transforms`` and
``--save-losses`` when registration diagnostics are needed.

Python API
==========

Use :class:`monai_physio.WorkflowReconstructHighres4DCT` for programmatic
access.

Related Pages
=============

* :doc:`../tutorials`
* :doc:`overview`
* :doc:`../api/workflows`
