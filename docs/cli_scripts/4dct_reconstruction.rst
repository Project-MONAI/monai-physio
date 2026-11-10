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

TCIA-4DLung's tutorial subset downloads automatically via
``monai-physio-download-data TCIA-4DLung``; the full TCIA 4D-Lung collection
is still obtained manually - downloading its DICOM series from TCIA and
running ``data/TCIA-4DLung/convert.py`` - if more than the tutorial subset is
needed. See ``data/TCIA-4DLung/README.md`` for the download and layout
instructions.

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
       --output-dir ./results

Composite Mode
===============

By default, the high-resolution reference image is warped back to each time
point. ``--composite-mode mean``, ``--composite-mode max``, or
``--composite-mode percentile`` instead build a pixel-by-pixel composite of
the reference image and every registered time-series image first, then warp
that composite back to each time point - useful when anatomy or contrast is
only visible in some frames.

.. code-block:: bash

   monai-physio-reconstruct-highres-4d-ct \
       --time-series-images frame_*.mha \
       --fixed-image highres_reference.mha \
       --composite-mode mean \
       --output-dir ./results

Percentile mode defaults to the 95th percentile. Use
``--composite-percentile`` to choose any percentile from 0 to 100 inclusive.
Percentiles use linear interpolation between samples at each voxel; samples
outside a registered image's coverage are excluded. The reference image
contributes one sample at every voxel. The composite preserves the reference
image's grid and pixel type, rounding interpolated values for integer images.
Exact percentile calculation retains the registered samples in memory and
requires more memory than the streaming mean and max reductions.

.. code-block:: bash

   monai-physio-reconstruct-highres-4d-ct \
       --time-series-images frame_*.mha \
       --fixed-image highres_reference.mha \
       --composite-mode percentile \
       --composite-percentile 95 \
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

Configure percentile reconstruction before calling ``workflow.process()``:

.. code-block:: python

   workflow.set_composite_mode("percentile")  # Defaults to the 95th percentile
   workflow.set_composite_mode("percentile", percentile=90.0)  # Custom percentile
   result = workflow.process()

Related Pages
=============

* :doc:`../tutorials`
* :doc:`overview`
* :doc:`../api/workflows`
