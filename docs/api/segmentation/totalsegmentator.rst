================
TotalSegmentator
================

.. module:: monai_physio.segment_chest_total_segmentator
.. currentmodule:: monai_physio

``SegmentChestTotalSegmentator`` groups a TotalSegmentator labelmap into the
anatomy masks used by MONAI Physio workflows.

``SegmentChestTotalSegmentatorWithContrast`` is the contrast-enhanced variant:
same interface, thresholds and grouping tuned for contrast CT. Pick it when
your scan has vascular contrast, and the plain class otherwise.

Class Reference
===============

.. autoclass:: SegmentChestTotalSegmentator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SegmentChestTotalSegmentatorWithContrast
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
===========

.. code-block:: python

   import itk

   from monai_physio import SegmentChestTotalSegmentator

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentChestTotalSegmentator()

   masks = segmenter.segment(image)

   heart = masks["heart"]
   lungs = masks["lung"]
   vessels = masks["major_vessels"]
   labelmap = masks["labelmap"]

   itk.imwrite(heart, "heart_mask.nrrd")
   itk.imwrite(lungs, "lung_mask.nrrd")
   itk.imwrite(vessels, "major_vessels_mask.nrrd")
   itk.imwrite(labelmap, "labelmap.nrrd")

Returned Keys
=============

For this segmenter, ``segment()`` returns a dictionary with the following
keys:

* ``labelmap``
* ``lung``
* ``heart``
* ``major_vessels``
* ``bone``
* ``soft_tissue``
* ``other``

The dictionary should be accessed by key. Do not unpack it positionally.
The exact key set is determined by the segmenter's :class:`AnatomyTaxonomy`
and may differ from other segmenters (see :doc:`base`). For
:class:`SegmentChestTotalSegmentator` specifically, all six groups plus
``labelmap`` are always present; downstream code that targets a different
segmenter should check membership.

For contrast-enhanced studies, use
:class:`SegmentChestTotalSegmentatorWithContrast` instead of
:class:`SegmentChestTotalSegmentator`. It adds a ``contrast`` key to the
returned dictionary and exposes a ``contrast_threshold`` attribute
(default 500) that can be overridden before calling ``segment()``:

.. code-block:: python

   from monai_physio import SegmentChestTotalSegmentatorWithContrast

   segmenter = SegmentChestTotalSegmentatorWithContrast()
   segmenter.contrast_threshold = 600  # optional override

   masks = segmenter.segment(image)
   contrast = masks["contrast"]

Operational Notes
=================

TotalSegmentator model inference may download model assets and can be slow on a
CPU-only environment. For repeatable workflows, prefer the tutorial scripts or
the ``monai-physio-convert-image-to-vtk`` CLI.

See Also
========

* :doc:`index`
* :doc:`../../cli_scripts/overview`
* :doc:`../../tutorials`
