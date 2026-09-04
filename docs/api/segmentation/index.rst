====================================
Segmentation Modules
====================================

.. currentmodule:: monai_physio

AI-powered anatomical structure identification from medical images using state-of-the-art deep learning models.

Overview
========

MONAI Physio supports multiple segmentation approaches:

* **TotalSegmentator**: Whole-body CT segmentation (100+ structures)
* **Simpleware**: Cardiac-focused segmentation (requires Simpleware Medical)
* **NV-Segment-CTMR**: Whole-body CT *and* MRI segmentation (345 structures,
  non-commercial license)

All segmentation classes inherit from :class:`SegmentAnatomyBase` and provide consistent interfaces.

Quick Links
===========

**Segmentation Classes**:
   * :doc:`base` - Base class for all segmentation methods
   * :doc:`totalsegmentator` - TotalSegmentator implementation
   * :doc:`simpleware` - Simpleware ASCardio cardiac segmentation
   * :doc:`nv_segment_ct_mri` - NVIDIA NV-Segment-CTMR CT/MRI segmentation

Choosing a Method
=================

+------------------+------------------+------------------+------------------+
| Method           | Speed            | Accuracy         | Best For         |
+==================+==================+==================+==================+
| TotalSegmentator | Fast (~30s)      | Good             | General purpose  |
+------------------+------------------+------------------+------------------+
| Simpleware       | Medium           | Excellent        | Cardiac imaging  |
+------------------+------------------+------------------+------------------+
| NV-Segment-CTMR  | Medium           | Good             | CT and MRI       |
+------------------+------------------+------------------+------------------+

Quick Start
===========

Basic Segmentation
------------------

.. code-block:: python

   from monai_physio import SegmentChestTotalSegmentator

   segmenter = SegmentChestTotalSegmentator()
   result = segmenter.segment(ct_image)
   labelmap = result['labelmap']

Module Documentation
====================

.. toctree::
   :maxdepth: 2

   base
   totalsegmentator
   simpleware
   nv_segment_ct_mri

Common Operations
=================

Structure Extraction
--------------------

Extract individual anatomical structures from segmentation results. The key
set returned by ``segment()`` is segmenter-specific (see :doc:`base` for the
anatomy taxonomy contract), so check membership before accessing:

.. code-block:: python

   result = segmenter.segment(ct_image)
   for group in ("heart", "lung", "bone"):
       if group in result:
           itk.imwrite(result[group], f"{group}_mask.mha")

Batch Processing
----------------

Process multiple images efficiently:

.. code-block:: python

   from pathlib import Path
   import itk

   segmenter = SegmentChestTotalSegmentator()

   for image_file in Path("data").glob("*.nrrd"):
       image = itk.imread(str(image_file))
       result = segmenter.segment(image)
       labelmap = result['labelmap']
       itk.imwrite(labelmap, f"{image_file.stem}_labels.mha")

Error Handling
--------------

.. code-block:: python

   try:
       result = segmenter.segment(image)
   except RuntimeError as e:
       print(f"Segmentation failed: {e}")

See Also
========

* :doc:`../workflows` - Using segmentation in workflows
* :doc:`../registration/index` - Register segmented images
* :doc:`../usd/index` - Convert segmentations to USD
* :doc:`../../cli_scripts/overview` - Command-line tools

.. rubric:: Navigation

:doc:`../index` | :doc:`base` | :doc:`totalsegmentator` | :doc:`simpleware`
