====================================
Utility Modules
====================================

.. currentmodule:: monai_physio

Common utilities for image processing, transforms, and file handling.

Overview
========

Utility modules provide low-level operations:

* **Image Tools**: Image I/O, preprocessing, manipulation
* **Labelmap Tools**: Labelmap to registration-mask conversion
* **Transform Tools**: Coordinate transforms and warping
* **Landmark Tools**: Landmark-based registration validation metrics
* **Contour Tools**: Contour extraction and processing
* **4D Image Conversion**: 4D image to 3D time-series conversion utilities
* **Test Tools**: Baseline and result comparison helpers
* **Data Download Tools**: Optional dataset download helpers

Quick Links
===========

**Utility Modules**:
   * :doc:`process_images` - Image processing utilities
   * :doc:`process_labelmaps` - Labelmap to registration-mask conversion
   * :doc:`process_transforms` - Transform operations
   * :doc:`process_landmarks` - Landmark-based registration validation
   * :doc:`process_contours` - Contour processing
   * :doc:`image_conversion` - 4D image to 3D time-series utilities
   * :doc:`process_tests` - Baseline / result comparison helpers
   * :doc:`download_data` - Optional dataset download helpers

Module Documentation
====================

.. toctree::
   :maxdepth: 2

   process_images
   process_labelmaps
   process_transforms
   process_landmarks
   process_contours
   image_conversion
   process_tests
   download_data

See Also
========

* :doc:`../workflows` - High-level workflows using utilities
* :doc:`../index` - Complete API reference

.. rubric:: Navigation

:doc:`../usd/vtk_conversion` | :doc:`../index` | :doc:`process_images`
