====================================
4D Image Conversion
====================================

.. currentmodule:: monai_physio

Utilities for converting 4D medical images into a 3D time-series sequence.
``.nrrd`` inputs (including Slicer ``.seq.nrrd`` heart sequences, whose
per-voxel vector dimension exceeds ITK Python's wrapped Vector sizes) are read
with ``pynrrd``; all other formats (NIfTI ``.nii.gz``, MHA, …) are read with
``itk.imread`` and must be true 4-dimensional images.

Module Reference
================

.. automodule:: monai_physio.convert_image_4d_to_3d
   :members:
   :undoc-members:

.. rubric:: Navigation

:doc:`contour_tools` | :doc:`index` | :doc:`../index`
