========================
Time-Series Registration
========================

.. module:: monai_physio.register_time_series_images
.. currentmodule:: monai_physio

``RegisterTimeSeriesImages`` registers ordered 3D image phases to a reference
frame using a caller-supplied :class:`RegisterImagesBase` backend (e.g.
:class:`RegisterImagesGreedy`, :class:`RegisterImagesICON`, or
:class:`RegisterImagesGreedyICON` for combined Greedy-then-ICON refinement).

Class Reference
===============

.. autoclass:: RegisterTimeSeriesImages
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: registration_method

Basic Usage
===========

.. code-block:: python

   import itk

   from monai_physio import RegisterImagesGreedy, RegisterTimeSeriesImages

   images = [itk.imread(f"phase_{idx:02d}.mha") for idx in range(10)]

   registrar = RegisterTimeSeriesImages(registration_method=RegisterImagesGreedy())
   registrar.set_fixed_image(images[0])

   result = registrar.register_time_series(
       moving_images=images,
       reference_frame=0,
       register_reference=False,
   )
   forward_transforms = result["forward_transforms"]
   inverse_transforms = result["inverse_transforms"]

See Also
========

* :doc:`ants`
* :doc:`greedy`
* :doc:`icon`
* :doc:`../../cli_scripts/4dct_reconstruction`
