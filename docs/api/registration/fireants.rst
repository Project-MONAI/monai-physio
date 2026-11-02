=====================
FireANTs Registration
=====================

.. module:: monai_physio.register_images_fireants
.. currentmodule:: monai_physio

``RegisterImagesFireants`` provides GPU-accelerated rigid, affine, and
deformable image registration using the FireANTs backend.

Class Reference
===============

.. autoclass:: RegisterImagesFireants
   :members:
   :undoc-members:
   :show-inheritance:

Basic Registration
==================

.. code-block:: python

   import itk

   from monai_physio import RegisterImagesFireants

   fixed = itk.imread("reference.mha")
   moving = itk.imread("moving.mha")

   registrar = RegisterImagesFireants()
   registrar.set_modality("ct")
   registrar.set_fixed_image(fixed)

   result = registrar.register(moving)

   fixed_to_moving_transform = result["fixed_to_moving_transform"]
   moving_to_fixed_transform = result["moving_to_fixed_transform"]
   registered = registrar.get_registered_image()

Installation Note
=================

``RegisterImagesFireants`` requires the ``fireants`` package, which is not
installed by default. Install it without its dependencies, so that it does not
replace the CUDA build of PyTorch already in the environment:

.. code-block:: bash

   pip install fireants --no-deps

See Also
========

* :doc:`greedy`
* :doc:`icon`
* :doc:`time_series`
