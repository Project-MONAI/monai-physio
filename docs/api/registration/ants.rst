=================
ANTs Registration
=================

.. module:: monai_physio.register_images_ants
.. currentmodule:: monai_physio

``RegisterImagesANTS`` provides optimization-based deformable image
registration through ANTs.

Class Reference
===============

.. autoclass:: RegisterImagesANTS
   :members:
   :undoc-members:
   :show-inheritance:

Basic Registration
==================

.. code-block:: python

   import itk

   from monai_physio import RegisterImagesANTS

   fixed = itk.imread("reference.mha")
   moving = itk.imread("moving.mha")

   registrar = RegisterImagesANTS()
   registrar.set_modality("ct")
   registrar.set_transform_type("SyN")
   registrar.set_number_of_iterations([30, 15, 7])
   registrar.set_fixed_image(fixed)

   result = registrar.register(moving)

   fixed_to_moving_transform = result["fixed_to_moving_transform"]
   moving_to_fixed_transform = result["moving_to_fixed_transform"]
   registered = registrar.get_registered_image()

See Also
========

* :doc:`icon`
* :doc:`time_series`
