================================
ICON Image Registration
================================

.. module:: monai_physio.register_images_icon
.. currentmodule:: monai_physio

``RegisterImagesICON`` performs deformable image registration using the
uniGradICON registration backend.

Class Reference
===============

.. autoclass:: RegisterImagesICON
   :members:
   :undoc-members:
   :show-inheritance:

Basic Registration
==================

.. code-block:: python

   import itk

   from monai_physio import RegisterImagesICON

   fixed = itk.imread("reference_frame.mha")
   moving = itk.imread("moving_frame.mha")

   registrar = RegisterImagesICON()
   registrar.set_modality("ct")
   registrar.set_number_of_iterations(50)
   registrar.set_fixed_image(fixed)

   result = registrar.register(moving)

   fixed_to_moving_transform = result["fixed_to_moving_transform"]
   moving_to_fixed_transform = result["moving_to_fixed_transform"]
   loss = result["loss"]
   registered = registrar.get_registered_image()

Result Dictionary
=================

``register()`` returns:

* ``fixed_to_moving_transform``: warps the moving image onto the fixed grid
  (or maps a fixed-space point to moving space)
* ``moving_to_fixed_transform``: warps the fixed image onto the moving grid
  (or maps a moving-space point to fixed space)
* ``loss``: registration loss value reported by the backend

Configuration
=============

Use ``set_number_of_iterations()`` to control per-pair refinement. Use
``set_multi_modality()`` and ``set_mass_preservation()`` for modality-specific
behavior. There is no public ``set_device()`` method; device selection is owned
by the underlying PyTorch/ICON runtime and installed CUDA environment.

See Also
========

* :doc:`ants`
* :doc:`time_series`
* :doc:`../../cli_scripts/heart_gated_ct`
