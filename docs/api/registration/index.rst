==========================
Image Registration Modules
==========================

.. currentmodule:: monai_physio

MONAI Physio image registration classes align moving 3D images to a fixed
3D image and return transform dictionaries.

.. toctree::
   :maxdepth: 2

   base
   ants
   greedy
   icon
   chained
   time_series

Common Result Shape
===================

``RegisterImagesANTS.register()`` and ``RegisterImagesICON.register()`` return:

* ``fixed_to_moving_transform``
* ``moving_to_fixed_transform``
* ``loss``

Use :meth:`RegisterImagesBase.get_registered_image` after ``register()`` when a
resampled moving image is needed.

Basic Example
=============

.. code-block:: python

   import itk

   from monai_physio import RegisterImagesANTS

   fixed = itk.imread("reference.mha")
   moving = itk.imread("moving.mha")

   registrar = RegisterImagesANTS()
   registrar.set_modality("ct")
   registrar.set_fixed_image(fixed)

   result = registrar.register(moving)
   registered = registrar.get_registered_image()

See Also
========

* :doc:`../model_registration/index`
* :doc:`../workflows`
