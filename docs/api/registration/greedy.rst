===================
Greedy Registration
===================

.. module:: monai_physio.register_images_greedy
.. currentmodule:: monai_physio

``RegisterImagesGreedy`` provides fast CPU-based deformable image registration
using the PICSL Greedy backend.

Class Reference
===============

.. autoclass:: RegisterImagesGreedy
   :members:
   :undoc-members:
   :show-inheritance:

Basic Registration
==================

.. code-block:: python

   import itk

   from monai_physio import RegisterImagesGreedy

   fixed = itk.imread("reference.mha")
   moving = itk.imread("moving.mha")

   registrar = RegisterImagesGreedy()
   registrar.set_modality("ct")
   registrar.set_fixed_image(fixed)

   result = registrar.register(moving)

   fixed_to_moving_transform = result["fixed_to_moving_transform"]
   moving_to_fixed_transform = result["moving_to_fixed_transform"]
   registered = registrar.get_registered_image()

Installation Note
=================

``RegisterImagesGreedy`` requires the ``picsl-greedy`` package, which is not
installed by default:

.. code-block:: bash

   pip install picsl-greedy

See Also
========

* :doc:`ants`
* :doc:`icon`
* :doc:`time_series`
