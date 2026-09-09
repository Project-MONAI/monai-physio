=========================
Utilities Developer Guide
=========================

MONAI Physio utility classes provide reusable operations for images,
transforms, contours, and USD files. They are class-based APIs, not module-level
free functions.

Transform Tools
===============

.. code-block:: python

   import itk

   from monai_physio import TransformTools

   tools = TransformTools()
   transform = itk.transformread("fixed_to_moving_transform.hdf")
   moving = itk.imread("moving.mha")
   reference = itk.imread("reference.mha")

   warped = tools.transform_image(moving, transform, reference)

For PyVista contours:

.. code-block:: python

   import itk
   import pyvista as pv

   from monai_physio import TransformTools

   mesh = pv.read("heart_t0.vtp")
   transform = itk.transformread("fixed_to_moving_transform.hdf")

   transformed = TransformTools().transform_pvcontour(mesh, transform)

Contour Tools
=============

.. code-block:: python

   import itk

   from monai_physio import ContourTools

   mask = itk.imread("heart_mask.nrrd")
   contour = ContourTools().extract_contours(mask)
   contour.save("heart_surface.vtp")

USD Tools
=========

.. code-block:: python

   from monai_physio import USDTools

   tools = USDTools()
   tools.merge_usd_files(
       "combined.usd",
       ["heart.usd", "lung.usd", "vessels.usd"],
   )

USD Anatomy Tools
=================

.. code-block:: python

   from pxr import Usd

   from monai_physio import USDAnatomyTools

   stage = Usd.Stage.Open("anatomy.usd")
   painter = USDAnatomyTools(stage)
   painter.apply_anatomy_material_to_mesh("/World/Heart", "heart")
   stage.Export("anatomy_painted.usd")

Image Tools
===========

``ImageTools`` contains conversion and small image helpers used internally by
registration and transform utilities. Prefer direct ITK I/O for ordinary
example code:

.. code-block:: python

   import itk

   image = itk.imread("input.nrrd")
   itk.imwrite(image, "output.mha")

See Also
========

* :doc:`../api/utilities/index`
* :doc:`../api/usd/index`
* :doc:`../developer/usd_generation`
