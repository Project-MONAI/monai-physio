==================================
Image Registration Developer Guide
==================================

MONAI Physio image registration classes register a moving ITK image to a
fixed ITK image.

Basic Pattern
=============

.. code-block:: python

   import itk

   from monai_physio import RegisterImagesANTS

   fixed = itk.imread("fixed.mha")
   moving = itk.imread("moving.mha")

   registrar = RegisterImagesANTS()
   registrar.set_modality("ct")
   registrar.set_fixed_image(fixed)

   result = registrar.register(moving)
   registered = registrar.get_registered_image()

The result dictionary contains ``fixed_to_moving_transform``,
``moving_to_fixed_transform``, and ``loss``. Applying the right one is
critical and direction-dependent: ``fixed_to_moving_transform`` warps the
moving image onto the fixed grid, while ``moving_to_fixed_transform`` warps
moving points/landmarks into fixed space (image and point warps use opposite
transforms). See :doc:`transform_conventions` for the full rules.

Time Series
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

Combining Registrars
=====================

Workflows that accept a ``registration_method`` (e.g.
:class:`WorkflowConvertImageToUSD`, :class:`RegisterTimeSeriesImages`) take
any :class:`RegisterImagesBase` instance, including a composite chain that
runs multiple backends in sequence. :class:`RegisterImagesChain` runs an
ordered list of registrars, each stage refining the previous stage's
``fixed_to_moving_transform`` through ``register_from()`` (see `Seeding a registration`_
below). :class:`RegisterImagesGreedyICON` is a named 2-stage convenience class
for the common case of a fast Greedy registration followed by ICON refinement:

.. code-block:: python

   from monai_physio import RegisterImagesChain, RegisterImagesGreedy, RegisterImagesICON

   # Arbitrary N-stage chain
   registrar = RegisterImagesChain([RegisterImagesGreedy(), RegisterImagesICON()])

   # Or, for the common Greedy-then-ICON case:
   from monai_physio import RegisterImagesGreedyICON

   registrar = RegisterImagesGreedyICON()
   registrar.greedy.set_number_of_iterations([30, 15, 7, 3])
   registrar.icon.set_number_of_iterations(20)

Seeding a registration
======================

To start from an alignment you already have, call ``register_from()`` instead of
``register()``:

.. code-block:: python

   result = registrar.register_from(known_fixed_to_moving_transform, moving_image)

It warps the moving image, mask and labelmap onto the fixed grid by that
transform, registers the residual, and composes the two, so the returned
transforms still map between the *original* moving image and the fixed image.
Every backend goes through this one implementation -- no registrar accepts an
initial transform of its own, which is what keeps the pre-warp, the composition
and the inversion identical for Greedy, ICON and ANTs.

Development Notes
=================

* Use masks when registration should focus on a specific anatomy.
* Check transform direction before applying transforms to contours or images.
* Use ``TransformTools.transform_image()`` for resampling images.
* Use ``TransformTools.transform_pvcontour()`` for PyVista contours.

See Also
========

* :doc:`transform_conventions`
* :doc:`../api/registration/index`
* :doc:`workflows`
