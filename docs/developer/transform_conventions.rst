===============================
Transform Direction Conventions
===============================

Registration in MONAI Physio produces a pair of transforms, named for the
literal spatial mapping they perform:

``fixed_to_moving_transform``
    ``TransformPoint()`` maps a **fixed**-space coordinate to the
    corresponding **moving**-space coordinate.

``moving_to_fixed_transform``
    ``TransformPoint()`` maps a **moving**-space coordinate to the
    corresponding **fixed**-space coordinate.

Every registration class -- image (:class:`monai_physio.RegisterImagesANTS`,
:class:`monai_physio.RegisterImagesICON`,
:class:`monai_physio.RegisterImagesGreedy`,
:class:`monai_physio.RegisterModelsDistanceMaps`) and model
(:class:`monai_physio.RegisterModelsPCA`,
:class:`monai_physio.RegisterModelsICP`) -- returns exactly these two names.
:class:`monai_physio.RegisterTimeSeriesImages` returns the list-valued
``fixed_to_moving_transforms`` / ``moving_to_fixed_transforms``.

Because the name states the direction literally, mapping a *point* needs no
lookup table -- just pick the transform whose name matches the source and
target point space you have. Warping an *image* is different:
:func:`TransformTools.transform_image` samples whichever grid you are
building the output on, so the transform is picked by output grid, not by
source/target space (see below).

Read this page before applying any transform to an image, mask, contour, or
landmark.

Image warping vs. point warping use opposite transforms
========================================================

ITK resampling is a *pull-back* operation. To build the warped image on the
fixed grid, :func:`TransformTools.transform_image` (an ``itk.ResampleImageFilter``)
visits every fixed-grid sample ``q`` and looks up the moving image at
``transform.TransformPoint(q)``. The transform it needs therefore maps
**fixed-space coordinates to moving-space coordinates** --
``fixed_to_moving_transform``.

Warping a *point* (landmark, contour vertex, mesh node) is a *push-forward*
operation: :func:`TransformTools.transform_pvcontour` /
:func:`TransformTools.transform_dataset` apply ``transform.TransformPoint(p)``
directly to each input point. To move a moving-space landmark to its location
in the fixed image, the transform must map **moving-space coordinates to
fixed-space coordinates** -- ``moving_to_fixed_transform``.

.. list-table:: Which transform to apply
   :header-rows: 1
   :widths: 50 25 25

   * - Goal
     - Transform
     - Helper
   * - Warp the **moving image** into fixed space (onto the fixed grid)
     - ``fixed_to_moving_transform``
     - :func:`TransformTools.transform_image`
   * - Warp **moving points / contours / landmarks** into fixed space
     - ``moving_to_fixed_transform``
     - :func:`TransformTools.transform_pvcontour`
   * - Warp the **fixed image** into moving space (e.g. time-series reconstruction)
     - ``moving_to_fixed_transform``
     - :func:`TransformTools.transform_image`
   * - Warp **fixed points / contours / landmarks** into moving space
     - ``fixed_to_moving_transform``
     - :func:`TransformTools.transform_pvcontour`

.. note::

   All registration classes (image and model alike) follow this same
   convention. ``transform_image(moving, fixed_to_moving_transform, fixed)``
   is the correct call to warp the moving image onto the fixed grid for every
   backend, and ``transform_pvcontour(points, moving_to_fixed_transform)`` is
   the correct call to warp moving points/mesh nodes into fixed space for
   every backend -- image or model.

PCA / ICP point transforms
===========================

:class:`monai_physio.RegisterModelsPCA` builds ``moving_to_fixed_transform``
directly from the template-to-target point displacement, so
``moving_to_fixed_transform.TransformPoint(template_point)`` returns the
corresponding *target* point -- treating the template as the moving object
and the patient/target as the fixed object. Deforming the template mesh onto
the patient (the usual PCA use, performed internally by
``transform_template_model()`` and ``transform_point()``) uses
``moving_to_fixed_transform``; resampling an image with the PCA result uses
``fixed_to_moving_transform``. :class:`monai_physio.RegisterModelsICP`
follows the same convention.

Rule of thumb
=============

* **Images pull back; points push forward.** For one registration result, the
  image and the points always use the two *different* members of the
  transform pair.
* **Image into the reference frame** -> ``fixed_to_moving_transform``.
* **Points into the reference frame** -> ``moving_to_fixed_transform``.
* When in doubt, warp a known landmark and a small image patch and confirm
  they land in the same place before trusting a pipeline.

See Also
========

* :doc:`registration_images`
* :doc:`registration_models`
* :doc:`utilities`
