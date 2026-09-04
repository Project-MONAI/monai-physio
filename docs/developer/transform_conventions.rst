===============================
Transform Direction Conventions
===============================

Registration in MONAI Physio produces a pair of transforms, and choosing the
wrong one of the pair is the single most common registration mistake. The rules
are simple but easy to get backwards, because **warping an image and warping a
point require opposite transforms**, and because **model (PCA) registration
returns its transforms in the opposite orientation from image registration**.

Read this page before applying any transform to an image, mask, contour, or
landmark.

The two transform families
===========================

Image registration
    :class:`monai_physio.RegisterImagesANTS`,
    :class:`monai_physio.RegisterImagesICON`, and
    :class:`monai_physio.RegisterImagesGreedy` register a *moving* image to a
    *fixed* image and return a dict with ``forward_transform`` and
    ``inverse_transform``. :class:`monai_physio.RegisterTimeSeriesImages`
    returns the list-valued ``forward_transforms`` / ``inverse_transforms``.

Model (PCA) registration
    :class:`monai_physio.RegisterModelsPCA` deforms a *template* model toward
    a *target* (patient) and, via ``compute_pca_transforms()``, returns
    ``forward_point_transform`` and ``inverse_point_transform``. These are
    **point transforms**, oriented opposite to the image-registration transforms
    (see `PCA point transforms`_ below).

Image warping vs. point warping use opposite transforms
========================================================

ITK resampling is a *pull-back* operation. To build the warped image on the
fixed grid, :func:`TransformTools.transform_image` (an ``itk.ResampleImageFilter``)
visits every fixed-grid sample ``q`` and looks up the moving image at
``transform.TransformPoint(q)``. The transform it needs therefore maps
**fixed-space coordinates to moving-space coordinates**.

Warping a *point* (landmark, contour vertex, mesh node) is a *push-forward*
operation: :func:`TransformTools.transform_pvcontour` /
:func:`TransformTools.transform_dataset` apply ``transform.TransformPoint(p)``
directly to each input point. To move a moving-space landmark to its location in
the fixed image, the transform must map **moving-space coordinates to
fixed-space coordinates** -- the inverse of the image-warp transform.

So for the **same** moving-to-fixed registration result:

.. list-table:: Image registration: which transform to apply
   :header-rows: 1
   :widths: 50 25 25

   * - Goal
     - Transform
     - Helper
   * - Warp the **moving image** into fixed space (onto the fixed grid)
     - ``forward_transform``
     - :func:`TransformTools.transform_image`
   * - Warp **moving points / contours / landmarks** into fixed space
     - ``inverse_transform``
     - :func:`TransformTools.transform_pvcontour`
   * - Warp the **fixed image** into moving space (e.g. time-series reconstruction)
     - ``inverse_transform``
     - :func:`TransformTools.transform_image`
   * - Warp **fixed points / contours / landmarks** into moving space
     - ``forward_transform``
     - :func:`TransformTools.transform_pvcontour`

The first two rows are the everyday case (warping the registered moving data
into the fixed/reference frame): the **image uses** ``forward_transform``, the
**points use** ``inverse_transform``. The last two rows are the mirror image;
:meth:`monai_physio.RegisterTimeSeriesImages.reconstruct_time_series` is the
canonical consumer of ``inverse_transform`` for image warping (it resamples the
fixed image back onto each moving frame's grid).

.. note::

   All three image-registration backends (ANTS, ICON, Greedy) follow this same
   convention. ``transform_image(moving, forward_transform, fixed)`` is the
   correct call to warp the moving image onto the fixed grid for every backend.

PCA point transforms
====================

:class:`monai_physio.RegisterModelsPCA` builds ``forward_point_transform``
directly from the template-to-target point displacement, so
``forward_point_transform.TransformPoint(template_point)`` returns the
corresponding *target* point. As a **point** map it goes template (moving) to
target (fixed) -- which is the same orientation as image registration's
``inverse_transform``, and therefore the **opposite** orientation of image
registration's ``forward_transform``.

Concretely, treating the template as the moving object and the patient/target as
the fixed object:

.. list-table:: Same goal, opposite transform names across the two families
   :header-rows: 1
   :widths: 50 25 25

   * - Goal
     - Image registration
     - PCA model registration
   * - Warp the **image** (moving/template space -> fixed/target grid)
     - ``forward_transform``
     - ``inverse_point_transform``
   * - Warp **points / meshes** (moving/template -> fixed/target)
     - ``inverse_transform``
     - ``forward_point_transform``

In other words, ``forward_point_transform`` plays the role that
``inverse_transform`` plays for image registration, and
``inverse_point_transform`` plays the role of ``forward_transform``. Deforming
the template mesh onto the patient (the usual PCA use, performed internally by
``transform_template_model()`` and ``transform_point()``) uses
``forward_point_transform``; resampling an image with the PCA result uses
``inverse_point_transform``.

Rule of thumb
=============

* **Images pull back; points push forward.** For one registration result, the
  image and the points always use the two *different* members of the transform
  pair.
* **Image into the reference frame** -> ``forward_transform`` (image
  registration) / ``inverse_point_transform`` (PCA).
* **Points into the reference frame** -> ``inverse_transform`` (image
  registration) / ``forward_point_transform`` (PCA).
* When in doubt, warp a known landmark and a small image patch and confirm they
  land in the same place before trusting a pipeline.

See Also
========

* :doc:`registration_images`
* :doc:`registration_models`
* :doc:`utilities`
