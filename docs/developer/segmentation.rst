============================
Segmentation Developer Guide
============================

Segmentation classes convert CT images into anatomy-group masks used by
workflows.

Current Segmentation Contract
=============================

.. code-block:: python

   import itk

   from monai_physio import SegmentChestTotalSegmentator

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentChestTotalSegmentator()
   masks = segmenter.segment(image)

   labelmap = masks["labelmap"]
   if "heart" in masks:
       heart = masks["heart"]

Segmentation outputs are dictionaries of ITK images. Access masks by key,
not by positional unpacking. The exact key set depends on the segmenter's
:class:`monai_physio.AnatomyTaxonomy` - see :doc:`../api/segmentation/base`
for the per-segmenter key sets and the general contract.

Implemented Segmenters
======================

Intensity pixel types
---------------------

Shared preprocessing accepts scalar 3-D images, including in-memory int32 CTs.
Before resampling, ``ProcessImages.convert_intensity_image(image)`` preserves
uint8, int16, uint16, float32, and float64 images and converts other integers
to float64 without clipping CT intensities. Integers outside the exact float64
range are rejected. ``make_isotropic_image`` uses the same preparation.

For a backend that explicitly requires signed short, request the lossy conversion::

   import numpy as np
   from monai_physio import ProcessImages

   image = ProcessImages().convert_intensity_image(image, dtype=np.int16)

This rounds ties to even, clips to the int16 range, rejects nonfinite values,
and preserves the image grid. Do not apply it to labelmaps.

Backend classes
---------------

* :class:`monai_physio.SegmentChestTotalSegmentator`
* :class:`monai_physio.SegmentHeartSimpleware`
* :class:`monai_physio.SegmentAnatomyBase`

Releasing cached model resources
================================

NV-Segment-CT and NV-Segment-CTMR retain a pipeline for reuse across patients.
After the last segmentation, the owner may explicitly call::

   outputs = segmenter.segment(image)
   segmenter.release_resources(empty_cuda_cache=True)

``release_resources()`` drops these backends' cached pipelines without changing
their configuration, returned labelmaps, or downloaded weights. The next
segmentation rebuilds the pipeline lazily. Other segmenters inherit a no-op
default. Workflows never release a caller-owned segmenter automatically. Call
only while the segmenter is idle, not concurrently with segmentation.

The optional ``empty_cuda_cache=True`` also collects unreachable objects and
releases unused PyTorch allocator blocks. It affects the process-wide CUDA cache
and can make later allocations slower; the default False avoids that side
effect. Live tensors held by the caller or another library remain allocated.

For memory measurements, separate process RSS, PyTorch allocated bytes and
reserved bytes before/after segmentation, after dropping owned resources, and
after cache release. Synchronize CUDA before recording device measurements.
Also measure fitting and a complete respiratory cycle on the original image
grid. Python object release and a lower allocated-byte count do not guarantee
that ITK, VTK or a system allocator returns memory to the operating system.
Applications may still benefit from a subprocess boundary.

Adding a New Segmenter
======================

A new segmenter subclass declares which anatomy groups and organ labels it
produces by populating ``self.taxonomy``. The base class owns the
:class:`AnatomyTaxonomy` instance and contributes two default placeholders
(``contrast`` at id 135, ``soft_tissue`` at id 133).

.. code-block:: python

   import logging

   from monai_physio import SegmentAnatomyBase


   class SegmentMySite(SegmentAnatomyBase):
       def __init__(self, log_level=logging.INFO):
           super().__init__(log_level=log_level)
           self.target_spacing = 1.5

           # Register organ labels under group names. New group names
           # can be introduced freely; downstream renderers fall back
           # to render_params["other"] for unregistered groups.
           for group_name, organs in (
               ("heart", {51: "heart", 61: "atrial_appendage_left"}),
               ("lung", {10: "lung_upper_lobe_left"}),
           ):
               for label_id, organ_name in organs.items():
                   self.taxonomy.add_organ(group_name, label_id, organ_name)

           # Fill in 'other' for any unclaimed ids in the [1, 256) range.
           self._finalize_other_group()

       def segmentation_method(self, preprocessed_image):
           # Run the actual model here; must return an itk.Image labelmap.
           ...

The base class then provides the shared pre/post-processing, contrast-agent
fusion, and ``create_anatomy_group_labelmaps()`` that walks the taxonomy and
emits one labelmap (preserving the original label ids) per registered group
plus ``"other"``.

Custom Anatomy Looks
====================

If the new segmenter introduces groups beyond the default chest set
(``heart``, ``lung``, ``bone``, ``major_vessels``, ``contrast``,
``soft_tissue``, ``other``), register a matching OmniSurface look so the
USD renderer doesn't fall back to the generic ``"other"`` material:

.. code-block:: python

   from monai_physio.process_usd_anatomy import DEFAULT_RENDER_PARAMS

   DEFAULT_RENDER_PARAMS["brain"] = {
       "name": "Brain",
       "diffuse_reflection_color": (0.85, 0.75, 0.7),
       # ... see existing entries for the full parameter list ...
   }

See :doc:`usd_generation` for the renderer-side contract.

Development Notes
=================

* Use ``log_info()`` and ``log_debug()`` inside runtime classes; never ``print``.
* Document the key set the segmenter produces; downstream callers should
  check membership rather than assume a fixed schema.
* Keep tests synthetic unless real model/data behavior is being validated.
* Real-data tests use the session fixtures (data downloads automatically);
  mark GPU- or Simpleware-bound tests with ``requires_gpu`` /
  ``requires_simpleware`` so they fall into the right opt-in bucket.

See Also
========

* :doc:`../api/segmentation/index`
* :doc:`../api/segmentation/base` - full AnatomyTaxonomy reference
* :doc:`workflows`
* :doc:`usd_generation` - how the taxonomy drives USD output
