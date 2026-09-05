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

* :class:`monai_physio.SegmentChestTotalSegmentator`
* :class:`monai_physio.SegmentHeartSimpleware`
* :class:`monai_physio.SegmentAnatomyBase`

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

   from monai_physio.usd_anatomy_tools import DEFAULT_RENDER_PARAMS

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
