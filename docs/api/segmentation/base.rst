========================
Segmentation Base Class
========================

.. module:: monai_physio.segment_anatomy_base
.. currentmodule:: monai_physio

``SegmentAnatomyBase`` defines the shared chest-anatomy segmentation contract
used by MONAI Physio segmentation implementations. It owns an
:class:`AnatomyTaxonomy` instance that subclasses populate to declare which
anatomy groups (and which organ labels within each group) they produce.

Class Reference
===============

.. autoclass:: SegmentAnatomyBase
   :members:
   :undoc-members:
   :show-inheritance:

Segmentation Contract
=====================

Concrete segmenters accept an ITK image and return a dictionary of ITK images:

.. code-block:: python

   import itk

   from monai_physio import SegmentChestTotalSegmentator

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentChestTotalSegmentator()
   masks = segmenter.segment(image)

   labelmap = masks["labelmap"]
   if "heart" in masks:
       heart = masks["heart"]

The returned dictionary always contains ``"labelmap"`` plus one entry per
anatomy group the segmenter registered in its taxonomy (and ``"other"`` for
unclassified labels). **The exact key set is segmenter-specific** — callers
must check membership (``"lung" in masks``) rather than assume a fixed
schema. For example, :class:`SegmentChestTotalSegmentator` returns the full
``heart, lung, bone, major_vessels, soft_tissue, contrast, other`` set,
while :class:`SegmentHeartSimpleware` returns only the groups its
ASCardio module actually populates (``heart``, ``major_vessels``,
``soft_tissue``, ``contrast``, ``other``).

Anatomy Taxonomy
================

The group-to-organ mapping is held by :class:`AnatomyTaxonomy`, a small
data class shared between the segmenter and downstream renderers
(:class:`USDAnatomyTools`, :class:`ConvertVTKToUSD`). It is independent of
ITK and OpenUSD so segmentation code can be reasoned about without pulling
in the rendering stack.

.. autoclass:: AnatomyGroup
   :no-members:

.. autoclass:: AnatomyTaxonomy
   :members:

Typical usage from a subclass ``__init__``:

.. code-block:: python

   class SegmentMySite(SegmentAnatomyBase):
       def __init__(self):
           super().__init__()
           self.taxonomy.add_organ("heart", 51, "myocardium")
           self.taxonomy.add_organ("heart", 61, "atrial_appendage_left")
           self.taxonomy.add_organ("lung", 10, "lung_upper_lobe_left")
           # ...
           self._finalize_other_group()

Downstream callers can introspect what a segmenter produces without
running it:

.. code-block:: python

   tax = segmenter.taxonomy
   print(tax.group_names())                       # ['heart', 'lung', ...]
   print(tax.labels_in_group("heart"))            # {51: 'myocardium', ...}
   print(tax.group_for_label("myocardium"))       # 'heart'
   print(tax.all_labels())                        # full id -> name dict

Extending Segmentation
======================

New runtime segmentation classes should:

1. Inherit from :class:`SegmentAnatomyBase` (or another :class:`MONAIPhysioBase`
   subclass if no anatomy taxonomy is needed).
2. Populate ``self.taxonomy`` with ``add_organ`` calls in ``__init__``.
3. Call ``self._finalize_other_group()`` once all groups have been registered.
4. Use ``log_info()`` / ``log_debug()`` instead of ``print``.
5. Document the key set the segmenter produces; downstream callers should
   check membership rather than assume a fixed schema.

Keep synthetic tests small. Real-data tests pull data through the session
fixtures (downloaded on first use); mark GPU- or Simpleware-bound tests with
``requires_gpu`` / ``requires_simpleware`` so they fall into the right
opt-in bucket.

See Also
========

* :doc:`totalsegmentator`
* :doc:`simpleware`
* :doc:`index`
* :doc:`../../developer/segmentation`
* :doc:`../usd/anatomy_tools` for the renderer side of the taxonomy
