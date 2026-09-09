===============
NV-Segment-CTMR
===============

.. module:: monai_physio.segment_nv_segment_ct_mri
.. currentmodule:: monai_physio

``SegmentNVSegmentCTMRI`` runs NVIDIA's NV-Segment-CTMR model (a VISTA3D
derivative finetuned on 30K+ CT and MRI scans) and groups its 345-class
labelmap into the anatomy masks used by MONAI Physio workflows.

.. warning::

   The NV-Segment-CTMR *weights* are released under the NVIDIA OneWay
   Non-Commercial License (academic research use only); the surrounding bundle
   code is Apache 2.0. Use ``SegmentChestTotalSegmentator`` or NV-Segment-CT if
   you need a commercially licensed model.

Class Reference
===============

.. autoclass:: SegmentNVSegmentCTMRI
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
===========

.. code-block:: python

   import itk

   from monai_physio import SegmentNVSegmentCTMRI

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentNVSegmentCTMRI()

   masks = segmenter.segment(image)

   heart = masks["heart"]
   lungs = masks["lung"]
   labelmap = masks["labelmap"]

   itk.imwrite(labelmap, "labelmap.nrrd", compression=True)

For MR studies, select the matching modality before calling ``segment()``:

.. code-block:: python

   segmenter = SegmentNVSegmentCTMRI()
   segmenter.set_modality("MRI_BODY")   # or "CT_BODY", "MRI_BRAIN"

``MRI_BRAIN`` expects a T1 volume that has already been skull-stripped and
affinely aligned to the LUMIR template; this class does not perform that
preprocessing.

Returned Keys
=============

For this segmenter, ``segment()`` returns a dictionary with the following
keys:

* ``labelmap``
* ``heart``
* ``major_vessels``
* ``lung``
* ``bone``
* ``soft_tissue``
* ``brain_parcellation``
* ``other``

Label Ids
=========

Unlike the other segmenters, the labelmap is ``uint16``: label ids are the
model's own published class indices (see ``configs/label_dict.json`` in
https://github.com/NVIDIA-Medtech/NV-Segment-CTMR), which run to 345. For
example, 6 is the aorta and 115 the heart. The full group→id mapping is
available through the segmenter's ``taxonomy`` attribute
(``segmenter.taxonomy.labels_in_group("heart")``,
``segmenter.taxonomy.all_labels()``).

Rendering
=========

``brain_parcellation`` is a group name this segmenter introduces. Its
group-level entry in
:data:`monai_physio.usd_anatomy_tools.DEFAULT_RENDER_PARAMS` is a grey-matter
look, which is the right default because most of its labels are cortical gyri
or deep grey nuclei (caudate, putamen, thalamus, amygdala, hippocampus).

The brain tissues whose gross appearance genuinely differs from cortex carry
organ-level overrides, which win over the group entry on a substring match
(longest key first):

* ``white_matter`` - glossy creamy off-white myelin; also claims the
  cerebellar white matter.
* ``3rd_ventricle``, ``4th_ventricle``, ``lateral_ventricle``,
  ``inf_lat_vent`` - a shared clear-fluid CSF look. Four keys because a bare
  ``ventricle`` key would lose to the heart's ``ventricle_left`` /
  ``ventricle_right`` overrides.
* ``brain_stem`` - pale, fiber-tract dominated.
* ``cerebell`` - darker, browner, more matte cerebellar cortex; matches both
  ``cerebellum_exterior_*`` and ``cerebellar_vermal_lobules_*``.
* ``pallidum`` - myelin-rich, paler than the neighboring putamen and caudate.
* ``basal_forebrain`` - grey matter; present only to outrank the whole-organ
  ``brain`` override.

Note that ``white_matter_hyperintensity`` is in the ``soft_tissue`` group, not
``brain_parcellation``; its ``hyperintensity`` override keeps the lesion dull
and matte instead of inheriting the glossy white-matter look.

Operational Notes
=================

The first call to ``segment()`` downloads ~872 MB of model weights from
https://huggingface.co/nvidia/NV-Segment-CTMR into the Hugging Face cache
(override the destination with the ``model_cache_dir`` attribute). Inference
requires a CUDA GPU.

See Also
========

* :doc:`index`
* :doc:`totalsegmentator`
* :doc:`../../tutorials`
