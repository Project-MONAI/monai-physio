=============
NV-Segment-CT
=============

.. module:: monai_physio.segment_nv_segment_ct
.. currentmodule:: monai_physio

``SegmentNVSegmentCT`` runs NVIDIA's NV-Segment-CT model (a VISTA3D
derivative finetuned on CT scans) and groups its labelmap into the anatomy
masks used by MONAI Physio workflows. It covers 117 "segment everything"
classes (132 addressable via label prompt) and supports only automatic
(label-prompt) segmentation - there is no point-click interactive branch
exposed here.

.. warning::

   The NV-Segment-CT *weights* are released under the NVIDIA Open Model
   License Agreement (research use only, not for clinical use); the
   surrounding bundle code is Apache 2.0. NV-Segment-CTMR's weights carry a
   different restrictive license (NVIDIA OneWay Non-Commercial License) -
   both models are research-use-only, just under different named terms, so
   check the license text for your use case. Use
   ``SegmentChestTotalSegmentator`` if you need no license restriction at
   all - its default task set is unrestricted; only its optional
   ``heartchambers_highres`` and ``tissue_4_types`` tasks
   (``set_has_academic_license(True)``) require a TotalSegmentator academic
   license.

Class Reference
===============

.. autoclass:: SegmentNVSegmentCT
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
===========

.. code-block:: python

   import itk

   from monai_physio import SegmentNVSegmentCT

   image = itk.imread("chest_ct.nrrd")
   segmenter = SegmentNVSegmentCT()

   masks = segmenter.segment(image)

   heart = masks["heart"]
   lungs = masks["lung"]
   labelmap = masks["labelmap"]

   itk.imwrite(labelmap, "labelmap.nrrd", compression=True)

NV-Segment-CT is CT-only: unlike ``SegmentNVSegmentCTMRI`` there is no
``set_modality()`` call or MRI code path.

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
* ``other``

Label Ids
=========

Label ids are the model's own published class indices, used verbatim (see
``NV-Segment-CT/configs/label_dict.json`` in
https://github.com/NVIDIA-Medtech/NV-Segment-CTMR), which run to 132. For
example, 6 is the aorta and 115 the heart. The full group->id mapping is
available through the segmenter's ``taxonomy`` attribute
(``segmenter.taxonomy.labels_in_group("heart")``,
``segmenter.taxonomy.all_labels()``).

Operational Notes
=================

The first call to ``segment()`` downloads ~872 MB of model weights from
https://huggingface.co/nvidia/NV-Segment-CT into the Hugging Face cache
(override the destination with the ``model_cache_dir`` attribute). Inference
requires a CUDA GPU.

See Also
========

* :doc:`index`
* :doc:`nv_segment_ct_mri`
* :doc:`totalsegmentator`
* :doc:`../../tutorials`
