.. _cookbook_new_segmenter:

=============================
Adding a Segmentation Method
=============================

Wire a new segmentation backend into MONAI Physio so every workflow and CLI
that takes a ``segmentation_method`` can use it.

Ingredients
===========

* **A model or algorithm** that turns one 3D image into an integer labelmap.
* **A label map of your own**: which integer id means which organ, and which
  anatomy group each organ belongs to.
* **A module** at ``src/monai_physio/segment_<name>.py``.

Steps
=====

**1. Subclass** :class:`~monai_physio.SegmentAnatomyBase`. The base owns
preprocessing, postprocessing, contrast fusion, and anatomy-group splitting;
you supply only the model call.

.. code-block:: python

   import logging

   import itk

   from monai_physio import SegmentAnatomyBase


   class SegmentBrainMyModel(SegmentAnatomyBase):
       def __init__(self, log_level: int | str = logging.INFO) -> None:
           super().__init__(log_level=log_level)
           self.target_spacing = 1.5

       def segmentation_method(self, preprocessed_image: itk.Image) -> itk.Image:
           return run_my_model(preprocessed_image)

``segmentation_method()`` is the one required override. Return an
``itk.Image`` labelmap on the preprocessed grid — the base resamples it back.

**2. Declare the taxonomy** in ``__init__``, then finalize. Group names are
free-form; new ones are allowed.

.. code-block:: python

   for group_name, organs in (
       ("brain", {1: "cerebrum", 2: "cerebellum"}),
       ("bone", {3: "skull"}),
   ):
       for label_id, organ_name in organs.items():
           self.taxonomy.add_organ(group_name, label_id, organ_name)

   self._finalize_other_group()

The base contributes ``contrast`` (id 135) and ``soft_tissue`` (id 133)
already. ``_finalize_other_group()`` claims the remaining ids in ``[1, 256)``
for ``"other"``.

**3. Register a look** if you introduced a group outside the default chest set
(``heart``, ``lung``, ``bone``, ``major_vessels``, ``contrast``,
``soft_tissue``, ``other``). Without one, USD export falls back to the generic
``"other"`` material.

.. code-block:: python

   from monai_physio.usd_anatomy_tools import DEFAULT_RENDER_PARAMS

   DEFAULT_RENDER_PARAMS["brain"] = {
       "name": "Brain",
       "diffuse_reflection_color": (0.85, 0.75, 0.7),
       # ... copy the parameter list from an existing entry ...
   }

**4. Export it** from ``src/monai_physio/__init__.py``, next to the other
``segment_*`` imports, and add it to ``__all__``.

**5. Add it to the CLI dispatch** in
``src/monai_physio/cli/_method_factories.py`` — the single place strings become
instances. Append the name to ``SEGMENTATION_METHODS`` and a branch to
``build_segmentation_method()``. Every ``--segmentation-method`` flag picks it
up from there.

**6. Test it.** Copy the shape of
``tests/test_segment_chest_total_segmentator.py``. Keep inputs synthetic where
possible; for real data use the session fixtures (``test_directories``,
``download_test_data``, ``test_images``). Mark GPU- or license-bound tests
``requires_gpu`` / ``requires_simpleware``.

.. code-block:: bash

   py -m pytest tests/test_segment_brain_my_model.py -v

**7. Document it.** Add an ``docs/api/segmentation/<name>.rst`` page, list it in
that directory's ``index.rst``, and add the class to the implemented-segmenters
list in :doc:`/developer/segmentation`. Then run ``graphify update .``.

Notes
=====

* Output is a dict of ITK images: ``"labelmap"`` plus one entry per anatomy
  group, keyed by group name, original label ids preserved. Callers should
  check key membership, not assume a fixed schema.
* Use ``self.log_info()`` / ``self.log_debug()``; never ``print()``.
* Set ``self.target_spacing`` to whatever resolution your model was trained at.

See Also
========

* :doc:`/developer/segmentation` — the extended guide
* :doc:`/api/segmentation/base` — ``AnatomyTaxonomy`` reference
* :doc:`/developer/usd_generation` — how the taxonomy drives USD materials
* :doc:`add_a_registration_method`
