.. _cookbook:

=====================
MONAI Physio Cookbook
=====================

Short, self-contained recipes for the things people actually ask for. Each one
lists its **ingredients** — what you must have on hand before you start — then
the **steps** that turn them into a result.

Recipes assume you already installed the package (:doc:`/installation`) and
know roughly what the pipeline does (:doc:`/architecture`). Where a recipe
needs more depth than a step can carry, it links to the reference page instead
of repeating it.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Recipe
     - Makes
   * - :doc:`train_and_infer_on_your_own_data`
     - A motion surrogate trained on your cohort, predicting unseen stages
   * - :doc:`add_a_segmentation_method`
     - A new ``SegmentAnatomyBase`` backend, usable from workflows and CLIs
   * - :doc:`add_a_registration_method`
     - A new ``RegisterImagesBase`` backend, usable anywhere a registrar is

.. toctree::
   :maxdepth: 1
   :hidden:

   train_and_infer_on_your_own_data
   add_a_segmentation_method
   add_a_registration_method
