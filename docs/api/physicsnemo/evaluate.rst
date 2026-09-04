==========================================
Scoring a Mesh-Stage Model Against Images
==========================================

.. module:: monai_physio.workflow_evaluate_movement
.. currentmodule:: monai_physio

A mm error against the surfaces a registration produced says how well the
network reproduces that registration. :class:`WorkflowEvaluateMovement` asks the
other question: how close are the size and shape of the inferred anatomy to the
anatomy that was actually imaged, structure by structure. See Tutorial 11 in
:doc:`../../tutorials`.

For every gated time point it carries the reference frame's labelmap into that
time point with the network's own deformation, and compares the result to the
labelmap of the frame that was acquired: volume difference, Dice and surface
RMSE per lung lobe or per heart chamber.

Per-structure scoring
=====================

.. autoclass:: WorkflowEvaluateMovement
   :members:
   :undoc-members:
   :show-inheritance:

Example
=======

.. code-block:: python

   from monai_physio import (
       WorkflowEvaluateMovement,
       WorkflowInferMovement,
       WorkflowInferPhysicsNeMo,
   )

   evaluate = WorkflowEvaluateMovement(
       movement_workflow=WorkflowInferMovement(
           WorkflowInferPhysicsNeMo(model_directory=model_dir)
       ),
       label_names={28: "lung_upper_lobe_left", 29: "lung_lower_lobe_left"},
   )
   result = evaluate.process(
       case_id="Case1Pack",
       shape_parameters=pca_coefficients_file,
       fitted_reference_mesh=fitted_reference_mesh_file,
       reference_labelmap=reference_labelmap,
       ground_truth_labelmaps={0.0: frame_00, 0.1: frame_10},
       output_directory=out_dir,
   )
   print(result["report_file"], result["csv_file"])

Notes
=====

**Why labelmaps rather than the model's surface.** The lung shape model carries
its five lobes as per-cell labels, but the heart model is a single structure ---
the whole heart minus its chamber cavities --- so its chambers exist only in the
acquired labelmaps. Warping those labelmaps scores every structure the
acquisition contains, whether or not the shape model represents it separately.

**The evaluation grid.** Everything is measured on one isotropic grid built
around the reference anatomy, so a case whose gated frames carry different slice
pitches is still scored on a single, stated voxel volume. Its pitch sets both
that voxel volume and the memory the per-stage deformation fields take, which
grows with its cube.

See Also
========

* :doc:`infer`
* :doc:`train`
