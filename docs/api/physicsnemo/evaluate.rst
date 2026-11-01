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

Ground truth and cohorts
========================

.. autoclass:: MovementGroundTruth
   :exclude-members: labelmaps, reference_labelmap, reference_stage, meshes
   :show-inheritance:

.. autoclass:: EvaluateMovementBase
   :members:
   :exclude-members: segmenter_class, label_ids, report_dice,
                     evaluation_spacing_mm, smoothing_sigma_mm
   :show-inheritance:

.. autoclass:: EvaluateMovementLung
   :members:
   :show-inheritance:

.. autoclass:: EvaluateMovementDukeHeart
   :members:
   :show-inheritance:

.. autoclass:: ReportEvaluateMovement
   :members:
   :undoc-members:
   :show-inheritance:

Example
=======

.. code-block:: python

   from monai_physio import (
       MovementGroundTruth,
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
   ground_truth = MovementGroundTruth(
       labelmaps={0.0: frame_00, 0.1: frame_10},
       reference_labelmap=frame_00,
       reference_stage=0.0,
       meshes={0.0: fitted_mesh_00_file, 0.1: fitted_mesh_10_file},
   )
   result = evaluate.process(
       case_id="103_HM10395",
       shape_parameters=pca_coefficients_file,
       fitted_reference_mesh=fitted_reference_mesh_file,
       ground_truth=ground_truth,
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
