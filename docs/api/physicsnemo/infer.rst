====================================
Predicting With a Mesh-Stage Model
====================================

.. module:: monai_physio.workflow_infer_physicsnemo
.. module:: monai_physio.workflow_infer_movement
.. module:: monai_physio.infer_physicsnemo_base
.. module:: monai_physio.infer_physicsnemo_mgn
.. module:: monai_physio.infer_physicsnemo_mlp
.. currentmodule:: monai_physio

Inference is split in two so that the generic half stays target-agnostic:

* :class:`WorkflowInferPhysicsNeMo` loads the checkpoint and returns the raw
  ``(n_points, n_target)`` prediction, whatever the target means.
* :class:`WorkflowInferMovement` wraps it and interprets three-component
  targets as displacements - deformed meshes, error statistics in millimetres,
  and rasterized deformation fields.

Generic prediction
==================

.. autoclass:: WorkflowInferPhysicsNeMo
   :members:
   :undoc-members:
   :show-inheritance:

Displacement interpretation
===========================

.. autoclass:: WorkflowInferMovement
   :members:
   :undoc-members:
   :show-inheritance:

Inference methods
=================

.. autoclass:: InferPhysicsNeMoBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: InferPhysicsNeMoMGN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: InferPhysicsNeMoMLP
   :members:
   :undoc-members:
   :show-inheritance:

Example
=======

.. code-block:: python

   from monai_physio import (
       InferPhysicsNeMoMGN,
       WorkflowInferMovement,
       WorkflowInferPhysicsNeMo,
   )

   infer = WorkflowInferPhysicsNeMo(
       model_directory=model_dir,
       inference_method=InferPhysicsNeMoMGN(),   # the default
   )

   # Raw targets, whatever the model was trained to predict.
   targets = infer.predict(pca_coefficients, stage=0.5)

   # Or, for a displacement model, geometry and mm error statistics.
   movement = WorkflowInferMovement(infer)
   result = movement.process(subject_manifest, output_directory=out_dir)

Notes
=====

**Where the displacements are applied.**
:meth:`WorkflowInferMovement.predict_single` and
:meth:`~WorkflowInferMovement.create_deformation_field` both require a
``fitted_reference_mesh`` - the patient's shape-model surface as fitted by
:class:`~monai_physio.WorkflowFitStatisticalModelToPatient`, which is shape
parameters *and* a deformable registration. The prediction stays in that mesh's
world frame, so a fit that carried a pose transform lands where the patient
actually is. A surface reconstructed from the shape parameters alone is not a
substitute and is not accepted.

**Arbitrary stages.** Nothing constrains ``stage`` to a phase that was
acquired; predicting between acquired phases is the reason to train a
surrogate at all.

**Deformation fields.** :meth:`~WorkflowInferMovement.create_deformation_field`
bins the per-vertex displacements and reference-surface normals onto a
caller-supplied image grid, giving an ITK vector image you can apply to
volumes and labelmaps with :class:`~monai_physio.TransformTools`.

See Also
========

* :doc:`manifest`
* :doc:`train`
* :doc:`../../cli_scripts/infer_physicsnemo`
