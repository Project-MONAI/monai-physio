=========================
PhysicsNeMo AI Surrogates
=========================

MONAI Physio trains and runs PhysicsNeMo mesh-stage models: given a subject's
shape parameters and a stage (a point in the cardiac or respiratory cycle),
predict a per-vertex target on the shared template mesh. When that target is a
displacement, the prediction replaces a per-phase registration solve with one
forward pass — see Tutorials 9 through 13 in :doc:`../../tutorials`.

The layer follows the same has-a shape as the rest of the workflow tier: a
workflow owns the data and the artifacts, and a *method* object owns the
network.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Class
     - Role
   * - :class:`~monai_physio.WorkflowTrainPhysicsNeMo`
     - Manifests, normalization, lazy datasets, checkpoints and metadata
   * - :class:`~monai_physio.WorkflowInferPhysicsNeMo`
     - Loads a trained model and predicts raw per-point targets
   * - :class:`~monai_physio.WorkflowInferMovement`
     - Interprets 3-component targets as displacements: deformed meshes, mm
       error statistics, rasterized deformation fields, warped images and USD
   * - :class:`~monai_physio.WorkflowEvaluateMovement`
     - Scores those predictions per structure against the acquired frames:
       volume difference, Dice and surface RMSE
   * - :class:`~monai_physio.TrainPhysicsNeMoMGN` /
       :class:`~monai_physio.TrainPhysicsNeMoMLP`
     - The networks to train: MeshGraphNet or fully connected
   * - :class:`~monai_physio.InferPhysicsNeMoMGN` /
       :class:`~monai_physio.InferPhysicsNeMoMLP`
     - The matching networks at inference time
   * - :class:`~monai_physio.TrainPhysicsNeMoPhysicsInformedMotion`
     - A MeshGraphNet whose loss also prices the tissue's strain energy. It
       predicts displacement, as the others do; the stress that deformation
       implies comes from ``NeoHookeanResidual.cauchy_stress`` afterwards, as
       Tutorial 18 does it

PhysicsNeMo is an optional dependency::

   pip install "monai-physio[physicsnemo]"
   pip install torch-geometric          # MeshGraphNet only

It requires Python >= 3.11. ``import monai_physio`` works without it; the
imports happen lazily inside the methods that need them.

.. toctree::
   :maxdepth: 2

   manifest
   train
   infer
   evaluate
   physics_informed_motion
