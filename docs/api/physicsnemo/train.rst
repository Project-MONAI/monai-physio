=============================
Training a Mesh-Stage Model
=============================

.. module:: monai_physio.workflow_train_physicsnemo
.. module:: monai_physio.train_physicsnemo_base
.. module:: monai_physio.train_physicsnemo_mgn
.. module:: monai_physio.train_physicsnemo_mlp
.. currentmodule:: monai_physio

:class:`WorkflowTrainPhysicsNeMo` owns the data side of training — manifests,
normalization statistics, lazy datasets, output directories, checkpoints,
metadata and logs. The network and its optimization loop live in the training
method it drives.

Workflow
========

.. autoclass:: WorkflowTrainPhysicsNeMo
   :members:
   :undoc-members:
   :show-inheritance:

Training methods
================

.. autoclass:: TrainPhysicsNeMoBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TrainPhysicsNeMoMGN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TrainPhysicsNeMoMLP
   :members:
   :undoc-members:
   :show-inheritance:

Example
=======

.. code-block:: python

   from monai_physio import TrainPhysicsNeMoMGN, WorkflowTrainPhysicsNeMo

   method = TrainPhysicsNeMoMGN()
   method.set_epochs(1500)
   method.set_batch_size(4)
   method.set_processor_size(3)      # message-passing hops
   method.set_hidden_dim(128)

   workflow = WorkflowTrainPhysicsNeMo(
       train_manifests=train_manifests,
       val_manifests=val_manifests,
       pca_mean_mesh=pca_mean_surface_file,
       output_directory=output_dir,
       training_method=method,
   )
   result = workflow.process()

   checkpoint = result["checkpoint"]

Swap ``TrainPhysicsNeMoMGN`` for :class:`TrainPhysicsNeMoMLP` to train the
fully connected network instead; nothing else changes.

Notes
=====

**The template mesh sets the node domain.** ``pca_mean_mesh`` defines the
points every subject is expressed on, and — for the MeshGraphNet — the graph
topology. Pass a ``.vtu`` to train on volume points, a ``.vtp`` to train on
surface points. When the PCA model is volumetric but your manifests reference
surfaces, set ``use_template_surface=True`` and the workflow trains on the
template's extracted surface.

**Target width is inferred.** The stored target array's column count becomes
the network's output size and is recorded in the checkpoint, so inference
rebuilds a matching network without being told.

**Resuming writes to a fresh directory.** Passing ``resume_from`` inherits the
prior run's normalization statistics — so the loaded weights stay valid — and
writes to a numbered sibling of ``output_directory``. Read the actual location
from ``result["output_directory"]``.

**Streaming.** ``set_cache_size()`` bounds how many decoded target arrays stay
in RAM, so an arbitrarily large training set streams from disk.

See Also
========

* :doc:`manifest`
* :doc:`infer`
* :doc:`../../cli_scripts/train_physicsnemo`
