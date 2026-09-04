==============================
Train a PhysicsNeMo Surrogate
==============================

``monai-physio-train-physicsnemo`` trains a mesh-stage model — given a
subject's PCA shape parameters and a stage, predict a per-vertex target on the
shared template mesh. It is the command-line form of
:class:`~monai_physio.WorkflowTrainPhysicsNeMo`; Tutorial 9 in
:doc:`../tutorials` is the same thing as a script.

Requires the optional extra::

   pip install "monai-physio[physicsnemo]"
   pip install torch-geometric          # MeshGraphNet only

PhysicsNeMo needs Python >= 3.11.

Basic Usage
===========

.. code-block:: bash

   monai-physio-train-physicsnemo \
       --network mgn \
       --train-manifest manifests/Case2Pack_manifest.json manifests/Case3Pack_manifest.json \
       --val-manifest manifests/Case4Pack_manifest.json \
       --pca-mean-mesh output/tutorial_06_lung/pca_mean_surface.vtp \
       --output output/mgn_run

Options
=======

``--network {mgn,mlp}``
   Required. ``mgn`` trains a MeshGraphNet, which passes messages along mesh
   edges and suits a continuum whose neighbouring vertices co-vary; ``mlp``
   trains a fully connected network that treats each vertex independently.

``--train-manifest JSON [JSON ...]``
   Required. One per-subject manifest per training subject. See
   :doc:`../api/physicsnemo/manifest` for the schema — producing these is the
   only work needed to train on your own data.

``--val-manifest JSON [JSON ...]``
   Validation subjects, used for the intermittent RMSE report. May be omitted.

``--pca-mean-mesh PATH``
   The PCA template mesh. Its points define the node domain and, for the MGN,
   the graph topology, so a ``.vtu`` trains on volume points and a ``.vtp`` on
   surface points. A sibling ``pca_model.json`` is copied into the output
   directory so the trained model stays self-contained.

``--output PATH``
   Directory for checkpoints, metadata and logs.

``--resume-from PATH``
   A prior ``<tag>_stage_model.pt``. Its normalization statistics are inherited
   so the loaded weights stay valid, and training writes to a fresh numbered
   sibling of ``--output``.

Shared tuning
   ``--epochs``, ``--batch-size`` (in samples), ``--learning-rate``,
   ``--num-layers``, and ``--cache-size`` (decoded target arrays held in RAM;
   ``0`` is unbounded).

MGN-specific
   ``--processor-size`` (message-passing hops), ``--hidden-dim``.

MLP-specific
   ``--layer-size``.

Output
======

.. code-block:: text

   <output>/
   ├── mgn_stage_model.pt                 # weights + normalization stats
   ├── mgn_stage_model_metadata.json      # features, target name and width
   ├── mgn_stage_model_epoch_00100.pt     # periodic, resumable checkpoints
   ├── training_losses.json
   ├── training_validation_rmse.{json,csv}
   ├── pca_mean_template.vtp              # the node domain, for inference
   ├── pca_model.json                     # when found beside the template
   └── shared_edge_{index,features}.pt    # MGN graph tensors

The checkpoint records the target width, so
``monai-physio-infer-physicsnemo`` rebuilds a matching network without being
told.

See Also
========

* :doc:`infer_physicsnemo`
* :doc:`../api/physicsnemo/index`
* :doc:`../tutorials`
