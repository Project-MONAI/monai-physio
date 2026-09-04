================================
Predict With a Trained Surrogate
================================

``monai-physio-infer-physicsnemo`` loads a model directory written by
``monai-physio-train-physicsnemo`` and predicts a subject's per-vertex targets
at any stage — including stages that were never acquired, which is the reason
to train a surrogate. It is the command-line form of
:class:`~monai_physio.WorkflowInferPhysicsNeMo` and
:class:`~monai_physio.WorkflowInferMovement`.

Requires the same optional extra as training.

Manifest Mode
=============

Predict every phase in a manifest and score against its stored targets:

.. code-block:: bash

   monai-physio-infer-physicsnemo \
       --model-dir output/mgn_run \
       --manifest manifests/Case1Pack_manifest.json \
       --output output/mgn_run/eval

Add ``--displacement`` when the targets are displacements from the subject's
reference mesh: the command then writes ``reference + prediction`` meshes, a
reference surface colored by per-point RMSE, and error statistics in
millimetres, instead of the raw target arrays.

Pass ``--stages 0.15 0.35`` to predict arbitrary stages instead of the
manifest's phases; no ground truth exists for those, so no statistics are
written.

Single-Subject Mode
===================

No manifest — just the subject's PCA coefficients:

.. code-block:: bash

   monai-physio-infer-physicsnemo \
       --model-dir output/mgn_run \
       --shape-parameters Case1Pack_ssm_pca_coefficients.json \
       --stage 0.7 \
       --fitted-reference-mesh Case1Pack_ssm_surface.vtp \
       --output output/prediction

``--fitted-reference-mesh`` is required in single-subject mode: it is the
patient's fitted shape-model surface, written by
``monai-physio-fit-statistical-model-to-patient``.

Deformation Fields
==================

With ``--reference-image``, the command rasterizes the predicted displacements
and the reference-surface normals onto that image's voxel grid:

.. code-block:: bash

   monai-physio-infer-physicsnemo \
       --model-dir output/mgn_run \
       --shape-parameters coefficients.json \
       --stage 0.5 \
       --fitted-reference-mesh patient_surface.vtp \
       --reference-image patient_ct.mha \
       --output output/fields

This writes ``deformation_field.mha`` and ``surface_normal_field.mha`` —
apply them to volumes and labelmaps with
:class:`~monai_physio.TransformTools`.

Options
=======

``--model-dir PATH``
   Required. The trained model directory.

``--network {mgn,mlp,auto}``
   Auto-detected from the checkpoint present in ``--model-dir`` by default.

``--epoch N``
   Load a periodic epoch checkpoint instead of the final weights.

``--manifest JSON``, ``--stages [FLOAT ...]``, ``--displacement``
   Manifest mode, as above.

``--shape-parameters JSON``, ``--stage FLOAT``, ``--fitted-reference-mesh PATH``, ``--reference-image PATH``
   Single-subject mode, as above.

``--output PATH``
   Output directory; defaults to a subdirectory of the model directory.

See Also
========

* :doc:`train_physicsnemo`
* :doc:`../api/physicsnemo/infer`
* :doc:`../tutorials`
