.. _cookbook_own_data:

==========================================
Training and Inference on Your Own Data
==========================================

Train a motion surrogate on your own cohort, then predict phases you never
acquired. Tutorials 4 through 12 are this recipe on public data.

Ingredients
===========

* **A cohort.** Several subjects, each with a gated series covering the cycle.
  Any ITK-readable input: a DICOM directory, ``.mha``, ``.nrrd``, ``.nii.gz``,
  or a 4D file.
* **One held-out subject**, kept out of training, for scoring.
* **A segmentation backend** that covers your anatomy — see
  :doc:`add_a_segmentation_method` if none does.
* **The optional extra**, plus a CUDA GPU::

     pip install "physiotwin4d[physicsnemo]"
     pip install torch-geometric      # MeshGraphNet only

  PhysicsNeMo requires Python >= 3.11.

Steps
=====

**1. Split 4D into a 3D time series.** Skip if your data is already per-phase
files.

.. code-block:: bash

   physiotwin4d-convert-image-4d-to-3d \
       --input-image subject_4d.seq.nrrd \
       --output-dir work/sub01/phases \
       --basename phase

**2. Segment each phase to surfaces.** Repeat per phase, per subject.

.. code-block:: bash

   physiotwin4d-convert-image-to-vtk \
       --input-image work/sub01/phases/phase_000.mha \
       --output-dir work/sub01/surfaces \
       --segmentation-method ChestTotalSegmentator \
       --anatomy-groups heart

**3. Build one PCA shape model for the cohort.** Every subject's fitted mesh
inherits this template's point count and ordering, which is what makes vertices
comparable across subjects.

.. code-block:: bash

   physiotwin4d-create-statistical-model \
       --sample-meshes-dir work/reference_surfaces \
       --reference-mesh work/reference_surfaces/sub01_heart.vtp \
       --output-dir work/ssm \
       --number-of-pca-components 20

**4. Fit the model to each subject, at every phase.** Fit the reference phase
first, then carry that fitted mesh through the remaining phases. This yields
the two per-subject artifacts training needs: a fitted reference surface and
its PCA coefficient JSON.

.. code-block:: bash

   physiotwin4d-fit-statistical-model-to-patient \
       --template-model work/ssm/pca_mean_surface.vtp \
       --pca-json work/ssm/pca_model.json \
       --patient-models work/sub01/surfaces/phase_000_heart.vtp \
       --patient-image work/sub01/phases/phase_000.mha \
       --output-dir work/sub01/fit \
       --output-prefix sub01_T00

**5. Write one manifest per subject.** This is the only per-subject artifact
the training stack requires, and the only place your data meets it. Name the reference mesh,
the PCA coefficients, the point-data array holding your targets, and one entry
per phase with its normalized ``stage``. Targets are read verbatim — write
``phase.points - reference.points`` for a motion model, or any other per-vertex
quantity for something else.

.. code-block:: json

   {
     "subject_id": "sub01",
     "fitted_reference_mesh": "sub01_ssm_surface.vtp",
     "pca_coefficients": "sub01_ssm_pca_coefficients.json",
     "target_array": "displacement",
     "phases": [
       {"mesh": "sub01_T00_target.vtp", "stage": 0.0},
       {"mesh": "sub01_T50_target.vtp", "stage": 0.5}
     ]
   }

See :doc:`/api/physicsnemo/manifest` for the full schema and its rules.
``tutorials/tutorial_09_lung_train_physicsnemo_mgn.py`` has a working writer.

**6. Train.** Hold your test subject back; list the rest.

.. code-block:: bash

   physiotwin4d-train-physicsnemo \
       --network mgn \
       --train-manifest work/manifests/sub0{2,3,4}_manifest.json \
       --val-manifest work/manifests/sub05_manifest.json \
       --pca-mean-mesh work/ssm/pca_mean_surface.vtp \
       --output work/mgn_run

**7. Predict.** Manifest mode predicts the stages the manifest stores;
``--stages`` asks for phases that were never acquired. Neither scores the
result --- for that, see :doc:`/tutorials` Tutorial 11.

.. code-block:: bash

   physiotwin4d-infer-physicsnemo \
       --model-dir work/mgn_run \
       --manifest work/manifests/sub01_manifest.json \
       --displacement \
       --output work/mgn_run/eval

   physiotwin4d-infer-physicsnemo \
       --model-dir work/mgn_run \
       --shape-parameters work/sub01/fit/sub01_ssm_pca_coefficients.json \
       --stage 0.35 \
       --fitted-reference-mesh work/sub01/fit/sub01_ssm_surface.vtp \
       --output work/prediction

**8. Score against the images.** Pass the inference workflow to
:class:`~physiotwin4d.WorkflowEvaluateMovement` with the structures to score,
which reports per-structure surface and volume error against the acquired
frames.

Notes
=====

* Every phase mesh must share the template's point count and ordering. A
  mismatch here is the most common training failure.
* ``stage`` is your own normalization of position in the cycle. Nothing parses
  filenames for it.
* ``.vtp`` trains on surface points, ``.vtu`` on volume points — the template
  mesh decides.
* Add ``--reference-image`` at inference to rasterize predictions into
  ``deformation_field.mha`` for warping volumes and labelmaps.

See Also
========

* :doc:`/api/physicsnemo/manifest`
* :doc:`/cli_scripts/train_physicsnemo`
* :doc:`/cli_scripts/infer_physicsnemo`
* :doc:`/cli_scripts/byod_tutorials` — your own data straight to USD, no training
* :doc:`/tutorials`
