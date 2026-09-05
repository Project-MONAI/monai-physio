=========================
The Per-Subject Manifest
=========================

.. module:: monai_physio.physicsnemo_tools
.. currentmodule:: monai_physio.physicsnemo_tools

The manifest is the contract between your data and the training stack. It is
the only thing you must produce to train on your own subjects: one JSON file
per subject, naming a fitted reference mesh, that subject's PCA shape parameters, the
point-data array holding the training targets, and one entry per phase.

.. code-block:: json

   {
     "subject_id": "Case1Pack",
     "fitted_reference_mesh": "Case1Pack_ssm_surface.vtp",
     "pca_coefficients": "Case1Pack_ssm_pca_coefficients.json",
     "target_array": "displacement",
     "phases": [
       {"mesh": "Case1Pack_T00_ssm_surface_target.vtp", "stage": 0.0},
       {"mesh": "Case1Pack_T50_ssm_surface_target.vtp", "stage": 0.5}
     ]
   }

Relative paths resolve against the manifest's own directory.

Targets are read verbatim
=========================

The stack never derives targets from geometry. Whatever array
``target_array`` names is what the network learns to predict, and its width
sets the network's output size - three columns for a displacement, one for a
scalar field, any number for something else. Tutorial 9 writes
``phase.points - reference.points`` into that array, which is what makes its
model a motion model; write something else and the same code trains on it.

Every phase mesh must share the template's point count and ordering, and
``stage`` is the caller's own normalization of where the phase sits in the
cycle - the workflow never parses filenames.

Meshes may be surfaces (``.vtp``) or volumes (``.vtu``); the template mesh
decides which domain the model lives on.

Reference
=========

These live in :mod:`monai_physio.physicsnemo_tools`, which is not re-exported
from the top-level package - import it by module:

.. code-block:: python

   from monai_physio.physicsnemo_tools import SubjectManifest, parse_manifest

.. autoclass:: SubjectManifest
   :exclude-members: subject_id, fitted_reference_mesh, pca_coefficients, target_array, phases

.. autoclass:: PhaseEntry
   :exclude-members: mesh, stage

.. autofunction:: parse_manifest

.. autofunction:: load_target_array

.. autofunction:: load_pca_coefficients

Supporting helpers
==================

.. autofunction:: build_node_features

.. autofunction:: mesh_to_edge_index

.. autofunction:: compute_edge_features

.. autoclass:: PhaseSampleDataset
   :members:
   :undoc-members:

See Also
========

* :doc:`train`
* :doc:`infer`
* :doc:`../../tutorials`
