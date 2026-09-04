==========================
Model Registration Modules
==========================

.. currentmodule:: monai_physio

Model registration classes align PyVista meshes and derived image masks. They
are used by :class:`WorkflowFitStatisticalModelToPatient`.

.. toctree::
   :maxdepth: 2

   icp
   icp_itk
   distance_maps
   pca

Available Classes
=================

* :class:`RegisterModelsICP`: initial surface alignment.
* :class:`RegisterModelsICPITK`: ITK point-set registration.
* :class:`RegisterModelsDistanceMaps`: distance-map based deformable
  registration.
* :class:`RegisterModelsPCA`: statistical shape model fitting.

Workflow-Level Use
==================

Most users should access model registration through the workflow:

.. code-block:: python

   import itk
   import pyvista as pv

   from monai_physio import WorkflowFitStatisticalModelToPatient

   workflow = WorkflowFitStatisticalModelToPatient(
       template_model=pv.read("template_heart.vtu"),
       patient_models=[pv.read("lv.vtp"), pv.read("rv.vtp")],
       patient_image=itk.imread("patient_ct.nii.gz"),
   )

   result = workflow.process()

See Also
========

* :doc:`../workflows`
* :doc:`../../cli_scripts/fit_statistical_model_to_patient`
