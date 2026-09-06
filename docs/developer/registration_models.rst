==================================
Model Registration Developer Guide
==================================

Model registration aligns template meshes to patient surfaces and masks. The
supported high-level entry point is
:class:`monai_physio.WorkflowFitStatisticalModelToPatient`.

Recommended Entry Point
=======================

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

Lower-Level Classes
===================

The workflow composes these lower-level registration classes:

* :class:`monai_physio.RegisterModelsICP`
* :class:`monai_physio.RegisterModelsICPITK`
* :class:`monai_physio.RegisterModelsDistanceMaps`
* :class:`monai_physio.RegisterModelsPCA`

Use these directly only when developing or testing a specific registration
stage. Their constructors and return dictionaries are documented in
:doc:`../api/model_registration/index`.

Development Notes
=================

* Prefer PyVista mesh objects at the public Python boundary.
* Convert volumetric meshes to surfaces before surface registration when needed.
* Treat ITK/PyVista coordinate transforms as high-risk and add focused tests.
* Keep synthetic test meshes small and deterministic.
* ``RegisterModelsPCA`` returns ``moving_to_fixed_transform`` /
  ``fixed_to_moving_transform``. These are **point** transforms; see
  :doc:`transform_conventions` before applying them to images or meshes.

See Also
========

* :doc:`transform_conventions`
* :doc:`../api/model_registration/index`
* :doc:`workflows`
