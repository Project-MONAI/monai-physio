================
Workflow Classes
================

.. module:: monai_physio.workflow_convert_image_to_usd
.. module:: monai_physio.workflow_convert_image_to_vtk
.. module:: monai_physio.workflow_convert_vtk_to_usd
.. module:: monai_physio.workflow_create_mean_surface
.. module:: monai_physio.workflow_create_statistical_model
.. module:: monai_physio.workflow_finetune_icon_registration
.. module:: monai_physio.workflow_fit_statistical_model_to_patient
.. module:: monai_physio.workflow_reconstruct_highres_4d_ct
.. currentmodule:: monai_physio

Workflow classes are the highest-level Python API in MONAI Physio. They
combine segmentation, registration, contour generation, and USD conversion into
repeatable pipelines. The installed CLI commands are thin wrappers around these
classes.

Available Workflows
===================

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Workflow
     - Typical use
   * - :class:`WorkflowConvertImageToUSD`
     - Convert a 4D cardiac CT sequence into animated USD anatomy.
   * - :class:`WorkflowConvertImageToVTK`
     - Segment one CT image and export anatomy-group VTK surfaces.
   * - :class:`WorkflowConvertVTKToUSD`
     - Convert VTK/VTP/VTU meshes or time series into USD.
   * - :class:`WorkflowCreateMeanSurface`
     - Build an unbiased mean surface (atlas) from a population, for use as the
       reference a shape model is built against.
   * - :class:`WorkflowCreateStatisticalModel`
     - Build a PCA shape model from sample meshes aligned to a reference.
   * - :class:`WorkflowFitStatisticalModelToPatient`
     - Fit a template/statistical heart model to patient-specific surfaces.
   * - :class:`WorkflowReconstructHighres4DCT`
     - Reconstruct a high-resolution 4D CT series from phase images and a
       high-resolution reference.
   * - :class:`WorkflowFinetuneICONRegistration`
     - Finetune uniGradICON on your own cohort and return the weights
       :class:`RegisterImagesICON` can load.

The PhysicsNeMo AI-surrogate workflows — :class:`WorkflowTrainPhysicsNeMo`,
:class:`WorkflowInferPhysicsNeMo`, :class:`WorkflowInferMovement` and
:class:`WorkflowEvaluateMovement` — have their own section, since they need the
optional ``[physicsnemo]`` extra. See :doc:`physicsnemo/index`.

Convert Image to USD
====================

.. autoclass:: WorkflowConvertImageToUSD
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   import itk

   from monai_physio import (
       RegisterImagesICON,
       SegmentChestTotalSegmentatorWithContrast,
       WorkflowConvertImageToUSD,
   )

   time_series_images = [itk.imread(str(path)) for path in frame_files]

   workflow = WorkflowConvertImageToUSD(
       time_series_images=time_series_images,
       reference_image=time_series_images[0],
       output_directory="./results",
       usd_project_name="patient_001",
       segmentation_method=SegmentChestTotalSegmentatorWithContrast(),
       registration_method=RegisterImagesICON(),
   )

   results = workflow.process()

Image to VTK
============

.. autoclass:: WorkflowConvertImageToVTK
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   import itk

   from monai_physio import (
       ContourTools,
       SegmentChestTotalSegmentatorWithContrast,
       WorkflowConvertImageToVTK,
   )

   image = itk.imread("chest_ct.nii.gz")
   workflow = WorkflowConvertImageToVTK(
       segmentation_method=SegmentChestTotalSegmentatorWithContrast()
   )
   result = workflow.process(
       input_image=image,
       anatomy_groups=["heart", "major_vessels"],
   )

   ContourTools.save_combined_surfaces(
       result["surfaces"],
       "./output/patient01_surfaces.vtp",
   )

VTK to USD
==========

.. autoclass:: WorkflowConvertVTKToUSD
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   import pyvista as pv
   from monai_physio import WorkflowConvertVTKToUSD

   input_meshes = [pv.read("heart_000.vtp"), pv.read("heart_001.vtp")]
   workflow = WorkflowConvertVTKToUSD(
       input_meshes=input_meshes,
       usd_project_name="heart",
       output_directory="./output",
       appearance="anatomy",
       anatomy_type="heart",
   )

   result = workflow.process()
   usd_file = result["usd_file"]

Statistical Shape Modeling
==========================

.. autoclass:: WorkflowCreateMeanSurface
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WorkflowCreateStatisticalModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WorkflowFitStatisticalModelToPatient
   :members:
   :undoc-members:
   :show-inheritance:

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

High-Resolution 4D CT Reconstruction
====================================

.. autoclass:: WorkflowReconstructHighres4DCT
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   import itk

   from monai_physio import RegisterImagesGreedyICON, WorkflowReconstructHighres4DCT

   time_series_images = [itk.imread(f"phase_{idx:02d}.mha") for idx in range(10)]
   workflow = WorkflowReconstructHighres4DCT(
       time_series_images=time_series_images,
       reference_image=time_series_images[0],
       reference_time_frame=0,
       registration_method=RegisterImagesGreedyICON(),
   )

   workflow.set_modality("ct")
   result = workflow.process()

Finetune ICON Registration
==========================

.. autoclass:: WorkflowFinetuneICONRegistration
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from monai_physio import RegisterImagesICON, WorkflowFinetuneICONRegistration

   workflow = WorkflowFinetuneICONRegistration(
       subject_image_files=subject_image_files,
       output_dir=weights_dir,
       finetune_name="my_cohort",
       subject_ids=subject_ids,
       epochs=100,
   )
   weights_path = workflow.process()

   registrar = RegisterImagesICON()
   registrar.set_weights_path(str(weights_path))

See Also
========

* :doc:`../tutorials`
* :doc:`physicsnemo/index`
* :doc:`../cli_scripts/overview`
* :doc:`../architecture`
