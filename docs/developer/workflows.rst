==========================
Workflow Development Guide
==========================

Workflow classes coordinate multiple processing steps behind a stable Python API
and, where useful, an installed CLI command.

Current Workflow Mapping
========================

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - CLI command
     - Workflow class
   * - ``monai-physio-convert-image-to-usd``
     - :class:`monai_physio.WorkflowConvertImageToUSD`
   * - ``monai-physio-convert-image-to-vtk``
     - :class:`monai_physio.WorkflowConvertImageToVTK`
   * - ``monai-physio-convert-vtk-to-usd``
     - :class:`monai_physio.WorkflowConvertVTKToUSD`
   * - ``monai-physio-create-statistical-model``
     - :class:`monai_physio.WorkflowCreateStatisticalModel`
   * - ``monai-physio-fit-statistical-model-to-patient``
     - :class:`monai_physio.WorkflowFitStatisticalModelToPatient`
   * - ``monai-physio-reconstruct-highres-4d-ct``
     - :class:`monai_physio.WorkflowReconstructHighres4DCT`
   * - ``monai-physio-train-physicsnemo``
     - :class:`monai_physio.WorkflowTrainPhysicsNeMo`
   * - ``monai-physio-infer-physicsnemo``
     - :class:`monai_physio.WorkflowInferPhysicsNeMo`
   * - ``monai-physio-convert-image-4d-to-3d``
     - :class:`monai_physio.ConvertImage4DTo3D` (a converter, not a workflow)
   * - ``monai-physio-download-data``
     - :class:`monai_physio.DownloadData` (a utility, not a workflow)
   * - ``monai-physio-visualize-pca-modes``
     - Reads a ``pca_model.json`` directly; no workflow class

That is all eleven installed commands. Two workflow classes have no CLI
wrapper: :class:`monai_physio.WorkflowFinetuneICONRegistration` and
:class:`monai_physio.WorkflowEvaluateMovement`.

Workflow Example
================

.. code-block:: python

   from pathlib import Path

   import itk

   from monai_physio import RegisterImagesICON, WorkflowConvertImageToUSD

   frame_files = sorted(Path("data/Slicer-Heart-CT").glob("slice_???.mha"))
   time_series_images = [itk.imread(str(path)) for path in frame_files]

   workflow = WorkflowConvertImageToUSD(
       time_series_images=time_series_images,
       reference_image=time_series_images[0],
       output_directory="./results",
       usd_project_name="patient_001",
       registration_method=RegisterImagesICON(),
   )

   results = workflow.process()

Selecting a fitting grid
========================

``WorkflowFitStatisticalModelToPatient`` accepts optional constructor arguments
``fitting_crop_margin_mm`` and ``fitting_spacing_mm``. Cropping happens before
isotropic resampling, avoiding a full-volume fine-grid allocation. The crop uses
the nonzero patient labelmap, which must share the input image grid. Provide an
anatomy-specific labelmap so neighboring organs do not contribute to PCA fitting.

For an initial lung configuration to validate on your data::

   fit = WorkflowFitStatisticalModelToPatient(
       template_model=mean_lung_model,
       patient_models=[patient_lung_surface],
       patient_image=ct,
       patient_labelmap=lung_labelmap,
       fitting_crop_margin_mm=40.0,
       fitting_spacing_mm=1.5,
   )

These values are examples, not heart or lung defaults. Omitting both options
preserves existing behavior. ``ProcessImages.crop_to_mask`` is also available
independently; it retains surrounding tissue and pixel type, unlike the
labelmap-only ``crop_to_nonzero`` with its added zero padding.

With either option enabled, patient labels are sampled by nearest neighbor on
the fitting grid before constructing distance maps. Label IDs and interior-label
exclusions retain their meanings. Registration grids include patient and aligned
template surfaces plus physical registration margins, including when a crop
touches the input boundary. The source labelmap is retained and sampled directly
onto expanded registration grids, preserving anatomy outside the CT extent.
Intermediate template labelmaps also use expanded grids, allowing later
registration stages to move anatomy back inside the crop without losing labels.
Meshes remain in patient coordinates. The original
CT remains available as ``fit.original_patient_image`` for full-resolution motion
warping; fitting stages, including intensity refinement, use ``fit.patient_image``.

Coarser sampling can remove small structures and change PCA coefficients and
motion. Compare lung and heart surface errors, coefficients, and predicted
motion against the full-resolution configuration before selecting a grid for a
new dataset. Agreement with another fit is not accuracy against ground truth.

Adding a Workflow
=================

1. Inherit from :class:`monai_physio.MONAIPhysioBase`.
2. Keep the constructor explicit and typed.
3. Use ``self.log_info()`` and ``self.log_debug()`` for runtime status.
4. Keep file I/O behavior predictable and documented.
5. Add a CLI wrapper only when the workflow is useful from the command line.
6. Add focused tests using synthetic data where possible.
7. Run ``graphify update .`` after public API changes - methods added,
   modified, or removed.

Risk Areas
==========

Changes at the ITK-to-PyVista boundary, time-series transform direction, or
LPS-to-USD-Y-up coordinate conversion are high risk and should include focused
tests plus visual or metadata validation.

See Also
========

* :doc:`../api/workflows`
* :doc:`../cli_scripts/overview`
* :doc:`../architecture`
