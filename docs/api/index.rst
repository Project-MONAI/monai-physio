====================
API Reference
====================

Complete API documentation for PhysioTwin4D modules.

This section provides detailed documentation for all PhysioTwin4D classes, functions, and modules organized by functionality.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   base
   workflows
   segmentation/index
   registration/index
   model_registration/index
   physicsnemo/index
   usd/index
   utilities/index
   cli/index

Quick Navigation
================

By Category
-----------

**Core Classes**
   * :class:`~physiotwin4d.PhysioTwin4DBase` - Base class for all components

**Workflows**
   * :class:`~physiotwin4d.WorkflowConvertImageToUSD` - CT time series to animated USD
   * :class:`~physiotwin4d.WorkflowConvertImageToVTK` - Segmentation to VTK surfaces
   * :class:`~physiotwin4d.WorkflowConvertVTKToUSD` - Meshes to USD
   * :class:`~physiotwin4d.WorkflowReconstructHighres4DCT` - High-resolution 4D reconstruction
   * :class:`~physiotwin4d.WorkflowFinetuneICONRegistration` - Finetune ICON on your cohort
   * :class:`~physiotwin4d.WorkflowCreateMeanSurface` - Unbiased mean surface / atlas
   * :class:`~physiotwin4d.WorkflowCreateStatisticalModel` - Create PCA statistical shape model
   * :class:`~physiotwin4d.WorkflowFitStatisticalModelToPatient` - Fit the model to a patient

**AI Surrogates (PhysicsNeMo)**
   * :class:`~physiotwin4d.WorkflowTrainPhysicsNeMo` - Train a mesh-stage model
   * :class:`~physiotwin4d.WorkflowInferPhysicsNeMo` - Predict per-point targets
   * :class:`~physiotwin4d.WorkflowInferMovement` - Turn predictions back into geometry
   * :class:`~physiotwin4d.WorkflowEvaluateMovement` - Score predictions against the acquired frames
   * :class:`~physiotwin4d.TrainPhysicsNeMoPhysicsInformedMotion` - Train against a neo-Hookean strain energy as well as displacement

**Segmentation**
   * :class:`~physiotwin4d.SegmentAnatomyBase` - Base segmentation class
   * :class:`~physiotwin4d.SegmentChestTotalSegmentator` - TotalSegmentator
   * :class:`~physiotwin4d.SegmentChestTotalSegmentatorWithContrast` - TotalSegmentator for contrast-enhanced CT
   * :class:`~physiotwin4d.SegmentHeartSimpleware` - Simpleware cardiac segmentation
   * :class:`~physiotwin4d.SegmentHeartSimplewareTrimmedBranches` - Simpleware with trimmed great vessels
   * :class:`~physiotwin4d.SegmentNVSegmentCTMRI` - NV-Segment-CTMR, CT *and* MRI

**Image Registration**
   * :class:`~physiotwin4d.RegisterImagesBase` - Base registration class
   * :class:`~physiotwin4d.RegisterImagesANTS` - ANTs registration
   * :class:`~physiotwin4d.RegisterImagesGreedy` - Greedy classical deformable registration
   * :class:`~physiotwin4d.RegisterImagesICON` - Icon deep learning registration
   * :class:`~physiotwin4d.RegisterImagesChain` - Run registrations back to back
   * :class:`~physiotwin4d.RegisterImagesGreedyICON` - Greedy then ICON, as a preset chain
   * :class:`~physiotwin4d.RegisterTimeSeriesImages` - 4D time series registration

**Model Registration**
   * :class:`~physiotwin4d.RegisterModelsICP` - Iterative Closest Point
   * :class:`~physiotwin4d.RegisterModelsICPITK` - ICP with ITK
   * :class:`~physiotwin4d.RegisterModelsDistanceMaps` - Distance map-based
   * :class:`~physiotwin4d.RegisterModelsPCA` - PCA-based registration

**USD Tools**
   * :mod:`~physiotwin4d.usd_tools` - USD file utilities
   * :mod:`~physiotwin4d.usd_anatomy_tools` - Anatomical structure tools
   * :class:`~physiotwin4d.ConvertVTKToUSD` - VTK to USD conversion

Module Index
============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
