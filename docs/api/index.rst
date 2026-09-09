====================
API Reference
====================

Complete API documentation for MONAI Physio modules.

This section provides detailed documentation for all MONAI Physio classes, functions, and modules organized by functionality.

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
   * :class:`~monai_physio.MONAIPhysioBase` - Base class for all components

**Workflows**
   * :class:`~monai_physio.WorkflowConvertImageToUSD` - CT time series to animated USD
   * :class:`~monai_physio.WorkflowConvertImageToVTK` - Segmentation to VTK surfaces
   * :class:`~monai_physio.WorkflowConvertVTKToUSD` - Meshes to USD
   * :class:`~monai_physio.WorkflowReconstructHighres4DCT` - High-resolution 4D reconstruction
   * :class:`~monai_physio.WorkflowFinetuneICONRegistration` - Finetune ICON on your cohort
   * :class:`~monai_physio.WorkflowCreateMeanSurface` - Unbiased mean surface / atlas
   * :class:`~monai_physio.WorkflowCreateStatisticalModel` - Create PCA statistical shape model
   * :class:`~monai_physio.WorkflowFitStatisticalModelToPatient` - Fit the model to a patient

**AI Surrogates (PhysicsNeMo)**
   * :class:`~monai_physio.WorkflowTrainPhysicsNeMo` - Train a mesh-stage model
   * :class:`~monai_physio.WorkflowInferPhysicsNeMo` - Predict per-point targets
   * :class:`~monai_physio.WorkflowInferMovement` - Turn predictions back into geometry
   * :class:`~monai_physio.WorkflowEvaluateMovement` - Score predictions against the acquired frames
   * :class:`~monai_physio.TrainPhysicsNeMoPhysicsInformedMotion` - Train against a neo-Hookean strain energy as well as displacement

**Segmentation**
   * :class:`~monai_physio.SegmentAnatomyBase` - Base segmentation class
   * :class:`~monai_physio.SegmentChestTotalSegmentator` - TotalSegmentator
   * :class:`~monai_physio.SegmentChestTotalSegmentatorWithContrast` - TotalSegmentator for contrast-enhanced CT
   * :class:`~monai_physio.SegmentHeartSimpleware` - Simpleware cardiac segmentation
   * :class:`~monai_physio.SegmentHeartSimplewareTrimmedBranches` - Simpleware with trimmed great vessels
   * :class:`~monai_physio.SegmentNVSegmentCTMRI` - NV-Segment-CTMR, CT *and* MRI

**Image Registration**
   * :class:`~monai_physio.RegisterImagesBase` - Base registration class
   * :class:`~monai_physio.RegisterImagesANTS` - ANTs registration
   * :class:`~monai_physio.RegisterImagesGreedy` - Greedy classical deformable registration
   * :class:`~monai_physio.RegisterImagesICON` - Icon deep learning registration
   * :class:`~monai_physio.RegisterImagesChain` - Run registrations back to back
   * :class:`~monai_physio.RegisterImagesGreedyICON` - Greedy then ICON, as a preset chain
   * :class:`~monai_physio.RegisterTimeSeriesImages` - 4D time series registration

**Model Registration**
   * :class:`~monai_physio.RegisterModelsICP` - Iterative Closest Point
   * :class:`~monai_physio.RegisterModelsICPITK` - ICP with ITK
   * :class:`~monai_physio.RegisterModelsDistanceMaps` - Distance map-based
   * :class:`~monai_physio.RegisterModelsPCA` - PCA-based registration

**USD Tools**
   * :mod:`~monai_physio.usd_tools` - USD file utilities
   * :mod:`~monai_physio.usd_anatomy_tools` - Anatomical structure tools
   * :class:`~monai_physio.ConvertVTKToUSD` - VTK to USD conversion

Module Index
============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
