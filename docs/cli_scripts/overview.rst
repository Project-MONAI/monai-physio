====================================
CLI & Scripts Overview
====================================

This section provides comprehensive guides for using MONAI Physio's command-line tools to process medical imaging data. These tools are designed for medical imaging experts and physiological simulation researchers who need efficient, reproducible pipelines for extracting anatomic models from 3D medical images and building the personalized physiological digital twins derived from them.

How to Use These Resources
==========================

MONAI Physio exposes the same toolkit through three user-facing layers:

* **Workflows** are Python classes that orchestrate complete processing
  pipelines. Use them when integrating MONAI Physio into Python applications
  or when you need programmatic control over inputs, outputs, and parameters.
* **CLIs** are installed command-line wrappers around workflow classes. Use them
  for repeatable processing runs, batch jobs, and environment validation without
  writing Python glue code.
* **Tutorials** are repository scripts that demonstrate each major workflow with
  concrete data preparation, commands, and expected outputs. Use them when first
  learning the toolkit or validating a local installation.

The ``experiments/`` directory tracks prior and ongoing research experiments
that helped define this toolkit. Those experiments are useful historical and
design context, but they are not intended to be examples for users or
developers. For supported usage patterns, start with the tutorials, CLIs, and
workflow API documentation.

Target Audience
===============

These CLI tools are intended for users with:

* Strong medical image analysis expertise
* Understanding of physiological simulation requirements
* Modest Python experience for running scripts
* Familiarity with command-line interfaces

If you are a Python developer looking to extend or integrate MONAI Physio into your applications, please refer to the :doc:`../developer/architecture` section.

Available Scripts
=================

Current Scripts
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Description
   * - :doc:`download_data`
     - Download supported MONAI Physio example datasets
   * - :doc:`heart_gated_ct`
     - Process cardiac gated CT to animated heart models with physiological motion
   * - :doc:`../api/cli/convert_image_to_vtk`
     - Segment one 3D image and export anatomy-group VTK surfaces
   * - :doc:`../api/cli/convert_image_4d_to_3d`
     - Split a 4D medical image into a 3D time series using ITK readers
   * - :doc:`create_statistical_model`
     - Build a PCA statistical shape model from sample meshes aligned to a reference
   * - :doc:`fit_statistical_model_to_patient`
     - Register generic heart models to patient-specific imaging data and surface models
   * - :doc:`4dct_reconstruction`
     - Reconstruct high-resolution 4D CT from time-series images and a reference
   * - :doc:`vtk_to_usd`
     - Convert VTK anatomical models to USD format with material painting
   * - :doc:`train_physicsnemo`
     - Train a PhysicsNeMo mesh-stage surrogate from per-subject manifests
   * - :doc:`infer_physicsnemo`
     - Predict motion with a trained surrogate, and rasterize deformation fields
   * - :doc:`../api/cli/visualize_pca_modes`
     - Render PCA model mode visualizations

Installation
============

All scripts are installed with the MONAI Physio package:

.. code-block:: bash

   pip install monai-physio

After installation, scripts are available as command-line tools with the prefix ``monai-physio-``:

.. code-block:: bash

   monai-physio-convert-image-to-usd --help
   monai-physio-download-data --help

General Workflow
================

All MONAI Physio scripts follow a similar pattern:

1. **Input Data**: Provide medical image files (NRRD, NII, MHA formats)
2. **Configuration**: Set processing parameters via command-line flags
3. **Processing Pipeline**: Automated execution of segmentation, registration, and conversion
4. **Output Generation**: USD files ready for Omniverse visualization

Typical Command Structure
--------------------------

.. code-block:: bash

   monai-physio-<command> --help

Use each command's ``--help`` output as the source of truth for required
arguments and script-specific options.

Output Organization
-------------------

Every script writes into the directory you give it, flat, with filenames
prefixed by the project name where one applies:

.. code-block:: text

   output_directory/
   ├── <project_name>.dynamic_painted.usd     # animated USD, when produced
   ├── <project_name>.static_painted.usd
   ├── <project_name>.all_painted.usd
   ├── patient_surfaces.vtp                   # meshes, from the VTK workflows
   ├── patient_labelmap.mha                   # labelmaps and volumes
   └── *.png                                  # screenshots, when requested

The tutorial scripts follow the same rule under
``tutorials/output/<tutorial_name>/``.

Getting Help
============

Each script provides detailed help:

.. code-block:: bash

   monai-physio-<script-name> --help

For troubleshooting and common issues, see :doc:`../troubleshooting`.

Next Steps
==========

* Start with :doc:`heart_gated_ct` for a complete example
* See :doc:`best_practices` for optimization tips
* Refer to script-specific pages for detailed usage
