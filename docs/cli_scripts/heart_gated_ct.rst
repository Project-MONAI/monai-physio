====================================
Heart Gated CT Processing
====================================

Process cardiac gated CT images into dynamic, animated heart models for visualization in NVIDIA Omniverse.

Overview
========

The ``monai-physio-convert-image-to-usd`` script processes 4D cardiac CT scans through a complete pipeline that includes:

* AI-based anatomical segmentation
* Deformable image registration across cardiac phases
* Surface mesh extraction and transformation
* USD file generation with anatomically-realistic materials

This tool is ideal for:

* Visualizing cardiac motion through the cardiac cycle
* Creating patient-specific animated heart models
* Research and educational demonstrations of cardiac physiology
* Pre-procedural planning with dynamic anatomical models

Input Requirements
==================

Data Format
-----------

Accepts two input formats:

**Option 1: Single 4D NRRD file**
   * Contains all cardiac phases in one file
   * Format: ``.nrrd`` or ``.seq.nrrd``
   * Example: ``cardiac_4d_ct.nrrd``

**Option 2: Multiple 3D files (time series)**
   * Separate file for each cardiac phase
   * Formats: ``.nrrd``, ``.nii``, ``.mha``
   * Example: ``phase_00.nrrd``, ``phase_10.nrrd``, ``phase_20.nrrd``

Image Quality Requirements
---------------------------

* **Modality**: Cardiac CT (contrast-enhanced recommended)
* **Phases**: Minimum 2, typically 10-20 cardiac phases
* **Resolution**: 0.5-2mm isotropic recommended
* **Field of View**: Must include entire heart and major vessels
* **Image Quality**: Minimal motion artifacts, good contrast

Basic Usage
===========

Single 4D File
--------------

.. code-block:: bash

   monai-physio-convert-image-to-usd cardiac_4d.nrrd --contrast

Multiple 3D Files
-----------------

.. code-block:: bash

   monai-physio-convert-image-to-usd phase_*.nrrd --contrast --project-name patient_001

With Output Directory
---------------------

.. code-block:: bash

   monai-physio-convert-image-to-usd cardiac.nrrd \
       --contrast \
       --output-dir ./results/patient_001 \
       --project-name patient_001

Command-Line Options
====================

Required Arguments
------------------

.. code-block:: text

   input_files         Path to 4D NRRD file OR list of 3D files

Optional Arguments
------------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--output-dir DIR``
     - ``./results``
     - Output directory for intermediate and final files
   * - ``--project-name NAME``
     - ``cardiac_model``
     - Project name used for USD file naming
   * - ``--contrast``
     - False
     - Flag for contrast-enhanced studies (recommended)
   * - ``--reference-image PATH``
     - Auto (70%)
     - Custom reference image for registration
   * - ``--registration-iterations N``
     - 1
     - Number of registration refinement iterations
   * - ``--registration-method``
     - ``ICON``
     - Registration method: ``ICON`` or ``Greedy``

Processing Pipeline
===================

The script executes these steps automatically:

1. **Load Time Series**
   * Converts 4D to 3D time series OR loads multiple files
   * Selects reference image (default: 70% cardiac phase)

2. **Segmentation**
   * Segments reference image using AI segmentation (TotalSegmentator or Simpleware)
   * Identifies: heart chambers, myocardium, vessels, lungs, bones

3. **Registration**
   * Registers each phase to reference image
   * Creates separate registrations for dynamic vs static anatomy
   * Uses mass-preserving deformable registration

4. **Mesh Generation**
   * Extracts VTK surface meshes from segmentation
   * Creates smooth anatomical surface representations

5. **Transform Application**
   * Applies registration transforms to meshes at each time point
   * Generates animated mesh sequences

6. **USD Creation**
   * Produces painted USD files with anatomical materials
   * Creates time-varying geometry for Omniverse

Output Files
============

Primary Outputs
---------------

Located in ``<output-dir>/``:

.. code-block:: text

   <project_name>.dynamic_anatomy_painted.usd
       Animated heart and vessels with cardiac motion
   
   <project_name>.static_anatomy_painted.usd
       Stationary anatomy (lungs, bones, tissues)
   
   <project_name>.all_anatomy_painted.usd
       Combined model with all structures

Intermediate Files
------------------

.. code-block:: text

   intermediate/
   ├── slice_*.mha                  # Individual 3D images per time point
   ├── slice_*.labelmap.mha         # Segmentation masks
   ├── slice_*.reg_*.moving_to_fixed_transform.hdf5  # Registration transforms
   └── slice_max.reg_*.mha          # Maximum intensity projections
   
   meshes/
   ├── *.vtk                        # VTK mesh files
   └── *_4d.vtk                     # Time series VTK files

Examples
========

Contrast-Enhanced Cardiac CTA
------------------------------

.. code-block:: bash

   monai-physio-convert-image-to-usd cardiac_cta.nrrd \
       --contrast \
       --output-dir ./output/patient_123 \
       --project-name PatientXYZ_CTA

Non-Contrast Study with Custom Reference
-----------------------------------------

.. code-block:: bash

   monai-physio-convert-image-to-usd phase_*.nrrd \
       --reference-image phase_05.mha \
       --output-dir ./results \
       --project-name noncontrast_cardiac

Research Dataset Processing
----------------------------

.. code-block:: bash

   # Batch process multiple cases
   for case_dir in /data/cardiac_studies/case_*/; do
       case_name=$(basename "$case_dir")
       monai-physio-convert-image-to-usd ${case_dir}/*.nrrd \
           --contrast \
           --output-dir ./results/${case_name} \
           --project-name ${case_name}
   done

With Greedy Registration
------------------------

.. code-block:: bash

   monai-physio-convert-image-to-usd cardiac.nrrd \
       --contrast \
       --registration-method Greedy \
       --registration-iterations 50

Best Practices
==============

Choosing Reference Image
-------------------------

* **Default (70% phase)**: Mid-diastole, suitable for most cases
* **Custom selection**: Choose phase with:
  
  * Best image quality
  * Least motion artifact
  * Optimal vessel enhancement (for CTA)

Performance Optimization
------------------------

* **GPU**: Enable CUDA for faster registration (automatic if available)
* **RAM**: 16GB+ recommended for large datasets
* **Storage**: Use SSD for faster I/O operations
* **Iterations**: Start with default (1), increase only if needed

Data Quality Tips
-----------------

* Verify cardiac phases are temporally ordered
* Check consistent spacing and orientation across phases
* Ensure uniform contrast enhancement (for CTA)
* Confirm heart structures are within field of view

Troubleshooting
===============

Input File Issues
-----------------

**Error: "Input file not found"**
   * Verify file paths are correct
   * Use absolute paths if relative paths fail
   * Check file permissions

**Error: "Unable to read image"**
   * Confirm file format is supported (NRRD, NII, MHA)
   * Check file is not corrupted
   * Verify image metadata is complete

Segmentation Quality Issues
----------------------------

**Poor heart segmentation**
   * Use ``--contrast`` flag for CTA studies
   * Verify cardiac structures are visible
   * Check image quality (no severe artifacts)

**Missing anatomical structures**
   * Ensure structures are in field of view
   * Check image resolution is adequate
   * Verify reference image quality

Registration Failures
---------------------

**Registration not converging**
   * Try different reference image (better quality phase)
   * Increase ``--registration-iterations``
   * Switch registration method (``--registration-method Greedy``)

**Excessive deformation**
   * Verify sufficient temporal overlap between phases
   * Check image spacing is reasonable
   * Ensure phases are properly ordered

Memory Issues
-------------

**Out of memory errors**
   * Reduce image resolution if possible
   * Process fewer time points
   * Close other applications
   * Consider using machine with more RAM

See :doc:`../troubleshooting` for additional help.

Expected Processing Time
========================

Approximate times on typical workstation (32GB RAM, NVIDIA GPU):

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Dataset
     - GPU Available
     - No GPU
   * - 10 phases, 512³ volume
     - 15-30 minutes
     - 2-4 hours
   * - 20 phases, 512³ volume
     - 30-60 minutes
     - 4-8 hours

Viewing Results
===============

1. Open the ``.usd`` file in an Omniverse Kit application, with the viewport
   renderer set to RTX
2. Switch the viewport to the scene's ``/World/Camera``
3. Press Play to view the cardiac motion animation
4. Adjust the timeline speed for the visualization you want

:doc:`../viewing_usd` covers installing the viewer and what to look at once
the scene is open.

Next Steps
==========

* See :doc:`best_practices` for optimization strategies
* Review :doc:`../troubleshooting` for common issues
* For Python API access, see :class:`monai_physio.WorkflowConvertImageToUSD` in :doc:`../developer/workflows`
