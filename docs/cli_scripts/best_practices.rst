====================================
Best Practices
====================================

This guide provides recommendations for optimal use of MONAI Physio CLI scripts, focusing on data preparation, processing strategies, and troubleshooting approaches.

Data Preparation
================

Image Quality Guidelines
-------------------------

**Spatial Resolution**
   * Cardiac CT: 0.5-1.0mm isotropic optimal
   * Lung CT: 1.0-2.0mm acceptable
   * Brain angiography: <0.5mm for small vessels

**Temporal Sampling**
   * Cardiac: 10-20 phases per cycle minimum
   * Respiratory: 8-10 phases typically sufficient
   * More phases = smoother motion but longer processing

**Field of View**
   * Include entire organ of interest
   * Include surrounding anatomy for context
   * Avoid truncation artifacts

**Image Artifacts**
   * Minimize motion artifacts (ECG/respiratory gating)
   * Ensure uniform contrast enhancement
   * Avoid metal artifacts in region of interest

File Organization
-----------------

Recommended directory structure:

.. code-block:: text

   project/
   ├── raw_data/
   │   ├── patient_001/
   │   │   ├── phase_00.nrrd
   │   │   ├── phase_01.nrrd
   │   │   └── ...
   │   ├── patient_002/
   │   └── ...
   ├── processing/
   │   ├── patient_001/
   │   └── ...
   └── results/
       ├── patient_001/
       └── ...

**Naming Conventions**
   * Use consistent, descriptive names
   * Include patient ID and phase information
   * Avoid spaces and special characters
   * Use leading zeros for sorting (``phase_00``, not ``phase_0``)

Processing Strategies
=====================

Computational Resources
-----------------------

**GPU Acceleration**
   * Enable CUDA for 5-10x speedup in registration
   * Verify GPU availability: ``nvidia-smi``
   * Most benefit in registration steps

**Memory Management**
   * 16GB RAM minimum for typical datasets
   * 32GB+ recommended for large volumes
   * Monitor memory usage: ``top`` or Task Manager
   * Consider downsampling for very large images

**Storage**
   * SSD strongly recommended for I/O performance
   * Plan for 5-10x input data size for outputs
   * Keep intermediate files until verification complete

Workflow Optimization
---------------------

**Start Simple**
   1. Process single case with default parameters
   2. Verify output quality
   3. Adjust parameters if needed
   4. Apply to full dataset

**Parameter Selection**
   * Use defaults first (they work for most cases)
   * Adjust only if quality issues observed
   * Document parameter changes for reproducibility

**Batch Processing**

.. code-block:: bash

   # Create processing script
   for patient_dir in raw_data/patient_*/; do
       patient_id=$(basename "$patient_dir")
       echo "Processing $patient_id"
       
       monai-physio-convert-image-to-usd \
           ${patient_dir}/*.nrrd \
           --contrast \
           --output-dir processing/${patient_id} \
           --project-name ${patient_id} \
           2>&1 | tee processing/${patient_id}/log.txt
   done

Quality Control
===============

Verification Checklist
----------------------

After processing, verify:

**Segmentation Quality**
   □ All expected structures identified
   □ No major segmentation errors
   □ Boundaries are smooth and anatomically correct

**Registration Quality**
   □ Good alignment across phases
   □ No excessive warping or distortions
   □ Smooth temporal transitions

**Mesh Quality**
   □ Surfaces are smooth and closed
   □ No gaps or overlaps
   □ Appropriate level of detail

**USD Files**
   □ Files open correctly in Omniverse
   □ Animation plays smoothly
   □ Materials are correctly applied
   □ Anatomical colors are appropriate

Automated Quality Metrics
--------------------------

MONAI Physio logs quality metrics during processing:

* **Segmentation confidence**: Check for low-confidence regions
* **Registration error**: Monitor convergence
* **Mesh statistics**: Verify vertex/face counts are reasonable

Common Issues and Solutions
============================

Poor Segmentation
-----------------

**Issue**: Important structures missing or incorrectly segmented

**Solutions**:
   * Use ``--contrast`` flag for contrast-enhanced studies
   * Verify image quality and contrast are adequate
   * Check that structures are in field of view
   * Consider manual correction if needed

Registration Failure
--------------------

**Issue**: Registration not converging or producing artifacts

**Solutions**:
   * Select better reference image (less motion, better quality)
   * Increase ``--registration-iterations``
   * Try alternative registration method (``--registration-method Greedy``)
   * Verify phases are temporally ordered correctly

Memory Errors
-------------

**Issue**: Out of memory during processing

**Solutions**:
   * Close other applications
   * Reduce image resolution (resample before processing)
   * Process fewer phases simultaneously
   * Use machine with more RAM

Slow Processing
---------------

**Issue**: Processing takes excessive time

**Solutions**:
   * Enable GPU acceleration
   * Use SSD for storage
   * Reduce registration iterations if acceptable quality
   * Consider cloud computing for large batches

Performance Benchmarks
======================

Typical processing times (NVIDIA RTX 3090, 32GB RAM):

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Task
     - Small Dataset
     - Large Dataset
   * - Cardiac CT (10 phases)
     - 20-30 min
     - 45-60 min
   * - Lung CT (10 phases)
     - 25-40 min
     - 60-90 min
   * - Model registration
     - 5-10 min
     - 15-25 min

*Small: 256³-384³ volumes; Large: 512³-768³ volumes*

Reproducibility
===============

Ensuring Reproducible Results
------------------------------

**Version Control**
   * Record MONAI Physio version: ``pip show monai_physio``
   * Document dependency versions
   * Use virtual environments

**Parameter Documentation**
   * Save command-line arguments used
   * Record any manual interventions
   * Note quality issues and solutions

**Example Documentation**:

.. code-block:: yaml

   # processing_metadata.yaml
   patient_id: patient_001
   monai_physio_version: {{ mphysio_project_version }}
   script: heart-gated-ct
   date: 2026-01-08
   
   parameters:
     contrast: true
     registration_method: icon
     registration_iterations: 1
     reference_image: auto
   
   hardware:
     gpu: NVIDIA RTX 3090
     ram: 32GB
   
   processing_time: 28min
   quality_notes: Good segmentation, smooth registration

Advanced Topics
===============

Custom Reference Selection
--------------------------

Choose reference image based on:

* **Image quality**: Sharpest, least artifact
* **Physiological phase**: Representative motion state
* **Contrast timing**: Peak enhancement (for CTA)
* **Clinical target**: Phase of interest for application

Multi-Phase Studies
-------------------

For non-gated multi-phase acquisitions (e.g., arterial/venous/delayed):

.. code-block:: bash

   monai-physio-convert-image-to-usd \
       arterial.nrrd \
       venous.nrrd \
       delayed.nrrd \
       --project-name multiphase_study

Parallel Processing
-------------------

Process multiple patients in parallel:

.. code-block:: bash

   # Using GNU parallel
   parallel -j 4 'monai-physio-convert-image-to-usd {}/*.nrrd \
       --output-dir results/{/} \
       --project-name {/}' ::: raw_data/patient_*

   # Adjust -j to match CPU cores/GPU availability

Getting Help
============

If you encounter issues not covered here:

1. Check :doc:`../troubleshooting` for detailed problem-solving
2. Review script-specific documentation
3. Consult :doc:`../faq` for common questions
4. Check GitHub issues for similar problems
5. Open new GitHub issue with:
   
   * MONAI Physio version
   * Complete command used
   * Error messages
   * System information

Next Steps
==========

* Apply these practices to your specific use case
* See individual script pages for detailed usage
* Consult :doc:`../developer/extending` to customize workflows
