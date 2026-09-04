====================================
Create Statistical Model
====================================

Overview
========

The ``monai-physio-create-statistical-model`` command-line tool builds a PCA
(Principal Component Analysis) statistical shape model from a sample of meshes
aligned to a reference mesh. This mirrors the pipeline in the
Heart-Create_Statistical_Model experiment scripts.

The workflow:

1. **Extract surfaces** from sample and reference meshes
2. **ICP alignment**: Affine align each sample surface to the reference surface
3. **Deformable registration**: ANTs SyN to establish dense correspondence
4. **Correspondence**: Build aligned shapes with reference topology
5. **PCA**: Compute mean shape and principal components

Outputs written to the output directory:

* ``pca_mean_surface.vtp`` — Mean shape as a surface (PolyData)
* ``pca_mean.vtu`` — Reference volume mesh in mean space (only if reference is volumetric)
* ``pca_model.json`` — PCA model (eigenvalues, components) for use with
  :class:`monai_physio.WorkflowFitStatisticalModelToPatient` or
  :class:`monai_physio.RegisterModelsPCA`

Installation
============

The script is installed with MONAI Physio:

.. code-block:: bash

   pip install monai-physio

Quick Start
===========

Basic Usage
-----------

Create a PCA model from a directory of sample meshes and a reference mesh:

.. code-block:: bash

   monai-physio-create-statistical-model \
       --sample-meshes-dir ./input_meshes \
       --reference-mesh average_mesh.vtk \
       --output-dir ./pca_output

Explicit Sample List
--------------------

Provide sample mesh paths explicitly instead of a directory:

.. code-block:: bash

   monai-physio-create-statistical-model \
       --sample-meshes 01.vtk 02.vtk 03.vtu 04.vtp \
       --reference-mesh average_mesh.vtk \
       --output-dir ./pca_output

With Custom Parameters
----------------------

.. code-block:: bash

   monai-physio-create-statistical-model \
       --sample-meshes-dir ./meshes \
       --reference-mesh average_mesh.vtk \
       --output-dir ./pca_output \
       --number-of-pca-components 20

Command-Line Arguments
======================

Required Arguments
------------------

``--sample-meshes-dir DIR`` or ``--sample-meshes PATH [PATH ...]``
   Either a directory containing sample mesh files (``.vtk``, ``.vtu``, ``.vtp``)
   or a list of paths to sample meshes. One of these is required.

``--reference-mesh PATH``
   Path to the reference mesh. Its surface is used as the alignment target for
   all samples.

``--output-dir DIR``
   Output directory. Writes ``pca_mean_surface.vtp``, ``pca_mean.vtu`` (if
   reference is volumetric), and ``pca_model.json``.

Optional Arguments
------------------

``--number-of-pca-components N``
   Number of PCA components to retain (default: 7).

See :class:`monai_physio.WorkflowCreateStatisticalModel` for the full API and
additional parameters (e.g. ``reference_spatial_resolution``,
``reference_buffer_factor``) that can be exposed in future CLI versions.
