=====================
Architecture Overview
=====================

MONAI Physio is an early-beta scientific Python package for creating
personalized physiological digital twins, built from workflow classes plus
reusable segmentation, registration, geometry, image, and USD components.
Runtime classes inherit from :class:`MONAIPhysioBase` for
logging and consistent configuration.

Architecture Diagram
====================

.. code-block:: text

   CLI scripts / tutorials
          |
          v
   Workflow classes
      |       |       |       |
      v       v       v       v
   ITK images  Segmentation  Registration  PyVista meshes
      |              |             |             |
      +--------------+-------------+-------------+
                         |
                         v
               ContourTools / TransformTools
                         |
                         v
              ConvertVTKToUSD / vtk_to_usd
                         |
                         v
                 OpenUSD / Omniverse

Workflow Classes
================

``WorkflowConvertImageToUSD``
   Orchestrates 4D CT loading, segmentation, image registration, contour
   transformation, and animated USD generation.

``WorkflowConvertImageToVTK``
   Converts one 3D image into labeled VTK surface outputs.

``WorkflowCreateStatisticalModel``
   Builds a PCA statistical shape model from aligned population meshes.

``WorkflowFitStatisticalModelToPatient``
   Applies a template or PCA model to patient surfaces with model-registration
   stages.

``WorkflowReconstructHighres4DCT``
   Reconstructs high-resolution 4D CT frames using time-series registration to
   a fixed high-resolution reference.

``WorkflowConvertVTKToUSD``
   Wraps VTK-to-USD conversion for repository workflows and CLI use.

Key Boundaries
==============

Image processing uses ITK images in LPS world space. Surface and volume meshes
use PyVista/VTK and inherit that LPS frame. USD export converts those meshes to
OpenUSD and applies the repository's LPS-to-USD-Y-up transform at the
VTK-to-USD boundary.

The installed CLI commands are thin wrappers around these workflow classes.
They are the best executable references for supported API usage.

See Also
========

* :doc:`extending`
* :doc:`migration_next`
