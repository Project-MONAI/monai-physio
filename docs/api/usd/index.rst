====================================
USD Generation Modules
====================================

.. currentmodule:: monai_physio

Convert medical imaging data to Universal Scene Description (USD) format for NVIDIA Omniverse.

Overview
========

USD generation tools for creating animated 3D models from medical images:

* **USD Tools**: Core USD file operations
* **Anatomy Tools**: Anatomical structure handling
* **VTK Conversion**: Convert VTK meshes to USD with :class:`ConvertVTKToUSD`

Quick Links
===========

**USD Modules**:
   * :doc:`tools` - Core USD utilities
   * :doc:`anatomy_tools` - Anatomical structure tools
   * :doc:`vtk_conversion` - VTK to USD conversion (preferred high-level API)
   * :doc:`vtk_to_usd_lib` - Low-level ``vtk_to_usd`` subpackage (advanced)

Module Documentation
====================

.. toctree::
   :maxdepth: 2

   tools
   anatomy_tools
   vtk_conversion
   vtk_to_usd_lib

Quick Start
===========

Convert VTK to USD
------------------

.. code-block:: python

   from monai_physio import ConvertVTKToUSD
   
   converter = ConvertVTKToUSD.from_files(
       data_basename="Heart",
       vtk_files=["heart_phase_00.vtk", "heart_phase_01.vtk"],
       time_codes=[0.0, 1.0],
   )
   stage = converter.convert("animated_heart.usd")

Create Anatomical Scene
-----------------------

.. code-block:: python

   from monai_physio import usd_anatomy_tools
   
   stage = usd_anatomy_tools.create_anatomical_stage()
   usd_anatomy_tools.add_heart_model(stage, "heart.vtk")
   usd_anatomy_tools.add_lungs_model(stage, "lungs.vtk")
   stage.Save()

See Also
========

* :doc:`../workflows` - Complete USD workflows
* :doc:`../segmentation/index` - Segmentation for USD generation

.. rubric:: Navigation

:doc:`../model_registration/pca` | :doc:`../index` | :doc:`tools`
