===============================
Low-Level vtk_to_usd Subpackage
===============================

.. module:: monai_physio.vtk_to_usd
.. currentmodule:: monai_physio.vtk_to_usd

``monai_physio.vtk_to_usd`` is a stable low-level API for advanced external
users. Inside this repository (experiments, workflows, CLIs, tutorials,
tests), use :class:`~monai_physio.ConvertVTKToUSD` from
:doc:`vtk_conversion` instead of importing this subpackage directly.

This subpackage exposes the readers, data containers, coordinate helpers, and
USD primitive writers that back ``ConvertVTKToUSD``. The public symbols are
documented in their defining submodules below; ``__init__`` re-exports them
for convenience.

File Facade
===========

.. automodule:: monai_physio.vtk_to_usd.converter
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
===============

.. automodule:: monai_physio.vtk_to_usd.data_structures
   :members:
   :undoc-members:
   :show-inheritance:

VTK Readers
===========

.. automodule:: monai_physio.vtk_to_usd.vtk_reader
   :members:
   :undoc-members:
   :show-inheritance:

USD Mesh Conversion
===================

.. automodule:: monai_physio.vtk_to_usd.usd_mesh_converter
   :members:
   :undoc-members:
   :show-inheritance:

Material Manager
================

.. automodule:: monai_physio.vtk_to_usd.material_manager
   :members:
   :undoc-members:
   :show-inheritance:

Mesh Utilities
==============

.. automodule:: monai_physio.vtk_to_usd.mesh_utils
   :members:
   :undoc-members:
   :show-inheritance:

USD Coordinate and Primvar Helpers
==================================

.. automodule:: monai_physio.vtk_to_usd.usd_utils
   :members:
   :undoc-members:
   :show-inheritance:

Primvar Derivations
===================

.. automodule:: monai_physio.vtk_to_usd.primvar_derivations
   :members:
   :undoc-members:
   :show-inheritance:

See Also
========

* :doc:`vtk_conversion`
* :doc:`../workflows`
