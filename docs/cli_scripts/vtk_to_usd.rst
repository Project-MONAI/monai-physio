=====================
VTK to USD Conversion
=====================

The ``monai-physio-convert-vtk-to-usd`` command converts VTK, VTP, or VTU
mesh files to USD for Omniverse visualization. Multiple input files are
treated as an ordered time series by default; pass ``--static-merge`` to
instead merge unrelated static meshes into one scene with no time samples.

Basic Usage
===========

.. code-block:: bash

   monai-physio-convert-vtk-to-usd heart.vtp \
       --output heart.usd

Time Series
===========

.. code-block:: bash

   monai-physio-convert-vtk-to-usd heart_*.vtp \
       --output heart_animation.usd \
       --fps 30

Appearance Options
==================

Solid color:

.. code-block:: bash

   monai-physio-convert-vtk-to-usd heart.vtp \
       --output heart_red.usd \
       --appearance solid \
       --color 1 0 0

One anatomy material for every mesh:

.. code-block:: bash

   monai-physio-convert-vtk-to-usd heart.vtp \
       --output heart_material.usd \
       --appearance anatomy \
       --anatomy-type heart

A material per structure. Omitting ``--anatomy-type`` picks each object's
material from its name, and with ``--static-merge`` the objects are named after
the structures recorded in each file's ``SegmentationLabelNames`` field data -
as written by the image-to-VTK workflow:

.. code-block:: bash

   monai-physio-convert-vtk-to-usd patient_highres_*.vtp \
       --output heart_structures.usd \
       --appearance anatomy \
       --static-merge

Colormap from a VTK point data array:

.. code-block:: bash

   monai-physio-convert-vtk-to-usd frame_*.vtk \
       --output stress.usd \
       --appearance colormap \
       --primvar vtk_point_stress_c0 \
       --cmap viridis \
       --intensity-range 0 500

Splitting
=========

By default, meshes are split by connected component. Use ``--no-split`` to keep
one mesh, or ``--by-cell-type`` to split by cell type.

.. code-block:: bash

   monai-physio-convert-vtk-to-usd mesh.vtu \
       --output mesh.usd \
       --by-cell-type

Python API
==========

Use :class:`monai_physio.WorkflowConvertVTKToUSD` for the workflow API
(splitting and appearance built in) and :class:`monai_physio.ConvertVTKToUSD`
for lower-level control (e.g. anatomical label splitting via ``mask_ids``).
Both take in-memory PyVista/VTK meshes.

Related Pages
=============

* :doc:`overview`
* :doc:`../api/usd/vtk_conversion`
* :doc:`../developer/usd_generation`
