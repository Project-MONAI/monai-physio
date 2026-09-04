========================
USD Generation
========================

MONAI Physio uses :class:`monai_physio.ConvertVTKToUSD` as the
application-level API for VTK-to-USD conversion. Workflows, command-line tools,
and experiments should use this class instead of importing
``monai_physio.vtk_to_usd`` directly.

``monai_physio.vtk_to_usd`` remains a public advanced low-level package for
external users who need file readers, data containers, or direct USD writer
primitives.

ConvertVTKToUSD
================

Use :class:`monai_physio.ConvertVTKToUSD` for application-level conversion.
The full API reference is in :doc:`../api/usd/vtk_conversion`.

Single File
-----------

.. code-block:: python

   from monai_physio import ConvertVTKToUSD

   stage = ConvertVTKToUSD.from_files(
       data_basename='Heart',
       vtk_files=['heart.vtp'],
       extract_surface=True,
   ).convert('heart.usd')

Time Series
-----------

.. code-block:: python

   from monai_physio import ConvertVTKToUSD

   stage = ConvertVTKToUSD.from_files(
       data_basename='Heart',
       vtk_files=['heart_t0.vtp', 'heart_t1.vtp', 'heart_t2.vtp'],
       time_codes=[0.0, 1.0, 2.0],
       frames_per_second=24.0,
   ).convert('animated_heart.usd')

In-Memory Meshes
----------------

.. code-block:: python

   from monai_physio import ConvertVTKToUSD
   import pyvista as pv

   meshes = [pv.read(path) for path in vtk_files]
   converter = ConvertVTKToUSD(
       data_basename='CardiacModel',
       input_polydata=meshes,
       compute_normals=True,
   )
   stage = converter.convert('cardiac_model.usd')

Labeled Meshes with Anatomy Grouping
------------------------------------

When the input has anatomical labels (``mask_ids`` and a ``boundary_labels``
cell array), pass a :class:`monai_physio.SegmentAnatomyBase` instance
through the ``segmenter`` argument so labeled prims are grouped by anatomy
type. The output layout becomes:

* Meshes: ``/World/{basename}/{group}/{organ_name}``
* Materials: ``/World/Looks/{group}/{organ_name}_material``

with one Xform per group (``heart``, ``lung``, ``bone``, ...) that only
appears if at least one labeled organ from the input falls into that group.

.. code-block:: python

   from monai_physio import (
       ConvertVTKToUSD,
       SegmentChestTotalSegmentator,
   )

   seg = SegmentChestTotalSegmentator()
   converter = ConvertVTKToUSD(
       data_basename='Patient',
       input_polydata=meshes,
       mask_ids=seg.taxonomy.all_labels(),
       segmenter=seg,
   )
   stage = converter.convert('patient.usd')
   # /World/Patient/heart/heart, /World/Patient/lung/lung_upper_lobe_left, ...
   # /World/Looks/heart/heart_material, ...

If no segmenter is supplied, labels are grouped under a single ``Anatomy``
Xform (``/World/{basename}/Anatomy/{organ_name}``) so the hierarchy stays
two-deep regardless.

The classification logic is delegated to the segmenter's
:class:`AnatomyTaxonomy`; new groups added there flow into the prim layout
without any change to ``ConvertVTKToUSD``. See
:doc:`../api/segmentation/base` for the taxonomy contract and
:doc:`segmentation` for how to add a new segmenter.

Colormaps
---------

.. code-block:: python

   converter.set_colormap(
       color_by_array='pressure',
       colormap='viridis',
       intensity_range=(0.0, 1.0),
   )
   stage = converter.convert('pressure.usd')

Derived Scalars: von Mises Stress
---------------------------------

For finite-element outputs that carry a 9-component stress tensor field
(e.g. cardiac valve simulations), call ``compute_von_mises_stress`` between
``from_files`` and ``convert`` to add a scalar ``von_mises_stress`` primvar
suitable for colormap rendering:

.. code-block:: python

   stage = (
       ConvertVTKToUSD.from_files(
           data_basename='Valve',
           vtk_files=valve_files,
           time_codes=time_codes,
           separate_by='connectivity',
       )
       .compute_von_mises_stress('stress')   # source array name in the VTK
       .convert('valve.usd')
   )

The default output array name is ``"von_mises_stress"``; the method finds
the source array in either ``point_data`` or ``cell_data`` and writes the
scalar back to the same data dict so the convert step picks it up as a
USD primvar.

Framing Camera and Light
------------------------

Every USD stage that ``ConvertVTKToUSD`` (and the lower-level
``convert_vtk_file`` facade) writes gets a ``/World/Camera`` prim with a
tight ``clippingRange`` sized to the geometry's bounding box. This avoids
the common "Omniverse Kit near-plane clips small geometry" problem for
medical-scale meshes (~0.03 m wide). Alongside it, ``add_framing_camera``
authors a ``/World/DistantLight`` sharing the camera's transform — a
DistantLight emits along its local -Z, the same direction the camera looks,
so the anatomy is lit from the viewing direction in stages that carry no
other light. The light is created only when valid, non-degenerate bounds
exist; otherwise no light is authored. The camera is also baked into stages
produced by ``TransformTools.convert_transform_to_usd_visualization`` and
``USDTools.merge_usd_files``; the helper is idempotent so re-merging a USD
that already has a Camera does not produce a duplicate transform op.

Anatomy Materials with USDAnatomyTools
=======================================

:class:`monai_physio.USDAnatomyTools` applies OmniSurface materials to
labeled meshes after conversion. It reads :class:`AnatomyTaxonomy` from the
segmenter to find which prim names map to which group, and looks up the
material parameters in its ``render_params`` dict (initialized from the
module-level :data:`monai_physio.usd_anatomy_tools.DEFAULT_RENDER_PARAMS`).

.. code-block:: python

   from monai_physio import USDAnatomyTools

   tools = USDAnatomyTools(stage)
   tools.enhance_meshes(seg)
   stage.Export('painted.usd')

Add a custom anatomy look without subclassing by mutating either the
module-level defaults (affects future instances) or a single instance:

.. code-block:: python

   from monai_physio.usd_anatomy_tools import DEFAULT_RENDER_PARAMS

   DEFAULT_RENDER_PARAMS["brain"] = {
       "name": "Brain",
       "diffuse_reflection_color": (0.85, 0.75, 0.7),
       # ... full parameter list mirrors the existing entries ...
   }

   # ...or per-instance:
   tools.render_params["brain"] = {...}

``enhance_meshes`` falls back to ``render_params["other"]`` for any group
without a registered entry, so newly registered groups still render — just
with the generic "other" look until a dedicated entry is added.

Advanced Low-Level Facade
=========================

Use the low-level facade only when the high-level class is not appropriate:

.. code-block:: python

   from monai_physio.vtk_to_usd import convert_vtk_file

   stage = convert_vtk_file('mesh.vtp', 'mesh.usd')
