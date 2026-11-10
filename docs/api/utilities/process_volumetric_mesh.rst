====================================
Volumetric Mesh Tools
====================================

.. currentmodule:: monai_physio

Tetrahedral meshes filling each label of a labelmap, their relaxation onto a surface, and the repair of inverted elements.

Overview
========

:class:`ProcessVolumetricMesh` offers three meshers that fill each label of
a labelmap with its own tetrahedral mesh; each method's suffix names the
mesher it uses (``_vtk``, ``_netgen``, ``_simpleware``).  Like
:meth:`ProcessSurfaceMesh.extract_label_surfaces`, all three take the
labelmap, an optional ``element_size_mm`` and optional ``label_ids``, and
return ``{label id: pv.UnstructuredGrid}`` - positively oriented ``TETRA``
cells in the labelmap's world frame, each grid tagged with
``field_data['SegmentationLabelIds']`` - or ``{}`` for an empty labelmap, so
switching meshers is a one-word change.  A binary mask yields a single mesh,
under its foreground value.  Neighboring labels' meshes meet on coincident
nodes along their shared walls (each mesh holds its own copy of them).  The
netgen and Simpleware meshers also take ``smoothing_sigma_mm``, the blur that
sets how smooth their boundaries are, with the same default; the VTK mesher's
boundaries come from the surfaces they are trimmed to instead.  Tutorial 1's
tetmesh variant
(``tutorials/tutorial_01_convert_to_usd_tetmesh_slicer_heart_4d_ct.py``)
builds, warps, and saves all three side by side, with one USD prim per
label.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - :meth:`~ProcessVolumetricMesh.extract_label_tetrahedra_vtk`
     - :meth:`~ProcessVolumetricMesh.extract_label_tetrahedra_netgen`
     - :meth:`~ProcessVolumetricMesh.extract_label_tetrahedra_simpleware`
   * - Method
     - Every voxel becomes a hexahedron that VTK splits into six tetrahedra
     - netgen fills each label's smooth surface from
       :meth:`ProcessSurfaceMesh.extract_label_surfaces`
     - Synopsys Simpleware Medical +FE Free meshes each label's mask
   * - Boundary
     - Voxel staircase; relax it with
       :meth:`~ProcessVolumetricMesh.trim_tetrahedra_to_surface`
     - Smooth, as given
     - Smooth: each mask is Gaussian-blurred in Simpleware before meshing
   * - Elements
     - Uniform, structured
     - Unstructured, graded
     - Unstructured, graded
   * - Requirements
     - GPU (when resampling)
     - GPU, the ``netgen`` extra
     - A licensed Simpleware Medical install with +FE Free
   * - Use when
     - Element-for-element correspondence across subjects matters, as for
       statistical shape models
     - An open-source, unstructured mesh is wanted
     - Simpleware's mesh quality controls are wanted, or it is already in the
       pipeline

Labels with stray voxels or enclosed cavities give every mesher extra
boundary shells; clean each label's mask first with
:meth:`ProcessLabelmaps.remove_islands` and :meth:`ProcessLabelmaps.fill_holes`
(netgen in particular fails when such shells crowd the outer surface).

Voxel-based meshes
------------------

.. code-block:: python

   from monai_physio import ProcessSurfaceMesh, ProcessVolumetricMesh

   surfaces = ProcessSurfaceMesh().extract_label_surfaces(labelmap)
   volume_tools = ProcessVolumetricMesh()
   meshes = volume_tools.extract_label_tetrahedra_vtk(labelmap, element_size_mm=1.5)
   meshes = {
       label_id: volume_tools.remove_boundary_slivers(
           volume_tools.trim_tetrahedra_to_surface(mesh, surfaces[label_id])
       )
       for label_id, mesh in meshes.items()
   }

Trimming moves each label's mesh on its own, so the nodes neighboring labels
share then match only as closely as their surfaces do.

netgen meshes
-------------

Install the optional extra first (source installs through ``dev_cuda12`` or
``dev_cuda13`` already include it):

.. code-block:: bash

   uv pip install "monai-physio[netgen]"

.. code-block:: python

   meshes = ProcessVolumetricMesh().extract_label_tetrahedra_netgen(
       labelmap, element_size_mm=2.0
   )

``element_size_mm`` sets both the grid the surface is contoured on and
netgen's maximum interior element size.  The contour's smoothing grows with
it, so coarse sizes shrink thin structures: on the 1.5 mm heart test labels a
3 mm size loses about 5% of the voxel volume, the mask's own spacing about 1%.
``None`` contours at the labelmap's finest spacing and lets netgen grade the
interior from the surface.

``shared_interfaces`` picks how the surfaces reach netgen, as for Simpleware.
``True`` (default) hands netgen the contoured surfaces as they are: they are
triangulated identically where labels meet and netgen keeps a boundary's
triangles, so neighboring labels' meshes share those nodes.  ``False`` has
netgen re-tile each label's surface on its own, to triangles about
``element_size_mm`` across, before filling it; the meshes are then
independent and may overlap or gap slightly where labels meet.  netgen's
re-tiling ends its whole process on a surface that intersects itself, so it
runs in a child Python process and such a failure is raised as a
``RuntimeError``.

``contour_spacing_mm`` sets the grid the surfaces are contoured on, which
otherwise is ``element_size_mm``.  Where a label's voxels are finer than that
grid, its surface carries steps of the grid's size, so a finer grid gives
smaller steps.  With shared interfaces netgen keeps every contoured triangle:
on the full Slicer heart labelmap at 2 mm elements, a 1 mm grid grows the
meshes from about 420,000 to 1,750,000 cells.  Without shared interfaces the
re-tiling brings the triangles back to the element size, about 760,000 cells
from a 1 mm grid.

.. code-block:: python

   # Smaller steps, shared interface nodes, four times the cells.
   meshes = ProcessVolumetricMesh().extract_label_tetrahedra_netgen(
       labelmap, element_size_mm=2.0, contour_spacing_mm=1.0
   )
   # Smaller steps at about the element size, independent meshes.
   meshes = ProcessVolumetricMesh().extract_label_tetrahedra_netgen(
       labelmap, element_size_mm=2.0, contour_spacing_mm=1.0,
       shared_interfaces=False,
   )

``smoothing_sigma_mm`` is passed to
:meth:`ProcessSurfaceMesh.extract_label_surfaces` as ``distance_sigma_mm``:
``None`` (default) blurs the distance map by the contour grid spacing and ``0``
turns the blur off.  The surface is resampled and Taubin-smoothed either way,
so the blur mostly trades volume for smoothness: on the 1.5 mm heart test
label at 3 mm elements, no blur keeps 97% of the voxel volume, the default
95%, and 6 mm 91%.

``smoothing_iterations`` (default 30) is the surfaces' Taubin smoothing.  The
labels' surfaces are smoothed together, which can pinch a thin wall between
two labels - such as the heart wall between the contrast-filled blood pool and
the background - until a surface touches itself, and netgen rejects such a
surface.  ``0`` avoids that, but leaves irregular triangles (roughness about
0.34 against 0.08 on the Slicer heart labelmap at 2 mm) and, with them,
thousands of sliver elements that a warp inverts beyond repair.  Fewer
iterations without the blur balance the two: with ``smoothing_iterations=10``
and ``smoothing_sigma_mm=0`` the full Slicer heart labelmap's shared 2 mm
surfaces leave no element below a scaled Jacobian of 0.08 once
:meth:`~ProcessVolumetricMesh.remove_boundary_slivers` has run.  Tutorial 1's
tetmesh variant adds ``contour_spacing_mm=1.0`` and
``shared_interfaces=False`` (below) for smaller voxel steps: about 760,000
cells, whose 21 warped phases needed one inverted element repaired.

Simpleware meshes
-----------------

.. code-block:: python

   volume_tools = ProcessVolumetricMesh()
   # Only needed when Simpleware is not at the default install path.
   volume_tools.set_simpleware_executable_path(
       "C:/Program Files/Synopsys/Simpleware Medical/Y-2026.03/"
       "ConsoleSimplewareMedical.exe"
   )
   meshes = volume_tools.extract_label_tetrahedra_simpleware(
       labelmap, element_size_mm=2.0
   )

The labelmap is cropped, padded clear of its blur, and handed to
``ConsoleSimplewareMedical.exe`` in one run, which executes
``simpleware_medical/SimplewareScript_tetrahedral_mesh.py``: each label
becomes a mask blurred with Simpleware's recursive Gaussian filter, the masks
become +FE Free model parts, ``element_size_mm`` becomes the maximum edge
length (with half of it as the target minimum; ``None`` keeps Simpleware's
default sizes), and the linear tetrahedra are exported as VTK volumes.
``shared_interfaces`` picks the model layout: ``True`` (default) meshes every
label as a part of one model, so neighboring labels' meshes share their
interface nodes, and Simpleware's part priority settles where blurred masks
overlap; ``False`` meshes each label in a model of its own, so the meshes are
independent and may overlap or gap slightly where labels meet.  Simpleware
receives the labelmap with an identity direction and a zero origin, and its
points are mapped back through the labelmap's own origin and direction, so
the result does not depend on how Simpleware reads image orientation.
``timeout_s`` (default 600) bounds the external process.

The blur is what keeps the voxel staircase out of the meshes: a 0/1 mask holds
no sub-voxel boundary position, and +FE Free meshes it as given.
``smoothing_sigma_mm`` sets its standard deviation; ``None`` (default) uses
``element_size_mm``, or the labelmap's finest spacing when that is ``None``
too - the same blur netgen's surfaces get - and ``0`` turns it off.  On the 1.5 mm
Slicer heart mask at 2 mm elements:

.. list-table::
   :header-rows: 1

   * - ``smoothing_sigma_mm``
     - Boundary roughness
     - Mean distance to the smooth surface
     - Volume vs. voxels
   * - 0 (staircase)
     - 0.129
     - 1.50 mm
     - 0.993
   * - 1.5
     - 0.097
     - 0.24 mm
     - 0.982
   * - 3.0
     - 0.079
     - 0.31 mm
     - 0.960
   * - netgen, for comparison
     - 0.069
     - 0.22 mm
     - 0.973

Roughness is a boundary point's mean offset from its neighbors' centroid, per
mean edge length.  Larger blurs smooth further but thin small structures.

Simpleware's own smart mask smoothing
(``Model.SetUseSmartMaskSmoothing``) did less: 10 iterations reached a
roughness of 0.112.

Mesh quality and repair
=======================

:meth:`~ProcessVolumetricMesh.remove_boundary_slivers` drops flat boundary
caps before a mesh is warped, and
:meth:`~ProcessVolumetricMesh.repair_inverted_tetrahedra` relaxes the nodes of
elements a warp has inverted.  Both work on meshes from any of the three
meshers.

Module Reference
================

.. automodule:: monai_physio.process_volumetric_mesh
   :members:
   :undoc-members:

.. rubric:: Navigation

:doc:`process_surface_mesh` | :doc:`index` | :doc:`process_mesh_kernels`
