====================================
Labelmap Tools
====================================

.. currentmodule:: monai_physio

Convert segmentation labelmaps into binary registration masks, with optional
label exclusion and physically isotropic dilation, and clean binary masks.

Mask Cleanup
============

:meth:`~ProcessLabelmaps.remove_islands` drops the disconnected pieces of a
mask (ITK ``ConnectedComponentImageFilter`` then
``RelabelComponentImageFilter``): by default it keeps only the largest
face-connected component, or with ``min_island_volume_mm3`` every component
at least that large.  :meth:`~ProcessLabelmaps.fill_holes` fills the cavities
the mask encloses (ITK ``BinaryFillholeImageFilter``) and leaves openings to
the image border alone.  Both return a 0/1 ``itk.UC`` mask.

Run them before tetrahedral meshing: every island and enclosed cavity becomes
an extra boundary shell, and
:meth:`~ProcessVolumetricMesh.extract_label_tetrahedra_netgen` can fail on a
surface whose shells crowd each other.

.. code-block:: python

   from monai_physio import ProcessLabelmaps, ProcessVolumetricMesh

   tools = ProcessLabelmaps()
   solid = tools.fill_holes(tools.remove_islands(mask))
   meshes = ProcessVolumetricMesh().extract_label_tetrahedra_netgen(
       solid, element_size_mm=2.0
   )

Module Reference
================

.. automodule:: monai_physio.process_labelmaps
   :members:
   :undoc-members:

.. rubric:: Navigation

:doc:`index` | :doc:`process_images` | :doc:`process_transforms`
