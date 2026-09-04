====================================
Landmark Tools
====================================

.. currentmodule:: monai_physio

Landmark reading, writing and comparison utilities. Landmarks are the
independent check on a registration: transform a fixed-image landmark set with
the recovered transform and measure the distance to the corresponding
moving-image landmarks — the metric ``WorkflowFinetuneICONRegistration``
reports and the DIR-Lab benchmark is scored on.

Module Reference
================

.. automodule:: monai_physio.landmark_tools
   :members:
   :undoc-members:

See Also
========

* :doc:`transform_tools`
* :doc:`../registration/index`

.. rubric:: Navigation

:doc:`transform_tools` | :doc:`index` | :doc:`contour_tools`
