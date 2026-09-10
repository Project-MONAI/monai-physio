====================================
Landmark Tools
====================================

.. currentmodule:: monai_physio

Landmark reading, writing and comparison utilities. Landmarks are the
independent check on a registration: transform a fixed-image landmark set with
the recovered transform and measure the distance to the corresponding
moving-image landmarks - the metric ``WorkflowFinetuneICONRegistration``
reports and the DIR-Lab benchmark is scored on.

Module Reference
================

.. automodule:: monai_physio.process_landmarks
   :members:
   :undoc-members:

See Also
========

* :doc:`process_transforms`
* :doc:`../registration/index`

.. rubric:: Navigation

:doc:`process_transforms` | :doc:`index` | :doc:`process_contours`
