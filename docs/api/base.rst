====================================
Base Class
====================================

.. module:: monai_physio.monai_physio_base
.. currentmodule:: monai_physio

``MONAIPhysioBase`` provides the shared logging behavior used by workflow,
segmentation, registration, transform, contour, and USD helper classes.

Class Reference
===============

.. autoclass:: MONAIPhysioBase
   :members:
   :undoc-members:
   :show-inheritance:

Logging
=======

Runtime classes should call ``log_info()``, ``log_debug()``, and
``log_warning()`` instead of printing directly. The base class also supports
global log filtering by class name.

.. code-block:: python

   import logging

   from monai_physio import MONAIPhysioBase

   class MyProcessor(MONAIPhysioBase):
       def __init__(self) -> None:
           super().__init__(class_name="MyProcessor", log_level=logging.INFO)

       def process(self) -> None:
           self.log_info("Starting processing")
           self.log_debug("Detailed diagnostic state")
           self.log_warning("Recoverable issue")

   processor = MyProcessor()
   processor.process()

   MONAIPhysioBase.set_log_classes(["MyProcessor"])
   MONAIPhysioBase.set_log_all_classes()

Extension Notes
===============

New runtime classes should inherit from ``MONAIPhysioBase`` and pass a
``class_name`` plus ``log_level`` to ``super().__init__``. Standalone scripts,
data containers, and small pure utility functions do not need to inherit from
the base class.

See Also
========

* :doc:`workflows`
* :doc:`../developer/architecture`
* :doc:`../developer/extending`
