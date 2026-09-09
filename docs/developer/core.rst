======================
Core Developer Guide
======================

This page summarizes the core conventions for extending MONAI Physio.

Base Class
==========

Runtime classes inherit from :class:`monai_physio.MONAIPhysioBase` and use
the shared logging methods.

.. code-block:: python

   import logging

   from monai_physio import MONAIPhysioBase

   class MyWorkflow(MONAIPhysioBase):
       def __init__(self, input_file: str) -> None:
           super().__init__(class_name="MyWorkflow", log_level=logging.INFO)
           self.input_file = input_file

       def process(self) -> str:
           self.log_info("Processing %s", self.input_file)
           return self.input_file

Use ``log_info()``, ``log_debug()``, and ``log_warning()`` inside runtime
classes. Standalone scripts may use ``print()`` for command-line status.

Class Boundaries
================

* Workflows orchestrate complete pipelines.
* Registration classes estimate transforms.
* Segmentation classes return ITK images or dictionaries of ITK masks.
* ``TransformTools`` applies transforms to images and PyVista contours.
* ``ContourTools`` creates and transforms VTK/PyVista surface data.
* ``USDTools`` and ``USDAnatomyTools`` operate on USD stages and files.

Public APIs should be documented in ``docs/api``.

Validation
==========

For most code changes, run:

.. code-block:: bash

   py -m pytest tests/ -v

(Slow / GPU / Simpleware / PhysicsNeMo / tutorial tests are
auto-skipped; opt in with ``--run-slow``, ``--run-gpu``, ``--run-simpleware``,
``--run-physicsnemo``, ``--run-tutorials``, or use
``--run-all`` to enable every bucket at once. Data-dependent tests download
their data through the session fixtures and run by default.)

After public API changes, refresh the graphify knowledge graph so AI assistants
keep resolving symbols correctly (see :doc:`ai_assistants`):

.. code-block:: bash

   graphify update .

See Also
========

* :doc:`architecture`
* :doc:`workflows`
* :doc:`utilities`
