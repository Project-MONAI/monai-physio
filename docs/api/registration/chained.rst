===========================
Chained Image Registration
===========================

.. module:: monai_physio.register_images_chain
.. module:: monai_physio.register_images_greedy_icon
.. currentmodule:: monai_physio

Coarse-to-fine registration composes two registrars: a fast, robust method
recovers the large motion, then a deformable method refines it.
``RegisterImagesChain`` is the general composition;
``RegisterImagesGreedyICON`` is the Greedy-then-ICON pairing, used by Tutorial 2
and by the distance-map stage of the statistical-model fit.

Both implement :class:`RegisterImagesBase`, so they drop into any workflow that
takes a ``registration_method``.

Class Reference
===============

.. autoclass:: RegisterImagesChain
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RegisterImagesGreedyICON
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
===========

.. code-block:: python

   from monai_physio import RegisterImagesGreedyICON

   registrar = RegisterImagesGreedyICON()
   # Coarse-to-fine iteration schedule for the Greedy stage.
   registrar.greedy.set_number_of_iterations([30, 15, 7, 3])
   # Mass preservation suits non-contrast CT; leave it off for contrast.
   registrar.icon.set_mass_preservation(True)

The two stages are reachable as ``.greedy`` and ``.icon``, so each is tuned
with its own setters rather than a merged parameter set.

See Also
========

* :doc:`greedy`
* :doc:`icon`
* :doc:`time_series`
