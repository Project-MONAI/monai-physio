=====================================
Physics-Informed Motion (Neo-Hookean)
=====================================

.. module:: physiotwin4d.train_physicsnemo_physics_informed_motion
.. currentmodule:: physiotwin4d

A MeshGraphNet trained on displacement alone has no opinion about whether the
motion it predicts is motion tissue could undergo. An element may inflate, thin
past what myocardium allows, or invert outright, and an L2 loss notices only to
the extent that the vertices land in the wrong place. This module adds a
neo-Hookean strain energy to that loss, which prices exactly those deformations,
and exposes the Cauchy stress the same constitutive law implies.

The energy is

.. math::

   W = \frac{\mu}{2}(I_1 - 3) - \mu \ln J + \frac{\lambda}{2} (\ln J)^2

with :math:`I_1 = \operatorname{tr}(F^T F)` and :math:`J = \det F`, evaluated
from the deformation gradient of each tetrahedron. Spatial derivatives come
from PhysicsNeMo Sym's least-squares gradient reconstruction, its method for
unstructured meshes.

The two appearances of :math:`J` are not the same quantity in code. An inverted
element makes :math:`\det F` non-positive, and :math:`\ln J` would then poison
the whole loss with a NaN, so the logarithmic terms use
:math:`\max(J, 10^{-6})`: an inversion costs a large finite penalty and stays
trainable rather than ending the run. The incompressibility penalty
:math:`(J - 1)^2` uses the raw determinant, which is signed and therefore
already prices an inversion correctly.

That clamp is also why the inversion count matters. It keeps an inverted element
finite, which is exactly what would let one pass unnoticed, so
``PhysicsInformedMotion.inverted_element_count`` reports the unclamped
determinant's non-positive entries and is the only signal that the predicted
motion turned tissue inside out.

Requirements
============

The shape model must be **volumetric**: a strain energy needs volume elements,
and the template's own cells are what supply them. A surface model has no
interior and no deformation gradient can be formed on it. Tutorial 16 builds
such a model; see :doc:`../../tutorials`.

``physicsnemo.sym`` supplies ``PhysicsInformer`` and ships inside
``nvidia-physicsnemo``, so no separate install is needed. It is imported lazily.

Training method
===============

.. autoclass:: TrainPhysicsNeMoPhysicsInformedMotion
   :members:
   :undoc-members:
   :show-inheritance:

Constitutive law and geometry
=============================

.. automodule:: physiotwin4d.train_physicsnemo_physics_informed_motion
   :members: NeoHookeanResidual, PhysicsInformedMotion, neo_hookean_pde,
             compute_deformation_gradient, tet_volumes, tet_edges, edge_matrix
   :undoc-members:

Notes
=====

**The reference configuration is the subject's own fit, not the mean.** The
stored targets are ``phase.points - fitted_reference.points``, so the fitted
reference is the undeformed state. Measuring the residual against the
population mean instead would charge every subject a strain energy for merely
being shaped unlike the mean, confusing variation between subjects with
deformation within one.

**Two formulations of one law.** The symbolic energy
(:func:`~physiotwin4d.train_physicsnemo_physics_informed_motion.neo_hookean_pde`)
is what PhysicsNeMo Sym differentiates during training; the tensor one
(:class:`~physiotwin4d.train_physicsnemo_physics_informed_motion.NeoHookeanResidual`)
computes the Cauchy stress for export, which the symbolic path does not hand
back. ``tests/test_physics_informed_motion.py`` cross-checks them against each
other on the same field.

**Loss scale.** The data term is scored on normalized displacement and the
physics term in millimeters and kilopascals, so ``lambda_physics`` is a value to
sweep rather than one to trust. The two components are accumulated separately so
they can be reported apart.

See Also
========

* :doc:`train`
* :doc:`evaluate`
