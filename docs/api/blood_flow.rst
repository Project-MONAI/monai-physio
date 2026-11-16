====================================
Blood Flow (Warp)
====================================

.. currentmodule:: monai_physio

GPU blood-flow building blocks written with NVIDIA Warp FEM. They are
demonstrated by the ``experiments/BloodFlow-*`` folders: 1D and 3D flow in a
rigid aorta and two-way fluid-structure interaction with a compliant wall.
The solvers are demonstrations, not validated physiology.

* :class:`WarpNavierStokes` - P2/P1 incompressible Navier-Stokes on
  tetrahedra, with a cuDSS-preconditioned GMRES solve in float32 or float64.
* :class:`Windkessel` - three-element (RCR) outlet model.
* :func:`~monai_physio.warp_navier_stokes.inlet_flow` - smooth periodic inlet
  flow from per-interval fluxes, such as left-ventricle volume loss.
* :class:`WarpHyperelasticWall` - prestressed Neo-Hookean vessel wall.
* :class:`WarpFluidStructure` - Dirichlet-Neumann coupling with IQN-ILS
  acceleration.
* :class:`VisualizeFlowFields` - interactive viewer and movie export of the
  saved flow fields.

The pressure preconditioner loads the NVIDIA cuDSS library at run time; set
``CUDSS_LIBRARY`` to its path when it is not found automatically.

Module Reference
================

.. autoclass:: WarpNavierStokes
   :members:

.. autoclass:: Windkessel
   :members:
   :exclude-members: proximal_resistance, distal_resistance, compliance, pressure

.. autofunction:: monai_physio.warp_navier_stokes.inlet_flow

.. autofunction:: monai_physio.warp_navier_stokes.find_cudss_library

.. autoclass:: WarpHyperelasticWall
   :members:

.. autoclass:: WarpFluidStructure
   :members:

.. autofunction:: monai_physio.warp_fluid_structure.quasi_newton_update

.. autoclass:: VisualizeFlowFields
   :members:
