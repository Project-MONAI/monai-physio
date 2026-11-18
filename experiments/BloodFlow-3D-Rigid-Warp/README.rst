Rigid aorta: 3D blood flow with Warp FEM
========================================

A demonstration of three-dimensional, time-dependent blood flow in the
Tutorial 13 aorta with a rigid wall. It shares its geometry cuts, inlet
waveform and outlet totals with the 1D experiment in
``../BloodFlow-1D-Rigid-Warp``, whose ``aorta_lumen.py`` prepares the
Tutorial 13 lumen for both and whose ``prepare.py`` writes the LV inflow
``waveform.npz`` (named by ``waveform`` in ``config.json``). The solver, the
outlet model, the inlet waveform smoothing and the viewer are reusable library
code: ``WarpNavierStokes``, ``Windkessel``, ``inlet_flow`` and
``VisualizeFlowFields`` in ``src/monai_physio``.

What it computes
----------------

* **Lumen.** The source aorta is frozen at sample 69 and cut into a closed
  lumen with five openings: a root inlet (a truncation, not a valve), the
  brachiocephalic (BCT), left subclavian (LSA) and left common carotid (LCCA)
  stumps, and a distal outlet 10 mm above the inferior CT crop. Netgen fills
  it with 77,865 tetrahedra: 2 mm elements, refined to eight elements across
  each opening too narrow for that (``port_cells_across``; 0.38 mm at the
  LCCA stump, 0.61 mm at the LSA, 0.74 mm at the BCT) within 1.5 opening
  radii of its centre. The refinement uses
  ``ProcessVolumetricMesh.create_element_size_image`` and the
  ``element_size_image`` of ``extract_label_tetrahedra_netgen``.
* **Inlet.** The left-ventricle volume loss of each 0.03 s source interval
  (ideal valve), three beats in 3 s (60 bpm), smoothed by a periodic Gaussian
  (sigma 0.03 s) that keeps Q >= 0 and each stroke volume (about 36 mL). The
  root velocity profile is flat except within 20% of the radius of the wall.
* **Outlets.** Each outlet has a three-element Windkessel (RCR), coupled
  implicitly, with outflow backflow stabilization (beta = 1). The whole-body
  resistance 3.33e8 Pa s/m3 and compliance 6.8e-9 m3/Pa are split
  distal/BCT/LSA/LCCA = 70/17/8/5% (resistance inversely, compliance
  directly); 6% of each resistance is proximal. All outlets start at
  80 mmHg.
* **Solver.** Quadratic velocity and linear pressure (P2/P1) on Warp FEM,
  backward Euler with a single-step semi-Lagrangian backtrace, dt = 5 ms,
  3 s from rest, in float32 (``precision`` in ``config.json`` or
  ``run.py --precision float64``). GMRES (relative tolerance 1e-5 in
  float32, where the attainable residual depends on the problem; 1e-6 in
  float64) uses a Schur-complement preconditioner factored once on the GPU
  by NVIDIA cuDSS.
  Blood density is 1060 kg/m3 and viscosity 0.004 Pa s.

How to run
----------

From the repository root, in an environment with this package installed and
a CUDA GPU. The solver needs NVIDIA cuDSS: install the ``nvidia-cudss-cu13``
wheel (or ``-cu12``) or set ``CUDSS_LIBRARY`` to the cuDSS library file.
The first line writes the inlet waveform::

  python experiments/BloodFlow-1D-Rigid-Warp/prepare.py --source PATH/heart_and_lung_motion.usd
  python experiments/BloodFlow-3D-Rigid-Warp/prepare.py --source PATH/heart_and_lung_motion.usd
  python experiments/BloodFlow-3D-Rigid-Warp/run.py
  python experiments/BloodFlow-3D-Rigid-Warp/view.py

Without ``--source`` the Tutorial 13 output named in ``config.json`` is used.
Results go to ``output/prepared`` and ``output/flow`` in this folder;
``run.py --dt``, ``--precision`` and ``--output`` run other cases. The
viewer serves http://127.0.0.1:8702: play or scrub time, show speed or
pressure on the lumen and on a movable slice, add velocity arrows, pick the
color range and rotate or zoom. ``--screenshot PATH`` saves one frame, and
this records the last beat off screen (MP4 output needs ``imageio-ffmpeg``,
part of the ``dev`` extra)::

  python experiments/BloodFlow-3D-Rigid-Warp/view.py --movie experiments/BloodFlow-3D-Rigid-Warp/output/movies/rigid-aorta-pressure.mp4

``compare_precision.py`` runs any missing float64 or float32 case of the
current configuration, compares both with each other and with a reference
run, and writes the timings and differences to ``OUT/precision-comparison``
as JSON and a Markdown table::

  python experiments/BloodFlow-3D-Rigid-Warp/compare_precision.py --float64 OUT/flow-float64 --float32 OUT/flow-float32 --reference OUT/flow-reference --output OUT/precision-comparison

Tests of all three blood-flow experiments::

  python -m pytest experiments/BloodFlow-1D-Rigid-Warp experiments/BloodFlow-3D-Rigid-Warp experiments/BloodFlow-3D-FSI-Simpleware-Warp --run-gpu

What it shows
-------------

The numbers in this section come from development runs of this code
(Windows, RTX PRO 6000 Blackwell Max-Q, Warp 1.17); their outputs are not
in the repository. ``run.py`` reproduces them, though float32 runs are not
bit-reproducible and timings are machine-specific.

Last beat (2.01-3.00 s) of the 3 s run with the default configuration
(float32). On an otherwise idle GPU it takes 0.8 minutes (600
steps, 0.078 s each):

==========================================  =====================
Root pressure, systolic / diastolic / mean  113 / 77 / 92 mmHg
Change of mean root pressure from beat 2    -0.6 mmHg
Mean inlet flow                             36.3 mL/s
Outlet flow split distal/BCT/LSA/LCCA       70.0 / 17.0 / 8.0 / 5.0%
Peak speed more than 10 mm from an outlet   0.81 m/s
Peak speed anywhere                         3.2 m/s (LCCA cap)
Net flow through all ports                  2e-7 of peak inflow
==========================================  =====================

The root pressure exceeds the distal-outlet pressure by up to 10.3 mmHg in
early systole and falls 7.7 mmHg below it during deceleration (inertia; the
1D experiment gives +11.7 / -10.5 mmHg). Each outlet receives its assigned
share because the Windkessel resistances dominate the lumen. The fastest
flow is at the small branch caps: 3.2, 2.8 and 2.1 m/s at the LCCA, BCT and
LSA caps. These cap speeds depend on the mesh there: refined meshes of
nearly the same surface gave 2.2-3.2 m/s at the LCCA. The movie command
above records the last beat: pressure on the lumen and on a slice 10 mm
above the root, with velocity arrows, and color limits fixed over the beat
(76.5-113.8 mmHg).

**Port refinement.** The LCCA stump's source cap is 7.2 mm2. Without the
refinement, the uniform 2 mm mesh tiled it with two triangles (4.5 mm2) or,
after a 20-50 um change of the Tutorial 13 surface, one (3.5 mm2). The
solver holds every velocity node of the wall at zero, so two triangles left
one free node, which carried the whole 5% share (5.3 m/s, a numerical
artifact), and one triangle left none: the LCCA then carried no flow and the
root pressure rose to 118 / 80 mmHg. With the refinement the branch caps
have 70-96 triangles and their areas are within 3.2% of the source (the
unrefined root and distal caps have 696 and 492, the distal one 5.8% under
its source area). Refined meshes of the two Tutorial 13 surfaces gave root
pressures within 0.2% and flow shares within 0.01 points, as close as their
inflows (0.2%).

Limitations
-----------

* Rigid wall: no wall motion or pulse-wave propagation.
* Time and mesh convergence are not established.
* The arch branches are short stumps. Their caps are resolved by eight
  elements across, but speeds next to the branch outlets vary by up to about
  45% (LCCA 2.2-3.2 m/s) between refined meshes of nearly the same surface;
  they are not mesh converged.
* The outlet totals are not typical adult values. The resistance,
  3.33e8 Pa s/m3 = 2.5 mmHg s/mL, is about twice a typical total
  peripheral resistance; it offsets the cardiac output of about 2.2 L/min,
  half the normal value, so that the pressures look normal. They are split by
  assumed fractions, not patient measurements. The run starts at 80 mmHg and
  the outlet time constant is 2.1 s, so the third beat is close to, but not
  exactly, periodic.
* The LV-derived stroke volume (36 mL) is about half a typical adult value,
  and 60 bpm is assumed. Density and viscosity are typical values.
