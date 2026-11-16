Rigid aorta: 1D blood flow
==========================

A quick, transparent estimate of aortic pressure and mean velocity: the
Tutorial 13 aorta is treated as a rigid tube driven by the left-ventricle
(LV) volume change. It shares its geometry cuts, inlet waveform and outlet
model with the 3D experiment in ``../BloodFlow-3D-Rigid-Warp``. Both prepare
the Tutorial 13 lumen and LV flux with ``aorta_lumen.py`` in this folder and
smooth the inflow with ``inlet_flow`` from
``src/monai_physio/warp_navier_stokes.py``.

What it computes
----------------

* **Geometry.** The source aorta is frozen at sample 69 (about 70% of the
  recorded motion). A radius-weighted path through the lumen joins the
  centroids of the root cut and the distal cut used by the 3D experiment;
  the distal cut is 10 mm above the inferior CT crop, clear of its rounded
  closure. Each section area is the closed plane section of the source
  surface normal to the path (255 sections over 380 mm).
* **Inlet flow.** Each 0.03 s source interval contributes its LV volume loss
  (an ideal valve: filling intervals carry no flow). Three beats are retimed
  to 3 s (60 bpm). The resulting staircase is smoothed by a periodic Gaussian
  (sigma 0.03 s), which keeps Q >= 0 and preserves every stroke volume
  (35.9, 36.1 and 36.2 mL).
* **Model.** In a rigid tube the flow is the same at every section and the
  mean velocity is Q/A. Warp evaluates, in float64 on the GPU,

  .. math::

     p(s,t) = P_\mathrm{out}(t) + \dot{Q}\int_s^L \frac{\rho}{A}\,ds
              + Q\int_s^L \frac{8\pi\mu}{A^2}\,ds .

  ``P_out`` comes from a three-element Windkessel with the whole-body
  resistance 3.33e8 Pa s/m3 (6% proximal) and compliance 6.8e-9 m3/Pa,
  starting at 80 mmHg: the totals of the 3D experiment's four outlets.
  Blood density is 1060 kg/m3 and viscosity 0.004 Pa s.

How to run
----------

From the repository root, in an environment with this package installed and
a CUDA GPU::

  python experiments/BloodFlow-1D-Rigid-Warp/prepare.py --source PATH/heart_and_lung_motion.usd
  python experiments/BloodFlow-1D-Rigid-Warp/run.py
  python experiments/BloodFlow-1D-Rigid-Warp/view.py

Without ``--source`` the Tutorial 13 output named in ``config.json`` is used.
Results go to ``output/prepared`` and ``output/flow`` in this folder
(``--output`` selects other folders). The viewer serves
http://127.0.0.1:8701: play or scrub the 3 s, choose velocity or pressure,
change the anatomy opacity and rotate or zoom. ``--screenshot PATH`` saves
one frame instead, and this records the last beat (5.5 s at 30 fps) with
color limits fixed over that beat (MP4 output needs ``imageio-ffmpeg``,
part of the ``dev`` extra)::

  python experiments/BloodFlow-1D-Rigid-Warp/view.py --field pressure --movie experiments/BloodFlow-1D-Rigid-Warp/output/movies/aorta-1d-pressure-last-beat.mp4

Tests of all three blood-flow experiments (without ``--run-gpu`` the GPU
checks are skipped)::

  python -m pytest experiments/BloodFlow-1D-Rigid-Warp experiments/BloodFlow-3D-Rigid-Warp experiments/BloodFlow-3D-FSI-Simpleware-Warp --run-gpu

What it shows
-------------

The numbers in this section come from development runs of this code
(Windows, RTX PRO 6000 Blackwell Max-Q, Warp 1.17); their outputs are not
in the repository. ``run.py`` reproduces them, though timings are
machine-specific.

Last beat (2.01-3.00 s) of the 3 s run, which takes 0.6 s:

==========================================  =================
Root pressure, systolic / diastolic / mean  112 / 78 / 92 mmHg
Outlet (Windkessel) pressure                112 / 77 mmHg
Root-to-outlet pressure difference          +11.7 / -10.6 mmHg
Peak inlet flow                             157 mL/s
Peak section-mean velocity                  0.95 m/s (distal)
==========================================  =================

The root-to-outlet difference, largest in early systole and negative
during deceleration, is mostly the inertia of the 38 cm blood column. In a
development comparison, the unsmoothed staircase inflow gave a largest
difference of about 37 mmHg.

Limitations
-----------

* Rigid wall: no wall motion or pulse-wave propagation.
* All flow leaves through the trunk; there is no branch flow, and the single
  Windkessel stands for the whole circulation.
* No convective acceleration, velocity profiles or wall shear.
* The LV-derived stroke volume (36 mL) is about half a typical adult value,
  and 60 bpm is assumed.
* The run starts at 80 mmHg and the Windkessel time constant is 2.1 s, so the
  third beat is close to, but not exactly, periodic.
* Density and viscosity are typical values, not measurements.
