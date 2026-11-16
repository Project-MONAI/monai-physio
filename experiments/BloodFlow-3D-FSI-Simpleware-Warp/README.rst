Compliant aorta: two-way blood flow and wall motion
===================================================

A demonstration of basic fluid-structure interaction (FSI) in an aorta: the
blood pressure pushes an elastic wall outward, and the moving wall changes the
blood flow. It reuses the flow model of ``../BloodFlow-3D-Rigid-Warp``
(experiment 2) on the Simpleware candidate-20 blood and wall meshes. It is a
demonstration, not validated physiology. The solvers are library classes:
``WarpNavierStokes`` and ``Windkessel`` (flow), ``WarpHyperelasticWall`` and
``WarpFluidStructure`` (wall and coupling) and ``VisualizeFlowFields``
(viewer) in ``src/monai_physio``.

What it models
--------------

* **Meshes.** Candidate 20 holds a blood mesh (342,795 linear tetrahedra) and
  a wall mesh (294,506 tetrahedra, about 1.5 mm thick) that share 83,504
  interface triangles. ``prepare.py`` reads only its ``geometry.npz`` and
  ``waveform.npz``. The ports (root inlet; BCT, LSA and LCCA stumps; distal
  outlet) are recomputed from the face regions. The distal cap lies about
  1.6 mm above the CT crop, so the lumen there differs from experiment 2.
* **Blood.** As experiment 2: P2/P1 Warp FEM in float32, backward Euler with
  a semi-Lagrangian backtrace, dt = 2.5 ms, GMRES relative tolerance 1e-5
  (``precision`` and ``gmres_tolerance`` in ``config.json``; ``null`` uses
  the precision's default, 1e-5 in float32 and 1e-6 in float64, and
  ``run.py --precision float64`` selects float64). Float32 residuals stall
  near 1e-6, so 1e-5 is the tightest reliably reachable value, and the
  coupling still converges with it (see below). The smoothed LV inflow
  (about 36 mL per beat, 60 bpm) enters with a blunt root profile, and
  three-element Windkessel (RCR) outlets carry backflow stabilization.
  Totals 3.55e8 Pa s/m3 and 2.2e-9 m3/Pa (see *Outlet calibration*) are
  split distal/BCT/LSA/LCCA = 70/17/8/5%, 6% proximal.
  The blood mesh moves with the wall (arbitrary Lagrangian-Eulerian): the wall
  displacement is extended harmonically into the lumen once per step.
* **Wall.** Quasi-static (massless) compressible Neo-Hookean solid on
  quadratic tetrahedra, E = 1 MPa, nu = 0.45. The imaged geometry is taken as
  the loaded state at 80 mmHg: a prestress that balances 80 mmHg on the
  imaged wall is computed once (one small-strain solve). The wall collar at
  each port (the wall vertices at or beyond the cap plane, about 1.6 mm deep)
  is clamped, so the ports do not move. Weak springs (5e5 Pa/m) on the outer
  surface stand in for the surrounding tissue. The wall and the mesh
  extension always run in float64: evaluated in float32 at 120 mmHg, the
  wall's internal force is off by 1.3e-4 of the load, far above the 1e-6
  at which its Newton solve stops, and the extension takes only 6 ms per
  step.
* **Coupling.** Each step iterates Dirichlet-Neumann: the blood takes the
  wall velocity, the wall takes the consistent blood force on the shared
  nodes, and an interface quasi-Newton method (IQN-ILS, Degroote et al.
  2009) updates the interface displacement. It fits a least-squares inverse
  Jacobian to the residual and wall-response differences of the current
  step and the previous ``reuse`` = 2 steps (QR on the GPU, nearly dependent
  columns dropped), takes the quasi-Newton step for the part of the
  residual that this model explains and relaxes the rest by 0.1
  (``coupling`` in ``config.json``). Standard IQN-ILS, which relaxes only
  the first iteration and none of the rest afterwards, inverts the light
  wall in the third iteration of the tube check's first step. A step
  converges when the RMS interface residual is at most 1e-3 times the RMS
  interface motion of the step plus 1e-9 m. The reported coupling residual
  is the relative part alone, so steps with little motion converge, through
  the 1 nm term, while reporting more than 1e-3. A step still unconverged
  after 20 iterations is capped: it is solved once more at its
  lowest-residual iterate, which it keeps, and the quasi-Newton columns
  are reset to those of the earlier steps (see *Limitations*). A capped
  step reports 21 iterations, and its re-solve may meet the criterion, so
  ``run.json`` counts capped steps (``capped_steps``) and steps left
  unconverged (``nonconverged_steps``) separately. Neither is rejected.

Outlet calibration
------------------

With experiment 2's outlet totals (3.33e8 Pa s/m3, 6.8e-9 m3/Pa) the root
pulse pressure here was 26 mmHg instead of 36 mmHg, because the wall adds
compliance. In that run (the Aitken float64 run under *Development runs*)
the lumen volume follows the root pressure with dV/dP = 2.08e-9 m3/Pa
(0.28 mL/mmHg, linear fit over beat 2).

The totals above were chosen with a 0D model: this wall compliance
``C_w`` at the root, behind it one ``Windkessel`` with the outlet totals
``R`` and ``C`` (6% of ``R`` proximal, ``R_p``, the rest distal, ``R_d``;
the split outlets share one time constant):

.. math::

   C_w \frac{dP}{dt} = Q_\mathrm{in} - Q_\mathrm{out}, \qquad
   P = R_p Q_\mathrm{out} + P_d, \qquad
   C \frac{dP_d}{dt} = Q_\mathrm{out} - \frac{P_d}{R_d}.

It is driven by the inflow of the 3D runs, stepped with backward Euler at
dt = 2.5 ms (implicit in the outlet flow, as in the 3D solver) for two beats
from 80 mmHg, and summarized over 1.02-2.01 s like the 3D runs. Root
pressure, systolic / diastolic / mean (pulse), in mmHg, from this 0D
development calculation and from the 3D development runs:

================  =========  ==========================  =========================
Totals R, C       C_w        0D model                    3D run
================  =========  ==========================  =========================
3.33e8, 6.8e-9    none       112.7 / 77.2 / 92.5 (35.5)  experiment 2: pulse 36.3 (36.6 at the distal outlet)
3.33e8, 6.8e-9    2.08e-9    105.7 / 79.5 / 90.9 (26.2)  Aitken float64 run: 106 / 80 / 91 (26)
3.55e8, 2.2e-9    2.08e-9    118.5 / 77.4 / 96.8 (41.1)  this experiment: 117.0 / 78.0 / 97.2 (38.9)
================  =========  ==========================  =========================

The last row lowers the outlet compliance to 2.2e-9 m3/Pa and raises the
total resistance 7% to 3.55e8 Pa s/m3, which lifts the mean from about 91
to 97 mmHg. The outlet time constant R_d C is then 0.73 s, and with the wall
R (C + C_w) is about 1.5 s.

How to run
----------

The input is the Simpleware candidate-20 export, meshing run
``POC3-S1-r20``, of which ``prepare.py`` reads only ``geometry.npz`` (SHA-256
``3f011c48...79af3``) and ``waveform.npz`` (``855e0322...d09dd``). It is not
in Git and cannot be regenerated from this repository alone: it was made on
2026-11-11 by Simpleware meshing scripts that are not in this repository and
that also read surfaces and label maps of earlier meshing runs. Place that
folder at the ``source`` named in ``config.json`` (relative to the
repository root) or pass ``--source``.

From the repository root, with a CUDA GPU and NVIDIA cuDSS (the
``nvidia-cudss-cu13`` wheel, or ``CUDSS_LIBRARY`` set to the library file)::

  python experiments/BloodFlow-3D-FSI-Simpleware-Warp/prepare.py --source PATH/POC3-S1-r20
  python experiments/BloodFlow-3D-FSI-Simpleware-Warp/run.py
  python experiments/BloodFlow-3D-FSI-Simpleware-Warp/view.py

``prepare.py`` writes ``output/prepared`` and ``run.py`` writes
``output/fsi`` in this folder (``run.json`` summarizing beat 2, per-step
``steps.npz`` and the saved fields); ``run.py --steps N`` runs a short smoke
test and ``--output`` and ``--precision`` run other cases. The viewer serves
http://127.0.0.1:8703 with the experiment 2 controls plus a *Wall
displacement* field; the lumen moves at true scale. ``--screenshot PATH``
saves one frame, and this records beat 2 off screen (MP4 output needs
``imageio-ffmpeg``, part of the ``dev`` extra)::

  python experiments/BloodFlow-3D-FSI-Simpleware-Warp/view.py --field displacement --movie experiments/BloodFlow-3D-FSI-Simpleware-Warp/output/movies/compliant-aorta-wall-displacement.mp4

Tests of all three blood-flow experiments, then the wall and coupling checks::

  python -m pytest experiments/BloodFlow-1D-Rigid-Warp experiments/BloodFlow-3D-Rigid-Warp experiments/BloodFlow-3D-FSI-Simpleware-Warp --run-gpu
  python -m pytest tests/test_warp_fluid_structure.py --run-gpu

The second command checks the wall and the coupling on a straight tube: the
inner-radius change under 100 Pa against the Lame thick-wall solution, zero
displacement once that load is taken as prestress, and the pulse-wave speed
from the quarter-wave ringing of a closed tube against Bramwell-Hill
corrected for viscosity (Womersley) and backward Euler.

What it shows
-------------

The numbers in this section come from development runs of this code
(Windows, RTX PRO 6000 Blackwell Max-Q, Warp 1.17); their outputs are not
in the repository. ``run.py`` reproduces them, though float32 runs are not
bit-reproducible and timings are machine-specific.

Beat 2 (1.02-2.01 s) of the default two-beat run, which starts at rest at
80 mmHg (float32 blood, IQN-ILS coupling, the outlet totals above). Its
time steps took 56 minutes on an otherwise idle GPU (804 steps, 4.3 s each
in beat 2), plus 26 s setup. ``run.json`` reports the capped and
unconverged step counts; the 1 nm count is inferred in a development
analysis of the per-step residuals, as converged steps whose relative
residual exceeds 1e-3:

==========================================  ==========================
Root pressure, systolic / diastolic / mean  117 / 78 / 97 mmHg
                                            (pulse 39 mmHg)
Mean inlet flow                             36.4 mL/s
Outlet flow split distal/BCT/LSA/LCCA       69.9 / 17.0 / 8.0 / 5.0%
Peak / mean wall displacement               0.99 / 0.19 mm
Lumen volume change over the beat           11.1 mL (8.2%)
Coupling iterations, mean / max             5.4 / 21
Capped steps (more than 20 iterations,      12 of 397, kept residuals
re-solved)                                  1.8e-3 to 4.1e-3; 11 left
                                            unconverged
1 nm term decided convergence (inferred)    266 of 386 converged steps
Median / largest coupling residual          1.3e-3 / 5.0e-3 (relative
                                            part)
Inflow - outflow - dV/dt                    at most 0.18% of peak inflow;
                                            0.014% over the beat
Largest GMRES relative residual             2.7e-6
==========================================  ==========================

The root pulse pressure, 38.9 mmHg, is close to the 40 mmHg target and to
the 0D prediction (118.5 / 77.4 mmHg, mean 96.8, pulse 41.1): the 3D run
gives 117.0 / 78.0 mmHg, mean 97.2. The lumen volume follows the root
pressure with dV/dP = 2.12e-9 m3/Pa, the wall compliance assumed in the
calibration (2.08e-9). The root pressure exceeds the distal-outlet pressure
by up to 7.4 mmHg in early systole and falls 4.3 mmHg below it during
deceleration. Beat 2 is not exactly periodic: the root pressure at the end
of the beats is 79.76 mmHg (1.00 s) and 79.48 mmHg (2.00 s), the beat
means are 98.16 and 97.24 mmHg, the two beats' inflow volumes differ
slightly (35.8 and 36.2 mL), and the 0D model, with the beat-2
inflow repeated 12 times to reach a periodic state, puts the two-beat
run's beat 2 about 0.3-0.4 mmHg (at most 0.5 mmHg) above that state, for wall compliances 2.08e-9
and 2.12e-9 m3/Pa. Of the 12 capped steps, 7 fall in late ejection and
deceleration (1.35-1.60 s, after the inflow peak at 1.25 s) and 5 in
diastole (1.695-1.98 s); one met the convergence test on its re-solve. A
13th step reached 20 iterations and converged on its 20th. The 1 nm term
decides most steps: the interface moves only micrometres per step, so
1e-3 of that motion is comparable to 1 nm. The Aitken development run
(see *Development runs*) used the same test (by the same inference, 286 of
its 348 converged steps needed the 1 nm term). ``steps.npz`` stores only
the relative residual, not the absolute residual or the step motion, so
the 1 nm counts are inferred and the pass or fail decision cannot be
recomputed from it. The median change of the mean
displacement magnitude, a proxy for the RMS interface motion, is 1.8 um
per step; for that motion the convergence bound (1e-3 of it plus 1 nm)
is about 3 nm, 3e-6 of the peak displacement, and the implied error in
the wall flux about 0.03 mL/s, 0.02% of peak inflow.

The wall bulges most in the ascending aorta and the arch, where the lumen is
widest, and least near the clamped collars. The movie command above shows
beat 2 with the wall displacement colored from 0 to its largest value in
the saved frames (0.98 mm), and velocity arrows on a slice 10 mm above the
root.

The wall is softer than a thin-wall estimate suggests. For E = 1 MPa, a 1.55
mm wall and a radius near 12 mm, a circular tube would bulge about 0.41-0.44
mm at the peak pulse and give dV/dP of about 1.5e-9 m3/Pa; the run gives
0.99 mm and 2.12e-9 m3/Pa, and a static wall solve at 120 mmHg in a
development calculation agrees (0.92 mm), so this is a property of the wall model and geometry, not of the
coupling. The non-circular, curved lumen is a plausible but unverified
cause. The resulting distensibility, 2.06e-3 per mmHg, and Bramwell-Hill
wave speed, 7.8 m/s, are plausible for an older, stiffer adult aorta.

On the straight-tube checks (``tests/test_warp_fluid_structure.py``) the
inner-radius change is within 0.5% of the Lame solution, and the
pulse-wave speed is 2.82 m/s (2.818-2.819, as the test prints it, in
repeated float32 runs), 0.8% below the expected 2.841 m/s:
Bramwell-Hill's 3.049 m/s lowered by viscosity (Womersley number 18) and
by backward Euler at 20 steps per period. The coupling needs 4 to 5
iterations per step there (4.2-4.5 in three test runs, at most 9);
Aitken needed 9.2 (at most 19). In a development run with 40 steps per
period (added-mass ratio about 40 instead of 10), relaxation 0.1 inverts
the wall in the first step; relaxation 0.05 converges every step (6.3
iterations, at most 16) and gives a wave speed 0.6% below the expected
2.909 m/s. The tube steps move enough that they test the relative
criterion, not the 1 nm term.

Development runs
----------------

The comparisons below come from development runs whose outputs are not in
the repository. Each is this experiment except for the stated differences;
the Aitken coupling and the midpoint backtrace came from earlier versions of
the code.

**Aitken float64 run**, two beats: Aitken coupling, float64 blood with
GMRES tolerance 1e-9 and an earlier midpoint backtrace, and experiment 2's
outlet totals. It took 11.6 hours (52 s per step in beat 2). Beat 2:

==========================================  ==========================
Root pressure, systolic / diastolic / mean  106 / 80 / 91 mmHg
Mean inlet flow                             36.4 mL/s
Outlet flow split distal/BCT/LSA/LCCA       69.9 / 17.0 / 8.0 / 5.0%
Peak / mean wall displacement               0.69 / 0.12 mm
Lumen volume change over the beat           7.2 mL (5.2%)
Coupling iterations, mean / max             13.6 / 20
Median coupling residual, all steps         2.0e-3
Steps stopped at 20 iterations              49 of 397 (median residual
                                            1.6e-2, largest 0.10)
Inflow - outflow - dV/dt                    at most 0.8% of peak inflow;
                                            0.01% over the beat
==========================================  ==========================

**Run before the capped-step handling**, two beats, same code and
configuration otherwise. A capped step then kept its last iterate and its
quasi-Newton columns (see *Limitations*):

===============================================  ==========  ==========
                                                 Before      Current
===============================================  ==========  ==========
Volume imbalance over both beats (mL)            -0.208      -0.011
Largest kept coupling residual, beat 2           0.068       5.0e-3
Largest step imbalance, beat 2 (of peak inflow)  1.41%       0.18%
Beat 2 steps at 20 iterations / unconverged      18 / 15     13 / 11
Beat 2 root pressure, mean (mmHg)                97.12       97.24
Beat 2 root pressure, diastolic (mmHg)           77.85       78.03
Peak wall displacement (mm)                      0.982       0.985
===============================================  ==========  ==========

The current beat-2 root pressures are 0.12-0.18 mmHg higher, as expected
from the volume the earlier run lost in a beat-1 glitch (see
*Limitations*); the pulse, split, peak displacement and dV/dP are
unchanged.

**Short windows** of the first 160, 60 or 44 steps (0-0.4 s for 160 steps,
from rest through the first systolic peak), float32 blood and the current
outlets unless stated:

===============================  ==========================================
Window                           Result
===============================  ==========================================
Aitken, 160 steps                17.3 / 20 coupling iterations (mean /
                                 max), 42 of 160 steps stopped at 20,
                                 11.5 s per step
IQN-ILS reuse 2, 160 steps       5.1 / 12 iterations, none stopped at 20,
                                 4.3 s per step (blood 43%, wall 45%,
                                 quasi-Newton update 1.4%); root
                                 pressures, port flows and peak wall
                                 displacements agree with Aitken to 3e-4,
                                 5e-4 and 9e-4 of the largest values
IQN-ILS reuse 4, 160 steps       5.2 iterations
IQN-ILS reuse 0/2/4/8, 60 steps  9.2 (2 steps stopped at 20), 5.5, 6.1
                                 and 4.9 iterations, so the default reuse
                                 is 2
IQN-ILS float64 blood, 60 steps  5.7 iterations (reuse 2); matches float32
                                 within 2e-5 in root pressure, 4e-5 in
                                 port flows and 3e-4 in peak wall
                                 displacement; 3.1 times longer per step
Pre-handling, 44 and 40 steps    the step after the first capped one ended
                                 at 0.69 and at 0.84 (the next at 12)
Current handling, 2 x 44 steps   one capped step between them (residual
                                 4.0e-3); the next step started with all 9
                                 columns of the step before the capped one
                                 and converged in 8 iterations; volume
                                 balance within about 2.1% of peak inflow,
                                 as in runs without capped steps
Current handling, 160 steps      no step capped; matches the reuse-2
                                 window to about 0.002 mmHg in root
                                 pressure, 2e-5 of the largest port flow
                                 and 5e-5 of the peak wall displacement
===============================  ==========================================

The IQN-ILS windows without the capped-step handling used the current
algorithm otherwise; the handling changes only steps that reach 20
iterations (two in the reuse-0 window, none in the others). The Aitken
window used an earlier version of the coupling module.

Limitations
-----------

* Demonstration only: no mesh or time convergence and no calibration to a
  patient. The wall is light and stiff compared with the blood it moves at
  dt = 2.5 ms; a smaller dt needs a smaller relaxation (see the tube check
  above). The outlet totals were chosen with a 0D model, so the 3D pulse
  pressure may differ from 40 mmHg by a few mmHg.
* The wall is massless and quasi-static, uniform in stiffness and thickness,
  and its prestress is a small-strain estimate. The clamps and tethers are
  modelling choices.
* The blood mesh moves once per step, to the predicted interface, with
  straight-sided elements, while the wall velocity follows the coupling
  iterate and the wall's quadratic edge nodes may curve. Both show up in
  the volume balance: up to 0.18% of peak inflow in beat 2 (0.81% in the
  Aitken run) and 2.1% in the first steps from rest, with either coupling
  method.
* The interface residual cannot fall below the noise of the inexact blood
  and wall solves (float32 GMRES stops at 1e-5, the wall's Newton solve at
  1e-6 of the load), about 1-10 nm RMS in the stalled iterations of the
  44-step development runs. Where the interface moves little, as in beat 1
  (RMS interface motion 0.1-0.6 um per step over 0.005-0.09 s in those
  runs; the saved proxy, the change of the mean displacement, is
  0.06-0.11 um near 0.08 s), that can exceed the 1.4 nm convergence
  threshold, and the step stalls until the 20-iteration cap. Before the
  capped-step handling (see *Coupling*), such a step kept its last iterate
  and up to 19 quasi-Newton columns of that noise, which the next two
  steps reused: their least-squares model fitted noise and they ended at
  relative residuals up to 12. The run before the handling capped
  0.080-0.085 s (residuals 6.2e-3, 0.45 and 10.7): the root pressure fell
  from 80.58 to 79.15 mmHg, the volume balance was off by -111% of peak
  inflow at 0.0875 s, 0.217 mL was lost over 0.080-0.090 s, and the
  pressures stayed about 0.3 mmHg low through 0.4 s. The current run
  capped 25 steps (13 before 1.02 s), and its root pressure rises smoothly
  from 80.46 to 81.14 mmHg over 0.080-0.095 s, with -0.0008 mL of volume
  imbalance over 0.080-0.090 s. Its largest kept residuals, 1.85e-2 and
  1.83e-2, are a capped pair at 0.4875 and 0.49 s in beat 1: with the
  mean displacement changing by about 2 um per step there, about 40 nm,
  above the noise floor. They give the largest step imbalance after
  start-up, 0.23% of peak inflow (0.49 s); beat 2 has no such pair. Steps
  that stall still take 21 iterations (about 14 s instead of 4 s).
* As in experiment 2: the branch outlets are short stumps (the LCCA stump has
  very few nodes), the outlet totals are split by assumed fractions, and
  the LV-derived stroke volume is about half a typical adult value. The
  total resistance, 3.55e8 Pa s/m3 = 2.7 mmHg s/mL, is about twice a
  typical total peripheral resistance; it offsets the cardiac output of
  about 2.2 L/min, half the normal value, so that the pressures look
  normal. The total compliance, outlets plus wall, is about 0.57 mL/mmHg.
