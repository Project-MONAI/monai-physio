============================
Viewing Meshes and USD Files
============================

MONAI Physio writes VTP surface meshes and OpenUSD anatomy scenes with
per-organ prims, materials, and optional time samples. The lightweight Trame
mesh viewer supports animated USD and static VTP overlays; use an NVIDIA
Omniverse Kit application when full USD material fidelity is required.

.. important::

   ``pip install monai-physio`` pulls in `usd-core
   <https://pypi.org/project/usd-core/>`_, which is the OpenUSD *libraries*
   only — enough to write and read stages, but it contains no viewer.

Browser mesh viewer with Trame
==============================

Install the optional viewer dependencies, then point the CLI at a USD scene or
one or more VTP surface meshes:

.. code-block:: bash

   pip install "monai-physio[viewer]"
   monai-physio-view-meshes \
       tutorials/output/tutorial_01_lung/lung_model.all_painted.usd

The mesh viewer supports orbit, pan, zoom, time scrubbing, and looping
playback. It evaluates the same time-sampled mesh points stored in the USD and
uses the stage's ``timeCodesPerSecond`` value for playback. Static stages show
a single disabled timeline frame. Pass ``--fps 3`` to override the viewing
rate without modifying the USD. The viewer preloads point samples for
fixed-topology animation so playback does not reopen and rebuild the full
scene at every frame.

The Trame renderer approximates each bound OmniSurface material with its
diffuse color. It does not reproduce RTX-only transmission, subsurface
scattering, or MDL shading, so use USD Composer when judging final material
appearance. It also uses flat shading: recomputing smooth vertex normals for
every deformation can dominate the frame time of multi-million-triangle
anatomy, while the facets are negligible at that mesh density.

The same command accepts one or more ``.vtp`` surfaces. Multiple surfaces are
overlaid with distinct colors and opacity, and each toolbar checkbox toggles
one surface. This is useful for comparing a target segmentation with a fitted
model without first converting the surfaces to USD:

.. code-block:: bash

   monai-physio-view-meshes patient_surface.vtp fitted_surface.vtp

Docker preview
--------------

The tutorial image includes the viewer dependencies and an EGL rendering
backend. After building the image, run:

.. code-block:: bash

   ./docker/view-meshes.sh \
       tutorials/output/tutorial_01_lung/lung_model.all_painted.usd

To compare the two Tutorial 7 lung surfaces directly:

.. code-block:: bash

   ./docker/view-meshes.sh --port 8080 \
       tutorials/output/tutorial_07_lung/tutorial_07_lung_lung_surface.vtp \
       tutorials/output/tutorial_07_lung/tutorial_07_lung_template_surface_registered.vtp

Open ``http://127.0.0.1:8080/index.html``. Use ``--port`` to select another
port and ``--fps 3`` to play a 10-phase respiratory cycle in about 3.3 seconds.
Each input directory is mounted read-only, so surfaces may come from different
directories and relative USD references remain available. The server is
published only on the local host.

For a viewer running on a remote instance, leave the Docker command running
there and create an SSH tunnel from the local computer:

.. code-block:: powershell

   ssh -N -L 8080:127.0.0.1:8080 USER@REMOTE_HOST

Then open ``http://127.0.0.1:8080/index.html`` in the local browser. Use the
same port in the Docker command and both sides of ``-L`` when 8080 is already
in use. Press ``Ctrl+C`` in the remote terminal to stop the viewer.

Omniverse Kit applications
==========================

For full material fidelity, use **USD Composer**, built from the
`usd_composer template
<https://github.com/NVIDIA-Omniverse/kit-app-template/tree/main/templates/apps/usd_composer>`_
in NVIDIA's `kit-app-template
<https://github.com/NVIDIA-Omniverse/kit-app-template>`_ repository. Clone the
repository and follow its README: ``template new`` to create an app from the
``usd_composer`` template, ``build`` to build it, and ``launch`` to run it —
driven through ``./repo.sh`` on Linux and macOS, or ``.\repo.bat`` on Windows:

.. code-block:: bat

   .\repo.bat template new
   .\repo.bat build
   .\repo.bat launch

The same repository holds a ``usd_viewer`` template if you want a
review-and-playback app or a starting point for embedding a viewer in your own
tool.

Omniverse needs an RTX-capable NVIDIA GPU and a current driver.

Opening a MONAI Physio scene:

1. Launch your **USD Composer** app.
2. ``File > Open`` and select the generated ``.usd`` file — for the tutorials,
   under ``tutorials/output/<tutorial_name>/``.
3. Switch the viewport to the **camera defined in the USD scene**
   (``/World/Camera``) — see below.
4. Press **Play** on the timeline to run the animation. The frame rate is the
   ``frames_per_second`` the workflow was given, so a value of ``1.0`` plays one
   phase per second; raise it for smoother playback.
5. Anatomy materials are already bound, so the organs arrive colored. Select a
   prim in the stage tree to adjust its material, or to hide organs that
   occlude the structure you care about.

Use RTX rendering
-----------------

The workflows assign each tissue an OmniSurface material carrying its visual
properties — color, roughness, transmission and subsurface scattering for
translucent tissue. Those properties are only evaluated by the **RTX**
renderers (``RTX - Real-Time`` or ``RTX - Interactive``). In a preview or
Storm-style render mode the organs fall back to flat approximate shading, so
tissues that should read as translucent or wet look uniformly opaque. Set the
viewport renderer to RTX before judging how a scene looks.

Use the camera in the scene
---------------------------

Each scene ships a ``/World/Camera`` prim framing the anatomy, with clipping
planes and focus distance fitted to the anatomy's scale — the near plane is set
from the geometry's bounding-box diagonal, so you can zoom in close without the
surfaces vanishing. The default Omniverse perspective camera is set up for
room- and building-sized content, so on an organ-sized scene it clips the
anatomy away and navigates awkwardly. In the viewport camera menu, select the
scene's ``Camera`` rather than ``Perspective``. If a scene opens but appears
empty, this is almost always why. See :doc:`developer/usd_generation` for the
coordinate and unit details.

Before USD: viewing meshes directly
====================================

The intermediate ``.vtp`` and ``.vtu`` files that Tutorials 4, 6, 7, 8, 9, 10,
11 and 12 write need no USD tooling at all — PyVista, already a dependency,
opens them:

.. code-block:: python

   import pyvista as pv

   pv.read("tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_lung.vtp").plot()

This is usually the faster way to check a segmentation or a fitted shape model
before spending time on the USD export. For remote work, pass the same VTP
paths to ``monai-physio-view-meshes`` or ``docker/view-meshes.sh`` to use the
Trame browser viewer instead of opening a desktop window.

See Also
========

* :doc:`tutorials` — the workflows that produce these scenes
* :doc:`cli_scripts/vtk_to_usd` — converting existing meshes to USD
* :doc:`developer/usd_generation` — coordinate frames, materials, time samples
* :doc:`troubleshooting` — when a scene does not play or looks wrong
