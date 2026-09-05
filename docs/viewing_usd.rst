==================
Viewing USD Files
==================

Every USD-producing workflow in MONAI Physio - Tutorials 1, 5, 10, 12 and 13,
and the
``monai-physio-convert-image-to-usd`` and ``monai-physio-convert-vtk-to-usd``
commands - writes an OpenUSD scene: anatomy split into per-organ prims, painted
with OmniSurface materials, and time-sampled when the input was a series. To
see the motion you need a USD viewer.

Use an **NVIDIA Omniverse Kit application**. It is built on OpenUSD, renders
with RTX in real time, and is the only viewer that shows these scenes as the
workflows intend them.

.. important::

   ``pip install monai-physio`` pulls in `usd-core
   <https://pypi.org/project/usd-core/>`_, which is the OpenUSD *libraries*
   only - enough to write and read stages, but it contains no viewer.

Omniverse Kit applications
==========================

The recommended application is **USD Composer**, built from the
`usd_composer template
<https://github.com/NVIDIA-Omniverse/kit-app-template/tree/main/templates/apps/usd_composer>`_
in NVIDIA's `kit-app-template
<https://github.com/NVIDIA-Omniverse/kit-app-template>`_ repository. Clone the
repository and follow its README: ``template new`` to create an app from the
``usd_composer`` template, ``build`` to build it, and ``launch`` to run it -
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
2. ``File > Open`` and select the generated ``.usd`` file - for the tutorials,
   under ``tutorials/output/<tutorial_name>/``.
3. Switch the viewport to the **camera defined in the USD scene**
   (``/World/Camera``) - see below.
4. Press **Play** on the timeline to run the animation. The frame rate is the
   ``frames_per_second`` the workflow was given, so a value of ``1.0`` plays one
   phase per second; raise it for smoother playback.
5. Anatomy materials are already bound, so the organs arrive colored. Select a
   prim in the stage tree to adjust its material, or to hide organs that
   occlude the structure you care about.

Use RTX rendering
-----------------

The workflows assign each tissue an OmniSurface material carrying its visual
properties - color, roughness, transmission and subsurface scattering for
translucent tissue. Those properties are only evaluated by the **RTX**
renderers (``RTX - Real-Time`` or ``RTX - Interactive``). In a preview or
Storm-style render mode the organs fall back to flat approximate shading, so
tissues that should read as translucent or wet look uniformly opaque. Set the
viewport renderer to RTX before judging how a scene looks.

Use the camera in the scene
---------------------------

Each scene ships a ``/World/Camera`` prim framing the anatomy, with clipping
planes and focus distance fitted to the anatomy's scale - the near plane is set
from the geometry's bounding-box diagonal, so you can zoom in close without the
surfaces vanishing. The default Omniverse perspective camera is set up for
room- and building-sized content, so on an organ-sized scene it clips the
anatomy away and navigates awkwardly. In the viewport camera menu, select the
scene's ``Camera`` rather than ``Perspective``. If a scene opens but appears
empty, this is almost always why. See :doc:`developer/usd_generation` for the
coordinate and unit details.

Before USD: viewing the meshes directly
=======================================

The intermediate ``.vtp`` and ``.vtu`` files that Tutorials 4, 6, 7, 8, 9, 10,
11 and 12 write need no USD tooling at all - PyVista, already a dependency,
opens them:

.. code-block:: python

   import pyvista as pv

   pv.read("tutorials/output/tutorial_04_heart/patient_surfaces.vtp").plot()

This is usually the faster way to check a segmentation or a fitted shape model
before spending time on the USD export.

See Also
========

* :doc:`tutorials` - the workflows that produce these scenes
* :doc:`cli_scripts/vtk_to_usd` - converting existing meshes to USD
* :doc:`developer/usd_generation` - coordinate frames, materials, time samples
* :doc:`troubleshooting` - when a scene does not play or looks wrong
