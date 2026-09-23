============================
Viewing Meshes and USD Files
============================

MONAI Physio includes a lightweight Trame viewer for animated USD and static
VTP surfaces. Use an Omniverse Kit application when RTX material fidelity is
required.

Install and run
===============

.. code-block:: bash

   pip install "monai-physio[viewer]"
   monai-physio-view-meshes output.usd

The viewer supports orbit, pan, zoom, time scrubbing, and looping playback.
Override the USD playback rate without modifying the file:

.. code-block:: bash

   monai-physio-view-meshes --fps 3 output.usd

Overlay one or more static VTP surfaces:

.. code-block:: bash

   monai-physio-view-meshes patient.vtp fitted.vtp

Docker
======

The tutorial image includes the viewer and EGL rendering dependencies:

.. code-block:: bash

   ./docker/view-meshes.sh --port 8080 --fps 3 \
       tutorials/output/tutorial_01_lung/lung_model.all_painted.usd

For a static surface comparison:

.. code-block:: bash

   ./docker/view-meshes.sh --port 8080 \
       tutorials/output/tutorial_07_lung/tutorial_07_lung_lung_surface.vtp \
       tutorials/output/tutorial_07_lung/tutorial_07_lung_template_surface_registered.vtp

Open ``http://127.0.0.1:8080/index.html``. Stop the server with ``Ctrl+C``.

Remote instances
================

Leave the viewer running on the remote instance. For Brev:

.. code-block:: bash

   brev port-forward INSTANCE_NAME -p 8080:8080

Or create an SSH tunnel from the local machine:

.. code-block:: bash

   ssh -N -L 8080:127.0.0.1:8080 USER@REMOTE_HOST

Then open ``http://127.0.0.1:8080/index.html`` locally.

Limitations
===========

The Trame viewer approximates USD materials with diffuse color and opacity.
It does not reproduce RTX-only transmission, subsurface scattering, or MDL
shading. VTP input is a static overlay; animation requires time-sampled USD.

For full material rendering, open the USD in an NVIDIA Omniverse Kit
application, select the scene's ``/World/Camera``, choose an RTX renderer, and
press Play. The OpenUSD package installed by ``usd-core`` does not include a
viewer.

See Also
========

* :doc:`tutorials` — workflows that produce VTP and USD output
* :doc:`cli_scripts/vtk_to_usd` — converting meshes to USD
* :doc:`developer/usd_generation` — coordinates, materials, and time samples
* :doc:`troubleshooting` — rendering and playback problems
