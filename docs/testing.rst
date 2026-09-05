=======
Testing
=======

Run the fast test suite during development:

.. code-block:: bash

   pytest tests/ -v

Slow, GPU, Simpleware, and tutorial tests are auto-skipped unless
their opt-in flag is passed. Tests that depend on downloadable data fetch it
automatically via the session fixtures, so no marker filter is needed for them.

Opt-in Buckets
==============

Each ``--run-<bucket>`` flag enables one marker family:

.. code-block:: bash

   pytest tests/ -v --run-slow         # tests marked 'slow'
   pytest tests/ -v --run-gpu          # tests marked 'requires_gpu'
   pytest tests/ -v --run-simpleware   # tests marked 'requires_simpleware'
   pytest tests/ -v --run-physicsnemo  # tests marked 'requires_physicsnemo'
   pytest tests/ -v --run-tutorials    # tests marked 'tutorial'

Flags compose. A typical local GPU profile is:

.. code-block:: bash

   pytest tests/ -v --run-gpu --run-slow

``--run-all`` is a convenience flag that turns on every ``--run-*`` bucket at
once. The self-hosted CI GPU runner uses it (after installing
``.[dev,cuda13]``):

.. code-block:: bash

   pytest tests/ -v --run-all

Three further markers are declared but not gated by a flag: ``unit`` and
``integration`` are descriptive, and ``xdist_group`` is a ``pytest-xdist``
builtin used to keep related tests on one worker. Tests carrying only these
markers always run.

``tests/test_tutorials.py`` holds the ``tutorial`` bucket. It is not
parametrized over the tutorials directory - it is one hand-written class per
covered script, currently all 29 tutorial scripts, so adding a tutorial does
not automatically add a test.

Test Categories
===============

.. code-block:: bash

   # CLI help smoke tests
   pytest tests/test_cli_smoke.py -v

   # Public import surface
   pytest tests/test_import_public_api.py -v

Specific Areas
==============

.. code-block:: bash

   pytest tests/test_convert_vtk_to_usd.py -v
   pytest tests/test_convert_image_4d_to_3d.py -v
   pytest tests/test_contour_tools.py -v
   pytest tests/test_transform_tools.py -v
   pytest tests/test_image_tools.py -v
   pytest tests/test_workflow_train_physicsnemo.py -v
   pytest tests/test_workflow_evaluate_movement.py -v

Real Data and GPU Tests
=======================

Tests that need downloadable data request the session fixtures
(``test_directories``, ``download_test_data``, ``test_images``); the data is
downloaded on first use, so these tests run by default. GPU-bound tests are
marked ``requires_gpu`` (opt-in via ``--run-gpu``); Simpleware-bound tests are
marked ``requires_simpleware`` (opt-in via ``--run-simpleware`` and require a
licensed Simpleware Medical installation locally).

Continuous Integration
======================

CI runs the fast subset by default. The self-hosted GPU runner installs
``.[dev,cuda13]`` and invokes pytest with ``--run-all`` (which
enables every ``--run-*`` bucket); tests whose host requirements aren't met
(e.g. a licensed Simpleware install on a runner without one) runtime-skip
cleanly via their internal guards.

See Also
========

* :doc:`contributing`
* :doc:`tutorials`
