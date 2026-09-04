====================
CLI Entry-Point API
====================

.. module:: monai_physio.cli
.. currentmodule:: monai_physio.cli

The ``monai_physio.cli`` subpackage contains the entry-point scripts that
back the installed ``monai-physio-*`` console commands. Each module exposes
a ``main()`` function that parses ``argparse`` arguments and dispatches into
the corresponding workflow class.

User-facing documentation for the command-line tools (flags, examples, recipes)
lives under :doc:`../../cli_scripts/overview`. This section documents the
Python entry-point modules themselves so they are reachable from the Python
Module Index.

.. toctree::
   :maxdepth: 1

   convert_image_4d_to_3d
   convert_image_to_usd
   convert_image_to_vtk
   convert_vtk_to_usd
   create_statistical_model
   download_data
   fit_statistical_model_to_patient
   infer_physicsnemo
   reconstruct_highres_4d_ct
   train_physicsnemo
   visualize_pca_modes

See Also
========

* :doc:`../../cli_scripts/overview`
* :doc:`../workflows`
