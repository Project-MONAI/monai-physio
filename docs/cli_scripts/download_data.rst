=====================
Download Example Data
=====================

The ``monai-physio-download-data`` command downloads example datasets used by
MONAI Physio tutorials and demos.

Supported Datasets
==================

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Data name
     - Description
   * - ``Slicer-Heart-CT``
     - Public 4D cardiac CT sample from SlicerHeart.
   * - ``KCL-Heart-Model``
     - King's College London four-chamber heart model dataset: 20
       individual heart meshes plus an average mesh, from Zenodo.
   * - ``CHOP-Valve4D``
     - CHOP Jolley Lab transcatheter pulmonary valve model, converted from
       the original FEBio model to VTK/ITK and segmented with Simpleware,
       from the MONAI Physio GitHub release. See
       ``data/CHOP-Valve4D/README.md``.
   * - ``Chest-CT``
     - Ungated 3D chest CT, a single static volume, from the MONAI Physio
       GitHub release. See ``data/Chest-CT/README.md`` for the data source
       and required citation.

Basic Usage
===========

Download a dataset into its default location:

.. code-block:: bash

   monai-physio-download-data Slicer-Heart-CT

Running the command with no arguments prints usage/help instead of
downloading anything.

Options
=======

.. code-block:: bash

   monai-physio-download-data [Slicer-Heart-CT|KCL-Heart-Model|CHOP-Valve4D|Chest-CT] [--directory DIRECTORY]

``data_name``
   Dataset to download. One of ``Slicer-Heart-CT``, ``KCL-Heart-Model``,
   ``CHOP-Valve4D``, or ``Chest-CT``. Required — omitting it prints help and
   exits.

``--directory``
   Directory where the dataset is stored. Defaults to ``data/<data_name>``.

Output
======

For ``Slicer-Heart-CT``, the command downloads the 4-D sequence and splits it
into per-phase 3-D volumes:

.. code-block:: text

   data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd
   data/Slicer-Heart-CT/slice_000.mha ... slice_020.mha

The command uses
:meth:`monai_physio.data_download_tools.DataDownloadTools.DownloadSlicerHeartCTData`,
so repeated runs reuse the existing non-empty file and skip the split once
the ``slice_???.mha`` files are present.

For ``KCL-Heart-Model``, the command downloads, extracts, and reuses:

.. code-block:: text

   data/KCL-Heart-Model/average_mesh.vtk
   data/KCL-Heart-Model/input_meshes/01.vtk ... 20.vtk

The command uses
:meth:`monai_physio.data_download_tools.DataDownloadTools.DownloadKCLHeartModelData`,
which fetches each per-model ``.tar.gz`` archive from Zenodo, extracts its
mesh, and skips archives whose target ``.vtk`` file is already present.

For ``CHOP-Valve4D``, the command downloads, extracts, and reuses:

.. code-block:: text

   data/CHOP-Valve4D/Alterra/   (valve mesh time series, >1 GB)
   data/CHOP-Valve4D/TPV25/     (valve mesh time series, >1 GB)
   data/CHOP-Valve4D/CT/        (source CT volume and Simpleware segmentation)

The command uses
:meth:`monai_physio.data_download_tools.DataDownloadTools.DownloadCHOPValve4DData`,
which fetches each subdirectory's zip archive from the MONAI Physio GitHub
release and skips a subdirectory once it has its expected extracted files
(the CT volume or Simpleware segmentation for ``CT/``, ``.vtk`` meshes for
``Alterra/`` and ``TPV25/``) — a subdirectory left behind by an interrupted
extraction is re-downloaded rather than treated as complete.

For ``Chest-CT``, the command downloads and reuses a single volume:

.. code-block:: text

   data/Chest-CT/Chest-CT.mha

The command uses
:meth:`monai_physio.data_download_tools.DataDownloadTools.DownloadChestCTData`,
which fetches the volume from the MONAI Physio GitHub release and reuses an
existing non-empty file, so re-running resumes an interrupted download.

See Also
========

* :doc:`../tutorials` — ``Slicer-Heart-CT`` drives Heart Tutorials 1, 3 and 4;
  ``KCL-Heart-Model`` drives Heart Tutorial 6; ``Chest-CT`` drives Lung
  Tutorial 7 and Tutorial 13. ``DirLab-4DCT`` — Lung Tutorials 1, 2, 3, 4, 6, 8
  and 10-12, plus Heart Tutorial 7 — is manual-only, see
  ``data/DirLab-4DCT/README.md``. ``Duke-Heart-4DLabelmaps``, which drives the
  ten ``duke_heart`` variants, is being released soon; see
  ``data/Duke-Heart-4DLabelmaps/README.md``.
* :doc:`byod_tutorials`
* :doc:`heart_gated_ct`
* :doc:`overview`
