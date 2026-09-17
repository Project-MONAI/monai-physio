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
   * - ``TCIA-4DLung``
     - Converted tutorial subset of the 4D-Lung collection, from the MONAI
       Physio GitHub release. See ``data/TCIA-4DLung/README.md`` for the
       data source, the full-collection manual download, and required
       citation.
   * - ``PhysicsNeMo-MGN-Lung-Motion``
     - Pretrained PhysicsNeMo MeshGraphNet checkpoint for lung motion, from
       the MONAI Physio GitHub release. Used by Lung Tutorial 9 (train) and
       Tutorial 10+ (infer).

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

   monai-physio-download-data [Slicer-Heart-CT|KCL-Heart-Model|CHOP-Valve4D|Chest-CT|TCIA-4DLung|PhysicsNeMo-MGN-Lung-Motion] [--directory DIRECTORY]

``data_name``
   Dataset to download. One of ``Slicer-Heart-CT``, ``KCL-Heart-Model``,
   ``CHOP-Valve4D``, ``Chest-CT``, ``TCIA-4DLung``, or
   ``PhysicsNeMo-MGN-Lung-Motion``. Required - omitting it prints help and
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
:meth:`monai_physio.download_data.DownloadData.DownloadSlicerHeartCTData`,
so repeated runs reuse the existing non-empty file and skip the split once
the ``slice_???.mha`` files are present.

For ``KCL-Heart-Model``, the command downloads, extracts, and reuses:

.. code-block:: text

   data/KCL-Heart-Model/average_mesh.vtk
   data/KCL-Heart-Model/input_meshes/01.vtk ... 20.vtk

The command uses
:meth:`monai_physio.download_data.DownloadData.DownloadKCLHeartModelData`,
which fetches each per-model ``.tar.gz`` archive from Zenodo, extracts its
mesh, and skips archives whose target ``.vtk`` file is already present.

For ``CHOP-Valve4D``, the command downloads, extracts, and reuses:

.. code-block:: text

   data/CHOP-Valve4D/Alterra/   (valve mesh time series, >1 GB)
   data/CHOP-Valve4D/TPV25/     (valve mesh time series, >1 GB)
   data/CHOP-Valve4D/CT/        (source CT volume and Simpleware segmentation)

The command uses
:meth:`monai_physio.download_data.DownloadData.DownloadCHOPValve4DData`,
which fetches each subdirectory's zip archive from the MONAI Physio GitHub
release and skips a subdirectory once it has its expected extracted files
(the CT volume or Simpleware segmentation for ``CT/``, ``.vtk`` meshes for
``Alterra/`` and ``TPV25/``) - a subdirectory left behind by an interrupted
extraction is re-downloaded rather than treated as complete.

For ``Chest-CT``, the command downloads and reuses a single volume:

.. code-block:: text

   data/Chest-CT/Chest-CT.mha

The command uses
:meth:`monai_physio.download_data.DownloadData.DownloadChestCTData`,
which fetches the volume from the MONAI Physio GitHub release and reuses an
existing non-empty file, so re-running resumes an interrupted download.

For ``TCIA-4DLung``, the command downloads, extracts, and reuses:

.. code-block:: text

   data/TCIA-4DLung/100_HM10395/100_HM10395_g000.nii.gz ...
   data/TCIA-4DLung/116_HM10395/116_HM10395_g000.nii.gz ...

The command uses
:meth:`monai_physio.download_data.DownloadData.DownloadTCIA4DLungData`,
which fetches both zip archives (``TCIA-4DLung-Part1.zip`` and
``TCIA-4DLung-Part2.zip``) from the MONAI Physio GitHub release and skips
the download entirely once any case's phase volumes are already present -
this is a converted tutorial subset; see ``data/TCIA-4DLung/README.md`` for
how to obtain the full TCIA 4D-Lung collection manually.

For ``PhysicsNeMo-MGN-Lung-Motion``, the command downloads, extracts, and
reuses:

.. code-block:: text

   tutorials/network_weights/physicsnemo_mgn_lung_motion/mgn_stage_model.pt
   tutorials/network_weights/physicsnemo_mgn_lung_motion/  (other epoch
       checkpoints and metadata)

The command uses
:meth:`monai_physio.download_data.DownloadData.DownloadPhysicsNeMoMGNLungMotionData`,
which fetches ``physicsnemo_mgn_lung_motion.zip`` from the MONAI Physio
GitHub release and skips the download once ``mgn_stage_model.pt`` is
already present. Unlike every other dataset, its default ``--directory`` is
``tutorials/network_weights``, not ``data/<data_name>``.

See Also
========

* :doc:`../tutorials` - ``Slicer-Heart-CT`` drives Heart Tutorials 1, 3 and 4;
  ``KCL-Heart-Model`` drives Heart Tutorial 6; ``Chest-CT`` drives Lung
  Tutorial 7 and Tutorial 13; ``TCIA-4DLung`` drives Lung Tutorials 1, 2, 3,
  4, 6, 8 and 10-12. ``DirLab-4DCT``, used by Heart Tutorial 7, has no
  ``monai-physio-download-data`` entry - DIR-Lab distributes each case
  individually and may require registration, so it is manual-only, see
  ``data/DirLab-4DCT/README.md``. ``Duke-Heart-4DLabelmaps``, which drives the
  ten ``duke_heart`` variants, is being released soon; see
  ``data/Duke-Heart-4DLabelmaps/README.md``. ``PhysicsNeMo-MGN-Lung-Motion``
  is the pretrained-checkpoint shortcut for Lung Tutorial 10 (and everything
  downstream of it: 11-14), letting a reader skip running Tutorial 9's
  training themselves.
* :doc:`byod_tutorials`
* :doc:`heart_gated_ct`
* :doc:`overview`
