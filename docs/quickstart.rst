===========
Quick Start
===========

This guide will help you get started with MONAI Physio quickly.

Prerequisites
=============

Before starting, ensure you have:

* MONAI Physio installed (see :doc:`installation`)
* NVIDIA GPU with CUDA 12.6 or CUDA 13 - recommended for production
  performance; see :doc:`installation` for the ``[cuda12]`` extra
  (recommended, installs entirely from prebuilt wheels), ``[cuda13]``, or
  ``--torch-backend=auto``. A CPU-only PyPI install works for evaluation but
  is significantly slower.
* Disk space for the sample datasets (~10-20 GB for the full set; each dataset
  README lists its own size)

.. _tutorial_scripts:

Get the Tutorial Scripts
========================

The tutorials ship with the source repository, not with the pip package.
``pip install monai-physio`` installs the library and the ``monai-physio-*``
commands, but it does not create a ``tutorials/`` directory. Clone the
repository to get the scripts:

.. code-block:: bash

   git clone https://github.com/Project-MONAI/monai-physio.git
   cd monai_physio

To match the scripts to an already-installed release, check out its tag. Tags
are bare version strings, so use the installed version directly:

.. code-block:: bash

   python -c "import monai_physio; print(monai_physio.__version__)"

   git clone --depth 1 --branch {{ mphysio_project_version }} \
       https://github.com/Project-MONAI/monai-physio.git
   cd monai_physio

A release tarball works just as well if you would rather not use git:
``https://github.com/Project-MONAI/monai-physio/archive/refs/tags/{{ mphysio_project_version }}.tar.gz``.

No further installation step is needed after cloning - the scripts import
``monai_physio`` from your environment.

Run every dataset download from the top level of the clone. The tutorials
resolve their inputs against the repository root (``<repo>/data/<dataset>``),
while ``monai-physio-download-data`` writes to ``data/<dataset>`` relative to
the current working directory, so downloading from anywhere else puts the data
where the tutorials will not find it.

Then fetch the sample datasets, again from the top level of the clone:

.. code-block:: bash

   # Heart Tutorials 1, 3 and 4
   monai-physio-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT

   # Heart Tutorial 6
   monai-physio-download-data KCL-Heart-Model --directory data/KCL-Heart-Model

   # Lung Tutorial 7
   monai-physio-download-data Chest-CT --directory data/Chest-CT

   # No tutorial - transcatheter pulmonary valve experiments only, >2 GB
   monai-physio-download-data CHOP-Valve4D --directory data/CHOP-Valve4D

Which dataset each tutorial needs:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Dataset
     - Download
     - Used by
   * - ``Slicer-Heart-CT``
     - CLI
     - Heart Tutorials 1, 3, 4
   * - ``DirLab-4DCT``
     - Manual
     - Lung Tutorials 1, 2, 3, 4, 6, 8, 10, 11, 12, and Heart Tutorial 7
   * - ``KCL-Heart-Model``
     - CLI
     - Heart Tutorial 6
   * - ``Chest-CT``
     - CLI
     - Lung Tutorial 7, and Tutorial 13
   * - ``Duke-Heart-4DLabelmaps``
     - Releasing soon
     - The ten ``duke_heart`` variants, Tutorials 2 and 4 through 12
   * - ``CHOP-Valve4D``
     - CLI
     - No tutorial - used by the valve experiments under ``experiments/``

Tutorials 5 and 9 need no dataset of their own: they consume the outputs of
Tutorials 4 and 8 respectively.

``Duke-Heart-4DLabelmaps`` is scheduled for public release soon. Until then the
ten ``duke_heart`` variants cannot be run; contact Stephen Aylward
(saylward@nvidia.com) to request access, and see
``data/Duke-Heart-4DLabelmaps/README.md``.

``DirLab-4DCT`` is the one dataset with no automatic downloader: DIR-Lab
distributes each case individually and may require registration, so download it
by hand following ``data/DirLab-4DCT/README.md``. See
:doc:`cli_scripts/download_data` for every dataset's size and source.

Tutorial 1 needs only ``Slicer-Heart-CT``; with that dataset in place it runs
as a plain script:

.. code-block:: bash

   python tutorials/tutorial_01_heart_gated_ct_to_usd.py

:doc:`tutorials` is the full guide - thirteen numbered stages with previews of
what each one produces, the run order, and per-tutorial notes on pointing them
at your own data. The rest of this page is the same functionality as a CLI call and as
a Python API call, for when you would rather not start from a script.

Basic Workflow
==============

Minimal Slicer-Heart Quickstart
-------------------------------

The public Slicer-Heart 4D CT sample can be downloaded automatically and used
as the smallest end-to-end cardiac workflow. Data downloading and a
CUDA-capable GPU are required for practical runtime.

.. code-block:: bash

   monai-physio-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT

   monai-physio-convert-image-to-usd data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd \
       --registration-method ICON \
       --output-dir output/quickstart \
       --project-name slicer_heart_quickstart

Command-Line Interface
----------------------

The same cardiac processing, packaged as a command for unattended runs:

.. code-block:: bash

   # Process a single 4D cardiac CT file
   monai-physio-convert-image-to-usd cardiac_4d.nrrd --contrast --output-dir ./results

   # Process multiple time frames
   monai-physio-convert-image-to-usd frame_*.nrrd --contrast --project-name patient_001

   # With custom settings
   monai-physio-convert-image-to-usd cardiac.nrrd \
       --contrast \
       --reference-image ref.mha \
       --registration-iterations 50 \
       --output-dir ./output

Python API
----------

For more control, use the Python API. The workflow takes **images, not
filenames**: read your series with ITK first, and pick one frame as the
segmentation and registration reference.

.. code-block:: python

   import itk
   from pathlib import Path

   from monai_physio import (
       RegisterImagesICON,
       SegmentChestTotalSegmentatorWithContrast,
       WorkflowConvertImageToUSD,
   )

   frame_files = sorted(Path("data/Slicer-Heart-CT").glob("slice_???.mha"))
   time_series_images = [itk.imread(str(path)) for path in frame_files]
   reference_image = time_series_images[int(0.7 * len(time_series_images))]

   workflow = WorkflowConvertImageToUSD(
       time_series_images=time_series_images,
       reference_image=reference_image,
       output_directory="./results",
       usd_project_name="cardiac_model",
       registration_method=RegisterImagesICON(),
       segmentation_method=SegmentChestTotalSegmentatorWithContrast(),
       save_assets=True,
   )
   results = workflow.process()

   print(f"USD model saved to: {results['dynamic']}")

That is the whole pipeline: 4D input to 3D frames, registration between
phases, AI segmentation of the reference, contour transformation across time,
and an animated USD scene. A 4D ``.seq.nrrd`` needs splitting into frames
first - use ``monai-physio-convert-image-4d-to-3d`` or
:class:`~monai_physio.ConvertImage4DTo3D`.

:doc:`tutorials` walks the same call through real data with screenshots, and
each tutorial section ends with the constants to change for your own scans.

Working with Individual Components
===================================

Segmentation Only
-----------------

If you only need segmentation:

.. code-block:: python

   from monai_physio import SegmentChestTotalSegmentatorWithContrast
   import itk

   # Initialize segmenter (use SegmentChestTotalSegmentator for non-contrast studies)
   segmenter = SegmentChestTotalSegmentatorWithContrast()

   # Load and segment image
   image = itk.imread("chest_ct.nrrd")
   masks = segmenter.segment(image)

   # Extract individual anatomy masks by key
   heart_mask = masks["heart"]
   vessels_mask = masks["major_vessels"]
   lungs_mask = masks["lung"]
   labelmap = masks["labelmap"]

   # Save results
   itk.imwrite(heart_mask, "heart_mask.nrrd")
   itk.imwrite(labelmap, "labelmap.nrrd")

Image Registration Only
-----------------------

For standalone registration:

.. code-block:: python

   from monai_physio.register_images_icon import RegisterImagesICON
   import itk

   # Initialize registration
   registerer = RegisterImagesICON()

   # Load images
   fixed_image = itk.imread("reference_frame.mha")
   moving_image = itk.imread("target_frame.mha")

   # Configure registration
   registerer.set_modality('ct')
   registerer.set_fixed_image(fixed_image)

   # Perform registration
   results = registerer.register(moving_image)

   # Get transformation fields
   inverse_transform = results["inverse_transform"]  # Fixed to moving space
   forward_transform = results["forward_transform"]  # Moving to fixed space

VTK to USD Conversion
---------------------

Convert VTK time series to USD:

.. code-block:: python

   from monai_physio import ConvertVTKToUSD

   vtk_files = [f"heart_frame_{i:03d}.vtp" for i in range(10)]
   time_codes = [float(i) for i in range(len(vtk_files))]

   stage = ConvertVTKToUSD.from_files(
       data_basename="Heart",
       vtk_files=vtk_files,
       time_codes=time_codes,
   ).convert("heart_animation.usd")

Sample Data
===========

Download Sample Datasets
-------------------------

``Slicer-Heart-CT``, ``KCL-Heart-Model``, ``CHOP-Valve4D``, and ``Chest-CT``
are all auto-downloadable, via the CLI:

.. code-block:: bash

   monai-physio-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT

or from Python:

.. code-block:: python

   from monai_physio import DataDownloadTools

   data_file = DataDownloadTools.DownloadSlicerHeartCTData("data/Slicer-Heart-CT")
   assert DataDownloadTools.VerifySlicerHeartCTData("data/Slicer-Heart-CT")

See :doc:`cli_scripts/download_data` for sizes, source URLs, and directory
layouts for every dataset.

DirLab-4DCT data is manual-only; see ``data/DirLab-4DCT/README.md``. It drives
the whole lung pipeline - Lung Tutorials 1, 2, 3, 4, 6 and 8, plus Heart
Tutorial 7 - which then feeds the AI-surrogate Tutorials 9 through 12.
PhysicsNeMo and ``torch-geometric`` (for the MeshGraphNet) are base
dependencies of monai-physio; PhysicsNeMo itself requires Python >= 3.11.

Visualizing Results
===================

The workflows write OpenUSD scenes, and viewing them needs a USD viewer - use
an Omniverse Kit application with RTX rendering, which is what evaluates the
material properties assigned to each tissue. Note that the ``usd-core``
package installed with MONAI Physio provides the OpenUSD *libraries* only and
contains no viewer.

:doc:`viewing_usd` covers where to get it, how to set it up, and how to open a
MONAI Physio scene - including switching to the camera defined in the scene,
whose clipping planes are fitted to the anatomy's scale.

The intermediate meshes need no USD tooling:

.. code-block:: python

   import pyvista as pv

   # Load and display
   mesh = pv.read("heart_frame_000.vtp")
   mesh.plot()

Next Steps
==========

Now that you've completed your first workflow:

* Work through :doc:`tutorials` - the primary guide to using and adapting the
  toolkit
* Read detailed :doc:`cli_scripts/overview` for unattended and production runs
* Learn about :doc:`api/segmentation/index` options
* Understand :doc:`api/registration/index` methods
* Check the :doc:`api/base` for advanced usage

.. important::

   **Where to learn the toolkit:**

   * **tutorials/** - the primary resource. Each tutorial runs end-to-end on
     downloadable data, shows the workflow classes in context, and closes with
     the constants to change for your own scans. See :doc:`tutorials`.

   * **CLI Commands** - the same workflows packaged as commands for unattended
     and production runs (``monai-physio-convert-image-to-usd``,
     ``monai-physio-create-statistical-model``,
     ``monai-physio-fit-statistical-model-to-patient``). See
     ``src/monai_physio/cli/`` for implementation details.

   * **experiments/** - Research prototypes and design explorations. These demonstrate conceptual
     approaches for adapting workflows to new anatomical regions and digital twin applications,
     but may contain outdated APIs and should not be copied directly into production code.

Common Issues
=============

**Out of memory errors**

* Resample or crop the input image before running the workflow
* Process fewer frames at once
* Use Greedy registration with ``--registration-method Greedy`` when CUDA is unavailable

**Segmentation quality issues**

* Adjust contrast parameters
* Preprocess images (denoising, normalization)

**USD not animating**

* Check that the input time series has more than one frame
* Validate the generated USD with ``usdchecker final_model.usd``

See :doc:`troubleshooting` for more solutions.
