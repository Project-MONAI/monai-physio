=========
Tutorials
=========

.. raw:: html

   <section class="mphysio-hero">
     <div class="mphysio-hero__brand">
       <img src="_static/nvidia-logo.svg" alt="NVIDIA logo">
     </div>
     <p class="mphysio-kicker">MONAI Physio tutorials</p>
     <h1>From a CT scan to an animated digital twin</h1>
     <p>
       Eighteen numbered stages across 36 Python scripts, 21 of them runnable
       today: the fifteen <code>duke_heart</code> variants wait on a dataset
       that is being released soon.
       Each one drives the real workflow classes end-to-end on downloadable
       data, shows what it produced, and ends with the handful of constants
       to change so it runs on your own scans.
     </p>
   </section>

Before You Start
================

**1. Get the scripts.** They ship with the source repository, not with the pip
package — ``pip install monai-physio`` gives you the library and the
``monai-physio-*`` commands but no ``tutorials/`` directory:

.. code-block:: bash

   git clone https://github.com/Project-MONAI/monai-physio.git
   cd monai_physio

See :doc:`quickstart` for version-matched clones and the release tarball link.

**2. Get the data**, running every download from the top level of the clone.
The tutorials resolve their inputs against the repository root
(``<repo>/data/<dataset>``), while the CLI writes to ``data/<dataset>``
relative to the current working directory:

.. code-block:: bash

   monai-physio-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT
   monai-physio-download-data KCL-Heart-Model --directory data/KCL-Heart-Model
   monai-physio-download-data Chest-CT --directory data/Chest-CT

That covers Heart Tutorials 1, 3, 4 and 6 (``Slicer-Heart-CT`` and
``KCL-Heart-Model``) and Lung Tutorial 7 (``Chest-CT``), which Tutorial 13 also
animates. ``DirLab-4DCT`` — used by Lung Tutorials 1, 2, 3, 4, 6, 8, 10, 11 and
12, and by Heart Tutorial 7 — is **not** auto-downloaded: DIR-Lab distributes
each case individually and may require registration.

Tutorials 5 and 9 need no dataset of their own; they consume the outputs of
Tutorials 4 and 8. ``Duke-Heart-4DLabelmaps`` drives the fifteen ``duke_heart``
variants: a fourteen-tutorial chain from Tutorial 4 through Tutorial 18, plus
the separate, optional Tutorial 2 ICON finetuning variant; the dataset is
being released soon, and until then access can be requested from Stephen Aylward
(saylward@nvidia.com). See ``data/DirLab-4DCT/README.md``,
``data/Duke-Heart-4DLabelmaps/README.md``, and
:doc:`cli_scripts/download_data` for every dataset's size and source.

**3. Know where output lands.** Every tutorial writes to
``tutorials/output/<tutorial_name>/`` and reuses what it finds there, so a
second run is cheap and later tutorials pick up earlier results automatically.

.. raw:: html

   <section class="mphysio-card-grid" aria-label="Tutorial cards">
     <a class="mphysio-card" href="#tutorial-1-gated-4d-ct-to-animated-usd">
       <span class="mphysio-card__number">01</span>
       <h2>Gated 4D CT to Animated USD</h2>
       <p>Segment, register and assemble a 4D CT series into an animated OpenUSD scene.</p>
       <span class="mphysio-card__meta">Slicer-Heart-CT &middot; DIR-Lab</span>
     </a>
     <a class="mphysio-card" href="#tutorial-2-finetune-icon-registration">
       <span class="mphysio-card__number">02</span>
       <h2>Finetune ICON Registration</h2>
       <p>Adapt uniGradICON to your own cohort and measure what the finetuning bought you.</p>
       <span class="mphysio-card__meta">DIR-Lab</span>
     </a>
     <a class="mphysio-card" href="#tutorial-3-reconstruct-high-resolution-4d-ct">
       <span class="mphysio-card__number">03</span>
       <h2>Reconstruct High-Resolution 4D CT</h2>
       <p>Register every phase to one reference and reconstruct the series at its resolution.</p>
       <span class="mphysio-card__meta">Slicer-Heart-CT &middot; DIR-Lab</span>
     </a>
     <a class="mphysio-card" href="#tutorial-4-ct-segmentation-to-vtk-surfaces">
       <span class="mphysio-card__number">04</span>
       <h2>CT Segmentation to VTK Surfaces</h2>
       <p>Segment one CT phase and export patient anatomy as VTK PolyData surfaces.</p>
       <span class="mphysio-card__meta">Slicer-Heart-CT &middot; DIR-Lab</span>
     </a>
     <a class="mphysio-card" href="#tutorial-5-vtk-surfaces-to-animated-usd">
       <span class="mphysio-card__number">05</span>
       <h2>VTK Surfaces to Animated USD</h2>
       <p>Convert meshes into a time-sampled USD scene for Omniverse playback.</p>
       <span class="mphysio-card__meta">Tutorial 4 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-6-create-a-pca-shape-model">
       <span class="mphysio-card__number">06</span>
       <h2>Create a PCA Shape Model</h2>
       <p>Turn a population of meshes into a statistical shape model and its modes.</p>
       <span class="mphysio-card__meta">KCL-Heart-Model &middot; DIR-Lab</span>
     </a>
     <a class="mphysio-card" href="#tutorial-7-fit-the-shape-model-to-a-patient">
       <span class="mphysio-card__number">07</span>
       <h2>Fit the Shape Model to a Patient</h2>
       <p>Fit the shape model to one ungated clinical scan, PCA coefficients and all.</p>
       <span class="mphysio-card__meta">Chest-CT &middot; Tutorial 6 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-8-propagate-the-shape-model-through-4d">
       <span class="mphysio-card__number">08</span>
       <h2>Propagate the Model Through 4D</h2>
       <p>Fit each case at its reference phase and carry the mesh through every phase.</p>
       <span class="mphysio-card__meta">DIR-Lab &middot; Tutorials 2 and 6</span>
     </a>
     <a class="mphysio-card" href="#tutorial-9-train-a-physicsnemo-surrogate">
       <span class="mphysio-card__number">09</span>
       <h2>Train a PhysicsNeMo Surrogate</h2>
       <p>Train a MeshGraphNet to predict per-vertex motion from shape and phase.</p>
       <span class="mphysio-card__meta">Tutorial 8 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-10-predict-motion-with-the-surrogate">
       <span class="mphysio-card__number">10</span>
       <h2>Predict Motion With the Surrogate</h2>
       <p>Replace the registration solve with one forward pass, then export to USD.</p>
       <span class="mphysio-card__meta">Tutorials 8 and 9 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-11-score-the-surrogate-against-the-images">
       <span class="mphysio-card__number">11</span>
       <h2>Score the Surrogate Against the Images</h2>
       <p>Volume and surface RMSE per lobe, plus Dice per chamber, on the held-out case.</p>
       <span class="mphysio-card__meta">Tutorials 8, 9 and 10 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-12-the-whole-inference-pipeline-in-one-script">
       <span class="mphysio-card__number">12</span>
       <h2>The Whole Inference Pipeline in One Script</h2>
       <p>Go from a gated series to an animated prediction without registering a single phase.</p>
       <span class="mphysio-card__meta">Tutorials 6 and 9 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-13-breathe-and-beat-a-static-clinical-ct">
       <span class="mphysio-card__number">13</span>
       <h2>Breathe and Beat a Static Clinical CT</h2>
       <p>Animate one ungated breath-hold scan with both rhythms, from two networks at once.</p>
       <span class="mphysio-card__meta">Chest-CT &middot; Tutorials 7 and 9 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-14-sweep-the-shape-parameters">
       <span class="mphysio-card__number">14</span>
       <h2>Sweep the Shape Parameters</h2>
       <p>Re-infer and rescore over a grid of PCA coefficients, to see how far the motion moves with them.</p>
       <span class="mphysio-card__meta">Tutorials 8 and 9 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-15-leave-one-out-cross-validation">
       <span class="mphysio-card__number">15</span>
       <h2>Leave-One-Out Cross-Validation</h2>
       <p>Rebuild the model, refit, retrain and rescore once per fold, for a spread rather than a number.</p>
       <span class="mphysio-card__meta">DIR-Lab &middot; Duke-Heart-4DLabelmaps</span>
     </a>
     <a class="mphysio-card" href="#tutorial-16-build-a-volumetric-shape-model-of-the-myocardium">
       <span class="mphysio-card__number">16</span>
       <h2>Build a Volumetric Shape Model of the Myocardium</h2>
       <p>Fill the mean surface with tetrahedra so a strain energy has a deformation gradient to act on.</p>
       <span class="mphysio-card__meta">Duke-Heart-4DLabelmaps &middot; Tutorial 4 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-17-train-a-mechanics-aware-cardiac-motion-surrogate">
       <span class="mphysio-card__number">17</span>
       <h2>Train a Mechanics-Aware Cardiac Motion Surrogate</h2>
       <p>Add a neo-Hookean strain energy to the displacement loss, and train an unweighted ablation to isolate it.</p>
       <span class="mphysio-card__meta">Tutorial 16 output</span>
     </a>
     <a class="mphysio-card" href="#tutorial-18-predict-myocardial-motion-and-the-stress-it-implies">
       <span class="mphysio-card__number">18</span>
       <h2>Predict Myocardial Motion and the Stress It Implies</h2>
       <p>Score the physics-informed surrogate against its ablation and export the motion colored by von Mises stress.</p>
       <span class="mphysio-card__meta">Tutorials 16 and 17 output</span>
     </a>
   </section>

Recommended Run Order
=====================

Tutorials are straightforward Python scripts: run one with
``python tutorials/tutorial_01_heart_gated_ct_to_usd.py``, or open it in your
editor and read it top to bottom. Numbers 1, 4 and 5 are the fastest way to see
the toolkit
work end-to-end; 6 through 18 build the statistical-model and AI-surrogate
pipeline on top.

1. **Tutorial 1** — after downloading Slicer-Heart-CT.
2. **Tutorial 2** — after obtaining DIR-Lab. It writes the finetuned ICON
   weights Tutorial 8 uses when present; optional if stock weights are
   acceptable.
3. **Tutorial 3** — after obtaining its dataset; it registers with Greedy and
   needs no finetuned weights.
4. **Tutorial 4** — after downloading Slicer-Heart-CT.
5. **Tutorial 5** — after Tutorial 4, whose surfaces it converts.
6. **Tutorial 6** — heart needs KCL-Heart-Model, lung needs DIR-Lab.
7. **Tutorial 7** — after Tutorial 6; the lung variant also needs Chest-CT.
8. **Tutorial 8** — after Tutorial 6 (lung); Tutorial 2 optional.
9. **Tutorial 9** — after Tutorial 8, whose fitted meshes it trains on.
10. **Tutorial 10** — after Tutorial 9, whose checkpoint it loads.
11. **Tutorial 11** — after Tutorial 9. The lung variant segments every gated
    frame of the held-out case, so it needs a GPU and the segmentation weights.
12. **Tutorial 12** — after Tutorial 6 and Tutorial 9 for its anatomy; it fits
    the model to the patient itself, so nothing is read from Tutorial 8.
13. **Tutorial 13** — after Tutorial 7 (lung) and Tutorial 9 for both anatomies.
    It also needs Simpleware Medical, which segments the heart it fits.
14. **Tutorial 14** — after Tutorial 8 and Tutorial 9, whose fit and
    checkpoint every point of the grid reuses. It scores each point the way
    Tutorial 11 does, so it needs a GPU and the segmentation weights too.
15. **Tutorial 15** — needs only the cohort. It rebuilds the shape model, the
    fits and the network per fold, so nothing from Tutorials 6, 8 or 9 is read;
    those outputs are reused as a cache when they happen to be there.
16. **Tutorials 16, 17 and 18** (Duke heart) — in that order, after Tutorial 4.
    They branch off the chain rather than continuing it: a strain energy needs
    volume elements, so Tutorial 16 rebuilds the shape model tetrahedrally
    instead of reusing the surface one from Tutorials 6 to 8. Tutorial 17 then
    trains against that energy and Tutorial 18 scores it and reads out stress.

Tutorial 1: Gated 4D CT to Animated USD
=======================================

Script
   ``tutorials/tutorial_01_heart_gated_ct_to_usd.py`` (Slicer-Heart-CT)

   ``tutorials/tutorial_01_lung_gated_ct_to_usd.py`` (DIR-Lab)

Workflow
   :class:`~monai_physio.WorkflowConvertImageToUSD`, driving
   :class:`~monai_physio.RegisterImagesGreedy` and a
   :class:`~monai_physio.SegmentAnatomyBase` subclass.

Dataset
   Slicer-Heart-CT (auto-download) for the heart, DIR-Lab (manual) for the
   lung. The phase roughly 70% through the series is the segmentation and
   registration reference.

Requirements
   Greedy registers every phase against the reference on the CPU; a GPU is
   still needed for segmentation.

Preview
   .. figure:: assets/tutorial_01_heart_4d.gif
      :alt: Animated cardiac USD produced by Tutorial 1
      :width: 90%

      The animated cardiac model, played back in Omniverse.

   .. figure:: assets/tutorial_01_lung_4d.gif
      :alt: Animated lung USD produced by Tutorial 1
      :width: 90%

      The same workflow on a DIR-Lab respiratory series.

Inner API usage
   .. code-block:: python

      workflow = WorkflowConvertImageToUSD(
          time_series_images=time_series_images,
          reference_image=reference_image,
          output_directory=str(output_dir),
          usd_project_name="cardiac_model",
          registration_method=registration_method,
          segmentation_method=segmentation_method,
          save_assets=True,
      )
      workflow_results = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_01_heart_gated_ct_to_usd.py
      python tutorials/tutorial_01_lung_gated_ct_to_usd.py

Outputs
   The animated USD named after ``usd_project_name``, the per-phase registered
   volumes and labelmaps, and screenshots — all under
   ``tutorials/output/tutorial_01_{heart,lung}/``.

Adapt to your data
   Point ``data_dir`` and the file glob near the top of the script at your own
   series: any set of 3D volumes ITK can read (``.mha``, ``.nrrd``,
   ``.nii.gz``) in acquisition order, or a 4D ``.seq.nrrd`` split first with
   ``monai-physio-convert-image-4d-to-3d``. Choose the reference phase by
   changing the index expression, and swap ``segmentation_method`` for the one
   matching your anatomy and contrast — see :doc:`api/segmentation/index`. For
   command-line use without editing code, run
   ``monai-physio-convert-image-to-usd`` (:doc:`cli_scripts/heart_gated_ct`).

Tutorial 2: Finetune ICON Registration
======================================

Script
   ``tutorials/tutorial_02_lung_finetune_icon.py``

   ``tutorials/tutorial_02_lung_distancemap_finetune_icon.py`` — the lung
   distance-map variant, which finetunes on distance maps rather than image
   intensities so the labelmap-to-labelmap stage of Tutorials 7 and 8 has
   in-distribution weights.

   ``tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py`` — the same
   for the heart, on Duke-Heart-4DLabelmaps. The heart needs its own run because
   it registers with a much tighter mask than the lungs, so its distance maps
   saturate over a shorter radius and do not share an intensity distribution
   with lung ones. The per-organ values live in
   ``tutorials/parameters_lung_ct_dirlab.py`` for the lung variant and
   ``tutorials/parameters_duke_heart_labelmaps.py`` for this one. This is a
   ``duke_heart`` tutorial: Duke-Heart-4DLabelmaps is being released soon (see
   `Before You Start`_), and until then access can be requested from Stephen
   Aylward (saylward@nvidia.com) — see
   ``data/Duke-Heart-4DLabelmaps/README.md``.

Workflow
   :class:`~monai_physio.WorkflowFinetuneICONRegistration`, then
   :class:`~monai_physio.RegisterImagesGreedy` and
   :class:`~monai_physio.RegisterImagesGreedyICON` to score the result, with
   :class:`~monai_physio.SegmentNVSegmentCTMRI` supplying the labelmaps.

Dataset
   DIR-Lab (manual). Every case except ``Case1Pack`` trains; ``Case1Pack`` is
   held out and registered three ways — Greedy alone with its defaults, then
   Greedy+ICON with the stock uniGradICON weights and with the finetuned ones —
   so the improvement is measured, not asserted.

Scoring
   The fixed image is segmented once, and each registered moving image is
   segmented again after warping. The table reports the mean, 5th percentile,
   median, 95th percentile, minimum and maximum of the per-class Dice scores,
   plus the mislabeled voxel count, with the unregistered moving image as a
   reference row. Segmenting each warped volume separately costs one GPU
   segmentation per method and folds segmentation variability into the scores.

Requirements
   GPU required. 100 epochs over nine cases: the longest-running tutorial
   before the AI-surrogate chain. The experiment directory is cleared on every
   run, so it does not resume.

Preview
   .. figure:: assets/tutorial_02_finetuning.png
      :alt: Registration accuracy table for the held-out case
      :width: 100%

      The held-out case scored per method — unregistered, Greedy, Greedy+ICON
      with the stock weights, and with the finetuned weights.

Inner API usage
   .. code-block:: python

      workflow = WorkflowFinetuneICONRegistration(
          subject_image_files=list(subject_image_files.values()),
          output_dir=weights_dir,
          finetune_name=finetune_name,
          subject_ids=list(subject_image_files.keys()),
          epochs=epochs,
          dice_loss_weight=0.0,
      )
      weights_path = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_02_lung_finetune_icon.py
      python tutorials/tutorial_02_lung_distancemap_finetune_icon.py

Outputs
   The finetuned checkpoint under
   ``tutorials/network_weights/icon_dirlab_4dct/``, plus
   ``registration_summary.csv``, the fixed-minus-registered difference images
   (residual structure is what separates the methods), the fixed and warped
   labelmaps, and before/after screenshots in
   ``tutorials/output/tutorial_02_lung/``.

Adapt to your data
   Replace the training cohort glob with your own volumes and set ``epochs``
   to fit your budget — the workflow needs only a list of image files and
   matching subject ids. Raise ``dice_loss_weight`` above ``0.0`` when you also
   have labelmaps to supervise with. Load the resulting weights anywhere by
   passing them to :class:`~monai_physio.RegisterImagesICON`.

Tutorial 3: Reconstruct High-Resolution 4D CT
=============================================

Script
   ``tutorials/tutorial_03_heart_reconstruct_highres_4d_ct.py``

   ``tutorials/tutorial_03_lung_reconstruct_highres_4d_ct.py``

Workflow
   :class:`~monai_physio.WorkflowReconstructHighres4DCT` with
   :class:`~monai_physio.RegisterImagesGreedy`.

Dataset
   Slicer-Heart-CT for the heart; DIR-Lab for the lung, which reconstructs
   against its T70 (end-exhale) phase — the same reference Tutorial 8 fits to.

Requirements
   CPU is enough. One coarse-to-fine registration per phase, greedy schedule
   ``[30, 15, 7, 3]``.

Preview
   .. figure:: assets/tutorial_03_heart_original.gif
      :alt: Acquired cardiac phases
      :width: 90%

      The acquired cardiac phases.

   .. figure:: assets/tutorial_03_heart_recon.gif
      :alt: Cardiac phases reconstructed at the reference resolution
      :width: 90%

      The same phases reconstructed at the reference resolution.

   .. figure:: assets/tutorial_03_output_comparison.gif
      :alt: Acquired phase beside the reconstructed high-resolution phase
      :width: 90%

      Side by side on the lung series.

Inner API usage
   .. code-block:: python

      registration_method = RegisterImagesGreedy()
      registration_method.set_number_of_iterations([30, 15, 7, 3])

      workflow = WorkflowReconstructHighres4DCT(
          time_series_images=time_series,
          reference_image=reference_image,
          reference_time_frame=reference_time_frame,
          registration_method=registration_method,
      )
      workflow.set_modality("ct")
      result = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_03_heart_reconstruct_highres_4d_ct.py
      python tutorials/tutorial_03_lung_reconstruct_highres_4d_ct.py

Outputs
   ``reconstructed_frame_<i>.mha`` plus forward and inverse transforms for
   every phase, and two screenshots, under
   ``tutorials/output/tutorial_03_{heart,lung}/``.

Adapt to your data
   Set ``case_glob`` and ``data_dir`` to your series and pick the reference
   with ``reference_time_frame``. If you have a separate breath-hold or
   contrast-enhanced volume, pass it as ``reference_image`` instead of one of
   the phases — that is what the workflow is really designed for. Tune
   ``number_of_iterations_greedy`` down for a fast smoke test. The saved
   ``.hdf`` transforms are reusable:
   :class:`~monai_physio.TransformTools` applies them to meshes and labelmaps.

Tutorial 4: CT Segmentation to VTK Surfaces
===========================================

Script
   ``tutorials/tutorial_04_heart_ct_to_vtk.py``

   ``tutorials/tutorial_04_lung_ct_to_vtk.py``

   ``tutorials/tutorial_04_duke_heart_labelmap_to_vtk.py`` — starts from gated
   labelmaps rather than CT, and also extracts tetrahedral meshes. Needs
   Duke-Heart-4DLabelmaps (see `Before You Start`_).

Workflow
   :class:`~monai_physio.WorkflowConvertImageToVTK` with
   :class:`~monai_physio.SegmentChestTotalSegmentatorWithContrast` (heart) or
   :class:`~monai_physio.SegmentChestTotalSegmentator` (lung).

Dataset
   One frame of Slicer-Heart-CT or DIR-Lab — a single static volume is enough.

Requirements
   GPU recommended for segmentation; no registration, so this is the quickest
   way to confirm your environment and model weights work.

Preview
   .. figure:: assets/tutorial_04_heart.gif
      :alt: Cardiac surfaces extracted from a CT phase
      :width: 90%

      Cardiac anatomy surfaces exported from one CT phase.

   .. figure:: assets/tutorial_04_lung.png
      :alt: Lung surfaces extracted from a CT phase
      :width: 90%

      The same workflow on a DIR-Lab respiratory case.

   .. figure:: assets/tutorial_04_duke_heart.png
      :alt: Heart surfaces extracted from a gated Duke labelmap
      :width: 90%

      The ``duke_heart`` variant, which starts from a gated labelmap rather than
      a CT and also writes tetrahedral meshes.

Inner API usage
   .. code-block:: python

      workflow = WorkflowConvertImageToVTK(
          segmentation_method=segmentation_method,
      )
      result = workflow.process(
          input_image=ct_image,
          surface_reduction_rate=HEART_CT_KCL.surface_reduction_rate,
          extract_label_surfaces=save_label_surfaces,
      )

Run
   .. code-block:: bash

      python tutorials/tutorial_04_heart_ct_to_vtk.py
      python tutorials/tutorial_04_lung_ct_to_vtk.py

Outputs
   ``patient_surfaces.vtp`` (all anatomy in one mesh, with a per-cell
   ``SegmentationLabelIds`` array so each cell still names the structure it came
   from), per-group and per-label ``.vtp`` files, ``patient_labelmap.mha`` and
   two screenshots, under ``tutorials/output/tutorial_04_{heart,lung}/``.

Adapt to your data
   Change the input volume path, then choose the segmenter matching your scan:
   contrast versus non-contrast CT, or
   :class:`~monai_physio.SegmentNVSegmentCTMRI` for CT **and** MRI. Raise
   ``surface_reduction_rate`` in the tutorial's parameter module toward ``1.0``
   for lighter meshes. Every
   segmenter declares its own labels through
   :class:`~monai_physio.AnatomyTaxonomy`, so downstream grouping and USD
   materials follow automatically — see :doc:`api/segmentation/index`.

Tutorial 5: VTK Surfaces to Animated USD
========================================

Script
   ``tutorials/tutorial_05_heart_vtk_to_usd.py``

   ``tutorials/tutorial_05_duke_heart_vtk_to_usd.py`` — the 4D counterpart,
   animating Tutorial 4 (duke heart)'s per-phase surfaces. Needs
   Duke-Heart-4DLabelmaps (see `Before You Start`_).

Workflow
   :class:`~monai_physio.WorkflowConvertVTKToUSD`.

Dataset
   Tutorial 4's per-structure ``patient_*.vtp`` surfaces — no image data, no
   download.

Requirements
   CPU only, seconds to run. The cheapest tutorial in the set.

Preview
   .. figure:: assets/tutorial_05_heart_vtk_to_usd.png
      :alt: Cardiac surfaces rendered from the exported USD scene
      :width: 90%

      The exported USD scene, split by anatomy and painted with OmniSurface
      materials.

Inner API usage
   .. code-block:: python

      workflow = WorkflowConvertVTKToUSD(
          input_meshes=meshes,
          usd_project_name=project_name,
          output_directory=output_dir,
          appearance="anatomy",
          static_merge=True,
          separate_by_connectivity=True,
      )
      results = workflow.process()

   Each input surface keeps the structure name that
   :class:`~monai_physio.WorkflowConvertImageToVTK` wrote into its
   ``field_data['SegmentationLabelNames']``. That name becomes the USD prim
   name and, with ``anatomy_type`` left unset, selects the prim's material —
   so the left chambers, right chambers, myocardium and great vessels each get
   their own look rather than one shared heart material.

Run
   .. code-block:: bash

      python tutorials/tutorial_05_heart_vtk_to_usd.py

Outputs
   The USD scene and a rendered screenshot under
   ``tutorials/output/tutorial_05_heart/``.

Adapt to your data
   ``input_meshes`` takes any list of PyVista meshes — pass one per time point,
   in order, for an animated scene instead of a static one (drop
   ``static_merge``), and set ``frames_per_second`` to control playback.
   ``appearance="anatomy"`` binds per-organ materials through
   :class:`~monai_physio.USDAnatomyTools`; set ``anatomy_type`` to force one
   palette onto every object, or ``object_names`` to name the prims yourself.
   For file-in, file-out conversion without Python, see
   :doc:`cli_scripts/vtk_to_usd`.

Tutorial 6: Create a PCA Shape Model
====================================

Script
   ``tutorials/tutorial_06_heart_create_statistical_model.py``

   ``tutorials/tutorial_06_lung_create_statistical_model.py``

   ``tutorials/tutorial_06_duke_heart_create_statistical_model.py`` — builds the
   cardiac model the ``duke_heart`` surrogate chain trains against. Needs
   Duke-Heart-4DLabelmaps (see `Before You Start`_).

Workflow
   :class:`~monai_physio.WorkflowCreateStatisticalModel`; the lung variant
   first builds an unbiased atlas with
   :class:`~monai_physio.WorkflowCreateMeanSurface`.

Dataset
   KCL-Heart-Model (auto-download) for the heart. The lung variant starts from
   raw DIR-Lab volumes, segmenting each case's T70 phase itself.

Requirements
   The heart variant is CPU-only and quick. **The lung variant is the slowest
   of Tutorials 1-7**: one GPU segmentation per case, then a deformable
   registration per case per atlas iteration. Every intermediate is cached, so
   a re-run costs almost nothing.

Preview
   .. figure:: assets/tutorial_06_heart_modes_of_variation.png
      :alt: Cardiac shape model modes of variation
      :width: 90%

      Heart model: the mean shape at ±2σ along its leading modes.

   .. figure:: assets/tutorial_06_lung_modes_of_variation.png
      :alt: Lung shape model modes of variation
      :width: 90%

      The same decomposition for the lung population.

Inner API usage
   .. code-block:: python

      mean_workflow = WorkflowCreateMeanSurface(surfaces=sample_surfaces)
      mean_workflow.set_number_of_iterations(mean_surface_iterations)
      reference_surface = mean_workflow.process()["mean_surface"]

      workflow = WorkflowCreateStatisticalModel(
          sample_meshes=sample_surfaces,
          reference_mesh=reference_surface,
          number_of_pca_components=number_of_pca_components,
      )
      result = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_06_heart_create_statistical_model.py
      python tutorials/tutorial_06_lung_create_statistical_model.py

Outputs
   ``pca_model.json``, ``pca_mean_surface.vtp``, the ±2σ mode surfaces and
   their renders, under ``tutorials/output/tutorial_06_{heart,lung}/``. The
   lung variant also leaves its per-case segmentations there, which Tutorial 8
   reuses.

Adapt to your data
   The workflow wants a population of meshes plus one reference; point
   ``sample_meshes`` at your own cohort and let
   :class:`~monai_physio.WorkflowCreateMeanSurface` build the reference when no
   natural template exists. ``number_of_pca_components`` trades fidelity
   against cohort size — you need more subjects than modes. The saved
   ``pca_model.json`` is the portable artifact: Tutorials 7 and 8 and
   :doc:`cli_scripts/create_statistical_model` all speak it.

Tutorial 7: Fit the Shape Model to a Patient
============================================

Script
   ``tutorials/tutorial_07_heart_fit_statistical_model_to_patient.py``

   ``tutorials/tutorial_07_lung_fit_statistical_model_to_patient.py``

   ``tutorials/tutorial_07_duke_heart_fit_statistical_model_to_patient.py`` —
   fits the Tutorial 6 (duke heart) model. Needs Duke-Heart-4DLabelmaps (see
   `Before You Start`_).

Workflow
   :class:`~monai_physio.WorkflowFitStatisticalModelToPatient`.

Dataset
   Tutorial 6's model plus one patient scan. The lung variant fits to
   ``Chest-CT`` — an ungated, single-acquisition chest CT, the kind a
   patient-specific model is normally fitted to. See
   ``data/Chest-CT/README.md`` for the data source and required citation.

Requirements
   One segmentation pass plus a PCA-constrained fit; GPU recommended for the
   segmentation, and no registration over time.

Preview
   .. figure:: assets/tutorial_07_heart_in_noncontrast_ct.gif
      :alt: Fitted heart model overlaid on a non-contrast CT
      :width: 90%

      The heart model fitted to a non-contrast scan.

   .. figure:: assets/tutorial_07_lung.gif
      :alt: Fitted lung model on the ungated Chest-CT scan
      :width: 90%

      The lung model fitted to the ungated ``Chest-CT`` volume.

Inner API usage
   .. code-block:: python

      workflow = WorkflowFitStatisticalModelToPatient(
          template_model=pca_mean,
          patient_models=[patient_surface],
          patient_image=patient_image,
          patient_labelmap=patient_labelmap,
      )
      workflow.set_use_pca_registration(
          use_pca_registration=True,
          pca_model=pca_model,
      )
      result = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_07_heart_fit_statistical_model_to_patient.py
      python tutorials/tutorial_07_lung_fit_statistical_model_to_patient.py

Outputs
   The registered template surface, the fitted mesh, and — the piece the rest
   of the pipeline needs — ``*_registered_coefficients.json``, the patient's
   position in shape space. Under
   ``tutorials/output/tutorial_07_{heart,lung}/``.

Adapt to your data
   Set the patient image path and keep the segmenter consistent with the one
   that built the model. ``labelmap_interior_object_ids`` (heart) tells the fit
   which labels are interior structures — those ids are TotalSegmentator's
   chamber labels, so change them if you change segmenter. Turn
   ``set_use_pca_registration`` off to fall back to an unconstrained
   template-to-patient fit when you have no model. The CLI equivalent is
   :doc:`cli_scripts/fit_statistical_model_to_patient`.

Tutorial 8: Propagate the Shape Model Through 4D
================================================

Script
   ``tutorials/tutorial_08_lung_fit_model_to_4d_patients.py``

   ``tutorials/tutorial_08_duke_heart_fit_model_to_4d_patients.py`` — the same
   fit-then-propagate pass over cardiac phases, using
   :class:`~monai_physio.RegisterModelsDistanceMaps` in place of the image
   registration. Needs Duke-Heart-4DLabelmaps (see `Before You Start`_).

Workflow
   :class:`~monai_physio.WorkflowFitStatisticalModelToPatient` at the reference
   phase, then :class:`~monai_physio.WorkflowReconstructHighres4DCT` to carry
   the fitted surface through every other phase.

Dataset
   DIR-Lab, plus Tutorial 6 (lung)'s model. Tutorial 2's finetuned distance-map
   ICON weights are used by the model fit when present; without them the
   tutorial warns and fits with the stock uniGradICON weights.

Requirements
   GPU required, and the heaviest registration workload in the set: one
   segmentation and one fit per case, plus one registration per phase per case.

Preview
   .. figure:: assets/tutorial_08_lung.gif
      :alt: Fitted lung shape model carried through every respiratory phase
      :width: 90%

      The fitted shape-model surface propagated across the phases of a DIR-Lab
      case.

   .. figure:: assets/tutorial_08_duke_heart_def_mag.gif
      :alt: Deformation magnitude over the propagated heart surface
      :width: 90%

      The ``duke_heart`` variant, coloured by deformation magnitude across the
      cardiac phases.

Inner API usage
   .. code-block:: python

      fit_workflow = WorkflowFitStatisticalModelToPatient(
          template_model=pca_mean_surface,
          patient_models=[lung_surface],
          patient_image=reference_image,
          patient_labelmap=lung_labelmap,
      )
      fit_workflow.set_use_pca_registration(True, pca_model=pca_model)

      reg_workflow = WorkflowReconstructHighres4DCT(
          time_series_images=time_series,
          reference_image=reference_image,
          reference_time_frame=phase_ids.index(reference_phase),
          register_reference_time_frame_to_reference_image=False,
          registration_method=registration_method,
      )

Run
   .. code-block:: bash

      python tutorials/tutorial_08_lung_fit_model_to_4d_patients.py

Outputs
   Per case, under ``tutorials/output/tutorial_08_lung/<case>/``: the fitted
   reference surface, its PCA coefficients, and one warped surface plus
   forward/inverse transform per phase. Those per-phase surfaces are exactly
   the training set Tutorial 9 consumes.

Adapt to your data
   Point ``data_dir`` at a directory of per-case 4D series and set
   ``reference_phase`` to the phase your model was built at; the case and phase
   file patterns are two globs near the top of the script. Everything is cached
   per case, so adding a subject re-runs only that subject.

Tutorial 9: Train a PhysicsNeMo Surrogate
=========================================

Script
   ``tutorials/tutorial_09_lung_train_physicsnemo_mgn.py``

   ``tutorials/tutorial_09_duke_heart_train_physicsnemo_mgn.py`` — trains the
   cardiac network Tutorial 13 uses for its heartbeat. Needs
   Duke-Heart-4DLabelmaps (see `Before You Start`_).

Workflow
   :class:`~monai_physio.WorkflowTrainPhysicsNeMo` driving
   :class:`~monai_physio.TrainPhysicsNeMoMGN`, then
   :class:`~monai_physio.WorkflowInferPhysicsNeMo` and
   :class:`~monai_physio.WorkflowInferMovement` to score the held-out case. A
   fully connected :class:`~monai_physio.TrainPhysicsNeMoMLP` method is a
   drop-in replacement; no separate tutorial ships for it.

Dataset
   Tutorial 8's per-phase surfaces and Tutorial 6 (lung)'s mean surface. The
   tutorial writes one JSON manifest per case, plus the per-vertex displacement
   targets those manifests point at.

Requirements
   GPU, plus the optional extra::

      pip install "monai-physio[physicsnemo]"
      pip install torch-geometric

   Python >= 3.11. 1500 epochs by default.

Preview
   .. figure:: assets/tutorial_09_lung_motion.gif
      :alt: Predicted lung motion across the respiratory cycle
      :width: 90%

      The held-out lung case, predicted at every stage by the trained network.

   .. figure:: assets/tutorial_09_lung_rmse.gif
      :alt: Per-vertex RMSE of the predicted lung surface
      :width: 90%

      The same surface coloured by per-vertex error against the registration
      that produced the training data.

   .. figure:: assets/tutorial_09_lung_deformation_magnitude.gif
      :alt: Deformation magnitude over the lung surface
      :width: 90%

      Deformation magnitude, which is what the error above should be read
      against — the largest errors sit where the motion is largest.

   .. figure:: assets/tutorial_09_duke_heart_motion.gif
      :alt: Predicted heart motion across the cardiac cycle
      :width: 90%

      The ``duke_heart`` variant over a cardiac cycle, with its own RMSE and
      deformation-magnitude captures in
      ``tutorial_09_duke_heart_rmse.gif`` and
      ``tutorial_09_duke_heart_deformation_magnitude.gif``.

Inner API usage
   .. code-block:: python

      training_method = TrainPhysicsNeMoMGN()
      training_method.set_epochs(epochs)
      training_method.set_processor_size(processor_size)

      train_workflow = WorkflowTrainPhysicsNeMo(
          train_manifests=train_manifests,
          val_manifests=val_manifests,
          pca_mean_mesh=ssm_mean_surface_file,
          output_directory=output_dir,
          training_method=training_method,
      )
      train_result = train_workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_09_lung_train_physicsnemo_mgn.py

Outputs
   ``mgn_stage_model.pt``, its metadata and loss/RMSE logs, in the shared
   weights directory Tutorial 10 reads
   (``tutorials/network_weights/physicsnemo_mgn_lung_motion/``, a fresh sibling
   of it when resuming). The per-case manifests and the held-out evaluation
   under ``eval_mgn/`` stay in ``tutorials/output/tutorial_09_lung_mgn/``.

Adapt to your data
   The contract is the manifest, not the tutorial. Each JSON names a reference
   mesh, a PCA coefficient file, a ``target_array`` name and one entry per
   phase; the workflow reads that array verbatim, so the target can be
   displacement — as here — or any per-point quantity of any width. Produce
   manifests in that shape from your own pipeline and nothing else changes.
   See :doc:`api/physicsnemo/index` for the schema, and
   :doc:`cli_scripts/train_physicsnemo` for the command-line path.

Tutorial 10: Predict Motion With the Surrogate
==============================================

Script
   ``tutorials/tutorial_10_lung_infer_physicsnemo_mgn.py``

   ``tutorials/tutorial_10_duke_heart_infer_physicsnemo_mgn.py`` — the same
   prediction over a cardiac cycle. Needs Duke-Heart-4DLabelmaps (see
   `Before You Start`_).

Workflow
   :class:`~monai_physio.WorkflowInferPhysicsNeMo` for the raw prediction,
   :class:`~monai_physio.WorkflowInferMovement` to turn it back into geometry,
   and :class:`~monai_physio.WorkflowConvertVTKToUSD` to export it.

Dataset
   Tutorial 8's fitted surfaces for one case, and Tutorial 9's checkpoint.

Requirements
   The ``[physicsnemo]`` extra; otherwise trivial — one forward pass per stage
   replaces the per-phase registration solve that produced the training data.

Preview
   .. figure:: assets/tutorial_10_lung_motion_usd.gif
      :alt: Animated USD of the predicted lung motion
      :width: 90%

      The exported USD scene, played back over the respiratory cycle — every
      frame a forward pass rather than a registration solve.

   .. figure:: assets/tutorial_10_duke_heart_motion_usd.gif
      :alt: Animated USD of the predicted heart motion
      :width: 90%

      The ``duke_heart`` variant over a cardiac cycle.

Inner API usage
   .. code-block:: python

      infer_workflow = WorkflowInferPhysicsNeMo(
          model_directory=model_dir,
          epoch=epoch,
      )
      infer_result = WorkflowInferMovement(infer_workflow).process_time_series(
          shape_parameters=pca_file,
          stages=stages,
          output_directory=output_dir,
          fitted_reference_mesh=fitted_reference_mesh_file,
          ground_truth=phase_files,
          reference_image=itk.imread(str(reference_ct_file)),
          warp_interpolation="linear",
          warp_background_value=-1000.0,
          usd_project_name=f"{case_id}_mgn_motion",
          anatomy_type="lung",
      )

Run
   .. code-block:: bash

      python tutorials/tutorial_10_lung_infer_physicsnemo_mgn.py

Outputs
   One predicted surface and one warped CT per stage, and one animated USD
   across all of them, under ``tutorials/output/tutorial_10_lung_mgn/<case>/``.
   The acquired phase surface is rendered beside the prediction for visual
   comparison; scoring it is Tutorial 11's job.

Adapt to your data
   Change ``case_id`` to predict a different subject, or pass ``stages`` that
   were never acquired — which is the point of the surrogate. Omit
   ``reference_image`` to write meshes without warping anything. Use
   :class:`~monai_physio.WorkflowInferPhysicsNeMo` on its own to get the raw
   target array when your model predicts something other than displacement.

Tutorial 11: Score the Surrogate Against the Images
===================================================

Script
   ``tutorials/tutorial_11_lung_evaluate_physicsnemo.py``

   ``tutorials/tutorial_11_duke_heart_evaluate_physicsnemo.py``

Workflow
   :class:`~monai_physio.WorkflowEvaluateMovement`, driving
   :class:`~monai_physio.WorkflowInferMovement` and, for the lung variant,
   :class:`~monai_physio.SegmentNVSegmentCTMRI`.

Dataset
   The gated sequence itself — DIR-Lab for the lung, Duke-Heart-4DLabelmaps for
   the heart — plus Tutorial 8's fitted surface and Tutorial 9's checkpoint for
   the held-out case.

Requirements
   The ``[physicsnemo]`` extra. The lung variant also segments every gated frame
   on first run, so it needs a GPU and the segmentation weights; the labelmaps
   are cached, and a re-run skips them.

Preview
   .. figure:: assets/tutorial_11_lung_volumes.png
      :alt: Acquired and predicted lobe volumes across the respiratory cycle
      :width: 90%

      ``volume_vs_stage.png`` for the held-out lung case: acquired volume solid,
      predicted dashed, one pair per lobe across every gated stage.

   .. figure:: assets/tutorial_11_lung_stats.png
      :alt: Per-lobe volume difference and surface RMSE for the lung case
      :width: 90%

      The same run summarised per lobe. No Dice column — see the note below.

   .. figure:: assets/tutorial_11_duke_heart_stats.png
      :alt: Per-chamber Dice, volume difference and surface RMSE for the heart
      :width: 90%

      The ``duke_heart`` variant, which does report Dice per chamber, alongside
      its own ``tutorial_11_duke_heart_volumes.png``.

Inner API usage
   .. code-block:: python

      evaluate = WorkflowEvaluateMovement(
          movement_workflow=WorkflowInferMovement(infer_workflow),
          label_names=lobe_names,
      )
      result = evaluate.process(
          case_id=case_id,
          shape_parameters=pca_file,
          fitted_reference_mesh=fitted_reference_mesh_file,
          reference_labelmap=itk.imread(str(reference_labelmap_file)),
          ground_truth_labelmaps=ground_truth_labelmaps,
          output_directory=output_dir,
          evaluation_spacing_mm=2.0,
          report_dice=False,
      )

Run
   .. code-block:: bash

      python tutorials/tutorial_11_lung_evaluate_physicsnemo.py

Outputs
   ``evaluation_report.md``, ``evaluation_metrics.csv`` and
   ``volume_vs_stage.png`` under ``tutorials/output/tutorial_11_lung/<case>/``,
   carrying volume difference and surface RMSE per lobe at every gated stage;
   the duke variant adds Dice per chamber. The plot traces each structure's
   acquired and predicted volume across the stages. Report and CSV both record
   the hold-out case name, its shape parameters, and the network weights path
   with its dates, so a number can be traced back to the run that produced it.

   The lung variant passes ``report_dice=False``. Dice is an overlap fraction,
   so a lobe that moves a few millimeters against its own bulk scores over 0.96
   however well or badly the motion is predicted; the column would describe the
   lobe rather than the model. Chambers change shape enough over a heartbeat for
   it to discriminate, so the duke variant keeps it.

Adapt to your data
   Change ``LOBE_LABEL_IDS`` (or ``HEART_LABEL_IDS``) to score a different set
   of structures — any label your segmenter writes and your reference frame
   contains. Raise ``evaluation_spacing_mm`` if the deformation fields do not
   fit in memory; lower it to resolve a thin wall, at the cost of its cube.

Tutorial 12: The Whole Inference Pipeline in One Script
=======================================================

Script
   ``tutorials/tutorial_12_lung_end_to_end_inference.py``

   ``tutorials/tutorial_12_duke_heart_end_to_end_inference.py``

Workflow
   :class:`~monai_physio.WorkflowConvertImageToVTK` (lung) or
   :class:`~monai_physio.ContourTools` (heart),
   :class:`~monai_physio.WorkflowFitStatisticalModelToPatient`, then
   :meth:`~monai_physio.WorkflowInferMovement.process_time_series`.

Dataset
   The gated sequence alone — DIR-Lab for the lung, Duke-Heart-4DLabelmaps for
   the heart — plus the Tutorial 6 shape model and the Tutorial 9 checkpoint.
   Unlike Tutorial 10, nothing is read from Tutorial 8: this script fits the
   model to the patient itself, so the chain from image to animation runs in one
   place.

Requirements
   The ``[physicsnemo]`` extra. The output directory is emptied at the start of
   every run, so nothing is reused and the reported runtimes are the whole
   pipeline's. Neither variant registers a phase — that is what the network
   replaces, and it is why this runs in minutes where Tutorial 8 runs in hours.

Preview
   .. figure:: assets/tutorial_12_lung.gif
      :alt: Lung motion predicted end-to-end from a gated series
      :width: 90%

      The whole chain on one DIR-Lab case: segment, fit, infer, animate — no
      phase registered anywhere in it.

   .. figure:: assets/tutorial_12_duke_heart.gif
      :alt: Heart motion predicted end-to-end from gated labelmaps
      :width: 90%

      The ``duke_heart`` variant, starting from gated labelmaps instead of CT.

Inner API usage
   .. code-block:: python

      # The fit puts the model in this patient: coefficients condition the
      # network, and the fitted surface is what its displacements move.
      fit = WorkflowFitStatisticalModelToPatient(
          template_model=pca_mean_surface,
          patient_models=[lung_surface],
          patient_image=reference_image,
          patient_labelmap=lung_labelmap,
      )
      fit.set_use_pca_registration(
          use_pca_registration=True,
          pca_model=pca_model,
          number_of_pca_components=6,
          use_surface=False,
      )
      fit_result = fit.process()

      infer_result = WorkflowInferMovement(infer_workflow).process_time_series(
          shape_parameters=pca_coefficients_file,
          stages=stages,
          output_directory=output_dir,
          fitted_reference_mesh=fitted_reference_mesh_file,
          reference_image=reference_image,
          usd_project_name=f"{case_id}_mgn_motion",
          anatomy_type="lung",
      )

Run
   .. code-block:: bash

      python tutorials/tutorial_12_lung_end_to_end_inference.py

Outputs
   Under ``tutorials/output/tutorial_12_lung/<case>/`` (or
   ``tutorial_12_duke_heart``): the patient's fitted
   ``<case>_ssm_surface.vtp`` and ``<case>_ssm_pca_coefficients.json``, one
   predicted ``*_pred.vtp`` surface and one ``*_warped.mha`` volume per stage,
   ``<case>_mgn_motion.usd`` animating the whole cycle, and
   ``<case>_runtimes.csv`` timing each step of the run.

Adapt to your data
   Point the script at any case of the same cohort by changing ``case_id``; the
   stages come from the filenames, so a sequence with a different number of
   phases needs no other change. To predict stages the acquisition never
   sampled, pass your own ``stages`` list — the network is continuous in stage,
   and nothing downstream requires a matching image.

Tutorial 13: Breathe and Beat a Static Clinical CT
==================================================

Script
   ``tutorials/tutorial_13_heart_and_lung_motion.py``

Workflow
   :class:`~monai_physio.WorkflowInferMovement` over both Tutorial 9 networks,
   :class:`~monai_physio.WorkflowFitStatisticalModelToPatient` for the heart fit,
   and :class:`~monai_physio.ConvertVTKToUSD` with
   :class:`~monai_physio.USDAnatomyTools` for the animation.

Dataset
   ``data/Chest-CT/Chest-CT.mha``, one ungated breath-hold scan, plus Tutorial 7
   (lung)'s fit of it and both Tutorial 9 checkpoints. No 4D acquisition is
   involved: every deformation comes from a network, none from a registration.
   See ``data/Chest-CT/README.md`` for the data source and required citation.

Requirements
   The ``[physicsnemo]`` extra, and Simpleware Medical for the heart
   segmentation. Both segmentations and the heart fit are cached, so a re-run
   goes straight to inference. Budget disk: 100 combined frames, each with its
   own warped CT and labelmap, come to roughly 43 GB.

Preview
   .. figure:: assets/tutorial_13_combined_motion_usd.gif
      :alt: Combined heart and lung surface motion on a static clinical CT
      :width: 90%

      ``heart_and_lung_motion.usd``: one ungated breath-hold scan, breathing and
      beating at once, with every deformation coming from a network and none
      from a registration.

   .. figure:: assets/tutorial_13_combined_motion_ct.gif
      :alt: The static CT warped by the same combined heart and lung motion
      :width: 90%

      The same per-frame deformation applied to the CT itself — the voxels move
      with the surfaces, so the scan breathes and beats along with them.

Inner API usage
   .. code-block:: python

      infer = WorkflowInferMovement(
          WorkflowInferPhysicsNeMo(model_directory=lung_model_dir)
      )
      # "forward" moves mesh vertices; "inverse" is what resampling an image
      # into the stage's frame needs.
      field = infer.create_deformation_field(
          shape_parameters=lung_coefficients_file,
          stage=0.0,
          reference_image=patient_image,
          fitted_reference_mesh=lung_fitted_reference_mesh_file,
          direction="forward",
      )
      transform = TransformTools().smooth_deformation_field_transform(
          field["deformation_field"], 15.0, field["weight_image"]
      )

Run
   .. code-block:: bash

      python tutorials/tutorial_13_heart_and_lung_motion.py

Outputs
   Under ``tutorials/output/tutorial_13_heart_and_lung/``: one 4D USD per rhythm
   (``breathing_lungs.usd``, ``beating_heart.usd``), 100 combined frames as VTP
   plus ``heart_and_lung_motion.usd`` split by anatomy and painted with organ
   materials, and the CT and labelmap warped by the same per-frame deformation.

Adapt to your data
   Point ``patient_image_file`` at your own chest CT and rerun Tutorial 7 (lung)
   on it to get the lung fit; the heart fit happens inside this script. Change
   ``cardiac_cycles_per_phase`` to re-time the heartbeat against the breath, and
   the two ``*_sigma_mm`` values to change how far each rhythm's surface motion
   is carried into the surrounding tissue.

Tutorial 14: Sweep the Shape Parameters
=======================================

Script
   ``tutorials/tutorial_14_lung_shape_parameter_sweep.py`` (DIR-Lab)

   ``tutorials/tutorial_14_duke_heart_shape_parameter_sweep.py``
   (Duke-Heart-4DLabelmaps)

Workflow
   :class:`~monai_physio.WorkflowInferPhysicsNeMo` driving
   :class:`~monai_physio.InferPhysicsNeMoMGN`, scored by
   :class:`~monai_physio.WorkflowEvaluateMovement` once per grid point.

Dataset
   The held-out case of Tutorial 9, plus its Tutorial 8 fit and the Tutorial 9
   checkpoint.

Requirements
   The ``[physicsnemo]`` extra plus ``torch-geometric``, a GPU, and the
   segmentation weights --- every grid point is scored against independently
   segmented frames, exactly as Tutorial 11 scores its one fit.

Preview
   .. figure:: assets/tutorial_14_duke_heart_parameter_sweep.png
      :alt: Displacement error across the shape-parameter sweep grid
      :width: 90%

      The Duke Heart variant's sweep summary: pooled displacement error at
      every grid point, with the all-zero combination as the unperturbed
      baseline. No lung-variant figure exists yet.

What it does
   Tutorial 11 scores the inferred motion at the one point in shape space the
   statistical-model fit happened to land on. This tutorial sweeps that point:
   it perturbs the first few PCA coefficients over a grid, re-infers the whole
   cycle at every combination, and scores each the way Tutorial 11 scores its
   single fit.

   Only the coefficients handed to the network change. The reference anatomy
   stays the Tutorial 8 fitted surface at every grid point, so what the
   perturbation moves is the displacement field the MeshGraphNet infers, not
   the patient's own shape. The sweep therefore isolates the network's
   sensitivity to its shape conditioning. Because the reference surface, the
   reference labelmap and the acquired frames are identical across the grid,
   every combination is scored on the same evaluation grid and the figures are
   directly comparable point to point.

   The all-zero combination is in the grid, so the unperturbed score comes out
   of the same code path as every perturbed one.
   ``number_of_modes_to_vary``, ``perturbation_range`` and
   ``perturbation_step`` set the grid; the default is ``5 ** 2 = 25``
   combinations, each costing one Tutorial 11 run.

   Read the sweep by the displacement columns rather than by Dice: a perturbed
   coefficient can leave a structure the same size in the same place and still
   move every point of it wrong, which the labelmap metrics cannot see.

Run
   .. code-block:: bash

      python tutorials/tutorial_14_lung_shape_parameter_sweep.py

      python tutorials/tutorial_14_duke_heart_shape_parameter_sweep.py

Outputs
   Under ``tutorials/output/tutorial_14_<anatomy>/<case>/``:
   ``shape_sweep_metrics.csv`` with one row per combination, stage and
   structure, ``shape_sweep_summary.csv`` with one row per combination carrying
   that combination's pooled displacement error, and one ``combo_<NNN>/``
   directory per grid point holding its own Tutorial 11 style report,
   predicted surfaces and warped labelmaps.

Adapt to your data
   ``number_of_modes_to_vary``, ``perturbation_step`` and
   ``evaluation_spacing_mm`` are the cost knobs --- the grid is exponential in
   the first. Point ``case_id`` at a different subject to sweep that one
   instead.

Tutorial 15: Leave-One-Out Cross-Validation
===========================================

Script
   ``tutorials/tutorial_15_lung_leave_one_out.py`` (DIR-Lab)

   ``tutorials/tutorial_15_duke_heart_leave_one_out.py``
   (Duke-Heart-4DLabelmaps)

Workflow
   :class:`~monai_physio.WorkflowCreateMeanSurface` and
   :class:`~monai_physio.WorkflowCreateStatisticalModel` per fold,
   :class:`~monai_physio.WorkflowFitStatisticalModelToPatient` per case,
   :class:`~monai_physio.WorkflowTrainPhysicsNeMo` driving
   :class:`~monai_physio.TrainPhysicsNeMoMGN`, and
   :class:`~monai_physio.WorkflowEvaluateMovement` on the held-out case.

Dataset
   The whole cohort, and nothing else. Tutorials 6, 8 and 9 outputs are reused
   as a cache when they are present, but every fold builds its own shape model,
   its own fits and its own network.

Requirements
   The ``[physicsnemo]`` extra plus ``torch-geometric``. Written for a
   multi-GPU Linux host, though it runs as a single process too.

.. TODO(image): no preview media exists yet for Tutorial 15. Add a
   ``loo_metrics_by_label.png``-derived figure (or similar) here once one is
   captured, matching the ``Preview`` pattern used by the other tutorials.

What it does
   Tutorials 6 through 11 report accuracy for one fixed held-out case, which is
   a single observation: it says nothing about how far the number would move
   had a different patient been held out. This tutorial runs that chain once
   per fold. Each fold rebuilds the PCA model from the population *without* its
   held-out case, refits the cohort to that model, retrains the MeshGraphNet on
   the other cases, infers the held-out case at every acquired stage, and scores
   it against that stage's own ground truth. Rebuilding is the point --- a model
   built once from everyone has already seen every case, so scoring against it
   measures recall rather than generalization.

   ``number_of_leave_one_out_runs`` near the top of each script sets the fold
   count and defaults to 5.

   Two things do not depend on which case is held out --- the segmentations
   and, for the lung, the phase-to-reference image registrations --- so they
   are computed once into ``shared/`` and reused. Hoisting the registrations is
   what makes the lung variant tractable. The Duke variant cannot hoist its
   frame registrations: they warp the fold's own fitted surface, which changes
   with the fold, so a Duke fold costs materially more than a lung one.

Run
   .. code-block:: bash

      # One process
      python tutorials/tutorial_15_lung_leave_one_out.py
      python tutorials/tutorial_15_duke_heart_leave_one_out.py

      # Data-parallel training and rank-split per-case loops
      torchrun --standalone --nproc_per_node=8 \
          tutorials/tutorial_15_lung_leave_one_out.py
      torchrun --standalone --nproc_per_node=8 \
          tutorials/tutorial_15_duke_heart_leave_one_out.py

Outputs
   Under ``tutorials/output/tutorial_15_<anatomy>/``: ``loo_metrics.csv`` with
   every metric row of every fold, ``loo_report.md`` with the per-structure mean
   and standard deviation across folds, ``loo_metrics_by_label.png`` as the
   matching box plot, and one ``fold_<case>/`` directory per fold holding that
   fold's shape model, fits, manifests, weights and evaluation.

Adapt to your data
   Raise ``number_of_leave_one_out_runs`` to the cohort size for a full
   leave-one-out study; the runtime is linear in it. ``epochs`` and
   ``batch_size`` mirror Tutorial 9 so each fold's network is comparable to the
   one that tutorial trains --- lower them for a quicker sweep, at the cost of
   comparability.

Tutorial 16: Build a Volumetric Shape Model of the Myocardium
==============================================================

Script
   ``tutorials/tutorial_16_duke_heart_physics_informed_motion_prep.py``

Workflow
   :class:`~monai_physio.WorkflowCreateStatisticalModel` with
   ``solve_for_surface_pca=False`` on a tetrahedral template built by
   :meth:`~monai_physio.ContourTools.extract_tetrahedra` and
   :meth:`~monai_physio.ContourTools.trim_tetrahedra_to_surface`, then
   :class:`~monai_physio.WorkflowFitStatisticalModelToPatient` per case and
   gated frame.

Dataset
   Duke-Heart-4DLabelmaps, plus Tutorial 4 (duke heart)'s surfaces and,
   optionally, Tutorial 2's finetuned distance-map ICON weights --- the stock
   uniGradICON weights are used when absent, which understates the
   population's variance. Nothing in Tutorials 1 to 15 is modified.

Requirements
   None beyond the base install; Tutorial 17 is what needs PhysicsNeMo. A CUDA
   GPU is required --- every registration runs on one. This is the most
   expensive tutorial in the chain: an atlas pass over the population, one
   deformable registration per case for the model, then one fit plus one
   registration per gated frame per case, against a template far denser than
   the surface one. Every stage is cached, so a re-run only redoes what is
   missing.

.. TODO(image): no preview media exists yet for Tutorial 16. Add a render of
   ``pca_mean_surface.vtp`` or the tetrahedral template once one is captured,
   matching the ``Preview`` pattern used by the other tutorials.

What it does
   Tutorials 9 and 10 predict cardiac motion from a surface shape model, which
   has no interior and so cannot form a deformation gradient --- the quantity
   Tutorial 17's strain energy needs. This tutorial rebuilds the same model
   volumetrically: it builds the unbiased mean surface exactly as Tutorial 6
   does, fills it with tetrahedra (holding every cell above a scaled Jacobian
   of 0.1), decomposes the population into shape modes against that
   tetrahedral template, and fits the model to every case and cardiac phase.
   Because every subject inherits the template's topology, one set of element
   node ids stays valid across the cohort.

   Only the reference mesh has to be volumetric: correspondence is
   established by warping it through each subject's displacement field, which
   carries interior nodes along, while the samples merely supply the distance
   maps that drive the registration.

   The last step writes one training manifest per case, naming each frame's
   displacement from that case's *own* fitted reference rather than the
   population mean --- the physics residual Tutorial 17 trains against is
   measured at the fitted reference's points, and measuring against the mean
   would charge every subject a strain energy for merely being shaped unlike
   it.

Run
   .. code-block:: bash

      python tutorials/tutorial_16_duke_heart_physics_informed_motion_prep.py

Outputs
   Under ``tutorials/output/tutorial_16_duke_heart_physics_informed_motion/``:
   ``ssm_template.vtu`` (the tetrahedral template), ``pca_model.json`` and
   ``pca_mean.vtu`` (the volumetric shape model), ``pca_mean_surface.vtp``
   (its boundary, for display), one ``<case>/`` directory of fitted and
   phase-propagated models per case, and ``manifests/`` for Tutorial 17.

Adapt to your data
   ``ssm_element_size_mm`` in ``parameters_duke_heart_physics_informed.py`` is
   the one number that decides how much of the myocardium the physics term
   ever sees, because ``extract_tetrahedra`` resamples with a vote and drops
   any wall thinner than the element size. Measured against the 208,259 mm^3
   the Duke mean surface encloses, the template holds 99.5% of it at 1.0 mm
   (305,696 nodes), 88.3% at 1.5 mm (100,903 nodes) and 72.1% at 2.0 mm
   (43,826 nodes); the default is 1.5 mm.

Tutorial 17: Train a Mechanics-Aware Cardiac Motion Surrogate
==============================================================

Script
   ``tutorials/tutorial_17_duke_heart_physics_informed_motion_train.py``

Workflow
   :class:`~monai_physio.WorkflowTrainPhysicsNeMo` driving
   :class:`~monai_physio.TrainPhysicsNeMoPhysicsInformedMotion`.

Dataset
   Tutorial 16 output: ``manifests/*_manifest.json``, ``pca_mean.vtu`` and
   ``ssm_template.vtu``.

Requirements
   The ``[physicsnemo]`` extra plus ``torch-geometric``:

   .. code-block:: bash

      pip install "monai-physio[physicsnemo]"
      pip install torch-geometric

   ``physicsnemo.sym``, which supplies ``PhysicsInformer``, ships inside
   ``nvidia-physicsnemo``; no separate install is needed. A CUDA GPU is
   required. The mesh graph is several times larger than Tutorial 9's surface
   one, so expect to lower ``batch_size`` and leave gradient checkpointing on;
   training the ablation baseline (on by default) doubles the run.

.. TODO(image): no preview media exists yet for Tutorial 17. Add the
   ``training_losses.png`` comparison of both runs' loss curves once one is
   captured, matching the ``Preview`` pattern used by the other tutorials.

What it does
   Trains the same MeshGraphNet architecture Tutorial 9 trains, on the
   volumetric shape model Tutorial 16 built, but scores it against a
   neo-Hookean strain energy as well as against measured displacement.
   Tutorial 9's loss is L2 on displacement alone, so it has no opinion about
   motion no myocardium could undergo --- an element may inflate, thin past
   what tissue allows, or invert outright, and the loss only notices to the
   extent the vertices land in the wrong place. A strain energy prices exactly
   those deformations. The loss becomes::

       data + lambda_physics * (strain_energy + incompressibility)

   with the strain energy of a compressible neo-Hookean solid,

       W = (mu / 2)(I1 - 3) - mu ln(J) + (lambda / 2) ln(J)^2

   evaluated from the deformation gradient of each tetrahedron via PhysicsNeMo
   Sym's least-squares gradient reconstruction, its method for unstructured
   meshes.

   The residual is measured against each case's own fitted reference model,
   not the population mean: the targets are displacements from that
   reference, so it is the undeformed state. Measuring against the mean would
   confuse variation between subjects with deformation within one.

   By default a second model trains on identical data with
   ``lambda_physics = 0``. That ablation is the only comparison that isolates
   the physics term --- comparing against Tutorial 9 instead would confound it
   with the change from a surface shape model to a volumetric one --- and
   Tutorial 18 scores the two against each other.

Run
   .. code-block:: bash

      python tutorials/tutorial_17_duke_heart_physics_informed_motion_train.py

Outputs
   Under ``tutorials/network_weights/physicsnemo_physics_informed_motion_duke_heart/``
   (and ``..._ablation/`` for the comparison model): the trained checkpoint
   (``physics_informed_motion_stage_model.pt``), per-epoch loss
   (``training_losses.json``), and intermittent validation RMSE
   (``training_validation_rmse.csv``). Under
   ``tutorials/output/tutorial_16_duke_heart_physics_informed_motion/``:
   ``training_losses.png``, both runs' loss curves.

Adapt to your data
   ``mu_kpa`` and ``lambda_lame_kpa`` in
   ``parameters_duke_heart_physics_informed.py`` are the tissue's constitutive
   parameters, and ``lambda_physics`` weighs the residual against a
   displacement loss scored in different units --- sweep it rather than
   trusting it. Set ``train_ablation_baseline`` to ``False`` to skip the
   comparison model.

Tutorial 18: Predict Myocardial Motion and the Stress It Implies
=================================================================

Script
   ``tutorials/tutorial_18_duke_heart_physics_informed_motion_infer.py``

Workflow
   :class:`~monai_physio.WorkflowInferMovement` driving
   :class:`~monai_physio.WorkflowInferPhysicsNeMo` for both models,
   :class:`~monai_physio.WorkflowEvaluateMovement` to score them, and
   :class:`~monai_physio.ConvertVTKToUSD` (with
   :meth:`~monai_physio.ConvertVTKToUSD.compute_von_mises_stress`) for the
   export.

Dataset
   Tutorial 16 output (fitted models, manifests, template), Tutorial 17
   output (the trained networks), and the held-out case's
   ``data/Duke-Heart-4DLabelmaps/<case>/*_labelmap.nii.gz``.

Requirements
   The ``[physicsnemo]`` extra plus ``torch-geometric``, and a CUDA GPU ---
   same as Tutorial 17. Far cheaper than Tutorials 16 and 17: two inference
   passes and two evaluations over the held-out case's gated frames, then one
   stress evaluation per frame.

.. TODO(image): no preview media exists yet for Tutorial 18. Add a still or
   clip of ``heart_physics_informed_motion.usd`` colored by von Mises stress
   once one is captured, matching the ``Preview`` pattern used by the other
   tutorials.

What it does
   Scores the mechanics-aware surrogate Tutorial 17 trained, and turns what it
   predicts into something a mechanics-aware model can say and a data-only one
   cannot: a stress field.

   It predicts the held-out case's motion with the physics-informed network
   and again with the ``lambda_physics = 0`` ablation Tutorial 17 trained on
   identical data --- the only honest comparison, since scoring against
   Tutorial 9 would confound the physics term with the change from a surface
   shape model to a volumetric one. Both are scored with
   ``WorkflowEvaluateMovement`` into one table comparing them per phase,
   including the minimum Jacobian, which says whether the predicted motion
   ever turns tissue inside out.

   It then derives the Cauchy stress each predicted deformation implies, from
   the same neo-Hookean law the training loss used, and reduces it to von
   Mises stress --- the tutorial supplies only the 9-component stress tensor,
   and ``ConvertVTKToUSD.compute_von_mises_stress`` derives the scalar ---
   before exporting the animated result to USD with a stress colormap.

   The success criterion worth holding this to is not that the
   physics-informed model wins on RMSE. It is that it is no worse while
   keeping every element's Jacobian positive: a strain energy is a prior, and
   a prior that improved the data fit would be suspicious. Read
   ``mechanics_comparison.csv`` and report what it says.

Run
   .. code-block:: bash

      python tutorials/tutorial_18_duke_heart_physics_informed_motion_infer.py

Outputs
   Under ``tutorials/output/tutorial_18_duke_heart_physics_informed_motion/<case>/``:
   ``mechanics_comparison.csv`` (per-phase scores, both models),
   ``physics_informed/`` and ``ablation/`` (each model's predictions and
   report), ``stress/<frame>_stress.vtu`` (predicted motion carrying stress),
   and ``heart_physics_informed_motion.usd`` (animated, colored by von Mises
   stress).

Adapt to your data
   Point ``case_id`` at a different held-out subject to score that one
   instead; everything it reads comes from Tutorials 16 and 17, so no other
   change is needed.


Where to Go Next
================

- :doc:`viewing_usd` — installing an Omniverse Kit application and opening the
  scenes these tutorials produce.
- :doc:`cli_scripts/byod_tutorials` — running the workflows on your own DICOM,
  NRRD or VTK data, including directory layout and conversion.
- :doc:`api/index` — every workflow, segmenter, registrar and utility class.
- :doc:`architecture` — how the workflow layer fits together and where to
  extend it.
- :doc:`testing` — ``tests/test_tutorials.py`` runs these scripts end-to-end
  behind the ``--run-tutorials`` flag.
