.. MONAI Physio documentation master file

.. title:: MONAI Physio Documentation

.. raw:: html

   <section class="mphysio-hero">
     <div class="mphysio-hero__brand">
       <img src="_static/nvidia-logo.svg" alt="NVIDIA logo">
     </div>
     <p class="mphysio-kicker">MONAI Physio</p>
     <h1>Build animated medical USD workflows for NVIDIA Omniverse</h1>
     <p>
       MONAI Physio is a collection of methods, workflows, tutorials, and CLI
       tools for creating personalized physiological digital twins from 3D
       medical images. Install it, clone the repository for the tutorial
       scripts, and work through the cards below - each tutorial runs on
       downloadable data and ends with the constants to change for your own.
       It is not validated for clinical use: MONAI Physio is a research and
       visualization toolkit, not a medical device, and must not be used for
       diagnosis, treatment planning, or clinical decision-making.
     </p>

     <p class="mphysio-hero__version">Version {{ mphysio_project_version }}</p>
   </section>

   <section class="mphysio-card-grid" aria-label="Tutorial cards">
     <a class="mphysio-card" href="installation.html">
       <span class="mphysio-card__number">00</span>
       <h2>Install and Clone</h2>
       <p>Install the package, then clone the repository - the tutorial scripts do not ship in the wheel.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-1-gated-4d-ct-to-animated-usd">
       <span class="mphysio-card__number">01</span>
       <h2>Gated 4D CT to Animated USD</h2>
       <p>Segment, register and assemble a 4D CT series into an animated OpenUSD scene.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-2-finetune-icon-registration">
       <span class="mphysio-card__number">02</span>
       <h2>Finetune ICON Registration</h2>
       <p>Adapt uniGradICON to your own cohort and measure what the finetuning bought you.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-3-reconstruct-high-resolution-4d-ct">
       <span class="mphysio-card__number">03</span>
       <h2>Reconstruct High-Resolution 4D CT</h2>
       <p>Register every phase to one reference and reconstruct the series at its resolution.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-4-ct-segmentation-to-vtk-surfaces">
       <span class="mphysio-card__number">04</span>
       <h2>CT Segmentation to VTK Surfaces</h2>
       <p>Segment one CT phase and export patient anatomy as VTK PolyData surfaces.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-5-vtk-surfaces-to-animated-usd">
       <span class="mphysio-card__number">05</span>
       <h2>VTK Surfaces to Animated USD</h2>
       <p>Convert meshes into a time-sampled USD scene for Omniverse playback.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-6-create-a-pca-shape-model">
       <span class="mphysio-card__number">06</span>
       <h2>Create a PCA Shape Model</h2>
       <p>Turn a population of meshes into a statistical shape model and its modes.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-7-fit-the-shape-model-to-a-patient">
       <span class="mphysio-card__number">07</span>
       <h2>Fit the Shape Model to a Patient</h2>
       <p>Fit the shape model to one ungated clinical scan, PCA coefficients and all.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-8-propagate-the-shape-model-through-4d">
       <span class="mphysio-card__number">08</span>
       <h2>Propagate the Model Through 4D</h2>
       <p>Fit each case at its reference phase and carry the mesh through every phase.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-9-train-a-physicsnemo-surrogate">
       <span class="mphysio-card__number">09</span>
       <h2>Train a PhysicsNeMo Surrogate</h2>
       <p>Train a MeshGraphNet to predict per-vertex motion from shape and phase.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-10-predict-motion-with-the-surrogate">
       <span class="mphysio-card__number">10</span>
       <h2>Predict Motion With the Surrogate</h2>
       <p>Replace the registration solve with one forward pass, then export to USD.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-11-score-the-surrogate-against-the-images">
       <span class="mphysio-card__number">11</span>
       <h2>Score the Surrogate Against the Images</h2>
       <p>Volume and surface RMSE per lobe, plus Dice per chamber, on the held-out case.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-12-the-whole-inference-pipeline-in-one-script">
       <span class="mphysio-card__number">12</span>
       <h2>The Whole Inference Pipeline in One Script</h2>
       <p>Go from a gated series to an animated prediction without registering a single phase.</p>
     </a>
     <a class="mphysio-card" href="tutorials.html#tutorial-13-breathe-and-beat-a-static-clinical-ct">
       <span class="mphysio-card__number">13</span>
       <h2>Breathe and Beat a Static Clinical CT</h2>
       <p>Animate one ungated breath-hold scan with both rhythms, from two networks at once.</p>
     </a>
   </section>

   <section class="mphysio-topic-section" aria-label="Documentation topics">
     <div class="mphysio-section-heading">
       <p class="mphysio-kicker">Documentation</p>
       <h2>Explore the rest of the docs</h2>
     </div>
     <div class="mphysio-topic-grid">
       <a class="mphysio-topic-card" href="installation.html">
         <h3>Installation</h3>
         <p>Set up MONAI Physio with CUDA extras, CPU-only options, and required system tools.</p>
       </a>
       <a class="mphysio-topic-card" href="quickstart.html">
         <h3>Getting Started</h3>
         <p>Run your first workflow and understand the basic CT-to-USD processing path.</p>
       </a>
       <a class="mphysio-topic-card" href="tutorials.html">
         <h3>Tutorials &amp; Examples</h3>
         <p>Runnable scripts covering cardiac, lung, segmentation, and USD tasks, with the inner workflow-class calls each one makes.</p>
       </a>
       <a class="mphysio-topic-card" href="cli_scripts/overview.html">
         <h3>CLI Workflows</h3>
         <p>Use production command-line workflows for conversion, reconstruction, modeling, and USD export.</p>
       </a>
       <a class="mphysio-topic-card" href="viewing_usd.html">
         <h3>Viewing USD Files</h3>
         <p>Set up an Omniverse Kit app with RTX rendering and open the scenes the workflows produce.</p>
       </a>
       <a class="mphysio-topic-card" href="cli_scripts/byod_tutorials.html">
         <h3>Bring Your Own Data</h3>
         <p>Point the workflows at your own DICOM, NRRD, or VTK data instead of the sample datasets.</p>
       </a>
       <a class="mphysio-topic-card" href="cookbook/index.html">
         <h3>MONAI Physio Cookbook</h3>
         <p>Short recipes - ingredients and steps - for training on your own data and adding new segmentation or registration methods.</p>
       </a>
       <a class="mphysio-topic-card" href="api/index.html">
         <h3>API Reference</h3>
         <p>Browse classes and modules for workflows, segmentation, registration, USD, and utilities.</p>
       </a>
     <a class="mphysio-topic-card" href="developer/architecture.html">
       <h3>Developer Docs</h3>
       <p>Understand architecture, extension points, coordinate transforms, and implementation boundaries.</p>
     </a>
      <a class="mphysio-topic-card" href="architecture.html">
        <h3>Architecture</h3>
        <p>Trace the actual workflow classes and data flow from CT inputs to USD outputs.</p>
      </a>
       <a class="mphysio-topic-card" href="contributing.html">
         <h3>Contributing</h3>
         <p>Follow repository conventions for code style, testing, documentation, and pull requests.</p>
       </a>
       <a class="mphysio-topic-card" href="testing.html">
         <h3>Testing</h3>
         <p>Run the fast test suite, data-gated tutorial tests, and regression checks.</p>
       </a>
       <a class="mphysio-topic-card" href="troubleshooting.html">
         <h3>Troubleshooting</h3>
         <p>Diagnose environment, data, segmentation, registration, and USD playback issues.</p>
       </a>
     </div>
   </section>

Tutorial Details
================

See :doc:`tutorials` for the recommended run order, commands, datasets,
per-tutorial implementation details, and the "adapt to your data" notes that
close every section.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart
   tutorials
   viewing_usd
   cli_scripts/byod_tutorials
   architecture

.. toctree::
   :maxdepth: 2
   :caption: CLI & Scripts Guide
   :hidden:

   cli_scripts/overview
   cli_scripts/download_data
   cli_scripts/heart_gated_ct
   cli_scripts/create_statistical_model
   cli_scripts/fit_statistical_model_to_patient
   cli_scripts/4dct_reconstruction
   cli_scripts/vtk_to_usd
   cli_scripts/train_physicsnemo
   cli_scripts/infer_physicsnemo
   cli_scripts/best_practices

.. toctree::
   :maxdepth: 2
   :caption: MONAI Physio Cookbook
   :hidden:

   cookbook/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guides
   :hidden:
   :glob:

   developer/architecture
   developer/extending
   developer/workflows
   developer/core
   developer/segmentation
   developer/registration_images
   developer/registration_models
   developer/transform_conventions
   developer/usd_generation
   developer/utilities
   developer/ai_assistants
   developer/migration_*

.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :hidden:

   contributing
   testing

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   faq
   troubleshooting
   references
   statistics

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
