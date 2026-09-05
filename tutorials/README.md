# MONAI Physio Tutorials

End-to-end Python scripts covering each major workflow in the library.
These are the recommended starting point for new users.

## Before You Begin

These scripts live only in the source repository - `pip install monai-physio`
does not install them. Clone the repository first:

```bash
git clone https://github.com/Project-MONAI/monai-physio.git
cd monai_physio
```

Each tutorial requires one or more public datasets.
**See [../data/README.md](../data/README.md)** for download instructions,
dataset licensing, and expected directory layout. Run every download from the
top level of the clone: the tutorials resolve their inputs against the
repository root (`<repo>/data/<dataset>`), while
`monai-physio-download-data` writes to `data/<dataset>` relative to the
current working directory.

## Tutorial Index

| # | Script | Primary API | Dataset |
|---|--------|-------------|---------|
| 1 | [tutorial_01_heart_gated_ct_to_usd.py](tutorial_01_heart_gated_ct_to_usd.py) | `WorkflowConvertImageToUSD` | Slicer-Heart-CT (prepare first) |
| 1 | [tutorial_01_lung_gated_ct_to_usd.py](tutorial_01_lung_gated_ct_to_usd.py) | `WorkflowConvertImageToUSD` | Lung gated 4D CT (prepare first) |
| 2 | [tutorial_02_lung_finetune_icon.py](tutorial_02_lung_finetune_icon.py) | `WorkflowFinetuneICONRegistration` | DirLab-4DCT (manual) |
| 2 | [lung distancemap variant](tutorial_02_lung_distancemap_finetune_icon.py) | `WorkflowFinetuneICONRegistration` on distance maps | DirLab-4DCT (manual) |
| 2 | [heart distancemap variant](tutorial_02_duke_heart_distancemap_finetune_icon.py) | `WorkflowFinetuneICONRegistration` on distance maps | Duke-Heart-4DLabelmaps (releasing soon) |
| 3 | [tutorial_03_heart_reconstruct_highres_4d_ct.py](tutorial_03_heart_reconstruct_highres_4d_ct.py) | `WorkflowReconstructHighres4DCT` | Slicer-Heart-CT (prepare first) |
| 3 | [tutorial_03_lung_reconstruct_highres_4d_ct.py](tutorial_03_lung_reconstruct_highres_4d_ct.py) | `WorkflowReconstructHighres4DCT` | DirLab-4DCT (manual) |
| 4 | [tutorial_04_heart_ct_to_vtk.py](tutorial_04_heart_ct_to_vtk.py) | `WorkflowConvertImageToVTK` | Slicer-Heart-CT (prepare first) |
| 4 | [tutorial_04_lung_ct_to_vtk.py](tutorial_04_lung_ct_to_vtk.py) | `WorkflowConvertImageToVTK` | Lung gated 4D CT (prepare first) |
| 4 | [duke heart labelmap variant](tutorial_04_duke_heart_labelmap_to_vtk.py) | `ContourTools.extract_label_surfaces`, `ContourTools.extract_tetrahedra` | Duke-Heart-4DLabelmaps (releasing soon) |
| 5 | [tutorial_05_heart_vtk_to_usd.py](tutorial_05_heart_vtk_to_usd.py) | `WorkflowConvertVTKToUSD` | Output of tutorial 4 |
| 5 | [duke heart variant](tutorial_05_duke_heart_vtk_to_usd.py) | `ConvertVTKToUSD`, `USDAnatomyTools` | Output of tutorial 4 (duke heart labelmap) |
| 6 | [tutorial_06_heart_create_statistical_model.py](tutorial_06_heart_create_statistical_model.py) | `WorkflowCreateStatisticalModel` | KCL-Heart-Model |
| 6 | [tutorial_06_lung_create_statistical_model.py](tutorial_06_lung_create_statistical_model.py) | `WorkflowCreateMeanSurface`, `WorkflowCreateStatisticalModel` | DirLab-4DCT `Case*T70.mha`, which it segments itself |
| 6 | [duke heart variant](tutorial_06_duke_heart_create_statistical_model.py) | `WorkflowCreateMeanSurface`, `WorkflowCreateStatisticalModel` | Reference-frame heart surfaces from Tutorial 4 (duke heart labelmap) |
| 7 | [tutorial_07_heart_fit_statistical_model_to_patient.py](tutorial_07_heart_fit_statistical_model_to_patient.py) | `WorkflowFitStatisticalModelToPatient` | DirLab-4DCT `Case1Pack_T70.mha` (manual) plus Tutorial 6 (heart) output |
| 7 | [tutorial_07_lung_fit_statistical_model_to_patient.py](tutorial_07_lung_fit_statistical_model_to_patient.py) | `WorkflowFitStatisticalModelToPatient` | Chest-CT plus Tutorial 6 (lung) output |
| 7 | [duke heart variant](tutorial_07_duke_heart_fit_statistical_model_to_patient.py) | `WorkflowFitStatisticalModelToPatient` | Duke-Heart-4DLabelmaps plus Tutorial 6 (duke heart) output |
| 8 | [tutorial_08_lung_fit_model_to_4d_patients.py](tutorial_08_lung_fit_model_to_4d_patients.py) | `WorkflowFitStatisticalModelToPatient`, `WorkflowReconstructHighres4DCT` | DirLab-4DCT plus Tutorial 6 (lung) and Tutorial 2 output |
| 8 | [duke heart variant](tutorial_08_duke_heart_fit_model_to_4d_patients.py) | `WorkflowFitStatisticalModelToPatient`, `RegisterModelsDistanceMaps` | Duke-Heart-4DLabelmaps plus Tutorial 6 (duke heart) and Tutorial 2 output |
| 9 | [tutorial_09_lung_train_physicsnemo_mgn.py](tutorial_09_lung_train_physicsnemo_mgn.py) | `WorkflowTrainPhysicsNeMo`, `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement` | Tutorial 8 (lung) output |
| 9 | [duke heart variant](tutorial_09_duke_heart_train_physicsnemo_mgn.py) | `WorkflowTrainPhysicsNeMo`, `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement` | Tutorial 8 (duke heart) output |
| 10 | [tutorial_10_lung_infer_physicsnemo_mgn.py](tutorial_10_lung_infer_physicsnemo_mgn.py) | `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement`, `WorkflowConvertVTKToUSD` | Tutorial 8 and 9 (lung) output |
| 10 | [duke heart variant](tutorial_10_duke_heart_infer_physicsnemo_mgn.py) | `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement`, `WorkflowConvertVTKToUSD` | Tutorial 8 and 9 (duke heart) output |
| 11 | [tutorial_11_lung_evaluate_physicsnemo.py](tutorial_11_lung_evaluate_physicsnemo.py) | `WorkflowEvaluateMovement`, `SegmentNVSegmentCTMRI` | DirLab-4DCT plus Tutorial 8 and 9 (lung) output |
| 11 | [duke heart variant](tutorial_11_duke_heart_evaluate_physicsnemo.py) | `WorkflowEvaluateMovement` | Duke-Heart-4DLabelmaps plus Tutorial 8 and 9 (duke heart) output |
| 12 | [tutorial_12_lung_end_to_end_inference.py](tutorial_12_lung_end_to_end_inference.py) | `WorkflowConvertImageToVTK`, `WorkflowFitStatisticalModelToPatient`, `WorkflowInferMovement` | DirLab-4DCT plus Tutorial 6 and 9 (lung) output |
| 12 | [duke heart variant](tutorial_12_duke_heart_end_to_end_inference.py) | `ContourTools`, `WorkflowFitStatisticalModelToPatient`, `WorkflowInferMovement` | Duke-Heart-4DLabelmaps plus Tutorial 6 and 9 (duke heart) output |
| 13 | [tutorial_13_heart_and_lung_motion.py](tutorial_13_heart_and_lung_motion.py) | `WorkflowInferMovement`, `WorkflowFitStatisticalModelToPatient`, `ConvertVTKToUSD` (requires Simpleware Medical) | Chest-CT plus Tutorial 7 (lung) and Tutorial 9 (lung and duke heart) output |
| 14 | [tutorial_14_lung_shape_parameter_sweep.py](tutorial_14_lung_shape_parameter_sweep.py) | `WorkflowEvaluateMovement`, `SegmentNVSegmentCTMRI` | DirLab-4DCT plus Tutorial 8 and 9 (lung) output |
| 14 | [duke heart variant](tutorial_14_duke_heart_shape_parameter_sweep.py) | `WorkflowEvaluateMovement` | Duke-Heart-4DLabelmaps plus Tutorial 8 and 9 (duke heart) output |
| 15 | [tutorial_15_lung_leave_one_out.py](tutorial_15_lung_leave_one_out.py) | `WorkflowCreateStatisticalModel`, `WorkflowFitStatisticalModelToPatient`, `WorkflowTrainPhysicsNeMo`, `WorkflowEvaluateMovement` | DirLab-4DCT |
| 15 | [duke heart variant](tutorial_15_duke_heart_leave_one_out.py) | `WorkflowCreateStatisticalModel`, `WorkflowFitStatisticalModelToPatient`, `WorkflowTrainPhysicsNeMo`, `WorkflowEvaluateMovement` | Duke-Heart-4DLabelmaps |
| 16 | [tutorial_16_duke_heart_physics_informed_motion_prep.py](tutorial_16_duke_heart_physics_informed_motion_prep.py) | `ContourTools.extract_tetrahedra`, `WorkflowCreateStatisticalModel`, `WorkflowFitStatisticalModelToPatient` | Duke-Heart-4DLabelmaps plus Tutorial 4 (duke heart) output |
| 17 | [tutorial_17_duke_heart_physics_informed_motion_train.py](tutorial_17_duke_heart_physics_informed_motion_train.py) | `TrainPhysicsNeMoPhysicsInformedMotion`, `WorkflowTrainPhysicsNeMo` | Tutorial 16 output |
| 18 | [tutorial_18_duke_heart_physics_informed_motion_infer.py](tutorial_18_duke_heart_physics_informed_motion_infer.py) | `WorkflowEvaluateMovement`, `ConvertVTKToUSD` | Tutorial 16 and 17 output |

The [tutorials page](https://project-monai.github.io/monai-physio/tutorials.html)
covers the same set with previews of what each one produces and per-tutorial
notes on running them against your own data.

## Running a Tutorial

Each tutorial is a standalone, straightforward Python script, executed
end-to-end. Paths are defined near the top of each script. By default, data
is read from the repository `data/` directory and outputs are written under
`tutorials/output/<tutorial_name>/`.

```bash
# Run the whole tutorial from the command line
python tutorials/tutorial_01_heart_gated_ct_to_usd.py
```

In VS Code or Cursor, open the tutorial and use **Run Python File** (or run
the cells in order with **Run Cell**). The script's `if __name__ ==
"__main__":` block executes the workflow and assigns the resulting
`tutorial_results` dict in the script's namespace; the same variable is what
`tests/test_tutorials.py` consumes via `runpy.run_path(..., run_name=
"__main__")`.

To use different paths, edit the constants near the top of the tutorial
script. For repeatable command-line execution with path arguments, use the
installed `monai-physio-*` CLI commands instead.

## Running as Pytest Tutorial Tests

Some tutorials are wired into the test suite under the `tutorial` marker -
currently 9 of the 29 scripts, as one hand-written class each in
`tests/test_tutorials.py` rather than a parametrized sweep, so adding a tutorial
does not automatically add a test. Those that are covered run end-to-end and
compare generated screenshots against baselines:

```bash
# Run all tutorial tests (requires data download first)
pytest tests/test_tutorials.py --run-tutorials -v

# Create baselines on first run
pytest tests/test_tutorials.py --run-tutorials --create-baselines -v

# Run a single tutorial test
pytest tests/test_tutorials.py::TestTutorial01HeartGatedCTToUSD --run-tutorials -v
```

## Recommended Order

Each numbered step has a heart variant, a lung variant, or both. Follow the
variants for the anatomy you care about: every tutorial consumes the output of
its own anatomy's earlier tutorials, never the other's.

1. **Tutorial 1** converts one gated 4D CT into an animated USD - the heart variant uses Slicer-Heart-CT, the lung variant DirLab-4DCT. Prepare the dataset for your anatomy per `data/README.md`, then start here.
2. **Tutorial 2** requires DirLab-4DCT (download it per `data/README.md`) and finetunes the ICON weights Tutorial 8 uses when they are present - it falls back to the stock uniGradICON weights otherwise.
3. **Tutorial 3** registers with Greedy and needs no finetuned weights; the heart variant uses Slicer-Heart-CT, the lung variant DirLab-4DCT.
4. **Tutorial 4** segments a CT into VTK surfaces; the heart variant uses Slicer-Heart-CT, the lung variant DirLab-4DCT.
5. **Tutorial 5** (heart only) uses the VTK surfaces produced by Tutorial 4 (heart) - run Tutorial 4 first.
6. **Tutorial 6** creates the PCA statistical model; the heart variant from KCL-Heart-Model, the lung variant from the DirLab-4DCT `Case*T70.mha` phases, which it segments itself. Both write `pca_model.json` and `pca_mean_surface.vtp` under their own output directory.
7. **Tutorial 7** applies the statistical model, consuming its own anatomy's Tutorial 6 output; the heart variant fits the Tutorial 6 (heart) model, the lung variant fits the Tutorial 6 (lung) model to the ungated `Chest-CT` scan (`monai-physio-download-data Chest-CT`; see `data/Chest-CT/README.md` for the data source and required citation).

The AI-surrogate pipeline (Tutorials 8 -> 9 -> 10 -> 11 -> 12, plus 14 and 15)
runs on DIR-Lab and the Tutorial 6 lung model, in order. Tutorials 14 and 15
branch off the chain rather than continuing it: 14 needs only the Tutorial 8 fit
and the Tutorial 9 checkpoint, and 15 needs neither, rebuilding both per fold:

8. **Tutorial 8** fits the lung PCA model to each case's reference phase and propagates the fitted SSM surface through every respiratory phase (output feeds Tutorial 9). It uses the Tutorial 2 ICON weights when they exist.
9. **Tutorial 9** trains a PhysicsNeMo MeshGraphNet to predict the per-vertex motion at any stage. A `TrainPhysicsNeMoMLP` method exists as a drop-in alternative, without its own tutorial.
10. **Tutorial 10** loads that checkpoint and predicts the held-out case's surface at every acquired stage, warping the reference-phase CT through the inferred deformation and exporting one animated USD. It renders the acquired frame surface beside the prediction for visual comparison but does not score it; scoring is Tutorial 11's job. The case and checkpoint epoch are constants near the top of the script; for command-line runs with path arguments, use the installed `monai-physio-infer-physicsnemo` CLI.
11. **Tutorial 11** scores the same prediction against the images rather than against the registration: it segments every gated frame independently, then reports volume difference and surface RMSE per lung lobe (per heart chamber, with Dice, in the duke variant) as `evaluation_report.md` and `evaluation_metrics.csv`. The lung variant leaves Dice out: a lobe moves little compared to its own size, so the overlap fraction describes the lobe rather than the motion.
12. **Tutorial 12** collapses the whole chain into one script: it segments the reference frame, fits the Tutorial 6 model to that patient itself, and infers every stage - so nothing is read from Tutorial 8 and no phase is ever registered. It needs only the gated series plus the Tutorial 6 model and the Tutorial 9 checkpoint, and it wipes its output directory on every run so the reported runtimes in `<case>_runtimes.csv` cover the entire pipeline.

**Tutorial 13** is where the two chains meet. It animates the ungated
`Chest-CT` scan of Tutorial 7 (lung) with both rhythms at once: respiratory
motion from the Tutorial 9 (lung) network applied to that scan's lung fit, and
cardiac motion from the Tutorial 9 (duke heart) network applied to a Duke heart
model it fits to the same scan. Nothing in it registers anything or needs a 4D
acquisition. Each model is fitted through the segmenter that built it, so the
heart step calls Simpleware Medical.

**Tutorial 14** asks how much the shape parameters matter. It re-runs the
Tutorial 11 scoring for the same hold-out case over a grid of PCA coefficient
offsets - the first few modes, swept from -1 to +1 standard deviations in steps
of 0.5 - feeding each perturbed vector to the Tutorial 9 network while holding
the patient's fitted surface fixed, so the only thing that changes is the motion
the network infers. Dice, volume difference and surface RMSE for every
combination land in `shape_sweep_metrics.csv`, averaged per combination in
`shape_sweep_summary.csv`. The default grid is 25 combinations, each costing one
Tutorial 11 run; `number_of_modes_to_vary`, `perturbation_range` and
`perturbation_step` near the top of the script set its size.

**Tutorial 15** stops trusting a single hold-out. It re-runs the whole chain -
shape model, cohort fit, MeshGraphNet training, inference and scoring - once per
fold, holding out a different case each time, and reports Dice, volume
difference and surface RMSE as a mean and a spread across folds rather than as
one number. `number_of_leave_one_out_runs` near the top of the script sets the
fold count and defaults to 5. Nothing from Tutorials 6, 8 or 9 is required: it
builds its own model per fold, because a model built once from everyone has
already seen every case. Segmentations and, for the lung, the phase
registrations do not depend on which case is held out, so they are cached under
`shared/` and reused. Written for a multi-GPU Linux host:
under `torchrun --standalone --nproc_per_node=<gpus>` the training is
data-parallel across ranks and the per-case loops are split across them.

**Tutorials 16, 17 and 18** ask a different question of the same cardiac data:
is the predicted motion something tissue could actually do? Tutorials 9 and 10
score displacement alone, so nothing in their loss rules out an element
inflating, thinning past what myocardium allows, or inverting outright. These
three add a neo-Hookean strain energy to the loss, which prices exactly those
deformations, and read out the stress it implies.

That energy needs a deformation gradient, which needs volume elements, which the
surface shape model of Tutorials 6 to 8 does not have. So **Tutorial 16** builds
the model again volumetrically: it fills the mean surface with tetrahedra and
decomposes the population against that template, then fits it to every case and
phase. **Tutorial 17** trains the surrogate with the residual added, and by
default trains a second model with the physics weight at zero on identical data
- the only comparison that isolates the physics term rather than confounding it
with the change of shape model. **Tutorial 18** predicts the held-out case with
both, scores them side by side, derives the Cauchy stress from the same
constitutive law the loss used, and exports the animation to USD colored by von
Mises stress. Nothing in Tutorials 1 to 15 is modified; only Tutorial 4's
surfaces and Tutorial 2's weights are read.

The success criterion is worth stating plainly: the physics-informed model is
not expected to *beat* the ablation on RMSE. A strain energy is a prior, and a
prior that improved the data fit would be suspicious. What it should do is match
it while keeping every element's Jacobian positive.

The `duke_heart` variants form their own chain on Duke-Heart-4DLabelmaps,
which no step above shares: Tutorial 4 (duke heart) -> 5 -> 6 -> 7 -> 8 -> 9 ->
10 -> 11 -> 12, each reading the previous one's output, with Tutorial 2 (heart
distancemap variant) supplying optional finetuned weights to Tutorials 7 and 8.
Tutorials 14 and 15 (duke heart) branch off the same chain on the same dataset,
14 from Tutorials 8 and 9, 15 from the cohort alone. Tutorials 16 to 18 branch
from Tutorial 4 alone, building their own volumetric shape model rather than
reusing the surface one.
That dataset is being released soon; until then this chain cannot be run, and
access can be requested from Stephen Aylward (<saylward@nvidia.com>). See
[../data/Duke-Heart-4DLabelmaps/README.md](../data/Duke-Heart-4DLabelmaps/README.md).

## For Contributors

Class-level API reference: [../docs/api/index.rst](../docs/api/index.rst)

To explore the code with an AI assistant, query the graphify knowledge graph
(`graphify query "<question>"`) instead of grepping - see
[../docs/developer/ai_assistants.rst](../docs/developer/ai_assistants.rst)
