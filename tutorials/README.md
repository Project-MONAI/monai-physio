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
| 00 | [tutorial_00_infer_respiratory_motion_tcia_lung_3d_ct.ipynb](tutorial_00_infer_respiratory_motion_tcia_lung_3d_ct.ipynb) | `WorkflowFitStatisticalModelToPatient`, `WorkflowInferMovement` | Chest-CT plus pretrained PhysicsNeMo-MGN-Lung-Motion (both auto-download) |
| 00 | [tutorial_00_infer_cardiac_motion_duke_heart_4d_labelmaps.ipynb](tutorial_00_infer_cardiac_motion_duke_heart_4d_labelmaps.ipynb) | `WorkflowFitStatisticalModelToPatient`, `WorkflowInferMovement` | Duke-Heart-4DLabelmaps plus pretrained PhysicsNeMo-MGN-Duke-Heart-Motion (both auto-download) |
| 00 | [tutorial_00_infer_physics_informed_motion_duke_heart_4d_labelmaps.ipynb](tutorial_00_infer_physics_informed_motion_duke_heart_4d_labelmaps.ipynb) | `WorkflowFitStatisticalModelToPatient`, `WorkflowInferMovement`, `InferPhysicsNeMoMGN`, `ConvertVTKToUSD` (`UsdGeomTetMesh`) | Duke-Heart-4DLabelmaps plus pretrained PhysicsNeMo-PhysicsInformed-Motion-Duke-Heart and its ablation (all auto-download); CUDA GPU |
| 1 | [tutorial_01_convert_to_usd_slicer_heart_4d_ct.py](tutorial_01_convert_to_usd_slicer_heart_4d_ct.py) | `WorkflowConvertImageToUSD` | Slicer-Heart-CT (prepare first) |
| 1 | [tutorial_01_convert_to_usd_tetmesh_slicer_heart_4d_ct.py](tutorial_01_convert_to_usd_tetmesh_slicer_heart_4d_ct.py) | `WorkflowConvertImageToVTK` (tetrahedral meshes), `WorkflowReconstructHighres4DCT`, `ConvertVTKToUSD` (`UsdGeomTetMesh`) | Slicer-Heart-CT (prepare first) |
| 1 | [tutorial_01_convert_to_usd_tcia_lung_4d_ct.py](tutorial_01_convert_to_usd_tcia_lung_4d_ct.py) | `WorkflowConvertImageToUSD` | TCIA-4DLung (manual) |
| 2 | [tutorial_02_finetune_icon_tcia_lung_4d_ct.py](tutorial_02_finetune_icon_tcia_lung_4d_ct.py) | `WorkflowFinetuneICONRegistration` | TCIA-4DLung (manual) |
| 2 | [tutorial_02_finetune_icon_distancemap_tcia_lung_4d_ct.py](tutorial_02_finetune_icon_distancemap_tcia_lung_4d_ct.py) | `WorkflowFinetuneICONRegistration` on distance maps | TCIA-4DLung (manual) |
| 2 | [tutorial_02_finetune_icon_distancemap_duke_heart_4d_labelmaps.py](tutorial_02_finetune_icon_distancemap_duke_heart_4d_labelmaps.py) | `WorkflowFinetuneICONRegistration` on distance maps | Duke-Heart-4DLabelmaps |
| 3 | [tutorial_03_reconstruct_highres_slicer_heart_4d_ct.py](tutorial_03_reconstruct_highres_slicer_heart_4d_ct.py) | `WorkflowReconstructHighres4DCT` | Slicer-Heart-CT (prepare first) |
| 3 | [tutorial_03_reconstruct_highres_tcia_lung_4d_ct.py](tutorial_03_reconstruct_highres_tcia_lung_4d_ct.py) | `WorkflowReconstructHighres4DCT` | TCIA-4DLung (manual) |
| 4 | [tutorial_04_convert_to_vtk_slicer_heart_4d_ct.py](tutorial_04_convert_to_vtk_slicer_heart_4d_ct.py) | `WorkflowConvertImageToVTK` | Slicer-Heart-CT (prepare first) |
| 4 | [tutorial_04_convert_to_vtk_tcia_lung_4d_ct.py](tutorial_04_convert_to_vtk_tcia_lung_4d_ct.py) | `WorkflowConvertImageToVTK` | TCIA-4DLung (manual) |
| 4 | [tutorial_04_convert_to_vtk_duke_heart_4d_labelmaps.py](tutorial_04_convert_to_vtk_duke_heart_4d_labelmaps.py) | `WorkflowConvertImageToVTK` (labelmap input, tetrahedral meshes) | Duke-Heart-4DLabelmaps |
| 5 | [tutorial_05_convert_vtk_to_usd_slicer_heart_4d_ct.py](tutorial_05_convert_vtk_to_usd_slicer_heart_4d_ct.py) | `WorkflowConvertVTKToUSD` | Output of Tutorial 4 (Slicer heart) |
| 5 | [tutorial_05_convert_vtk_to_usd_duke_heart_4d_labelmaps.py](tutorial_05_convert_vtk_to_usd_duke_heart_4d_labelmaps.py) | `ConvertVTKToUSD`, `ProcessUSDAnatomy` | Output of Tutorial 4 (duke heart) |
| 6 | [tutorial_06_create_statistical_model_kcl_heart_4d_meshes.py](tutorial_06_create_statistical_model_kcl_heart_4d_meshes.py) | `WorkflowCreateStatisticalModel` | KCL-Heart-Model |
| 6 | [tutorial_06_create_statistical_model_tcia_lung_4d_ct.py](tutorial_06_create_statistical_model_tcia_lung_4d_ct.py) | `WorkflowCreateMeanSurface`, `WorkflowCreateStatisticalModel` | TCIA-4DLung `*_HM10395_g070.nii.gz`, which it segments itself |
| 6 | [tutorial_06_create_statistical_model_duke_heart_4d_labelmaps.py](tutorial_06_create_statistical_model_duke_heart_4d_labelmaps.py) | `WorkflowCreateMeanSurface`, `WorkflowCreateStatisticalModel` | Reference-frame heart surfaces from Tutorial 4 (duke heart) |
| 7 | [tutorial_07_fit_statistical_model_to_patient_kcl_heart_4d_meshes.py](tutorial_07_fit_statistical_model_to_patient_kcl_heart_4d_meshes.py) | `WorkflowFitStatisticalModelToPatient` | DirLab-4DCT `Case1Pack_T70.mha` (manual) plus Tutorial 6 (KCL heart) output |
| 7 | [tutorial_07_fit_statistical_model_to_patient_tcia_lung_4d_ct.py](tutorial_07_fit_statistical_model_to_patient_tcia_lung_4d_ct.py) | `WorkflowFitStatisticalModelToPatient` | Chest-CT plus Tutorial 6 (lung) output |
| 7 | [tutorial_07_fit_statistical_model_to_patient_duke_heart_4d_labelmaps.py](tutorial_07_fit_statistical_model_to_patient_duke_heart_4d_labelmaps.py) | `WorkflowFitStatisticalModelToPatient` | Duke-Heart-4DLabelmaps plus Tutorial 6 (duke heart) output |
| 8 | [tutorial_08_fit_model_to_4d_patients_tcia_lung_4d_ct.py](tutorial_08_fit_model_to_4d_patients_tcia_lung_4d_ct.py) | `WorkflowFitStatisticalModelToPatient`, `WorkflowReconstructHighres4DCT` | TCIA-4DLung plus Tutorial 6 (lung) and Tutorial 2 output |
| 8 | [tutorial_08_fit_model_to_4d_patients_duke_heart_4d_labelmaps.py](tutorial_08_fit_model_to_4d_patients_duke_heart_4d_labelmaps.py) | `WorkflowFitStatisticalModelToPatient`, `RegisterModelsDistanceMaps` | Duke-Heart-4DLabelmaps plus Tutorial 6 (duke heart) and Tutorial 2 output |
| 9 | [tutorial_09_train_physicsnemo_mgn_tcia_lung_4d_ct.py](tutorial_09_train_physicsnemo_mgn_tcia_lung_4d_ct.py) | `WorkflowTrainPhysicsNeMo`, `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement` | Tutorial 8 (lung) output |
| 9 | [tutorial_09_train_physicsnemo_mgn_duke_heart_4d_labelmaps.py](tutorial_09_train_physicsnemo_mgn_duke_heart_4d_labelmaps.py) | `WorkflowTrainPhysicsNeMo`, `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement` | Tutorial 8 (duke heart) output |
| 10 | [tutorial_10_infer_physicsnemo_mgn_tcia_lung_4d_ct.py](tutorial_10_infer_physicsnemo_mgn_tcia_lung_4d_ct.py) | `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement`, `WorkflowConvertVTKToUSD` | Tutorial 8 and 9 (lung) output |
| 10 | [tutorial_10_infer_physicsnemo_mgn_duke_heart_4d_labelmaps.py](tutorial_10_infer_physicsnemo_mgn_duke_heart_4d_labelmaps.py) | `WorkflowInferPhysicsNeMo`, `WorkflowInferMovement`, `WorkflowConvertVTKToUSD` | Tutorial 8 and 9 (duke heart) output |
| 11 | [tutorial_11_evaluate_physicsnemo_tcia_lung_4d_ct.py](tutorial_11_evaluate_physicsnemo_tcia_lung_4d_ct.py) | `WorkflowEvaluateMovement`, `SegmentNVSegmentCTMRI` | TCIA-4DLung plus Tutorial 8 and 9 (lung) output |
| 11 | [tutorial_11_evaluate_physicsnemo_duke_heart_4d_labelmaps.py](tutorial_11_evaluate_physicsnemo_duke_heart_4d_labelmaps.py) | `WorkflowEvaluateMovement` | Duke-Heart-4DLabelmaps plus Tutorial 8 and 9 (duke heart) output |
| 12 | [tutorial_12_infer_respiratory_motion_tcia_lung_4d_ct.py](tutorial_12_infer_respiratory_motion_tcia_lung_4d_ct.py) | `WorkflowConvertImageToVTK`, `WorkflowFitStatisticalModelToPatient`, `WorkflowInferMovement` | TCIA-4DLung plus Tutorial 6 and 9 (lung) output |
| 12 | [tutorial_12_infer_cardiac_motion_duke_heart_4d_labelmaps.py](tutorial_12_infer_cardiac_motion_duke_heart_4d_labelmaps.py) | `ProcessSurfaceMesh`, `WorkflowFitStatisticalModelToPatient`, `WorkflowInferMovement` | Duke-Heart-4DLabelmaps plus Tutorial 6 and 9 (duke heart) output |
| 13 | [tutorial_13_combine_heart_and_lung_motion_chest_3d_ct.py](tutorial_13_combine_heart_and_lung_motion_chest_3d_ct.py) | `WorkflowInferMovement`, `WorkflowFitStatisticalModelToPatient`, `ConvertVTKToUSD` (requires Simpleware Medical) | Chest-CT plus Tutorial 7 (lung) and Tutorial 9 (lung and duke heart) output |
| 14 | [tutorial_14_shape_parameter_sweep_tcia_lung_4d_ct.py](tutorial_14_shape_parameter_sweep_tcia_lung_4d_ct.py) | `WorkflowEvaluateMovement`, `SegmentNVSegmentCTMRI` | TCIA-4DLung plus Tutorial 8 and 9 (lung) output |
| 14 | [tutorial_14_shape_parameter_sweep_duke_heart_4d_labelmaps.py](tutorial_14_shape_parameter_sweep_duke_heart_4d_labelmaps.py) | `WorkflowEvaluateMovement` | Duke-Heart-4DLabelmaps plus Tutorial 8 and 9 (duke heart) output |
| 15 | [tutorial_15_leave_one_out_tcia_lung_4d_ct.py](tutorial_15_leave_one_out_tcia_lung_4d_ct.py) | `WorkflowCreateStatisticalModel`, `WorkflowFitStatisticalModelToPatient`, `WorkflowTrainPhysicsNeMo`, `WorkflowEvaluateMovement` | TCIA-4DLung |
| 15 | [tutorial_15_leave_one_out_duke_heart_4d_labelmaps.py](tutorial_15_leave_one_out_duke_heart_4d_labelmaps.py) | `WorkflowCreateStatisticalModel`, `WorkflowFitStatisticalModelToPatient`, `WorkflowTrainPhysicsNeMo`, `WorkflowEvaluateMovement` | Duke-Heart-4DLabelmaps |
| 16 | [tutorial_16_physics_informed_motion_prep_duke_heart_4d_labelmaps.py](tutorial_16_physics_informed_motion_prep_duke_heart_4d_labelmaps.py) | `ProcessVolumetricMesh.extract_tetrahedra`, `WorkflowCreateStatisticalModel`, `WorkflowFitStatisticalModelToPatient` | Duke-Heart-4DLabelmaps plus Tutorial 4 (duke heart) output |
| 17 | [tutorial_17_physics_informed_motion_train_duke_heart_4d_labelmaps.py](tutorial_17_physics_informed_motion_train_duke_heart_4d_labelmaps.py) | `TrainPhysicsNeMoPhysicsInformedMotion`, `WorkflowTrainPhysicsNeMo` | Tutorial 16 output |
| 18 | [tutorial_18_physics_informed_motion_infer_duke_heart_4d_labelmaps.py](tutorial_18_physics_informed_motion_infer_duke_heart_4d_labelmaps.py) | `WorkflowEvaluateMovement`, `ConvertVTKToUSD` | Tutorial 16 and 17 output |

The [tutorials page](https://project-monai.github.io/monai-physio/tutorials.html)
covers the same set with previews of what each one produces and per-tutorial
notes on running them against your own data.

## Running a Tutorial

Each numbered tutorial is a standalone, straightforward Python script,
executed end-to-end - except Tutorial 00, which is a Jupyter notebook. Paths
are defined near the top of each script. By default, data is read from the
repository `data/` directory and outputs are written under
`tutorials/output/<tutorial_name>/`.

```bash
# Run the whole tutorial from the command line
python tutorials/tutorial_01_convert_to_usd_slicer_heart_4d_ct.py

# Tutorial 00 is a notebook - run it from Jupyter instead
jupyter notebook tutorials/tutorial_00_infer_respiratory_motion_tcia_lung_3d_ct.ipynb
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

The tutorial scripts are wired into the test suite under the `tutorial`
marker - currently every numbered `.py` script, though not the Tutorial 00
notebooks - as one hand-written class each in `tests/test_tutorials.py` rather
than a parametrized sweep, so adding a tutorial does not automatically add a
test. Those that are covered run end-to-end and compare generated screenshots
against baselines:

```bash
# Run all tutorial tests (requires data download first)
pytest tests/test_tutorials.py --run-tutorials -v

# Create baselines on first run
pytest tests/test_tutorials.py --run-tutorials --create-baselines -v

# Run a single tutorial test
pytest tests/test_tutorials.py::TestTutorial01ConvertToUSDSlicerHeart4DCT --run-tutorials -v
```

## Recommended Order

**Tutorial 00** is an optional, self-contained demo notebook - it downloads
its own data and a pretrained network, then predicts lung motion end-to-end
with nothing from the numbered chain below required first.

Each numbered step has one or more variants, each named after the data source
it runs on - `slicer_heart_4d_ct`, `kcl_heart_4d_meshes`,
`duke_heart_4d_labelmaps` or `tcia_lung_4d_ct` - whose shared settings live in
the matching `parameters_<data source>.py` module beside the scripts. Follow the
variants for the anatomy you care about: every tutorial consumes the output of
its own anatomy's earlier tutorials, never the other's.

1. **Tutorial 1** converts one gated 4D CT into an animated USD - the Slicer heart variants (surface and tetmesh) use Slicer-Heart-CT, the lung variant TCIA-4DLung. Prepare the dataset for your anatomy per `data/README.md`, then start here.
2. **Tutorial 2** (lung variants) requires TCIA-4DLung (download it per `data/README.md`) and finetunes the ICON weights Tutorial 8 uses when they are present - it falls back to the stock uniGradICON weights otherwise.
3. **Tutorial 3** registers with Greedy and needs no finetuned weights; the Slicer heart variant uses Slicer-Heart-CT, the lung variant TCIA-4DLung.
4. **Tutorial 4** segments a CT into VTK surfaces; the Slicer heart variant uses Slicer-Heart-CT, the lung variant TCIA-4DLung.
5. **Tutorial 5** (Slicer heart) uses the VTK surfaces produced by Tutorial 4 (Slicer heart) - run Tutorial 4 first.
6. **Tutorial 6** creates the PCA statistical model; the KCL heart variant from KCL-Heart-Model, the lung variant from the TCIA-4DLung `*_HM10395_g070.nii.gz` phases, which it segments itself. Both write `pca_model.json` and `pca_mean_surface.vtp` under their own output directory.
7. **Tutorial 7** applies the statistical model, consuming its own anatomy's Tutorial 6 output; the KCL heart variant fits the Tutorial 6 (KCL heart) model to the DirLab-4DCT `Case1Pack_T70.mha` scan, the lung variant fits the Tutorial 6 (lung) model to the ungated `Chest-CT` scan (`monai-physio-download-data Chest-CT`; see `data/Chest-CT/README.md` for the data source and required citation).

The AI-surrogate pipeline (Tutorials 8 -> 9 -> 10 -> 11 -> 12, plus 14 and 15)
runs on TCIA-4DLung and the Tutorial 6 lung model, in order. Tutorials 14 and 15
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
10 -> 11 -> 12, each reading the previous one's output, with Tutorial 2 (duke heart
distancemap variant) supplying optional finetuned weights to Tutorials 7 and 8.
Tutorials 14 and 15 (duke heart) branch off the same chain on the same dataset,
14 from Tutorials 8 and 9, 15 from the cohort alone. Tutorials 16 to 18 branch
from Tutorial 4 alone, building their own volumetric shape model rather than
reusing the surface one.
Pretrained checkpoints stand in for Tutorial 9 (duke heart) and Tutorial 17:
Tutorials 12 to 14 and 18 accept them, and the two cardiac Tutorial 00
notebooks run on them with no other tutorial first. Download the dataset with
`monai-physio-download-data Duke-Heart-4DLabelmaps`; see
[../data/Duke-Heart-4DLabelmaps/README.md](../data/Duke-Heart-4DLabelmaps/README.md).

## For Contributors

Class-level API reference: [../docs/api/index.rst](../docs/api/index.rst)

To explore the code with an AI assistant, query the graphify knowledge graph
(`graphify query "<question>"`) instead of grepping - see
[../docs/developer/ai_assistants.rst](../docs/developer/ai_assistants.rst)
