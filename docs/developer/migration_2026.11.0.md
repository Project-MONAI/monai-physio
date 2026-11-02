# Migration Guide - 2026.11.0

Breaking changes in this release, and how to update code that depends on
them.

MONAI Physio prefers compatibility: public APIs are broken only when the change
is generally beneficial to future users. When a break is unavoidable, the
project does **not** ship deprecation shims or removed-symbol stubs. Instead,
substantial changes ship with code that automates the conversion, and every
break is recorded here in the commit that introduces it.

At release time this file is renamed `migration_<version>.md` and a fresh
`migration_next.md` is started for the next cycle.

## Entry template

Append one section per breaking change, newest last, using this shape:

````markdown
## <symbol, module, or CLI flag> - <one-line summary>

**Change:** what moved, was renamed, or changed signature.

**Why:** the benefit to future users that justified the break.

**Before**

```python
old_call(argument)
```

**After**

```python
new_call(argument, required_option="value")
```

**Automated conversion:** `<path to script or CLI>`, or `None needed`.
````

## InferPhysicsNeMoBase, InferPhysicsNeMoMGN, InferPhysicsNeMoMLP - `model_tag` is now a constructor argument

**Change:** `model_tag` moved from a class attribute to the first `__init__`
argument of `InferPhysicsNeMoBase` and its MGN and MLP subclasses
(defaults `"base"`, `"mgn"`, `"mlp"`). `log_level` is now the second
positional argument. A subclass that only sets a class-level `model_tag`
now gets the constructor default instead, because `__init__` assigns the
instance attribute.

**Why:** the checkpoint tag must match the training method that wrote the
weights (for example `"physics_informed_motion"`). It can now be chosen per
instance without defining a subclass.

**Before**

```python
class InferPhysicsInformed(InferPhysicsNeMoMGN):
    model_tag = "physics_informed_motion"


method = InferPhysicsInformed("DEBUG")
```

**After**

```python
method = InferPhysicsNeMoMGN(model_tag="physics_informed_motion", log_level="DEBUG")
```

**Automated conversion:** `None needed`. Pass `log_level` by keyword and
replace class-level `model_tag` overrides with the constructor argument.

## `ProcessContours` - split into `ProcessSurfaceMesh` and `ProcessVolumetricMesh`

**Change:** `monai_physio.process_contours.ProcessContours` no longer exists.
- Its tetrahedral methods moved to
  `monai_physio.process_volumetric_mesh.ProcessVolumetricMesh`:
  - `extract_tetrahedra`
  - `trim_tetrahedra_to_surface`
  - `repair_inverted_tetrahedra`
- `apply_anatomy_color` moved to `ProcessUSDAnatomy`, which holds the color
  tables, next to the new `select_anatomy_type`.
- The savers `save_surfaces` and `save_combined_surfaces` moved to
  `WorkflowConvertImageToVTK`, whose results they write. The in-memory merge
  behind `save_combined_surfaces` is the new
  `ProcessSurfaceMesh.combine_surfaces`.
- Every other method moved to
  `monai_physio.process_surface_mesh.ProcessSurfaceMesh`; the surface
  extractors changed too (see the next entry).
- The GPU primitives behind both are the new public class
  `monai_physio.process_mesh_kernels.ProcessMeshKernels`. It provides closest
  points, inside/outside tests, tetrahedron quality, distance maps, the blur
  and the label-vote resample.
- All three classes are exported from `monai_physio`.
- Apart from the extractors, method names, signatures and results are
  unchanged.

**Why:** Tutorial 4's GPU work grew `process_contours.py` past 2100 lines,
mixing surface code, tetrahedral code and roughly 500 lines of warp/torch
kernels. Splitting them gives each class one kind of mesh. It also gives the
GPU primitives a documented home that tutorials and tests can call, instead
of importing private module functions. Process classes now work in memory
only, and anatomy colors come from one place, so a surface and a volume mesh
of the same structure always get the same color.

**Before**

```python
from monai_physio import ProcessContours

tools = ProcessContours()
surface = tools.extract_label_surfaces(mask)[1]
tools.apply_anatomy_color(surface, ["myocardium", "heart"])
mesh = tools.extract_tetrahedra(mask, element_size_mm=1.5)
mesh = tools.trim_tetrahedra_to_surface(mesh, surface)
ProcessContours.save_combined_surfaces(surfaces, "out.vtp")
```

**After**

```python
from monai_physio import (
    ProcessSurfaceMesh,
    ProcessUSDAnatomy,
    ProcessVolumetricMesh,
    WorkflowConvertImageToVTK,
)

surface_tools = ProcessSurfaceMesh()
volume_tools = ProcessVolumetricMesh()
surface = surface_tools.extract_label_surfaces(mask)[1]
ProcessUSDAnatomy(stage=None).apply_anatomy_color(surface, ["myocardium", "heart"])
mesh = volume_tools.extract_tetrahedra(mask, element_size_mm=1.5)
mesh = volume_tools.trim_tetrahedra_to_surface(mesh, surface)
WorkflowConvertImageToVTK.save_combined_surfaces(surfaces, "out.vtp")
```

**Automated conversion:**
`py utils/migrate_process_contours.py [--dry-run] PATH [PATH ...]`

It rewrites imports and the class name, and moves calls on a surface-tools
object to `ProcessVolumetricMesh().<method>(...)`,
`ProcessUSDAnatomy(stage=None).apply_anatomy_color(...)` and
`WorkflowConvertImageToVTK.save_*(...)`, adding the imports. It prints every
change and flags any file where an import must be added by hand. Hoist the
per-call `ProcessVolumetricMesh()` and `ProcessUSDAnatomy(stage=None)` into one
instance where they run in a loop.

## `ProcessSurfaceMesh` extractors - one conforming, watertight extractor

**Change:**
- `extract_contours` and `extract_watertight_surface` are removed.
  `extract_label_surfaces` is the one surface extractor: it returns every
  label's closed, outward surface, and neighboring labels share their walls
  vertex for vertex, including where three labels meet.
- `extract_label_surfaces` gains two trailing parameters:
  - `label_ids` extracts a subset and treats every other label as background;
  - `surface_reduction_rate` remeshes each surface with ACVD, after which
    neighboring surfaces no longer share their vertices. Lower the triangle
    count with a coarser `isotropic_spacing_mm` to keep them matching.
- Each surface it returns carries `field_data["SegmentationLabelIds"]`.
- Its memory no longer grows with the number of labels.
- Nothing produces or reads the per-cell `boundary_labels` array any more:
  `ConvertVTKToUSD`, `WorkflowConvertVTKToUSD` and
  `ProcessSurfaceMesh.remesh_and_smooth_surface` use per-cell
  `SegmentationLabelIds`, which `ProcessSurfaceMesh.combine_surfaces` writes.
- `ProcessMeshKernels.resample_labelmap_isotropic` is removed, as only
  `extract_contours` used it; `vote_resample_labelmap` is the GPU equivalent.

**Why:** the three extractors gave three different surfaces for the same
labelmap: `extract_contours` was not watertight and could pinch, and
`extract_watertight_surface` traced each label on its own, so neighbors
overlapped or left gaps. One extractor gives every workflow, tutorial and USD
export the same geometry.

**Before**

```python
contours = tools.extract_contours(labelmap, surface_reduction_rate=0.5)
surface = tools.extract_watertight_surface(mask, anatomy_names=["lung"])
```

**After**

```python
surfaces = tools.extract_label_surfaces(labelmap, surface_reduction_rate=0.5)
contours = ProcessSurfaceMesh.combine_surfaces(surfaces)

surface = tools.extract_label_surfaces(mask)[1]
ProcessUSDAnatomy(stage=None).apply_anatomy_color(surface, ["lung"])
```

**Automated conversion:** `py utils/migrate_process_contours.py --dry-run PATH`
flags each line that uses a removed extractor or `boundary_labels`; the
replacement depends on whether the caller wants one multi-label mesh
(`combine_surfaces`) or one mask's surface (`[1]`).

## `WorkflowConvertImageToVTK.process` - labelmap input, shared walls, ACVD reduction

**Change:**
- Surfaces come from `ProcessSurfaceMesh.extract_label_surfaces`, so they are
  watertight and neighbors share their walls; the group surfaces are extracted
  together, and so are the per-label ones.
- `surface_reduction_rate` remeshes with ACVD instead of `decimate_pro`.
- A label surface is colored by its own name before its group's, so
  `myocardium` gets its own color rather than the heart's, and a name nothing
  matches falls back to the `other` color instead of grey.
- The `segmentation_masks` key is removed from the result.
- New keyword-only options: `labelmap_image` (skip segmentation),
  `extract_group_surfaces`, `groups` (custom unions of labels),
  `isotropic_spacing_mm`, `smoothing_iterations` and
  `tetrahedra_element_size_mm`, which adds `tetrahedra` and
  `label_tetrahedra` to the result.

**Why:** tutorials built labelmap-to-surface and labelmap-to-mesh pipelines by
hand, each slightly differently. One workflow that also accepts a labelmap
gives every tutorial the same surfaces, colors and annotations.

**Before**

```python
result = workflow.process(ct, surface_reduction_rate=0.5)
heart_mask = result["segmentation_masks"]["heart"]
```

**After**

```python
result = workflow.process(ct, surface_reduction_rate=0.5)
heart_mask = segmenter.create_anatomy_group_labelmaps(result["labelmap"])["heart"]
```

**Automated conversion:** None for the geometry, which changes, so regenerate
saved outputs. `py utils/migrate_process_contours.py --dry-run PATH` flags
uses of `segmentation_masks`.

## `ProcessUSD` stage methods - take an open stage, never a path

**Change:** `list_mesh_primvars`, `apply_colormap_from_primvar`,
`set_solid_display_color`, `list_mesh_paths_under` and
`repair_mesh_primvar_element_sizes` take a `Usd.Stage` only, and none of them
saves it; `repair_mesh_primvar_element_sizes` loses its `save` parameter. Their
first parameter is renamed from `stage_or_path` to `stage`, so a call passing
it by keyword must use the new name. The
new `ProcessUSD.open_stage(path)` and `ProcessUSD.save_stage(stage)` open and
write a stage.

**Why:** given a path, each call opened the file and the editing ones saved
it, so a loop over meshes reopened and rewrote the whole USD file once per
mesh. Opening once and saving once is faster and makes the write explicit.

**Before**

```python
for mesh_path in mesh_paths:
    usd_tools.set_solid_display_color("scene.usd", mesh_path, (1.0, 0.0, 0.0))
```

**After**

```python
stage = usd_tools.open_stage("scene.usd")
for mesh_path in mesh_paths:
    usd_tools.set_solid_display_color(stage, mesh_path, (1.0, 0.0, 0.0))
usd_tools.save_stage(stage)
```

**Automated conversion:** `py utils/migrate_process_contours.py --dry-run PATH`
flags calls that pass a path string or `str(...)`.

## `EvaluateMovementLung` - scores NVSegmentCTMRI lobes

**Change:** `EvaluateMovementLung.segmenter_class` is `SegmentNVSegmentCTMRI`
instead of `SegmentChestTotalSegmentator`, and `label_ids` are its lobe ids
`(28, 29, 30, 31, 32)` instead of `(10, 11, 12, 13, 14)`.

**Why:** the lung tutorials now segment with one segmenter,
`SegmentNVSegmentCTMRI`, so the lobes a shape model is built from and the
lobes its predictions are scored against carry the same ids.

**Before**

```python
EvaluateMovementLung().label_ids  # (10, 11, 12, 13, 14)
```

**After**

```python
EvaluateMovementLung().label_ids  # (28, 29, 30, 31, 32)
```

**Automated conversion:** None needed. Delete cached ground-truth labelmaps
made with TotalSegmentator so they are re-segmented.

## `WorkflowCreateMeanSurface` - transform setters renamed

**Change:** `set_alignment_transform_type` is now `set_icp_transform_type`, and
`set_registration_transform_type` is now `set_distance_map_transform_type`.

**Why:** the names say which stage each transform type configures, and
`set_icp_transform_type` matches the same setter on the statistical-model and
patient-fit workflows.

**Before**

```python
workflow.set_alignment_transform_type("Rigid")
workflow.set_registration_transform_type("Affine")
```

**After**

```python
workflow.set_icp_transform_type("Rigid")
workflow.set_distance_map_transform_type("Affine")
```

**Automated conversion:**
`sed -i 's/set_alignment_transform_type/set_icp_transform_type/g; s/set_registration_transform_type/set_distance_map_transform_type/g' PATH`

## `ProcessSurfaceMesh.extract_label_surfaces` - surfaces facing the background no longer shrink

**Change:** a label's wall against the background now lies midway between
the label's outermost voxels and the background's, as walls between two
labels already did. It used to lie on the label's outermost voxel centers,
half a voxel inside. Surfaces, and the tetrahedral meshes trimmed to them,
are therefore larger than before: a 12 mm sphere now encloses 97% of its
volume on a 1 mm grid, up from 79%, and walls three voxels thick stay in one
piece instead of fragmenting. `ProcessMeshKernels.signed_distance_mm` with
`label_id=0` now measures to the background's own edge voxels rather than
returning the negated distance to the union of every label. A labelmap in
which no label survives the isotropic grid now returns `{}` with a warning
instead of raising `KeyError: 'LabelId'`.

**Why:** the half-voxel inset removed up to half the volume of small or
coarsely sampled structures, which every downstream surface, mesh, and
volume measurement inherited.

**Before**

```python
surfaces = ProcessSurfaceMesh().extract_label_surfaces(labelmap)
surfaces[1].volume  # about 0.8 x the label's voxel volume on a 1 mm grid
```

**After**

```python
surfaces = ProcessSurfaceMesh().extract_label_surfaces(labelmap)
surfaces[1].volume  # about 0.97 x the label's voxel volume on a 1 mm grid
```

**Automated conversion:** `None needed`. Regenerate cached surfaces, meshes,
and any statistics or baselines computed from them.

## Tutorials and their `parameters_*` modules - renamed to `<step>_<task>_<dataset>`

**Change:** every tutorial script and notebook is renamed so that its name says
what it does and on which dataset, and each writes to a
`tutorials/output/<new script name>/` directory. The per-dataset parameter
modules, their classes, and their shared instances are renamed to match. The
TCIA-4DLung tutorials now hold out `103_HM10395` instead of `100_HM10395`
(`ParametersTCIALung4DCT.mgn_hold_out_case`), and `100_HM10395` is no longer in
`ParametersTCIALung4DCT.case_ids`.

**Why:** the old names mixed organ, modality, and task in no fixed order, so
the variants of one tutorial did not sort together and the dataset each used
was not visible from its name.

| Before | After |
|---|---|
| `tutorial_00_lung_demo.ipynb` | `tutorial_00_infer_respiratory_motion_tcia_lung_3d_ct.ipynb` |
| `tutorial_01_heart_gated_ct_to_usd.py` | `tutorial_01_convert_to_usd_slicer_heart_4d_ct.py` |
| `tutorial_01_lung_gated_ct_to_usd.py` | `tutorial_01_convert_to_usd_tcia_lung_4d_ct.py` |
| `tutorial_01_lung_gated_ct_to_usd_tetmesh.py` | `tutorial_01_convert_to_usd_tetmesh_slicer_heart_4d_ct.py` |
| `tutorial_02_duke_heart_distancemap_finetune_icon.py` | `tutorial_02_finetune_icon_distancemap_duke_heart_4d_labelmaps.py` |
| `tutorial_02_lung_distancemap_finetune_icon.py` | `tutorial_02_finetune_icon_distancemap_tcia_lung_4d_ct.py` |
| `tutorial_02_lung_finetune_icon.py` | `tutorial_02_finetune_icon_tcia_lung_4d_ct.py` |
| `tutorial_03_heart_reconstruct_highres_4d_ct.py` | `tutorial_03_reconstruct_highres_slicer_heart_4d_ct.py` |
| `tutorial_03_lung_reconstruct_highres_4d_ct.py` | `tutorial_03_reconstruct_highres_tcia_lung_4d_ct.py` |
| `tutorial_04_duke_heart_labelmap_to_vtk.py` | `tutorial_04_convert_to_vtk_duke_heart_4d_labelmaps.py` |
| `tutorial_04_heart_ct_to_vtk.py` | `tutorial_04_convert_to_vtk_slicer_heart_4d_ct.py` |
| `tutorial_04_lung_ct_to_vtk.py` | `tutorial_04_convert_to_vtk_tcia_lung_4d_ct.py` |
| `tutorial_05_duke_heart_vtk_to_usd.py` | `tutorial_05_convert_vtk_to_usd_duke_heart_4d_labelmaps.py` |
| `tutorial_05_heart_vtk_to_usd.py` | `tutorial_05_convert_vtk_to_usd_slicer_heart_4d_ct.py` |
| `tutorial_06_duke_heart_create_statistical_model.py` | `tutorial_06_create_statistical_model_duke_heart_4d_labelmaps.py` |
| `tutorial_06_heart_create_statistical_model.py` | `tutorial_06_create_statistical_model_kcl_heart_4d_meshes.py` |
| `tutorial_06_lung_create_statistical_model.py` | `tutorial_06_create_statistical_model_tcia_lung_4d_ct.py` |
| `tutorial_07_duke_heart_fit_statistical_model_to_patient.py` | `tutorial_07_fit_statistical_model_to_patient_duke_heart_4d_labelmaps.py` |
| `tutorial_07_heart_fit_statistical_model_to_patient.py` | `tutorial_07_fit_statistical_model_to_patient_kcl_heart_4d_meshes.py` |
| `tutorial_07_lung_fit_statistical_model_to_patient.py` | `tutorial_07_fit_statistical_model_to_patient_tcia_lung_4d_ct.py` |
| `tutorial_08_duke_heart_fit_model_to_4d_patients.py` | `tutorial_08_fit_model_to_4d_patients_duke_heart_4d_labelmaps.py` |
| `tutorial_08_lung_fit_model_to_4d_patients.py` | `tutorial_08_fit_model_to_4d_patients_tcia_lung_4d_ct.py` |
| `tutorial_09_duke_heart_train_physicsnemo_mgn.py` | `tutorial_09_train_physicsnemo_mgn_duke_heart_4d_labelmaps.py` |
| `tutorial_09_lung_train_physicsnemo_mgn.py` | `tutorial_09_train_physicsnemo_mgn_tcia_lung_4d_ct.py` |
| `tutorial_10_duke_heart_infer_physicsnemo_mgn.py` | `tutorial_10_infer_physicsnemo_mgn_duke_heart_4d_labelmaps.py` |
| `tutorial_10_lung_infer_physicsnemo_mgn.py` | `tutorial_10_infer_physicsnemo_mgn_tcia_lung_4d_ct.py` |
| `tutorial_11_duke_heart_evaluate_physicsnemo.py` | `tutorial_11_evaluate_physicsnemo_duke_heart_4d_labelmaps.py` |
| `tutorial_11_lung_evaluate_physicsnemo.py` | `tutorial_11_evaluate_physicsnemo_tcia_lung_4d_ct.py` |
| `tutorial_12_duke_heart_end_to_end_inference.py` | `tutorial_12_infer_cardiac_motion_duke_heart_4d_labelmaps.py` |
| `tutorial_12_lung_end_to_end_inference.py` | `tutorial_12_infer_respiratory_motion_tcia_lung_4d_ct.py` |
| `tutorial_13_heart_and_lung_motion.py` | `tutorial_13_combine_heart_and_lung_motion_chest_3d_ct.py` |
| `tutorial_14_duke_heart_shape_parameter_sweep.py` | `tutorial_14_shape_parameter_sweep_duke_heart_4d_labelmaps.py` |
| `tutorial_14_lung_shape_parameter_sweep.py` | `tutorial_14_shape_parameter_sweep_tcia_lung_4d_ct.py` |
| `tutorial_15_duke_heart_leave_one_out.py` | `tutorial_15_leave_one_out_duke_heart_4d_labelmaps.py` |
| `tutorial_15_lung_leave_one_out.py` | `tutorial_15_leave_one_out_tcia_lung_4d_ct.py` |
| `tutorial_16_duke_heart_physics_informed_motion_prep.py` | `tutorial_16_physics_informed_motion_prep_duke_heart_4d_labelmaps.py` |
| `tutorial_17_duke_heart_physics_informed_motion_train.py` | `tutorial_17_physics_informed_motion_train_duke_heart_4d_labelmaps.py` |
| `tutorial_18_duke_heart_physics_informed_motion_infer.py` | `tutorial_18_physics_informed_motion_infer_duke_heart_4d_labelmaps.py` |

| Module before | Module after | Class | Instance |
|---|---|---|---|
| `parameters_tcia_4d_lung` | `parameters_tcia_lung_4d_ct` | `ParametersTCIA4DLung` -> `ParametersTCIALung4DCT` | `TCIA_4D_LUNG` -> `TCIA_LUNG_4D_CT` |
| `parameters_heart_ct_kcl` | `parameters_kcl_heart_4d_meshes` | `ParametersHeartCTKCL` -> `ParametersKCLHeart4DMeshes` | `HEART_CT_KCL` -> `KCL_HEART_4D_MESHES` |
| `parameters_lung_ct_dirlab` | `parameters_dirlab_lung_4d_ct` | `ParametersLungCTDirLab` -> `ParametersDirLabLung4DCT` | `LUNG_CT_DIRLAB` -> `DIRLAB_LUNG_4D_CT` |
| `parameters_duke_heart_physics_informed` | `parameters_duke_heart_4d_labelmaps_physics` | `ParametersDukeHeartPhysicsInformed` -> `ParametersDukeHeart4DLabelmapsPhysics` | `DUKE_HEART_PHYSICS_INFORMED` -> `DUKE_HEART_4D_LABELMAPS_PHYSICS` |
| `parameters_duke_heart_labelmaps` | `parameters_duke_heart_4d_labelmaps` | `ParametersDukeHeartLabelmaps` -> `ParametersDukeHeart4DLabelmaps` | `DUKE_HEART` -> `DUKE_HEART_4D_LABELMAPS` |

**Before**

```python
from parameters_tcia_4d_lung import TCIA_4D_LUNG
```

**After**

```python
from parameters_tcia_lung_4d_ct import TCIA_LUNG_4D_CT
```

**Automated conversion:** for code that imports the parameter modules,
`sed -i -E 's/\bparameters_tcia_4d_lung\b/parameters_tcia_lung_4d_ct/g; s/\bParametersTCIA4DLung\b/ParametersTCIALung4DCT/g; s/\bTCIA_4D_LUNG\b/TCIA_LUNG_4D_CT/g; s/\bparameters_heart_ct_kcl\b/parameters_kcl_heart_4d_meshes/g; s/\bParametersHeartCTKCL\b/ParametersKCLHeart4DMeshes/g; s/\bHEART_CT_KCL\b/KCL_HEART_4D_MESHES/g; s/\bparameters_lung_ct_dirlab\b/parameters_dirlab_lung_4d_ct/g; s/\bParametersLungCTDirLab\b/ParametersDirLabLung4DCT/g; s/\bLUNG_CT_DIRLAB\b/DIRLAB_LUNG_4D_CT/g; s/\bparameters_duke_heart_physics_informed\b/parameters_duke_heart_4d_labelmaps_physics/g; s/\bParametersDukeHeartPhysicsInformed\b/ParametersDukeHeart4DLabelmapsPhysics/g; s/\bDUKE_HEART_PHYSICS_INFORMED\b/DUKE_HEART_4D_LABELMAPS_PHYSICS/g; s/\bparameters_duke_heart_labelmaps\b/parameters_duke_heart_4d_labelmaps/g; s/\bParametersDukeHeartLabelmaps\b/ParametersDukeHeart4DLabelmaps/g; s/\bDUKE_HEART\b/DUKE_HEART_4D_LABELMAPS/g' PATH`. Run the tutorials under their new names; outputs
written under the old `tutorials/output/` directory names are not reused.

## `WorkflowFinetuneICONRegistration` - finetuning is seeded by default

**Change:** the new keyword `seed` (default `0`) seeds Python, NumPy, and
PyTorch at the start of finetuning, so two runs with the same inputs start
from the same state, and results differ from an unseeded 2026.09.1 run.
`seed` is the last parameter, so positional calls are unaffected.

**Why:** reproducible starting points make finetuning results comparable
between runs.

**Before**

```python
WorkflowFinetuneICONRegistration(files, output_dir, "lung")  # unseeded
```

**After**

```python
WorkflowFinetuneICONRegistration(files, output_dir, "lung", seed=None)  # unseeded
```

**Automated conversion:** `None needed`. Pass `seed=None` to keep the old,
unseeded behavior.

## `[cuda12]` extra - CUDA 12.8 PyTorch, torch below 2.12

**Change:** the `cuda12` and `dev_cuda12` extras now take PyTorch from the CUDA
12.8 index (`cu128`) instead of CUDA 12.6 (`cu126`), and pin torch to
`>=2.10,<2.12` instead of `<2.13`. The matching prebuilt `torch-scatter` wheels
are on `https://data.pyg.org/whl/torch-2.11.0+cu128.html`, and installs from
PyPI must pass `--torch-backend=cu128` to uv.

**Why:** the CUDA 12.6 build has no kernels for Blackwell GPUs, and without
`--torch-backend` uv took torch from PyPI, which is CPU-only on Windows and a
CUDA 13 build on Linux. Torch 2.11 is the newest release on the CUDA 12.8 index,
and PyG publishes `torch-scatter` wheels for it on Linux and Windows.

**Before**

```bash
uv pip install "monai-physio[cuda12]" -f https://data.pyg.org/whl/torch-2.12.0+cu126.html
```

**After**

```bash
uv pip install "monai-physio[cuda12]" --torch-backend=cu128 -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

**Automated conversion:** `None needed`. Recreate existing CUDA 12.6
environments with the command above.
