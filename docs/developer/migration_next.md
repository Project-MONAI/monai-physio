# Migration Guide - Unreleased

Breaking changes committed since the last release, and how to update code that
depends on them.

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
saves it; `repair_mesh_primvar_element_sizes` loses its `save` parameter. The
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
