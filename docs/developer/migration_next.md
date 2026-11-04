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

## Lung shape-model fits - pass only the lung labels as `patient_labelmap`

**Change:** the TCIA-lung tutorials (7, 8, 12, 15) and the lung Tutorial 00
notebook now pass `WorkflowFitStatisticalModelToPatient` a labelmap holding
only the segmenter's lung labels, instead of its whole-body labelmap. Code that
fits the lung model itself should do the same. The lung tutorials also stay on
`SegmentNVSegmentCTMRI`, the segmenter the shape model and the pretrained
network were built from.

**Why:** the fit's PCA stage builds its distance map from `patient_labelmap`
and scores every labelled voxel as on-target. Given the whole body, the lung
model could grow into the heart, liver and chest wall at no cost. Other
segmenters' "lung" groups also hold structures the model never saw, such as
TotalSegmentator's pulmonary vessels and airways, which pin the fit's
coefficients at their bounds.

**Before**

```python
WorkflowFitStatisticalModelToPatient(
    template_model=pca_mean_surface,
    patient_models=[lung_surface],
    patient_image=image,
    patient_labelmap=segmentation_result["labelmap"],
)
```

**After**

```python
lung_labelmap = segmenter.create_anatomy_group_labelmaps(
    segmentation_result["labelmap"]
)["lung"]
WorkflowFitStatisticalModelToPatient(
    template_model=pca_mean_surface,
    patient_models=[lung_surface],
    patient_image=image,
    patient_labelmap=lung_labelmap,
)
```

Fits, and so the coefficients a lung network is trained on, change: re-run
Tutorials 8 and 9 to retrain a lung network on them.

**Automated conversion:** None needed.

## `RegisterModelsICPITK` - rebuilt as point-set ICP matching `RegisterModelsICP`

**Change:** `RegisterModelsICPITK` no longer fits a model to a distance map
built from `reference_image` with SciPy. It now subclasses `RegisterModelsICP`
and runs the same pipeline (centroid alignment, bounding-box scaling, then
rigid, similarity or affine ICP stages), with each stage minimizing ITK's
`EuclideanDistancePointSetToPointSetMetricv4`. The constructor drops
`reference_image` and `point_subsample_step` and gains `rotation_axis` and
`number_of_points`. `register()` takes the arguments of
`RegisterModelsICP.register()`: `initial_transform`, `method`, `scale_bound`,
`skew_bound`, `versor_bound` and `translation_bound` are gone, and the result
no longer holds `mean_distance`. `set_reference_image()` and
`set_fixed_model()` are removed. `WorkflowFitStatisticalModelToPatient` now
runs its Stage 1 ICP with `RegisterModelsICPITK`, so its `icp_registrar` is a
`RegisterModelsICPITK` and its fits change: re-run Tutorials 8 and 9 to
retrain a network on them.

**Why:** the class is now a drop-in alternative to `RegisterModelsICP` that
can hold transform parameters fixed: `rotation_axis` limits every stage to
rotation about one world axis. Its gradient-based stages also converge closer
than VTK's ICP on the lung models.

**Before**

```python
registrar = RegisterModelsICPITK(fixed_model=patient_surface, reference_image=image)
result = registrar.register(
    moving_model=model_surface,
    transform_type="Affine",
    max_iterations=100,
    method="L-BFGS-B",
)
```

**After**

```python
registrar = RegisterModelsICPITK(fixed_model=patient_surface)
result = registrar.register(
    moving_model=model_surface,
    transform_type="Affine",
    max_iterations=100,
)
```

**Automated conversion:** None needed.
