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

## Exterior falloff distance - replaced by a fade proportional to the surface displacement

**Change:** `ProcessTransforms.restrict_deformation_field_to_normal_falloff_outside_mask`
no longer takes `falloff_distance_mm`, and
`WorkflowInferMovement.process_time_series` no longer takes
`exterior_falloff_distance_mm`. Outside the mask, the displacement now comes
from the nearest mask voxel and fades over
`falloff_length_per_displacement` (in `process_time_series`,
`exterior_falloff_length_per_displacement`; default 3.0) times that voxel's
displacement, never less than the smallest voxel spacing.

**Why:** a fixed fade distance shorter than about 1.5 times the organ's
displacement folds the warp. When a heart or lung contracts, the space it
vacates was then filled with its own unmoved tissue instead of the tissue
around it. A fade proportional to the displacement cannot fold, adapts to each
point of the surface, and no longer reads the unreliable far tail of the
spread field.

**Before**

```python
transform_tools.restrict_deformation_field_to_normal_falloff_outside_mask(
    field, normals, mask, direction_offset_mm, direction_transition_mm, 2.5
)
WorkflowInferMovement(infer).process_time_series(
    ..., exterior_mask=mask, exterior_falloff_distance_mm=10.0
)
```

**After**

```python
transform_tools.restrict_deformation_field_to_normal_falloff_outside_mask(
    field, normals, mask, direction_offset_mm, direction_transition_mm
)
WorkflowInferMovement(infer).process_time_series(..., exterior_mask=mask)
```

Pass `falloff_length_per_displacement` (or
`exterior_falloff_length_per_displacement`) only to change the default 3.0;
values below 1.5 can fold the warp.

**Automated conversion:** None needed. Delete the `falloff_distance_mm` /
`exterior_falloff_distance_mm` argument.

## `combine_transforms_with_masks`, `reduce_folding_in_field` - removed

**Change:** `ProcessTransforms.combine_transforms_with_masks` and
`ProcessTransforms.reduce_folding_in_field` are removed.
`detect_folding_in_field` and `compute_jacobian_determinant_from_field` remain.

**Why:** nothing in MONAI Physio used them. Scaling the displacement only where
the Jacobian determinant is low does not remove folding; it creates new steps
at the edges of the scaled region, and on a heart warp it grew 982 folded
voxels to 4,098. Keeping them suggested a fix that does not work.

**Before**

```python
combined = transform_tools.combine_transforms_with_masks(
    heart_transform, lung_transform, heart_mask, lung_mask, reference_ct
)
field = transform_tools.reduce_folding_in_field(field, jacobian)
```

**After**

```python
jacobian = transform_tools.compute_jacobian_determinant_from_field(field)
if transform_tools.detect_folding_in_field(jacobian, threshold=0.0):
    ...  # fix the field's source, e.g. a smoother or longer falloff
```

To combine two motions, restrict each with
`restrict_deformation_field_to_normal_falloff_outside_mask` against its own
organ mask and compose them, as tutorial 13 does for heart and lung motion.

**Automated conversion:** None needed; no replacement is one-to-one.
