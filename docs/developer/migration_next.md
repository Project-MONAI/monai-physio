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

## Registration transform naming - `forward`/`inverse` renamed to `fixed_to_moving`/`moving_to_fixed`

**Change:** every registration class now returns/exposes exactly two transform
names, spelled out by direction instead of by "forward"/"inverse":

- `fixed_to_moving_transform` - `TransformPoint()` maps a fixed-space
  coordinate to the corresponding moving-space coordinate.
- `moving_to_fixed_transform` - `TransformPoint()` maps a moving-space
  coordinate to the corresponding fixed-space coordinate.

This affects `RegisterImagesANTS`/`RegisterImagesICON`/`RegisterImagesGreedy`/
`RegisterImagesGreedyICON`/`RegisterImagesChain`/`RegisterTimeSeriesImages`/
`RegisterModelsDistanceMaps` (previously `forward_transform`/
`inverse_transform`, or the plural `forward_transforms`/`inverse_transforms`
for time series), and `RegisterModelsPCA`/`RegisterModelsICP`/
`RegisterModelsICPITK` (previously `forward_point_transform`/
`inverse_point_transform`). `WorkflowCreateStatisticalModel`,
`WorkflowCreateMeanSurface`, and `WorkflowFitStatisticalModelToPatient`
follow the same rename on their own `forward_transforms`/`l2l_forward_transform`/
`l2i_forward_transform`/etc. attributes. `post_pca_transform` is unaffected.

**Why:** "forward"/"inverse" did not name one fixed direction. Warping an
image is a pull-back (needs a fixed-to-moving point map) while warping a
point/mesh is a push-forward (needs the opposite map), so the same word meant
opposite things depending on whether you were warping an image or a point -
for the *same* registration result. Worse, image registration's
`forward_transform` and PCA/ICP's `forward_point_transform` pointed in
opposite real-world directions for the same "moving to fixed" concept. See
`docs/developer/transform_conventions.rst` for the full explanation. Naming
the transform by its literal source/target space removes both ambiguities:
the correct transform for any operation is now just "the one whose name
matches the space you have and the space you want."

**The rename is not symmetric between the two families** - this is the
detail most likely to be missed migrating by hand:

| Old name | Class family | New name |
|---|---|---|
| `forward_transform(s)` | Image registration | `fixed_to_moving_transform(s)` |
| `inverse_transform(s)` | Image registration | `moving_to_fixed_transform(s)` |
| `forward_point_transform` | PCA / ICP | `moving_to_fixed_transform` |
| `inverse_point_transform` | PCA / ICP | `fixed_to_moving_transform` |

**Before**

```python
# Image registration
result = RegisterImagesANTS().register(moving_image)
warped = transform_tools.transform_image(
    moving_image, result["forward_transform"], fixed_image
)
warped_points = transform_tools.transform_pvcontour(
    points, result["inverse_transform"]
)

# PCA/ICP model registration
result = registrar.compute_pca_transforms(reference_image)
warped_template = transform_tools.transform_pvcontour(
    template_points, result["forward_point_transform"]
)
```

**After**

```python
# Image registration
result = RegisterImagesANTS().register(moving_image)
warped = transform_tools.transform_image(
    moving_image, result["fixed_to_moving_transform"], fixed_image
)
warped_points = transform_tools.transform_pvcontour(
    points, result["moving_to_fixed_transform"]
)

# PCA/ICP model registration
result = registrar.compute_pca_transforms(reference_image)
warped_template = transform_tools.transform_pvcontour(
    template_points, result["moving_to_fixed_transform"]
)
```

**Automated conversion:** none provided. A blind find-replace is unsafe here
because the same old name maps to a *different* new name depending on which
class produced the dict/attribute (see table above), and static analysis
cannot always tell which family a bare `result["forward_transform"]` belongs
to without tracing back to the registrar that built `result`. Search your
code for `forward_transform`, `inverse_transform`, `forward_point_transform`,
and `inverse_point_transform`, and at each hit, check which `Register*` class
produced the value: `RegisterImages*`/`RegisterModelsDistanceMaps`/
`RegisterTimeSeriesImages` use the top row of the table, `RegisterModelsPCA`/
`RegisterModelsICP`/`RegisterModelsICPITK` use the bottom row.

## `WorkflowCreateStatisticalModel.inverse_transforms` - removed

**Change:** the attribute is gone. `forward_transforms` (the surviving
attribute) was itself later renamed to `fixed_to_moving_transforms` by the
registration transform naming change above.

**Why:** it was populated but never read - not by the workflow, nor by any
tutorial, test or CLI in the repository. Each entry held a `CompositeTransform`
owning dense full-grid displacement fields, so on a cohort of a few dozen
samples the list retained gigabytes of host memory for the life of the workflow.
That was enough, on a memory-capped Linux host, to get the process killed by the
OOM killer partway through building a shape model. Removing it roughly halves
the workflow's peak memory and costs nothing, because the value had no consumer.

Callers that genuinely need the inverse of a correspondence can invert the
matching `fixed_to_moving_transforms` entry with `itk.Transform.GetInverse`,
which computes it on demand rather than holding one per sample for the whole
run.

**Before**

```python
workflow.process()
inverse = workflow.inverse_transforms[index]
```

**After**

```python
workflow.process()
inverse = itk.CompositeTransform[itk.D, 3].New()
if not workflow.fixed_to_moving_transforms[index].GetInverse(inverse):
    raise RuntimeError(f"Correspondence transform {index} is not invertible")
```

``GetInverse`` returns whether it succeeded rather than raising, and leaves
*inverse* unusable when it does not, so the result has to be checked.

**Automated conversion:** `None needed` - no caller in the repository, the
tutorials, the tests or the CLI referenced the attribute.

## Install extras - replaced by `cuda12`, `cuda13`, `dev`, `dev_cuda12`, `dev_cuda13`

**Change:** the `all`, `physicsnemo`, `docs`, and `test` extras are removed,
and `dev` is redefined. The five extras going forward are `cuda12`/`cuda13`
(CUDA-matched PyTorch plus CuPy), `dev` (now test/lint/docs tooling, no CUDA
component, replacing the old `docs`/`test` extras), and
`dev_cuda12`/`dev_cuda13` (both combined). The AI-surrogate packages formerly
behind `[physicsnemo]` -- `nvidia-physicsnemo`, `torch-geometric`,
`torch-scatter` -- are base dependencies, installed with every install.

**Why:** each extra now names the CUDA version it targets, so there is one
full-bundle name per CUDA version and no version-unaware `all`. The
AI-surrogate packages are CPU-capable, so gating them behind an extra split
the workflow set without a supporting technical constraint; as base
dependencies, every install runs every workflow.

**Consequence:** `torch-scatter` compiles against the installed torch and
declares no build-system dependency, so installs that resolve it without a
matching prebuilt wheel need `setuptools` present and build isolation
disabled. `uv` applies this from `pyproject.toml`; `pip` needs the flag
explicitly. `[cuda12]` pins torch inside the prebuilt-wheel range and avoids
the build entirely.

**Before**

```bash
uv pip install "monai-physio[all]"
uv pip install -e ".[cuda13]"
pip install "monai-physio[physicsnemo]"
pip install monai-physio[dev]
pip install monai-physio[test]
pip install monai-physio[docs]
```

**After**

```bash
uv pip install "monai-physio[cuda12]"              # recommended; [cuda13] for CUDA 13
uv pip install -e ".[dev_cuda12]"                  # source/dev install
uv pip install --torch-backend=auto monai-physio   # auto-detected PyTorch, no CuPy
```

**Automated conversion:** `None needed` - these are install-time extra names,
not Python symbols; no code referenced `[physicsnemo]` or the other removed
extras, so only install commands change.

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
