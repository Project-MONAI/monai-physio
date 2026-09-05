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

## `WorkflowCreateStatisticalModel.inverse_transforms` - removed

**Change:** the attribute is gone. `forward_transforms` is unaffected.

**Why:** it was populated but never read - not by the workflow, nor by any
tutorial, test or CLI in the repository. Each entry held a `CompositeTransform`
owning dense full-grid displacement fields, so on a cohort of a few dozen
samples the list retained gigabytes of host memory for the life of the workflow.
That was enough, on a memory-capped Linux host, to get the process killed by the
OOM killer partway through building a shape model. Removing it roughly halves
the workflow's peak memory and costs nothing, because the value had no consumer.

Callers that genuinely need the inverse of a correspondence can invert the
matching `forward_transforms` entry with `itk.Transform.GetInverse`, which
computes it on demand rather than holding one per sample for the whole run.

**Before**

```python
workflow.process()
inverse = workflow.inverse_transforms[index]
```

**After**

```python
workflow.process()
inverse = itk.CompositeTransform[itk.D, 3].New()
if not workflow.forward_transforms[index].GetInverse(inverse):
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
