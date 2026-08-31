# Migration Guide — Unreleased

Breaking changes committed since the last release, and how to update code that
depends on them.

PhysioTwin4D prefers compatibility: public APIs are broken only when the change
is generally beneficial to future users. When a break is unavoidable, the
project does **not** ship deprecation shims or removed-symbol stubs. Instead,
substantial changes ship with code that automates the conversion, and every
break is recorded here in the commit that introduces it.

At release time this file is renamed `migration_<version>.md` and a fresh
`migration_next.md` is started for the next cycle.

## `WorkflowCreateStatisticalModel.inverse_transforms` — removed

**Change:** the attribute is gone. `forward_transforms` is unaffected.

**Why:** it was populated but never read — not by the workflow, nor by any
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

**Automated conversion:** `None needed` — no caller in the repository, the
tutorials, the tests or the CLI referenced the attribute.

## Entry template

Append one section per breaking change, newest last, using this shape:

````markdown
## <symbol, module, or CLI flag> — <one-line summary>

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
