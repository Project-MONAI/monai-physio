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
