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

## `ProcessVolumetricMesh.extract_tetrahedra` - now `extract_label_tetrahedra_vtk`, one mesh per label

**Change:** `ProcessVolumetricMesh.extract_tetrahedra(mask_image, element_size_mm,
anatomy_names)` is now
`ProcessVolumetricMesh.extract_label_tetrahedra_vtk(labelmap_image,
element_size_mm, label_ids)`.  It meshes every non-zero label of the image and
returns `{label id: pv.UnstructuredGrid}`, each grid tagged with
`field_data["SegmentationLabelIds"]`, mirroring
`ProcessSurfaceMesh.extract_label_surfaces`; a 0/1 mask gives `{1: mesh}`, and
an empty one `{}` instead of an empty grid.  `anatomy_names` is gone: color a
mesh with `ProcessUSDAnatomy.apply_anatomy_color`.  It sits beside two new
meshers with the same shape, `extract_label_tetrahedra_netgen` (the optional
`netgen` extra) and `extract_label_tetrahedra_simpleware` (a licensed
Simpleware Medical install).

**Why:** a labelmap's structures - chambers, myocardium, vessels - are now
meshed apart in one call, each keeping its identity through warping and into
the USD, rather than fused into one solid; neighboring labels' meshes meet on
coincident nodes.  The `extract_label_*` names and the labelmap argument match
`extract_label_surfaces`, and the suffix says which mesher a call uses, so
switching meshers is a one-word change.

**Before**

```python
from monai_physio import ProcessVolumetricMesh

mesh = ProcessVolumetricMesh().extract_tetrahedra(
    mask, element_size_mm=1.5, anatomy_names=["heart"]
)
```

**After**

```python
from monai_physio import ProcessUSDAnatomy, ProcessVolumetricMesh

meshes = ProcessVolumetricMesh().extract_label_tetrahedra_vtk(mask, element_size_mm=1.5)
mesh = meshes[1]
ProcessUSDAnatomy(stage=None).apply_anatomy_color(mesh, ["heart"])
```

**Automated conversion:** `py utils/migrate_extract_label_tetrahedra.py [--dry-run] PATH [PATH ...]`
rewrites each call and appends `[1]`; it reports calls passing
`anatomy_names` for a hand edit.  Masks whose foreground is not 1, or that may
be empty, also need a hand edit (index by the foreground value, or use
`.get(1)`).  Run `utils/migrate_process_contours.py` first for code still
using `ProcessContours`.

## Registration dependencies - published fine-tuning support

**Change:** registration now requires `icon-registration>=1.1.9` and
`unigradicon>=1.0.6` from the package index. The uniGradICON fine-tuning branch
override is removed. Public workflow signatures and checkpoint paths are
unchanged.

**Why:** uniGradICON 1.0.6 includes `unigradicon.finetuning.finetune`, so
registration and fine-tuning can use the same published dependencies without a
branch checkout or local package patches.

**Before**

```bash
pip install "unigradicon @ git+https://github.com/uncbiag/uniGradICON.git@feat-add-finetuning"
```

**After**

```bash
pip install --upgrade "icon-registration>=1.1.9" "unigradicon>=1.0.6"
```

Remove any old uniGradICON source path from `PYTHONPATH` and leave
`unigradicon_src_path=None` unless deliberately testing a source checkout.

**Automated conversion:** None needed.
