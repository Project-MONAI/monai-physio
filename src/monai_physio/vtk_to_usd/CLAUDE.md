# CLAUDE.md - vtk_to_usd

This subpackage is a public advanced low-level library. Understand this before
touching it.

## Preferred In-Repo API

Do not add calls to `vtk_to_usd` from experiments, workflows, CLIs, or other
top-level MONAI Physio modules. The in-repository entry point for VTK-to-USD
conversion is:

```python
from monai_physio.convert_vtk_to_usd import ConvertVTKToUSD
```

External advanced users may import `monai_physio.vtk_to_usd` directly.

## Role of This Subpackage

`vtk_to_usd/` is the workhorse used by `ConvertVTKToUSD`. It provides:

- `convert_vtk_file()` for single-file low-level conversion
- VTK readers for `.vtk`, `.vtp`, and `.vtu`
- Low-level data structures: `MeshData`, `ConversionSettings`, `MaterialData`
- USD primitive writing: normals, primvars, time samples, materials

## When To Edit This Subpackage

Edit code here only when the change belongs in the file-based conversion layer
itself: readers, USD writers, data structures, coordinate transforms, or the
single-file facade. Otherwise, prefer `convert_vtk_to_usd.py`.

## Module Responsibilities

| File | Responsibility |
| --- | --- |
| `converter.py` | `convert_vtk_file()` single-file facade |
| `data_structures.py` | `MeshData`, `MaterialData`, etc. |
| `vtk_reader.py` | Read `.vtk`, `.vtp`, `.vtu` files into `MeshData` |
| `usd_utils.py` | Coordinate conversion and primvar helpers |
| `material_manager.py` | `UsdPreviewSurface` creation and binding |
| `usd_mesh_converter.py` | Write `MeshData` to a USD prim |
| `mesh_utils.py` | Mesh splitting helpers |

## Coordinate System

LPS-to-USD-Y-up conversion: `USD(x, y, z) = LPS(x, z, -y) * 0.001`.

MONAI Physio keeps images and surfaces in ITK's native LPS frame; the
resulting USD frame is right-handed Y-up with USD +X = patient Left,
USD +Y = patient Superior, USD +Z = patient Anterior.

This conversion happens inside `usd_utils.lps_to_usd()` and
`lps_points_to_usd()`. It must not be applied more than once.

## Testing Policy

Tests should exercise this subpackage through `ConvertVTKToUSD`. Do not add
direct tests for `vtk_to_usd` internals unless the project explicitly changes
this policy.
