# VTK to USD Advanced Library

`monai_physio.vtk_to_usd` is the public low-level VTK-to-USD conversion
layer used by `monai_physio.ConvertVTKToUSD`.

Repository workflows, experiments, and CLIs should use `ConvertVTKToUSD`.
Import this subpackage directly only when you need advanced file readers, data
containers, coordinate helpers, or USD writer primitives.

## Features

- File support: legacy VTK (`.vtk`), XML PolyData (`.vtp`), XML
  UnstructuredGrid (`.vtu`)
- Geometry: points, faces, topology, normals, vertex colors
- Data arrays: VTK point arrays as USD primvars; cell-array primvars are
  limited by surface topology and controlled by
  `ConversionSettings.preserve_cell_arrays`
- Materials: `UsdPreviewSurface` materials
- Coordinates: RAS millimeter coordinates converted to USD Y-up meters

## Modules

```text
vtk_to_usd/
  converter.py           # convert_vtk_file facade
  data_structures.py     # MeshData, MaterialData, ConversionSettings
  vtk_reader.py          # VTK file readers
  usd_utils.py           # coordinate conversion and primvar helpers
  material_manager.py    # UsdPreviewSurface creation and binding
  usd_mesh_converter.py  # MeshData to UsdGeom.Mesh writer
  mesh_utils.py          # splitting by connectivity or cell type
```

## Quick Start

```python
from monai_physio.vtk_to_usd import convert_vtk_file

stage = convert_vtk_file('mesh.vtp', 'output.usd')
```

## Custom Settings

```python
from monai_physio.vtk_to_usd import (
    ConversionSettings,
    MaterialData,
    convert_vtk_file,
)

settings = ConversionSettings(
    triangulate_meshes=True,
    compute_normals=True,
    meters_per_unit=1.0,
    up_axis='Y',
)

material = MaterialData(
    name='cardiac_tissue',
    diffuse_color=(0.9, 0.3, 0.3),
    roughness=0.4,
)

stage = convert_vtk_file(
    'heart.vtp',
    'heart.usd',
    data_basename='Heart',
    settings=settings,
    material=material,
)
```

## Time Series

Use the high-level package API for time series, colormaps, labels, and
application workflows:

```python
from monai_physio import ConvertVTKToUSD

stage = ConvertVTKToUSD.from_files(
    data_basename='AnimatedMesh',
    vtk_files=['frame_0.vtp', 'frame_1.vtp', 'frame_2.vtp'],
    time_codes=[0.0, 1.0, 2.0],
).convert('animated.usd')
```

## MeshData Inspection

```python
from monai_physio.vtk_to_usd import read_vtk_file

mesh_data = read_vtk_file('mesh.vtp')
print(len(mesh_data.points))
print(len(mesh_data.face_vertex_counts))
for array in mesh_data.generic_arrays:
    print(array.name, array.num_components, array.data_type)
```

## Public API

- `convert_vtk_file()`: Convert one VTK file to one USD stage.
- `read_vtk_file()`: Read `.vtk`, `.vtp`, or `.vtu` into `MeshData`.
- `ConversionSettings`: Conversion settings for the low-level writer.
- `MaterialData`: USD material settings.
- `MeshData`, `GenericArray`, `DataType`: data containers used by the writer.
- `MaterialManager`, `UsdMeshConverter`: advanced USD authoring primitives.

## Coordinate System

Input VTK coordinates are assumed to be RAS in millimeters:

- X: patient right
- Y: patient anterior
- Z: patient superior

USD output is Y-up in meters:

```text
USD(x, y, z) = RAS(x, z, -y) * 0.001
```

The stage metadata is authored as `metersPerUnit=1.0` and `upAxis='Y'`.

## Testing Policy

Repository tests validate this library through `ConvertVTKToUSD`, the supported
application-level API. Avoid adding direct tests for `vtk_to_usd` internals unless
the project explicitly changes this policy.
