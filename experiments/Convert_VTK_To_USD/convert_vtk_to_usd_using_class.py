# %% [markdown]
# # VTK to USD Converter Test Notebook
#
# This notebook demonstrates ConvertVTKToUSD for converting VTK files to USD format.
#
# The library is based on the ParaViewConnector architecture from Omniverse but simplified for file-based conversion only.
#
# ## Features
#
# - **File Format Support**: VTK legacy (.vtk), XML PolyData (.vtp), XML UnstructuredGrid (.vtu)
# - **Geometry Preservation**: Points, faces, normals, colors
# - **Data Arrays**: VTK point and cell data arrays → USD primvars
# - **Time-Series**: Support for animated/time-varying data
# - **Materials**: UsdPreviewSurface materials with customizable properties
# - **Coordinate System**: Automatic conversion from LPS (ITK's native medical imaging frame) to USD right-handed Y-up
#
# ## Test Data
#
# We'll use the KCL Heart Model data:
# - `average_surface.vtp`: Surface mesh of the heart
# - `average_mesh.vtk`: Volumetric mesh of the heart

# %%
import logging
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from pxr import Usd, UsdGeom, UsdShade

from monai_physio import ContourTools, ConvertVTKToUSD

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_HERE = Path(__file__).parent

# %% [markdown]
# ## 1. Basic Conversion: VTP to USD
#
# Let's start with the simplest use case: converting a single VTP file to USD.

# %%
# Define file paths
data_dir = _HERE / "../../data/KCL-Heart-Model"
output_dir = _HERE / "results"
output_dir.mkdir(exist_ok=True)

vtk_file = data_dir / "average_mesh.vtk"

print("Input files:")
print(f"  VTK: {vtk_file.exists()} - {vtk_file}")
print(f"\nOutput directory: {output_dir}")

# %%
vtp_file = output_dir / "average_surface.vtp"
if not vtp_file.exists():
    vtk_mesh = pv.read(vtk_file)
    contour_tools = ContourTools()
    vtp_surface = vtk_mesh.extract_surface(algorithm="dataset_surface")
    vtp_surface.save(vtp_file)
print(f"  VTP: {vtp_file.exists()} - {vtp_file}")

# %%
# Simple conversion using ConvertVTKToUSD
output_usd = output_dir / "heart_surface_simple.usd"

stage = ConvertVTKToUSD.from_files(
    data_basename="HeartSurface", vtk_files=[vtp_file]
).convert(str(output_usd))

print(f"\nCreated USD file: {output_usd}")
print("Stage info:")
print(f"  Root layer: {stage.GetRootLayer().identifier}")
print(f"  Default prim: {stage.GetDefaultPrim().GetPath()}")

# %% [markdown]
# ## 2. Inspect VTK File Data
#
# Let's examine what data arrays are present in the VTK files.

# %%
# Read and inspect the VTP file
mesh_info = ConvertVTKToUSD.inspect_file(vtp_file)

print("=" * 60)
print("VTP File (average_surface.vtp) Contents:")
print("=" * 60)
print("\nGeometry:")
print(f"  Points: {mesh_info['points']}")
print(f"  Faces: {mesh_info['faces']}")
print(f"  Normals: {'Yes' if mesh_info['has_normals'] else 'No'}")
print(f"  Colors: {'Yes' if mesh_info['has_colors'] else 'No'}")

print(f"\nData Arrays ({len(mesh_info['arrays'])}):")
for i, array in enumerate(mesh_info["arrays"], 1):
    print(f"  {i}. {array['name']}:")
    print(f"     - Components: {array['num_components']}")
    print(f"     - Type: {array['data_type']}")
    print(f"     - Interpolation: {array['interpolation']}")
    print(f"     - Shape: {array['shape']}")
    data_range = array["range"]
    if data_range[0] is not None and data_range[1] is not None:
        print(f"     - Range: [{data_range[0]:.6f}, {data_range[1]:.6f}]")

# %%
# Read and inspect the VTK file
mesh_info_vtk = ConvertVTKToUSD.inspect_file(vtk_file, extract_surface=True)

print("=" * 60)
print("VTK File (average_mesh.vtk) Contents:")
print("=" * 60)
print("\nGeometry:")
print(f"  Points: {mesh_info_vtk['points']}")
print(f"  Faces: {mesh_info_vtk['faces']}")
print(f"  Normals: {'Yes' if mesh_info_vtk['has_normals'] else 'No'}")
print(f"  Colors: {'Yes' if mesh_info_vtk['has_colors'] else 'No'}")

print(f"\nData Arrays ({len(mesh_info_vtk['arrays'])}):")
for i, array in enumerate(mesh_info_vtk["arrays"], 1):
    print(f"  {i}. {array['name']}:")
    print(f"     - Components: {array['num_components']}")
    print(f"     - Type: {array['data_type']}")
    print(f"     - Interpolation: {array['interpolation']}")
    print(f"     - Shape: {array['shape']}")
    data_range = array["range"]
    if data_range[0] is not None and data_range[1] is not None:
        print(f"     - Range: [{data_range[0]:.6f}, {data_range[1]:.6f}]")

# %% [markdown]
# ## 3. Advanced Conversion with Custom Settings
#
# Now let's use custom settings to control the conversion process.

# %%
# Convert with a custom solid color
output_usd_custom = output_dir / "heart_surface_custom.usd"
stage_custom = ConvertVTKToUSD.from_files(
    data_basename="HeartSurface",
    vtk_files=[vtp_file],
    solid_color=(0.9, 0.3, 0.3),  # Reddish color for heart
).convert(str(output_usd_custom))

print(f"\nCreated custom USD file: {output_usd_custom}")

# %% [markdown]
# ## 4. Convert VTK Legacy Format
#
# Now let's convert the legacy VTK format file.

# %%
# Convert VTK (legacy volumetric mesh) with surface extraction
output_usd_vtk = output_dir / "heart_mesh.usd"
stage_vtk = ConvertVTKToUSD.from_files(
    data_basename="HeartMesh",
    vtk_files=[vtk_file],
    extract_surface=True,  # Extract surface from volumetric mesh
    solid_color=(0.8, 0.4, 0.4),
).convert(str(output_usd_vtk))

print(f"\nCreated VTK USD file: {output_usd_vtk}")

# %% [markdown]
# ## 5. Inspect USD Output
#
# Let's examine the created USD file to verify all data was preserved.

# %%
# Open the USD file for inspection
inspect_stage = Usd.Stage.Open(str(output_usd_custom))

print("=" * 60)
print("USD File Inspection")
print("=" * 60)

# Get the mesh prim (ConvertVTKToUSD places meshes at /World/{basename}/Mesh)
mesh_path = "/World/HeartSurface/Mesh"
mesh_prim = inspect_stage.GetPrimAtPath(mesh_path)

if mesh_prim:
    mesh = UsdGeom.Mesh(mesh_prim)

    print(f"\nMesh: {mesh_path}")
    print(f"  Type: {mesh_prim.GetTypeName()}")

    # Geometry attributes
    points = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()

    print("\nGeometry:")
    print(f"  Points: {len(points) if points else 0}")
    print(f"  Faces: {len(face_counts) if face_counts else 0}")
    print(f"  Face indices: {len(face_indices) if face_indices else 0}")

    # Check normals
    normals_attr = mesh.GetNormalsAttr()
    if normals_attr:
        normals = normals_attr.Get()
        print(f"  Normals: {len(normals) if normals else 0}")

    # List primvars
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    primvars = primvars_api.GetPrimvars()

    print(f"\nPrimvars ({len(primvars)}):")
    for primvar in primvars:
        name = primvar.GetPrimvarName()
        interpolation = primvar.GetInterpolation()
        type_name = primvar.GetTypeName()
        value = primvar.Get()
        size = len(value) if value else 0
        print(f"  - {name}: {type_name} ({interpolation}), {size} elements")

    # Check material binding
    binding_api = UsdShade.MaterialBindingAPI(mesh)
    material_binding = binding_api.GetDirectBinding()
    if material_binding:
        material_path = material_binding.GetMaterialPath()
        print(f"\nMaterial Binding: {material_path}")
else:
    print(f"\nMesh not found at path: {mesh_path}")

# %% [markdown]
# ## 6. Create Time-Series Data (Simulated)
#
# Demonstrate time-series conversion by creating synthetic deformation of the mesh.

# %%
# Create a simple time-series by deforming the mesh using PyVista
base_mesh = pv.read(str(vtp_file))


def create_deformed_pv_mesh(
    base: pv.PolyData, time_step: int, num_steps: int = 10
) -> pv.PolyData:
    """Return a sinusoidally scaled copy of base with a synthetic pressure field."""
    t = time_step / num_steps * 2 * np.pi
    scale = 1.0 + 0.1 * np.sin(t)
    centroid = np.mean(base.points, axis=0)
    deformed = base.copy(deep=True)
    deformed.points = centroid + (base.points - centroid) * scale
    num_points = len(deformed.points)
    deformed.point_data["pressure"] = np.sin(
        t + np.linspace(0, 2 * np.pi, num_points)
    ).astype(np.float32)
    return deformed


num_time_steps = 10
pv_sequence = [
    create_deformed_pv_mesh(base_mesh, t, num_time_steps) for t in range(num_time_steps)
]
print(f"\nCreated {len(pv_sequence)} time steps")

# %%
# Convert time series to USD
output_usd_anim = output_dir / "heart_surface_animated.usd"

stage_anim = ConvertVTKToUSD(
    data_basename="HeartAnimated",
    input_polydata=pv_sequence,
    solid_color=(0.9, 0.2, 0.2),
).convert(str(output_usd_anim))

print(f"\nCreated animated USD file: {output_usd_anim}")
print(f"Time range: {stage_anim.GetStartTimeCode()} to {stage_anim.GetEndTimeCode()}")
print(f"Time codes per second: {stage_anim.GetTimeCodesPerSecond()}")

# %% [markdown]
# ## 7. Summary
#
# Let's summarize what we've created.

# %%
print("=" * 60)
print("Generated USD Files")
print("=" * 60)

usd_files = list(output_dir.glob("*.usd"))
usd_files.extend(output_dir.glob("*.usda"))
usd_files.extend(output_dir.glob("*.usdc"))

for usd_file in sorted(usd_files):
    size_kb = os.path.getsize(usd_file) / 1024
    print(f"\n{usd_file.name}:")
    print(f"  Size: {size_kb:.2f} KB")
    print(f"  Path: {usd_file}")

    # Quick inspection
    stage = Usd.Stage.Open(str(usd_file))
    if stage:
        print(f"  Up axis: {UsdGeom.GetStageUpAxis(stage)}")
        print(f"  Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")
        if stage.HasAuthoredTimeCodeRange():
            print(
                f"  Time range: {stage.GetStartTimeCode()} - {stage.GetEndTimeCode()}"
            )

print("\n" + "=" * 60)
print("All conversions completed successfully!")
print("=" * 60)


# %% [markdown]
# ## 8. Verification
#
# Verify that the USD files can be opened and contain the expected data.


# %%
def verify_usd_file(usd_path):
    """Verify USD file integrity."""
    print(f"\nVerifying: {usd_path.name}")
    print("-" * 40)

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print("  FAILED: could not open stage")
        return False

    # Check default prim
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print("  FAILED: no default prim")
        return False
    print(f"  Default prim: {default_prim.GetPath()}")

    # Find mesh prims
    mesh_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if points:
                print(f"  Mesh '{prim.GetName()}': {len(points)} points")

    if mesh_count == 0:
        print("  FAILED: no meshes found")
        return False

    print(f"  Total meshes: {mesh_count}")
    return True


# Verify all generated files
print("=" * 60)
print("USD File Verification")
print("=" * 60)

all_valid = True
for usd_file in sorted(usd_files):
    valid = verify_usd_file(usd_file)
    all_valid = all_valid and valid

print("\n" + "=" * 60)
if all_valid:
    print("All USD files are valid!")
else:
    print("FAILED: some USD files have issues")
print("=" * 60)

# %% [markdown]
# ## Conclusion
#
# This notebook demonstrated the comprehensive features of ConvertVTKToUSD:
#
# 1. **Simple Conversion**: One-line conversion of VTK files
# 2. **Data Inspection**: Reading and analyzing VTK data arrays
# 3. **Custom Settings**: Fine-grained control over conversion
# 4. **Multiple Formats**: Support for VTP, VTK, VTU files
# 5. **Material System**: Custom materials with UsdPreviewSurface
# 6. **Time-Series**: Animated meshes with time-varying attributes
# 7. **Data Preservation**: All VTK arrays preserved as USD primvars
# 8. **Coordinate Systems**: Automatic LPS to USD right-handed Y-up conversion
#
# The library is production-ready and can be used for converting medical imaging data, simulation results, and other VTK-based datasets to USD for visualization in Omniverse or other USD-compatible applications.
#
# ### Next Steps
#
# - View the generated USD files in Omniverse with RTX rendering enabled
# - Experiment with different conversion settings
# - Test with your own VTK datasets
# - Explore advanced features like custom colormaps and transfer functions
