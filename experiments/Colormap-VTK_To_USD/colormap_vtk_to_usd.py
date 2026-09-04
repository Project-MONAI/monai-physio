# %% [markdown]
# # Colormap Features for VTK to USD Conversion
#
# This notebook demonstrates the colormap features of `ConvertVTKToUSD` for visualizing point data arrays in NVIDIA Omniverse.
#
# ## Features Demonstrated
#
# 1. **Pre-defined colormaps**: plasma, viridis, rainbow, heat, coolwarm, grayscale, random
# 2. **Custom intensity ranges**: Control value-to-color mapping
# 3. **Point data visualization**: Map scalar data to colors on 3D meshes
# 4. **Time-varying data**: Create animated USD files with colored meshes
# %% [markdown]
# ## Setup and Imports

# %%
from pathlib import Path

import numpy as np
import pyvista as pv

from monai_physio import ConvertVTKToUSD

_HERE = Path(__file__).parent


# Create output directory
output_dir = _HERE / "output"
output_dir.mkdir(exist_ok=True)

print("MONAI Physio Colormap Examples")
print("=" * 50)


# %% [markdown]
# ## Helper Function: Create Example Meshes
#
# This function creates sphere meshes with synthetic time-varying data to demonstrate colormap functionality.


# %%
def create_example_mesh_with_data(time_step):
    """
    Create a sphere mesh with synthetic data for demonstration.

    Parameters
    ----------
    time_step : int
        Current time step for animation

    Returns
    -------
    pyvista.PolyData
        Sphere mesh with point data arrays
    """
    # Create a sphere
    sphere = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)

    # Add synthetic point data (e.g., simulating transmembrane potential)
    points = sphere.points
    z_coords = points[:, 2]
    theta = 2 * np.pi * time_step / 10.0  # Full cycle every 10 frames

    # Simulate transmembrane potential: -80 mV (rest) to +20 mV (depolarized)
    potential = -80.0 + 100.0 * (0.5 + 0.5 * np.sin(3 * z_coords + theta))
    sphere.point_data["transmembrane_potential"] = potential

    # Add temperature data example
    temperature = 20.0 + 15.0 * (0.5 + 0.5 * np.cos(2 * z_coords - theta))
    sphere.point_data["temperature"] = temperature

    return sphere


print("Helper function defined successfully")

# %% [markdown]
# ## Example 1: Default Plasma Colormap with Automatic Range
#
# The plasma colormap is the default and provides a perceptually uniform gradient from purple to pink to orange.

# %%
print("\nExample 1: Plasma colormap (default) with auto range")
print("-" * 50)

# Create time series of meshes
meshes = [create_example_mesh_with_data(t) for t in range(20)]

# Initialize converter
converter = ConvertVTKToUSD(
    data_basename="CardiacModel", input_polydata=meshes, mask_ids=None
)

# List available arrays for coloring
print("Available point data arrays:")
available = converter.list_available_arrays()
for name, info in available.items():
    print(f"  - {name}: range={info['range']}, dtype={info['dtype']}")

# Set colormap (automatic range)
converter.set_colormap(
    color_by_array="transmembrane_potential",
    colormap="plasma",
    intensity_range=None,  # Auto-detect from data
)

# Convert to USD
output_file = output_dir / "example1_plasma_auto.usd"
converter.convert(str(output_file))
print(f"\nCreated: {output_file}")

# %% [markdown]
# ## Example 2: Rainbow Colormap with Custom Range
#
# The rainbow colormap provides a classic ROYGBIV spectrum. Here we specify a custom physiological range for cardiac action potentials.

# %%
print("\nExample 2: Rainbow colormap with custom range [-80, 20] mV")
print("-" * 50)

meshes = [create_example_mesh_with_data(t) for t in range(20)]

converter = ConvertVTKToUSD(
    data_basename="CardiacModel", input_polydata=meshes, mask_ids=None
)

print("Setting colormap")
converter.set_colormap(
    color_by_array="transmembrane_potential",
    colormap="rainbow",
    intensity_range=(-80.0, 20.0),  # Physiological range for action potential
)

output_file = output_dir / "example2_rainbow_custom.usd"
print("Creating file:", output_file)
stage = converter.convert(str(output_file))
print(f"Created: {output_file}")

# %% [markdown]
# ## Example 3: Heat Colormap for Temperature Data
#
# The heat colormap (black-red-yellow-white) is ideal for temperature or intensity visualizations.

# %%
print("\nExample 3: Heat colormap for temperature visualization")
print("-" * 50)

meshes = [create_example_mesh_with_data(t) for t in range(20)]

converter = ConvertVTKToUSD(
    data_basename="TemperatureModel", input_polydata=meshes, mask_ids=None
)

converter.set_colormap(
    color_by_array="temperature",
    colormap="hot",  # 'heat' alias maps to 'hot' colormap
    intensity_range=(15.0, 40.0),  # Temperature range in Celsius
)

output_file = output_dir / "example3_heat_temperature.usd"
stage = converter.convert(str(output_file))
print(f"Created: {output_file}")

# %% [markdown]
# ## Example 4: Coolwarm (Diverging) Colormap
#
# The coolwarm colormap is a diverging colormap (blue-white-red) useful for data centered around a midpoint.

# %%
print("\nExample 4: Coolwarm colormap for diverging data")
print("-" * 50)

meshes = [create_example_mesh_with_data(t) for t in range(20)]

converter = ConvertVTKToUSD(
    data_basename="CardiacModel", input_polydata=meshes, mask_ids=None
)

converter.set_colormap(
    color_by_array="transmembrane_potential",
    colormap="coolwarm",
    intensity_range=(-80.0, 20.0),
)

output_file = output_dir / "example4_coolwarm_diverging.usd"
stage = converter.convert(str(output_file))
print(f"Created: {output_file}")

# %% [markdown]
# ## Example 5: Grayscale Colormap
#
# The grayscale colormap provides a simple black-to-white gradient for monochrome visualizations.

# %%
print("\nExample 5: Grayscale colormap")
print("-" * 50)

meshes = [create_example_mesh_with_data(t) for t in range(20)]

converter = ConvertVTKToUSD(
    data_basename="CardiacModel", input_polydata=meshes, mask_ids=None
)

converter.set_colormap(
    color_by_array="transmembrane_potential", colormap="grayscale", intensity_range=None
)

output_file = output_dir / "example5_grayscale.usd"
stage = converter.convert(str(output_file))
print(f"Created: {output_file}")

# %% [markdown]
# ## Example 6: Random Colormap for Categorical Data
#
# The random colormap assigns random colors to different values, making it useful for visualizing categorical or region-based data.

# %%
print("\nExample 6: Random colormap for categorical visualization")
print("-" * 50)

meshes = [create_example_mesh_with_data(t) for t in range(20)]

# Add categorical-like data (discrete regions)
for mesh in meshes:
    z_values = mesh.points[:, 2]
    regions = np.floor(3 * (z_values + 1) / 2)  # 3 regions
    mesh.point_data["region_id"] = regions

converter = ConvertVTKToUSD(
    data_basename="RegionModel", input_polydata=meshes, mask_ids=None
)

converter.set_colormap(
    color_by_array="region_id", colormap="random", intensity_range=None
)

output_file = output_dir / "example6_random_categorical.usd"
stage = converter.convert(str(output_file))
print(f"Created: {output_file}")

# %% [markdown]
# ## Example 7: Method Chaining for Concise API Usage
#
# The `set_colormap()` method supports chaining, allowing for more concise code.

# %%
print("\nExample 7: Method chaining with viridis colormap")
print("-" * 50)

meshes = [create_example_mesh_with_data(t) for t in range(20)]

output_file = output_dir / "example7_viridis_chained.usd"

# Method chaining for concise code
stage = (
    ConvertVTKToUSD(data_basename="CardiacModel", input_polydata=meshes, mask_ids=None)
    .set_colormap(
        color_by_array="transmembrane_potential",
        colormap="viridis",
        intensity_range=(-80.0, 20.0),
    )
    .convert(str(output_file))
)

print(f"Created: {output_file}")

# %% [markdown]
# ## Summary: Available Colormaps and Features
#
# ### Colormap Options
#
# | Colormap | Description | Best For |
# |----------|-------------|----------|
# | `plasma` | Purple-pink-orange gradient (default) | General purpose, perceptually uniform |
# | `viridis` | Blue-green-yellow gradient | General purpose, colorblind-friendly |
# | `rainbow` | Classic rainbow spectrum (ROYGBIV) | Full range visualization |
# | `heat` | Black-red-yellow-white | Temperature, intensity data |
# | `coolwarm` | Blue-white-red (diverging) | Data centered around zero/midpoint |
# | `grayscale` | Black to white linear | Monochrome, publication figures |
# | `random` | Random colors per value | Categorical/discrete data |
#
# ### Intensity Range Options
#
# - **`None`**: Automatic range from data min/max
# - **`(vmin, vmax)`**: Custom range tuple, e.g., `(-80.0, 20.0)`
#
# ### Key API Methods
#
# - **`list_available_arrays()`**: List all point data arrays available for coloring
# - **`set_colormap()`**: Configure colormap settings (supports method chaining)
# - **`convert(output_file)`**: Perform USD conversion with specified output path

# %%
print("\n" + "=" * 50)
print("All examples completed!")
print("=" * 50)
print(f"\nOutput files created in: {output_dir.absolute()}")
print("\nView these USD files in NVIDIA Omniverse to see the colormap visualizations.")
