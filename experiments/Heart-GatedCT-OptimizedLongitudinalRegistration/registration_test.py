# %% [markdown]
# # Registration test: pm0003 time point 20 -> time point 60
#
# Registers pm0003 gated CT time point 20 (moving) to time point 60
# (fixed) with deformable registration, then warps time point 20 into
# time point 60's space and writes it to disk.
#
# Switch backends by editing the single ``method`` variable below
# ("ANTS", "ICON", or "Greedy").  All paths are hard-coded; run the
# cells top to bottom.

# %%
import time
from pathlib import Path

import itk

from monai_physio.register_images_ants import RegisterImagesANTS
from monai_physio.register_images_greedy import RegisterImagesGreedy
from monai_physio.register_images_icon import RegisterImagesICON
from monai_physio.transform_tools import TransformTools

# %% [markdown]
# ## 1. Configuration and hard-coded paths
#
# Change ``method`` to switch backends.  Time point 20 is the moving
# image; time point 60 is the fixed image.

# %%
method = "Greedy"  # one of: "ANTS", "ICON", "Greedy"

data_dir = Path("d:/MONAI-Physio/duke_data/gated_nii/pm0003")
moving_path = data_dir / "pm0003_dupr_135-0094_135_4700_g020_s2.000_n0058_11.nii.gz"
fixed_path = data_dir / "pm0003_dupr_135-0094_135_4700_g060_s2.000_n0058_15.nii.gz"

output_dir = Path(__file__).parent / "results" / "registration_test"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"pm0003_g020_to_g060_{method.lower()}.mha"

# %% [markdown]
# ## 2. Load the fixed (time point 60) and moving (time point 20) images

# %%
fixed_image = itk.imread(str(fixed_path), pixel_type=itk.F)
moving_image = itk.imread(str(moving_path), pixel_type=itk.F)
print(f"Fixed  (g060): {fixed_path.name}")
print(f"Moving (g020): {moving_path.name}")

# %% [markdown]
# ## 3. Build and configure the registration backend
#
# ANTS and Greedy share ``set_transform_type``/``set_metric`` and take a
# per-level iteration list; ICON takes a single iteration count and has
# no transform-type/metric setters.

# %%
if method == "ANTS":
    reg = RegisterImagesANTS()
    reg.set_transform_type("Deformable")
    reg.set_metric("MeanSquares")
    reg.set_number_of_iterations([40, 20, 10])
elif method == "Greedy":
    reg = RegisterImagesGreedy()
    reg.set_transform_type("Deformable")
    # NCC (CC) beats SSD for same-modality CT; tighter update-field smoothing
    # (first sigma) captures more cardiac motion while staying diffeomorphic.
    reg.set_metric("CC")
    reg.set_number_of_iterations([40, 20, 10])
    reg.deformable_smoothing = "1.0vox 0.5vox"
elif method == "ICON":
    reg = RegisterImagesICON()
    reg.set_number_of_iterations(50)
else:
    raise ValueError(f"Unknown method: {method}")

reg.set_modality("ct")
reg.set_fixed_image(fixed_image)

# %% [markdown]
# ## 4. Register time point 20 to time point 60

# %%
t_start = time.perf_counter()
reg_result = reg.register(moving_image=moving_image)
elapsed = time.perf_counter() - t_start

forward_transform = reg_result["forward_transform"]
loss = float(reg_result["loss"])
print(f"{method} registration done in {elapsed:.1f} s, loss={loss:.4f}")

# %% [markdown]
# ## 5. Warp time point 20 into time point 60's space and save
#
# ``forward_transform`` is the transform consumed by ``transform_image`` to
# resample the moving image onto the fixed grid (it supplies the fixed->moving
# sampling map the ITK resampler needs). ``inverse_transform`` is the opposite
# direction, used to warp the fixed image onto the moving grid (e.g. in
# ``RegisterTimeSeriesImages.reconstruct_time_series``). This holds for all
# three backends (ANTS, ICON, Greedy).

# %%
transform_tools = TransformTools()
warp_t_start = time.perf_counter()
warped_image = transform_tools.transform_image(
    moving_image,
    forward_transform,
    fixed_image,
    interpolation_method="linear",
)
itk.imwrite(warped_image, str(output_path), compression=True)
warp_elapsed = time.perf_counter() - warp_t_start
print(f"Wrote warped time point 20 -> 60: {output_path}")

# %% [markdown]
# ## 6. Timing report
#
# Wall-clock seconds for the registration and the warp/write step.  The
# registration time dominates and is the figure to compare across backends;
# for ICON it includes the one-time network load on this first (and only)
# pair.

# %%
total_elapsed = elapsed + warp_elapsed
print()
print("=" * 44)
print(f"Timing report ({method})")
print("=" * 44)
print(f"{'Step':<22}{'seconds':>12}")
print("-" * 44)
print(f"{'register':<22}{elapsed:>12.2f}")
print(f"{'warp + write':<22}{warp_elapsed:>12.2f}")
print("-" * 44)
print(f"{'total':<22}{total_elapsed:>12.2f}")
print("=" * 44)
