# %% [markdown]
# # 4D CT Reconstruction Using RegisterTimeSeriesImages Class
#
# This notebook demonstrates the use of the `RegisterTimeSeriesImages` class to register a time series of CT images to a common reference frame.
#
# This is a refactored version of `reconstruct_4d_ct.ipynb` that uses the new class-based approach, including:
# - Registration of time series images using Greedy, ICON, or Greedy+ICON methods
# - Reconstruction of time series using the `reconstruct_time_series()` method
# - Optional upsampling to fixed image resolution while preserving spatial positioning
#

# %%
import os
from typing import Optional

import itk
import numpy as np

from monai_physio import (
    RegisterImagesBase,
    RegisterImagesGreedy,
    RegisterImagesGreedyICON,
    RegisterImagesICON,
    RegisterTimeSeriesImages,
    TransformTools,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _build_registrar(method_name: str, iterations=None) -> Optional[RegisterImagesBase]:
    """Build a registrar instance for one of "Greedy", "ICON", or
    "Greedy_ICON". When `iterations` is given, it matches this experiment's
    `number_of_iterations_list` shape: a list for Greedy, an int for ICON,
    or [greedy_list, icon_int] for Greedy_ICON. `iterations=None` builds an
    unconfigured instance (used where only the transform-reconstruction
    step is needed, which does not depend on iteration counts)."""
    if method_name == "Greedy":
        greedy = RegisterImagesGreedy()
        if iterations is not None:
            greedy.set_number_of_iterations(iterations)
        return greedy
    if method_name == "ICON":
        icon = RegisterImagesICON()
        if iterations is not None:
            icon.set_number_of_iterations(iterations)
        return icon
    if method_name == "Greedy_ICON":
        greedy_icon = RegisterImagesGreedyICON()
        if iterations is not None:
            greedy_icon.greedy.set_number_of_iterations(iterations[0])
            greedy_icon.icon.set_number_of_iterations(iterations[1])
        return greedy_icon
    if method_name == "Default":
        return None
    raise ValueError(f"Unknown registration method: {method_name}")


# %%
# Load image files
data_dir = os.path.join(_HERE, "..", "..", "data", "Slicer-Heart-CT")
files = [
    os.path.join(data_dir, f)
    for f in sorted(os.listdir(data_dir))
    if f.endswith(".mha") and f.startswith("slice_")
]

print(f"Found {len(files)} slice files")

# %%
num_files = len(files)
files_indx = list(range(num_files))
reference_image_num = 7

# Registration parameters - Greedy_ICON is the recommended method
registration_method_names = [
    "Default"
]  # Use default, or ["Greedy", "ICON", "Greedy_ICON"]
number_of_iterations_list = [None]  # [
# [30, 15, 7, 3],
# 20,  # For ICON
# [[30, 15, 7, 3], 20],  # For Greedy_ICON
# ]

# Common parameters
reference_image_file = os.path.join(
    data_dir, f"slice_{files_indx[reference_image_num]:03d}.mha"
)
register_start_to_reference = False
portion_of_prior_transform_to_init_next_transform = 0.0

print(f"Number of files: {num_files}")
print(f"Reference image: slice_{files_indx[reference_image_num]:03d}.mha")
print(f"Registration method names: {registration_method_names}")
print(f"Number of iterations: {number_of_iterations_list}")

# %% [markdown]
# ## Load Images
#

# %%
# Load fixed/reference image
fixed_image = itk.imread(reference_image_file, pixel_type=itk.F)
print(f"Fixed image size: {itk.size(fixed_image)}")
print(f"Fixed image spacing: {itk.spacing(fixed_image)}")

# Save fixed image for reference
_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
out_file = os.path.join(_RESULTS_DIR, "slice_fixed.mha")
itk.imwrite(fixed_image, out_file)
print(f"Saved fixed image to: {out_file}")

images = []
for file in files:
    img = itk.imread(file, pixel_type=itk.F)
    images.append(img)

# %% [markdown]
# ## Perform Time Series Registration
#
# Loop through each registration method and perform registration.
#
# The registration produces:
# - **Forward transforms**: Transform moving images to fixed space (moving → fixed)
# - **Inverse transforms**: Transform fixed image to moving space (fixed → moving)
# - **Losses**: Registration quality metric for each time point
#

# %%
tfm_tools = TransformTools()

# Loop through each registration method
for method_idx, registration_method_name in enumerate(registration_method_names):
    number_of_iterations = number_of_iterations_list[method_idx]

    print("\n" + "=" * 70)
    print(f"Starting registration with {registration_method_name.upper()}")
    print("=" * 70)
    print(f"  Starting index: {reference_image_num}")
    print(f"  Register start to reference: {register_start_to_reference}")
    print(
        f"  Prior transform weight: {portion_of_prior_transform_to_init_next_transform}"
    )
    print(f"  Number of iterations: {number_of_iterations}")

    # Create registrar for this method
    registration_method = _build_registrar(
        registration_method_name, number_of_iterations
    )
    registrar = RegisterTimeSeriesImages(registration_method=registration_method)

    registrar.set_modality("ct")
    registrar.set_fixed_image(fixed_image)

    # Perform registration
    result = registrar.register_time_series(
        moving_images=images,
        reference_frame=reference_image_num,
        register_reference=register_start_to_reference,
        prior_weight=portion_of_prior_transform_to_init_next_transform,
    )

    forward_transforms = result["forward_transforms"]
    inverse_transforms = result["inverse_transforms"]
    losses = result["losses"]

    print(f"\n{registration_method_name.upper()} registration complete!")
    print(f"  Average loss: {np.mean(losses):.6f}")
    print(f"  Min loss: {np.min(losses):.6f}")
    print(f"  Max loss: {np.max(losses):.6f}")

    # Reconstruct time series using the new method (moving to fixed space)
    # This applies the inverse transforms to each moving image
    print("  Reconstructing time series in fixed image space...")
    reconstructed_images = registrar.reconstruct_time_series(
        moving_images=images,
        inverse_transforms=inverse_transforms,
        upsample_to_fixed_resolution=True,
    )

    # Save reconstructed images and inverse transforms
    for i, img_indx in enumerate(files_indx):
        print(f"  Saving slice {img_indx:03d}...")

        # Save reconstructed image (moving to fixed using inverse transform)
        out_file = os.path.join(
            _RESULTS_DIR,
            f"slice_{registration_method_name}_recon{img_indx:03d}.mha",
        )
        itk.imwrite(reconstructed_images[i], out_file, compression=True)

        # Also save forward-transformed images (moving to fixed using forward transform)
        # This shows the moving image aligned to fixed space
        reg_image = tfm_tools.transform_image(
            images[i], forward_transforms[i], fixed_image
        )
        out_file = os.path.join(
            _RESULTS_DIR,
            f"slice_{registration_method_name}_{img_indx:03d}_fixedSpace.mha",
        )
        itk.imwrite(reg_image, out_file, compression=True)

        # Save transforms
        itk.transformwrite(
            forward_transforms[i],
            os.path.join(
                _RESULTS_DIR,
                f"slice_{registration_method_name}_forward_{img_indx:03d}.hdf",
            ),
            compression=True,
        )
        itk.transformwrite(
            inverse_transforms[i],
            os.path.join(
                _RESULTS_DIR,
                f"slice_{registration_method_name}_inverse_{img_indx:03d}.hdf",
            ),
            compression=True,
        )

    for i, img_indx in enumerate(files_indx):
        status = "(reference)" if i == reference_image_num else ""
        print(f"  Slice {img_indx:03d}: {losses[i]:.6f} {status}")

    print(f"  Mean loss: {np.mean(losses):.6f}")
    print(f"  Std loss: {np.std(losses):.6f}")
    print(f"  Min loss: {np.min(losses):.6f}")
    print(f"  Max loss: {np.max(losses):.6f}")

    # Generate grid image for visualization
    grid_image = tfm_tools.generate_grid_image(fixed_image, 30, 1)

    print(f"Generating {registration_method_name.upper()} grid visualizations...")
    for i, img_indx in enumerate(files_indx):
        print(f"  Generating grid for slice {img_indx:03d}...")

        # Transform grid with inverse transform (FM)
        inverse_grid_image = tfm_tools.transform_image(
            grid_image,
            inverse_transforms[i],
            fixed_image,
        )
        itk.imwrite(
            inverse_grid_image,
            os.path.join(
                _RESULTS_DIR,
                f"slice_fixed_{registration_method_name}_inverse_grid_{img_indx:03d}.mha",
            ),
            compression=True,
        )

        # Save displacement field as image
        inverse_transform_image = tfm_tools.convert_transform_to_displacement_field(
            inverse_transforms[i],
            fixed_image,
            np_component_type=np.float32,
        )
        itk.imwrite(
            inverse_transform_image,
            os.path.join(
                _RESULTS_DIR,
                f"slice_{registration_method_name}_inverse_{img_indx:03d}_field.mha",
            ),
            compression=True,
        )
    print(f"Grid visualizations saved for {registration_method_name.upper()}")
