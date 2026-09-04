# %%
import os

import itk
import numpy as np

from monai_physio import RegisterImagesGreedy, TransformTools

_HERE = os.path.dirname(os.path.abspath(__file__))

# %%
data_dir = os.path.join(_HERE, "..", "..", "data", "Slicer-Heart-CT")
files = [
    os.path.join(data_dir, f)
    for f in sorted(os.listdir(data_dir))
    if f.endswith(".mha") and f.startswith("slice_")
]

num_files = len(files)
files_indx = list(range(num_files))
reference_image_num = 7
reg_method_data = zip(["Greedy"], [RegisterImagesGreedy()], [[30, 15, 5]])
# reg_method_data = zip(["ICON"], [RegisterImagesICON()], [20])
# reg_method_data = zip(["ICON","ANTs"], [RegisterImagesICON(), RegisterImagesANTS()], [20, [40, 20, 10]])
# (requires importing RegisterImagesANTS above before uncommenting)

reference_image_file = os.path.join(
    data_dir, f"slice_{files_indx[reference_image_num]:03d}.mha"
)
reference_image_reg_use_identity = True

fixed_image = itk.imread(reference_image_file, pixel_type=itk.F)
_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
out_file = os.path.join(_RESULTS_DIR, "slice_fixed.mha")
itk.imwrite(fixed_image, out_file)

images = []
for file in files:
    img = itk.imread(file, pixel_type=itk.F)
    images.append(img)


# %%
def register_slices(
    reg_tool,
    reg_tool_name,
    fixed_image,
    images,
    files_indx,
    reference_image_num,
    reference_image_reg_use_identity,
    portion_of_prior_to_use=0.0,
):
    tfm_tools = TransformTools()

    img = images[reference_image_num]
    forward_transform = None
    inverse_transform = None
    results = None
    reg_image = None
    prior_forward_transform = None

    reference_image = images[reference_image_num]
    reference_image_indx = files_indx[reference_image_num]

    identity_tfm = itk.IdentityTransform[itk.D, 3].New()
    identity_tfm = tfm_tools.convert_transform_to_displacement_field_transform(
        identity_tfm, reference_image
    )

    if reference_image_reg_use_identity:
        print(
            f"Registering reference slice {reference_image_indx} using identify transform"
        )
        forward_transform = identity_tfm
        inverse_transform = identity_tfm
        if portion_of_prior_to_use > 0.0:
            prior_forward_transform = identity_tfm
        reg_image = img
        reg_image_inv = fixed_image
    else:
        print(f"Registering reference slice {reference_image_indx} to reference image.")
        results = reg_tool.register(img)
        forward_transform = results["forward_transform"]
        inverse_transform = results["inverse_transform"]
        if portion_of_prior_to_use > 0.0:
            prior_forward_transform = tfm_tools.combine_displacement_field_transforms(
                identity_tfm,
                forward_transform,
                reference_image,
                tfm1_weight=1.0,
                tfm2_weight=portion_of_prior_to_use,
                tfm1_blur_sigma=0.0,
                tfm2_blur_sigma=0.5,
                mode="add",
            )
        reg_image = tfm_tools.transform_image(img, forward_transform, fixed_image)
        reg_image_inv = tfm_tools.transform_image(fixed_image, inverse_transform, img)

    num_images = len(images)

    forward_transform_arr = [itk.Transform[itk.D, 3].New() for _ in range(num_images)]
    inverse_transform_arr = [itk.Transform[itk.D, 3].New() for _ in range(num_images)]
    forward_transform_arr[reference_image_num] = forward_transform
    inverse_transform_arr[reference_image_num] = inverse_transform

    debug_mode = True

    if debug_mode:
        out_file = os.path.join(
            _RESULTS_DIR,
            f"slice_{reg_tool_name}_forward_{reference_image_indx:03d}.mha",
        )
        itk.imwrite(reg_image, out_file, compression=True)

        out_file = os.path.join(
            _RESULTS_DIR,
            f"slice_fixed_{reg_tool_name}_inverse_{reference_image_indx:03d}.mha",
        )
        itk.imwrite(reg_image_inv, out_file, compression=True)

        itk.transformwrite(
            forward_transform,
            os.path.join(
                _RESULTS_DIR,
                f"slice_{reg_tool_name}_forward_{reference_image_indx:03d}.hdf",
            ),
            compression=True,
        )
        itk.transformwrite(
            inverse_transform,
            os.path.join(
                _RESULTS_DIR,
                f"slice_{reg_tool_name}_inverse_{reference_image_indx:03d}.hdf",
            ),
            compression=True,
        )

    prior_forward_transform_ref = prior_forward_transform

    for step_i in [1, -1]:
        start_i = 0
        end_i = num_files
        if step_i == -1:
            start_i = reference_image_num - 1
            end_i = -1
        else:
            start_i = reference_image_num + 1
            end_i = num_files

        prior_forward_transform = prior_forward_transform_ref

        if start_i != end_i:
            print(
                f"registering: from {files_indx[start_i]} to {files_indx[end_i - step_i]} step {step_i}"
            )
        for img_indx in range(start_i, end_i, step_i):
            img = images[img_indx]
            img_file_indx = files_indx[img_indx]
            print("   Registering slice", img_file_indx)

            # Try identity as initial transform
            print("     Trying init with identity.")
            results_init_identity = reg_tool.register(img)
            inverse_tranform_init_identity = results_init_identity["inverse_transform"]
            forward_transform_init_identity = results_init_identity["forward_transform"]
            loss_init_identity = results_init_identity["loss"]
            print("        Final loss:", results_init_identity["loss"])

            if portion_of_prior_to_use > 0.0:
                # Try with prior transform
                print("     Trying with init prior.")
                results_init_prior = reg_tool.register_from(
                    prior_forward_transform, img
                )
                inverse_transform_init_prior = results_init_prior["inverse_transform"]
                forward_transform_init_prior = results_init_prior["forward_transform"]
                loss_init_prior = results_init_prior["loss"]
                print("        Final loss:", results_init_prior["loss"])

                if loss_init_identity < loss_init_prior:
                    print("     Using identity.")
                    prior_forward_transform = identity_tfm
                    inverse_transform = inverse_tranform_init_identity
                    forward_transform = forward_transform_init_identity
                else:
                    print("     Using prior.")
                    inverse_transform = inverse_transform_init_prior
                    forward_transform = forward_transform_init_prior

                prior_forward_transform = (
                    tfm_tools.combine_displacement_field_transforms(
                        identity_tfm,
                        forward_transform,
                        reference_image,
                        tfm1_weight=1.0,
                        tfm2_weight=portion_of_prior_to_use,
                        tfm1_blur_sigma=0.0,
                        tfm2_blur_sigma=0.5,
                        mode="add",
                    )
                )
            else:
                inverse_transform = inverse_tranform_init_identity
                forward_transform = forward_transform_init_identity

            forward_transform_arr[img_indx] = forward_transform
            inverse_transform_arr[img_indx] = inverse_transform

            if debug_mode:
                reg_image = tfm_tools.transform_image(
                    img, forward_transform, fixed_image
                )
                out_file = os.path.join(
                    _RESULTS_DIR,
                    f"slice_{reg_tool_name}_forward_{img_file_indx:03d}.mha",
                )
                itk.imwrite(reg_image, out_file, compression=True)

                reg_image = tfm_tools.transform_image(
                    fixed_image, inverse_transform, img
                )
                out_file = os.path.join(
                    _RESULTS_DIR,
                    f"slice_fixed_{reg_tool_name}_inverse_{img_file_indx:03d}.mha",
                )
                itk.imwrite(reg_image, out_file, compression=True)

                itk.transformwrite(
                    forward_transform,
                    os.path.join(
                        _RESULTS_DIR,
                        f"slice_{reg_tool_name}_forward_{img_file_indx:03d}.hdf",
                    ),
                    compression=True,
                )
                itk.transformwrite(
                    inverse_transform,
                    os.path.join(
                        _RESULTS_DIR,
                        f"slice_{reg_tool_name}_inverse_{img_file_indx:03d}.hdf",
                    ),
                    compression=True,
                )

    return {
        "forward_transforms": forward_transform_arr,
        "inverse_transforms": inverse_transform_arr,
    }


# %%
forward_transform_arr = None
inverse_transform_arr = None
for reg_tool_name, reg_tool, num_iterations in reg_method_data:
    reg_tool.set_fixed_image(fixed_image)
    reg_tool.set_number_of_iterations(num_iterations)
    results = register_slices(
        reg_tool,
        reg_tool_name,
        fixed_image,
        images,
        files_indx,
        reference_image_num,
        reference_image_reg_use_identity,
        portion_of_prior_to_use=0.0,
    )
    forward_transform_arr = results["forward_transforms"]
    inverse_transform_arr = results["inverse_transforms"]

# %%
tfm_tool = TransformTools()

load_data = True

if load_data:
    files = []
    files_indx = []
    for f in sorted(os.listdir(_RESULTS_DIR)):
        if f.endswith(".hdf") and f.startswith("slice_ANTS_forward_"):
            files.append(os.path.join(_RESULTS_DIR, f))
            files_indx.append(int(f.split("_")[3].split(".")[0]))

    num_files = len(files)

    fixed_image = itk.imread(
        os.path.join(_RESULTS_DIR, "slice_fixed.mha"), pixel_type=itk.F
    )

grid_image = tfm_tool.generate_grid_image(fixed_image, 30, 1)

for i in range(num_files):
    print(files_indx[i])
    inverse_transform = itk.transformread(
        os.path.join(_RESULTS_DIR, f"slice_ANTS_inverse_{files_indx[i]:03d}.hdf")
    )[0]

    inverse_image = tfm_tool.convert_transform_to_displacement_field(
        inverse_transform,
        fixed_image,
        np_component_type=np.float32,
    )
    itk.imwrite(
        inverse_image,
        os.path.join(_RESULTS_DIR, f"slice_ANTS_inverse_{files_indx[i]:03d}_hdf.mha"),
        compression=True,
    )

    inverse_grid_image = tfm_tool.transform_image(
        grid_image,
        inverse_transform,
        fixed_image,
    )
    itk.imwrite(
        inverse_grid_image,
        os.path.join(
            _RESULTS_DIR, f"slice_fixed_ANTS_inverse_grid_{files_indx[i]:03d}.mha"
        ),
        compression=True,
    )
