"""
Tutorial 3 (Lung): Reconstruct High-Resolution 4D CT

Purpose
-------
Register a respiratory CT time series to a fixed reference phase and save the
reconstructed frames. DIR-Lab does not provide a separate high-resolution
breath-hold reference image, so this tutorial uses the T70 (end-exhale) phase
as the fixed reference - the same reference Tutorial 8 fits its lung SSM to.

Data Required
-------------
Full data: ``data/DirLab-4DCT/Case1Pack_T??.mha``
Test data: ``data/test/DirLab-4DCT/Case1Pack_T??.mha``

Outputs (under ``tutorials/output/tutorial_03_lung/``)
-----------------------------------------------------
- ``reconstructed_frame_<i>.mha`` plus ``_fwd.hdf`` / ``_inv.hdf`` transforms
  for every phase
- ``reference_frame.png`` and ``reconstructed_frame.png`` screenshots
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path

import itk

from parameters_base import ParametersBase
from monai_physio import (
    RegisterImagesGreedy,
    TestTools,
    WorkflowReconstructHighres4DCT,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_03_lung_reconstruct_highres_4d_ct"

    # Only the shared directory roots are needed here; no dataset-specific
    # parameters module applies to this tutorial.
    tutorial_paths = ParametersBase()
    test_mode = TestTools.running_as_test()

    output_dir = tutorial_paths.output_directory(test_mode) / "tutorial_03_lung"
    baselines_dir = repo_root / "tests" / "baselines"

    # .mha files are DirLab-4DCT data already converted to HU by
    # data/DirLab-4DCT/fix_downloaded_data.py.
    case_glob = "Case1Pack_T??.mha"

    if test_mode:
        data_dir = tutorial_paths.data_directory(test_mode) / "DirLab-4DCT"
        number_of_iterations_greedy = [1, 0]
    else:
        data_dir = tutorial_paths.data_directory(test_mode) / "DirLab-4DCT"
        number_of_iterations_greedy = [30, 15, 7, 3]

    log_level = logging.INFO

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    registration_method = RegisterImagesGreedy(log_level=log_level)
    registration_method.set_number_of_iterations(number_of_iterations_greedy)

    phase_files = sorted(data_dir.glob(case_glob))
    if not phase_files:
        raise FileNotFoundError(
            f"No DirLab phase images found under {data_dir}.\n"
            "See data/README.md for download instructions."
        )

    time_series = [itk.imread(str(path)) for path in phase_files]
    # T70 (end-exhale) is the DIR-Lab reference phase used throughout the
    # tutorials; fall back to the last phase when it is absent (test data).
    reference_time_frame = next(
        (index for index, path in enumerate(phase_files) if path.stem.endswith("T70")),
        len(time_series) - 1,
    )
    reference_image = time_series[reference_time_frame]

    # Workflow initialization

    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=time_series,
        reference_image=reference_image,
        reference_time_frame=reference_time_frame,
        # The reference image *is* the reference frame here, so registering the
        # two would be a self-registration; use an identity transform instead.
        register_reference_time_frame_to_reference_image=False,
        registration_method=registration_method,
        log_level=log_level,
    )
    workflow.set_modality("ct")

    # Workflow execution
    result = workflow.process()

    # Result saving
    fixed_to_moving_transform = result["fixed_to_moving_transforms"]
    moving_to_fixed_transform = result["moving_to_fixed_transforms"]
    reconstructed_images: list[itk.Image] = result["reconstructed_images"]
    reconstructed_files: list[Path] = []
    for frame_index, image in enumerate(reconstructed_images):
        out_path = output_dir / f"reconstructed_frame_{frame_index:03d}.mha"
        itk.imwrite(image, str(out_path), compression=True)
        reconstructed_files.append(out_path)

        out_path = output_dir / f"reconstructed_frame_{frame_index:03d}_fwd.hdf"
        itk.transformwrite(fixed_to_moving_transform[frame_index], str(out_path))

        out_path = output_dir / f"reconstructed_frame_{frame_index:03d}_inv.hdf"
        itk.transformwrite(moving_to_fixed_transform[frame_index], str(out_path))

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    screenshots.append(
        tt.save_screenshot_image_slice(
            reference_image,
            "reference_frame.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
        )
    )
    if reconstructed_images:
        screenshots.append(
            tt.save_screenshot_image_slice(
                reconstructed_images[0],
                "reconstructed_frame.png",
                axis=0,
                slice_fraction=0.5,
                colormap="gray",
            )
        )

    tutorial_results = {
        "reconstructed_images": reconstructed_images,
        "reconstructed_files": reconstructed_files,
        "screenshots": screenshots,
    }
