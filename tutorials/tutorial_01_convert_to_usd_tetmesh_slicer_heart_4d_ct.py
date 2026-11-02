"""
Tutorial 1 (Slicer Heart, Tetmesh): Heart-Gated 4D CT to Animated Tetrahedral Mesh

Purpose
-------
Convert a cardiac-gated 4D CT scan (multiple cardiac phases) into a time series
of tetrahedral volume meshes of the heart, one per phase. Unlike Tutorial 1,
which builds a surface-only animated USD model of the whole labelmap, this
variant takes only the heart and contrast structures, fills their
reference-frame surface with tetrahedra and warps that single volume
mesh through every phase's registration transform, giving a 4D tetmesh
suitable for physics-based simulation (e.g. finite-element cardiac motion)
rather than visualization alone.

Inputs
------
- A set of 3D CT volumes (``slice_???.mha``) representing successive cardiac
  phases.
  Expected location: ``data/Slicer-Heart-CT/slice_???.mha``.
- The phase roughly 70% through the series is used as the reference frame for
  segmentation, meshing, and registration.

Outputs
-------
Under ``tutorials/output/tutorial_01_convert_to_usd_tetmesh_slicer_heart_4d_ct/``:

- One tetrahedral mesh (``*.vtu``) per cardiac phase, named
  ``heart_tetmesh_<phase>.vtu``, all sharing the reference mesh's connectivity.
- ``heart_tetmesh.usd``: the phases as one animated ``UsdGeomTetMesh``, one
  time sample per phase, for viewing in NVIDIA Omniverse.
- A screenshot (PNG) of the reference-phase heart surface for documentation and
  regression testing: ``heart_surface_test.png``.

Strengths
---------
- Single reference tetmesh, warped rather than re-meshed per phase, so every
  frame shares the same connectivity -- required for a 4D finite-element
  simulation to track individual elements across phases.
- Registers on the CPU with ``RegisterImagesGreedy``; no GPU needed for this
  stage.  Its stationary-velocity mode and a 1.0-voxel warp smoothing keep
  the warps from folding and inverting the tetrahedra they carry.
- Flat boundary slivers are removed from the reference tetmesh before it is
  warped (``ProcessVolumetricMesh.remove_boundary_slivers``), and any element
  a deformation still inverts is repaired per frame via
  ``ProcessVolumetricMesh.repair_inverted_tetrahedra``.

Weaknesses / Limitations
------------------------
- Segmentation quality depends on the segmenter's training distribution;
  unusual pathologies or pediatric anatomy may degrade results.
- The heart and contrast structures are meshed as one solid, so chambers,
  myocardium and blood pool share a single tetmesh rather than one per
  structure.  ``tetmesh_anatomy_groups`` in the script picks the structures.
- Large 4D datasets (>20 phases, high resolution) can require 32 GB+ RAM.

Classes Used
------------
- WorkflowConvertImageToVTK (workflow_convert_image_to_vtk.py):
    Segments the reference phase with ``SLICER_HEART_4D_CT.segmenter``
    (SegmentChestTotalSegmentatorWithContrast) and builds one surface and
    tetmesh of its heart and contrast structures.
- WorkflowReconstructHighres4DCT (workflow_reconstruct_highres_4d_ct.py):
    Registers every phase to the reference with ``SLICER_HEART_4D_CT.registrar``
    (RegisterImagesGreedy, with ``set_stationary_velocity(True)`` and
    ``set_deformable_smoothing("2.0vox", "1.0vox")``).
- ProcessTransforms (process_transforms.py):
    Warps the reference tetmesh to each phase.
- ConvertVTKToUSD (convert_vtk_to_usd.py):
    Writes the phase tetmeshes as an animated ``UsdGeomTetMesh``.
- ProcessVolumetricMesh (process_volumetric_mesh.py):
    Removes boundary slivers from the reference tetmesh and repairs the warped
    tetmesh per phase.

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: Slicer-Heart-CT - https://github.com/SlicerHeart/SlicerHeart
This script expects the ``slice_???.mha`` phase volumes to already exist under
``data/Slicer-Heart-CT/``, as for Tutorial 1.
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import itk
from parameters_slicer_heart_4d_ct import SLICER_HEART_4D_CT

from monai_physio import (
    ConvertVTKToUSD,
    MONAIPhysioBase,
    ProcessTests,
    ProcessTransforms,
    ProcessVolumetricMesh,
    RegisterImagesGreedy,
    WorkflowConvertImageToVTK,
    WorkflowReconstructHighres4DCT,
)

# Only run if this script is not imported as a module

# nnUNetv2 (used by TotalSegmentator) spawns a multiprocessing.Pool. On
# Windows the spawn start method re-imports this script in each child;
# without the __name__ == "__main__" guard around the top-level work, that
# re-import fires the pipeline again and Python's spawn-cascade detector
# raises RuntimeError.
if __name__ == "__main__":
    # Data directory specification

    class_name = "tutorial_01_convert_to_usd_tetmesh_slicer_heart_4d_ct"

    test_mode = ProcessTests.running_as_test()

    output_dir = (
        SLICER_HEART_4D_CT.output_directory(test_mode)
        / "tutorial_01_convert_to_usd_tetmesh_slicer_heart_4d_ct"
    )

    data_dir = SLICER_HEART_4D_CT.input_directory(test_mode)

    if test_mode:
        frame_files = sorted(data_dir.glob("slice_???.mha"))[0:2]
    else:
        frame_files = sorted(data_dir.glob("slice_???.mha"))

    log_level = logging.INFO
    reporter = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    registration_method = cast(
        RegisterImagesGreedy,
        SLICER_HEART_4D_CT.registrar(test_mode, log_level=log_level),
    )
    # The reference tetmesh is carried through every phase's warp, and a warp
    # that folds inverts the elements it folds.  Greedy's default 0.5-voxel
    # warp smoothing folds on this series even with -sv; 1.0 voxel does not.
    registration_method.set_stationary_velocity(True)
    registration_method.set_deformable_smoothing("2.0vox", "1.0vox")
    segmentation_method = SLICER_HEART_4D_CT.segmenter(test_mode, log_level=log_level)

    transform_tools = ProcessTransforms(log_level=log_level)
    volume_tools = ProcessVolumetricMesh(log_level=log_level)

    # Directory setup and data reading

    output_dir.mkdir(parents=True, exist_ok=True)

    input_filenames = [str(path) for path in frame_files]
    if not input_filenames:
        raise FileNotFoundError(
            "Slicer-Heart-CT data not found. Checked:\n"
            + f"  - {data_dir}"
            + "\n"
            + "See data/README.md for download instructions."
        )

    time_series_images = [itk.imread(str(path)) for path in input_filenames]
    # Frame 7 of the full series; test mode loads fewer frames than that.
    reference_index = min(7, len(time_series_images) - 1)
    reference_image = time_series_images[reference_index]

    reporter.log_info("Number of time-series images: %d", len(time_series_images))

    # Reference-frame segmentation and tetmesh construction

    # The heart and the contrast-filled blood pool are meshed together, as one
    # structure, so the tetmesh fills the chambers rather than only the walls.
    tetmesh_anatomy_groups = ["heart", "contrast"]
    tetmesh_name = "heart_and_contrast"
    taxonomy = segmentation_method.taxonomy
    tetmesh_label_ids = [
        label_id
        for group in tetmesh_anatomy_groups
        for label_id in taxonomy.labels_in_group(group)
    ]

    heart_result = WorkflowConvertImageToVTK(
        segmentation_method=segmentation_method, log_level=log_level
    ).process(
        input_image=reference_image,
        anatomy_groups=tetmesh_anatomy_groups,
        groups={tetmesh_name: tetmesh_label_ids},
        surface_reduction_rate=SLICER_HEART_4D_CT.surface_reduction_rate,
        tetrahedra_element_size_mm=SLICER_HEART_4D_CT.mesh_element_size_mm,
    )
    if tetmesh_name not in heart_result["tetrahedra"]:
        raise ValueError(
            f"No {tetmesh_name} tetrahedra: the segmented {tetmesh_anatomy_groups} "
            "structures are missing or thinner than one element."
        )
    reference_surface = heart_result["surfaces"][tetmesh_name]
    # Flat boundary caps invert under modest compression and cannot be
    # repaired after the warp, so they are dropped from the reference first.
    reference_tetmesh = volume_tools.remove_boundary_slivers(
        heart_result["tetrahedra"][tetmesh_name]
    )
    reporter.log_info(
        "Reference tetmesh: %d points, %d cells",
        reference_tetmesh.n_points,
        reference_tetmesh.n_cells,
    )

    # Registration and per-phase tetmesh warping

    fixed_to_moving_transforms = WorkflowReconstructHighres4DCT(
        time_series_images=time_series_images,
        reference_image=reference_image,
        reference_time_frame=reference_index,
        register_reference_time_frame_to_reference_image=False,
        registration_method=registration_method,
        log_level=log_level,
    ).register_time_series()["fixed_to_moving_transforms"]

    tetmesh_files: list[Path] = []
    phase_tetmeshes = []
    for i, fixed_to_moving_transform in enumerate(fixed_to_moving_transforms):
        phase_tetmesh = volume_tools.repair_inverted_tetrahedra(
            transform_tools.transform_pvcontour(
                reference_tetmesh, fixed_to_moving_transform
            )
        )
        tetmesh_file = output_dir / f"heart_tetmesh_{i:03d}.vtu"
        phase_tetmesh.save(tetmesh_file)
        tetmesh_files.append(tetmesh_file)
        phase_tetmeshes.append(phase_tetmesh)

    # Animated USD: every phase shares the reference connectivity, so the
    # stage holds one UsdGeomTetMesh whose points are sampled per phase.

    usd_file = output_dir / "heart_tetmesh.usd"
    ConvertVTKToUSD(
        "heart_tetmesh",
        phase_tetmeshes,
        frames_per_second=float(len(phase_tetmeshes)),
        solid_color=cast(
            tuple[float, float, float],
            tuple(float(c) for c in reference_tetmesh.field_data["AnatomyColor"][:3]),
        ),
        output_tet_mesh=True,
        log_level=log_level,
    ).convert(str(usd_file))
    reporter.log_info("USD: %s", usd_file)

    # Result saving
    tt = ProcessTests(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = [
        tt.save_screenshot_mesh(
            reference_surface,
            "heart_surface_test.png",
            camera_position="iso",
            color="indianred",
            opacity=0.85,
        )
    ]

    tutorial_results = {
        "tetmesh_files": [str(path) for path in tetmesh_files],
        "usd_file": str(usd_file),
        "screenshots": screenshots,
    }
