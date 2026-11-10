"""
Tutorial 1 (Slicer Heart, Tetmesh): Heart-Gated 4D CT to Animated Tetrahedral Mesh

Purpose
-------
Convert a cardiac-gated 4D CT scan (multiple cardiac phases) into a time series
of tetrahedral volume meshes of the heart, one per phase. Unlike Tutorial 1,
which builds a surface-only animated USD model of the whole labelmap, this
variant takes only the heart and contrast structures, fills each of their
reference-frame labels with its own tetrahedral mesh and warps those volume
meshes through every phase's registration transform, giving a 4D tetmesh per
structure suitable for physics-based simulation (e.g. finite-element cardiac
motion) rather than visualization alone.

The reference labelmap is meshed three ways, and each mesher's meshes are
warped and saved on their own, so the meshers can be compared on the same
anatomy and motion:

- ``vtk``: ``ProcessVolumetricMesh.extract_label_tetrahedra_vtk`` (voxels
  split into tetrahedra by VTK), each label relaxed onto its surface by
  ``trim_tetrahedra_to_surface``.
- ``netgen``: ``ProcessVolumetricMesh.extract_label_tetrahedra_netgen`` (needs
  the ``netgen`` extra).
- ``simpleware``: ``ProcessVolumetricMesh.extract_label_tetrahedra_simpleware``,
  only when Simpleware Medical is installed at
  ``ProcessVolumetricMesh.simpleware_exe_path``; skipped otherwise.

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

For each mesher ``<mesher>`` (``vtk``, ``netgen``, and ``simpleware`` when
available):

- One tetrahedral mesh (``*.vtu``) per cardiac phase, named
  ``heart_tetmesh_<mesher>_<phase>.vtu``, holding every label's tetrahedra
  (cell array ``SegmentationLabelIds``), all sharing that mesher's reference
  connectivity.
- ``heart_tetmesh_<mesher>.usd``: the phases as one animated
  ``UsdGeomTetMesh`` per label, grouped and colored by anatomy, one time
  sample per phase, for viewing in NVIDIA Omniverse.

And once:

- A screenshot (PNG) of the reference-phase heart surface for documentation and
  regression testing: ``heart_surface_test.png``.

Strengths
---------
- Single reference tetmesh per mesher, warped rather than re-meshed per
  phase, so every frame shares the same connectivity -- required for a 4D
  finite-element simulation to track individual elements across phases.
- Three meshers side by side: structured voxel-based elements, and graded
  unstructured elements from netgen and from Simpleware Medical.
- Each reference label is cleaned first (largest connected piece, enclosed
  cavities filled where no other label is), so every mesher fills one solid
  per label.
- Each label stays a separate, named structure through the warp and into the
  USD.
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
- Each label is its own tetmesh, and neighbors' interface nodes are not
  shared.  Simpleware's coincide but are not merged; trimming moves the vtk
  meshes' apart; netgen re-tiles each label on its own, so its meshes may
  overlap or gap slightly.  A simulation coupling the structures must tie
  them.
  ``tetmesh_anatomy_groups`` in the script picks the structures.
- Large 4D datasets (>20 phases, high resolution) can require 32 GB+ RAM.
- All three tetmeshes use 2 mm elements (``mesh_element_size_mm``): at
  1 mm the smoothed surface netgen meshes touches itself at thin walls and
  netgen rejects it.
- Needs the ``netgen`` extra (``uv pip install "monai-physio[netgen]"``); the
  Simpleware mesh additionally needs a Simpleware Medical install licensed for
  +FE Free.

Classes Used
------------
- WorkflowConvertImageToVTK (workflow_convert_image_to_vtk.py):
    Segments the reference phase with ``SLICER_HEART_4D_CT.segmenter``
    (SegmentChestTotalSegmentatorWithContrast) and builds one surface of its
    heart and contrast structures.
- WorkflowReconstructHighres4DCT (workflow_reconstruct_highres_4d_ct.py):
    Registers every phase to the reference with ``SLICER_HEART_4D_CT.registrar``
    (RegisterImagesGreedy, with ``set_stationary_velocity(True)`` and
    ``set_deformable_smoothing("2.0vox", "1.0vox")``).
- ProcessTransforms (process_transforms.py):
    Warps the reference tetmesh to each phase.
- ConvertVTKToUSD (convert_vtk_to_usd.py):
    Writes the phase tetmeshes as one animated ``UsdGeomTetMesh`` per label.
- ProcessUSDAnatomy (process_usd_anatomy.py):
    Gives each label's prim its anatomy's material.
- ProcessLabelmaps (process_labelmaps.py):
    Cleans each reference label before meshing (``remove_islands``,
    ``fill_holes``).
- ProcessSurfaceMesh (process_surface_mesh.py):
    Builds the cleaned labels' surfaces that the voxel tetmeshes are trimmed
    to (``extract_label_surfaces``).
- ProcessVolumetricMesh (process_volumetric_mesh.py):
    Builds the reference tetmeshes (``extract_label_tetrahedra_vtk`` with
    ``trim_tetrahedra_to_surface``, ``extract_label_tetrahedra_netgen``,
    ``extract_label_tetrahedra_simpleware``), removes their boundary slivers,
    and repairs the warped tetmeshes per phase.

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
import numpy as np
import pyvista as pv
from parameters_slicer_heart_4d_ct import SLICER_HEART_4D_CT

from monai_physio import (
    ConvertVTKToUSD,
    MONAIPhysioBase,
    ProcessLabelmaps,
    ProcessSurfaceMesh,
    ProcessTests,
    ProcessTransforms,
    ProcessUSDAnatomy,
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

    labelmap_tools = ProcessLabelmaps(log_level=log_level)
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

    # Every heart and contrast label (chambers, myocardium, blood pools, ...)
    # is meshed as its own structure, so the tetmeshes fill the chambers as
    # well as the walls and each keeps its anatomy in the USD.
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
    )
    if tetmesh_name not in heart_result["surfaces"]:
        raise ValueError(
            f"No {tetmesh_name} surface: the segmented {tetmesh_anatomy_groups} "
            "structures are missing."
        )
    reference_surface = heart_result["surfaces"][tetmesh_name]
    labelmap = heart_result["labelmap"]
    labels = itk.GetArrayFromImage(labelmap)
    labels[~np.isin(labels, tetmesh_label_ids)] = 0
    # Stray voxels and enclosed cavities would become extra boundary shells,
    # which netgen rejects where they crowd the outer surface.  So each label
    # keeps only its largest piece and fills the cavities it encloses, where
    # no other label already is.
    for label_id in np.unique(labels[labels != 0]).tolist():
        label_mask = itk.GetImageFromArray((labels == label_id).astype(np.uint8))
        label_mask.CopyInformation(labelmap)
        cleaned = (
            itk.GetArrayViewFromImage(
                labelmap_tools.fill_holes(labelmap_tools.remove_islands(label_mask))
            )
            != 0
        )
        labels[(labels == label_id) & ~cleaned] = 0
        labels[cleaned & (labels == 0)] = label_id
    reference_labelmap = itk.GetImageFromArray(labels)
    reference_labelmap.CopyInformation(labelmap)
    label_names = {
        label_id: taxonomy.all_labels()[label_id]
        for label_id in np.unique(labels[labels != 0]).tolist()
    }

    # The same labelmap, meshed three ways, one tetmesh per label.  The voxel
    # meshes' boundaries are staircases, so each is relaxed onto its label's
    # surface; netgen and Simpleware produce smooth boundaries themselves.
    element_size_mm = SLICER_HEART_4D_CT.mesh_element_size_mm
    trim_surfaces = ProcessSurfaceMesh(log_level=log_level).extract_label_surfaces(
        reference_labelmap
    )
    raw_tetmeshes: dict[str, dict[int, pv.UnstructuredGrid]] = {
        "vtk": {
            label_id: volume_tools.trim_tetrahedra_to_surface(
                tetmesh, trim_surfaces[label_id]
            )
            for label_id, tetmesh in volume_tools.extract_label_tetrahedra_vtk(
                reference_labelmap, element_size_mm
            ).items()
            if label_id in trim_surfaces
        },
        # The heart wall is thin between the contrast and the background.  The
        # default blur plus 30 Taubin iterations pinches it until a surface
        # touches itself, which netgen rejects; no Taubin smoothing leaves
        # rough surfaces whose sliver elements the warp inverts beyond repair.
        # Light Taubin smoothing without the blur avoids both.  Contouring on
        # a 1 mm grid keeps the voxel steps small, and netgen re-tiles each
        # label's surface to the element size, so the labels' meshes are
        # independent rather than sharing their interface nodes.
        "netgen": volume_tools.extract_label_tetrahedra_netgen(
            reference_labelmap,
            element_size_mm,
            smoothing_sigma_mm=0.0,
            smoothing_iterations=10,
            shared_interfaces=False,
            contour_spacing_mm=1.0,
        ),
    }
    if Path(volume_tools.simpleware_exe_path).exists():
        raw_tetmeshes["simpleware"] = volume_tools.extract_label_tetrahedra_simpleware(
            reference_labelmap, element_size_mm
        )
    else:
        reporter.log_info(
            "Simpleware Medical not found at %s; skipping its tetmesh.",
            volume_tools.simpleware_exe_path,
        )

    # Flat boundary caps invert under modest compression and cannot be
    # repaired after the warp, so they are dropped from each label's mesh
    # first.  Each mesher's label meshes are then merged into one grid, its
    # cells tagged by label, which is warped as a whole and split back into
    # one prim per label in the USD.
    reference_tetmeshes: dict[str, pv.UnstructuredGrid] = {}
    for mesher, label_tetmeshes in raw_tetmeshes.items():
        tagged = []
        for label_id, tetmesh in label_tetmeshes.items():
            tetmesh = volume_tools.remove_boundary_slivers(tetmesh)
            tetmesh.cell_data["SegmentationLabelIds"] = np.full(
                tetmesh.n_cells, label_id, dtype=np.int32
            )
            tagged.append(tetmesh)
        if not tagged:
            raise ValueError(
                f"No {mesher} tetrahedra: the segmented {tetmesh_anatomy_groups} "
                "structures are thinner than one element."
            )
        reference_tetmeshes[mesher] = cast(
            pv.UnstructuredGrid, pv.merge(tagged, merge_points=False)
        )
        reporter.log_info(
            "Reference %s tetmeshes: %d labels, %d points, %d cells",
            mesher,
            len(tagged),
            reference_tetmeshes[mesher].n_points,
            reference_tetmeshes[mesher].n_cells,
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

    tetmesh_files: dict[str, list[Path]] = {}
    usd_files: dict[str, Path] = {}
    for mesher, reference_tetmesh in reference_tetmeshes.items():
        tetmesh_files[mesher] = []
        phase_tetmeshes = []
        for i, fixed_to_moving_transform in enumerate(fixed_to_moving_transforms):
            phase_tetmesh = volume_tools.repair_inverted_tetrahedra(
                transform_tools.transform_pvcontour(
                    reference_tetmesh, fixed_to_moving_transform
                )
            )
            tetmesh_file = output_dir / f"heart_tetmesh_{mesher}_{i:03d}.vtu"
            phase_tetmesh.save(tetmesh_file)
            tetmesh_files[mesher].append(tetmesh_file)
            phase_tetmeshes.append(phase_tetmesh)

        # Animated USD: every phase shares the reference connectivity, so the
        # stage holds one UsdGeomTetMesh per label, grouped by anatomy, whose
        # points are sampled per phase and whose material follows its anatomy.
        usd_file = output_dir / f"heart_tetmesh_{mesher}.usd"
        stage = ConvertVTKToUSD(
            f"heart_tetmesh_{mesher}",
            phase_tetmeshes,
            mask_ids=label_names,
            segmenter=segmentation_method,
            frames_per_second=float(len(phase_tetmeshes)),
            output_tet_mesh=True,
            log_level=log_level,
        ).convert(str(usd_file))
        ProcessUSDAnatomy(stage, log_level=log_level).enhance_meshes(
            segmentation_method
        )
        stage.Save()
        usd_files[mesher] = usd_file
        reporter.log_info("USD (%s): %s", mesher, usd_file)

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
        "tetmesh_files": {
            mesher: [str(path) for path in paths]
            for mesher, paths in tetmesh_files.items()
        },
        "usd_files": {mesher: str(path) for mesher, path in usd_files.items()},
        "screenshots": screenshots,
    }
