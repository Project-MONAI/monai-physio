"""
Tutorial 4 (Duke Heart): 4D Labelmaps to VTK Surfaces and Tetrahedral Meshes

Purpose
-------
Turn the Duke-Heart-4DLabelmaps labelmaps into VTK geometry with
``WorkflowConvertImageToVTK``, which takes these labelmaps in place of a CT
image.  ``outputs`` below chooses how much of it to build, because the full
pass costs about a hundred times the short one:

``"shape_model"``
    Each case's reference frame only (``*_ref_labelmap.nii.gz``), and only the
    whole heart, as ``<frame_stem>_heart_minus_interior_chambers.vtp`` and its
    ``.vtu`` tetrahedral mesh.  Tutorial 6 (Duke Heart) reads the surfaces to
    build its shape model.

``"full"``
    Every gated frame and every label, each with its own surface and mesh.
    Tutorial 5 (Duke Heart) needs this, since it animates the frames.

Files written per frame, the per-label ones only under ``"full"``:

- ``<frame_stem>_heart_minus_interior_chambers.vtp`` / ``.vtu`` -- the whole
  heart, without the labels ``interior_object_ids`` names in
  ``parameters_duke_heart_labelmaps.py``: the chamber cavities, and the vessels
  whose extent varies too much between patients.  This is the same "heart"
  that Tutorials 2, 6 and 7 use.
- ``<frame_stem>_surfaces.vtp`` -- every label's watertight surface in one
  file, per-cell ``SegmentationLabelIds`` saying which label each triangle came
  from.  The labels are extracted together, so neighbors share the wall
  between them vertex for vertex.
- ``<frame_stem>_<name>.vtu`` -- one tetrahedral mesh per label, relaxed onto
  that label's surface.

Every output carries its structure's ``ProcessUSDAnatomy`` color, as
``AnatomyColor`` and as a per-cell ``Color``.  What remains between a surface
and its mesh is faceting at the element size, logged per structure in
millimeters.

Label names come from ``SegmentHeartSimplewareTrimmedBranches``'s taxonomy,
the segmenter that produced these labelmaps.  Labels with no taxonomy entry are
named ``label_<id>``.

Data Required
-------------
``data/Duke-Heart-4DLabelmaps/pm????/*_labelmap.nii.gz`` (multi-label)
"""

# Imports
from __future__ import annotations

import functools
import importlib
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import itk
import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from monai_physio import WorkflowConvertImageToVTK

CLASS_NAME = "tutorial_04_duke_heart_labelmap_to_vtk"

# Frames run in at most this many worker processes, each limited to a couple of
# ITK threads: the ITK filters stop scaling past a few threads, so several
# frames at a time use the machine better than one frame on all of it.  A large
# frame (439x512x512) peaks near 3.5 GB resident and 9.6 GB committed, so the
# count is also capped by the memory free at start, at WORKER_COMMIT_BYTES each.
MAX_WORKERS = 12
WORKER_ITK_THREADS = 2
WORKER_COMMIT_BYTES = 12 * 2**30


@functools.cache
def _workflow(log_level: int) -> WorkflowConvertImageToVTK:
    """Build this process's workflow once; a worker reuses it for every frame."""
    from parameters_duke_heart_labelmaps import DUKE_HEART

    from monai_physio import WorkflowConvertImageToVTK

    # The segmenter only names the labels: these labelmaps are its output.
    return WorkflowConvertImageToVTK(
        DUKE_HEART.segmenter(False, log_level=logging.WARNING), log_level=log_level
    )


def surface_to_mesh_displacement(
    surface: pv.PolyData, mesh: pv.UnstructuredGrid
) -> float:
    """Return the mean distance, in mm, from *surface*'s points to *mesh*."""
    from monai_physio import ProcessMeshKernels

    boundary = mesh.extract_surface(algorithm="dataset_surface")
    if boundary.n_points == 0:
        return 0.0
    points = np.asarray(surface.points, dtype=np.float64)
    kernels = ProcessMeshKernels()
    closest = kernels.closest_points_on_surface(
        kernels.build_surface_mesh(boundary), points
    )
    return float(np.linalg.norm(closest - points, axis=1).mean())


def frame_stem(labelmap_file: Path) -> str:
    """Return the name *labelmap_file*'s outputs start with."""
    return labelmap_file.name[: -len("_labelmap.nii.gz")]


def convert_frame(
    labelmap_file: Path, output_dir: Path, full: bool, log_level: int
) -> list[float]:
    """Write one labelmap's surfaces and meshes; the worker's entry point.

    Returns:
        The mean surface-to-mesh displacement of every mesh written, in mm.
    """
    from parameters_duke_heart_labelmaps import DUKE_HEART

    from monai_physio import WorkflowConvertImageToVTK

    workflow = _workflow(log_level)
    whole_heart = DUKE_HEART.whole_heart_name
    stem = frame_stem(labelmap_file)

    labelmap = itk.imread(str(labelmap_file))
    result = workflow.process(
        labelmap_image=labelmap,
        groups=DUKE_HEART.surface_groups(labelmap),
        extract_label_surfaces=full,
        isotropic_spacing_mm=DUKE_HEART.surface_spacing_mm,
        smoothing_iterations=DUKE_HEART.surface_smoothing_iterations,
        tetrahedra_element_size_mm=DUKE_HEART.mesh_element_size_mm,
    )
    if whole_heart not in result["surfaces"]:
        workflow.log_warning("%s: no whole-heart surface; skipping", labelmap_file.name)
        return []

    if full:
        WorkflowConvertImageToVTK.save_combined_surfaces(
            result["label_surfaces"], str(output_dir / f"{stem}_surfaces.vtp")
        )
    meshes = {**result["tetrahedra"], **result["label_tetrahedra"]}
    WorkflowConvertImageToVTK.save_surfaces(meshes, str(output_dir), prefix=stem)

    surfaces = {**result["surfaces"], **result["label_surfaces"]}
    displacements = []
    for name, mesh in meshes.items():
        displacements.append(surface_to_mesh_displacement(surfaces[name], mesh))
        workflow.log_info(
            "  %-24s %7d triangles  %8d tetrahedra  %6.3f mm",
            name,
            surfaces[name].n_cells,
            mesh.n_cells,
            displacements[-1],
        )

    # Written last, so an interrupted frame is redone rather than skipped by
    # the overwrite check.
    WorkflowConvertImageToVTK.save_surfaces(
        {whole_heart: result["surfaces"][whole_heart]}, str(output_dir), prefix=stem
    )
    return displacements


def available_memory_bytes() -> int:
    """Return the memory a new process can still commit, in bytes."""
    if sys.platform != "win32":
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_uint32),
            ("memory_load", ctypes.c_uint32),
            ("total_phys", ctypes.c_uint64),
            ("avail_phys", ctypes.c_uint64),
            ("total_commit", ctypes.c_uint64),
            ("avail_commit", ctypes.c_uint64),
            ("total_virtual", ctypes.c_uint64),
            ("avail_virtual", ctypes.c_uint64),
            ("avail_extended_virtual", ctypes.c_uint64),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return int(min(status.avail_commit, status.avail_phys))


def worker_count(available_bytes: int) -> int:
    """Return how many workers *available_bytes* holds at a frame's peak."""
    return max(1, min(MAX_WORKERS, available_bytes // WORKER_COMMIT_BYTES))


def _init_worker() -> None:
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(WORKER_ITK_THREADS)


def _warm_up() -> None:
    """Import what a frame needs and open the GPU, before any frame arrives."""
    import torch
    import warp as wp

    importlib.import_module("monai_physio")
    wp.init()
    torch.zeros(1, device="cuda:0")


def start_workers(count: int) -> ProcessPoolExecutor:
    """Spawn *count* worker processes and have each warm itself up.

    Workers are spawned and the parent never touches the GPU, so no CUDA
    context exists before they start.  Importing ``monai_physio`` takes longer
    than a frame, so the caller starts them before it imports it itself and the
    two imports overlap.
    """
    pool = ProcessPoolExecutor(
        max_workers=count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
    )
    for _ in range(count):
        pool.submit(_warm_up)
    return pool


# Only run if this script is not imported as a module
if __name__ == "__main__":
    workers = start_workers(worker_count(available_memory_bytes()))

    from parameters_duke_heart_labelmaps import DUKE_HEART

    from monai_physio import MONAIPhysioBase, ProcessTests

    # Data directory specification

    class_name = CLASS_NAME

    test_mode = ProcessTests.running_as_test()

    output_dir = DUKE_HEART.input_directory(test_mode)
    data_dir = DUKE_HEART.hold_out_directory(test_mode)

    # How much to build; see this module's docstring.  "shape_model" is what
    # Tutorial 6 (Duke Heart) consumes and runs in minutes over the cohort;
    # "full" also feeds Tutorial 5 (Duke Heart) and takes much longer, because
    # it contours and meshes every label of all 348 gated frames rather than
    # the whole heart of the 29 reference ones.  Frames run in parallel worker
    # processes, so the time shrinks with the cores and memory available.
    outputs = "full"
    if outputs not in ("shape_model", "full"):
        raise ValueError(f"outputs must be 'shape_model' or 'full'; got {outputs!r}")
    full = outputs == "full"

    # Set False to keep outputs a previous run already wrote.
    overwrite = True

    log_level = logging.INFO
    reporter = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    # Cohort discovery
    case_dirs = sorted(
        path for path in data_dir.glob("pm[0-9][0-9][0-9][0-9]") if path.is_dir()
    )
    if not case_dirs:
        raise FileNotFoundError(
            f"No pm???? case directories found under {data_dir}.\n"
            "See data/README.md for download instructions."
        )

    # Every case writes into this one directory: the frame stems already start
    # with their case id, so the names stay unique and the readers downstream
    # group on that prefix rather than on a directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    def whole_heart_file(labelmap_file: Path) -> Path:
        return (
            output_dir
            / f"{frame_stem(labelmap_file)}_{DUKE_HEART.whole_heart_name}.vtp"
        )

    labelmap_pattern = "*_labelmap.nii.gz" if full else "*_ref_labelmap.nii.gz"
    labelmap_files = [
        labelmap_file
        for case_dir in case_dirs
        for labelmap_file in sorted(case_dir.glob(labelmap_pattern))
    ]
    pending_files = [
        labelmap_file
        for labelmap_file in labelmap_files
        if overwrite or not whole_heart_file(labelmap_file).exists()
    ]
    reporter.log_section(
        f"{len(pending_files)} of {len(labelmap_files)} labelmaps to convert"
    )

    # Extraction
    with workers:
        frame_displacements = list(
            workers.map(
                functools.partial(
                    convert_frame, output_dir=output_dir, full=full, log_level=log_level
                ),
                pending_files,
            )
        )
    displacements = [value for frame in frame_displacements for value in frame]
    mean_displacement = float(np.mean(displacements)) if displacements else 0.0
    reporter.log_section(
        f"Wrote {len(displacements)} tetrahedral meshes, mean surface-to-mesh "
        f"displacement {mean_displacement:.3f} mm"
    )

    # Testing
    tt = ProcessTests(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    if labelmap_files and whole_heart_file(labelmap_files[0]).exists():
        screenshots.append(
            tt.save_screenshot_mesh(
                pv.read(str(whole_heart_file(labelmap_files[0]))),
                f"{DUKE_HEART.whole_heart_name}.png",
                camera_position="iso",
                color="lightblue",
                opacity=0.85,
            )
        )

    tutorial_results = {
        "output_dir": output_dir,
        "case_dirs": case_dirs,
        "n_pairs": len(displacements),
        "mean_surface_to_mesh_displacement_mm": mean_displacement,
        "screenshots": screenshots,
    }
