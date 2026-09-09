"""
Tutorial 4 (Duke Heart): 4D Labelmaps to VTK Surfaces and Tetrahedral Meshes

Purpose
-------
Turn the Duke-Heart-4DLabelmaps labelmaps into VTK geometry.  ``outputs``
below chooses how much of it to build, because the full pass costs about a
hundred times the short one:

``"shape_model"``
    Each case's reference frame only (``*_ref_labelmap.nii.gz``), and only the
    whole heart, as ``<frame_stem>_heart_minus_interior_chambers.vtp`` and its
    ``.vtu`` tetrahedral mesh.  Tutorial 6 (Duke Heart) reads the surfaces to
    build its shape model.

``"full"``
    Every gated frame and every label, each with its own surface and mesh.
    Tutorial 5 (Duke Heart) needs this, since it animates the frames.

Files written per frame, the per-label ones only under ``"full"``:

- ``<frame_stem>_surfaces.vtp`` -- every label's watertight, outward-oriented
  surface in one file, via ``ContourTools.extract_label_surfaces`` on the whole
  labelmap.  Extracting the labels together is what keeps neighbors touching:
  a wall between two of them is contoured from the same field on the same
  isotropic grid, so both surfaces carry the same vertices there.  Per-cell
  ``SegmentationLabelIds`` says which label each triangle came from.
- ``<frame_stem>_<name>.vtu`` -- one tetrahedral mesh per structure, six
  ``VTK_TETRA`` per isotropic voxel, via ``ContourTools.extract_tetrahedra``,
  then relaxed onto that structure's surface by
  ``ContourTools.trim_tetrahedra_to_surface``.

The mesh starts as a voxel staircase and ends up bounded by the smooth
surface: the relaxation projects its boundary onto that surface while smoothing
the interior to make room, so no trace of the voxel blocks is left.  What
remains between the two geometries is faceting at the element size, logged per
structure in millimeters.

Both carry the structure's ``USDAnatomyTools`` color, as ``AnatomyColor`` and
as a per-cell ``Color``, so they render the same way as the surfaces
``WorkflowConvertImageToVTK`` writes.

The whole-heart structure drops the labels ``interior_object_ids`` names in
``parameters_duke_heart_labelmaps.py`` -- the chamber cavities, and the vessels
whose extent varies too much between patients -- which is the same definition of
"heart" that Tutorials 2, 6 and 7 use.

Label names come from ``SegmentHeartSimplewareTrimmedBranches``'s taxonomy,
the segmenter that produced these labelmaps.  Labels with no taxonomy entry are
named ``label_<id>``.

Data Required
-------------
``data/Duke-Heart-4DLabelmaps/pm????/*_labelmap.nii.gz`` (multi-label)
"""

# Imports
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import itk
import numpy as np
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART

from monai_physio import (
    ContourTools,
    MONAIPhysioBase,
    SegmentHeartSimplewareTrimmedBranches,
    TestTools,
)

# Only run if this script is not imported as a module
if __name__ == "__main__":
    # Data directory specification

    class_name = "tutorial_04_duke_heart_labelmap_to_vtk"

    test_mode = TestTools.running_as_test()

    output_dir = (
        DUKE_HEART.output_directory(test_mode) / "tutorial_04_duke_heart_labelmap"
    )

    # How much to build; see this module's docstring.  "shape_model" is what
    # Tutorial 6 (Duke Heart) consumes and runs in minutes over the cohort;
    # "full" also feeds Tutorial 5 (Duke Heart) and takes hours, because it
    # contours and meshes every label of all 348 gated frames rather than the
    # whole heart of the 29 reference ones.
    outputs = "shape_model"
    if outputs not in ("shape_model", "full"):
        raise ValueError(f"outputs must be 'shape_model' or 'full'; got {outputs!r}")
    reference_frames_only = outputs == "shape_model"
    write_per_label_surfaces = outputs == "full"

    # Taubin smoothing iterations applied to the extracted surfaces (0 disables).
    smoothing_iterations = DUKE_HEART.surface_smoothing_iterations

    # Surfaces are contoured on an isotropic grid of this pitch, which also sets
    # the triangle count in place of decimation: decimating a label would move
    # its vertices off the neighbor it shares them with.
    surface_spacing_mm = DUKE_HEART.surface_spacing_mm

    # Element size of the tetrahedral meshes.  Isotropic, so the elements do
    # not inherit the slice pitch, and small enough that relaxing them onto the
    # surface leaves no trace of the voxel staircase.
    mesh_element_size_mm = DUKE_HEART.mesh_element_size_mm

    # Labels left out of the whole-heart structure; see the parameters module.
    interior_object_ids = DUKE_HEART.interior_object_ids
    whole_heart_name = "heart_minus_interior_chambers"

    # Set False to keep outputs a previous run already wrote.
    overwrite = True

    if test_mode:
        data_dir = DUKE_HEART.data_directory(test_mode) / "Duke-Heart-4DLabelmaps"
    else:
        data_dir = DUKE_HEART.data_directory(test_mode) / "Duke-Heart-4DLabelmaps"

    log_level = logging.INFO
    reporter = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    contour_tools = ContourTools(log_level=log_level)
    taxonomy = SegmentHeartSimplewareTrimmedBranches(log_level=logging.WARNING).taxonomy
    label_names = taxonomy.all_labels()

    def mask_from(labelmap_image: itk.Image, keep: np.ndarray) -> itk.Image:
        """Return *keep* as a binary mask carrying *labelmap_image*'s geometry."""
        mask = itk.GetImageFromArray(keep.astype(np.uint8))
        mask.CopyInformation(labelmap_image)
        return mask

    def annotate(
        mesh: pv.DataSet, label_ids: list[int], name: str, source: str
    ) -> None:
        """Record the originating labels and file on *mesh* in-place.

        ``SegmentationLabelIds`` holds one id for a per-label structure and
        every retained id for the whole-heart one; a single id is what
        ``ContourTools.save_combined_surfaces`` needs to tag the merged file's
        cells with the structure they came from.  The anatomy color is attached
        separately, by ``ContourTools.apply_anatomy_color``.
        """
        mesh.field_data["SegmentationLabelIds"] = np.asarray(label_ids, dtype=np.int32)
        mesh.field_data["LabelName"] = np.array([name])
        mesh.field_data["SourceLabelmap"] = np.array([source])

    def surface_to_mesh_displacement(
        surface: pv.PolyData, mesh: pv.UnstructuredGrid
    ) -> float:
        """Return the mean distance, in mm, from *surface*'s points to *mesh*."""
        boundary = mesh.extract_surface(algorithm="dataset_surface")
        # On a copy, so the distances are not written into the saved surface.
        distances = surface.copy().compute_implicit_distance(boundary)
        return float(np.abs(distances["implicit_distance"]).mean())

    def write_tetrahedra(
        mask: itk.Image,
        surface: pv.PolyData,
        case_output_dir: Path,
        stem: str,
        name: str,
        label_ids: list[int],
        source: str,
    ) -> Optional[float]:
        """Write one structure's VTU mesh, relaxed onto its *surface*.

        Returns:
            The mean displacement, in millimeters, between the two geometries,
            or ``None`` for a structure too thin to hold an element.
        """
        # Resampling to the element size eats a structure thinner than it --
        # the coronary arteries here are a voxel or two across -- so the size
        # is halved until the mesh accounts for most of the surface's volume,
        # or until it is as fine as the labelmap itself.
        finest_spacing = float(np.min(np.asarray(mask.GetSpacing())))
        element_size = mesh_element_size_mm
        while True:
            # USDAnatomyTools has no override for names like "left_ventricle",
            # so the structure's anatomy group is offered as the fallback color.
            tet_mesh = contour_tools.extract_tetrahedra(
                mask,
                element_size_mm=element_size,
                anatomy_names=[name, taxonomy.group_for_label(name)],
            )
            volume = float(np.sum(tet_mesh.compute_cell_sizes(volume=True)["Volume"]))
            if volume > 0.5 * surface.volume or element_size <= finest_spacing:
                break
            element_size *= 0.5
            reporter.log_debug(
                "  %s: %.3g mm elements hold %.0f%% of it; halving",
                name,
                element_size * 2.0,
                100.0 * volume / float(surface.volume),
            )
        if tet_mesh.n_cells == 0:
            reporter.log_warning("  %-24s thinner than one element; no mesh", name)
            return None
        tet_mesh = contour_tools.trim_tetrahedra_to_surface(tet_mesh, surface)
        annotate(tet_mesh, label_ids, name, source)
        tet_mesh.save(case_output_dir / f"{stem}_{name}.vtu")

        displacement = surface_to_mesh_displacement(surface, tet_mesh)
        reporter.log_info(
            "  %-24s %7d triangles  %8d tetrahedra  %6.3f mm",
            name,
            surface.n_cells,
            tet_mesh.n_cells,
            displacement,
        )
        return displacement

    # Cohort discovery
    case_dirs = sorted(
        path for path in data_dir.glob("pm[0-9][0-9][0-9][0-9]") if path.is_dir()
    )
    if not case_dirs:
        raise FileNotFoundError(
            f"No pm???? case directories found under {data_dir}.\n"
            "See data/README.md for download instructions."
        )

    # Extraction
    labelmap_pattern = (
        "*_ref_labelmap.nii.gz" if reference_frames_only else "*_labelmap.nii.gz"
    )
    displacements: list[float] = []
    surface_count = 0
    whole_heart_surface: Optional[pv.PolyData] = None

    # Every case writes into this one directory: the frame stems already start
    # with their case id, so the names stay unique and the readers downstream
    # group on that prefix rather than on a directory.
    case_output_dir = output_dir
    case_output_dir.mkdir(parents=True, exist_ok=True)

    for case_dir in case_dirs:
        labelmap_files = sorted(case_dir.glob(labelmap_pattern))
        reporter.log_section(f"{case_dir.name}: {len(labelmap_files)} labelmaps")

        for labelmap_file in labelmap_files:
            stem = labelmap_file.name[: -len("_labelmap.nii.gz")]
            surfaces_file = case_output_dir / f"{stem}_surfaces.vtp"
            whole_heart_file = case_output_dir / f"{stem}_{whole_heart_name}.vtp"
            # The whole-heart surface is written last, so its presence means
            # the frame finished rather than stopped part way.
            if not overwrite and whole_heart_file.exists():
                reporter.log_debug("%s: outputs exist, skipping", labelmap_file.name)
                continue

            reporter.log_info("%s", labelmap_file.name)
            labelmap_image = itk.imread(str(labelmap_file))
            labels = itk.GetArrayViewFromImage(labelmap_image)
            present_ids = [int(value) for value in np.unique(labels) if value != 0]

            # The whole heart, hollowed of the chamber cavities.  It overlaps
            # the per-label surfaces, so it is kept out of their file.
            whole_heart_ids = [
                label_id
                for label_id in present_ids
                if label_id not in interior_object_ids
            ]
            whole_heart_mask = mask_from(
                labelmap_image, np.isin(labels, whole_heart_ids)
            )
            # The mask carries one label, so its surface is the only entry; an
            # empty mapping means the frame held no heart to contour.
            whole_heart_surfaces = contour_tools.extract_label_surfaces(
                whole_heart_mask,
                isotropic_spacing_mm=surface_spacing_mm,
                smoothing_iterations=smoothing_iterations,
            )
            if 1 not in whole_heart_surfaces:
                reporter.log_warning(
                    "%s: no whole-heart surface; skipping the frame",
                    labelmap_file.name,
                )
                continue
            whole_heart = whole_heart_surfaces[1]
            contour_tools.apply_anatomy_color(whole_heart, [whole_heart_name, "heart"])
            annotate(whole_heart, whole_heart_ids, whole_heart_name, labelmap_file.name)
            surface_count += 1
            displacement = write_tetrahedra(
                whole_heart_mask,
                whole_heart,
                case_output_dir,
                stem,
                whole_heart_name,
                whole_heart_ids,
                labelmap_file.name,
            )
            if displacement is not None:
                displacements.append(displacement)

            if write_per_label_surfaces:
                # Every label at once, so that neighbors share the wall between
                # them rather than each contouring its own copy of it.
                surfaces = contour_tools.extract_label_surfaces(
                    labelmap_image,
                    isotropic_spacing_mm=surface_spacing_mm,
                    smoothing_iterations=smoothing_iterations,
                )
                named_surfaces: dict[str, pv.PolyData] = {}
                for label_id, surface in surfaces.items():
                    name = label_names.get(label_id, f"label_{label_id}")
                    contour_tools.apply_anatomy_color(
                        surface, [name, taxonomy.group_for_label(name)]
                    )
                    annotate(surface, [label_id], name, labelmap_file.name)
                    named_surfaces[name] = surface
                    surface_count += 1
                    displacement = write_tetrahedra(
                        mask_from(labelmap_image, labels == label_id),
                        surface,
                        case_output_dir,
                        stem,
                        name,
                        [label_id],
                        labelmap_file.name,
                    )
                    if displacement is not None:
                        displacements.append(displacement)
                ContourTools.save_combined_surfaces(named_surfaces, str(surfaces_file))

            # Written last, so an interrupted frame is redone rather than
            # skipped by the check above.
            whole_heart.save(whole_heart_file)
            if whole_heart_surface is None:
                whole_heart_surface = whole_heart

    mean_displacement = float(np.mean(displacements)) if displacements else 0.0
    if displacements:
        reporter.log_section(
            f"Wrote {surface_count} surfaces and {len(displacements)} tetrahedral "
            f"meshes, mean surface-to-mesh displacement {mean_displacement:.3f} mm"
        )
    else:
        reporter.log_section(f"Wrote {surface_count} surfaces")

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []
    if whole_heart_surface is not None:
        screenshots.append(
            tt.save_screenshot_mesh(
                whole_heart_surface,
                f"{whole_heart_name}.png",
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
