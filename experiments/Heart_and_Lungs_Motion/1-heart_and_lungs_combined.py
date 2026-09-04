"""Combine respiratory and cardiac motion into a 4D surface sequence and USD.

Purpose
-------
Produce a sequence of combined-motion surfaces for the DirLab ``Case1Pack``
thorax by composing two motions on one static patient surface:

- **Cardiac** motion from the ``temp_heart_and_lungs_beating_heart`` deformation
  fields (one per cardiac stage), which displace the heart region. These fields
  are defined on the reference (``T70``) grid.
- **Respiratory** motion from the ``tutorial_01_lung`` forward transforms
  (one per breath phase), which warp the reference-frame surface to each phase
  of the breathing cycle.

One cardiac cycle (10 stages) is rendered per breath phase, for
``10 phases x 10 stages = 100`` output frames.

Composition order
-----------------
The cardiac deformation is applied **first, at the reference frame** where the
cardiac fields are defined, and the breathing transform is then applied to the
already cardiac-deformed surface::

    p_combined = forward_r( p + cardiac_s(p) )

This carries the heart's cardiac deformation into each respiratory phase (the
breathing deformation is applied to the heart deformation), rather than sampling
the reference-frame cardiac field at breathing-displaced positions where it is
not valid.

Smooth respiratory progression
------------------------------
Rather than holding the breath fixed for a whole cardiac cycle, the respiratory
transform is interpolated from the current breath phase toward the next by the
fraction of the way through the cardiac cycle (``stage / n_stages``). The
interpolation is a per-vertex linear blend of the two phases' warped positions,
equivalent to interpolating the two displacement transforms::

    p_combined = (1 - t) * forward_r(cs) + t * forward_{r+1}(cs),   t = stage / n_stages

with ``cs = p + cardiac_s(p)``. The next phase wraps around (phase 9 -> phase 0)
so the breathing loops smoothly.

Pipeline
--------
1. Smooth each of the 10 cardiac deformation fields
   (``output/temp_beating_heart/deformation_field_s0*.mha``) with a Gaussian of
   ``SMOOTHING_SIGMA_MM`` (10 mm) into a ``DisplacementFieldTransform``. The raw
   fields are a thin surface shell, so smoothing spreads them into a continuous
   deformation (and, as a side effect, attenuates the peak magnitude).
2. Load the combined patient surface
   (``output/tutorial_04_lung/patient_surfaces.vtp``), optionally decimate and
   smooth it once (``SURFACE_DECIMATION_REDUCTION`` /
   ``SURFACE_SMOOTHING_ITERATIONS``; both disabled by default), then
   cardiac-deform it once per stage (reused across all breath phases).
3. Warp each cardiac-deformed surface to every breath phase with the
   ``tutorial_01_lung`` forward transforms ``slice_{r:03d}_all_forward.hdf``.
4. Blend consecutive breath phases per frame and assemble the 100 frames into a
   single animated 4D USD.

Outputs (under ``output/temp_heart_and_lungs_combined``)
-------------------------------------------------------
- ``combined_frame_<iii>.vtp`` for ``iii`` in ``000 .. 099`` - the combined
  respiratory + cardiac surface, ordered breath-phase-major then cardiac.
- ``heart_and_lungs_combined.usd`` - the 100-frame animated 4D USD.
"""

from __future__ import annotations

import logging
from pathlib import Path

import itk
import numpy as np
import pyvista as pv

from monai_physio import ImageTools, TransformTools, WorkflowConvertVTKToUSD

# Gaussian sigma (mm) used to smooth the sparse cardiac deformation fields.
SMOOTHING_SIGMA_MM = 10.0
# USD playback rate; 10 stages/second ~= one heartbeat per second.
FRAMES_PER_SECOND = 10.0
# One-time conditioning of the patient surface, applied before any warping.
# Fraction of triangles to remove (0.0 = no decimation, e.g. 0.5 halves them).
SURFACE_DECIMATION_REDUCTION = 0.0
# Taubin (non-shrinking) smoothing iterations (0 = no smoothing).
SURFACE_SMOOTHING_ITERATIONS = 0

_transform_tools = TransformTools()


def _condition_surface(
    surface: pv.PolyData,
    decimation_reduction: float,
    smoothing_iterations: int,
) -> pv.PolyData:
    """Optionally decimate then smooth the model surface (no-op when disabled).

    Applied once to the patient surface so every warped frame inherits the same
    resolution and smoothing. Decimation uses ``decimate_pro`` on a triangulated
    copy; smoothing uses non-shrinking Taubin smoothing.
    """
    conditioned = surface
    if decimation_reduction > 0.0:
        conditioned = conditioned.triangulate().decimate_pro(decimation_reduction)
    if smoothing_iterations > 0:
        conditioned = conditioned.smooth_taubin(n_iter=smoothing_iterations)
    return conditioned


def _smoothed_cardiac_transform(
    field_file: Path, sigma_mm: float
) -> itk.DisplacementFieldTransform:
    """Load a cardiac deformation field and return a smoothed field transform.

    The ``.mha`` field is a float vector image; it is converted to a
    double-precision vector field, wrapped as a ``DisplacementFieldTransform``,
    and Gaussian-smoothed by ``sigma_mm`` (in physical millimeters).
    """
    field = itk.imread(str(field_file))
    field_double = ImageTools().convert_array_to_image_of_vectors(
        itk.array_from_image(field), reference_image=field, ptype=itk.D
    )
    field_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    field_transform.SetDisplacementField(field_double)
    return _transform_tools.smooth_transform(
        field_transform, sigma=sigma_mm, reference_image=field
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("temp_heart_and_lungs_combined")

    tutorials_dir = Path(__file__).resolve().parent

    beating_heart_dir = tutorials_dir / "output" / "temp_beating_heart"
    respiratory_dir = tutorials_dir / "output" / "tutorial_01_lung"
    repo_tutorials_dir = Path(__file__).resolve().parents[2] / "tutorials"
    patient_surface_file = (
        repo_tutorials_dir / "output" / "tutorial_04_lung" / "patient_surfaces.vtp"
    )
    output_dir = tutorials_dir / "output" / "temp_heart_and_lungs_combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in output_dir.glob("combined_frame_*.vtp"):
        stale_frame.unlink()

    cardiac_field_files = sorted(beating_heart_dir.glob("deformation_field_s*.mha"))
    forward_transform_files = sorted(respiratory_dir.glob("slice_*_all_forward.hdf"))

    if not cardiac_field_files:
        raise FileNotFoundError(
            f"No cardiac deformation fields found in {beating_heart_dir}. "
            "Run temp_heart_and_lungs_beating_heart.py first."
        )
    if not forward_transform_files:
        raise FileNotFoundError(
            f"No respiratory forward transforms found in {respiratory_dir}. "
            "Run tutorial_01_lung_gated_ct_to_usd.py first."
        )
    if not patient_surface_file.exists():
        raise FileNotFoundError(
            f"Patient surface not found: {patient_surface_file}. "
            "Run tutorial_04_lung_ct_to_vtk.py first."
        )

    n_phases = len(forward_transform_files)
    n_stages = len(cardiac_field_files)
    logger.info(
        "Respiratory phases: %d, cardiac stages: %d (1 cardiac cycle/phase)",
        n_phases,
        n_stages,
    )

    # Smooth every cardiac field once; reused across all breath phases.
    cardiac_transforms = [
        _smoothed_cardiac_transform(field_file, SMOOTHING_SIGMA_MM)
        for field_file in cardiac_field_files
    ]
    logger.info(
        "Smoothed %d cardiac deformation fields by %.1f mm",
        len(cardiac_transforms),
        SMOOTHING_SIGMA_MM,
    )

    patient_surface = pv.read(str(patient_surface_file))
    # Condition the model once (decimate/smooth) so every frame inherits it.
    patient_surface = _condition_surface(
        patient_surface, SURFACE_DECIMATION_REDUCTION, SURFACE_SMOOTHING_ITERATIONS
    )
    logger.info(
        "Patient surface: %d points (decimation=%.2f, smoothing_iters=%d)",
        patient_surface.n_points,
        SURFACE_DECIMATION_REDUCTION,
        SURFACE_SMOOTHING_ITERATIONS,
    )

    # Cardiac motion is applied at the reference frame (where the fields are
    # defined), once per stage, then carried into each breath phase below.
    cardiac_surfaces = [
        _transform_tools.transform_pvcontour(patient_surface, cardiac_transform)
        for cardiac_transform in cardiac_transforms
    ]

    # Respiratory-warped vertex positions for every (phase, stage):
    # resp_points[phase][stage] = forward_phase(cardiac_surface[stage]).points.
    resp_points: list[list[np.ndarray]] = []
    for phase_idx, forward_file in enumerate(forward_transform_files):
        forward_transform = itk.transformread(str(forward_file))
        resp_points.append(
            [
                np.asarray(
                    _transform_tools.transform_pvcontour(
                        cardiac_surface, forward_transform
                    ).points,
                    dtype=np.float32,
                )
                for cardiac_surface in cardiac_surfaces
            ]
        )
        logger.info("Respiratory warp phase %d/%d done", phase_idx + 1, n_phases)

    # Blend the current and next breath phase by the fraction through the cardiac
    # cycle, so breathing advances smoothly across the heartbeat (next wraps).
    combined_files: list[Path] = []
    usd_frames: list[pv.PolyData] = []
    for phase_idx in range(n_phases):
        next_phase_idx = (phase_idx + 1) % n_phases
        for stage_idx in range(n_stages):
            blend = stage_idx / n_stages
            points = (1.0 - blend) * resp_points[phase_idx][stage_idx] + (
                blend * resp_points[next_phase_idx][stage_idx]
            )
            combined_surface = patient_surface.copy(deep=True)
            combined_surface.points = points

            frame_idx = phase_idx * n_stages + stage_idx
            frame_file = output_dir / f"combined_frame_{frame_idx:03d}.vtp"
            combined_surface.save(str(frame_file))
            combined_files.append(frame_file)
            usd_frames.append(combined_surface)

    del resp_points
    logger.info(
        "Wrote %d combined-motion surfaces to %s", len(combined_files), output_dir
    )

    # Assemble the ordered frames into a single animated 4D USD.
    usd_workflow = WorkflowConvertVTKToUSD(
        input_meshes=usd_frames,
        usd_project_name="heart_and_lungs_combined",
        output_directory=output_dir,
        appearance="solid",
        solid_color=(0.82, 0.70, 0.66),
        separate_by_connectivity=False,
        frames_per_second=FRAMES_PER_SECOND,
    )
    usd_result = usd_workflow.process()
    logger.info("Wrote 4D USD: %s", usd_result["usd_file"])

    tutorial_results = {
        "combined_surfaces": combined_files,
        "usd_file": usd_result["usd_file"],
    }
