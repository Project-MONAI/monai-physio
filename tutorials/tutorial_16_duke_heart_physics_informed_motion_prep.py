"""
Tutorial 16 (Duke Heart, PINN): Build a Volumetric Shape Model of the Myocardium

Purpose
-------
Prepare everything the physics-informed motion surrogate of Tutorial 17 trains
on.  That surrogate prices its predictions against a neo-Hookean strain energy
as well as against measured displacement, and a strain energy needs a
*deformation gradient*, which needs volume elements whose nodes are the values
the network predicts.  The shape model Tutorials 6 to 8 build is a surface: it
has no interior, so no deformation gradient can be formed on it.

This tutorial therefore builds the same model volumetrically:

1. Build the unbiased mean surface of the population, exactly as Tutorial 6
   does.
2. Fill that surface with tetrahedra
   (``ProcessVolumetricMesh.extract_tetrahedra`` then ``trim_tetrahedra_to_surface``).
   Those cells become the elements the strain energy is summed over, and
   because every subject and phase inherits the template's topology, one set of
   element node ids stays valid across the whole cohort.
3. Decompose the population into shape modes against that tetrahedral template
   (``WorkflowCreateStatisticalModel`` with ``solve_for_surface_pca=False``).
   Only the *reference* mesh has to be volumetric: correspondence is
   established by warping it through each subject's displacement field, which
   carries interior nodes along, while the samples merely supply the distance
   maps that drive the registration.
4. Fit that model to every case and propagate it through the cardiac phases,
   writing one ``.vtu`` per frame.
5. Write one training manifest per case, naming each frame's displacement from
   the case's own fitted reference.  The physics residual is measured against
   that fitted reference rather than against the population mean, since the
   targets are defined at its points -- measuring against the mean would charge
   every subject a strain energy for merely being shaped unlike the mean.

Nothing here reads or writes anything belonging to Tutorials 1 to 15 except
Tutorial 4's surfaces and Tutorial 2's ICON weights, both read-only.

Hardware Required
-----------------
A CUDA GPU is required.  Every registration below runs on one, and a
CPU-only run is not a supported configuration.

Data Required
-------------
Surfaces: Tutorial 4 (Duke Heart) output
(``output/tutorial_04_duke_heart_labelmap/*_ref_heart_minus_interior_chambers.vtp``)
Labelmaps: ``data/Duke-Heart-4DLabelmaps/pm????/*_labelmap.nii.gz``
ICON weights: Tutorial 2 (Duke Heart) output, optional -- the stock uniGradICON
weights are used when absent, which understates the population's variance.

Outputs (under ``output/tutorial_16_duke_heart_physics_informed_motion/``)
-------------------------------------------------------------------------
  * ``ssm_template.vtu``                  - the tetrahedral template
  * ``pca_model.json``, ``pca_mean.vtu``  - the volumetric shape model
  * ``pca_mean_surface.vtp``              - its boundary, for display
  * ``<case>/<case>_ssm_model.vtu``       - fitted reference-frame model
  * ``<case>/<frame>_ssm_model.vtu``      - model warped to that gated frame
  * ``manifests/<case>_manifest.json``    - what Tutorial 17 trains from

Cost
----
The most expensive tutorial in this chain: an atlas pass over the population,
one deformable registration per case for the model, then one fit plus one
registration per gated frame per case.  The tetrahedral template is far denser
than the surface one, so the fits are correspondingly slower.  Every stage is
cached on disk, so a re-run only redoes what is missing.
"""

# Imports
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv
from parameters_duke_heart_physics_informed import DUKE_HEART_PHYSICS_INFORMED

from monai_physio import (
    EvaluateMovementDukeHeart,
    MONAIPhysioBase,
    ProcessPhysicsNemo,
    ProcessSurfaceMesh,
    ProcessTests,
    ProcessVolumetricMesh,
    WorkflowCreateMeanSurface,
    WorkflowCreateStatisticalModel,
    WorkflowFitStatisticalModelToPatient,
)

LABELMAP_SUFFIX = "_labelmap.nii.gz"

# Gated frames carry a ``g{PPP}`` tag naming their percentage of the R-R
# interval; this is what a per-frame model is matched and staged by.
PHASE_MODEL_PATTERN = "*_g[0-9][0-9][0-9]_*_ssm_model.vtu"


def _write_case_manifest(
    case_dir: Path,
    manifests_dir: Path,
    logger: MONAIPhysioBase,
    evaluator: EvaluateMovementDukeHeart,
) -> Optional[Path]:
    """Write a per-case manifest JSON; return its path (or None if incomplete).

    A case needs a fitted reference model, a PCA coefficient file and at least
    two gated frames.  One missing any of them is skipped with the reason
    logged, so a half-finished run is distinguishable from one that never ran.
    Each frame's target is its per-node displacement from the case's own
    fitted reference model, written as a volume so the network's outputs land
    on the nodes the physics elements are indexed against; *evaluator* stages
    it by its ``g{PPP}`` tag.
    """
    case_id = case_dir.name
    fitted_reference_model_file = case_dir / f"{case_id}_ssm_model.vtu"
    pca_file = case_dir / f"{case_id}_ssm_pca_coefficients.json"
    phase_files = sorted(case_dir.glob(PHASE_MODEL_PATTERN))
    missing = []
    if not fitted_reference_model_file.exists():
        missing.append(f"reference model {fitted_reference_model_file.name}")
    if not pca_file.exists():
        missing.append(f"PCA coefficients {pca_file.name}")
    if len(phase_files) < 2:
        missing.append(f"at least 2 frame models (found {len(phase_files)})")
    if missing:
        logger.log_warning("Skipping %s: missing %s", case_id, "; ".join(missing))
        return None

    manifest_path: Path = ProcessPhysicsNemo.write_manifest(
        subject_id=case_id,
        fitted_reference_mesh=fitted_reference_model_file,
        pca_coefficients=pca_file,
        phase_meshes=phase_files,
        stages=[evaluator.stage_from_filename(path) for path in phase_files],
        output_directory=manifests_dir,
    )
    return manifest_path


# Only run if this script is not imported as a module

# The registration backends spawn worker processes.  On Windows the spawn start
# method re-imports this script in each child; without the
# __name__ == "__main__" guard around top-level work, that re-import would
# restart the whole cohort in every worker.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent

    class_name = "tutorial_16_duke_heart_physics_informed_motion_prep"

    test_mode = ProcessTests.running_as_test()
    parameters = DUKE_HEART_PHYSICS_INFORMED

    output_dir = parameters.prep_directory(test_mode)
    baselines_dir = repo_root / "tests" / "baselines"

    # Tutorial 4's reference-frame surfaces, which the model is built from.
    input_dir = parameters.input_directory(test_mode)
    # The gated labelmaps, one directory per case.
    data_dir = parameters.hold_out_directory(test_mode)

    number_of_pca_components = parameters.pca_components(test_mode)

    # Edge length of the template's tetrahedra.  This is the one number that
    # decides how much of the myocardium the physics term ever sees; see the
    # parameters module for the measured coverage at each value.
    ssm_element_size_mm = parameters.ssm_element_size_mm

    # Pitch of the grid the mean surface is voxelized on before it is meshed.
    # Finer than the element size, so the voxelization is not what limits the
    # template's fidelity to the surface.
    voxelization_spacing_mm = 0.5

    # Atlas iterations used to build the reference surface; 1 is a single
    # template-biased pass.
    mean_surface_iterations = 1 if test_mode else 3

    # Points kept per surface; see the parameters module.
    model_points = parameters.model_points

    # Labels left out of the whole-heart structure, the same ones Tutorials 4
    # and 6 drop, so the frames and the model describe the same structure.
    interior_object_ids = parameters.interior_object_ids

    # Distance-map weights finetuned by
    # tutorial_02_duke_heart_distancemap_finetune_icon.py.  Stock uniGradICON
    # weights are out of distribution for distance maps, so without these the
    # correspondences barely move off the template and the modes come out far
    # too tight.
    icon_weights_path = parameters.distancemap_icon_weights_file(test_mode)

    log_level = logging.INFO

    # Directory setup and data reading
    logger = MONAIPhysioBase(class_name=class_name, log_level=log_level)

    # A GPU is assumed. Every registration, fit and network pass below runs on
    # one, so fail here rather than hours into the cohort at the first call.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device is visible. Tutorials 16 to 18 assume a GPU; a "
            "CPU-only run is not supported and would take days."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    contour_tools = ProcessSurfaceMesh(log_level=log_level)
    volume_tools = ProcessVolumetricMesh(log_level=log_level)

    tutorial_results: dict[str, Any] = {"cases": {}, "screenshots": []}

    # One reference-frame surface per case, less the held-out one: Tutorial 18
    # scores the surrogate on that case, so the model must not have seen it.
    sample_files = [
        path
        for path in sorted(input_dir.glob(f"*_ref_{parameters.whole_heart_name}.vtp"))
        if not path.name.startswith(parameters.hold_out_case)
    ]
    if test_mode:
        sample_files = sample_files[:3]
    if len(sample_files) < 3:
        raise FileNotFoundError(
            f"Need at least 3 reference-frame heart surfaces under {input_dir}; "
            f"found {len(sample_files)}.\n"
            "Run tutorial_04_duke_heart_labelmap_to_vtk.py first."
        )

    sample_surfaces: list[pv.DataSet] = []
    for sample_file in sample_files:
        surface = cast(pv.PolyData, pv.read(str(sample_file)))
        sample_surfaces.append(
            contour_tools.remesh_and_smooth_surface(
                surface, 1.0 - model_points / surface.n_points, 0
            )
        )

    # Step 1: the unbiased mean surface, as Tutorial 6 builds it.  Picking one
    # case instead would make the model inherit that case's shape.  Cached: it
    # costs one deformable registration per case per atlas iteration.
    reference_surface_file = output_dir / "reference_mean_surface.vtp"
    # Keyed on the settings the atlas was corresponded with, not on the file
    # merely being there: reusing an atlas built at one dilation, saturation
    # radius or checkpoint while the model below corresponds at another is the
    # one way the two can disagree without saying so.
    mean_surface_settings = {
        "iterations": mean_surface_iterations,
        "mask_dilation_mm": parameters.mask_dilation_mm,
        "distance_squared_max": parameters.distancemap_squared_max,
        "icon_weights": (
            [str(icon_weights_path), icon_weights_path.stat().st_mtime_ns]
            if icon_weights_path.exists()
            else None
        ),
    }
    settings_file = output_dir / "reference_mean_surface_settings.json"
    cached_settings = (
        json.loads(settings_file.read_text(encoding="utf-8"))
        if reference_surface_file.exists() and settings_file.exists()
        else None
    )
    if cached_settings != mean_surface_settings:
        mean_workflow = WorkflowCreateMeanSurface(
            surfaces=sample_surfaces,
            template_surface=sample_surfaces[len(sample_surfaces) // 2],
            log_level=log_level,
        )
        mean_workflow.set_number_of_iterations(mean_surface_iterations)
        mean_workflow.set_mask_dilation_mm(parameters.mask_dilation_mm)
        mean_workflow.set_distance_squared_max(parameters.distancemap_squared_max)
        if icon_weights_path.exists():
            mean_workflow.set_icon_weights_path(str(icon_weights_path))
        mean_result = mean_workflow.process()
        mean_result["mean_surface"].save(str(reference_surface_file))
        settings_file.write_text(
            json.dumps(mean_surface_settings, indent=2), encoding="utf-8"
        )
    reference_surface = cast(pv.PolyData, pv.read(str(reference_surface_file)))

    # Step 2: fill that surface with tetrahedra.  The mesh starts as a voxel
    # staircase and is then relaxed onto the surface, which holds every cell
    # above a scaled Jacobian of 0.25 -- cutting at the surface instead would
    # shatter the boundary cells into slivers nothing downstream could repair.
    # 0.25 (rather than trim_tetrahedra_to_surface's own 0.1 default) costs
    # a still-modest amount of boundary fit accuracy but removes the
    # population of near-floor slivers that a per-subject fit reliably tips
    # into inverted or degenerate elements no amount of repair smoothing can
    # undo; 0.2 fixed most of these but left a handful of subjects still
    # pinned right at that floor.
    template_file = parameters.ssm_template_file(test_mode)
    if not template_file.exists():
        voxelization_grid = contour_tools.create_reference_image(
            mesh=reference_surface,
            spatial_resolution=voxelization_spacing_mm,
            ptype=itk.F,
        )
        reference_mask = contour_tools.create_mask_from_mesh(
            reference_surface, voxelization_grid
        )
        template_mesh = volume_tools.trim_tetrahedra_to_surface(
            volume_tools.extract_tetrahedra(
                reference_mask, element_size_mm=ssm_element_size_mm
            ),
            reference_surface,
            min_scaled_jacobian=0.25,
        )
        template_mesh.save(str(template_file))
    template_mesh = cast(pv.UnstructuredGrid, pv.read(str(template_file)))

    # Every downstream stage trusts these elements, so report what they are.
    element_quality = np.asarray(
        template_mesh.cell_quality(["scaled_jacobian"]).cell_data["scaled_jacobian"]
    )
    logger.log_info(
        "Template: %d nodes, %d elements, %.0f mm^3 (%.1f%% of the %.0f mm^3 the "
        "mean surface encloses), scaled Jacobian min %.3f mean %.3f",
        template_mesh.n_points,
        template_mesh.n_cells,
        template_mesh.volume,
        100.0 * template_mesh.volume / max(reference_surface.volume, 1.0),
        reference_surface.volume,
        float(element_quality.min()),
        float(element_quality.mean()),
    )

    # Step 3: decompose the population against the tetrahedral template.  The
    # samples stay surfaces: they only supply the distance maps that drive the
    # correspondence, and step 4 of the workflow warps the reference alone.
    model_file = parameters.ssm_model_file(test_mode)
    mean_volume_file = parameters.ssm_mean_volume_file(test_mode)
    if not (model_file.exists() and mean_volume_file.exists()):
        model_workflow = WorkflowCreateStatisticalModel(
            sample_meshes=sample_surfaces,
            reference_mesh=template_mesh,
            number_of_pca_components=number_of_pca_components,
            icp_transform_type=parameters.icp_transform_type,
            mask_dilation_mm=parameters.mask_dilation_mm,
            distance_squared_max=parameters.distancemap_squared_max,
            solve_for_surface_pca=False,
            log_level=log_level,
        )
        if icon_weights_path.exists():
            model_workflow.set_icon_weights_path(str(icon_weights_path))
        else:
            model_workflow.log_warning(
                "Finetuned distance-map ICON weights not found at %s; building "
                "the model with the stock uniGradICON weights, which are out of "
                "distribution for distance maps and will understate the "
                "population's variance. Run "
                "tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py "
                "to create them.",
                icon_weights_path,
            )
        model_result = model_workflow.process()

        # The components are 3 * n wide for the *tetrahedral* mesh's n points,
        # so this model and Tutorial 6's surface one are not interchangeable.
        with model_file.open("w", encoding="utf-8") as f:
            json.dump(model_result["pca_model"], f, indent=2)
        model_result["pca_mean_mesh"].save(str(mean_volume_file))
        model_result["pca_mean_surface"].save(
            str(parameters.ssm_mean_boundary_file(test_mode))
        )

    with model_file.open(encoding="utf-8") as f:
        pca_model = json.load(f)
    pca_mean_volume = cast(pv.DataSet, pv.read(str(mean_volume_file)))

    # Step 4: fit that model to every case and propagate it through the phases.
    use_finetuned_weights = icon_weights_path.exists()
    case_dirs = sorted(
        path for path in data_dir.glob("pm[0-9][0-9][0-9][0-9]") if path.is_dir()
    )
    if not case_dirs:
        raise FileNotFoundError(
            f"No pm???? case directories found under {data_dir}.\n"
            "See data/Duke-Heart-4DLabelmaps/README.md."
        )

    for case_dir in case_dirs:
        case_id = case_dir.name
        frame_files = sorted(case_dir.glob(f"*{LABELMAP_SUFFIX}"))
        reference_files = [
            path for path in frame_files if path.name.endswith(f"_ref{LABELMAP_SUFFIX}")
        ]
        if not reference_files:
            logger.log_warning("Skipping %s: no *_ref_labelmap.nii.gz frame", case_id)
            continue
        reference_file = reference_files[0]

        logger.log_section(
            "Processing case %s: %d gated frames", case_id, len(frame_files), width=48
        )

        case_output_dir = output_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        reference_labelmap = itk.imread(str(reference_file))
        reference_frame_surface = parameters.whole_heart_surface(
            reference_file, test_mode, log_level
        )

        fitted_reference_model_file = case_output_dir / f"{case_id}_ssm_model.vtu"
        pca_coefficients_file = case_output_dir / f"{case_id}_ssm_pca_coefficients.json"
        if not (
            fitted_reference_model_file.exists() and pca_coefficients_file.exists()
        ):
            # A volumetric template's interior nodes have no image evidence, so
            # the fit must score against a distance map that leaves them
            # unpenalized.  ``patient_labelmap`` is what selects that map; with
            # none, the fit falls back to unsigned distance-to-surface
            # everywhere and would collapse the volume onto its own boundary.
            assert reference_labelmap is not None, (
                f"{case_id} has no reference labelmap, so the volumetric fit "
                "has no distance map that spares its interior nodes."
            )
            fit_workflow = WorkflowFitStatisticalModelToPatient(
                template_model=pca_mean_volume,
                patient_models=[reference_frame_surface],
                patient_image=None,
                patient_labelmap=reference_labelmap,
                labelmap_interior_object_ids=interior_object_ids,
                log_level=log_level,
            )
            fit_workflow.set_use_pca_registration(
                use_pca_registration=True,
                pca_model=pca_model,
                number_of_pca_components=number_of_pca_components,
                use_surface=False,
            )
            fit_workflow.set_icp_transform_type(parameters.icp_transform_type)
            fit_workflow.set_mask_dilation_mm(parameters.mask_dilation_mm)
            fit_workflow.set_distancemap_squared_max(parameters.distancemap_squared_max)
            if use_finetuned_weights:
                fit_workflow.set_labelmap_to_labelmap_icon_weights_path(
                    str(icon_weights_path)
                )
            fit_result = fit_workflow.process()

            pca_coefficients = fit_workflow.pca_coefficients
            assert pca_coefficients is not None
            with pca_coefficients_file.open(mode="w", encoding="utf-8") as f:
                json.dump(pca_coefficients.tolist(), f)

            # The model is volumetric here, so "fitted_reference_model" (the
            # tetrahedral mesh) and "fitted_reference_mesh" (its boundary) are
            # different geometries.  The volume is what the network predicts on.
            #
            # The fit warps the template per subject with no cell-quality
            # constraint, so it can flip a handful of elements even though the
            # template itself was checked; repair before saving so Tutorial 17
            # never has to. A subject whose fit folds badly enough that
            # repair can't recover it needs a real re-fit, not a batch job
            # that dies on it -- skip the subject and keep the population
            # going rather than losing every case after it.
            try:
                fitted_reference_model = volume_tools.repair_inverted_tetrahedra(
                    cast(pv.UnstructuredGrid, fit_result["fitted_reference_model"])
                )
            except ValueError as error:
                logger.log_warning(
                    "Skipping %s: fitted reference model could not be repaired: %s",
                    case_id,
                    error,
                )
                continue
            fitted_reference_model.save(str(fitted_reference_model_file))
            fit_result["fitted_reference_mesh"].save(
                str(case_output_dir / f"{case_id}_ssm_surface.vtp")
            )
        fitted_reference_model = cast(
            pv.UnstructuredGrid, pv.read(str(fitted_reference_model_file))
        )

        # One grid is built around the reference frame's heart and reused by
        # every frame, so the whole case is registered in a common space.
        registration_grid = parameters.frame_registration_grid(reference_frame_surface)

        phase_outputs = []
        for frame_file in frame_files:
            stem = frame_file.name[: -len(LABELMAP_SUFFIX)]
            model_path = case_output_dir / f"{stem}_ssm_model.vtu"
            # The evaluation stack scores against per-frame ``*_ssm_surface.vtp``
            # fits, so each frame's boundary is written beside its volume.  Every
            # frame shares the template's topology, so the boundaries share a
            # point ordering too.
            surface_path = case_output_dir / f"{stem}_ssm_surface.vtp"
            if not (model_path.exists() and surface_path.exists()):
                if frame_file == reference_file:
                    # The fit already placed the model on this frame.
                    logger.log_info("Case %s: reference frame %s", case_id, stem)
                    phase_model: pv.DataSet = fitted_reference_model
                else:
                    logger.log_info("Case %s: warping to frame %s", case_id, stem)
                    # The transform is applied point-wise, so the tetrahedra
                    # ride along and the warped frame keeps the template's
                    # topology.
                    phase_model = parameters.register_to_frame(
                        fitted_reference_model,
                        parameters.whole_heart_surface(
                            frame_file, test_mode, log_level
                        ),
                        registration_grid,
                        test_mode,
                        log_level,
                    )
                phase_model.save(str(model_path))
                phase_model.extract_surface(algorithm="dataset_surface").save(
                    str(surface_path)
                )
            phase_outputs.append(
                {
                    "frame_stem": stem,
                    "model_file": model_path,
                    "surface_file": surface_path,
                }
            )

        tutorial_results["cases"][case_id] = {
            "pca_coefficients_file": pca_coefficients_file,
            "fitted_reference_model_file": fitted_reference_model_file,
            "phase_outputs": phase_outputs,
        }

    if not tutorial_results["cases"]:
        raise RuntimeError(
            f"No case under {data_dir} carried a reference frame; nothing was fitted."
        )

    # Step 5: one manifest per case, which is what Tutorial 17 trains from.
    manifests_dir = output_dir / "manifests"
    manifests: dict[str, Path] = {}
    evaluator = EvaluateMovementDukeHeart(log_level=log_level)
    for case_id in tutorial_results["cases"]:
        manifest_path = _write_case_manifest(
            output_dir / case_id, manifests_dir, logger, evaluator
        )
        if manifest_path is not None:
            manifests[case_id] = manifest_path
    if not manifests:
        raise RuntimeError(
            f"No case under {output_dir} yielded a complete manifest; "
            "Tutorial 17 has nothing to train on."
        )
    tutorial_results["manifests"] = manifests
    tutorial_results["template_file"] = template_file
    tutorial_results["model_file"] = model_file
    tutorial_results["mean_volume_file"] = mean_volume_file
    logger.log_info("Wrote %d manifests under %s", len(manifests), manifests_dir)

    # Testing
    tt = ProcessTests(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    # A cut plane rather than the outside, because what matters about this
    # template is that it has an interior at all.
    template_section = template_mesh.clip(
        normal="x", origin=template_mesh.center, crinkle=True
    )
    last_case = list(tutorial_results["cases"].values())[-1]
    tutorial_results["screenshots"] = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, template_section),
            "ssm_template_section.png",
            camera_position="yz",
            color="lightsteelblue",
        ),
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(last_case["fitted_reference_model_file"]))),
            "fitted_reference_model.png",
            camera_position="iso",
            color="steelblue",
            opacity=0.9,
        ),
    ]
