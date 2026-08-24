"""
Tutorial 13: Combined Heart and Lung Motion on a Static Clinical CT

Purpose
-------
Animate one ungated breath-hold chest CT (``data/Chest-CT/Chest-CT.mha``) with
both of its rhythms, taking every deformation from a trained MeshGraphNet rather
than from a registration. Nothing here needs a 4D acquisition: the input is a
single 3D scan.

1. **Respiratory motion.** The lung MeshGraphNet of Tutorial 9 (lung), applied to
   the Chest-CT lung fit of Tutorial 7 (lung) at each respiratory stage.
2. **Cardiac motion.** The heart MeshGraphNet of Tutorial 9 (Duke heart), applied
   to a Duke-heart-SSM fit of this patient's heart, layered on top.

Each stage's per-vertex displacements are rasterized onto the Chest-CT grid by
``WorkflowInferMovement.create_deformation_field`` and spread into a continuous
deformation by ``TransformTools.smooth_deformation_field_transform``. The 100
combined frames are written as VTP surfaces and assembled into a single animated
4D USD split by anatomy label, alongside the CT and labelmap warped by the same
per-frame deformation.

Segmenters
----------
Each statistical model is fitted through the segmenter that built it, so the
patient geometry the network sees is the geometry it was trained on:

- ``SegmentNVSegmentCTMRI`` for the lungs, the segmenter behind the Tutorial 6
  (lung) shape model. Its whole-thorax labelmap is also what the animated
  surface is contoured from and what names the anatomy prims in the USD.
- ``SegmentHeartSimplewareTrimmedBranches`` for the heart, the segmenter that
  produced the Duke-Heart-4DLabelmaps the cardiac shape model and network were
  trained on. It calls Synopsys Simpleware Medical, so this tutorial needs a
  Simpleware installation, unlike the rest of the lung chain.

``SegmentNVSegmentCTMRI`` resolves the heart as a single oversized blob on this
scan, so before anything downstream reads the thorax labelmap its heart labels
are demoted to one ``mediastinal_tissue`` id and the Simpleware heart is painted
in over them. The Simpleware structures are written under the NV-Segment ids
that name the same thing, which keeps one taxonomy naming every prim but
stretches two of those names: ``ventricle_myocardium_left`` then holds the whole
Simpleware myocardium, and ``heart`` its dilated whole-heart envelope.

Forward and inverse deformations
--------------------------------
Warping a *mesh* moves each vertex by the reference-to-stage displacement, while
resampling an *image* maps each output voxel back to where it should be sampled
from. Those are opposite mappings, so every stage is rasterized twice, as
``direction="forward"`` for the surfaces and ``direction="inverse"`` for the CT
and labelmap, and the inverse composite is built in the reverse order.

Sliding at organ boundaries
---------------------------
Spreading a surface's displacement outward carries the whole vector with it,
which would drag the chest wall, the ribs and the mediastinum along with a lung
that is sliding past them. A real thorax slips there instead: the visceral
pleura slides against the parietal pleura, the epicardium against the
pericardium, and only the motion *along the surface normal* -- the organ filling
and emptying -- is transmitted outward. Each stage's samples are therefore split
into their normal and tangential components, and outside the organ only the
normal component is spread. Inside it both are, so the parenchyma and the
myocardium still follow their own surfaces. The organ masks that select between
the two are softened by ``slip_transition_mm``, so the sliding stops over a band
rather than at a step that would tear the warped volumes.

Reference stage
---------------
The lung network predicts displacement relative to the ``T70`` phase its training
cases were fitted at, so the breath-hold Chest-CT fit plays the role of ``T70``:
the respiratory displacement is near zero at stage 0.70 and that frame reproduces
the input scan.

Composition order
-----------------
Cardiac deformation is applied first, at the reference frame where the cardiac
fields are defined, giving one warped surface per (breath phase, cardiac stage).
Each rendered frame then bilinearly interpolates that precomputed grid: the
respiratory axis advances with the breath phase, while the cardiac axis advances
**independently and continuously** at ``cardiac_cycles_per_phase`` beats per
phase. A value below 1.0 therefore lets a single heartbeat carry across a phase
boundary (0.75 = each breath phase covers three-quarters of a beat). Both axes
wrap, so the sequence loops.

Anatomy materials
-----------------
The per-cell ``boundary_labels`` produced by contouring the labelmap propagate
unchanged through every warp (each frame is a deep copy with only its points
moved; remeshing carries them over to the new cells when enabled).
``ConvertVTKToUSD``,
given ``segmenter.taxonomy.all_labels()`` and the segmenter, splits each frame
into per-organ prims, and ``USDAnatomyTools.enhance_meshes`` then binds the
matching OmniSurface material (diffuse color, subsurface scattering, etc.).

Prerequisites
-------------
Tutorial 6 (lung), Tutorial 7 (lung), Tutorial 6 (Duke heart) and Tutorial 9 for
both anatomies. Requires the ``[physicsnemo]`` extra and Simpleware Medical.

Data Required
-------------
  * ``data/Chest-CT/Chest-CT.mha`` -- ``physiotwin4d-download-data Chest-CT``
    (see ``data/Chest-CT/README.md`` for the data source and required citation)
  * ``output/tutorial_07_lung/`` -- lung fit of that scan
  * ``output/tutorial_06_duke_heart/`` -- Duke heart shape model
  * ``network_weights/physicsnemo_mgn_lung_motion/``,
    ``network_weights/physicsnemo_mgn_duke_heart_motion/`` -- Tutorial 9 weights

Outputs (under ``output/tutorial_13_heart_and_lung``, which needs about 43 GB
free: the hundred warped CT volumes are 40 GB of it)
-------
- ``chest_ct_labelmap.mha`` / ``chest_ct_heart_labelmap.mha`` - the two cached
  segmentations of the input scan.
- ``chest_ct_merged_labelmap.mha`` - the thorax labelmap with the Simpleware
  heart substituted in, the one everything downstream reads.
- ``heart_fit/`` - this patient's Duke-heart-SSM coefficients and fitted mesh.
- ``deformation_field_<rhythm>_s<sss>.mha`` /
  ``surface_normal_field_<rhythm>_s<sss>.mha`` /
  ``deformed_<rhythm>_surface_s<sss>.vtp`` - per-stage inferred motion.
- ``interior_mask_<rhythm>.mha`` - the softened organ mask each rhythm's sliding
  is confined to.
- ``breathing_lungs.usd`` / ``beating_heart.usd`` - each rhythm on its own.
- ``combined_frame_<iii>.vtp`` (``000..099``) + ``heart_and_lung_motion.usd`` -
  the combined respiratory + cardiac 4D motion, painted with anatomy materials.
- ``combined_ct_<iii>.mha`` / ``combined_labelmap_<iii>.mha`` (``000..099``) - the
  original CT and labelmap warped by the same per-frame combined deformation
  (signed-short), so their anatomy tracks the displaced surfaces.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional, cast

import itk
import numpy as np
import pyvista as pv
from parameters_duke_heart_labelmaps import DUKE_HEART
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

from physiotwin4d import (
    ContourTools,
    ConvertVTKToUSD,
    ImageTools,
    SegmentHeartSimplewareTrimmedBranches,
    SegmentNVSegmentCTMRI,
    TestTools,
    TransformTools,
    USDAnatomyTools,
    WorkflowConvertVTKToUSD,
    WorkflowFitStatisticalModelToPatient,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)

# Only run if this script is not imported as a module

# nnUNetv2 (inside SegmentNVSegmentCTMRI) and torch spawn worker processes. On
# Windows the spawn start method re-imports this script in each child; without
# the __name__ == "__main__" guard around top-level work, that re-import would
# restart the whole pipeline in every worker.
if __name__ == "__main__":
    log_level = logging.INFO
    logging.basicConfig(level=log_level)
    logger = logging.getLogger("tutorial_13_heart_and_lung_motion")

    tutorials_dir = Path(__file__).resolve().parent

    # ---- Inputs ------------------------------------------------------------
    test_mode = TestTools.running_as_test()

    # The ungated clinical scan every rhythm is inferred onto.
    patient_image_file = (
        LUNG_CT_DIRLAB.hold_out_directory(test_mode) / LUNG_CT_DIRLAB.hold_out_case
    )

    # Tutorial 9 weights for each rhythm, and the Tutorial 6 shape model the
    # cardiac one was trained on.
    lung_model_dir = LUNG_CT_DIRLAB.mgn_weights_dir
    heart_model_dir = DUKE_HEART.mgn_weights_dir
    heart_pca_json = DUKE_HEART.pca_json_file
    heart_pca_mean_file = DUKE_HEART.pca_mean_file

    # Tutorial 7 (lung) fit of this same scan: the network's reference geometry
    # and the shape parameters it is conditioned on.
    tutorial_07_lung_dir = tutorials_dir / "output" / "tutorial_07_lung"
    lung_coefficients_file = (
        tutorial_07_lung_dir / "tutorial_07_lung_registered_coefficients.json"
    )
    lung_fitted_reference_mesh_file = (
        tutorial_07_lung_dir / "tutorial_07_lung_template_surface_registered.vtp"
    )

    # Distance-map weights finetuned by
    # tutorial_02_duke_heart_distancemap_finetune_icon.py, used by the heart fit
    # below; optional, the stock uniGradICON weights are used without them.
    heart_icon_weights_path = (
        tutorials_dir
        / "network_weights"
        / "icon_duke_heart_distancemap"
        / "icon_duke_heart_distancemap_model"
        / "checkpoints"
        / "network_weights_final.trch"
    )

    # Needs about 43 GB free: a hundred frames of warped CT is 40 GB of it,
    # since resampling costs the compression the original scan had.
    output_dir = tutorials_dir / "output" / "tutorial_13_heart_and_lung"

    # ---- Parameters --------------------------------------------------------
    # Respiratory stages, the DIR-Lab T00..T90 phases the lung network was
    # trained on, so every inference is in distribution.
    respiratory_stages = [round(0.1 * k, 2) for k in range(10)]
    # Cardiac stages sampled over one heartbeat (fraction of the RR interval).
    cardiac_stages = [round(0.1 * k, 2) for k in range(10)]
    # Fraction of a cardiac cycle advanced per respiratory phase. The heart is
    # decoupled from breathing, so a value < 1 means a single beat continues into
    # the next phase (1.0 locks exactly one full beat to each phase).
    cardiac_cycles_per_phase = 0.75
    # Gaussian sigma (mm) that spreads each rhythm's surface-sampled
    # displacements into a continuous field. The lung shell is sparse relative
    # to the thorax it has to fill, so it is spread further than the heart.
    respiratory_sigma_mm = 15.0
    cardiac_sigma_mm = 10.0
    # How far each rhythm's push and pull carries *beyond* its own organ, as a
    # separate sigma for the normal component spread outside the interior mask.
    # The lungs drive the whole thorax, so they keep their full reach. The heart
    # sits in tissue that barely moves with it, so its influence is confined to
    # a quarter of that distance: the pericardial neighborhood still follows the
    # myocardium at full strength, while the mediastinum and chest wall further
    # out stop being pumped by it. Only the reach changes -- the displacement at
    # the heart surface, and everything inside it, is untouched.
    cardiac_exterior_sigma_mm = 0.25 * cardiac_sigma_mm
    # How wide a band (mm) the sliding motion dies out over at the pleura and
    # the pericardium. Zero would make each slip boundary a step, and shear the
    # voxels either side of it in opposite directions.
    slip_transition_mm = 5.0
    # How far past the labels that band starts. The fitted shape-model surface
    # and the segmentation of the same organ disagree by about a
    # deformation-grid voxel (3 mm here), so a fall-off that begins at the label
    # edge catches the surface the network predicted on: it costs a quarter of
    # the lung's tangential motion and a third of the heart's.
    slip_offset_mm = 3.0
    # Grid every deformation field is sampled on, as a fraction of the CT's own
    # voxel pitch per axis. The fields are Gaussian-smoothed by the sigmas above,
    # so they carry no detail a sub-millimeter grid could resolve, while a
    # full-resolution field of a 512x512x526 scan costs 1.6 GB and the frame loop
    # blends four of them at a time. Raise it toward 1.0 to sample them finer.
    deformation_field_scale = 0.3
    # Pitch the animated surface is contoured at. ContourTools.extract_contours
    # works on an isotropic grid of the labelmap's finest pitch, which for this
    # scan is 0.6 mm and yields a 4.5-million-point thorax: a hundred frames of
    # it is 5.5 GB of warped points held at once and 135 MB per frame on disk.
    # Coarsening the labelmap first sets that pitch instead, and the cost falls
    # with its square. Lower it for a finer animation, if the memory is there.
    surface_spacing_mm = 2.0
    # One-time conditioning of the patient surface, reused for every frame.
    surface_reduction_rate = 0.0  # fraction of triangles removed by remeshing
    surface_smoothing_iterations = 0  # Taubin (non-shrinking) iterations
    # USD playback rate; 10 stages/second ~= one heartbeat per second.
    frames_per_second = 10.0

    output_dir.mkdir(parents=True, exist_ok=True)

    if not patient_image_file.exists():
        raise FileNotFoundError(
            f"Patient chest CT not found: {patient_image_file}\n"
            "Run: physiotwin4d-download-data Chest-CT --directory data/Chest-CT"
        )
    for required, hint in (
        (
            lung_coefficients_file,
            "tutorial_07_lung_fit_statistical_model_to_patient.py",
        ),
        (
            lung_fitted_reference_mesh_file,
            "tutorial_07_lung_fit_statistical_model_to_patient.py",
        ),
        (heart_pca_json, "tutorial_06_duke_heart_create_statistical_model.py"),
        (heart_pca_mean_file, "tutorial_06_duke_heart_create_statistical_model.py"),
        (
            lung_model_dir / "mgn_stage_model.pt",
            "tutorial_09_lung_train_physicsnemo_mgn.py",
        ),
        (
            heart_model_dir / "mgn_stage_model.pt",
            "tutorial_09_duke_heart_train_physicsnemo_mgn.py",
        ),
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"Required input not found: {required}\nRun tutorials/{hint} first."
            )

    contour_tools = ContourTools(log_level=log_level)
    transform_tools = TransformTools(log_level=log_level)
    image_tools = ImageTools(log_level=log_level)

    patient_image = itk.imread(str(patient_image_file))

    # The grid the deformation fields live on. Only its geometry is used, so the
    # resampled intensities are irrelevant; it covers the same physical extent as
    # the scan, which is what a field that has to displace the whole thorax needs.
    deformation_grid = image_tools.resample_image_by_scale(
        patient_image, deformation_field_scale
    )
    logger.info(
        "Deformation grid: %s voxels at %s mm (scan is %s at %s mm)",
        list(itk.size(deformation_grid)),
        [round(s, 2) for s in deformation_grid.GetSpacing()],
        list(itk.size(patient_image)),
        [round(s, 2) for s in patient_image.GetSpacing()],
    )

    # ========================================================================
    # Stage 0: segment the scan once per segmenter, cached.
    # ========================================================================
    # The whole-thorax labelmap: the surface every frame is a deformation of,
    # and the taxonomy the final USD is split by. This is also the segmenter
    # the lung shape model was built with. Its heart is replaced below, but its
    # taxonomy still names the lung labels the slip mask selects, the id the
    # replaced heart is demoted to, and every id the animated surface carries
    # when Stage 3 splits it into USD prims.
    segmenter = SegmentNVSegmentCTMRI(log_level=log_level)
    chest_labelmap_file = output_dir / "chest_ct_labelmap.mha"
    if not chest_labelmap_file.exists():
        itk.imwrite(
            segmenter.segment(patient_image)["labelmap"],
            str(chest_labelmap_file),
            compression=True,
        )
    chest_labelmap = itk.imread(str(chest_labelmap_file))

    # The heart labelmap, from the segmenter that produced the labelmaps the
    # cardiac shape model and network were trained on. It supplies the shape
    # model's fit target, the cardiac slip mask, and the heart of the thorax
    # labelmap above.
    heart_labelmap_file = output_dir / "chest_ct_heart_labelmap.mha"
    if not heart_labelmap_file.exists():
        heart_segmenter = SegmentHeartSimplewareTrimmedBranches(log_level=log_level)
        itk.imwrite(
            heart_segmenter.segment(patient_image)["labelmap"],
            str(heart_labelmap_file),
            compression=True,
        )
    heart_labelmap = itk.imread(str(heart_labelmap_file))

    # NV-Segment resolves this scan's heart as a single undifferentiated blob
    # that is far larger than the organ, while the Simpleware heart above is
    # the geometry the cardiac model was trained on. So every NV-Segment heart
    # label is first demoted to NV-Segment's own mediastinal_tissue, then the
    # Simpleware structures are painted in under the NV-Segment ids that name
    # the same thing. Demoting rather than erasing keeps the tissue the blob
    # covered in the labelmap -- as mediastinum, which is what it is -- so the
    # warped volumes do not grow a hole around the heart, and the animated
    # surface and USD prims take the heart's shape from Simpleware alone.
    #
    # Reusing NV-Segment's ids keeps one taxonomy naming every prim, at the
    # cost of two stretched names: "ventricle_myocardium_left" then holds the
    # whole Simpleware myocardium and "heart" its whole-heart envelope.
    mediastinal_tissue_id = 159
    heart_label_remap = {
        1: 151,  # left_ventricle  -> ventricle_left
        2: 152,  # right_ventricle -> ventricle_right
        3: 149,  # left_atrium     -> atrium_left
        4: 153,  # right_atrium    -> atrium_right
        5: 154,  # myocardium      -> ventricle_myocardium_left
        6: 115,  # heart           -> heart
    }
    # Both segmenters ran on patient_image and SegmentAnatomyBase.segment copies
    # its geometry onto the result, so the two label arrays are voxelwise
    # aligned. Simpleware labels do not overlap, so the write order is free.
    # array_from_image copies, so the cached NV-Segment labelmap on disk stays
    # the raw segmentation and re-runs stay idempotent.
    merged_labels = itk.array_from_image(chest_labelmap)
    heart_labels = itk.GetArrayViewFromImage(heart_labelmap)
    assert merged_labels.shape == heart_labels.shape, (
        "the two segmentations must share the input scan's grid"
    )
    merged_labels[
        np.isin(merged_labels, list(segmenter.taxonomy.labels_in_group("heart")))
    ] = mediastinal_tissue_id
    for simpleware_id, chest_id in heart_label_remap.items():
        merged_labels[heart_labels == simpleware_id] = chest_id
    chest_labelmap = itk.GetImageFromArray(merged_labels)
    chest_labelmap.CopyInformation(heart_labelmap)
    itk.imwrite(
        chest_labelmap,
        str(output_dir / "chest_ct_merged_labelmap.mha"),
        compression=True,
    )

    # ========================================================================
    # Stage 1: fit the Duke heart shape model to this patient's heart.
    # ========================================================================
    # The cardiac network is conditioned on those coefficients and predicts
    # displacements of that fitted mesh, so this is what puts the beating heart
    # in this patient rather than in the model's own frame. Tutorial 7 (lung)
    # already did the equivalent for the lungs.
    heart_fit_dir = output_dir / "heart_fit"
    heart_coefficients_file = heart_fit_dir / "heart_registered_coefficients.json"
    heart_fitted_reference_mesh_file = (
        heart_fit_dir / "heart_template_surface_registered.vtp"
    )
    if not (
        heart_coefficients_file.exists() and heart_fitted_reference_mesh_file.exists()
    ):
        heart_fit_dir.mkdir(parents=True, exist_ok=True)

        # The whole heart minus its chamber cavities and the vessels whose
        # extent varies too much between patients: the structure the shape
        # model describes, named by DUKE_HEART.interior_object_ids.
        heart_labels = itk.GetArrayViewFromImage(heart_labelmap)
        wall_ids = [
            int(value)
            for value in np.unique(heart_labels)
            if value != 0 and int(value) not in DUKE_HEART.interior_object_ids
        ]
        heart_mask = itk.GetImageFromArray(
            np.isin(heart_labels, wall_ids).astype(np.uint8)
        )
        heart_mask.CopyInformation(heart_labelmap)
        heart_surface = contour_tools.extract_label_surfaces(
            heart_mask,
            isotropic_spacing_mm=DUKE_HEART.surface_spacing_mm,
            smoothing_iterations=DUKE_HEART.surface_smoothing_iterations,
        )[1]

        with heart_pca_json.open(encoding="utf-8") as f:
            heart_pca_model: dict[str, Any] = json.load(f)

        heart_fit_workflow = WorkflowFitStatisticalModelToPatient(
            template_model=cast(pv.DataSet, pv.read(str(heart_pca_mean_file))),
            patient_models=[heart_surface],
            patient_image=patient_image,
            patient_labelmap=heart_labelmap,
            # The labels the whole-heart surface leaves out are the ones a
            # distance map must not measure to either.
            labelmap_interior_object_ids=DUKE_HEART.interior_object_ids,
            log_level=log_level,
        )
        heart_fit_workflow.set_icp_transform_type(DUKE_HEART.icp_transform_type)
        heart_fit_workflow.set_mask_dilation_mm(DUKE_HEART.mask_dilation_mm)
        heart_fit_workflow.set_distancemap_squared_max(
            DUKE_HEART.distancemap_squared_max
        )
        heart_fit_workflow.set_use_pca_registration(
            use_pca_registration=True,
            pca_model=heart_pca_model,
            number_of_pca_components=DUKE_HEART.pca_components(test_mode),
            use_surface=False,
        )
        if heart_icon_weights_path.exists():
            heart_fit_workflow.set_labelmap_to_labelmap_icon_weights_path(
                str(heart_icon_weights_path)
            )
        else:
            heart_fit_workflow.log_warning(
                "Finetuned distance-map ICON weights not found at %s; fitting with "
                "the stock uniGradICON weights. Run "
                "tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py to "
                "create them.",
                heart_icon_weights_path,
            )

        heart_fit_result = heart_fit_workflow.process()

        heart_coefficients = heart_fit_workflow.pca_coefficients
        assert heart_coefficients is not None, (
            "pca_coefficients must be set after a PCA fit"
        )
        with heart_coefficients_file.open(mode="w", encoding="utf-8") as f:
            json.dump(heart_coefficients.tolist(), f)
        heart_fit_result["fitted_reference_mesh"].save(
            str(heart_fitted_reference_mesh_file)
        )
        logger.info("Fitted the Duke heart model to %s", patient_image_file.name)

    # ========================================================================
    # Interior masks: where sliding propagates, and where only expansion does.
    # ========================================================================
    def interior_mask_on_grid(labelmap: itk.Image, label_ids: list[int]) -> itk.Image:
        """Ramp an organ's labels into the blend weight the spreading uses.

        1 inside the organ, where a stage's full displacement is propagated,
        falling to 0 outside it, where only the component along the surface
        normal is. The fall-off is what keeps a slip boundary from shearing
        neighboring voxels in opposite directions and tearing the warped CT.

        It is placed by distance rather than by blurring the labels, because a
        symmetric blur would put the half-way point of that fall-off *on* the
        organ boundary -- which is where the network's displacements were
        predicted, and where the animated surface sits. Those samples would
        then lose a third of their tangential motion to a band meant for the
        tissue beyond them. The mask instead stays 1 until ``slip_offset_mm``
        past the labels and decays over the ``slip_transition_mm`` after that.
        """
        labels = itk.GetArrayViewFromImage(labelmap)
        binary = itk.GetImageFromArray(np.isin(labels, label_ids).astype(np.float32))
        binary.CopyInformation(labelmap)
        on_grid = transform_tools.transform_image(
            binary,
            itk.IdentityTransform[itk.D, 3].New(),
            deformation_grid,
        )
        interior = itk.GetImageFromArray(
            (itk.array_from_image(on_grid) > 0.5).astype(np.uint8)
        )
        interior.CopyInformation(on_grid)
        distance_mm = itk.array_from_image(
            itk.signed_maurer_distance_map_image_filter(
                interior,
                InsideIsPositive=False,
                SquaredDistance=False,
                UseImageSpacing=True,
            )
        )
        # Smoothstep rather than a straight ramp, so the mask has no kink at
        # either end for the warped volumes to crease along.
        ramp = np.clip((distance_mm - slip_offset_mm) / slip_transition_mm, 0.0, 1.0)
        mask = itk.GetImageFromArray(
            (1.0 - ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)
        )
        mask.CopyInformation(on_grid)
        return mask

    lung_interior_mask = interior_mask_on_grid(
        chest_labelmap, list(segmenter.taxonomy.labels_in_group("lung"))
    )
    heart_interior_mask = interior_mask_on_grid(
        heart_labelmap,
        [
            int(value)
            for value in np.unique(itk.GetArrayViewFromImage(heart_labelmap))
            if value != 0
        ],
    )
    for tag, mask_image in (
        ("respiratory", lung_interior_mask),
        ("cardiac", heart_interior_mask),
    ):
        itk.imwrite(
            mask_image, str(output_dir / f"interior_mask_{tag}.mha"), compression=True
        )

    # ========================================================================
    # Stage 2: infer each rhythm, one deformation per stage.
    # ========================================================================
    def stage_transforms(
        model_directory: Path,
        coefficients_file: Path,
        fitted_reference_mesh_file: Path,
        stages: list[float],
        sigma_mm: float,
        tag: str,
        interior_mask: itk.Image,
        exterior_sigma_mm: Optional[float] = None,
    ) -> tuple[list[itk.Transform], list[itk.Transform], list[pv.DataSet]]:
        """Infer one rhythm across ``stages`` as smoothed deformations.

        Each stage is rasterized twice: the forward field, which moves mesh
        vertices from the reference frame to the stage, and the inverse field,
        which is what resampling an image into that stage's frame needs. Both
        are spread into continuous transforms by the vertex counts the
        rasterization reports, so the smoothing keeps the displacement
        magnitude the network predicted.

        The spreading is given ``interior_mask`` and the surface normals the
        rasterization reports, so beyond the organ it carries only the motion
        along those normals: surrounding tissue is pushed and pulled by the
        organ without being dragged along it. ``exterior_sigma_mm`` spreads that
        outward motion by its own sigma, which is how far into the surrounding
        tissue the organ reaches; it defaults to ``sigma_mm``. The mask arrives
        in the reference frame, which is the frame the forward field is indexed
        in; the inverse field is indexed in the stage's own frame, so the mask is
        resampled into it first through the unrestricted inverse deformation.
        """
        infer = WorkflowInferMovement(
            WorkflowInferPhysicsNeMo(
                model_directory=model_directory, epoch=None, log_level=log_level
            ),
            log_level=log_level,
        )
        reference_points = np.asarray(
            pv.read(str(fitted_reference_mesh_file)).points, dtype=np.float64
        )
        forward_transforms: list[itk.Transform] = []
        inverse_transforms: list[itk.Transform] = []
        deformed_surfaces: list[pv.DataSet] = []
        directions: tuple[Literal["forward", "inverse"], ...] = ("forward", "inverse")
        for stage in stages:
            fields = {
                direction: infer.create_deformation_field(
                    shape_parameters=coefficients_file,
                    stage=float(stage),
                    reference_image=deformation_grid,
                    fitted_reference_mesh=fitted_reference_mesh_file,
                    direction=direction,
                )
                for direction in directions
            }
            pct = int(round(stage * 100))
            itk.imwrite(
                fields["forward"]["deformation_field"],
                str(output_dir / f"deformation_field_{tag}_s{pct:03d}.mha"),
                compression=True,
            )
            itk.imwrite(
                fields["forward"]["normal_image"],
                str(output_dir / f"surface_normal_field_{tag}_s{pct:03d}.mha"),
                compression=True,
            )
            fields["forward"]["deformed_surface"].save(
                str(output_dir / f"deformed_{tag}_surface_s{pct:03d}.vtp")
            )
            deformed_surfaces.append(fields["forward"]["deformed_surface"])

            forward_transforms.append(
                transform_tools.smooth_deformation_field_transform(
                    fields["forward"]["deformation_field"],
                    sigma_mm,
                    fields["forward"]["weight_image"],
                    fields["forward"]["normal_image"],
                    interior_mask,
                    exterior_sigma_mm,
                )
            )

            # The inverse field maps points in this stage's frame back to the
            # reference frame, which is exactly the mapping resampling the
            # reference mask into that frame needs. Spreading it unrestricted
            # first is what makes the mask available to restrict it with.
            unrestricted_inverse = transform_tools.smooth_deformation_field_transform(
                fields["inverse"]["deformation_field"],
                sigma_mm,
                fields["inverse"]["weight_image"],
            )
            inverse_transforms.append(
                transform_tools.smooth_deformation_field_transform(
                    fields["inverse"]["deformation_field"],
                    sigma_mm,
                    fields["inverse"]["weight_image"],
                    fields["inverse"]["normal_image"],
                    transform_tools.transform_image(
                        interior_mask, unrestricted_inverse, deformation_grid
                    ),
                    exterior_sigma_mm,
                )
            )

            # The magnitude that survives the smoothing, beside the magnitude
            # the network predicted: they should be close.
            smoothed = itk.array_from_image(
                forward_transforms[-1].GetDisplacementField()
            )
            predicted = (
                np.asarray(
                    fields["forward"]["deformed_surface"].points, dtype=np.float64
                )
                - reference_points
            )
            logger.info(
                "%s stage %.2f: max %.2f mm predicted, %.2f mm after smoothing",
                tag,
                stage,
                float(np.linalg.norm(predicted, axis=1).max()),
                float(np.linalg.norm(smoothed, axis=3).max()),
            )
        return forward_transforms, inverse_transforms, deformed_surfaces

    respiratory_forward, respiratory_inverse, lung_surfaces = stage_transforms(
        lung_model_dir,
        lung_coefficients_file,
        lung_fitted_reference_mesh_file,
        respiratory_stages,
        respiratory_sigma_mm,
        "respiratory",
        lung_interior_mask,
    )
    cardiac_forward, cardiac_inverse, heart_surfaces = stage_transforms(
        heart_model_dir,
        heart_coefficients_file,
        heart_fitted_reference_mesh_file,
        cardiac_stages,
        cardiac_sigma_mm,
        "cardiac",
        heart_interior_mask,
        cardiac_exterior_sigma_mm,
    )

    # Each rhythm on its own, as a reference for the combined animation below.
    for rhythm_meshes, rhythm_name, anatomy, by_connectivity in (
        (lung_surfaces, "breathing_lungs", "lung", True),
        (heart_surfaces, "beating_heart", "heart", False),
    ):
        WorkflowConvertVTKToUSD(
            input_meshes=rhythm_meshes,
            usd_project_name=rhythm_name,
            output_directory=output_dir,
            appearance="anatomy",
            anatomy_type=anatomy,
            separate_by_connectivity=by_connectivity,
            frames_per_second=float(len(rhythm_meshes)),
            log_level=log_level,
        ).process()

    # ========================================================================
    # Stage 3: combine the two rhythms.
    # ========================================================================
    # Labeled contour surface: each cell carries `boundary_labels`, which survive
    # the warps (deep copies) and let ConvertVTKToUSD split the USD by anatomy so
    # USDAnatomyTools.enhance_meshes can bind per-organ OmniSurface materials.
    # Contoured from a coarsened copy of the labelmap, so the animated surface
    # is sampled at surface_spacing_mm rather than at the scan's finest pitch.
    # Its heart cells come from the Simpleware segmentation merged in at Stage 0.
    # The full-resolution labelmap is still what gets warped per frame below.
    surface_labelmap = image_tools.resample_image_by_scale(
        chest_labelmap,
        min(1.0, float(np.min(chest_labelmap.GetSpacing())) / surface_spacing_mm),
        interpolate=False,
    )
    patient_surface = contour_tools.extract_contours(surface_labelmap)
    patient_surface = contour_tools.remesh_and_smooth_surface(
        patient_surface, surface_reduction_rate, surface_smoothing_iterations
    )
    logger.info(
        "Patient surface: %d points, %d cells (reduction=%.2f, smoothing_iters=%d)",
        patient_surface.n_points,
        patient_surface.n_cells,
        surface_reduction_rate,
        surface_smoothing_iterations,
    )

    # Cardiac motion at the reference frame, once per stage, reused across phases.
    cardiac_surfaces = [
        transform_tools.transform_pvcontour(patient_surface, cardiac_transform)
        for cardiac_transform in cardiac_forward
    ]

    n_phases = len(respiratory_forward)
    n_stages = len(cardiac_surfaces)

    # Respiratory-warped vertex positions for every (phase, stage):
    # resp_points[phase][stage] = forward_phase(cardiac_surface[stage]).points.
    # All of them are held at once, so this is the run's memory high-water mark;
    # raise surface_reduction_rate if it does not fit.
    logger.info(
        "Precomputed warped points: %d x %d x %d vertices, %.1f GB",
        n_phases,
        n_stages,
        patient_surface.n_points,
        n_phases * n_stages * patient_surface.n_points * 3 * 4 / 1e9,
    )
    resp_points: list[list[np.ndarray]] = []
    for phase_idx, respiratory_transform in enumerate(respiratory_forward):
        resp_points.append(
            [
                np.asarray(
                    transform_tools.transform_pvcontour(
                        cardiac_surface, respiratory_transform
                    ).points,
                    dtype=np.float32,
                )
                for cardiac_surface in cardiac_surfaces
            ]
        )
        logger.info("respiratory warp phase %d/%d done", phase_idx + 1, n_phases)

    for pattern in (
        "combined_frame_*.vtp",
        "combined_ct_*.mha",
        "combined_labelmap_*.mha",
    ):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    # The image counterpart of the surface warp above, sampled on the reference
    # grid, one per (phase, stage). Resampling maps each output point back
    # through the transform, so these are the *inverse* fields: the surfaces move
    # by forward_p(cardiac_s(x)), so the image is resampled through its inverse,
    # cardiac_s^-1(forward_p^-1(x)). ITK CompositeTransform applies its
    # last-added transform first, so the respiratory inverse is added last to act
    # before the cardiac one - the mirror of the forward order. Blending these
    # fields per frame is affine, as the surface point-blend is, so the warped
    # CT and labelmap track the displaced surfaces. Fields are built lazily and
    # evicted per phase so only the two phases bracketing the current frame are
    # ever held.
    field_cache: dict[tuple[int, int], np.ndarray] = {}

    def combined_field(phase: int, stage: int) -> np.ndarray:
        """Return the inverse displacement field of one (phase, stage) pair."""
        cached = field_cache.get((phase, stage))
        if cached is not None:
            return cached
        composite = itk.CompositeTransform[itk.D, 3].New()
        composite.AddTransform(cardiac_inverse[stage])
        composite.AddTransform(respiratory_inverse[phase])
        field = transform_tools.convert_transform_to_displacement_field(
            composite, deformation_grid, np_component_type=np.float32
        )
        arr: np.ndarray = itk.array_from_image(field)
        field_cache[(phase, stage)] = arr
        return arr

    def to_signed_short(image: itk.Image) -> itk.Image:
        """Cast a labelmap to signed short, whatever integer type it came in as."""
        result = itk.image_from_array(itk.array_from_image(image).astype(np.int16))
        result.CopyInformation(image)
        return result

    # Render frames by bilinearly interpolating the precomputed (phase, stage)
    # warped-point grid: respiratory advances with the breath phase, while the
    # cardiac cycle advances continuously and independently at
    # ``cardiac_cycles_per_phase`` beats per phase, so a heartbeat carries across
    # phase boundaries. Both axes wrap, so the sequence loops.
    frames_per_phase = n_stages
    n_frames = n_phases * frames_per_phase
    combined_files: list[Path] = []
    ct_files: list[Path] = []
    labelmap_files: list[Path] = []
    usd_frames: list[pv.PolyData] = []
    for frame_idx in range(n_frames):
        # Respiratory position: current breath phase and fraction into it.
        phase_pos = frame_idx / frames_per_phase
        phase_idx = int(phase_pos)
        next_phase_idx = (phase_idx + 1) % n_phases
        resp_blend = phase_pos - phase_idx

        # Cardiac position: continuous across phases, wrapping within the cycle.
        card_pos = (phase_pos * cardiac_cycles_per_phase * n_stages) % n_stages
        stage_idx = int(card_pos)
        next_stage_idx = (stage_idx + 1) % n_stages
        card_blend = card_pos - stage_idx

        # Interpolate the cardiac cycle within each bounding phase, then between
        # the two phases.
        phase_a = (1.0 - card_blend) * resp_points[phase_idx][stage_idx] + (
            card_blend * resp_points[phase_idx][next_stage_idx]
        )
        phase_b = (1.0 - card_blend) * resp_points[next_phase_idx][stage_idx] + (
            card_blend * resp_points[next_phase_idx][next_stage_idx]
        )
        points = (1.0 - resp_blend) * phase_a + resp_blend * phase_b

        combined_surface = patient_surface.copy(deep=True)
        combined_surface.points = points

        frame_file = output_dir / f"combined_frame_{frame_idx:03d}.vtp"
        combined_surface.save(str(frame_file))
        combined_files.append(frame_file)
        usd_frames.append(combined_surface)

        # Warp the original CT and labelmap by the same combined deformation, using
        # the field bilinearly blended over the same four (phase, stage) corners.
        field_a = (1.0 - card_blend) * combined_field(phase_idx, stage_idx) + (
            card_blend * combined_field(phase_idx, next_stage_idx)
        )
        field_b = (1.0 - card_blend) * combined_field(next_phase_idx, stage_idx) + (
            card_blend * combined_field(next_phase_idx, next_stage_idx)
        )
        field_arr = (1.0 - resp_blend) * field_a + resp_blend * field_b
        field_img = image_tools.convert_array_to_image_of_vectors(
            field_arr, ptype=itk.D, reference_image=deformation_grid
        )
        field_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
        field_transform.SetDisplacementField(field_img)

        # -1000 HU is air, the value a CT grid samples outside itself.
        warped_ct = transform_tools.transform_image(
            patient_image, field_transform, patient_image, "linear", -1000.0
        )
        warped_labelmap = transform_tools.transform_image(
            chest_labelmap, field_transform, patient_image, "nearest"
        )

        ct_file = output_dir / f"combined_ct_{frame_idx:03d}.mha"
        labelmap_file = output_dir / f"combined_labelmap_{frame_idx:03d}.mha"
        itk.imwrite(warped_ct, str(ct_file), compression=True)
        itk.imwrite(
            to_signed_short(warped_labelmap), str(labelmap_file), compression=True
        )
        ct_files.append(ct_file)
        labelmap_files.append(labelmap_file)

        # Keep only the two phases bracketing the current frame in the field cache.
        for key in [k for k in field_cache if k[0] not in (phase_idx, next_phase_idx)]:
            del field_cache[key]

    del resp_points
    logger.info(
        "Wrote %d combined-motion surfaces, CT and labelmap volumes to %s",
        len(combined_files),
        output_dir,
    )

    # Assemble the ordered frames into a single animated 4D USD, split by anatomy
    # label (via the mesh `boundary_labels`) so each organ becomes its own prim
    # under /World/heart_and_lung_motion/{group}/{label}. enhance_meshes then
    # binds the per-organ OmniSurface materials (colors, subsurface scatter, ...).
    # The segmenter is the one Stage 0 wrote the labelmap with, so its taxonomy
    # names the ids the surface carries.
    converter = ConvertVTKToUSD(
        "heart_and_lung_motion",
        usd_frames,
        segmenter.taxonomy.all_labels(),
        segmenter=segmenter,
        frames_per_second=frames_per_second,
        log_level=log_level,
    )
    usd_file = output_dir / "heart_and_lung_motion.usd"
    stage = converter.convert(str(usd_file))
    USDAnatomyTools(stage).enhance_meshes(segmenter)
    stage.Save()
    logger.info("Wrote 4D USD with anatomy materials: %s", usd_file)

    # Testing: the first combined frame, as geometry and as the labelmap the
    # same frame rasterizes to.
    class_name = "tutorial_13_heart_and_lung_motion"
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=tutorials_dir.parent / "tests" / "baselines" / class_name,
        log_level=log_level,
    )
    screenshots = [
        tt.save_screenshot_mesh(
            cast(pv.DataSet, pv.read(str(combined_files[0]))),
            "combined_motion_surface.png",
            camera_position="iso",
            color="lightcoral",
        ),
        tt.save_screenshot_image_slice(
            itk.imread(str(labelmap_files[0])),
            "combined_labelmap.png",
            axis=0,
            slice_fraction=0.5,
            colormap="viridis",
        ),
    ]

    tutorial_results = {
        "respiratory_stage_count": len(respiratory_stages),
        "cardiac_stage_count": len(cardiac_stages),
        "combined_surfaces": combined_files,
        "combined_ct_volumes": ct_files,
        "combined_labelmap_volumes": labelmap_files,
        "breathing_lungs_usd": str(output_dir / "breathing_lungs.usd"),
        "beating_heart_usd": str(output_dir / "beating_heart.usd"),
        "usd_file": str(usd_file),
        "screenshots": screenshots,
    }
