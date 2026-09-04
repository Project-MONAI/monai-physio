"""Apply the trained PhysicsNeMo MeshGraphNet cardiac motion model to Case1Pack.

Purpose
-------
Drive the cardiac MeshGraphNet (MGN) motion model trained by
``tutorial_09_byod_train_physicsnemo_mgn.py`` over the DirLab ``Case1Pack``
heart and, for each requested cardiac stage, rasterize the inferred per-vertex
displacement onto the ``Case1Pack`` image grid as a deformation field via
:meth:`WorkflowInferMovement.create_deformation_field`.

Inputs
------
- MGN model: the epoch-300 checkpoint in ``output/tutorial_09_byod_mgn_3``.
  That run directory holds only intermittent epoch checkpoints (training did
  not finalize), so the finalize-time inference assets are assembled here from
  the self-describing epoch checkpoint plus the source PCA template.
- Reference image (defines the deformation-field grid):
  ``data/DirLab-4DCT/Case1Pack_T70.mha``.
- Patient heart shape: the 15-mode ``pca-vol-kcl`` fit produced by re-running
  Tutorial 7 against Case1Pack:
  * coefficients: ``output/tutorial_07_heart/
    tutorial_07_heart_registered_coefficients.json``
  * patient-space mesh (binning positions): ``output/tutorial_07_heart/
    tutorial_07_heart_template_mesh_registered.vtu``

Coordinate-frame note
---------------------
The MGN was trained in the ``pca-vol-kcl`` basis, so its PCA reconstruction of a
subject lands in the model's *canonical* frame (near the origin). The Tutorial 7
fit aligned that model to Case1Pack with a pose transform that the shape
coefficients do not carry, so the patient heart sits in the scanner frame of
``Case1Pack_T70``. The displacements the network predicts depend only on the
coefficients and the stage, not on where the surface sits; therefore this script
passes the patient-space registered *mesh* as ``reference_surface`` so the
displacements are binned at positions that actually overlap the reference image.
The registered volume mesh is passed (not the surface ``.vtp``) because its
extracted surface reproduces the exact mean-shape point ordering the network
outputs, keeping the displacement/position correspondence intact.

Caveat: this assumes the network's displacement vectors and the Case1Pack image
share a common anatomical (LPS) orientation, which holds for supine cardiac/lung
CT but is not re-derived here.

Outputs (under ``output/temp_beating_heart``)
--------------------------------------------
Per stage ``s`` (percent of the RR interval):
- ``deformation_field_s<sss>.mha``    - ITK vector image of ``(dx, dy, dz)`` mm
- ``surface_normal_field_s<sss>.mha`` - ITK vector image of reference normals
- ``deformed_heart_surface_s<sss>.vtp`` - the patient-space heart surface at
  that stage (reference surface displaced by the network)

The per-stage surfaces are then assembled, in stage order, into a single
animated 4D ``beating_heart.usd`` with a heart anatomy material via
:class:`WorkflowConvertVTKToUSD`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import itk
import numpy as np
import pyvista as pv

from monai_physio import (
    WorkflowConvertVTKToUSD,
    WorkflowInferMovement,
    WorkflowInferPhysicsNeMo,
)
from monai_physio import physicsnemo_tools as pnt


def _ensure_mgn_inference_assets(
    model_dir: Path, epoch: int, pca_mean_volume: Path
) -> None:
    """Complete an interrupted MGN run directory so it can be loaded for inference.

    ``WorkflowInferPhysicsNeMo`` expects a finalized run directory
    (``mgn_stage_model.pt``, ``pca_mean_template.vtp`` and the shared graph
    tensors). The ``tutorial_09_byod_mgn_3`` directory holds only epoch
    checkpoints, so this regenerates the missing assets deterministically:

    - ``mgn_stage_model.pt`` from the self-describing epoch checkpoint (it
      carries the normalization stats and architecture the loader reads);
    - ``pca_mean_template.vtp`` and the shared MGN graph tensors from the PCA
      template volume, using the same steps the trainer used.

    All writes are idempotent (skipped when the target already exists).
    """
    import torch

    epoch_ckpt = model_dir / f"mgn_stage_model_epoch_{epoch:05d}.pt"
    if not epoch_ckpt.exists():
        raise FileNotFoundError(f"Epoch checkpoint not found: {epoch_ckpt}")

    final_ckpt = model_dir / "mgn_stage_model.pt"
    if not final_ckpt.exists():
        shutil.copy2(epoch_ckpt, final_ckpt)

    surface_file = model_dir / "pca_mean_template.vtp"
    if not surface_file.exists():
        volume = pv.read(str(pca_mean_volume))
        mean_surface = volume.extract_surface(algorithm="dataset_surface")
        mean_surface.save(str(surface_file))
    mean_surface = pv.read(str(surface_file))

    edge_index_file = model_dir / "shared_edge_index.pt"
    edge_feats_file = model_dir / "shared_edge_features.pt"
    if not edge_index_file.exists() or not edge_feats_file.exists():
        edge_index = pnt.mesh_to_edge_index(mean_surface)
        coords = np.asarray(mean_surface.points, dtype=np.float32)
        edge_feats = pnt.compute_edge_features(coords, edge_index)
        torch.save(edge_index, str(edge_index_file))
        torch.save(edge_feats, str(edge_feats_file))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    # MGN model produced by tutorial_09_byod_train_physicsnemo_mgn.py.
    model_dir = tutorials_dir / "output" / "tutorial_09_byod_mgn_3"
    epoch = 300
    # Source PCA template used to train the model (regenerates missing assets).
    pca_mean_volume = Path("D:/MONAI-Physio/kcl-heart-pca/pca-vol-kcl/pca_mean.vtu")

    # Case1Pack reference image (defines the deformation-field grid).
    reference_image_file = repo_root / "data" / "DirLab-4DCT" / "Case1Pack_T70.mha"

    # 15-mode pca-vol-kcl fit of the Case1Pack heart (Tutorial 7, re-run).
    tutorial_07_dir = tutorials_dir / "output" / "tutorial_07_heart"
    coefficients_file = (
        tutorial_07_dir / "tutorial_07_heart_registered_coefficients.json"
    )
    registered_mesh_file = (
        tutorial_07_dir / "tutorial_07_heart_template_mesh_registered.vtu"
    )

    output_dir = tutorials_dir / "output" / "temp_beating_heart"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cardiac stages (fraction of the RR interval) to render as a beating heart.
    stages = [round(0.1 * k, 2) for k in range(10)]

    for required in (
        reference_image_file,
        coefficients_file,
        registered_mesh_file,
        pca_mean_volume,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")

    _ensure_mgn_inference_assets(model_dir, epoch, pca_mean_volume)

    reference_image = itk.imread(str(reference_image_file))

    infer = WorkflowInferMovement(
        WorkflowInferPhysicsNeMo(model_directory=model_dir, epoch=epoch)
    )

    deformation_files: list[Path] = []
    normal_files: list[Path] = []
    surface_files: list[Path] = []
    deformed_surfaces: list[pv.PolyData] = []
    for stage in stages:
        result = infer.create_deformation_field(
            shape_parameters=coefficients_file,
            stage=float(stage),
            reference_image=reference_image,
            fitted_reference_mesh=registered_mesh_file,
        )
        pct = int(round(stage * 100))
        field_path = output_dir / f"deformation_field_s{pct:03d}.mha"
        normal_path = output_dir / f"surface_normal_field_s{pct:03d}.mha"
        surface_path = output_dir / f"deformed_heart_surface_s{pct:03d}.vtp"
        itk.imwrite(result["deformation_field"], str(field_path), compression=True)
        itk.imwrite(result["normal_image"], str(normal_path), compression=True)
        result["deformed_surface"].save(str(surface_path))
        deformation_files.append(field_path)
        normal_files.append(normal_path)
        surface_files.append(surface_path)
        deformed_surfaces.append(result["deformed_surface"])
        logging.info("stage %.2f -> %s", stage, field_path.name)

    # Assemble the ordered per-stage surfaces into one animated 4D USD. The
    # stages loop over one RR interval, so play them back at one beat per second.
    usd_workflow = WorkflowConvertVTKToUSD(
        input_meshes=deformed_surfaces,
        usd_project_name="beating_heart",
        output_directory=output_dir,
        appearance="anatomy",
        anatomy_type="heart",
        separate_by_connectivity=True,
        frames_per_second=float(len(stages)),
        log_level=logging.INFO,
    )
    usd_result = usd_workflow.process()

    tutorial_results = {
        "deformation_fields": deformation_files,
        "normal_fields": normal_files,
        "deformed_surfaces": surface_files,
        "usd_file": usd_result["usd_file"],
    }
