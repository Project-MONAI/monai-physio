"""Prepare lung SSM inputs on a downloaded MeshGraphNet checkpoint domain.

The lung motion checkpoint is defined on the PCA template, vertex ordering,
and mesh graph saved beside its weights.  A statistical model regenerated from
the source cohort is scientifically related but is not the same discrete
domain.  This module projects a fitted Tutorial 8 reference surface onto the
checkpoint PCA model and propagates that compatible surface with Tutorial 8's
already-computed respiratory transforms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import itk
import pyvista as pv

from monai_physio import TransformTools, WorkflowFitStatisticalModelToPatient


@dataclass(frozen=True)
class CheckpointCompatibleLungCase:
    """Paths for one case expressed on the checkpoint's mesh topology."""

    reference_mesh: Path
    pca_coefficients: Path
    phase_meshes: list[Path]


def _read_checkpoint_domain(
    model_dir: Path,
) -> tuple[pv.DataSet, dict[str, Any], int, int]:
    """Load and cross-check the checkpoint template, PCA model, and metadata."""
    template_file = model_dir / "pca_mean_template.vtp"
    pca_model_file = model_dir / "pca_model.json"
    metadata_file = model_dir / "mgn_stage_model_metadata.json"
    for required_file in (template_file, pca_model_file, metadata_file):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Checkpoint domain asset not found: {required_file}"
            )

    template = cast(pv.DataSet, pv.read(str(template_file)))
    with pca_model_file.open(encoding="utf-8") as file_handle:
        pca_model = cast(dict[str, Any], json.load(file_handle))
    with metadata_file.open(encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)

    expected_points = int(metadata["n_mesh_points"])
    expected_coefficients = len(metadata["pca_mean"])
    input_coefficients = sum(
        feature.startswith("pca_c") for feature in metadata["input_features"]
    )
    component_width = len(pca_model["components"][0])
    if template.n_points != expected_points:
        raise ValueError(
            f"Checkpoint template has {template.n_points} points, but metadata "
            f"declares {expected_points}."
        )
    if component_width != 3 * expected_points:
        raise ValueError(
            f"Checkpoint PCA components have width {component_width}, expected "
            f"{3 * expected_points} for {expected_points} template points."
        )
    if not (
        expected_coefficients
        == input_coefficients
        == len(pca_model["eigenvalues"])
        == len(pca_model["components"])
    ):
        raise ValueError(
            "Checkpoint metadata and PCA model disagree on the number of "
            "shape coefficients."
        )
    return template, pca_model, expected_points, expected_coefficients


def _output_is_complete(
    provenance_file: Path,
    output_files: list[Path],
    expected_points: int,
    expected_coefficients: int,
    provenance: dict[str, Any],
) -> bool:
    """Return whether every cached artifact belongs to this checkpoint domain."""
    if not provenance_file.exists() or not all(path.exists() for path in output_files):
        return False
    try:
        cached_provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        if cached_provenance != provenance:
            return False
        meshes_valid = all(
            cast(pv.DataSet, pv.read(str(path))).n_points == expected_points
            for path in output_files
            if path.suffix == ".vtp"
        )
        coefficient_files = [path for path in output_files if path.suffix == ".json"]
        coefficients_valid = all(
            len(json.loads(path.read_text(encoding="utf-8"))) == expected_coefficients
            for path in coefficient_files
        )
        return meshes_valid and coefficients_valid
    except (KeyError, OSError, ValueError):
        return False


def prepare_checkpoint_compatible_lung_case(
    source_case_dir: Path,
    model_dir: Path,
    output_dir: Path,
    mask_dilation_mm: float,
    distancemap_squared_max: float,
    icon_weights_path: Optional[Path] = None,
    log_level: int | str = logging.INFO,
) -> CheckpointCompatibleLungCase:
    """Project one Tutorial 8 case onto the checkpoint PCA and graph domain.

    The source case supplies its fitted reference anatomy and the respiratory
    transforms computed by Tutorial 8.  The checkpoint supplies the PCA model
    and exact template topology.  Existing compatible outputs are reused only
    when every required artifact and its provenance are present.

    Parameters
    ----------
    source_case_dir
        Tutorial 8 output directory for one DIR-Lab case.
    model_dir
        Directory containing the MGN checkpoint and domain assets.
    output_dir
        Directory for checkpoint-compatible reference and phases.
    mask_dilation_mm
        Registration-mask dilation in millimeters.
    distancemap_squared_max
        Distance-map saturation in squared millimeters.
    icon_weights_path
        Optional finetuned distance-map ICON weights.
    log_level
        Logging level.

    Returns
    -------
    CheckpointCompatibleLungCase
        Paths to the compatible reference mesh, PCA coefficients, and phases.
    """
    source_case_dir = Path(source_case_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    case_id = source_case_dir.name
    logger = logging.getLogger("lung_mgn_checkpoint_tools")
    logger.setLevel(log_level)

    source_reference = source_case_dir / f"{case_id}_ssm_surface.vtp"
    transform_files = sorted(source_case_dir.glob(f"{case_id}_T??_forward_tfm.hdf"))
    if not source_reference.exists():
        raise FileNotFoundError(
            f"Tutorial 8 fitted reference not found: {source_reference}"
        )
    if not transform_files:
        raise FileNotFoundError(
            f"No Tutorial 8 respiratory transforms found in {source_case_dir}"
        )

    template, pca_model, expected_points, expected_coefficients = (
        _read_checkpoint_domain(model_dir)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_mesh = output_dir / f"{case_id}_ssm_surface.vtp"
    pca_coefficients = output_dir / f"{case_id}_ssm_pca_coefficients.json"
    phase_meshes = [
        output_dir
        / f"{transform_file.stem.removesuffix('_forward_tfm')}_ssm_surface.vtp"
        for transform_file in transform_files
    ]
    provenance_file = output_dir / "checkpoint_domain.json"
    provenance = {
        "model_dir": str(model_dir.resolve()),
        "expected_points": expected_points,
        "pca_components": expected_coefficients,
        "template_size": (model_dir / "pca_mean_template.vtp").stat().st_size,
        "pca_model_size": (model_dir / "pca_model.json").stat().st_size,
        "source_reference": str(source_reference.resolve()),
        "source_reference_size": source_reference.stat().st_size,
        "source_reference_modified_ns": source_reference.stat().st_mtime_ns,
        "transforms": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "modified_ns": path.stat().st_mtime_ns,
            }
            for path in transform_files
        ],
    }
    required_outputs = [reference_mesh, pca_coefficients, *phase_meshes]
    if _output_is_complete(
        provenance_file,
        required_outputs,
        expected_points,
        expected_coefficients,
        provenance,
    ):
        logger.info("Reusing checkpoint-compatible lung fit in %s", output_dir)
        return CheckpointCompatibleLungCase(
            reference_mesh=reference_mesh,
            pca_coefficients=pca_coefficients,
            phase_meshes=phase_meshes,
        )

    logger.info(
        "Refitting %s from %d to the checkpoint's %d-point PCA domain",
        case_id,
        cast(pv.DataSet, pv.read(str(source_reference))).n_points,
        expected_points,
    )
    patient_surface = cast(pv.DataSet, pv.read(str(source_reference)))
    fit_workflow = WorkflowFitStatisticalModelToPatient(
        template_model=template,
        patient_models=[patient_surface],
        log_level=log_level,
    )
    fit_workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model=pca_model,
        number_of_pca_components=expected_coefficients,
        use_surface=False,
    )
    fit_workflow.set_mask_dilation_mm(mask_dilation_mm)
    fit_workflow.set_distancemap_squared_max(distancemap_squared_max)
    if icon_weights_path is not None and icon_weights_path.exists():
        fit_workflow.set_labelmap_to_labelmap_icon_weights_path(str(icon_weights_path))
    fit_result = fit_workflow.process()

    coefficients = fit_workflow.pca_coefficients
    if coefficients is None:
        raise RuntimeError("Checkpoint-domain PCA fit produced no coefficients.")
    pca_coefficients.write_text(json.dumps(coefficients.tolist()), encoding="utf-8")

    fitted_reference = cast(
        pv.PolyData, fit_result["registered_template_model_surface"]
    )
    if fitted_reference.n_points != expected_points:
        raise ValueError(
            f"Checkpoint-domain fit produced {fitted_reference.n_points} points, "
            f"expected {expected_points}."
        )
    fitted_reference.save(str(reference_mesh))

    transform_tools = TransformTools(log_level=log_level)
    for index, (transform_file, phase_mesh) in enumerate(
        zip(transform_files, phase_meshes, strict=True), start=1
    ):
        logger.info(
            "Propagating checkpoint topology: %d/%d (%s)",
            index,
            len(transform_files),
            transform_file.stem,
        )
        transform = itk.transformread(str(transform_file))
        phase = transform_tools.transform_pvcontour(
            fitted_reference,
            transform,
            with_deformation_magnitude=True,
        )
        phase.save(str(phase_mesh))

    provenance_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return CheckpointCompatibleLungCase(
        reference_mesh=reference_mesh,
        pca_coefficients=pca_coefficients,
        phase_meshes=phase_meshes,
    )
