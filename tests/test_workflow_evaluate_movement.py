"""Per-structure scoring of an inferred motion model.

The metric definitions are checked against volumes whose overlap, volume and
separation are known by construction, then the whole workflow is run once on a
tiny synthetic case so the report and the CSV are exercised end to end --- that
is where a mis-indexed stage or a missing provenance field would hide.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import itk
import numpy as np
import pytest
import pyvista as pv

from monai_physio import MovementGroundTruth, WorkflowEvaluateMovement
from monai_physio.monai_physio_base import MONAIPhysioBase

_SPACING_MM = 1.0
_GRID_SIZE = 40
_RADIUS_MM = 10.0


def _sphere() -> pv.PolyData:
    """Small sphere shared by the template, reference and phase meshes."""
    return pv.Sphere(radius=_RADIUS_MM, theta_resolution=8, phi_resolution=8)


def _ball_labelmap(radius_mm: float, label: int = 1) -> itk.Image:
    """A centered ball of ``label`` on a grid whose origin puts it at the middle."""
    axis = (np.arange(_GRID_SIZE) - (_GRID_SIZE - 1) / 2.0) * _SPACING_MM
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    array = np.where(x**2 + y**2 + z**2 <= radius_mm**2, label, 0).astype(np.uint8)
    image = itk.GetImageFromArray(array)
    image.SetSpacing([_SPACING_MM] * 3)
    image.SetOrigin([-(_GRID_SIZE - 1) / 2.0 * _SPACING_MM] * 3)
    return image


def test_dice_of_a_half_overlap() -> None:
    """Dice is twice the intersection over the summed sizes."""
    truth = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
    predicted = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)

    assert WorkflowEvaluateMovement.dice(truth, predicted, 1) == 0.5


def test_dice_is_not_a_number_when_neither_volume_has_the_label() -> None:
    """A label neither volume contains has no overlap to report, not a zero one."""
    empty = np.zeros(8, dtype=np.uint8)

    assert np.isnan(WorkflowEvaluateMovement.dice(empty, empty, 3))


def test_volume_counts_voxels_in_cubic_millimeters() -> None:
    """Volume is the label's voxel count times the voxel volume."""
    labels = np.array([2, 2, 2, 0, 1], dtype=np.uint8)

    assert WorkflowEvaluateMovement.volume_mm3(labels, 2, 0.5) == 1.5


def test_surface_rmse_of_two_concentric_spheres_is_their_radius_gap() -> None:
    """A surface offset by 1 mm everywhere scores 1 mm, from either direction."""
    inner = pv.Sphere(radius=10.0, theta_resolution=60, phi_resolution=60)
    outer = pv.Sphere(radius=11.0, theta_resolution=60, phi_resolution=60)

    assert WorkflowEvaluateMovement.surface_rmse_mm(inner, outer) == pytest.approx(
        1.0, abs=0.05
    )


def _evaluator(label_names: dict) -> WorkflowEvaluateMovement:
    """A workflow whose geometry helpers can be exercised without a network."""
    workflow = WorkflowEvaluateMovement.__new__(WorkflowEvaluateMovement)
    MONAIPhysioBase.__init__(
        workflow, class_name="WorkflowEvaluateMovement", log_level=logging.WARNING
    )
    workflow.label_names = label_names
    return workflow


def _labelled_mesh() -> pv.PolyData:
    """Two disjoint triangles, each tagged with a different structure id."""
    mesh = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([3, 0, 1, 2, 3, 3, 4, 5]),
    )
    mesh.cell_data["SegmentationLabelIds"] = np.array([28, 29], dtype=np.int32)
    return mesh


def test_a_mesh_that_names_its_structures_is_taken_at_its_word() -> None:
    """No inference beats the model's own per-cell ids."""
    workflow = _evaluator({28: "upper", 29: "lower"})

    indices = workflow._mesh_label_points(_labelled_mesh(), [28, 29])

    assert indices is not None
    assert indices[28].tolist() == [0, 1, 2]
    assert indices[29].tolist() == [3, 4, 5]


def test_a_mesh_without_the_ids_falls_back() -> None:
    """The heart's single surface names nothing, so the caller must infer."""
    workflow = _evaluator({1: "heart"})
    bare = pv.Sphere(radius=5.0)

    assert workflow._mesh_label_points(bare, [1]) is None
    # Carrying ids that are not the ones being scored is the same situation.
    assert workflow._mesh_label_points(_labelled_mesh(), [1, 2]) is None


def test_a_point_shared_by_two_structures_is_counted_once() -> None:
    """Two triangles meeting at a fissure must not both claim the shared points."""
    workflow = _evaluator({28: "upper", 29: "lower"})
    mesh = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([3, 0, 1, 2, 3, 1, 2, 3]),
    )
    mesh.cell_data["SegmentationLabelIds"] = np.array([28, 29], dtype=np.int32)

    indices = workflow._mesh_label_points(mesh, [28, 29])

    assert indices is not None
    # Points 1 and 2 touch both triangles; the lower id keeps them.
    assert indices[28].tolist() == [0, 1, 2]
    assert indices[29].tolist() == [3]


def test_every_mesh_point_goes_to_the_structure_it_is_nearest() -> None:
    """The fallback: the nearest-label partition the labelmap metrics use."""
    workflow = _evaluator({1: "left", 2: "right"})
    # Two spheres 100 mm apart, and a mesh straddling them: the points around
    # x = -50 belong to one, those around x = +50 to the other.
    surfaces = {
        1: pv.Sphere(radius=5.0, center=(-50.0, 0.0, 0.0)),
        2: pv.Sphere(radius=5.0, center=(50.0, 0.0, 0.0)),
    }
    mesh = pv.PolyData(
        np.array(
            [
                [-50.0, 0.0, 0.0],
                [-40.0, 1.0, 0.0],
                [40.0, 0.0, 1.0],
                [50.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
            ]
        )
    )

    indices = workflow._nearest_label_points(mesh, surfaces)

    assert indices[1].tolist() == [0, 1]
    assert indices[2].tolist() == [2, 3, 4]
    # A partition: every point claimed once, none left over.
    assert sorted(np.concatenate(list(indices.values())).tolist()) == list(range(5))


def test_a_structure_no_point_is_nearest_to_gets_no_points() -> None:
    """It is scored on nothing, which the caller turns into nan rather than 0."""
    workflow = _evaluator({1: "near", 2: "far"})
    surfaces = {
        1: pv.Sphere(radius=5.0, center=(0.0, 0.0, 0.0)),
        2: pv.Sphere(radius=5.0, center=(500.0, 0.0, 0.0)),
    }
    mesh = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    indices = workflow._nearest_label_points(mesh, surfaces)

    assert indices[1].tolist() == [0, 1]
    assert indices[2].size == 0


def test_pooled_displacement_error_pools_the_points_not_the_stages() -> None:
    """A percentile of the whole, which a percentile of percentiles is not."""
    # Stage 0 holds the 90 smallest errors, stage 1 the 10 largest, so the 95th
    # percentile of the pooled 100 lies inside the second stage and neither
    # stage's own 95th percentile is anywhere near it.
    stage_errors = [
        np.arange(90, dtype=np.float32),
        np.arange(90, 100, dtype=np.float32),
    ]

    pooled = WorkflowEvaluateMovement.pool_displacement_error(stage_errors)

    assert pooled["p95_mm"] == pytest.approx(
        float(np.percentile(np.arange(100, dtype=np.float32), 95.0))
    )
    assert pooled["max_mm"] == 99.0
    assert pooled["rms_mm"] == pytest.approx(
        float(np.sqrt(np.mean(np.arange(100, dtype=np.float64) ** 2)))
    )


def test_pooled_displacement_error_is_not_a_number_without_stages() -> None:
    """No stage was scored point by point, so there is no error to report."""
    pooled = WorkflowEvaluateMovement.pool_displacement_error([])

    assert np.isnan(pooled["rms_mm"])
    assert np.isnan(pooled["p95_mm"])
    assert np.isnan(pooled["max_mm"])


def test_displacement_rows_carry_both_displacements_and_their_error() -> None:
    """Each row is the point's own predicted and true displacement, and the gap."""
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    predicted = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 6.0]])
    true = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 5.0]])
    errors = np.linalg.norm(predicted - true, axis=1)

    rows = WorkflowEvaluateMovement._displacement_rows(
        "subject", 0.5, reference, predicted, true, errors
    )

    assert len(rows) == 2
    # Point 0 was predicted to move 1 mm in x and did not move at all: the
    # predicted displacement, the true displacement and the error all say so.
    assert rows[0] == [
        "subject",
        0.5,
        0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert rows[1][-4:] == [0.0, 0.0, 2.0, 1.0]


def test_the_fitted_reference_mesh_cannot_be_omitted() -> None:
    """A PCA-only reconstruction must never stand in for the patient's fit."""
    import inspect

    from monai_physio import WorkflowInferMovement

    for name in ("predict_single", "process_time_series", "create_deformation_field"):
        parameter = inspect.signature(getattr(WorkflowInferMovement, name)).parameters[
            "fitted_reference_mesh"
        ]
        assert parameter.default is inspect.Parameter.empty, name


def _trained_model_directory(tmp_path: Path) -> Path:
    """Train a two-epoch MeshGraphNet on two synthetic subjects."""
    from monai_physio import TrainPhysicsNeMoMGN, WorkflowTrainPhysicsNeMo

    template = _sphere()
    template_dir = tmp_path / "template"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_file = template_dir / "pca_mean_surface.vtp"
    template.save(str(template_file))
    (template_dir / "pca_model.json").write_text(
        json.dumps(
            {
                "mean": np.asarray(template.points, dtype=np.float64).ravel().tolist(),
                "components": np.zeros((2, template.n_points * 3)).tolist(),
            }
        ),
        encoding="utf-8",
    )

    manifests = []
    for index, offset in enumerate((0.5, -0.5)):
        subject_dir = tmp_path / f"subject_{index:02d}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        _sphere().save(str(subject_dir / "reference.vtp"))
        (subject_dir / "coefficients.json").write_text(
            json.dumps([offset, -offset]), encoding="utf-8"
        )
        phases = []
        for phase_index, stage in enumerate((0.0, 1.0)):
            phase_mesh = _sphere()
            phase_mesh.point_data["displacement"] = np.full(
                (phase_mesh.n_points, 3), offset * stage, dtype=np.float32
            )
            phase_file = subject_dir / f"phase_{phase_index}.vtp"
            phase_mesh.save(str(phase_file))
            phases.append({"mesh": str(phase_file), "stage": stage})
        manifest_file = subject_dir / "manifest.json"
        manifest_file.write_text(
            json.dumps(
                {
                    "subject_id": f"subject_{index:02d}",
                    "fitted_reference_mesh": str(subject_dir / "reference.vtp"),
                    "pca_coefficients": str(subject_dir / "coefficients.json"),
                    "target_array": "displacement",
                    "phases": phases,
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest_file)

    method = TrainPhysicsNeMoMGN()
    method.set_epochs(2)
    method.set_batch_size(1)
    method.set_processor_size(1)
    method.set_hidden_dim(8)
    method.set_num_layers(1)

    model_directory = tmp_path / "weights"
    WorkflowTrainPhysicsNeMo(
        train_manifests=manifests,
        val_manifests=[],
        pca_mean_mesh=template_file,
        output_directory=model_directory,
        training_method=method,
    ).process()
    return model_directory


def test_every_stage_and_structure_reaches_the_report(tmp_path: Path) -> None:
    """One row per stage and structure, with the run's provenance on it."""
    pytest.importorskip("torch")
    pytest.importorskip("physicsnemo")
    pytest.importorskip("torch_geometric")

    from monai_physio import WorkflowInferMovement, WorkflowInferPhysicsNeMo

    model_directory = _trained_model_directory(tmp_path)

    case_dir = tmp_path / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    fitted_reference_mesh_file = case_dir / "reference.vtp"
    _sphere().save(str(fitted_reference_mesh_file))
    shape_parameters = case_dir / "coefficients.json"
    shape_parameters.write_text(json.dumps([0.25, -0.25]), encoding="utf-8")

    reference_labelmap = _ball_labelmap(_RADIUS_MM)
    ground_truth = {0.0: _ball_labelmap(_RADIUS_MM), 1.0: _ball_labelmap(9.0)}

    # The true surface of each stage, sharing the reference mesh's topology:
    # the sphere itself at stage 0, shrunk by a tenth at stage 1, matching what
    # the two ground-truth labelmaps say.
    ground_truth_meshes = {}
    for stage, radius in ((0.0, _RADIUS_MM), (1.0, 9.0)):
        stage_mesh = _sphere()
        stage_mesh.points = np.asarray(stage_mesh.points) * (radius / _RADIUS_MM)
        stage_file = case_dir / f"truth_{int(stage * 100):03d}.vtp"
        stage_mesh.save(str(stage_file))
        ground_truth_meshes[stage] = stage_file

    output_directory = tmp_path / "evaluation"
    workflow = WorkflowEvaluateMovement(
        movement_workflow=WorkflowInferMovement(
            WorkflowInferPhysicsNeMo(model_directory=model_directory)
        ),
        label_names={1: "ball"},
    )

    # A stage with a labelmap but no fitted surface has nothing to measure a
    # per-point displacement against.
    with pytest.raises(ValueError, match="fitted surface for every stage"):
        workflow.process(
            case_id="synthetic_case",
            shape_parameters=shape_parameters,
            fitted_reference_mesh=fitted_reference_mesh_file,
            ground_truth=MovementGroundTruth(
                labelmaps=ground_truth,
                reference_labelmap=reference_labelmap,
                reference_stage=0.0,
                meshes={0.0: ground_truth_meshes[0.0]},
            ),
            output_directory=output_directory,
            include_displacement_error=True,
        )

    result = workflow.process(
        case_id="synthetic_case",
        shape_parameters=shape_parameters,
        fitted_reference_mesh=fitted_reference_mesh_file,
        ground_truth=MovementGroundTruth(
            labelmaps=ground_truth,
            reference_labelmap=reference_labelmap,
            reference_stage=0.0,
            meshes=ground_truth_meshes,
        ),
        output_directory=output_directory,
        smoothing_sigma_mm=2.0,
        evaluation_spacing_mm=_SPACING_MM,
        report_displacement_data=True,
        include_predicted_displacements=True,
        include_true_displacements=True,
        include_displacement_error=True,
    )

    # Two stages, one structure.
    assert len(result["rows"]) == 2
    assert len(result["predicted_surfaces"]) == 2
    assert len(result["warped_labelmaps"]) == 2

    with Path(result["csv_file"]).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {
        "case_id",
        "stage",
        "label_id",
        "label_name",
        "dice",
        "volume_truth_mm3",
        "volume_predicted_mm3",
        "volume_difference_mm3",
        "volume_difference_percent",
        "surface_rmse_mm",
        "shape_parameters_file",
        "pca_c01",
        "pca_c02",
        "network_weights_file",
        "network_weights_created",
        "network_weights_modified",
        "network_epoch",
        "displacement_rms_mm",
        "displacement_95th_mm",
        "displacement_max_mm",
    } <= set(rows[0])
    assert all(0.0 <= float(row["dice"]) <= 1.0 for row in rows)
    # One structure owns every mesh point here, so its per-structure figures are
    # the whole-model ones; an indexing slip would break that.
    assert [float(row["displacement_max_mm"]) for row in rows] == [
        pytest.approx(float(entry["max_error_mm"]))
        for entry in result["displacement_statistics"]
    ]
    assert max(float(row["displacement_95th_mm"]) for row in rows) == pytest.approx(
        max(float(entry["p95_error_mm"]) for entry in result["displacement_statistics"])
    )

    report = Path(result["report_file"]).read_text(encoding="utf-8")
    assert "synthetic_case" in report
    assert str(model_directory) in report
    assert "ball" in report
    assert Path(result["volume_plot_file"]).name in report
    assert Path(result["volume_plot_file"]).exists()
    # Mean alone would hide the stage a structure is predicted worst at.
    assert "Worst case, over every structure and stage" in report
    assert "Dice (min)" in report
    assert "## Displacement error" in report
    assert "Displacement 95th (mean mm)" in report
    assert "Worst Displacement 95th percentile" in report

    # The per-point CSV, the mesh arrays and the error statistics all describe
    # the same displacements, so they have to agree.
    n_points = _sphere().n_points
    with Path(result["displacement_data_file"]).open(
        newline="", encoding="utf-8"
    ) as fh:
        displacement_rows = list(csv.DictReader(fh))
    assert len(displacement_rows) == 2 * n_points
    assert {
        "subject_id",
        "stage",
        "point_id",
        "fitted_reference_x_mm",
        "predicted_dx_mm",
        "true_dx_mm",
        "error_mm",
    } <= set(displacement_rows[0])

    predicted_surface = pv.read(str(result["predicted_surfaces"][0]))
    predicted = np.asarray(predicted_surface["predicted_displacement_mm"])
    true = np.asarray(predicted_surface["true_displacement_mm"])
    error = np.asarray(predicted_surface["displacement_error_mm"])
    assert predicted.shape == (n_points, 3)
    assert true.shape == (n_points, 3)
    assert error.shape == (n_points,)
    assert error == pytest.approx(np.linalg.norm(predicted - true, axis=1), abs=1e-4)

    # The first stage's rows describe the first stage's mesh.
    first_stage = [row for row in displacement_rows if float(row["stage"]) == 0.0]
    assert [float(row["error_mm"]) for row in first_stage] == pytest.approx(
        error, abs=1e-4
    )

    statistics = result["displacement_statistics"]
    assert len(statistics) == 2
    assert statistics[0]["max_error_mm"] == pytest.approx(float(error.max()), abs=1e-4)
    assert result["displacement_max_mm"] == max(
        float(row["max_error_mm"]) for row in statistics
    )
    assert result["displacement_rms_mm"] > 0.0
    # Pooled over both stages, so neither can exceed the worst single point.
    assert result["displacement_95th_mm"] <= result["displacement_max_mm"]
    assert result["displacement_rms_mm"] <= result["displacement_max_mm"]

    # The same case with none of the displacement options asked for. The true
    # surfaces are still on hand, so nothing stops the error being measured ---
    # but the pooled figures are only reported when the error was asked for, so
    # measuring it here would put per-stage rows in the report beside a pooled
    # RMS of nan.
    plain = workflow.process(
        case_id="synthetic_case",
        shape_parameters=shape_parameters,
        fitted_reference_mesh=fitted_reference_mesh_file,
        ground_truth=MovementGroundTruth(
            labelmaps=ground_truth,
            reference_labelmap=reference_labelmap,
            reference_stage=0.0,
            meshes=ground_truth_meshes,
        ),
        output_directory=tmp_path / "evaluation_plain",
        smoothing_sigma_mm=2.0,
        evaluation_spacing_mm=_SPACING_MM,
    )
    assert plain["displacement_statistics"] == []
    assert np.isnan(plain["displacement_rms_mm"])
    assert plain["displacement_data_file"] is None
    assert "Displacement error" not in Path(plain["report_file"]).read_text(
        encoding="utf-8"
    )
    # The labelmap metrics do not depend on any of it.
    assert len(plain["rows"]) == 2
