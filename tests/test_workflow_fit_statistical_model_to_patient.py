"""Synthetic tests for model-to-patient workflow helpers."""

from __future__ import annotations

import inspect
from typing import Any

import itk
import numpy as np
import pytest
import pyvista as pv

from monai_physio.segment_heart_simpleware import SegmentHeartSimpleware
from monai_physio.segment_heart_simpleware_trimmed_branches import (
    SegmentHeartSimplewareTrimmedBranches,
)
from monai_physio.workflow_convert_image_to_vtk import WorkflowConvertImageToVTK
from monai_physio.workflow_fit_statistical_model_to_patient import (
    WorkflowFitStatisticalModelToPatient,
)


def test_transform_model_applies_staged_transform() -> None:
    """Transform helper updates mesh points with image shape (Z, Y, X) = (3, 3, 3)."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    model = pv.PolyData(points)
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=model,
        patient_models=[model],
        patient_image=image,
    )

    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetIdentity()
    transform.SetTranslation((1.0, 2.0, 3.0))
    workflow.icp_forward_point_transform = transform
    workflow.pca_coefficients = None
    workflow.use_l2l_registration = False
    workflow.use_l2i_registration = False

    output = workflow.transform_model()

    assert output is not None
    np.testing.assert_allclose(output.points, points + np.array([1.0, 2.0, 3.0]))


def test_fit_workflow_default_segmentation_method_is_trimmed_branches() -> None:
    """Default segmentation_method is None; see
    test_fit_workflow_routes_default_to_image_to_vtk_with_trimmed_branches for
    confirmation that the None default resolves to
    SegmentHeartSimplewareTrimmedBranches, matching the KCL-Heart-Model fit
    contract."""
    default = (
        inspect.signature(WorkflowFitStatisticalModelToPatient.__init__)
        .parameters["segmentation_method"]
        .default
    )
    assert default is None


def test_fit_workflow_routes_default_to_image_to_vtk_with_trimmed_branches(
    monkeypatch: Any,
) -> None:
    """When patient_models is omitted, the workflow must invoke
    WorkflowConvertImageToVTK with a SegmentHeartSimplewareTrimmedBranches
    instance."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    template = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    heart_surface = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )

    captured: dict[str, Any] = {}

    class _FakeConvertImageToVTK:
        def __init__(self, **kwargs: Any) -> None:
            captured["init_kwargs"] = kwargs

        def process(self, **kwargs: Any) -> dict[str, Any]:
            captured["process_kwargs"] = kwargs
            return {"surfaces": {"heart": heart_surface}}

    monkeypatch.setattr(
        "monai_physio.workflow_fit_statistical_model_to_patient."
        "WorkflowConvertImageToVTK",
        _FakeConvertImageToVTK,
    )

    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=template,
        patient_image=image,
    )

    assert isinstance(
        captured["init_kwargs"]["segmentation_method"],
        SegmentHeartSimplewareTrimmedBranches,
    )
    assert captured["process_kwargs"]["anatomy_groups"] == ["heart"]
    assert workflow.patient_models == [heart_surface]


def test_image_to_vtk_segmenter_uses_supplied_instance() -> None:
    """WorkflowConvertImageToVTK must use whatever segmenter instance it's
    given as-is: a SegmentHeartSimplewareTrimmedBranches instance stays a
    SegmentHeartSimplewareTrimmedBranches, and a plain SegmentHeartSimpleware
    instance stays a SegmentHeartSimpleware (not upgraded to the trimmed
    subclass)."""
    trimmed = WorkflowConvertImageToVTK(
        segmentation_method=SegmentHeartSimplewareTrimmedBranches()
    )._segmenter
    assert isinstance(trimmed, SegmentHeartSimplewareTrimmedBranches)

    plain_instance = SegmentHeartSimpleware()
    plain = WorkflowConvertImageToVTK(segmentation_method=plain_instance)._segmenter
    assert plain is plain_instance
    assert not isinstance(plain, SegmentHeartSimplewareTrimmedBranches)


def test_transform_model_preserves_unstructured_grid_topology() -> None:
    """Transform helper preserves cells with image shape (Z, Y, X) = (3, 3, 3)."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.array([4, 0, 1, 2, 3])
    celltypes = np.array([pv.CellType.TETRA])
    model = pv.UnstructuredGrid(cells, celltypes, points)
    model.cell_data["label"] = np.array([3], dtype=np.uint8)
    model.point_data["weights"] = np.arange(model.n_points, dtype=np.float64)
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=model,
        patient_models=[model],
        patient_image=image,
    )

    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetIdentity()
    transform.SetTranslation((1.0, 2.0, 3.0))
    workflow.icp_forward_point_transform = transform
    workflow.pca_coefficients = None
    workflow.use_l2l_registration = False
    workflow.use_l2i_registration = False

    output = workflow.transform_model()

    assert isinstance(output, pv.UnstructuredGrid)
    assert output.n_cells == model.n_cells
    np.testing.assert_array_equal(output.celltypes, model.celltypes)
    np.testing.assert_array_equal(output.cell_data["label"], model.cell_data["label"])
    np.testing.assert_array_equal(
        output.point_data["weights"], model.point_data["weights"]
    )
    np.testing.assert_allclose(output.points, points + np.array([1.0, 2.0, 3.0]))


def test_fit_icp_transform_type_defaults_to_affine_and_validates() -> None:
    """The fit must align the way the model was built, so this is tunable."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    model = pv.PolyData(np.zeros((3, 3), dtype=np.float64))
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=model,
        patient_models=[model],
        patient_image=image,
    )
    assert workflow.icp_transform_type == "Affine"

    workflow.set_icp_transform_type("Rigid")
    assert workflow.icp_transform_type == "Rigid"

    with pytest.raises(ValueError, match="Invalid ICP transform"):
        workflow.set_icp_transform_type("Deformable")


def _fit_workflow_for_pca() -> WorkflowFitStatisticalModelToPatient:
    """A minimal fit workflow, for exercising the PCA configuration only."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    model = pv.PolyData(np.zeros((3, 3), dtype=np.float64))
    return WorkflowFitStatisticalModelToPatient(
        template_model=model,
        patient_models=[model],
        patient_image=image,
    )


def test_requested_pca_components_drop_to_what_the_model_carries() -> None:
    """A model built from a small population carries fewer modes than asked for.

    WorkflowCreateStatisticalModel caps a model at one fewer mode than it had
    samples, so a count configured for a full population is too large for a
    model built from a handful of cases.  Asking the optimizer for modes that
    do not exist raises partway through the fit, so the count is reduced here.
    """
    workflow = _fit_workflow_for_pca()
    pca_model = {"eigenvalues": [4.0, 1.0], "components": [[0.0], [0.0]]}

    workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model=pca_model,
        number_of_pca_components=5,
    )

    assert workflow.number_of_pca_components == 2


def test_requested_pca_components_are_kept_when_the_model_carries_them() -> None:
    """Reducing the count must not touch a request the model can satisfy."""
    workflow = _fit_workflow_for_pca()
    pca_model = {
        "eigenvalues": [4.0, 2.0, 1.0],
        "components": [[0.0], [0.0], [0.0]],
    }

    workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model=pca_model,
        number_of_pca_components=2,
    )

    assert workflow.number_of_pca_components == 2


def test_pca_component_count_of_zero_still_means_every_mode() -> None:
    """0 is the documented "use all" sentinel and must survive the reduction."""
    workflow = _fit_workflow_for_pca()
    pca_model = {"eigenvalues": [4.0, 1.0], "components": [[0.0], [0.0]]}

    workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model=pca_model,
        number_of_pca_components=0,
    )

    assert workflow.number_of_pca_components == 0


def test_empty_pca_model_reduces_a_positive_request_to_zero() -> None:
    """A model carrying no modes must not leave a positive request standing.

    ``RegisterModelsPCA`` raises when asked for more modes than it holds, so a
    request that survived an empty model would fail inside the optimizer rather
    than here.  Zero is the documented "use every mode" sentinel, which for an
    empty model is zero modes.
    """
    workflow = _fit_workflow_for_pca()

    workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model={"eigenvalues": [], "components": []},
        number_of_pca_components=5,
    )

    assert workflow.number_of_pca_components == 0


def test_clamped_component_count_reaches_the_pca_registrar() -> None:
    """The reduced count must be what register_model_to_model_pca passes on.

    Reducing the stored count would be pointless if the registrar were built
    from the originally requested one.
    """
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise RuntimeError("stop after capturing the registrar configuration")

    workflow = _fit_workflow_for_pca()
    workflow.set_use_pca_registration(
        use_pca_registration=True,
        pca_model={"eigenvalues": [4.0, 1.0], "components": [[0.0], [0.0]]},
        number_of_pca_components=5,
    )

    from monai_physio import workflow_fit_statistical_model_to_patient as module

    original = module.RegisterModelsPCA.from_pca_model
    module.RegisterModelsPCA.from_pca_model = staticmethod(_capture)
    try:
        with pytest.raises(RuntimeError):
            workflow.register_model_to_model_pca()
    finally:
        module.RegisterModelsPCA.from_pca_model = original

    assert captured["pca_number_of_modes"] == 2
