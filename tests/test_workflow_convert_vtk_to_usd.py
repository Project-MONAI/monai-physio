"""Tests for the appearance and object-naming behavior of the VTK-to-USD workflow.

Synthetic meshes only - no segmentation or image data required.
"""

from pathlib import Path
from typing import Optional, cast

import numpy as np
import pytest
import pyvista as pv
from pxr import Usd, UsdGeom, UsdShade

from monai_physio import SegmentHeartSimpleware, WorkflowConvertVTKToUSD


def _labeled_sphere(
    center: tuple[float, float, float],
    label_name: str,
    group: Optional[str] = None,
) -> pv.PolyData:
    """Return a sphere annotated the way WorkflowConvertImageToVTK annotates one."""
    surface = pv.Sphere(radius=1.0, center=center, theta_resolution=8, phi_resolution=8)
    surface.field_data["SegmentationLabelNames"] = np.array([label_name])
    if group is not None:
        surface.field_data["AnatomyGroup"] = np.array([group])
    return surface


def _two_structure_frame(theta_resolution: int) -> pv.PolyData:
    """Return one frame holding two structures tagged per cell by label id.

    *theta_resolution* varies the triangulation, so a series built from several
    frames shares neither point count nor face count -- the case of surfaces
    contoured independently per frame rather than propagated by registration.
    """
    parts = []
    for label_id, center in ((1, (0.0, 0.0, 0.0)), (5, (5.0, 0.0, 0.0))):
        part = pv.Sphere(
            radius=1.0,
            center=center,
            theta_resolution=theta_resolution,
            phi_resolution=8,
        )
        part.cell_data["SegmentationLabelIds"] = np.full(
            part.n_cells, label_id, dtype=np.int32
        )
        parts.append(part)
    return cast(pv.PolyData, pv.merge(parts, merge_points=False))


def _bound_material_path(stage: Usd.Stage, mesh_path: str) -> str:
    prim = stage.GetPrimAtPath(mesh_path)
    assert prim.IsValid(), f"Missing prim: {mesh_path}"
    binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
    return str(binding.GetMaterialPath())


class TestAnatomyAppearance:
    """Per-structure materials must follow the structure names on the meshes."""

    def test_label_names_drive_prim_names_and_materials(self, tmp_path: Path) -> None:
        """Each labeled mesh becomes its own prim with its own anatomy material."""
        meshes = [
            _labeled_sphere((0.0, 0.0, 0.0), "highres_myocardium"),
            _labeled_sphere((3.0, 0.0, 0.0), "highres_ventricle_left"),
        ]

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=meshes,
            usd_project_name="heart",
            output_directory=tmp_path,
            appearance="anatomy",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        # separate_by_connectivity defaults to True, and each sphere is a
        # single connected component, so every object gains one "_object1" part.
        myocardium = _bound_material_path(
            stage, "/World/heart/highres_myocardium_object1"
        )
        ventricle = _bound_material_path(
            stage, "/World/heart/highres_ventricle_left_object1"
        )
        assert myocardium.endswith("OmniSurface_Myocardium")
        assert ventricle.endswith("OmniSurface_Ventricle_Left")

    def test_explicit_anatomy_type_overrides_names(self, tmp_path: Path) -> None:
        """A caller-supplied anatomy_type still paints every object the same."""
        meshes = [
            _labeled_sphere((0.0, 0.0, 0.0), "highres_myocardium"),
            _labeled_sphere((3.0, 0.0, 0.0), "highres_ventricle_left"),
        ]

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=meshes,
            usd_project_name="heart",
            output_directory=tmp_path,
            appearance="anatomy",
            anatomy_type="heart",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        for name in ("highres_myocardium", "highres_ventricle_left"):
            material = _bound_material_path(stage, f"/World/heart/{name}_object1")
            assert material.endswith("OmniSurface_Heart")

    def test_unmatched_name_falls_back_to_group(self, tmp_path: Path) -> None:
        """No material is named "rib_left_3", so its anatomy group decides."""
        meshes = [
            _labeled_sphere((0.0, 0.0, 0.0), "rib_left_3", group="bone"),
            _labeled_sphere((3.0, 0.0, 0.0), "vertebrae_T7", group="bone"),
        ]

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=meshes,
            usd_project_name="chest",
            output_directory=tmp_path,
            appearance="anatomy",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        for name in ("rib_left_3", "vertebrae_T7"):
            material = _bound_material_path(stage, f"/World/chest/{name}_object1")
            assert material.endswith("OmniSurface_Bone")

    def test_structure_name_wins_over_group(self, tmp_path: Path) -> None:
        """A structure with its own material must not collapse onto its group."""
        mesh = _labeled_sphere((0.0, 0.0, 0.0), "highres_myocardium", group="heart")

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=[mesh],
            usd_project_name="heart",
            output_directory=tmp_path,
            appearance="anatomy",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        material = _bound_material_path(
            stage, "/World/heart/highres_myocardium_object1"
        )
        assert material.endswith("OmniSurface_Myocardium")

    def test_unmatched_name_falls_back_to_other(self, tmp_path: Path) -> None:
        """A mesh whose name matches no anatomy still gets a material."""
        mesh = _labeled_sphere((0.0, 0.0, 0.0), "calibration_phantom")

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=[mesh],
            usd_project_name="scan",
            output_directory=tmp_path,
            appearance="anatomy",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        material = _bound_material_path(
            stage, "/World/scan/calibration_phantom_object1"
        )
        assert material.endswith("OmniSurface_Other")

    def test_unlabeled_meshes_keep_positional_names(self, tmp_path: Path) -> None:
        """Without SegmentationLabelNames, naming stays {project}_{index}."""
        meshes = [
            pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8),
            pv.Sphere(
                radius=1.0, center=(3.0, 0.0, 0.0), theta_resolution=8, phi_resolution=8
            ),
        ]

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=meshes,
            usd_project_name="scan",
            output_directory=tmp_path,
            appearance="anatomy",
            anatomy_type="heart",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        assert stage.GetPrimAtPath("/World/scan/scan_0_object1").IsValid()
        assert stage.GetPrimAtPath("/World/scan/scan_1_object1").IsValid()


class TestLabelTimeSeries:
    """label_names keeps structure identity across the frames of a series."""

    def test_label_names_produce_per_structure_animated_prims(
        self, tmp_path: Path
    ) -> None:
        """Each label becomes one prim, grouped and painted by the taxonomy."""
        frames = [_two_structure_frame(resolution) for resolution in (8, 9, 10)]
        segmenter = SegmentHeartSimpleware()

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=frames,
            usd_project_name="pm0001",
            output_directory=tmp_path,
            appearance="anatomy",
            label_names={1: "left_ventricle", 5: "myocardium"},
            segmenter=segmenter,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        ventricle_path = "/World/pm0001/heart/left_ventricle"
        myocardium_path = "/World/pm0001/heart/myocardium"

        # "left_ventricle" matches no material of its own, so it lands on the
        # heart group's; "myocardium" has one.
        assert _bound_material_path(stage, ventricle_path).endswith("OmniSurface_Heart")
        assert _bound_material_path(stage, myocardium_path).endswith(
            "OmniSurface_Myocardium"
        )

        for mesh_path in (ventricle_path, myocardium_path):
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
            assert mesh.GetPointsAttr().GetTimeSamples() == [0.0, 1.0, 2.0]

    def test_per_cell_labels_are_found_without_label_names(
        self, tmp_path: Path
    ) -> None:
        """The ids on the meshes are enough; the taxonomy supplies the names."""
        workflow = WorkflowConvertVTKToUSD(
            input_meshes=[_two_structure_frame(8), _two_structure_frame(9)],
            usd_project_name="pm0002",
            output_directory=tmp_path,
            appearance="anatomy",
            segmenter=SegmentHeartSimpleware(),
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        assert stage.GetPrimAtPath("/World/pm0002/heart/left_ventricle").IsValid()
        assert stage.GetPrimAtPath("/World/pm0002/heart/myocardium").IsValid()

    def test_static_merge_splits_a_labeled_mesh_by_structure(
        self, tmp_path: Path
    ) -> None:
        """One merged surface must not collapse onto a single static prim."""
        workflow = WorkflowConvertVTKToUSD(
            input_meshes=[_two_structure_frame(8)],
            usd_project_name="patient",
            output_directory=tmp_path,
            appearance="anatomy",
            segmenter=SegmentHeartSimpleware(),
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        assert stage.GetPrimAtPath("/World/patient/heart/left_ventricle").IsValid()
        assert stage.GetPrimAtPath("/World/patient/heart/myocardium").IsValid()
        assert not stage.GetPrimAtPath("/World/patient/patient_0_object1").IsValid(), (
            "Per-object naming used despite the per-cell label array"
        )

    def test_unlabeled_meshes_still_name_per_object(self, tmp_path: Path) -> None:
        """Without the per-cell array, naming falls back to the old path."""
        workflow = WorkflowConvertVTKToUSD(
            input_meshes=[_labeled_sphere((0.0, 0.0, 0.0), "highres_myocardium")],
            usd_project_name="heart",
            output_directory=tmp_path,
            appearance="anatomy",
            segmenter=SegmentHeartSimpleware(),
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        material = _bound_material_path(
            stage, "/World/heart/highres_myocardium_object1"
        )
        assert material.endswith("OmniSurface_Myocardium")

    def test_nameless_labels_fall_back_to_per_object(self, tmp_path: Path) -> None:
        """Without a segmenter the ids name nothing, so splitting on them would
        replace "scan_0" with "label_1" and gain nothing."""
        workflow = WorkflowConvertVTKToUSD(
            input_meshes=[_two_structure_frame(8)],
            usd_project_name="scan",
            output_directory=tmp_path,
            appearance="anatomy",
            anatomy_type="heart",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        assert stage.GetPrimAtPath("/World/scan/scan_0_object1").IsValid()
        assert not stage.GetPrimAtPath("/World/scan/heart/label_1").IsValid()

    def test_several_labeled_static_meshes_raise(self, tmp_path: Path) -> None:
        """Two static objects holding the same labels would collide on one path."""
        workflow = WorkflowConvertVTKToUSD(
            input_meshes=[_two_structure_frame(8), _two_structure_frame(9)],
            usd_project_name="pm0003",
            output_directory=tmp_path,
            appearance="anatomy",
            segmenter=SegmentHeartSimpleware(),
            static_merge=True,
        )
        with pytest.raises(ValueError, match="static_merge with mask_ids"):
            workflow.process()
