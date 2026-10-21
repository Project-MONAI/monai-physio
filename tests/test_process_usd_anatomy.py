"""Tests for ProcessUSDAnatomy's stage-free color lookups."""

import logging

import numpy as np
import pytest
import pyvista as pv

from monai_physio.process_usd_anatomy import ProcessUSDAnatomy


@pytest.fixture
def anatomy_tools() -> ProcessUSDAnatomy:
    return ProcessUSDAnatomy(stage=None)


def test_select_anatomy_type_prefers_first_resolvable(
    anatomy_tools: ProcessUSDAnatomy,
) -> None:
    assert anatomy_tools.select_anatomy_type(["myocardium", "heart"]) == "myocardium"
    assert anatomy_tools.select_anatomy_type(["unknown_xyz", "heart"]) == "heart"


def test_select_anatomy_type_matches_substring(
    anatomy_tools: ProcessUSDAnatomy,
) -> None:
    assert anatomy_tools.select_anatomy_type(["kidney_left"]) == "kidney_left"


def test_select_anatomy_type_falls_back_to_other(
    anatomy_tools: ProcessUSDAnatomy, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    assert anatomy_tools.select_anatomy_type(["unknown_xyz", "nothing_qq"]) == "other"
    assert "unknown_xyz or nothing_qq" in caplog.text


def test_apply_anatomy_color_sets_field_and_cell_color(
    anatomy_tools: ProcessUSDAnatomy,
) -> None:
    mesh = pv.Sphere()
    anatomy_tools.apply_anatomy_color(mesh, ["heart"])
    expected = np.array(anatomy_tools.get_anatomy_diffuse_color("heart"))
    assert np.allclose(mesh.field_data["AnatomyColor"], expected)
    assert mesh.cell_data["Color"].shape == (mesh.n_cells, 4)
    assert mesh.cell_data["Color"].dtype == np.uint8
