"""Synthetic tests for ITK ICP model registration."""

from __future__ import annotations

from typing import cast

import itk
import numpy as np
import pytest
import pyvista as pv
from scipy.spatial import cKDTree

from monai_physio import RegisterModelsICPITK


def _rotation(axis: int, degrees: float) -> np.ndarray:
    """Return the matrix rotating by ``degrees`` about world ``axis``."""
    angle = np.radians(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    first, second = [a for a in range(3) if a != axis]
    rotation = np.eye(3)
    rotation[first, first] = cosine
    rotation[first, second] = -sine
    rotation[second, first] = sine
    rotation[second, second] = cosine
    return rotation


def _ellipsoid() -> pv.PolyData:
    """Return an ellipsoid with three distinct radii, so its pose is unique."""
    return cast(
        pv.PolyData, pv.ParametricEllipsoid(60.0, 40.0, 100.0).triangulate().clean()
    )


def _mean_closest_distance(model: pv.PolyData, target: pv.PolyData) -> float:
    """Return the mean distance from model points to their closest target point."""
    distances = np.asarray(cKDTree(target.points).query(model.points)[0])
    return float(distances.mean())


@pytest.mark.parametrize("transform_type", ["Rigid", "Similarity", "Affine"])
def test_register_recovers_a_transform_it_can_represent(transform_type: str) -> None:
    """Each transform type recovers a target its transform can reach."""
    moving = _ellipsoid()
    matrix = _rotation(0, 3.0) @ _rotation(2, 3.0)
    if transform_type == "Similarity":
        matrix = 1.1 * matrix
    elif transform_type == "Affine":
        matrix = matrix @ np.diag([1.1, 0.95, 1.05])
    fixed = pv.PolyData(np.asarray(moving.points) @ matrix.T + [5.0, -3.0, 4.0])

    result = RegisterModelsICPITK(fixed_model=fixed).register(
        moving_model=moving, transform_type=transform_type
    )

    assert _mean_closest_distance(result["registered_model"], fixed) < 0.25


def test_rotation_axis_allows_rotation_about_that_axis_only() -> None:
    """With rotation_axis=2 the z-axis maps to itself, however the target tilts.

    The target is rotated 3 degrees in-plane and tilted 3 degrees about x. The
    in-plane part must be recovered; the tilt must not, by rotation or by skew.
    """
    moving = _ellipsoid()
    in_plane = _rotation(2, 3.0)
    fixed = pv.PolyData(np.asarray(moving.points) @ (_rotation(0, 3.0) @ in_plane).T)

    result = RegisterModelsICPITK(fixed_model=fixed, rotation_axis=2).register(
        moving_model=moving, transform_type="Affine"
    )
    matrix = itk.array_from_matrix(result["moving_to_fixed_transform"].GetMatrix())

    assert np.allclose(matrix[2, :2], 0.0, atol=1e-6)
    assert np.allclose(matrix[:2, 2], 0.0, atol=1e-6)
    upright = pv.PolyData(np.asarray(moving.points) @ in_plane.T)
    upright_result = RegisterModelsICPITK(
        fixed_model=upright, rotation_axis=2
    ).register(moving_model=moving, transform_type="Rigid")
    assert _mean_closest_distance(upright_result["registered_model"], upright) < 0.25


def test_invalid_rotation_axis_is_rejected() -> None:
    """rotation_axis must name a world axis."""
    with pytest.raises(ValueError, match="rotation_axis"):
        RegisterModelsICPITK(fixed_model=_ellipsoid(), rotation_axis=3)
