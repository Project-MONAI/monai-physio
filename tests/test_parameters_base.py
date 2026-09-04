"""Tests for the three directory roots every tutorial resolves paths through.

Two properties matter here. First, a clone with no environment set must behave
exactly as it always has, because the roots are read by every tutorial and a
silent change of default would scatter results. Second, the ``test`` subtree
must never coincide with the full root, because that separation is what stops a
test run from overwriting the datasets, results and checkpoints of a full run.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from parameters_base import ParametersBase
from parameters_duke_heart_labelmaps import DUKE_HEART
from parameters_heart_ct_kcl import HEART_CT_KCL
from parameters_lung_ct_dirlab import LUNG_CT_DIRLAB

_REPO_ROOT = Path(__file__).resolve().parent.parent

# (accessor name, environment variable, default relative to the repo root)
_ROOTS = [
    ("data_directory", "MONAI_PHYSIO_INPUT_DATA_DIR", Path("data")),
    ("output_directory", "MONAI_PHYSIO_OUTPUT_DATA_DIR", Path("tutorials/output")),
    (
        "weights_directory",
        "MONAI_PHYSIO_WEIGHTS_DIR",
        Path("tutorials/network_weights"),
    ),
]

# Every parameters module a tutorial imports, plus the bare base the tutorials
# with no dataset-specific module use.
_PARAMETERS = [ParametersBase(), DUKE_HEART, LUNG_CT_DIRLAB, HEART_CT_KCL]


def _resolver(parameters: ParametersBase, name: str) -> Callable[[bool], Path]:
    """Return the named root accessor bound to this parameters object."""
    resolver: Callable[[bool], Path] = getattr(parameters, name)
    return resolver


@pytest.mark.parametrize("parameters", _PARAMETERS, ids=lambda p: type(p).__name__)
@pytest.mark.parametrize(("name", "variable", "default"), _ROOTS)
def test_unset_variable_keeps_the_in_repo_default(
    parameters: ParametersBase,
    name: str,
    variable: str,
    default: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clone with no environment set behaves exactly as it always has."""
    monkeypatch.delenv(variable, raising=False)
    assert _resolver(parameters, name)(False) == _REPO_ROOT / default


@pytest.mark.parametrize("parameters", _PARAMETERS, ids=lambda p: type(p).__name__)
@pytest.mark.parametrize(("name", "variable", "default"), _ROOTS)
def test_empty_variable_is_treated_as_unset(
    parameters: ParametersBase,
    name: str,
    variable: str,
    default: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CI expression interpolating an undefined value yields the empty string.

    Reading that as a path would resolve to the working directory, scattering
    results wherever pytest happened to be invoked from.
    """
    monkeypatch.setenv(variable, "")
    assert _resolver(parameters, name)(False) == _REPO_ROOT / default


@pytest.mark.parametrize("parameters", _PARAMETERS, ids=lambda p: type(p).__name__)
@pytest.mark.parametrize(("name", "variable", "default"), _ROOTS)
def test_variable_redirects_both_run_modes(
    parameters: ParametersBase,
    name: str,
    variable: str,
    default: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Set a root and both the full and the test subtree move with it."""
    monkeypatch.setenv(variable, str(tmp_path))
    assert _resolver(parameters, name)(False) == tmp_path
    assert _resolver(parameters, name)(True) == tmp_path / "test"


@pytest.mark.parametrize(("name", "variable", "default"), _ROOTS)
def test_test_subtree_is_never_the_full_root(
    name: str,
    variable: str,
    default: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The separation that stops a test run overwriting a full run's files."""
    monkeypatch.delenv(variable, raising=False)
    resolve = _resolver(ParametersBase(), name)
    assert resolve(True) != resolve(False)
    assert resolve(True).parent == resolve(False)


def test_the_three_roots_are_distinct() -> None:
    """Datasets, results and trained networks must not share a directory."""
    roots = {_resolver(ParametersBase(), name)(False) for name, _, _ in _ROOTS}
    assert len(roots) == 3


@pytest.mark.parametrize("parameters", _PARAMETERS[1:], ids=lambda p: type(p).__name__)
def test_every_parameters_module_derives_from_the_base(
    parameters: ParametersBase,
) -> None:
    """Each dataset module inherits the roots rather than restating them."""
    assert isinstance(parameters, ParametersBase)


@pytest.mark.parametrize("parameters", _PARAMETERS[1:], ids=lambda p: type(p).__name__)
def test_dataset_paths_hang_off_the_roots(
    parameters: ParametersBase,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Redirecting a root moves every path a module derives from it.

    This is the property that lets a runner keep its data off the checkout: no
    dataset path may be pinned to the repository independently of its root.
    """
    monkeypatch.setenv("MONAI_PHYSIO_INPUT_DATA_DIR", str(tmp_path / "in"))
    monkeypatch.setenv("MONAI_PHYSIO_OUTPUT_DATA_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("MONAI_PHYSIO_WEIGHTS_DIR", str(tmp_path / "weights"))

    for name in (
        "input_directory",
        "hold_out_directory",
        "pca_model_file",
        "pca_mean_surface_file",
        "mgn_weights_directory",
    ):
        accessor = getattr(parameters, name, None)
        if accessor is None:  # ParametersHeartCTKCL trains no network.
            continue
        for test_mode in (False, True):
            assert tmp_path in accessor(test_mode).parents, (
                f"{type(parameters).__name__}.{name}({test_mode}) escaped the "
                "redirected roots"
            )
