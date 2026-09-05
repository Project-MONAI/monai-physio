"""Tutorial tests that run each tutorial end-to-end and compare its output.

Each test class maps to one tutorial script.  Tests are gated behind
``--run-tutorials`` (handled by conftest.py) and require the relevant dataset
to be present; the ``*_test_data`` fixtures in conftest.py build the small
subsets a tutorial reads when run as a test.

Every tutorial runs with ``MONAI_PHYSIO_RUNNING_AS_TEST`` set, which sends its
reads and writes to the ``test`` subtree of each root in
``parameters_base.ParametersBase``: it reads the downsampled datasets under
``<input root>/test``, writes its results under ``<output root>/test`` and
trains its networks into ``<weights root>/test``.  Running this suite therefore
cannot read or overwrite the datasets, results or checkpoints of a full run.
Each root defaults to its current in-repo location and is overridable by
environment variable, so a runner can keep them off the checkout.

Output is compared two ways, because neither catches what the other does:

1. Screenshots.  Each tutorial saves PNGs to its output directory, which
   ``TestTools.compare_result_to_baseline_image`` compares against a stored
   baseline with tolerances loose enough to survive a driver change.
2. Metrics.  The JSON and CSV of numbers each tutorial already reports, which
   ``TestTools.compare_result_to_baseline_metrics`` compares within tolerance.
   Two renderings can agree pixel for pixel while the numbers behind them move,
   and only this catches that.  Wall-clock files (``*_runtimes.csv``) are left
   out, being a property of the host rather than of the result.

Run all tutorial tests (they are marked ``slow`` as well as ``tutorial``)::

    pytest tests/test_tutorials.py --run-all -v

Create baselines on first run, then review the generated PNGs before
committing them -- ``--create-baselines`` blesses whatever came out::

    pytest tests/test_tutorials.py --run-all --create-baselines -v
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import pytest

from parameters_base import ParametersBase
from monai_physio.test_tools import TestTools

from .conftest import skip_or_fail_missing_data, tutorial_data_is_required

# Tolerances for screenshot comparison. Loose to survive minor rendering
# differences across OS / GPU / driver versions.
_PX_TOL = 10.0  # per-pixel absolute error (0-255 range)
_MAX_PX = 2000  # maximum number of pixels allowed above _PX_TOL
_TOT_TOL = float("inf")  # use the pixel-count criterion only
# Tolerances for the numeric metrics each tutorial reports.  Loose enough to
# survive the nondeterminism of GPU reductions and a driver change, tight
# enough that a real change in accuracy shows up.  The absolute term is what
# covers the metrics that sit near zero, where a relative one never triggers.
_METRIC_REL_TOL = 0.05
_METRIC_ABS_TOL = 1.0e-3
_REPO_ROOT = Path(__file__).parent.parent
# Where every tutorial writes under ``MONAI_PHYSIO_RUNNING_AS_TEST``, and where
# Tutorial 9 trains its networks to.  Kept apart from the ``output`` and
# ``network_weights`` directories a full run uses, so that running the suite
# cannot overwrite a full run's results or its trained checkpoints.
_TUTORIAL_PATHS = ParametersBase()
_TUTORIAL_OUTPUT = _TUTORIAL_PATHS.output_directory(test_mode=True)
_TUTORIAL_WEIGHTS = _TUTORIAL_PATHS.weights_directory(test_mode=True)


@pytest.fixture(autouse=True)
def _enable_tutorial_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run tutorials against repo data/test through TestTools mode switching."""
    monkeypatch.setenv("MONAI_PHYSIO_RUNNING_AS_TEST", "1")


def _compare_screenshots(
    screenshots: list[Path],
    tt: TestTools,
) -> None:
    """Read each PNG as itk.Image and compare against baseline."""
    if not screenshots:
        pytest.fail("No screenshots produced by tutorial script")

    for png_path in screenshots:
        if not png_path.exists():
            pytest.fail(f"Screenshot not created: {png_path}")
        assert tt.compare_result_to_baseline_image(
            png_path.name,
            per_pixel_absolute_error_tol=_PX_TOL,
            max_number_of_pixels_above_tol=_MAX_PX,
            total_absolute_error_tol=_TOT_TOL,
        ), f"Screenshot baseline mismatch: {png_path.name}"


def _compare_metrics(tt: TestTools, filenames: list[str]) -> None:
    """Compare each reported metrics file against its baseline.

    Screenshots catch what a rendering shows; these catch the accuracy drift
    that a rendering can hide.
    """
    for filename in filenames:
        assert tt.compare_result_to_baseline_metrics(
            filename,
            relative_tol=_METRIC_REL_TOL,
            absolute_tol=_METRIC_ABS_TOL,
        ), f"Metrics baseline mismatch: {filename}"


def _run_tutorial_script(script_name: str) -> dict[str, Any]:
    """Run a tutorial script with no command-line arguments."""
    # runpy does not put the script's own directory on sys.path the way the
    # interpreter does, and the tutorials import their sibling
    # ``parameters_*`` modules by plain name.
    tutorials_dir = str(_REPO_ROOT / "tutorials")
    sys.path.insert(0, tutorials_dir)
    try:
        namespace = runpy.run_path(
            str(_REPO_ROOT / "tutorials" / script_name),
            run_name="__main__",
        )
    finally:
        sys.path.remove(tutorials_dir)
    results = namespace.get("tutorial_results")
    assert isinstance(results, dict), f"{script_name} did not set tutorial_results"
    return results


def _require_files(directory: Path, pattern: str, reason: str) -> None:
    """Skip unless ``directory`` holds at least one entry matching ``pattern``.

    Under ``--require-tutorial-data`` this fails instead, so a runner that is
    supposed to hold every dataset cannot report a green run that tested
    nothing.
    """
    if not list(directory.glob(pattern)):
        skip_or_fail_missing_data(f"No {pattern} under {directory}. {reason}")


def _baseline_tools(class_name: str, out_dir: Path, baselines_root: Path) -> TestTools:
    """TestTools reading the tutorial's own output directory."""
    return TestTools(
        class_name=class_name,
        results_dir=out_dir,
        baselines_dir=baselines_root / class_name,
    )


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial01HeartGatedCTToUSD:
    """End-to-end test for tutorial_01_heart_gated_ct_to_usd.py."""

    _class_name = "tutorial_01_heart_gated_ct_to_usd"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        out_dir = _TUTORIAL_OUTPUT / "tutorial_01_heart"
        results = _run_tutorial_script("tutorial_01_heart_gated_ct_to_usd.py")
        assert results["usd_file"], "USD file path should not be empty"
        assert Path(results["usd_file"]).exists(), "USD file should exist"
        assert results["screenshots"], "Tutorial 1 should produce screenshots"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial01LungGatedCTToUSD:
    """End-to-end test for tutorial_01_lung_gated_ct_to_usd.py."""

    _class_name = "tutorial_01_lung_gated_ct_to_usd"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        _require_files(
            _TUTORIAL_PATHS.data_directory(test_mode=True) / "DirLab-4DCT",
            "Case1Pack_T??.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_01_lung"
        results = _run_tutorial_script("tutorial_01_lung_gated_ct_to_usd.py")
        assert Path(results["usd_file"]).exists(), "USD file should exist"

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


# -----------------------------------------------------------------------------
# Tutorial 2 - Finetune ICON registration
#
# Each variant finetunes uniGradICON, so they are GPU tests as well as slow
# ones, and each needs its own manually acquired dataset.
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestTutorial02DukeHeartDistancemapFinetuneICON:
    """End-to-end test for tutorial_02_duke_heart_distancemap_finetune_icon.py."""

    _class_name = "tutorial_02_duke_heart_distancemap_finetune_icon"

    def test_run(
        self,
        test_directories: dict[str, Path],
        duke_heart_test_data: Path,
    ) -> None:
        _require_files(
            test_directories["data"] / "Duke-Heart-4DLabelmaps",
            "pm*",
            "Duke-Heart-4DLabelmaps is not yet public; see its data/ README.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_02_heart_distancemap"
        results = _run_tutorial_script(
            "tutorial_02_duke_heart_distancemap_finetune_icon.py"
        )
        assert Path(results["weights_path"]).exists(), "Finetuned weights should exist"
        assert results["summary_file"].exists(), "Registration summary should exist"

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestTutorial02LungDistancemapFinetuneICON:
    """End-to-end test for tutorial_02_lung_distancemap_finetune_icon.py."""

    _class_name = "tutorial_02_lung_distancemap_finetune_icon"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        _require_files(
            test_directories["data"] / "DirLab-4DCT",
            "Case*_T??.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_02_lung_distancemap"
        results = _run_tutorial_script("tutorial_02_lung_distancemap_finetune_icon.py")
        assert Path(results["weights_path"]).exists(), "Finetuned weights should exist"
        assert results["registered_distance_maps"], "Registered distance maps expected"

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestTutorial02LungFinetuneICON:
    """End-to-end test for tutorial_02_lung_finetune_icon.py."""

    _class_name = "tutorial_02_lung_finetune_icon"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        _require_files(
            test_directories["data"] / "DirLab-4DCT",
            "Case*_T??.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_02_lung"
        results = _run_tutorial_script("tutorial_02_lung_finetune_icon.py")
        assert Path(results["weights_path"]).exists(), "Finetuned weights should exist"
        assert results["registered_images"], "Registered images expected"

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


# -----------------------------------------------------------------------------
# Tutorial 3 - Reconstruct High-Resolution 4D CT
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial03HeartReconstructHighres4DCT:
    """End-to-end test for tutorial_03_heart_reconstruct_highres_4d_ct.py."""

    _class_name = "tutorial_03_heart_reconstruct_highres_4d_ct"

    def test_run(
        self, test_directories: dict[str, Path], test_images: list[Any]
    ) -> None:
        out_dir = _TUTORIAL_OUTPUT / "tutorial_03_heart"
        results = _run_tutorial_script("tutorial_03_heart_reconstruct_highres_4d_ct.py")
        assert results["reconstructed_files"], (
            "At least one reconstructed frame expected"
        )
        for f in results["reconstructed_files"]:
            assert f.exists(), f"Reconstructed frame missing: {f}"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial03LungReconstructHighres4DCT:
    """End-to-end test for tutorial_03_lung_reconstruct_highres_4d_ct.py."""

    _class_name = "tutorial_03_lung_reconstruct_highres_4d_ct"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        # Match the phase files the script itself globs, not a directory layout
        # it never uses.
        dirlab_dir = test_directories["data"] / "DirLab-4DCT"
        if not list(dirlab_dir.glob("Case1Pack_T??.mha")):
            pytest.skip(
                "DirLab-4DCT Case1Pack phases not downloaded. See data/README.md "
                "for instructions."
            )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_03_lung"
        results = _run_tutorial_script("tutorial_03_lung_reconstruct_highres_4d_ct.py")
        assert results["reconstructed_files"], (
            "At least one reconstructed frame expected"
        )
        for f in results["reconstructed_files"]:
            assert f.exists(), f"Reconstructed frame missing: {f}"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


# -----------------------------------------------------------------------------
# Tutorial 4 - CT Segmentation to VTK
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial04HeartCTToVTK:
    """End-to-end test for tutorial_04_heart_ct_to_vtk.py."""

    _class_name = "tutorial_04_heart_ct_to_vtk"

    def test_run(
        self, test_directories: dict[str, Path], test_images: list[Any]
    ) -> None:
        out_dir = _TUTORIAL_OUTPUT / "tutorial_04_heart"
        results = _run_tutorial_script("tutorial_04_heart_ct_to_vtk.py")
        assert results["surface_file"].exists(), "Combined VTP surface should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial04DukeHeartLabelmapToVTK:
    """End-to-end test for tutorial_04_duke_heart_labelmap_to_vtk.py."""

    _class_name = "tutorial_04_duke_heart_labelmap_to_vtk"

    def test_run(
        self,
        test_directories: dict[str, Path],
        duke_heart_test_data: Path,
    ) -> None:
        _require_files(
            test_directories["data"] / "Duke-Heart-4DLabelmaps",
            "pm[0-9][0-9][0-9][0-9]",
            "Duke-Heart-4DLabelmaps is not yet public; see its data/ README.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_04_duke_heart_labelmap"
        results = _run_tutorial_script("tutorial_04_duke_heart_labelmap_to_vtk.py")
        assert results["case_dirs"], "At least one case should be meshed"
        assert list(out_dir.glob("*.vtp")), f"No surfaces written to {out_dir}"

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial04LungCTToVTK:
    """End-to-end test for tutorial_04_lung_ct_to_vtk.py."""

    _class_name = "tutorial_04_lung"  # the script's project_name, its TestTools key

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        _require_files(
            test_directories["data"] / "DirLab-4DCT",
            "Case*_T??.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_04_lung"
        results = _run_tutorial_script("tutorial_04_lung_ct_to_vtk.py")
        assert results["surface_file"].exists(), "Lung VTP surface should exist"
        assert results["labelmap_file"].exists(), "Lung labelmap should exist"

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


# -----------------------------------------------------------------------------
# Tutorial 5 - VTK to USD
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial05HeartVTKToUSD:
    """End-to-end test for tutorial_05_heart_vtk_to_usd.py."""

    _class_name = "tutorial_05_heart_vtk_to_usd"

    def test_run(
        self, test_directories: dict[str, Path], test_images: list[Any]
    ) -> None:
        # The script reads this exact directory and offers no input override,
        # so bootstrap Tutorial 4 rather than pointing it at other surfaces.
        input_dir = _TUTORIAL_OUTPUT / "tutorial_04_heart"
        if not list(input_dir.glob("patient_*.vtp")):
            _run_tutorial_script("tutorial_04_heart_ct_to_vtk.py")
            assert list(input_dir.glob("patient_*.vtp")), (
                f"Tutorial 4 bootstrap did not create surfaces in: {input_dir}"
            )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_05_heart"
        results = _run_tutorial_script("tutorial_05_heart_vtk_to_usd.py")
        assert results["usd_file"], "USD file path should not be empty"
        assert Path(results["usd_file"]).exists(), "USD file should exist"
        assert len(results["structures"]) > 1, (
            "Per-structure surfaces expected, so that each becomes its own prim"
        )

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial05DukeHeartVTKToUSD:
    """End-to-end test for tutorial_05_duke_heart_vtk_to_usd.py."""

    _class_name = "tutorial_05_duke_heart_vtk_to_usd"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        # The script reads this exact directory and offers no input override,
        # so bootstrap Tutorial 4 rather than pointing it at other surfaces.
        input_dir = _TUTORIAL_OUTPUT / "tutorial_04_duke_heart_labelmap"
        if not list(input_dir.glob("*.vtp")):
            _require_files(
                test_directories["data"] / "Duke-Heart-4DLabelmaps",
                "pm[0-9][0-9][0-9][0-9]",
                "Duke-Heart-4DLabelmaps is not yet public; see its data/ README.",
            )
            _run_tutorial_script("tutorial_04_duke_heart_labelmap_to_vtk.py")
            assert list(input_dir.glob("*.vtp")), (
                f"Tutorial 4 bootstrap did not create surfaces in: {input_dir}"
            )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_05_duke_heart"
        results = _run_tutorial_script("tutorial_05_duke_heart_vtk_to_usd.py")
        assert results["usd_files"], "At least one USD file expected"
        for usd_file in results["usd_files"]:
            assert Path(usd_file).exists(), f"USD file missing: {usd_file}"
        assert len(results["structures"]) > 1, (
            "Per-structure surfaces expected, so that each becomes its own prim"
        )

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


# -----------------------------------------------------------------------------
# Tutorial 6 - Create Statistical Shape Model
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial06CreateStatisticalModel:
    """End-to-end test for tutorial_06_heart_create_statistical_model.py."""

    _class_name = "tutorial_06_heart_create_statistical_model"

    def test_run(
        self, test_directories: dict[str, Path], download_kcl_heart_model: Path
    ) -> None:
        out_dir = _TUTORIAL_OUTPUT / "tutorial_06_heart"
        results = _run_tutorial_script("tutorial_06_heart_create_statistical_model.py")
        assert results["model_file"].exists(), "pca_model.json should exist"
        assert results["mean_surface_file"].exists(), "Mean surface VTP should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial06DukeHeartCreateStatisticalModel:
    """End-to-end test for tutorial_06_duke_heart_create_statistical_model.py."""

    _class_name = "tutorial_06_duke_heart_create_statistical_model"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        # The model is built from Tutorial 4's surfaces, not from the dataset.
        _require_files(
            _TUTORIAL_OUTPUT / "tutorial_04_duke_heart_labelmap",
            "*.vtp",
            "Run tutorial_04_duke_heart_labelmap_to_vtk.py first.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_06_duke_heart"
        results = _run_tutorial_script(
            "tutorial_06_duke_heart_create_statistical_model.py"
        )
        assert results["model_file"].exists(), "pca_model.json should exist"
        assert results["mean_surface_file"].exists(), "Mean surface VTP should exist"

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial06LungCreateStatisticalModel:
    """End-to-end test for tutorial_06_lung_create_statistical_model.py."""

    _class_name = "tutorial_06_lung_create_statistical_model"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        # This variant segments the T70 phases itself, so the images are input.
        _require_files(
            test_directories["data"] / "DirLab-4DCT",
            "Case*T70.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_06_lung"
        results = _run_tutorial_script("tutorial_06_lung_create_statistical_model.py")
        assert results["model_file"].exists(), "pca_model.json should exist"
        assert results["mean_surface_file"].exists(), "Mean surface VTP should exist"

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial07FitStatisticalModelToPatient:
    """End-to-end test for tutorial_07_heart_fit_statistical_model_to_patient.py."""

    _class_name = "tutorial_07_heart_fit_statistical_model_to_patient"

    def test_run(
        self,
        test_directories: dict[str, Path],
        download_kcl_heart_model: Path,
        dirlab_test_data: Path,
    ) -> None:
        # The patient scan comes from DIR-Lab, which must be acquired manually.
        if not (
            test_directories["data"] / "DirLab-4DCT" / "Case1Pack_T70.mha"
        ).exists():
            pytest.skip(
                "DirLab-4DCT Case1Pack_T70 not downloaded. See data/README.md "
                "for instructions."
            )

        pca_json = _TUTORIAL_OUTPUT / "tutorial_06_heart" / "pca_model.json"
        if not pca_json.exists():
            _run_tutorial_script("tutorial_06_heart_create_statistical_model.py")
            assert pca_json.exists(), (
                "Tutorial 6 bootstrap did not create the expected PCA model file: "
                f"{pca_json}"
            )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_07_heart"
        results = _run_tutorial_script(
            "tutorial_07_heart_fit_statistical_model_to_patient.py"
        )
        # ``out_dir.name`` is the tutorial's ``project_name`` file prefix.
        registered_surface_file = (
            out_dir / f"{out_dir.name}_template_surface_registered.vtp"
        )
        assert registered_surface_file.exists(), "Registered surface VTP should exist"

        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            ["tutorial_07_heart_registered_coefficients.json"],
        )


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial07DukeHeartFitStatisticalModelToPatient:
    """End-to-end test for tutorial_07_duke_heart_fit_statistical_model_to_patient."""

    _class_name = "tutorial_07_duke_heart"  # the script's project_name

    def test_run(
        self,
        test_directories: dict[str, Path],
        duke_heart_test_data: Path,
    ) -> None:
        _require_files(
            test_directories["data"] / "Duke-Heart-4DLabelmaps",
            "pm*",
            "Duke-Heart-4DLabelmaps is not yet public; see its data/ README.",
        )
        _require_files(
            _TUTORIAL_OUTPUT / "tutorial_06_duke_heart",
            "pca_model.json",
            "Run tutorial_06_duke_heart_create_statistical_model.py first.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_07_duke_heart"
        results = _run_tutorial_script(
            "tutorial_07_duke_heart_fit_statistical_model_to_patient.py"
        )
        # ``out_dir.name`` is the tutorial's ``project_name`` file prefix.
        assert (out_dir / f"{out_dir.name}_template_surface_registered.vtp").exists(), (
            "Registered surface VTP should exist"
        )

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            ["tutorial_07_duke_heart_registered_coefficients.json"],
        )


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial07LungFitStatisticalModelToPatient:
    """End-to-end test for tutorial_07_lung_fit_statistical_model_to_patient.py."""

    _class_name = "tutorial_07_lung"  # the script's project_name

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
        chest_ct_test_data: Path,
    ) -> None:
        # The lung variant fits the ungated Chest-CT scan, which the
        # download CLI provides, rather than a gated DIR-Lab phase.
        _require_files(
            test_directories["data"] / "Chest-CT",
            "Chest-CT.mha",
            "Fetch it with: monai-physio-download-data Chest-CT.",
        )
        _require_files(
            _TUTORIAL_OUTPUT / "tutorial_06_lung",
            "pca_model.json",
            "Run tutorial_06_lung_create_statistical_model.py first.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_07_lung"
        results = _run_tutorial_script(
            "tutorial_07_lung_fit_statistical_model_to_patient.py"
        )
        assert (out_dir / f"{out_dir.name}_template_surface_registered.vtp").exists(), (
            "Registered surface VTP should exist"
        )

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            ["tutorial_07_lung_registered_coefficients.json"],
        )


# -----------------------------------------------------------------------------
# Tutorial 8 - Fit the model to every gated frame of several patients
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial08DukeHeartFitModelTo4DPatients:
    """End-to-end test for tutorial_08_duke_heart_fit_model_to_4d_patients.py."""

    _class_name = "tutorial_08_duke_heart_fit_model_to_4d_patients"

    def test_run(
        self,
        test_directories: dict[str, Path],
        duke_heart_test_data: Path,
    ) -> None:
        _require_files(
            test_directories["data"] / "Duke-Heart-4DLabelmaps",
            "pm*",
            "Duke-Heart-4DLabelmaps is not yet public; see its data/ README.",
        )
        _require_files(
            _TUTORIAL_OUTPUT / "tutorial_06_duke_heart",
            "pca_model.json",
            "Run tutorial_06_duke_heart_create_statistical_model.py first.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_08_duke_heart"
        results = _run_tutorial_script(
            "tutorial_08_duke_heart_fit_model_to_4d_patients.py"
        )
        assert results["cases"], "At least one case should be fitted"
        for case_id, case in results["cases"].items():
            assert case["fitted_reference_mesh_file"].exists(), (
                f"{case_id}: fitted SSM surface should exist"
            )

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial08LungFitModelTo4DPatients:
    """End-to-end test for tutorial_08_lung_fit_model_to_4d_patients.py."""

    _class_name = "tutorial_08_lung_fit_model_to_4d_patients"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        _require_files(
            _TUTORIAL_PATHS.data_directory(test_mode=True) / "DirLab-4DCT",
            "Case*_T70.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )
        _require_files(
            _TUTORIAL_OUTPUT / "tutorial_06_lung",
            "pca_model.json",
            "Run tutorial_06_lung_create_statistical_model.py first.",
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_08_lung"
        results = _run_tutorial_script("tutorial_08_lung_fit_model_to_4d_patients.py")
        assert results["cases"], "At least one case should be fitted"
        for case_id, case in results["cases"].items():
            assert case["fitted_reference_mesh_file"].exists(), (
                f"{case_id}: fitted SSM surface should exist"
            )

        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


# -----------------------------------------------------------------------------
# Tutorials 9 and 10 - PhysicsNeMo train and infer
#
# Both need PhysicsNeMo/PyTorch Geometric (base dependencies) and the
# Tutorial 8 fitted meshes, so they skip rather than fail when either is
# absent.
# -----------------------------------------------------------------------------


def _require_physicsnemo() -> None:
    """Skip unless both MGN dependencies are installed.

    Fails instead of skipping under ``--require-tutorial-data``: these are
    unconditional base dependencies of monai-physio, so their absence there
    means the environment is broken, not merely that optional data is
    missing.
    """
    if importlib.util.find_spec("physicsnemo") is None:
        reason = (
            "PhysicsNeMo not installed, though it is a base dependency of "
            "monai-physio; the environment may be broken."
        )
        if tutorial_data_is_required():
            pytest.fail(f"{reason} (--require-tutorial-data is set)")
        pytest.skip(reason)
    if importlib.util.find_spec("torch_geometric") is None:
        reason = (
            "PyTorch Geometric not installed, though it is a base dependency "
            "of monai-physio; the environment may be broken."
        )
        if tutorial_data_is_required():
            pytest.fail(f"{reason} (--require-tutorial-data is set)")
        pytest.skip(reason)


def _require_physicsnemo_and_tutorial_08() -> Path:
    """Skip unless the MGN dependencies and three Tutorial 8 cases are present."""
    _require_physicsnemo()
    data_dir = _TUTORIAL_OUTPUT / "tutorial_08_lung"
    if len(list(data_dir.glob("Case*Pack"))) < 3:
        skip_or_fail_missing_data(
            "Fewer than three Tutorial 8 cases under "
            "the Tutorial 8 lung output directory. "
            "Run tutorial_08_lung_fit_model_to_4d_patients.py first."
        )
    return data_dir


def _require_physicsnemo_and_tutorial_08_duke() -> Path:
    """Skip unless the MGN dependencies and the Tutorial 8 Duke cases are present."""
    _require_physicsnemo()
    data_dir = _TUTORIAL_OUTPUT / "tutorial_08_duke_heart"
    if not list(data_dir.glob("pm*")):
        pytest.skip(
            "No Tutorial 8 cases under the Tutorial 8 Duke heart output directory. "
            "Run tutorial_08_duke_heart_fit_model_to_4d_patients.py first."
        )
    return data_dir


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial09LungTrainPhysicsNeMoMGN:
    """End-to-end test for tutorial_09_lung_train_physicsnemo_mgn.py."""

    _class_name = "tutorial_09_lung_train_physicsnemo_mgn"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_physicsnemo_and_tutorial_08()

        results = _run_tutorial_script("tutorial_09_lung_train_physicsnemo_mgn.py")
        model_dir = Path(results["model_directory"])
        assert (model_dir / "mgn_stage_model.pt").exists(), "Checkpoint should exist"
        assert results["cases"], "At least one held-out case should be evaluated"

        # The model goes to the shared weights directory; the manifests, the
        # evaluation and the screenshots stay under the tutorial's output.
        tt = TestTools(
            class_name=self._class_name,
            results_dir=_TUTORIAL_OUTPUT / "tutorial_09_lung_mgn",
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial09DukeHeartTrainPhysicsNeMoMGN:
    """End-to-end test for tutorial_09_duke_heart_train_physicsnemo_mgn.py."""

    _class_name = "tutorial_09_duke_heart_train_physicsnemo_mgn"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_physicsnemo_and_tutorial_08_duke()

        results = _run_tutorial_script(
            "tutorial_09_duke_heart_train_physicsnemo_mgn.py"
        )
        model_dir = Path(results["model_directory"])
        assert (model_dir / "mgn_stage_model.pt").exists(), "Checkpoint should exist"
        assert results["cases"], "At least one held-out case should be evaluated"

        # The model goes to the shared weights directory; the manifests, the
        # evaluation and the screenshots stay under the tutorial's output.
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(
                self._class_name,
                _TUTORIAL_OUTPUT / "tutorial_09_duke_heart_mgn",
                test_directories["baselines"],
            ),
        )


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial10LungInferPhysicsNeMoMGN:
    """End-to-end test for tutorial_10_lung_infer_physicsnemo_mgn.py."""

    _class_name = "tutorial_10_lung_infer_physicsnemo_mgn"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_physicsnemo_and_tutorial_08()

        # ParametersLungCTDirLab.mgn_weights_directory under test mode, which is
        # where Tutorial 9 trains to. Reading the full-run directory instead
        # would look for a checkpoint the test run never writes.
        model_dir = _TUTORIAL_WEIGHTS / "physicsnemo_mgn_lung_motion"
        if not (model_dir / "mgn_stage_model.pt").exists():
            _run_tutorial_script("tutorial_09_lung_train_physicsnemo_mgn.py")
            assert (model_dir / "mgn_stage_model.pt").exists(), (
                f"Tutorial 9 bootstrap did not create a checkpoint under {model_dir}"
            )

        results = _run_tutorial_script("tutorial_10_lung_infer_physicsnemo_mgn.py")
        assert results["predicted_surfaces"], "At least one predicted surface expected"
        assert Path(results["predicted_surfaces"][0]).exists(), (
            "Predicted surface should exist"
        )
        assert Path(results["usd_file"]).exists(), "USD file should exist"

        out_dir = _TUTORIAL_OUTPUT / "tutorial_10_lung_mgn" / "Case1Pack"
        tt = TestTools(
            class_name=self._class_name,
            results_dir=out_dir,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )
        _compare_screenshots(results["screenshots"], tt)


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial10DukeHeartInferPhysicsNeMoMGN:
    """End-to-end test for tutorial_10_duke_heart_infer_physicsnemo_mgn.py."""

    _class_name = "tutorial_10_duke_heart_infer_physicsnemo_mgn"

    def test_run(
        self,
        test_directories: dict[str, Path],
        duke_heart_test_data: Path,
    ) -> None:
        _require_physicsnemo_and_tutorial_08_duke()

        # ParametersDukeHeartLabelmaps.mgn_weights_directory, where Tutorial 9
        # trains to.
        model_dir = _TUTORIAL_WEIGHTS / "physicsnemo_mgn_duke_heart_motion"
        if not (model_dir / "mgn_stage_model.pt").exists():
            _run_tutorial_script("tutorial_09_duke_heart_train_physicsnemo_mgn.py")
            assert (model_dir / "mgn_stage_model.pt").exists(), (
                f"Tutorial 9 bootstrap did not create a checkpoint under {model_dir}"
            )

        results = _run_tutorial_script(
            "tutorial_10_duke_heart_infer_physicsnemo_mgn.py"
        )
        assert results["predicted_surfaces"], "At least one predicted surface expected"
        assert Path(results["usd_file"]).exists(), "USD file should exist"

        # ParametersDukeHeartLabelmaps.hold_out_case names the output subdirectory.
        out_dir = _TUTORIAL_OUTPUT / "tutorial_10_duke_heart_mgn" / "pm0027"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


# -----------------------------------------------------------------------------
# Tutorial 11 - Score the surrogate against the acquired frames
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial11DukeHeartEvaluatePhysicsNeMo:
    """End-to-end test for tutorial_11_duke_heart_evaluate_physicsnemo.py."""

    _class_name = "tutorial_11_duke_heart_evaluate_physicsnemo"

    def test_run(
        self,
        test_directories: dict[str, Path],
        duke_heart_test_data: Path,
    ) -> None:
        _require_physicsnemo_and_tutorial_08_duke()
        # The acquired labelmaps every metric is measured against.
        _require_files(
            test_directories["data"] / "Duke-Heart-4DLabelmaps",
            "pm*",
            "Duke-Heart-4DLabelmaps is not yet public; see its data/ README.",
        )
        _require_files(
            _TUTORIAL_WEIGHTS / "physicsnemo_mgn_duke_heart_motion",
            "mgn_stage_model.pt",
            "Run tutorial_09_duke_heart_train_physicsnemo_mgn.py first.",
        )

        results = _run_tutorial_script("tutorial_11_duke_heart_evaluate_physicsnemo.py")
        assert results["rows"], "At least one structure should be scored"
        assert results["csv_file"].exists(), "Metrics CSV should exist"
        assert results["report_file"].exists(), "Markdown report should exist"
        assert results["displacement_statistics"], (
            "Per-point displacement error should be reported"
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_11_duke_heart" / "pm0027"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            ["evaluation_metrics.csv"],
        )


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial11LungEvaluatePhysicsNeMo:
    """End-to-end test for tutorial_11_lung_evaluate_physicsnemo.py."""

    _class_name = "tutorial_11_lung_evaluate_physicsnemo"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        _require_physicsnemo_and_tutorial_08()
        # The acquired phases this variant segments to get its ground truth.
        _require_files(
            test_directories["data"] / "DirLab-4DCT",
            "Case1Pack_T??.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )
        _require_files(
            _TUTORIAL_WEIGHTS / "physicsnemo_mgn_lung_motion",
            "mgn_stage_model.pt",
            "Run tutorial_09_lung_train_physicsnemo_mgn.py first.",
        )

        results = _run_tutorial_script("tutorial_11_lung_evaluate_physicsnemo.py")
        assert results["rows"], "At least one structure should be scored"
        assert results["csv_file"].exists(), "Metrics CSV should exist"
        assert results["report_file"].exists(), "Markdown report should exist"
        assert results["displacement_statistics"], (
            "Per-point displacement error should be reported"
        )

        # ParametersLungCTDirLab.mgn_hold_out_case names the output subdirectory.
        out_dir = _TUTORIAL_OUTPUT / "tutorial_11_lung" / "Case1Pack"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            ["evaluation_metrics.csv"],
        )


# -----------------------------------------------------------------------------
# Tutorial 12 - End-to-end inference, from a raw image to a moving anatomy
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial12DukeHeartEndToEndInference:
    """End-to-end test for tutorial_12_duke_heart_end_to_end_inference.py."""

    _class_name = "tutorial_12_duke_heart_end_to_end_inference"

    def test_run(
        self,
        test_directories: dict[str, Path],
        duke_heart_test_data: Path,
    ) -> None:
        _require_physicsnemo()
        # This tutorial starts from the raw labelmaps, so it needs Tutorial 6's
        # model and Tutorial 9's weights but not Tutorial 8's fits.
        _require_files(
            test_directories["data"] / "Duke-Heart-4DLabelmaps",
            "pm*",
            "Duke-Heart-4DLabelmaps is not yet public; see its data/ README.",
        )
        _require_files(
            _TUTORIAL_OUTPUT / "tutorial_06_duke_heart",
            "pca_model.json",
            "Run tutorial_06_duke_heart_create_statistical_model.py first.",
        )
        _require_files(
            _TUTORIAL_WEIGHTS / "physicsnemo_mgn_duke_heart_motion",
            "mgn_stage_model.pt",
            "Run tutorial_09_duke_heart_train_physicsnemo_mgn.py first.",
        )

        results = _run_tutorial_script("tutorial_12_duke_heart_end_to_end_inference.py")
        assert results["pca_coefficients_file"].exists(), "Fitted shape parameters"
        assert results["predicted_surfaces"], "At least one predicted surface expected"
        assert Path(results["usd_file"]).exists(), "USD file should exist"
        assert results["runtime_file"].exists(), "Per-step runtime record should exist"

        out_dir = _TUTORIAL_OUTPUT / "tutorial_12_duke_heart" / "pm0027"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            ["pm0027_ssm_pca_coefficients.json"],
        )


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial12LungEndToEndInference:
    """End-to-end test for tutorial_12_lung_end_to_end_inference.py."""

    _class_name = "tutorial_12_lung_end_to_end_inference"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
        chest_ct_test_data: Path,
    ) -> None:
        _require_physicsnemo()
        _require_files(
            test_directories["data"] / "DirLab-4DCT",
            "Case1Pack_T??.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )
        _require_files(
            _TUTORIAL_OUTPUT / "tutorial_06_lung",
            "pca_model.json",
            "Run tutorial_06_lung_create_statistical_model.py first.",
        )
        _require_files(
            _TUTORIAL_WEIGHTS / "physicsnemo_mgn_lung_motion",
            "mgn_stage_model.pt",
            "Run tutorial_09_lung_train_physicsnemo_mgn.py first.",
        )

        results = _run_tutorial_script("tutorial_12_lung_end_to_end_inference.py")
        assert results["pca_coefficients_file"].exists(), "Fitted shape parameters"
        assert results["predicted_surfaces"], "At least one predicted surface expected"
        assert Path(results["usd_file"]).exists(), "USD file should exist"
        assert results["runtime_file"].exists(), "Per-step runtime record should exist"

        out_dir = _TUTORIAL_OUTPUT / "tutorial_12_lung" / "Case1Pack"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            ["Case1Pack_ssm_pca_coefficients.json"],
        )


# -----------------------------------------------------------------------------
# Tutorial 13 - Both rhythms on one scan
#
# It segments the heart with Simpleware and infers both networks, so it needs
# every optional dependency the toolkit has.
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
@pytest.mark.requires_simpleware
class TestTutorial13HeartAndLungMotion:
    """End-to-end test for tutorial_13_heart_and_lung_motion.py."""

    _class_name = "tutorial_13_heart_and_lung_motion"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
        duke_heart_test_data: Path,
        chest_ct_test_data: Path,
    ) -> None:
        _require_physicsnemo()
        # The ungated clinical scan both rhythms are inferred onto.
        _require_files(
            test_directories["data"] / "Chest-CT",
            "Chest-CT.mha",
            "Fetch it with: monai-physio-download-data Chest-CT.",
        )
        # Tutorial 7 (lung) fits that same scan; its output is the reference
        # geometry the respiratory network is conditioned on.
        _require_files(
            _TUTORIAL_OUTPUT / "tutorial_07_lung",
            "tutorial_07_lung_registered_coefficients.json",
            "Run tutorial_07_lung_fit_statistical_model_to_patient.py first.",
        )
        for weights_dir, tutorial in (
            (
                "physicsnemo_mgn_lung_motion",
                "tutorial_09_lung_train_physicsnemo_mgn.py",
            ),
            (
                "physicsnemo_mgn_duke_heart_motion",
                "tutorial_09_duke_heart_train_physicsnemo_mgn.py",
            ),
        ):
            _require_files(
                _TUTORIAL_WEIGHTS / weights_dir,
                "mgn_stage_model.pt",
                f"Run {tutorial} first.",
            )

        results = _run_tutorial_script("tutorial_13_heart_and_lung_motion.py")
        assert results["combined_surfaces"], "Combined-motion frames expected"
        assert Path(results["usd_file"]).exists(), "Combined 4D USD should exist"

        out_dir = _TUTORIAL_OUTPUT / "tutorial_13_heart_and_lung"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


# -----------------------------------------------------------------------------
# Tutorial 14 - Sweep the shape parameters and score every combination
#
# In test mode the tutorials vary one mode over three offsets, so the sweep is
# three evaluations rather than the twenty-five of the default grid.
# -----------------------------------------------------------------------------

_TUTORIAL_14_TEST_MODE_COMBINATIONS = 3


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial14DukeHeartShapeParameterSweep:
    """End-to-end test for tutorial_14_duke_heart_shape_parameter_sweep.py."""

    _class_name = "tutorial_14_duke_heart_shape_parameter_sweep"

    def test_run(
        self,
        test_directories: dict[str, Path],
        duke_heart_test_data: Path,
    ) -> None:
        _require_physicsnemo_and_tutorial_08_duke()
        # The acquired labelmaps every metric is measured against.
        _require_files(
            test_directories["data"] / "Duke-Heart-4DLabelmaps",
            "pm*",
            "Duke-Heart-4DLabelmaps is not yet public; see its data/ README.",
        )
        _require_files(
            _TUTORIAL_WEIGHTS / "physicsnemo_mgn_duke_heart_motion",
            "mgn_stage_model.pt",
            "Run tutorial_09_duke_heart_train_physicsnemo_mgn.py first.",
        )

        results = _run_tutorial_script(
            "tutorial_14_duke_heart_shape_parameter_sweep.py"
        )
        assert results["rows"], "At least one structure should be scored"
        assert results["metrics_csv_file"].exists(), "Sweep metrics CSV should exist"
        assert results["summary_csv_file"].exists(), "Sweep summary CSV should exist"
        combinations = {row["combination"] for row in results["rows"]}
        assert len(combinations) == _TUTORIAL_14_TEST_MODE_COMBINATIONS, (
            "Every combination of the test-mode grid should reach the CSV"
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_14_duke_heart" / "pm0027"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            [
                "shape_sweep_metrics.csv",
                "shape_sweep_summary.csv",
            ],
        )


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial14LungShapeParameterSweep:
    """End-to-end test for tutorial_14_lung_shape_parameter_sweep.py."""

    _class_name = "tutorial_14_lung_shape_parameter_sweep"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        _require_physicsnemo_and_tutorial_08()
        # The acquired phases this variant segments to get its ground truth.
        _require_files(
            test_directories["data"] / "DirLab-4DCT",
            "Case1Pack_T??.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )
        _require_files(
            _TUTORIAL_WEIGHTS / "physicsnemo_mgn_lung_motion",
            "mgn_stage_model.pt",
            "Run tutorial_09_lung_train_physicsnemo_mgn.py first.",
        )

        results = _run_tutorial_script("tutorial_14_lung_shape_parameter_sweep.py")
        assert results["rows"], "At least one structure should be scored"
        assert results["metrics_csv_file"].exists(), "Sweep metrics CSV should exist"
        assert results["summary_csv_file"].exists(), "Sweep summary CSV should exist"
        combinations = {row["combination"] for row in results["rows"]}
        assert len(combinations) == _TUTORIAL_14_TEST_MODE_COMBINATIONS, (
            "Every combination of the test-mode grid should reach the CSV"
        )

        # ParametersLungCTDirLab.mgn_hold_out_case names the output subdirectory.
        out_dir = _TUTORIAL_OUTPUT / "tutorial_14_lung" / "Case1Pack"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            [
                "shape_sweep_metrics.csv",
                "shape_sweep_summary.csv",
            ],
        )


# Both Tutorial 15 variants clamp themselves to this many folds under
# TestTools.running_as_test, which the autouse fixture above turns on.
_TUTORIAL_15_TEST_MODE_FOLDS = 2


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial15LungLeaveOneOut:
    """End-to-end test for tutorial_15_lung_leave_one_out.py."""

    _class_name = "tutorial_15_lung_leave_one_out"

    def test_run(
        self,
        test_directories: dict[str, Path],
        dirlab_test_data: Path,
    ) -> None:
        _require_physicsnemo()
        # Nothing from Tutorials 6, 8 or 9 is needed: every fold builds its own
        # shape model, fits and network. The cohort itself is the only input.
        _require_files(
            test_directories["data"] / "DirLab-4DCT",
            "Case*_T70.mha",
            "DirLab-4DCT is acquired manually; see data/README.md.",
        )

        results = _run_tutorial_script("tutorial_15_lung_leave_one_out.py")
        assert results["rows"], "At least one structure should be scored"
        assert results["metrics_file"].exists(), "Pooled metrics CSV should exist"
        assert results["report_file"].exists(), "Cross-fold report should exist"
        assert len(results["held_out_cases"]) == _TUTORIAL_15_TEST_MODE_FOLDS
        scored = {row["held_out_case"] for row in results["rows"]}
        assert scored == set(results["held_out_cases"]), (
            "Every fold's held-out case should reach the pooled metrics"
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_15_lung"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            ["loo_metrics.csv"],
        )


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial15DukeHeartLeaveOneOut:
    """End-to-end test for tutorial_15_duke_heart_leave_one_out.py."""

    _class_name = "tutorial_15_duke_heart_leave_one_out"

    def test_run(
        self,
        test_directories: dict[str, Path],
        duke_heart_test_data: Path,
    ) -> None:
        _require_physicsnemo()
        _require_files(
            test_directories["data"] / "Duke-Heart-4DLabelmaps",
            "pm*/*_ref_labelmap.nii.gz",
            "Duke-Heart-4DLabelmaps is not public yet; "
            "see data/Duke-Heart-4DLabelmaps/README.md.",
        )

        results = _run_tutorial_script("tutorial_15_duke_heart_leave_one_out.py")
        assert results["rows"], "At least one structure should be scored"
        assert results["metrics_file"].exists(), "Pooled metrics CSV should exist"
        assert results["report_file"].exists(), "Cross-fold report should exist"
        assert len(results["held_out_cases"]) == _TUTORIAL_15_TEST_MODE_FOLDS
        scored = {row["held_out_case"] for row in results["rows"]}
        assert scored == set(results["held_out_cases"]), (
            "Every fold's held-out case should reach the pooled metrics"
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_15_duke_heart"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
        _compare_metrics(
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
            ["loo_metrics.csv"],
        )


def _require_physicsnemo_sym() -> None:
    """Skip unless PhysicsNeMo Sym is importable.

    It supplies ``PhysicsInformer``, which the physics-informed trainer builds
    its residual on.  It ships inside ``nvidia-physicsnemo`` rather than as a
    separate distribution, so an older PhysicsNeMo is what makes this fail.
    """
    _require_physicsnemo()
    if importlib.util.find_spec("physicsnemo.sym") is None:
        skip_or_fail_missing_data(
            "physicsnemo.sym not available; it ships inside nvidia-physicsnemo, "
            "so upgrade that package to get PhysicsInformer."
        )


def _require_tutorial_16() -> Path:
    """Skip unless Tutorial 16 left manifests and a template behind."""
    prep_dir = _TUTORIAL_OUTPUT / "tutorial_16_duke_heart_physics_informed_motion"
    if not list(prep_dir.glob("manifests/*_manifest.json")):
        skip_or_fail_missing_data(
            "No Tutorial 16 manifests under its output directory. Run "
            "tutorial_16_duke_heart_physics_informed_motion_prep.py first."
        )
    return prep_dir


def _require_tutorial_17() -> Path:
    """Skip unless Tutorial 17 trained a physics-informed checkpoint."""
    weights_dir = _TUTORIAL_WEIGHTS / "physicsnemo_physics_informed_motion_duke_heart"
    if not list(weights_dir.glob("*_stage_model.pt")):
        skip_or_fail_missing_data(
            "No physics-informed checkpoint under its weights directory. Run "
            "tutorial_17_duke_heart_physics_informed_motion_train.py first."
        )
    return weights_dir


@pytest.mark.requires_gpu
@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial16DukeHeartPhysicsInformedMotionPrep:
    """End-to-end test for tutorial_16_duke_heart_physics_informed_motion_prep.py."""

    _class_name = "tutorial_16_duke_heart_physics_informed_motion_prep"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_files(
            _TUTORIAL_OUTPUT / "tutorial_04_duke_heart_labelmap",
            "*_ref_heart_minus_interior_chambers.vtp",
            "Run tutorial_04_duke_heart_labelmap_to_vtk.py first.",
        )
        _require_files(
            test_directories["data"] / "Duke-Heart-4DLabelmaps",
            "pm*/*_ref_labelmap.nii.gz",
            "Duke-Heart-4DLabelmaps is not public yet; "
            "see data/Duke-Heart-4DLabelmaps/README.md.",
        )

        results = _run_tutorial_script(
            "tutorial_16_duke_heart_physics_informed_motion_prep.py"
        )
        assert results["manifests"], "At least one case should yield a manifest"
        assert results["template_file"].exists(), "The tet template should exist"
        assert results["model_file"].exists(), "The volumetric model should exist"

        # The whole point of this tutorial is that the model has an interior,
        # so assert the template is volumetric rather than merely present.
        template = pv.read(str(results["template_file"]))
        tets = template.cells_dict[np.uint8(pv.CellType.TETRA)]
        assert len(tets) > 0, "The template should carry tetrahedra"
        quality = np.asarray(
            template.cell_quality(["scaled_jacobian"]).cell_data["scaled_jacobian"]
        )
        assert quality.min() >= 0.1, (
            "trim_tetrahedra_to_surface holds every cell above a scaled Jacobian "
            f"of 0.1; the worst here is {quality.min():.3f}"
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_16_duke_heart_physics_informed_motion"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


@pytest.mark.requires_gpu
@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial17DukeHeartPhysicsInformedMotionTrain:
    """End-to-end test for tutorial_17_duke_heart_physics_informed_motion_train.py."""

    _class_name = "tutorial_17_duke_heart_physics_informed_motion_train"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_physicsnemo_sym()
        _require_tutorial_16()

        results = _run_tutorial_script(
            "tutorial_17_duke_heart_physics_informed_motion_train.py"
        )
        assert results["checkpoint"].exists(), "A checkpoint should be written"
        assert results["runs"]["physics_informed"]["training_loss"], (
            "The physics-informed run should report a per-epoch loss"
        )

        out_dir = _TUTORIAL_OUTPUT / "tutorial_16_duke_heart_physics_informed_motion"
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )


@pytest.mark.requires_gpu
@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial18DukeHeartPhysicsInformedMotionInfer:
    """End-to-end test for tutorial_18_duke_heart_physics_informed_motion_infer.py."""

    _class_name = "tutorial_18_duke_heart_physics_informed_motion_infer"

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_physicsnemo_sym()
        _require_tutorial_16()
        _require_tutorial_17()

        results = _run_tutorial_script(
            "tutorial_18_duke_heart_physics_informed_motion_infer.py"
        )
        assert results["comparison_file"].exists(), "The comparison CSV should exist"
        assert results["usd_file"].exists(), "The animated USD should exist"
        assert results["stress_files"], "Every predicted phase should carry stress"

        # The USD is colored by a scalar this tutorial never writes directly:
        # it supplies the tensor, and ConvertVTKToUSD derives the scalar.
        stressed = pv.read(str(results["stress_files"][0]))
        assert stressed.point_data["stress"].shape[1] == 9, (
            "compute_von_mises_stress needs a 9-component tensor to reduce"
        )

        out_dir = (
            _TUTORIAL_OUTPUT
            / "tutorial_18_duke_heart_physics_informed_motion"
            / "pm0027"
        )
        _compare_screenshots(
            results["screenshots"],
            _baseline_tools(self._class_name, out_dir, test_directories["baselines"]),
        )
