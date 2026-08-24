"""Tutorial tests that run each tutorial end-to-end and compare screenshots.

Each test class maps to one tutorial script.  Tests are gated behind
``--run-tutorials`` (handled by conftest.py) and require the relevant dataset
to be present (see data/README.md).

Screenshot comparison uses the existing ITK-based baseline infrastructure:

1. Each tutorial script saves PNGs directly to its ``OUTPUT_DIR``.
2. ``TestTools.compare_result_to_baseline_image`` reads each PNG from that
   directory and compares it against a stored baseline with loose tolerances.

Run all tutorial tests::

    pytest tests/test_tutorials.py --run-tutorials -v

Create baselines on first run::

    pytest tests/test_tutorials.py --run-tutorials --create-baselines -v
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from physiotwin4d.test_tools import TestTools

# Tolerances for screenshot comparison. Loose to survive minor rendering
# differences across OS / GPU / driver versions.
_PX_TOL = 10.0  # per-pixel absolute error (0-255 range)
_MAX_PX = 2000  # maximum number of pixels allowed above _PX_TOL
_TOT_TOL = float("inf")  # use the pixel-count criterion only
_REPO_ROOT = Path(__file__).parent.parent
# Where every tutorial writes, and where Tutorial 9 trains its networks to.
_TUTORIAL_OUTPUT = _REPO_ROOT / "tutorials" / "output"
_TUTORIAL_WEIGHTS = _REPO_ROOT / "tutorials" / "network_weights"


@pytest.fixture(autouse=True)
def _enable_tutorial_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run tutorials against repo data/test through TestTools mode switching."""
    monkeypatch.setenv("PHYSIOTWIN_RUNNING_AS_TEST", "1")


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
    """Skip unless ``directory`` holds at least one entry matching ``pattern``."""
    if not list(directory.glob(pattern)):
        pytest.skip(f"No {pattern} under {directory}. {reason}")


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
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_01_heart"
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
        # This script reads the full dataset directory in either mode.
        _require_files(
            _REPO_ROOT / "data" / "DirLab-4DCT",
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
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
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_03_heart"
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
        # Match the phase files the script itself globs, not a directory layout
        # it never uses.
        dirlab_dir = test_directories["data"] / "DirLab-4DCT"
        if not list(dirlab_dir.glob("Case1Pack_T??.mha")):
            pytest.skip(
                "DirLab-4DCT Case1Pack phases not downloaded. See data/README.md "
                "for instructions."
            )

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_03_lung"
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
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_04_heart"
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
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
        input_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_04_heart"
        if not list(input_dir.glob("patient_*.vtp")):
            _run_tutorial_script("tutorial_04_heart_ct_to_vtk.py")
            assert list(input_dir.glob("patient_*.vtp")), (
                f"Tutorial 4 bootstrap did not create surfaces in: {input_dir}"
            )

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_05_heart"
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
        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_06_heart"
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
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
        self, test_directories: dict[str, Path], download_kcl_heart_model: Path
    ) -> None:
        # The patient scan comes from DIR-Lab, which must be acquired manually.
        if not (
            test_directories["data"] / "DirLab-4DCT" / "Case1Pack_T70.mha"
        ).exists():
            pytest.skip(
                "DirLab-4DCT Case1Pack_T70 not downloaded. See data/README.md "
                "for instructions."
            )

        pca_json = (
            _REPO_ROOT / "tutorials" / "output" / "tutorial_06_heart" / "pca_model.json"
        )
        if not pca_json.exists():
            _run_tutorial_script("tutorial_06_heart_create_statistical_model.py")
            assert pca_json.exists(), (
                "Tutorial 6 bootstrap did not create the expected PCA model file: "
                f"{pca_json}"
            )

        out_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_07_heart"
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


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial07DukeHeartFitStatisticalModelToPatient:
    """End-to-end test for tutorial_07_duke_heart_fit_statistical_model_to_patient."""

    _class_name = "tutorial_07_duke_heart"  # the script's project_name

    def test_run(self, test_directories: dict[str, Path]) -> None:
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


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial07LungFitStatisticalModelToPatient:
    """End-to-end test for tutorial_07_lung_fit_statistical_model_to_patient.py."""

    _class_name = "tutorial_07_lung"  # the script's project_name

    def test_run(self, test_directories: dict[str, Path]) -> None:
        # The lung variant fits the ungated Chest-CT scan, which the
        # download CLI provides, rather than a gated DIR-Lab phase.
        _require_files(
            test_directories["data"] / "Chest-CT",
            "Chest-CT.mha",
            "Fetch it with: physiotwin4d-download-data Chest-CT.",
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


# -----------------------------------------------------------------------------
# Tutorial 8 - Fit the model to every gated frame of several patients
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
class TestTutorial08DukeHeartFitModelTo4DPatients:
    """End-to-end test for tutorial_08_duke_heart_fit_model_to_4d_patients.py."""

    _class_name = "tutorial_08_duke_heart_fit_model_to_4d_patients"

    def test_run(self, test_directories: dict[str, Path]) -> None:
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
        # This script reads the full dataset directory in either mode.
        _require_files(
            _REPO_ROOT / "data" / "DirLab-4DCT",
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
# Both need the optional [physicsnemo] extra and the Tutorial 8 fitted meshes,
# so they skip rather than fail when either is absent.
# -----------------------------------------------------------------------------


def _require_physicsnemo() -> None:
    """Skip unless both MGN dependencies are installed."""
    if importlib.util.find_spec("physicsnemo") is None:
        pytest.skip("PhysicsNeMo not installed (optional [physicsnemo] extra).")
    if importlib.util.find_spec("torch_geometric") is None:
        pytest.skip(
            "PyTorch Geometric not installed; the MGN trainer needs it in addition "
            'to PhysicsNeMo. Install with: pip install "physiotwin4d[physicsnemo]" '
            "&& pip install torch-geometric"
        )


def _require_physicsnemo_and_tutorial_08() -> Path:
    """Skip unless the MGN dependencies and three Tutorial 8 cases are present."""
    _require_physicsnemo()
    data_dir = _REPO_ROOT / "tutorials" / "output" / "tutorial_08_lung"
    if len(list(data_dir.glob("Case*Pack"))) < 3:
        pytest.skip(
            "Fewer than three Tutorial 8 cases under tutorials/output/tutorial_08_lung. "
            "Run tutorial_08_lung_fit_model_to_4d_patients.py first."
        )
    return data_dir


def _require_physicsnemo_and_tutorial_08_duke() -> Path:
    """Skip unless the MGN dependencies and the Tutorial 8 Duke cases are present."""
    _require_physicsnemo()
    data_dir = _TUTORIAL_OUTPUT / "tutorial_08_duke_heart"
    if not list(data_dir.glob("pm*")):
        pytest.skip(
            "No Tutorial 8 cases under tutorials/output/tutorial_08_duke_heart. "
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
            results_dir=_REPO_ROOT / "tutorials" / "output" / "tutorial_09_lung_mgn",
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

        # ParametersLungCTDirLab.mgn_weights_dir, where Tutorial 9 trains to.
        model_dir = (
            _REPO_ROOT / "tutorials" / "network_weights" / "physicsnemo_mgn_lung_motion"
        )
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

        out_dir = (
            _REPO_ROOT / "tutorials" / "output" / "tutorial_10_lung_mgn" / "Case1Pack"
        )
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_physicsnemo_and_tutorial_08_duke()

        # ParametersDukeHeartLabelmaps.mgn_weights_dir, where Tutorial 9 trains to.
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
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


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial11LungEvaluatePhysicsNeMo:
    """End-to-end test for tutorial_11_lung_evaluate_physicsnemo.py."""

    _class_name = "tutorial_11_lung_evaluate_physicsnemo"

    def test_run(self, test_directories: dict[str, Path]) -> None:
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


# -----------------------------------------------------------------------------
# Tutorial 12 - End-to-end inference, from a raw image to a moving anatomy
# -----------------------------------------------------------------------------


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial12DukeHeartEndToEndInference:
    """End-to-end test for tutorial_12_duke_heart_end_to_end_inference.py."""

    _class_name = "tutorial_12_duke_heart_end_to_end_inference"

    def test_run(self, test_directories: dict[str, Path]) -> None:
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


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial12LungEndToEndInference:
    """End-to-end test for tutorial_12_lung_end_to_end_inference.py."""

    _class_name = "tutorial_12_lung_end_to_end_inference"

    def test_run(self, test_directories: dict[str, Path]) -> None:
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
        _require_physicsnemo()
        # The ungated clinical scan both rhythms are inferred onto.
        _require_files(
            test_directories["data"] / "Chest-CT",
            "Chest-CT.mha",
            "Fetch it with: physiotwin4d-download-data Chest-CT.",
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

    def test_run(self, test_directories: dict[str, Path]) -> None:
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


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial14LungShapeParameterSweep:
    """End-to-end test for tutorial_14_lung_shape_parameter_sweep.py."""

    _class_name = "tutorial_14_lung_shape_parameter_sweep"

    def test_run(self, test_directories: dict[str, Path]) -> None:
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


# Both Tutorial 15 variants clamp themselves to this many folds under
# TestTools.running_as_test, which the autouse fixture above turns on.
_TUTORIAL_15_TEST_MODE_FOLDS = 2


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial15LungLeaveOneOut:
    """End-to-end test for tutorial_15_lung_leave_one_out.py."""

    _class_name = "tutorial_15_lung_leave_one_out"

    def test_run(self, test_directories: dict[str, Path]) -> None:
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


@pytest.mark.tutorial
@pytest.mark.slow
@pytest.mark.requires_physicsnemo
class TestTutorial15DukeHeartLeaveOneOut:
    """End-to-end test for tutorial_15_duke_heart_leave_one_out.py."""

    _class_name = "tutorial_15_duke_heart_leave_one_out"

    def test_run(self, test_directories: dict[str, Path]) -> None:
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
