"""
Shared pytest fixtures for MONAI Physio tests.

This file defines fixtures that are available to all test modules
in the tests directory via pytest's automatic fixture discovery.
"""

import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import itk
import numpy as np
import pytest

from parameters_base import ParametersBase
from monai_physio.contour_tools import ContourTools
from monai_physio.data_download_tools import DataDownloadTools
from monai_physio.register_images_ants import RegisterImagesANTS
from monai_physio.register_images_greedy import RegisterImagesGreedy
from monai_physio.register_images_icon import RegisterImagesICON
from monai_physio.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from monai_physio.segment_chest_total_segmentator_with_contrast import (
    SegmentChestTotalSegmentatorWithContrast,
)
from monai_physio.segment_heart_simpleware import SegmentHeartSimpleware
from monai_physio.segment_nv_segment_ct_mri import SegmentNVSegmentCTMRI
from monai_physio.transform_tools import TransformTools

logger = logging.getLogger(__name__)

# ============================================================================
# Pytest Configuration - Command Line Options
# ============================================================================

# Module-level variable to store config for access in hooks
_pytest_config: Optional[pytest.Config] = None


_RUN_BUCKET_FLAGS = (
    "--run-tutorials",
    "--run-simpleware",
    "--run-slow",
    "--run-gpu",
    "--run-physicsnemo",
)


def _run_bucket_enabled(config: pytest.Config, flag: str) -> bool:
    """Return True if ``flag`` (a --run-* bucket) is on, directly or via --run-all."""
    return bool(
        config.getoption(flag, default=False)
        or config.getoption("--run-all", default=False),
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--run-tutorials",
        action="store_true",
        default=False,
        help="Run tutorial tests (data/GPU gated tutorial scripts)",
    )
    parser.addoption(
        "--run-simpleware",
        action="store_true",
        default=False,
        help=(
            "Run tests that require a local Synopsys Simpleware Medical "
            "installation (ASCardio module)"
        ),
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked 'slow' (skipped by default)",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests marked 'requires_gpu' (skipped by default)",
    )
    parser.addoption(
        "--run-physicsnemo",
        action="store_true",
        default=False,
        help=(
            "Run tests marked 'requires_physicsnemo' (need the optional "
            "[physicsnemo] extra installed)"
        ),
    )
    parser.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help=("Enable every --run-* bucket: " + ", ".join(_RUN_BUCKET_FLAGS) + "."),
    )
    parser.addoption(
        "--create-baselines",
        action="store_true",
        default=False,
        help="Create baseline files from current test outputs when missing (otherwise missing baseline fails)",
    )
    parser.addoption(
        "--max-test-seconds",
        type=float,
        default=0.0,
        help=(
            "Fail any test whose call phase runs longer than this many seconds. "
            "Unlike the pytest-timeout backstop, the test is allowed to finish, "
            "so the failure carries its duration and reaches the JUnit XML. "
            "0 (the default) disables the budget."
        ),
    )
    parser.addoption(
        "--require-tutorial-data",
        action="store_true",
        default=False,
        help=(
            "Fail rather than skip when a tutorial's dataset is missing. For "
            "runners that are supposed to have every dataset, where a skip "
            "would report a green run that tested nothing."
        ),
    )


def tutorial_data_is_required() -> bool:
    """True when --require-tutorial-data turns missing-data skips into failures."""
    if _pytest_config is None:
        return False
    return bool(_pytest_config.getoption("--require-tutorial-data", default=False))


def skip_or_fail_missing_data(reason: str) -> None:
    """Skip because a dataset is absent, or fail if the run demands it be there."""
    if tutorial_data_is_required():
        pytest.fail(f"{reason} (--require-tutorial-data is set)")
    pytest.skip(reason)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and settings."""
    global _pytest_config
    _pytest_config = config

    from monai_physio import test_tools as _test_tools

    _test_tools.set_create_baseline_if_missing(
        config.getoption("--create-baselines", default=False)
    )

    config.addinivalue_line(
        "markers",
        "tutorial: marks tests that run tutorial scripts (data/GPU gated, manual only)",
    )
    config.addinivalue_line(
        "markers",
        "requires_simpleware: marks tests that need a local Synopsys Simpleware "
        "Medical installation (skipped unless --run-simpleware is passed)",
    )
    config.addinivalue_line(
        "markers",
        "requires_physicsnemo: marks tests that need the optional "
        "[physicsnemo] extra installed (skipped unless --run-physicsnemo is passed)",
    )
    # Initialize test timing storage
    config._test_timings = {  # type: ignore[attr-defined]
        "tests": [],
        "total_time": 0.0,
        "start_time": datetime.now(),
    }


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Automatically skip bucketed tests unless their opt-in flags are passed.

    This ensures that long-running tests are opt-in only and won't run
    accidentally when running the normal test suite.
    """
    for item in items:
        if "tutorial" in item.keywords and not _run_bucket_enabled(
            config, "--run-tutorials"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="Tutorial tests require --run-tutorials (or --run-all) to run"
                )
            )
        if "requires_simpleware" in item.keywords and not _run_bucket_enabled(
            config, "--run-simpleware"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "Simpleware tests require --run-simpleware (or --run-all) "
                        "and a local Synopsys Simpleware Medical installation"
                    )
                )
            )
        if "slow" in item.keywords and not _run_bucket_enabled(config, "--run-slow"):
            item.add_marker(
                pytest.mark.skip(
                    reason="Slow tests require --run-slow (or --run-all) to run",
                )
            )
        if "requires_gpu" in item.keywords and not _run_bucket_enabled(
            config, "--run-gpu"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="GPU tests require --run-gpu (or --run-all) to run",
                )
            )
        if "requires_physicsnemo" in item.keywords and not _run_bucket_enabled(
            config, "--run-physicsnemo"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "PhysicsNeMo tests require --run-physicsnemo (or --run-all) "
                        "and the optional [physicsnemo] extra installed"
                    )
                )
            )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Any:
    """Fail a test that passed but took longer than --max-test-seconds.

    The pytest-timeout ``timeout`` setting in pyproject.toml is a backstop for a
    genuine hang: on Windows it can only kill the whole process, which loses the
    JUnit XML, the coverage data and every other test's result along with it. A
    budget checked after the test has finished costs none of that -- the run
    continues and the overrun is reported as an ordinary failure naming the
    duration and the limit.
    """
    report = yield
    if report.when != "call" or not report.passed or _pytest_config is None:
        return report
    limit = float(_pytest_config.getoption("--max-test-seconds", default=0.0) or 0.0)
    if limit > 0.0 and report.duration > limit:
        report.outcome = "failed"
        report.longrepr = (
            f"Exceeded the {limit:g}s runtime budget: took {report.duration:.1f}s. "
            "Reduce the work this test does in test mode, or raise "
            "--max-test-seconds if the cost is expected."
        )
    return report


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """
    Collect test timing information after each test completes.

    This hook is called for each phase of test execution (setup, call, teardown).
    We only collect timing from the 'call' phase which is the actual test execution.
    """
    if report.when == "call":
        # Use the module-level config reference
        if _pytest_config is None:
            return

        # Store test timing information
        test_info = {
            "nodeid": report.nodeid,
            "duration": report.duration,
            "outcome": report.outcome,
            "is_tutorial": "tutorial" in report.keywords,
        }

        _pytest_config._test_timings["tests"].append(test_info)  # type: ignore[attr-defined]
        _pytest_config._test_timings["total_time"] += report.duration  # type: ignore[attr-defined]


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """
    Print comprehensive test timing report after all tests complete.

    This hook is called at the end of the test session to display
    timing statistics for all tests, including tutorial tests.
    """
    timings = config._test_timings  # type: ignore[attr-defined]
    tests = timings["tests"]

    if not tests:
        return

    # Calculate session duration
    session_duration = datetime.now() - timings["start_time"]

    # Separate regular and tutorial tests
    regular_tests = [t for t in tests if not t["is_tutorial"]]
    tutorial_tests = [t for t in tests if t["is_tutorial"]]

    # Write the timing report
    terminalreporter.write_sep("=", "TEST TIMING REPORT", bold=True)
    terminalreporter.write_line("")

    # Session summary
    terminalreporter.write_line(f"Session Duration: {session_duration}")
    terminalreporter.write_line(
        f"Total Test Time: {timedelta(seconds=int(timings['total_time']))}"
    )
    terminalreporter.write_line(f"Total Tests: {len(tests)}")
    terminalreporter.write_line("")

    sorted_regular = sorted(regular_tests, key=lambda x: x["duration"], reverse=True)

    # Regular tests section
    if regular_tests:
        terminalreporter.write_sep("-", "Regular Tests", bold=True)
        terminalreporter.write_line(f"Count: {len(regular_tests)}")

        # Calculate total time
        regular_total = sum(t["duration"] for t in regular_tests)
        terminalreporter.write_line(
            f"Total Time: {timedelta(seconds=int(regular_total))}"
        )
        terminalreporter.write_line("")
        terminalreporter.write_line("Individual Test Times:")
        for test in sorted_regular:
            outcome_symbol = "+" if test["outcome"] == "passed" else "x"
            duration_str = _format_duration(test["duration"])
            terminalreporter.write_line(
                f"  {outcome_symbol} {duration_str:>10s}  {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Tutorial tests section
    if tutorial_tests:
        terminalreporter.write_sep("-", "Tutorial Tests", bold=True)
        terminalreporter.write_line(f"Count: {len(tutorial_tests)}")
        sorted_tutorials = sorted(
            tutorial_tests, key=lambda x: x["duration"], reverse=True
        )
        tutorial_total = sum(t["duration"] for t in tutorial_tests)
        terminalreporter.write_line(
            f"Total Time: {timedelta(seconds=int(tutorial_total))}"
        )
        terminalreporter.write_line("")
        terminalreporter.write_line("Individual Test Times:")
        for test in sorted_tutorials:
            outcome_symbol = "+" if test["outcome"] == "passed" else "x"
            duration_str = _format_duration(test["duration"])
            terminalreporter.write_line(
                f"  {outcome_symbol} {duration_str:>10s}  {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Top 10 slowest tests overall
    if len(tests) > 10:
        terminalreporter.write_sep("-", "Top 10 Slowest Tests", bold=True)
        sorted_all = sorted(tests, key=lambda x: x["duration"], reverse=True)[:10]

        for i, test in enumerate(sorted_all, 1):
            outcome_symbol = "+" if test["outcome"] == "passed" else "x"
            duration_str = _format_duration(test["duration"])
            test_type = "[TUT]" if test["is_tutorial"] else "[REG]"
            terminalreporter.write_line(
                f"  {i:2d}. {outcome_symbol} {duration_str:>10s} {test_type} {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Statistics by outcome
    passed = sum(1 for t in tests if t["outcome"] == "passed")
    failed = sum(1 for t in tests if t["outcome"] == "failed")
    skipped = sum(1 for t in tests if t["outcome"] == "skipped")

    terminalreporter.write_sep("-", "Test Outcomes", bold=True)
    terminalreporter.write_line(f"Passed:  {passed}")
    terminalreporter.write_line(f"Failed:  {failed}")
    terminalreporter.write_line(f"Skipped: {skipped}")
    terminalreporter.write_line("")


def _format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


# Directory and Data Download Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_directories() -> dict[str, Path]:
    """Set up test directories for data and results."""
    data_dir = ParametersBase().data_directory(test_mode=True)
    slicer_heart_data_dir = data_dir / "slicer_heart"
    slicer_heart_small_data_dir = data_dir / "slicer_heart_small"
    output_dir = Path(__file__).parent / "results"
    baselines_dir = Path(__file__).parent / "baselines"

    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    slicer_heart_data_dir.mkdir(parents=True, exist_ok=True)
    slicer_heart_small_data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    return {
        "data": data_dir,
        "slicer_heart_data": slicer_heart_data_dir,
        "slicer_heart_small_data": slicer_heart_small_data_dir,
        "output": output_dir,
        "baselines": baselines_dir,
    }


@pytest.fixture(scope="session")
def download_test_data(test_directories: dict[str, Path]) -> Path:
    """Download Slicer-Heart-CT data."""
    data_dir = test_directories["slicer_heart_data"]

    try:
        input_image_filename = DataDownloadTools.DownloadSlicerHeartCTData(data_dir)
    except OSError as e:
        msg = (
            f"Could not download test data: {e}. "
            "Please manually place "
            f"{DataDownloadTools.SLICER_HEART_CT_FILENAME} in {data_dir}"
        )
        if os.environ.get("CI"):
            pytest.fail(msg)
        else:
            pytest.skip(msg)

    return input_image_filename


@pytest.fixture(scope="session")
def download_kcl_heart_model(test_directories: dict[str, Path]) -> Path:
    """Download KCL-Heart-Model data."""
    data_dir = test_directories["data"] / "KCL-Heart-Model"

    try:
        data_dir = DataDownloadTools.DownloadKCLHeartModelData(data_dir)
    except OSError as e:
        msg = (
            f"Could not download KCL-Heart-Model data: {e}. "
            "See data/README.md for manual download instructions."
        )
        if os.environ.get("CI"):
            pytest.fail(msg)
        else:
            pytest.skip(msg)

    return data_dir


# ============================================================================
# Image Conversion Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_images(
    download_test_data: Path,
    test_directories: dict[str, Path],
) -> list[Any]:
    """Resample DownloadSlicerHeartCTData's 3D time series; return time points."""
    data_dir = test_directories["slicer_heart_data"]
    small_data_dir = test_directories["slicer_heart_small_data"]

    # DownloadSlicerHeartCTData() already split the 4D NRRD into
    # slice_???.mha 3D time-series volumes.
    # Resample each slice_???.mha to 1.5x1.5x1.5 mm into slicer_heart_small.
    target_spacing = [1.5, 1.5, 1.5]
    for slice_file in sorted(data_dir.glob("slice_???.mha")):
        small_file = small_data_dir / slice_file.name
        if not small_file.exists():
            logger.info("Resampling %s -> %s", slice_file.name, small_file.name)
            img = itk.imread(str(slice_file))
            input_spacing = list(img.GetSpacing())
            input_size = list(itk.size(img))
            output_size = [
                int(round(input_size[i] * input_spacing[i] / target_spacing[i]))
                for i in range(3)
            ]
            interpolator = itk.LinearInterpolateImageFunction.New(img)
            resampler = itk.ResampleImageFilter.New(Input=img)
            resampler.SetInterpolator(interpolator)
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(output_size)
            resampler.SetOutputOrigin(img.GetOrigin())
            resampler.SetOutputDirection(img.GetDirection())
            resampler.Update()
            itk.imwrite(resampler.GetOutput(), str(small_file), compression=True)
    logger.info("Resampled slice files up to date")

    slice_files = sorted(small_data_dir.glob("slice_???.mha"))
    if len(slice_files) < 3:
        pytest.skip("Resampled slice files not found.")

    images = [itk.imread(str(f)) for f in slice_files]
    logger.info("Loaded %d time points for testing", len(images))
    return images


# Tutorial Test-Data Fixtures
# ============================================================================
#
# The tutorials read <input root>/<dataset> in a full run and
# <input root>/test/<dataset> under MONAI_PHYSIO_RUNNING_AS_TEST, where the root is
# ParametersBase().data_directory() -- MONAI_PHYSIO_INPUT_DATA_DIR, or the clone's
# data/ when that is unset.  These fixtures build the latter from the former,
# small enough that a tutorial test finishes in minutes rather than hours.  Each
# skips when its source dataset is absent, so a clone that has not downloaded
# every dataset still runs whatever it can.


def _downsample_image(source: Path, destination: Path, spacing_mm: float) -> None:
    """Resample source to an isotropic pitch and write it to destination."""
    image = itk.imread(str(source))
    target_spacing = [spacing_mm] * 3
    input_spacing = list(image.GetSpacing())
    input_size = list(itk.size(image))
    output_size = [
        max(1, int(round(input_size[i] * input_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    resampler = itk.ResampleImageFilter.New(Input=image)
    resampler.SetInterpolator(itk.LinearInterpolateImageFunction.New(image))
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(output_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.Update()
    destination.parent.mkdir(parents=True, exist_ok=True)
    itk.imwrite(resampler.GetOutput(), str(destination), compression=True)


def _downsample_labelmap(source: Path, destination: Path, spacing_mm: float) -> None:
    """Resample a labelmap nearest-neighbour, so no label is interpolated away."""
    image = itk.imread(str(source))
    target_spacing = [spacing_mm] * 3
    input_spacing = list(image.GetSpacing())
    input_size = list(itk.size(image))
    output_size = [
        max(1, int(round(input_size[i] * input_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    resampler = itk.ResampleImageFilter.New(Input=image)
    resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction.New(image))
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(output_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.Update()
    destination.parent.mkdir(parents=True, exist_ok=True)
    itk.imwrite(resampler.GetOutput(), str(destination), compression=True)


# Cases kept in the test subsets.  Each list leads with the case the tutorials
# hold out, and carries enough others that a model can still be built without it
# (Tutorials 6 and 9 both refuse to run on fewer than three).
_DIRLAB_TEST_CASES = ["Case1Pack", "Case2Pack", "Case3Pack"]
_DUKE_HEART_TEST_CASES = ["pm0027", "pm0002", "pm0003", "pm0004"]


@pytest.fixture(scope="session")
def dirlab_test_data(test_directories: dict[str, Path]) -> Path:
    """Build the DIR-Lab test subset: a few cases, downsampled to 3 mm.

    Reads ``<input root>/DirLab-4DCT`` and writes ``<input root>/test/
    DirLab-4DCT``, where the root is whatever ``MONAI_PHYSIO_INPUT_DATA_DIR``
    names and defaults to the clone's ``data/``.
    """
    source_dir = ParametersBase().data_directory(test_mode=False) / "DirLab-4DCT"
    target_dir = test_directories["data"] / "DirLab-4DCT"
    if not source_dir.is_dir():
        skip_or_fail_missing_data(
            f"DIR-Lab data not found at {source_dir}. "
            "See data/DirLab-4DCT/README.md; it must be downloaded by hand."
        )

    for case_id in _DIRLAB_TEST_CASES:
        for phase_file in sorted(source_dir.glob(f"{case_id}_T??.mha")):
            small_file = target_dir / phase_file.name
            if not small_file.exists():
                logger.info("Downsampling %s -> %s", phase_file.name, small_file)
                _downsample_image(phase_file, small_file, 3.0)

    if not list(target_dir.glob("*_T??.mha")):
        skip_or_fail_missing_data(f"No DIR-Lab cases could be built under {target_dir}")
    return target_dir


@pytest.fixture(scope="session")
def duke_heart_test_data(test_directories: dict[str, Path]) -> Path:
    """Build the Duke heart test subset: a few cases, downsampled to 2 mm.

    Reads ``<input root>/Duke-Heart-4DLabelmaps`` and writes ``<input root>/
    test/Duke-Heart-4DLabelmaps``, where the root is whatever
    ``MONAI_PHYSIO_INPUT_DATA_DIR`` names and defaults to the clone's ``data/``.
    """
    source_dir = (
        ParametersBase().data_directory(test_mode=False) / "Duke-Heart-4DLabelmaps"
    )
    target_dir = test_directories["data"] / "Duke-Heart-4DLabelmaps"
    if not source_dir.is_dir():
        skip_or_fail_missing_data(f"Duke heart labelmaps not found at {source_dir}")

    for case_id in _DUKE_HEART_TEST_CASES:
        case_dir = source_dir / case_id
        if not case_dir.is_dir():
            continue
        for labelmap_file in sorted(case_dir.glob("*_labelmap.nii.gz")):
            small_file = target_dir / case_id / labelmap_file.name
            if not small_file.exists():
                logger.info("Downsampling %s -> %s", labelmap_file.name, small_file)
                _downsample_labelmap(labelmap_file, small_file, 2.0)
        # The landmarks are JSON in world coordinates, so resampling the
        # labelmaps leaves them valid and they are copied as they are.
        for landmark_file in sorted(case_dir.glob("*_landmark.mrk.json")):
            small_file = target_dir / case_id / landmark_file.name
            if not small_file.exists():
                small_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(landmark_file), str(small_file))

    built = [d for d in target_dir.glob("pm*") if d.is_dir()]
    if len(built) < 3:
        skip_or_fail_missing_data(
            f"Only {len(built)} Duke heart case(s) under {target_dir}; need 3."
        )
    return target_dir


@pytest.fixture(scope="session")
def chest_ct_test_data(test_directories: dict[str, Path]) -> Path:
    """Build the Chest-CT test subset: the single study, downsampled to 3 mm.

    Reads ``<input root>/Chest-CT`` and writes ``<input root>/test/Chest-CT``,
    where the root is whatever ``MONAI_PHYSIO_INPUT_DATA_DIR`` names and
    defaults to the clone's ``data/``.
    """
    source_file = (
        ParametersBase().data_directory(test_mode=False) / "Chest-CT" / "Chest-CT.mha"
    )
    target_dir = test_directories["data"] / "Chest-CT"
    if not source_file.exists():
        skip_or_fail_missing_data(f"Chest-CT data not found at {source_file}")

    target_file = target_dir / source_file.name
    if not target_file.exists():
        logger.info("Downsampling %s -> %s", source_file.name, target_file)
        _downsample_image(source_file, target_file, 3.0)
    return target_dir


@pytest.fixture(scope="session")
def test_labelmaps(
    segmenter_total_segmentator: SegmentChestTotalSegmentator,
    test_images: list[Any],
    test_directories: dict[str, Path],
) -> list[dict[str, Any]]:
    """
    Segment each time point with TotalSegmentator and return result dicts.
    Labelmaps are cached under slicer_heart_small_data.
    """
    small_data_dir = test_directories["slicer_heart_small_data"]
    slice_files = sorted(small_data_dir.glob("slice_???.mha"))

    results: list[dict[str, Any]] = []
    for img, slice_file in zip(test_images, slice_files):
        labelmap_file = slice_file.with_name(f"{slice_file.stem}_labelmap.mha")
        if not labelmap_file.exists():
            logger.info("Segmenting %s", slice_file.name)
            result = segmenter_total_segmentator.segment(img)
            itk.imwrite(result["labelmap"], str(labelmap_file), compression=True)

        labelmap = itk.imread(str(labelmap_file))
        labelmaps = segmenter_total_segmentator.create_anatomy_group_labelmaps(labelmap)
        results.append(
            {
                "labelmap": labelmap,
                "lung": labelmaps["lung"],
                "heart": labelmaps["heart"],
                "major_vessels": labelmaps["major_vessels"],
                "bone": labelmaps["bone"],
                "soft_tissue": labelmaps["soft_tissue"],
                "other": labelmaps["other"],
            }
        )

    return results


@pytest.fixture(scope="session")
def test_transforms(
    registrar_ANTS: RegisterImagesANTS,
    test_images: list[Any],
    test_directories: dict[str, Path],
) -> dict[str, Any]:
    """
    Perform ANTs registration and return results.
    Generates them if not already present, otherwise loads from disk.
    Transforms are cached under slicer_heart_small_data.
    """
    small_data_dir = test_directories["slicer_heart_small_data"]
    frame_tag = "001_to_007"
    inverse_transform_path = small_data_dir / f"ants_inverse_transform_{frame_tag}.hdf"
    forward_transform_path = small_data_dir / f"ants_forward_transform_{frame_tag}.hdf"

    if inverse_transform_path.exists() and forward_transform_path.exists():
        logger.info("Loading existing ANTs registration results")
        try:
            inverse_transform = itk.transformread(str(inverse_transform_path))
            forward_transform = itk.transformread(str(forward_transform_path))
            return {
                "inverse_transform": inverse_transform,
                "forward_transform": forward_transform,
            }
        except (RuntimeError, Exception) as e:
            logger.warning("Error loading transforms: %s; regenerating", e)
            inverse_transform_path.unlink(missing_ok=True)
            forward_transform_path.unlink(missing_ok=True)

    # Perform registration if files don't exist or loading failed
    logger.info("Performing ANTs registration")
    fixed_image = test_images[7]
    moving_image = test_images[1]

    registrar_ANTS.set_fixed_image(fixed_image)
    result = registrar_ANTS.register(moving_image=moving_image)

    inverse_transform = result["inverse_transform"]
    forward_transform = result["forward_transform"]

    itk.transformwrite(inverse_transform, str(inverse_transform_path), compression=True)
    itk.transformwrite(forward_transform, str(forward_transform_path), compression=True)
    return {
        "inverse_transform": inverse_transform,
        "forward_transform": forward_transform,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def segmenter_total_segmentator() -> SegmentChestTotalSegmentator:
    """Create a SegmentChestTotalSegmentator instance."""
    return SegmentChestTotalSegmentator()


@pytest.fixture(scope="session")
def segmenter_total_segmentator_with_contrast() -> (
    SegmentChestTotalSegmentatorWithContrast
):
    """Create a SegmentChestTotalSegmentatorWithContrast instance."""
    return SegmentChestTotalSegmentatorWithContrast()


@pytest.fixture(scope="session")
def segmenter_nv_segment_ct_mri() -> SegmentNVSegmentCTMRI:
    """Create a SegmentNVSegmentCTMRI instance."""
    return SegmentNVSegmentCTMRI()


@pytest.fixture(scope="session")
def segmenter_simpleware() -> SegmentHeartSimpleware:
    """Create a SegmentHeartSimpleware instance."""
    return SegmentHeartSimpleware()


@pytest.fixture(scope="session")
def contour_tools() -> ContourTools:
    """Create a ContourTools instance."""
    return ContourTools()


@pytest.fixture(scope="session")
def registrar_ANTS() -> RegisterImagesANTS:
    """Create a RegisterImagesANTS instance."""
    return RegisterImagesANTS()


@pytest.fixture(scope="session")
def registrar_greedy() -> RegisterImagesGreedy:
    """Create a RegisterImagesGreedy instance."""
    return RegisterImagesGreedy()


@pytest.fixture(scope="session")
def registrar_ICON() -> RegisterImagesICON:
    """Create a RegisterImagesICON instance."""
    return RegisterImagesICON()


@pytest.fixture(scope="session")
def transform_tools() -> TransformTools:
    """Create a TransformTools instance."""
    return TransformTools()


class KnownShiftCase:
    """A registration case whose correct answer is known exactly.

    ``moving`` is built by resampling ``fixed`` through a translation of
    ``shift_mm``, so ``moving(q) == fixed(q + shift_mm)``. Warping ``moving``
    back onto the fixed grid therefore requires a ``forward_transform`` of
    ``-shift_mm``, which gives an absolute accuracy target instead of the
    "did it return something" checks that let a sign error go unnoticed.
    """

    def __init__(self, fixed_image: itk.Image, shift_mm: tuple[float, float, float]):
        """Build the shifted pair.

        Args:
            fixed_image: Image used as the registration target.
            shift_mm: Content displacement applied to build the moving image.
                Use a different magnitude and sign per axis so an axis swap or a
                sign flip cannot pass.
        """
        self.transform_tools = TransformTools()
        self.fixed = fixed_image
        self.shift_mm = shift_mm
        self.expected_displacement = np.array([-v for v in shift_mm])

        shift = itk.TranslationTransform[itk.D, 3].New()
        shift.SetOffset(list(shift_mm))
        self.moving = self.transform_tools.transform_image(
            fixed_image, shift, fixed_image, interpolation_method="linear"
        )

        size = itk.size(fixed_image)
        self._center = list(
            fixed_image.TransformIndexToPhysicalPoint(
                [int(size[i]) // 2 for i in range(3)]
            )
        )
        # Score over the brightest 30% of the fixed image (tissue and blood
        # pool); background air correlates trivially and would mask errors.
        self._fixed_array = itk.array_from_image(fixed_image)
        self._foreground = self._fixed_array >= np.percentile(self._fixed_array, 70)

    def center_error_mm(self, forward_transform: itk.Transform) -> float:
        """Distance, in mm, between the recovered and true displacement."""
        displacement = np.array(
            list(forward_transform.TransformPoint(self._center))
        ) - np.array(self._center)
        return float(np.linalg.norm(displacement - self.expected_displacement))

    def foreground_ncc(self, forward_transform: itk.Transform) -> float:
        """Normalized cross-correlation after warping moving onto the fixed grid."""
        warped = self.transform_tools.transform_image(
            self.moving, forward_transform, self.fixed, interpolation_method="linear"
        )
        moved = itk.array_from_image(warped)[self._foreground]
        target = self._fixed_array[self._foreground]
        moved = moved - moved.mean()
        target = target - target.mean()
        denominator = np.sqrt((moved**2).sum() * (target**2).sum())
        return float((moved * target).sum() / denominator) if denominator else 0.0

    def unregistered_ncc(self) -> float:
        """Baseline score with no registration, for a floor to beat."""
        return self.foreground_ncc(itk.TranslationTransform[itk.D, 3].New())


@pytest.fixture(scope="session")
def known_shift_case(test_images: list[Any]) -> KnownShiftCase:
    """A moving/fixed pair separated by a known (6, -4, 3) mm shift."""
    return KnownShiftCase(test_images[0], (6.0, -4.0, 3.0))


class KnownAffineCase:
    """A registration case with a known *non-identity* affine, placed anywhere.

    :class:`KnownShiftCase` moves content by a pure translation, so the affine's
    linear block is the identity. That makes it blind to how the linear block is
    interpreted -- about the world origin, or about the data -- because both
    readings agree when the block is ``I``.

    They disagree everywhere else, and the disagreement grows with distance from
    the origin, since a linear block applied about the origin displaces a point
    in proportion to ``|p|``. On a grid at ``z ~ 1800 mm`` -- CT table
    coordinates, where this project's cardiac cohorts live -- a few degrees of
    rotation is tens of millimeters. So this case rotates as well as translates
    and can be placed far from the origin, which is what lets it tell a correct
    conversion from a misread convention.

    ``moving`` is built by resampling ``fixed`` through ``A``, so
    ``moving(q) == fixed(A(q))``. Warping ``moving`` back onto the fixed grid
    therefore needs a ``forward_transform`` of ``A^-1``, which is the exact
    answer every probe is measured against.
    """

    def __init__(
        self,
        rotation_degrees: tuple[float, float, float] = (2.0, 1.0, 3.0),
        translation_mm: tuple[float, float, float] = (4.0, -3.0, 2.0),
        origin_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Build the pair.

        Args:
            rotation_degrees: Rotation about each axis, applied about the image
                centroid. Different per axis so an axis swap cannot pass.
            translation_mm: Translation applied after the rotation.
            origin_offset_mm: Placed the grid's origin here, moving the data in
                world space. Use it to sit far from the world origin.
        """
        self.transform_tools = TransformTools()

        self.fixed = self._synthetic_volume(origin_offset_mm)
        self.origin_offset_mm = origin_offset_mm

        size = itk.size(self.fixed)
        self._center = np.array(
            list(
                self.fixed.TransformIndexToPhysicalPoint(
                    [int(size[i]) // 2 for i in range(3)]
                )
            )
        )

        matrix = self._rotation_matrix(rotation_degrees)
        # A rotation about the image centroid, written as a world affine: the
        # translation absorbs the center, which is the form a bare 4x4
        # homogeneous matrix carries.
        offset = self._center - matrix @ self._center + np.asarray(translation_mm)

        self.applied = itk.AffineTransform[itk.D, 3].New()
        self.applied.SetCenter(itk.Point[itk.D, 3]())
        self.applied.SetMatrix(itk.GetMatrixFromArray(matrix))
        applied_translation = itk.Vector[itk.D, 3]()
        for i in range(3):
            applied_translation[i] = float(offset[i])
        self.applied.SetTranslation(applied_translation)

        self.expected = itk.AffineTransform[itk.D, 3].New()
        self.applied.GetInverse(self.expected)

        self.moving = self.transform_tools.transform_image(
            self.fixed, self.applied, self.fixed, interpolation_method="linear"
        )

    @staticmethod
    def _synthetic_volume(origin_mm: tuple[float, float, float]) -> itk.Image:
        """Return a blocky test volume whose origin sits at *origin_mm*.

        Synthetic rather than a real scan on purpose. The question here is only
        whether recovery depends on distance from the world origin, and a real
        cardiac volume answers it unreliably: Greedy's optimizer sometimes fails
        outright on one (``vnl_lbfgs`` reports a Netlib failure and the recovered
        affine is off by more than a hundred millimeters), which swamps the
        millimeter-scale effect being measured. A high-contrast block converges
        every time, so a failure here means what the assertion says it means.
        """
        volume = np.zeros((70, 70, 70), dtype=np.float32)
        volume[15:55, 15:55, 15:55] = 400.0
        volume[25:45, 20:50, 22:48] = 900.0
        image = itk.GetImageFromArray(volume)
        image.SetSpacing([1.5, 1.5, 1.5])
        image.SetOrigin(list(origin_mm))
        return image

    @staticmethod
    def _rotation_matrix(degrees: tuple[float, float, float]) -> np.ndarray:
        """Return the rotation matrix for x, then y, then z rotations."""
        ax, ay, az = (np.deg2rad(value) for value in degrees)
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(ax), -np.sin(ax)],
                [0.0, np.sin(ax), np.cos(ax)],
            ]
        )
        ry = np.array(
            [
                [np.cos(ay), 0.0, np.sin(ay)],
                [0.0, 1.0, 0.0],
                [-np.sin(ay), 0.0, np.cos(ay)],
            ]
        )
        rz = np.array(
            [
                [np.cos(az), -np.sin(az), 0.0],
                [np.sin(az), np.cos(az), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation: np.ndarray = rz @ ry @ rx
        return rotation

    def probe_points(self) -> list[list[float]]:
        """Return points spread across the volume, not just its center.

        A linear-block misreading is invisible at one point -- any single
        displacement can be absorbed by the translation -- and grows with
        distance from the origin, so the spread is what catches it.
        """
        size = itk.size(self.fixed)
        indices = [
            [int(size[0]) // 4, int(size[1]) // 4, int(size[2]) // 4],
            [3 * int(size[0]) // 4, int(size[1]) // 4, int(size[2]) // 4],
            [int(size[0]) // 4, 3 * int(size[1]) // 4, int(size[2]) // 4],
            [int(size[0]) // 4, int(size[1]) // 4, 3 * int(size[2]) // 4],
            [3 * int(size[0]) // 4, 3 * int(size[1]) // 4, 3 * int(size[2]) // 4],
        ]
        points = [list(self.fixed.TransformIndexToPhysicalPoint(i)) for i in indices]
        points.append(self._center.tolist())
        return points

    def probe_errors_mm(self, forward_transform: itk.Transform) -> np.ndarray:
        """Return the per-probe distance, in mm, from the exact answer."""
        errors = []
        for point in self.probe_points():
            recovered = np.array(list(forward_transform.TransformPoint(point)))
            exact = np.array(list(self.expected.TransformPoint(point)))
            errors.append(float(np.linalg.norm(recovered - exact)))
        return np.asarray(errors)


@pytest.fixture(scope="session")
def known_affine_case_near_origin() -> KnownAffineCase:
    """A known rotation plus translation, on a grid near the world origin."""
    return KnownAffineCase()


@pytest.fixture(scope="session")
def known_affine_case_far_from_origin() -> KnownAffineCase:
    """The same known affine, on a grid at ``z ~ 1800 mm``.

    This is where the Duke heart cohort lives, and where an origin-based reading
    of the linear block differs from a data-centered one by tens of millimeters.
    """
    return KnownAffineCase(origin_offset_mm=(0.0, 0.0, 1800.0))
