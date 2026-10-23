"""Tests for MONAIPhysioBase progress reporting and caller-attached log handlers."""

import logging

import pytest

from monai_physio.monai_physio_base import MONAIPhysioBase


class _Recorder:
    """Callback that records every (fraction, message) pair it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, str]] = []

    def __call__(self, fraction: float, message: str) -> None:
        self.calls.append((fraction, message))

    @property
    def fractions(self) -> list[float]:
        """The fractions received, in order."""
        return [fraction for fraction, _ in self.calls]

    @property
    def messages(self) -> list[str]:
        """The messages received, in order."""
        return [message for _, message in self.calls]


@pytest.fixture
def recorder() -> _Recorder:
    recorder = _Recorder()
    MONAIPhysioBase.set_progress_callback(recorder)
    return recorder


def test_without_callback_nothing_is_reported() -> None:
    obj = MONAIPhysioBase(class_name="ProgressTest")
    obj._report_progress(0.5, "ignored")
    obj.log_progress(1, 2, log=False)


def test_report_clamps_and_never_decreases(recorder: _Recorder) -> None:
    obj = MONAIPhysioBase(class_name="ProgressTest")
    for fraction in (-0.2, 0.3, 0.1, 0.6, 1.4):
        obj._report_progress(fraction, "step")
    assert recorder.fractions == [0.0, 0.3, 0.3, 0.6, 1.0]


def test_range_maps_and_prefixes(recorder: _Recorder) -> None:
    obj = MONAIPhysioBase(class_name="ProgressTest")
    with obj._progress_range(0.2, 0.6, "ICP"):
        obj._report_progress(0.0, "start")
        obj._report_progress(0.5, "half")
        obj._report_progress(1.0, "done")
    obj._report_progress(0.7, "after")
    assert recorder.fractions == pytest.approx([0.2, 0.4, 0.6, 0.7])
    assert recorder.messages == ["ICP: start", "ICP: half", "ICP: done", "after"]


def test_skipped_range_is_jumped_over(recorder: _Recorder) -> None:
    """Ranges are fixed; a step that does not run leaves a gap, not a rescale."""
    obj = MONAIPhysioBase(class_name="ProgressTest")
    with obj._progress_range(0.0, 0.5, "first"):
        obj._report_progress(1.0, "done")
    # A disabled step would own 0.5-0.8.
    with obj._progress_range(0.8, 1.0, "last"):
        obj._report_progress(0.5, "half")
    assert recorder.fractions == pytest.approx([0.5, 0.9])


def test_another_object_reports_into_the_enclosing_range(recorder: _Recorder) -> None:
    """A registrar run inside a workflow's range needs no plumbing."""
    workflow = MONAIPhysioBase(class_name="Workflow")
    registrar = MONAIPhysioBase(class_name="Registrar")
    with (
        workflow._progress_range(0.2, 0.6, "Fit"),
        workflow._progress_range(0.5, 1.0, "PCA"),
    ):
        registrar._start_progress("starting")
        registrar.log_progress(50, 100, "iteration", log=False)
    assert recorder.fractions == pytest.approx([0.4, 0.5])
    assert recorder.messages == ["Fit: PCA: starting", "Fit: PCA: iteration: 50/100"]


def test_a_new_run_starts_again_at_zero(recorder: _Recorder) -> None:
    obj = MONAIPhysioBase(class_name="ProgressTest")
    for _ in range(2):
        obj._start_progress("start")
        obj._report_progress(1.0, "done")
    assert recorder.fractions == [0.0, 1.0, 0.0, 1.0]


def test_log_progress_logs_unless_told_not_to(
    recorder: _Recorder, caplog: pytest.LogCaptureFixture
) -> None:
    obj = MONAIPhysioBase(class_name="ProgressTest")
    logger = logging.getLogger("MONAIPhysio")
    logger.addHandler(caplog.handler)
    try:
        obj.log_progress(1, 4, "Frames")
        obj.log_progress(2, 4, "Frames", log=False)
    finally:
        logger.removeHandler(caplog.handler)
    assert recorder.calls == [(0.25, "Frames: 1/4"), (0.5, "Frames: 2/4")]
    assert [r.getMessage() for r in caplog.records] == [
        "ProgressTest Frames: 1/4 (25.0%)"
    ]


def test_caller_handler_survives_logger_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handler attached before the first MONAI Physio object is kept."""
    logger = logging.getLogger("MONAIPhysio")
    monkeypatch.setattr(MONAIPhysioBase, "_logger_initialized", False)
    monkeypatch.setattr(MONAIPhysioBase, "_class_filter", MONAIPhysioBase._class_filter)
    monkeypatch.setattr(logger, "handlers", list(logger.handlers))
    # Logger caches isEnabledFor() results per level; restore that cache with
    # the level so later tests do not see the level set here.
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(logger, "_cache", {})

    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    caller_handler = _Collect(level=logging.DEBUG)
    logger.addHandler(caller_handler)

    obj = MONAIPhysioBase(class_name="ProgressTest")
    MONAIPhysioBase.set_log_level(logging.INFO)
    obj.log_info("stage %.3f -> %s", 0.1, "file.vtp")

    assert caller_handler in logger.handlers
    assert caller_handler.level == logging.DEBUG
    assert [record.getMessage() for record in records] == [
        "ProgressTest stage 0.100 -> file.vtp"
    ]
