"""Opt-in GPU tests for experiment folders, as ``tests/conftest.py`` does.

When ``tests/`` is collected in the same run, whichever conftest loads first
adds --run-gpu and the other skips it; ``tests/conftest.py`` adds --run-all.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --run-gpu unless ``tests/conftest.py`` has added it."""
    try:
        parser.addoption(
            "--run-gpu",
            action="store_true",
            default=False,
            help="Run tests marked 'requires_gpu' (skipped by default)",
        )
    except ValueError:
        pass


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests marked 'requires_gpu' unless --run-gpu (or --run-all) is given."""
    if config.getoption("--run-gpu") or config.getoption("--run-all", default=False):
        return
    skip = pytest.mark.skip(reason="GPU tests require --run-gpu to run")
    for item in items:
        if "requires_gpu" in item.keywords:
            item.add_marker(skip)
