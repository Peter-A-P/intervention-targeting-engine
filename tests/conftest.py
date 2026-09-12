"""Shared fixtures, and the switch that keeps the real datasets out of a fast run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from itx.data import synthetic

if TYPE_CHECKING:
    from itx.types import UpliftDataset


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the two switches that turn on the tests which need a real download."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run the tests that need a downloaded dataset or a full-size fit",
    )
    # Separate from --run-slow rather than a degree of it. Criteo and Lenta are 435 MB and
    # a scan of fourteen million rows, and the CI job that runs the rest of the suite is
    # not allowed to pull them (PLAN.md changes 31 and 33). One flag could not express
    # that, so a run has to ask for these by name.
    parser.addoption(
        "--run-large",
        action="store_true",
        default=False,
        help="run the tests that load Criteo or Lenta (435 MB of downloads)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip anything marked slow or large unless that marker was asked for."""
    skips = {
        "slow": ("--run-slow", "downloads data or fits at full size"),
        "large": ("--run-large", "435 MB of downloads: Criteo and Lenta"),
    }
    for marker, (option, why) in skips.items():
        if config.getoption(option):
            continue
        skip = pytest.mark.skip(reason=f"needs {option} ({why})")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def constant_data() -> UpliftDataset:
    """A dataset whose treatment effect is 1.0 for every unit."""
    return synthetic.constant_effect(6_000, effect=1.0, seed=1)


@pytest.fixture(scope="session")
def heterogeneous_data() -> UpliftDataset:
    """A dataset whose effect depends on two of five covariates."""
    return synthetic.heterogeneous_effect(8_000, seed=2)


@pytest.fixture(scope="session")
def binary_data() -> UpliftDataset:
    """A binary-outcome dataset with a known per-unit uplift."""
    return synthetic.binary_outcome(12_000, seed=3)
