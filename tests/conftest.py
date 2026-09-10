"""Shared fixtures, and the switch that keeps the real datasets out of a fast run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from itx.data import synthetic

if TYPE_CHECKING:
    from itx.types import UpliftDataset


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--run-slow``, which turns on the tests that need a real download."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run the tests that need a downloaded dataset or a full-size fit",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip anything marked slow unless it was asked for."""
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow (downloads data or fits at full size)")
    for item in items:
        if "slow" in item.keywords:
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
