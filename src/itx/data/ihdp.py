"""IHDP: 747 units, 100 replicates, simulated outcomes with the truth written down.

The covariates are real (Infant Health and Development Program), the outcomes are
simulated from a published response surface, and a non-random subset of the treated units
was removed to induce confounding. That is what makes it useful here: the individual
treatment effect ``mu1 - mu0`` is known for every unit, so PEHE can be measured rather
than argued about, and the confounding is real enough that an estimator can fail.

The file is the standard ``ihdp_npci_1-100`` benchmark used by CFRNet, CEVAE and
Dragonnet, so the numbers this package reports are comparable with the literature. It
ships as a pre-split pair of archives; both are loaded and rejoined, because this project
applies its own committed splits to every dataset (PLAN.md section 4). Card:
docs/data/ihdp.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from itx.data.download import fetch
from itx.types import FloatArray, UpliftDataset

if TYPE_CHECKING:
    from collections.abc import Iterator

N_REPLICATES = 100
N_UNITS = 747
N_COVARIATES = 25
FEATURES = tuple(f"x{i + 1}" for i in range(N_COVARIATES))

#: Columns 7 to 25 of the IHDP covariate block are indicators (x14 takes the values 1 and 2
#: rather than 0 and 1); the first six are continuous. Recorded here so a reader does not
#: have to infer it from the values.
CONTINUOUS_FEATURES = FEATURES[:6]
BINARY_FEATURES = FEATURES[6:]

_REJOINED_KEYS = ("x", "t", "yf", "mu0", "mu1")


def load_ihdp(replicate: int = 0) -> UpliftDataset:
    """Load one IHDP replicate.

    Args:
        replicate: Replicate index, 0 to 99.

    Returns:
        The dataset, with ``true_effect`` set to ``mu1 - mu0`` and ``propensity`` left
        None: assignment was not random and the treated units were thinned on purpose.

    Raises:
        ValueError: If the replicate index is out of range.
    """
    if not 0 <= replicate < N_REPLICATES:
        msg = f"replicate must be in 0..{N_REPLICATES - 1}, got {replicate}"
        raise ValueError(msg)

    arrays = _rejoined()
    covariates = arrays["x"][:, :, replicate]
    features = pl.DataFrame(
        {name: covariates[:, index].astype(np.float64) for index, name in enumerate(FEATURES)}
    )
    true_effect: FloatArray = (
        arrays["mu1"][:, replicate] - arrays["mu0"][:, replicate]
    ).astype(np.float64)
    return UpliftDataset(
        name="ihdp",
        features=features,
        treatment=arrays["t"][:, replicate].astype(np.int64),
        outcome=arrays["yf"][:, replicate].astype(np.float64),
        true_effect=true_effect,
        replicate=replicate,
    )


def load_ihdp_replicates(count: int = N_REPLICATES) -> Iterator[UpliftDataset]:
    """Yield the first ``count`` replicates in order.

    Args:
        count: How many replicates to yield, 1 to 100. Metrics on IHDP are averaged over
            replicates with an interval taken across them, so this is the usual entry point.

    Yields:
        One dataset per replicate.
    """
    for replicate in range(min(count, N_REPLICATES)):
        yield load_ihdp(replicate)


def _rejoined() -> dict[str, FloatArray]:
    """Concatenate the shipped train and test archives back into all 747 units."""
    train = np.load(fetch("ihdp-train", quiet=True))
    test = np.load(fetch("ihdp-test", quiet=True))
    joined = {
        key: np.concatenate([train[key], test[key]], axis=0).astype(np.float64)
        for key in _REJOINED_KEYS
    }
    shape = joined["t"].shape
    if shape != (N_UNITS, N_REPLICATES):
        msg = f"IHDP: expected {(N_UNITS, N_REPLICATES)} units by replicates, got {shape}"
        raise ValueError(msg)
    return joined
