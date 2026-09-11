"""ACIC 2016: 4,802 units, 58 real covariates, simulated treatment and outcome.

The second ground-truth dataset, and it is in the benchmark because it disagrees with the
first one in the ways that matter. IHDP is 747 units with an effect whose standard
deviation is a fifth of its mean; ACIC is six times larger with an effect whose standard
deviation is nearly twice its mean. An estimator that looks good on IHDP alone has been
shown to work on a small, nearly homogeneous problem, which is one problem rather than a
class of them.

The covariates are real, from a study of twins linked to birth records. The treatment
assignment and the outcome are both simulated, by the organisers of the 2016 Atlantic
Causal Inference Conference competition, from response surfaces that were not published in
advance: the competition existed to find out which methods worked without being told the
answer. Each replicate is a different simulation setting, so the ten are ten different
problems over the same people rather than ten draws from one problem, and their true
average effects range from 1.5 to 4.8.

Both potential outcomes are in the file, so the observed outcome is assembled here rather
than read: ``y1`` for the treated units and ``y0`` for the control ones. The true individual
effect is ``mu1 - mu0``, the difference of the noiseless response surfaces, which is what
PEHE should be scored against; differencing the noisy ``y1`` and ``y0`` instead would add
the outcome noise to the target twice over and quietly inflate every estimator's error.

Card: docs/data/acic.md.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from itx.data.download import fetch
from itx.types import FloatArray, UpliftDataset

if TYPE_CHECKING:
    from collections.abc import Iterator

N_UNITS = 4_802
N_REPLICATES = 10
N_COVARIATES = 58

#: The three covariates the competition published as letters rather than numbers. They are
#: integer-encoded here, in sorted order so the encoding does not depend on row order, and
#: declared to LightGBM as categories rather than left to look like a quantity.
CATEGORICAL: tuple[str, ...] = ("x_2", "x_21", "x_24")


def load_acic(replicate: int = 0) -> UpliftDataset:
    """Load one ACIC 2016 replicate.

    Args:
        replicate: Replicate index, 0 to 9. Each is a different simulation setting.

    Returns:
        The dataset, with ``true_effect`` set to ``mu1 - mu0`` and ``propensity`` left
        None: assignment was simulated from the covariates and the analyst is not told how.

    Raises:
        ValueError: If the replicate index is out of range.
    """
    if not 0 <= replicate < N_REPLICATES:
        msg = f"replicate must be in 0..{N_REPLICATES - 1}, got {replicate}"
        raise ValueError(msg)

    features = _covariates()
    outcomes = pl.read_csv(
        fetch(f"acic-zymu-{replicate + 1}", quiet=True), infer_schema_length=None
    )
    treatment = outcomes["z"].cast(pl.Int64).to_numpy()
    observed: FloatArray = np.where(
        treatment == 1, outcomes["y1"].to_numpy(), outcomes["y0"].to_numpy()
    ).astype(np.float64)
    true_effect: FloatArray = (outcomes["mu1"] - outcomes["mu0"]).to_numpy().astype(np.float64)

    return UpliftDataset(
        name="acic2016",
        features=features,
        treatment=treatment,
        outcome=observed,
        categorical=CATEGORICAL,
        true_effect=true_effect,
        replicate=replicate,
    )


def load_acic_replicates(count: int = N_REPLICATES) -> Iterator[UpliftDataset]:
    """Yield the first ``count`` replicates in order.

    Args:
        count: How many replicates to yield, 1 to 10.

    Yields:
        One dataset per replicate.
    """
    for replicate in range(min(count, N_REPLICATES)):
        yield load_acic(replicate)


@cache
def _covariates() -> pl.DataFrame:
    """The shared covariate matrix, encoded once and reused across replicates.

    The same 4,802 people appear in every replicate, so reading and encoding the 58 columns
    once saves ten times the work when the benchmark averages over replicates.
    """
    frame = pl.read_csv(fetch("acic-x", quiet=True), infer_schema_length=None)
    if frame.shape != (N_UNITS, N_COVARIATES):
        msg = f"ACIC: expected {(N_UNITS, N_COVARIATES)}, got {frame.shape}"
        raise ValueError(msg)
    return frame.select(
        [
            _encode(column) if column in CATEGORICAL else pl.col(column).cast(pl.Float64)
            for column in frame.columns
        ]
    )


def _encode(column: str) -> pl.Expr:
    """Map a lettered category to its position in the sorted list of levels."""
    return (
        pl.col(column)
        .rank("dense")
        .cast(pl.Float64)
        .sub(1.0)  # rank is 1-based; LightGBM category codes start at 0
        .alias(column)
    )
