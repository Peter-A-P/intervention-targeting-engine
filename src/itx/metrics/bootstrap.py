"""Bootstrap intervals. Nothing in this package reports a number without one.

The resampling unit is the row of the test split, drawn with replacement, and the interval
is the percentile interval of the resampled statistic (PLAN.md section 4): 1,000 resamples
on the small datasets, 200 on the Criteo subsample, where each resample costs real time.

Two things the percentile interval does not do, stated here because they are the usual
misreadings. It does not correct for bias in the statistic; a Qini coefficient computed on
the same split the model chose its cutoffs on stays optimistic no matter how many
resamples are drawn, which is why the split is held out instead. And it says nothing about
model uncertainty: the model is fixed, only the evaluation sample moves. Refitting inside
each resample would answer a different and more expensive question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from itx.types import FloatArray, IntArray

DEFAULT_RESAMPLES = 1_000
DEFAULT_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class Estimate:
    """A statistic with the interval it is never reported without.

    Attributes:
        value: The statistic on the full sample.
        low: Lower percentile bound.
        high: Upper percentile bound.
        level: Nominal coverage, for example 0.95.
        n_resamples: How many bootstrap resamples produced the bounds.
    """

    value: float
    low: float
    high: float
    level: float = DEFAULT_LEVEL
    n_resamples: int = DEFAULT_RESAMPLES

    def __str__(self) -> str:
        """Compact rendering: ``0.0116 (0.0089, 0.0143)``."""
        return self.format()

    def format(self, digits: int = 4) -> str:
        """Render for a table.

        A statistic that is undefined on the full sample renders as a dash even when some
        resamples produced a number. Printing ``nan`` next to a finite-looking interval
        invites a reader to believe the interval means something.

        Args:
            digits: Decimal places.

        Returns:
            The value with its interval in brackets, or a dash if it is undefined.
        """
        if not math.isfinite(self.value):
            return "-"
        return f"{self.value:.{digits}f} ({self.low:.{digits}f}, {self.high:.{digits}f})"

    @property
    def width(self) -> float:
        """Width of the interval."""
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval is on one side of zero."""
        return self.low > 0.0 or self.high < 0.0


def bootstrap_ci(
    statistic: Callable[[IntArray], float],
    n_units: int,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap interval for a statistic of a sample.

    Args:
        statistic: Takes an array of row positions and returns the statistic on those
            rows. Row positions rather than arrays, so a caller can resample a whole
            dataset, several arrays, or a fitted model's predictions consistently.
        n_units: Number of rows in the sample.
        n_resamples: Bootstrap resamples to draw.
        level: Nominal coverage.
        seed: Seed for the resampling.

    Returns:
        The statistic on the full sample with its interval. Resamples that return NaN, for
        instance a budget prefix that lost one of the arms, are dropped from the
        percentiles and reduce ``n_resamples`` accordingly.

    Raises:
        ValueError: If the level is not in ``(0, 1)`` or every resample was NaN.
    """
    if not 0.0 < level < 1.0:
        msg = f"level must be in (0, 1), got {level}"
        raise ValueError(msg)

    full = np.arange(n_units)
    point = statistic(full)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        draws[index] = statistic(rng.integers(0, n_units, size=n_units))

    usable: FloatArray = draws[np.isfinite(draws)]
    if usable.size == 0:
        msg = (
            "every bootstrap resample was undefined; the sample is too small for this statistic"
        )
        raise ValueError(msg)

    tail = (1.0 - level) / 2.0
    low, high = np.percentile(usable, [100.0 * tail, 100.0 * (1.0 - tail)])
    return Estimate(
        value=point,
        low=float(low),
        high=float(high),
        level=level,
        n_resamples=int(usable.size),
    )


def bootstrap_many(
    statistics: Mapping[str, Callable[[IntArray], float]],
    n_units: int,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> dict[str, Estimate]:
    """Bootstrap several statistics in one pass over the resamples.

    Each resample is drawn once and every statistic is computed on it, which matters
    because drawing and sorting a Criteo-sized test split a thousand times per metric is
    the slowest thing in the benchmark. It also means the intervals for two metrics come
    from the same resamples, so their widths are directly comparable.

    Args:
        statistics: Named functions from row positions to a value.
        n_units: Number of rows in the sample.
        n_resamples: Bootstrap resamples to draw.
        level: Nominal coverage.
        seed: Seed for the resampling.

    Returns:
        One estimate per named statistic.

    Raises:
        ValueError: If the level is not in ``(0, 1)``.
    """
    if not 0.0 < level < 1.0:
        msg = f"level must be in (0, 1), got {level}"
        raise ValueError(msg)

    names = list(statistics)
    full = np.arange(n_units)
    points = {name: statistics[name](full) for name in names}

    rng = np.random.default_rng(seed)
    draws = np.empty((n_resamples, len(names)), dtype=np.float64)
    for index in range(n_resamples):
        resample = rng.integers(0, n_units, size=n_units)
        for column, name in enumerate(names):
            draws[index, column] = statistics[name](resample)

    return _percentiles(points, draws, names, level=level)


def bootstrap_vector(
    statistics: Callable[[IntArray], Mapping[str, float]],
    n_units: int,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> dict[str, Estimate]:
    """Bootstrap a function that returns several named metrics at once.

    The version to reach for when the metrics share expensive work, which the ranking
    metrics do: they all walk the same sorted ranking, and sorting it once per resample
    instead of once per metric is most of the benchmark's running time.

    Args:
        statistics: Takes row positions and returns a mapping of metric name to value. The
            same keys must come back from every call.
        n_units: Number of rows in the sample.
        n_resamples: Bootstrap resamples to draw.
        level: Nominal coverage.
        seed: Seed for the resampling.

    Returns:
        One estimate per metric name.

    Raises:
        ValueError: If the level is not in ``(0, 1)``.
    """
    if not 0.0 < level < 1.0:
        msg = f"level must be in (0, 1), got {level}"
        raise ValueError(msg)

    points = dict(statistics(np.arange(n_units)))
    names = list(points)

    rng = np.random.default_rng(seed)
    draws = np.empty((n_resamples, len(names)), dtype=np.float64)
    for index in range(n_resamples):
        values = statistics(rng.integers(0, n_units, size=n_units))
        draws[index] = [values[name] for name in names]

    return _percentiles(points, draws, names, level=level)


def bootstrap_over(
    values: FloatArray,
    *,
    level: float = DEFAULT_LEVEL,
) -> Estimate:
    """Interval across an existing set of values, for replicates and seeds.

    IHDP ships 100 replicates and the protocol runs five seeds per dataset. The spread
    across those runs is a different quantity from the spread across resamples of one test
    split, and it is reported as a percentile interval across the runs themselves rather
    than by resampling them.

    Args:
        values: One value per replicate or seed.
        level: Nominal coverage.

    Returns:
        The mean across runs, with the percentile interval across runs.

    Raises:
        ValueError: If no finite values were given.
    """
    finite: FloatArray = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        msg = "no finite values to summarise"
        raise ValueError(msg)
    tail = (1.0 - level) / 2.0
    low, high = np.percentile(finite, [100.0 * tail, 100.0 * (1.0 - tail)])
    return Estimate(
        value=float(finite.mean()),
        low=float(low),
        high=float(high),
        level=level,
        n_resamples=int(finite.size),
    )


def _percentiles(
    points: Mapping[str, float],
    draws: FloatArray,
    names: Sequence[str],
    *,
    level: float,
) -> dict[str, Estimate]:
    """Turn a matrix of resampled values into one estimate per column.

    Resamples that came back undefined, for instance a budget prefix that lost one of the
    arms, are dropped from the percentiles and reduce the reported resample count rather
    than being treated as zeros.
    """
    tail = (1.0 - level) / 2.0
    results: dict[str, Estimate] = {}
    for column, name in enumerate(names):
        usable = draws[:, column][np.isfinite(draws[:, column])]
        if usable.size == 0:
            results[name] = Estimate(
                value=points[name],
                low=float("nan"),
                high=float("nan"),
                level=level,
                n_resamples=0,
            )
            continue
        low, high = np.percentile(usable, [100.0 * tail, 100.0 * (1.0 - tail)])
        results[name] = Estimate(
            value=points[name],
            low=float(low),
            high=float(high),
            level=level,
            n_resamples=int(usable.size),
        )
    return results
