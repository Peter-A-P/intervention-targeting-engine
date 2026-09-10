"""What random targeting is worth, measured rather than assumed to be zero.

A Qini coefficient is defined against a straight line, so the random baseline is zero by
construction and there is nothing to compute. Every other metric in the table is not: the
uplift inside a random top 20 percent, or the outcome a random policy buys, is a real
quantity with real sampling noise, and on a 64,000-row split it is not reliably zero.

So the random baseline is drawn rather than assumed: 200 random rankings (PLAN.md section
4), the metric computed on each, and the spread across those rankings reported as the
interval. A model whose interval overlaps this one has not been shown to beat a coin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from itx.metrics.bootstrap import DEFAULT_LEVEL, Estimate, bootstrap_over

if TYPE_CHECKING:
    from collections.abc import Callable

    from itx.types import FloatArray

DEFAULT_RANDOM_RANKINGS = 200


def random_ranking_reference(
    statistic: Callable[[FloatArray], float],
    n_units: int,
    *,
    n_rankings: int = DEFAULT_RANDOM_RANKINGS,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> Estimate:
    """Average a ranking metric over many random rankings.

    Args:
        statistic: Takes a score vector and returns the metric for that ranking.
        n_units: Number of rows being ranked.
        n_rankings: How many random rankings to draw.
        level: Nominal coverage of the reported interval.
        seed: Seed for the draws.

    Returns:
        The mean of the metric across random rankings, with the interval across them. The
        interval is the spread of what random targeting could have given you, not a
        bootstrap of one draw.
    """
    rng = np.random.default_rng(seed)
    values = np.array(
        [statistic(rng.random(n_units)) for _ in range(n_rankings)], dtype=np.float64
    )
    return bootstrap_over(values, level=level)
