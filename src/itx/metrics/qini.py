"""The reported ranking metrics: Qini coefficient, normalised AUUC, uplift at k.

Definitions, because these three names are used for at least five different quantities in
the literature and a table is worthless if the reader has to guess which one:

``qini_coefficient``
    Area between the Qini curve and the random line, divided by the number of units. Zero
    means the ranking is worth nothing over picking at random; negative means it is worse
    than random, which happens more often than the literature suggests. The units are
    incremental outcome per unit of population, so on a binary outcome a coefficient of
    0.01 means the ranking buys one extra event per hundred people over random targeting
    among the *treated* units of the prefix, divided by the whole population, so it scales
    with the treated share and is not comparable across datasets with different shares
    (Criteo 85%, Hillstrom 50%, IHDP 19%); compare it within a table, not between them,
    averaged across every budget.

``auuc_normalised``
    Area under the cumulative-gain curve, divided by the area under the oracle ordering's
    curve on the same sample. At most 1. This is the readable one, but it is a ratio to an
    oracle and cannot be compared across datasets with different levels of noise.

``uplift_at_k``
    The plain difference in outcome rate between the treated and the control units inside
    the top ``k`` share of the ranking. This is the number a budget holder actually asks
    for, and it is an estimate on the sample, not a fitted quantity.

None of these is reported bare. Every one goes through :mod:`itx.metrics.bootstrap` before
it reaches a table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from itx.metrics.curves import (
    DEFAULT_TIE_SEED,
    area_under,
    optimal_scores,
    prefix_sums,
    qini_curve,
    qini_gain,
    rank_order,
    uplift_curve,
    uplift_gain,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itx.types import FloatArray, IntArray


def qini_coefficient(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    seed: int = DEFAULT_TIE_SEED,
) -> float:
    """Area between the Qini curve and the random line, per unit of population.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Targeting scores, higher meaning treat sooner.
        seed: Seed for tie-breaking.

    Returns:
        The coefficient. Zero for a ranking no better than random.
    """
    curve = qini_curve(outcome, treatment, scores, seed=seed).per_unit()
    return curve.area() - curve.random_area()


def auuc(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    seed: int = DEFAULT_TIE_SEED,
) -> float:
    """Area under the cumulative-gain curve, per unit of population.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Targeting scores, higher meaning treat sooner.
        seed: Seed for tie-breaking.

    Returns:
        The unnormalised area. Use :func:`auuc_normalised` in tables.
    """
    return uplift_curve(outcome, treatment, scores, seed=seed).per_unit().area()


def auuc_normalised(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    seed: int = DEFAULT_TIE_SEED,
) -> float:
    """AUUC as a share of the oracle ordering's AUUC on the same sample.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Targeting scores, higher meaning treat sooner.
        seed: Seed for tie-breaking.

    Returns:
        The ratio. Not bounded by 1: the outcome-ordered reference earns nothing while its
        prefix holds one arm, so a ranking that interleaves control non-responders can beat
        it, and random targeting lands anywhere from 0.04 to 0.83 under it depending on the
        dataset. Read it against the random row. Returns 0 when the oracle area is 0, which
        happens only when no ordering of this sample buys anything at all.
    """
    ceiling = auuc(outcome, treatment, optimal_scores(outcome, treatment), seed=seed)
    if ceiling == 0.0:
        return 0.0
    return auuc(outcome, treatment, scores, seed=seed) / ceiling


def uplift_at_k(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    k: float,
    *,
    seed: int = DEFAULT_TIE_SEED,
) -> float:
    """Difference in outcome rate between arms inside the top ``k`` of the ranking.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Targeting scores, higher meaning treat sooner.
        k: Share of the population targeted, in ``(0, 1]``.
        seed: Seed for tie-breaking.

    Returns:
        Treated mean minus control mean within the targeted prefix. NaN if the prefix
        contains no units from one of the arms, which is a real answer: at that budget the
        sample cannot say.

    Raises:
        ValueError: If ``k`` is outside ``(0, 1]``.
    """
    if not 0.0 < k <= 1.0:
        msg = f"k must be in (0, 1], got {k}"
        raise ValueError(msg)
    order = rank_order(scores, seed=seed)
    cut = max(1, int(np.ceil(k * outcome.size)))
    prefix = order[:cut]
    treated = treatment[prefix] == 1
    if not treated.any() or treated.all():
        return float("nan")
    return float(outcome[prefix][treated].mean() - outcome[prefix][~treated].mean())


def ate(outcome: FloatArray, treatment: IntArray) -> float:
    """Difference in means between the arms, the whole-population average effect.

    Only an unbiased estimate of the average treatment effect when assignment was random
    or the propensity is constant. On an observational dataset it is the naive comparison
    the rest of this package exists to improve on.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.

    Returns:
        Treated mean minus control mean.

    Raises:
        ValueError: If either arm is empty.
    """
    treated = treatment == 1
    if not treated.any() or treated.all():
        msg = "cannot compute an ATE with an empty arm"
        raise ValueError(msg)
    return float(outcome[treated].mean() - outcome[~treated].mean())


def ranking_metrics(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    budgets: Sequence[float] = (),
    seed: int = DEFAULT_TIE_SEED,
) -> dict[str, float]:
    """Every ranking metric for one ranking, from a single sorted pass.

    Identical results to calling :func:`qini_coefficient`, :func:`auuc_normalised` and
    :func:`uplift_at_k` in turn, at roughly a third of the cost, because the ranking is
    sorted once and the cumulative sums are shared. That difference is what makes a
    thousand-resample bootstrap on a large test split finish in minutes rather than an
    hour, so this is the entry point the benchmark uses.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Targeting scores, higher meaning treat sooner.
        budgets: Shares of the population to report uplift at.
        seed: Seed for tie-breaking.

    Returns:
        ``qini``, ``auuc``, and one ``uplift@k`` entry per budget.
    """
    n_units = outcome.size
    order = rank_order(scores, seed=seed)
    sums = prefix_sums(outcome, treatment, order)

    gain = qini_gain(sums)
    qini_area = area_under(gain, n_units) / n_units
    endpoint = float(gain[-1]) / n_units
    metrics = {"qini": qini_area - endpoint / 2.0}

    model_auuc = area_under(uplift_gain(sums), n_units) / n_units
    oracle_order = rank_order(optimal_scores(outcome, treatment), seed=seed)
    oracle_auuc = (
        area_under(uplift_gain(prefix_sums(outcome, treatment, oracle_order)), n_units)
        / n_units
    )
    metrics["auuc"] = 0.0 if oracle_auuc == 0.0 else model_auuc / oracle_auuc

    for budget in budgets:
        cut = max(1, int(np.ceil(budget * n_units)))
        n_treated = sums.n_treated[cut - 1]
        n_control = sums.n_control[cut - 1]
        if n_treated == 0 or n_control == 0:
            metrics[f"uplift@{budget:.0%}"] = float("nan")
        else:
            metrics[f"uplift@{budget:.0%}"] = float(
                sums.treated_sum[cut - 1] / n_treated - sums.control_sum[cut - 1] / n_control
            )
    return metrics
