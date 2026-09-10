"""Qini and cumulative-gain curves, and the two reference curves they are read against.

Both curves answer the same shape of question: walk down a ranking, treating people as you
go, and plot how much extra outcome you have bought by the time you have reached a given
fraction of the population. They differ in how they compare the treated to the untreated
inside the prefix.

**Qini curve** (Radcliffe): ``y_t(k) - y_c(k) * n_t(k) / n_c(k)``. The control sum is
rescaled to the size of the treated group in the prefix, so the y axis is in units of
incremental outcomes actually delivered to the treated.

**Cumulative gain curve** (the one AUUC integrates): ``(y_t(k)/n_t(k) - y_c(k)/n_c(k)) *
k``. The two arms are compared as rates and then scaled by the prefix size, which is
steadier at the top of the ranking where one arm can be thin.

Two reference curves matter. The **random line** is the straight line to the same endpoint:
what you get with no ranking at all, because a random prefix of the population contains a
random share of the total effect. The **optimal curve** is the best ordering achievable on
this sample, obtained by putting the treated responders first and the control responders
last; it is an oracle, since it reads the outcomes it is meant to be predicting, and it
exists only to give the normalised AUUC a ceiling of 1.

Ties are broken by a seeded shuffle before a stable sort. Tree ensembles produce ties
constantly, and leaving the order to whatever the sort does with equal keys makes a metric
depend on row order, which is a bug that hides for a long time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from itx.types import FloatArray, IntArray

DEFAULT_TIE_SEED = 0


@dataclass(frozen=True, slots=True)
class Curve:
    """A targeting curve: cumulative gain against the fraction of the population targeted.

    Attributes:
        fraction: Share of the population treated, from 0 to 1, length ``n + 1``.
        gain: Cumulative incremental outcome at that share, length ``n + 1``, starting at 0.
        n_units: Number of units the curve was computed on.
    """

    fraction: FloatArray
    gain: FloatArray
    n_units: int

    @property
    def endpoint(self) -> float:
        """Total incremental outcome when everyone is targeted."""
        return float(self.gain[-1])

    def area(self) -> float:
        """Area under the curve, integrating gain over the fraction axis."""
        return float(np.trapezoid(self.gain, self.fraction))

    def random_area(self) -> float:
        """Area under the straight line to the same endpoint."""
        return self.endpoint / 2.0

    def per_unit(self) -> Curve:
        """The same curve with the gain axis divided by the number of units.

        Areas from the per-unit curve are comparable across datasets of different sizes,
        which is what the reported coefficients use.
        """
        return Curve(
            fraction=self.fraction, gain=self.gain / self.n_units, n_units=self.n_units
        )


def rank_order(scores: FloatArray, *, seed: int = DEFAULT_TIE_SEED) -> IntArray:
    """Row positions sorted by score, highest first, with ties broken reproducibly.

    Args:
        scores: Targeting scores, higher meaning treat sooner.
        seed: Seed for the tie-breaking shuffle.

    Returns:
        Indices into ``scores``, best first.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(scores.size)
    order: IntArray = shuffled[np.argsort(-scores[shuffled], kind="stable")]
    return order


@dataclass(frozen=True, slots=True)
class PrefixSums:
    """Cumulative arm sums and counts down a ranking, the raw material of both curves.

    Computing these once and deriving both curves from them is what makes the bootstrap
    affordable: the sort, not the arithmetic, is the expensive part, and a thousand
    resamples times five metrics is five thousand sorts if each metric starts from scratch.

    Attributes:
        treated_sum: Cumulative outcome among treated units in the prefix.
        control_sum: Cumulative outcome among control units in the prefix.
        n_treated: Cumulative count of treated units in the prefix.
        n_control: Cumulative count of control units in the prefix.
    """

    treated_sum: FloatArray
    control_sum: FloatArray
    n_treated: FloatArray
    n_control: FloatArray

    @property
    def n_units(self) -> int:
        """Number of units the prefix sums were built from."""
        return int(self.treated_sum.size)


def prefix_sums(outcome: FloatArray, treatment: IntArray, order: IntArray) -> PrefixSums:
    """Cumulative sums and counts, walking down a given ranking.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        order: Row positions, best first.

    Returns:
        The cumulative arrays.
    """
    ordered_outcome = outcome[order]
    treated = treatment[order].astype(np.float64)
    control = 1.0 - treated
    return PrefixSums(
        treated_sum=np.cumsum(ordered_outcome * treated),
        control_sum=np.cumsum(ordered_outcome * control),
        n_treated=np.cumsum(treated),
        n_control=np.cumsum(control),
    )


def qini_gain(sums: PrefixSums) -> FloatArray:
    """Qini gain at every prefix: treated outcome minus rescaled control outcome.

    Args:
        sums: Prefix sums for the ranking.

    Returns:
        The gain at each prefix length, without the leading zero.
    """
    ratio = np.divide(
        sums.n_treated,
        sums.n_control,
        out=np.zeros_like(sums.n_treated),
        where=sums.n_control > 0,
    )
    gain: FloatArray = sums.treated_sum - sums.control_sum * ratio
    return gain


def uplift_gain(sums: PrefixSums) -> FloatArray:
    """Cumulative gain at every prefix: difference of arm rates, scaled by prefix size.

    Args:
        sums: Prefix sums for the ranking.

    Returns:
        The gain at each prefix length, without the leading zero. Prefixes holding only
        one arm contribute 0, since no difference of rates can be formed there.
    """
    both = (sums.n_treated > 0) & (sums.n_control > 0)
    treated_rate = np.divide(
        sums.treated_sum,
        sums.n_treated,
        out=np.zeros_like(sums.treated_sum),
        where=sums.n_treated > 0,
    )
    control_rate = np.divide(
        sums.control_sum,
        sums.n_control,
        out=np.zeros_like(sums.control_sum),
        where=sums.n_control > 0,
    )
    gain: FloatArray = np.where(
        both, (treated_rate - control_rate) * (sums.n_treated + sums.n_control), 0.0
    )
    return gain


def qini_curve(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    seed: int = DEFAULT_TIE_SEED,
) -> Curve:
    """The Qini curve for one ranking.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Targeting scores, higher meaning treat sooner.
        seed: Seed for tie-breaking.

    Returns:
        The curve, with gain in units of incremental outcome.
    """
    _check_lengths(outcome, treatment, scores)
    sums = prefix_sums(outcome, treatment, rank_order(scores, seed=seed))
    return curve_from_gain(qini_gain(sums), outcome.size)


def uplift_curve(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    seed: int = DEFAULT_TIE_SEED,
) -> Curve:
    """The cumulative-gain curve for one ranking, the one AUUC integrates.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Targeting scores, higher meaning treat sooner.
        seed: Seed for tie-breaking.

    Returns:
        The curve. Prefixes containing only one arm contribute a gain of 0, since no
        difference of rates can be formed there.
    """
    _check_lengths(outcome, treatment, scores)
    sums = prefix_sums(outcome, treatment, rank_order(scores, seed=seed))
    return curve_from_gain(uplift_gain(sums), outcome.size)


def optimal_scores(outcome: FloatArray, treatment: IntArray) -> FloatArray:
    """Scores for the best ordering achievable on this sample.

    Treated units with a high outcome push the curve up, control units with a high outcome
    push it down, so the ordering that maximises the curve puts the first group first and
    the second group last. This reads the outcomes, so it is an oracle and is only ever
    used as the denominator of a normalised score.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.

    Returns:
        Oracle scores.
    """
    sign = 2.0 * treatment.astype(np.float64) - 1.0
    oracle: FloatArray = outcome * sign
    return oracle


def curve_from_gain(gain: FloatArray, n_units: int) -> Curve:
    """Prepend the origin and build the fraction axis.

    Args:
        gain: Gain at each prefix length, without the leading zero.
        n_units: Number of units.

    Returns:
        The finished curve.
    """
    full_gain = np.concatenate([[0.0], gain])
    fraction = np.arange(n_units + 1, dtype=np.float64) / n_units
    return Curve(fraction=fraction, gain=full_gain, n_units=n_units)


def area_under(gain: FloatArray, n_units: int) -> float:
    """Area under a gain array on the fraction axis, without building a Curve.

    Args:
        gain: Gain at each prefix length, without the leading zero.
        n_units: Number of units.

    Returns:
        The integral of the gain over the fraction of the population targeted.
    """
    full_gain = np.concatenate([[0.0], gain])
    return float(np.trapezoid(full_gain, dx=1.0 / n_units))


def _check_lengths(outcome: FloatArray, treatment: IntArray, scores: FloatArray) -> None:
    """Reject mismatched inputs, the commonest way a metric silently lies."""
    sizes = {outcome.size, treatment.size, scores.size}
    if len(sizes) != 1:
        msg = (
            "outcome, treatment and scores must be the same length; got "
            f"{outcome.size}, {treatment.size}, {scores.size}"
        )
        raise ValueError(msg)
    if outcome.size == 0:
        msg = "cannot compute a curve on an empty sample"
        raise ValueError(msg)
