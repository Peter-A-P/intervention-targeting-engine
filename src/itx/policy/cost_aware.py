"""When interventions cost different amounts, the ranking that spends the budget changes.

:mod:`itx.policy.rank_and_cut` treats the top b% of a list. That is the right rule when
every intervention costs the same, and it is wrong as soon as they do not. A fraud review
costs an analyst an hour, a text message costs a fraction of a cent, and a clinic visit
costs more than either: a budget is money, not a headcount, and the question becomes which
subset of people to treat so that the predicted effect is largest and the bill fits.

That is the 0/1 knapsack problem, and this module solves it the way it should be solved in
this setting rather than the way a textbook solves it.

## Rank by effect per dollar, not by effect

The rule is to sort by ``uplift / cost`` and take from the top until the money runs out.
Two things about it are worth stating because both are easy to get wrong.

It is a different list from the uplift ranking. A unit with twice the effect and three
times the cost sits lower here than one with half the effect and a fifth of the cost, and
the two orderings agree only when every cost is the same. When they are, this reduces
exactly to :func:`itx.policy.rank_and_cut.rank_and_cut` at the matching share, which the
tests assert rather than assume.

It never buys predicted harm. Under a headcount budget, refusing to treat a unit with a
negative predicted effect leaves part of the budget unspent and makes two policies
non-comparable, which is why ``rank_and_cut`` treats the requested share by default. Under
a money budget there is no such tension: unspent money is saved, so a maximiser simply never
takes a negative-effect unit, and the default is the right one with nothing to trade off.

## The guarantee, and why this reports a measured gap instead of quoting one

The greedy ratio rule is not optimal for the 0/1 problem, and the 0/1 problem is NP-hard,
so there is no cheap exact answer for a million rows. The usual move is to quote the
textbook bound: greedy-by-ratio, taken together with the best single affordable item, is
within a factor of two of optimal. That is true and nearly useless, because a factor of two
is enormous and the actual gap on a real targeting instance is nothing like it.

So this reports the gap instead of quoting the bound. Relaxing the problem to allow
fractions of a unit makes it solvable exactly, by the same ratio ordering, and its value is
an upper bound on what any integer allocation could achieve. Every :class:`Allocation`
carries that bound and the distance to it, so a caller can see on their own instance that
the answer is, for example, within 0.002% of anything achievable. The gap is large only when
a single unit costs an appreciable share of the whole budget, which is the case this
docstring exists to warn about: with ten units and a budget of three, greedy can be badly
beaten, and with a hundred thousand units and a budget of thousands it cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itx.metrics.curves import DEFAULT_TIE_SEED, rank_order

if TYPE_CHECKING:
    from itx.types import BoolArray, FloatArray


@dataclass(frozen=True, slots=True)
class Allocation:
    """Who a money budget buys an intervention for, and what that is expected to be worth.

    Attributes:
        treat: Boolean mask, True for the units to treat.
        spend: What treating them costs.
        budget: What was available.
        predicted_gain: Sum of the predicted effects of the treated units. A prediction,
            not a measurement: what the chosen set is actually worth is a question for
            :mod:`itx.policy.policy_value` on held-out data.
        upper_bound: The most any allocation could possibly have bought within this budget,
            from the fractional relaxation. Never smaller than ``predicted_gain``.
    """

    treat: BoolArray
    spend: float
    budget: float
    predicted_gain: float
    upper_bound: float

    @property
    def n_treated(self) -> int:
        """How many units are treated."""
        return int(self.treat.sum())

    @property
    def unspent(self) -> float:
        """Budget left over, which happens when nothing affordable is worth buying."""
        return self.budget - self.spend

    @property
    def optimality_gap(self) -> float:
        """Share of the achievable gain this allocation might be leaving on the table.

        Measured against the fractional relaxation, so it is an over-estimate of the true
        gap against the best integer allocation: zero here means provably optimal, and a
        small number means optimal for any purpose a budget holder has.
        """
        if self.upper_bound <= 0.0:
            return 0.0
        return (self.upper_bound - self.predicted_gain) / self.upper_bound

    def describe(self) -> str:
        """One line for logs and the demo."""
        return (
            f"{self.n_treated:,} of {self.treat.size:,} units, "
            f"spend {self.spend:,.2f} of {self.budget:,.2f}, "
            f"predicted gain {self.predicted_gain:,.4f} "
            f"(within {self.optimality_gap:.4%} of the best possible)"
        )


def cost_aware_policy(
    scores: FloatArray,
    costs: FloatArray,
    budget: float,
    *,
    seed: int = DEFAULT_TIE_SEED,
) -> Allocation:
    """Choose whom to treat so the predicted effect is largest and the bill fits.

    Sorts by predicted effect per unit of cost and takes from the top, skipping units that
    no longer fit rather than stopping at the first one that does not: skipping can only
    increase the total, and the budget is usually left with room for something small.

    Args:
        scores: Predicted uplift per unit, on the outcome's scale.
        costs: Cost of treating each unit, strictly positive.
        budget: Money available, in the same units as ``costs``.
        seed: Seed for tie-breaking, shared with the metrics and with
            :func:`itx.policy.rank_and_cut.rank_and_cut` so that equal-ratio units are
            ordered the same way everywhere.

    Returns:
        The allocation, carrying what it spends and how far it could be from optimal.

    Raises:
        ValueError: If the inputs are different lengths, a cost is not strictly positive,
            or the budget is negative.
    """
    _check(scores, costs, budget)

    # Units predicted to do harm are never bought: under a money budget, declining to spend
    # costs nothing, so there is no version of this where treating them is right.
    worthwhile = scores > 0.0
    ratio = np.where(worthwhile, scores / costs, -np.inf)
    order = rank_order(ratio, seed=seed)

    treat: BoolArray = np.zeros(scores.size, dtype=bool)
    spend = 0.0
    for unit in order:
        if not worthwhile[unit]:
            break
        if costs[unit] <= budget - spend:
            treat[unit] = True
            spend += float(costs[unit])

    return Allocation(
        treat=treat,
        spend=spend,
        budget=budget,
        predicted_gain=float(scores[treat].sum()),
        upper_bound=fractional_bound(scores, costs, budget, seed=seed),
    )


def fractional_bound(
    scores: FloatArray,
    costs: FloatArray,
    budget: float,
    *,
    seed: int = DEFAULT_TIE_SEED,
) -> float:
    """The most a budget could buy if units could be treated in fractions.

    The relaxation of the knapsack, which the same ratio ordering solves exactly: fill from
    the top and take whatever fraction of the first unit that does not fit the remaining
    money covers. No integer allocation can beat it, so it is the yardstick
    :attr:`Allocation.optimality_gap` is measured against.

    Args:
        scores: Predicted uplift per unit.
        costs: Cost of treating each unit, strictly positive.
        budget: Money available.
        seed: Seed for tie-breaking.

    Returns:
        The relaxation's value.

    Raises:
        ValueError: If the inputs are different lengths, a cost is not strictly positive,
            or the budget is negative.
    """
    _check(scores, costs, budget)
    worthwhile = scores > 0.0
    ratio = np.where(worthwhile, scores / costs, -np.inf)
    order = rank_order(ratio, seed=seed)

    remaining = budget
    total = 0.0
    for unit in order:
        if not worthwhile[unit] or remaining <= 0.0:
            break
        cost = float(costs[unit])
        if cost <= remaining:
            total += float(scores[unit])
            remaining -= cost
        else:
            total += float(scores[unit]) * remaining / cost
            remaining = 0.0
    return total


def budget_for_share(costs: FloatArray, share: float) -> float:
    """The money that would treat ``share`` of the population at the average cost.

    A convenience for comparing a cost-aware policy with a headcount one on the same axis:
    the two are only comparable if they are allowed to spend the same amount, and a
    headcount budget of 20% is a money budget of twenty percent of the total bill.

    Args:
        costs: Cost of treating each unit.
        share: Share of the population, in ``(0, 1]``.

    Returns:
        The equivalent money budget.

    Raises:
        ValueError: If the share is outside ``(0, 1]``.
    """
    if not 0.0 < share <= 1.0:
        msg = f"share must be in (0, 1], got {share}"
        raise ValueError(msg)
    return float(costs.sum()) * share


def _check(scores: FloatArray, costs: FloatArray, budget: float) -> None:
    """Reject inputs a knapsack cannot be stated on."""
    if scores.size != costs.size:
        msg = f"scores and costs must be the same length; got {scores.size}, {costs.size}"
        raise ValueError(msg)
    if scores.size == 0:
        msg = "cannot allocate a budget over an empty population"
        raise ValueError(msg)
    if not (costs > 0.0).all():
        msg = (
            "every cost must be strictly positive; a free or negatively priced "
            f"intervention has no effect-per-dollar to rank on. Range is {costs.min():.4g} "
            f"to {costs.max():.4g}"
        )
        raise ValueError(msg)
    if budget < 0.0:
        msg = f"budget must not be negative, got {budget}"
        raise ValueError(msg)
