"""What a fixed number of analyst hours buys, under four ways of choosing the queue.

The rest of this package answers "who should we treat" with a share of the population,
because that is what the five public datasets support: every unit costs the same to treat, so
a budget is a headcount and rank-and-cut spends it optimally. The fraud worked case is the
one where that stops being true. A review takes an analyst somewhere between nine and
eighteen minutes depending on how complete the record is, so the budget is hours rather than
headcount, and the question changes from "who has the largest effect" to "who has the largest
effect per minute".

That is what :mod:`itx.policy.cost_aware` was built for in week 5 and what it had no data to
run on until now.

## The four queues

``uplift-knapsack`` takes units in order of predicted effect per minute until the hours run
out. ``uplift-rank-and-cut`` takes them in order of predicted effect, which is what a team
would do if it had an uplift model and had not thought about cost. ``risk`` takes them in
order of predicted loss, which is what a fraud team does now and is the baseline this whole
project exists to test. ``random`` is the floor.

## What they are scored on

Two numbers, because the case is semi-synthetic and can afford both. The doubly robust
estimate is what this package would report on real data where the truth is missing. The true
value is what the simulation actually wrote, summed over the units each queue picked, and it
is available here only because the effect was invented. Reporting them side by side is the
point: a reader can see how close the estimate gets without being told.

Both are totals in dollars rather than per-head rates, because a fraud manager's question is
how much a shift of analyst time is worth, and a per-transaction average over half a million
transactions is not a number anybody can hold.

The doubly robust total carries a bootstrap interval over the test rows, with the queue held
fixed, because a point beside a true value invites the reader to read their difference as an
error, and on 4,000 reviewed transactions that difference is inside the noise (PLAN.md change
56). The "never buy predicted harm" stop applies only to queues whose score is a predicted
effect; a risk score or a random draw has no zero that means harm, so those queues spend the
whole budget the way a real queue would.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itx.metrics.bootstrap import DEFAULT_LEVEL, DEFAULT_RESAMPLES, bootstrap_ci
from itx.policy.cost_aware import cost_aware_policy
from itx.policy.policy_value import dr_gain

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from itx.policy.policy_value import Nuisances
    from itx.types import BoolArray, FloatArray, IntArray

#: Minutes in an analyst hour, so the reported budget is in a unit a manager uses.
MINUTES_PER_HOUR = 60.0


@dataclass(frozen=True, slots=True)
class Queue:
    """One way of choosing a review queue, and what it turned out to be worth.

    Attributes:
        name: Which rule chose it.
        n_reviewed: How many transactions it sends to review.
        minutes: Analyst minutes it spends.
        dr_value: Doubly robust estimate of what reviewing them is worth, in dollars. This
            is the number this package would report on data whose truth is missing.
        true_value: What it is actually worth, from the effects the simulation wrote. Only
            available because the case is semi-synthetic.
        dr_low: Lower bound of the bootstrap interval on ``dr_value``.
        dr_high: Upper bound.
    """

    name: str
    n_reviewed: int
    minutes: float
    dr_value: float
    true_value: float
    dr_low: float = math.nan
    dr_high: float = math.nan

    @property
    def dollars_per_hour(self) -> float:
        """What an hour of analyst time bought, by the true effect."""
        if self.minutes <= 0.0:
            return float("nan")
        return self.true_value / (self.minutes / MINUTES_PER_HOUR)

    @property
    def error(self) -> float:
        """How far the doubly robust estimate is from the truth, in dollars."""
        return self.dr_value - self.true_value


def compare_queues(
    scores: Mapping[str, FloatArray],
    costs: FloatArray,
    budget_hours: float,
    *,
    outcome: FloatArray,
    treatment: IntArray,
    nuisances: Nuisances,
    truth: FloatArray,
    knapsack_for: Sequence[str] = (),
    harm_aware_for: Sequence[str] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> list[Queue]:
    """Score several queues against the same budget of analyst time.

    Args:
        scores: Ranking score per unit, by queue name. Higher means review sooner.
        costs: Minutes a review of each unit takes, strictly positive.
        budget_hours: Analyst hours available.
        outcome: Observed outcome per unit, in dollars of retained value.
        treatment: Whether each unit was in fact reviewed.
        nuisances: Propensity and outcome models for these rows.
        truth: The true per-unit effect the simulation wrote.
        knapsack_for: Which queues spend by effect per minute rather than by effect. The
            rest take units in score order until the budget runs out.
        harm_aware_for: Which of the rest stop at a score of zero because their score is a
            predicted effect and a non-positive one means review is expected to do harm.
            None, the default, means all of them, which is right when every score is an
            effect; a risk score or a random draw should be listed out of it.
        n_resamples: Bootstrap resamples behind the interval on the doubly robust total.
        level: Nominal coverage of that interval.
        seed: Tie-breaking seed, shared with the metrics, and the bootstrap seed.

    Returns:
        One :class:`Queue` per entry in ``scores``, in the order given.
    """
    budget = budget_hours * MINUTES_PER_HOUR
    n_units = outcome.size
    queues: list[Queue] = []
    for name, score in scores.items():
        stop_at_harm = harm_aware_for is None or name in harm_aware_for
        treat = (
            cost_aware_policy(score, costs, budget, seed=seed).treat
            if name in knapsack_for
            else _spend_in_score_order(
                score, costs, budget, seed=seed, stop_at_harm=stop_at_harm
            )
        )

        # dr_gain is a per-head rate over the whole population, so multiplying by the
        # population returns the total the queue is worth. The queue itself is fixed and
        # the rows are resampled, so the interval is about the estimate, not the policy.
        def dr_total(index: IntArray, treat: BoolArray = treat) -> float:
            return (
                float(
                    dr_gain(
                        outcome[index], treatment[index], treat[index], nuisances.take(index)
                    )
                )
                * n_units
            )

        estimate = bootstrap_ci(
            dr_total, n_units, n_resamples=n_resamples, level=level, seed=seed
        )
        queues.append(
            Queue(
                name=name,
                n_reviewed=int(treat.sum()),
                minutes=float(costs[treat].sum()),
                dr_value=estimate.value,
                true_value=float(truth[treat].sum()),
                dr_low=estimate.low,
                dr_high=estimate.high,
            )
        )
    return queues


def _spend_in_score_order(
    scores: FloatArray,
    costs: FloatArray,
    budget: float,
    *,
    seed: int,
    stop_at_harm: bool = True,
) -> BoolArray:
    """Take units in score order until the money runs out, skipping what no longer fits.

    The cost-blind comparison. It is rank-and-cut with a money budget rather than a
    headcount one, which is what a team with an uplift model and no cost model would do.
    Skipping rather than stopping matches :func:`itx.policy.cost_aware.cost_aware_policy`,
    so the only difference between the two queues is what they sort on, which is the whole
    point of putting them side by side.

    Args:
        scores: Ranking score per unit.
        costs: Cost of treating each unit.
        budget: Money available.
        seed: Tie-breaking seed.
        stop_at_harm: Stop at the first non-positive score. Right when the score is a
            predicted effect, wrong when it is a risk score or a random draw, whose zero
            means nothing about harm.

    Returns:
        A boolean mask of the units to treat.
    """
    from itx.metrics.curves import rank_order

    treat: BoolArray = np.zeros(scores.size, dtype=bool)
    spent = 0.0
    for position in rank_order(scores, seed=seed):
        # Never buy a unit the model expects to do harm, however much budget is left. On
        # this case that is most of the population.
        if stop_at_harm and scores[position] <= 0.0:
            break
        if spent + costs[position] <= budget:
            treat[position] = True
            spent += float(costs[position])
    return treat


def to_markdown(queues: Sequence[Queue], budget_hours: float) -> str:
    """Render the comparison as the table the write-up carries.

    Args:
        queues: What :func:`compare_queues` returned.
        budget_hours: The budget they were given, for the heading.

    Returns:
        A markdown table.
    """
    lines = [
        f"| Queue at {budget_hours:,.0f} analyst hours | Reviewed | Hours used "
        f"| True value | DR estimate (95% CI) | $/analyst hour |",
        "|---|---|---|---|---|---|",
    ]
    for queue in queues:
        interval = (
            f" ({_dollars(queue.dr_low)}, {_dollars(queue.dr_high)})"
            if math.isfinite(queue.dr_low) and math.isfinite(queue.dr_high)
            else ""
        )
        lines.append(
            f"| `{queue.name}` | {queue.n_reviewed:,} "
            f"| {queue.minutes / MINUTES_PER_HOUR:,.0f} "
            f"| {_dollars(queue.true_value)} | {_dollars(queue.dr_value)}{interval} "
            f"| {_dollars(queue.dollars_per_hour)} |"
        )
    return "\n".join(lines)


def _dollars(value: float) -> str:
    """Whole dollars with the sign before the symbol, so a loss reads -$3,395."""
    if not math.isfinite(value):
        return "-"
    return f"-${-value:,.0f}" if value < 0 else f"${value:,.0f}"
