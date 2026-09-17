"""How many units a targeting decision needs, asked before the data exists.

Every other module here reads data that has already been collected. This one is for the
question that comes first, and that the rest of this repository cannot answer: is it worth
running the experiment at all, and how big does it have to be. Somebody with no prior
treatment data cannot use any uplift model, and the honest advice to them is not "come back
when you have data" but "here is the smallest holdout that could answer your question".

## The two thresholds, and why the gap between them is the point

Detecting that an intervention does anything is a different problem from ranking who should
get it, and the second is far more expensive. Two things compound.

First, only a fraction of the population sits in the targeted prefix. Spending a budget on
the top 10% means the number that decides the precision is a tenth of the sample, so the
standard error on the prefix effect is larger than the standard error on the average effect
by a factor of one over the square root of the budget share.

Second, the quantity to be detected is smaller. Detecting an average effect means telling
the average from zero. Showing that targeting beats random means telling the prefix's effect
from the average, which is the heterogeneity rather than the effect, and heterogeneity is
usually the smaller number.

The ratio between the two requirements is::

    to rank / to detect an effect  =  1 / (budget x (lift ratio - 1)^2)

At a 20% budget with a top group that responds twice as well as average, ranking needs five
times the sample. At a 10% budget with a top group half again as good, it needs forty times.
A pilot sized to detect the average effect will report a targeting result that is noise, and
it will not look like noise: it will look like a table.

That gap is the whole reason this module exists, and it is why Lenta is in this repository.
687,029 customers, a real randomised campaign, and nothing separates from random targeting,
including every uplift model. It is large enough to measure the average effect many times
over and nowhere near large enough to rank on it.

## What this calculation is, exactly

A two-sample normal approximation on the difference of means, applied twice: once to the
whole sample against a null of no effect, and once to the targeted prefix against a null of
no heterogeneity. For a binary outcome the standard deviation is the usual ``sqrt(p(1-p))``;
for a continuous one the caller supplies it.

## What it is not, and these matter

**It assumes the ranking is handed to you.** The arithmetic prices measuring the prefix
effect once a ranking exists. In practice the same data has to learn the ranking as well,
and an estimate of heterogeneity fitted on the data it is evaluated on is worse than one
handed down from outside. So every number here is a floor: the real requirement is larger by
an amount that depends on the estimator and the signal, and this module will not pretend to
know it. Treat these as "certainly not less than".

**The lift ratio is a guess about the thing you are trying to discover.** How much better
the top group responds is exactly what the study is for, so it cannot be known in advance.
It also enters squared, which makes it the dominant uncertainty in the whole calculation:
being wrong by a factor of two moves the requirement by a factor of four. That is why
:func:`requirement_table` reports a row per lift ratio rather than a single number, and why
a single number from this module should be distrusted on principle.

**It says nothing about whether the design is valid.** A sample size large enough to detect
an effect is not an argument that the effect is identified. That is the assignment
mechanism's job, and on anything other than a randomised design it is an assumption the
analyst has to argue for rather than a quantity this file can compute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
#: Lift ratios reported by :func:`requirement_table`. A ratio of 1.25 is a top group a
#: quarter better than average, which is a modest and entirely plausible amount of
#: heterogeneity, and 3.0 is a strong one. The span is wide on purpose: the point of the
#: table is that the answer moves by orders of magnitude across it.
DEFAULT_LIFT_RATIOS = (1.25, 1.5, 2.0, 3.0)


def binary_outcome_sd(base_rate: float) -> float:
    """Standard deviation of a 0/1 outcome with the given base rate.

    Args:
        base_rate: Share of the population with the outcome when nobody intervenes, in
            ``(0, 1)``.

    Returns:
        ``sqrt(p(1-p))``.

    Raises:
        ValueError: If the base rate is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < base_rate < 1.0:
        msg = f"base rate must be in (0, 1), got {base_rate}"
        raise ValueError(msg)
    return math.sqrt(base_rate * (1.0 - base_rate))


def _multiplier(alpha: float, power: float) -> float:
    """The squared sum of the two normal quantiles, common to every formula here.

    Args:
        alpha: Two-sided significance level.
        power: Target power.

    Returns:
        ``(z_(1 - alpha/2) + z_power) ** 2``.

    Raises:
        ValueError: If either argument is outside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    if not 0.0 < power < 1.0:
        msg = f"power must be in (0, 1), got {power}"
        raise ValueError(msg)
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    return (z_alpha + z_power) ** 2


def _arm_factor(treated_share: float) -> float:
    """The cost of an unbalanced split, minimised at four when the arms are equal.

    Args:
        treated_share: Share of units assigned to treatment.

    Returns:
        ``1/h + 1/(1-h)``.

    Raises:
        ValueError: If the share is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < treated_share < 1.0:
        msg = f"treated share must be in (0, 1), got {treated_share}"
        raise ValueError(msg)
    return 1.0 / treated_share + 1.0 / (1.0 - treated_share)


def units_to_detect_an_effect(
    *,
    outcome_sd: float,
    average_effect: float,
    treated_share: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Units needed to tell the average effect from zero.

    This is the floor. It answers "did the intervention do anything", which is a weaker
    question than any question about targeting, and a study sized only for this one cannot
    support a targeting decision.

    Args:
        outcome_sd: Standard deviation of the outcome. For a rate use
            :func:`binary_outcome_sd`.
        average_effect: The average treatment effect worth detecting, in outcome units.
        treated_share: Share of units assigned to treatment.
        alpha: Two-sided significance level.
        power: Probability of detecting an effect of that size if it is real.

    Returns:
        Total units across both arms, rounded up.

    Raises:
        ValueError: If the standard deviation is not positive or the effect is zero.
    """
    if outcome_sd <= 0.0:
        msg = f"outcome_sd must be positive, got {outcome_sd}"
        raise ValueError(msg)
    if average_effect == 0.0:
        msg = "average_effect must be non-zero; an effect of zero needs infinite data"
        raise ValueError(msg)
    units = (
        _multiplier(alpha, power)
        * outcome_sd**2
        * _arm_factor(treated_share)
        / average_effect**2
    )
    return math.ceil(units)


def units_to_rank(
    *,
    outcome_sd: float,
    average_effect: float,
    lift_ratio: float,
    budget: float,
    treated_share: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Units needed to show that targeting the top ``budget`` share beats random targeting.

    The null is that the prefix responds exactly like the population, which is what random
    targeting delivers. The alternative is that it responds ``lift_ratio`` times as well.

    Args:
        outcome_sd: Standard deviation of the outcome.
        average_effect: The average treatment effect across the population.
        lift_ratio: How much better the targeted prefix responds than the average. 2.0 means
            twice as well. Must exceed 1: a prefix that responds like the average is what
            random targeting already gives, and no sample size distinguishes a thing from
            itself.
        budget: Share of the population the budget covers, in ``(0, 1]``.
        treated_share: Share of units assigned to treatment.
        alpha: Two-sided significance level.
        power: Probability of detecting heterogeneity of that size if it is real.

    Returns:
        Total units across both arms, rounded up.

    Raises:
        ValueError: If the budget is outside ``(0, 1]``, the lift ratio is not above one, or
            the standard deviation is not positive.
    """
    if not 0.0 < budget <= 1.0:
        msg = f"budget must be in (0, 1], got {budget}"
        raise ValueError(msg)
    if lift_ratio <= 1.0:
        msg = (
            f"lift_ratio must be above 1, got {lift_ratio}. A prefix that responds like the "
            f"population is what random targeting already gives you, and no sample size can "
            f"distinguish a thing from itself."
        )
        raise ValueError(msg)
    if outcome_sd <= 0.0:
        msg = f"outcome_sd must be positive, got {outcome_sd}"
        raise ValueError(msg)
    gap = (lift_ratio - 1.0) * average_effect
    units = (
        _multiplier(alpha, power)
        * outcome_sd**2
        * _arm_factor(treated_share)
        / (budget * gap**2)
    )
    return math.ceil(units)


def detectable_lift(
    *,
    n_units: int,
    outcome_sd: float,
    average_effect: float,
    budget: float,
    treated_share: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """The smallest lift ratio a study of this size could have detected.

    The inverse of :func:`units_to_rank`, and the more useful direction once data already
    exists: not "was this study big enough" but "how strong would the heterogeneity have had
    to be for this study to have seen it".

    Args:
        n_units: Total units across both arms.
        outcome_sd: Standard deviation of the outcome.
        average_effect: The average treatment effect across the population.
        budget: Share of the population the budget covers.
        treated_share: Share of units assigned to treatment.
        alpha: Two-sided significance level.
        power: Power the answer is quoted at.

    Returns:
        The lift ratio at the detection boundary. A returned 4.0 means only a top group
        responding four times as well as average would have been found by this study.

    Raises:
        ValueError: If ``n_units`` is not positive or the budget is outside ``(0, 1]``.
    """
    if n_units <= 0:
        msg = f"n_units must be positive, got {n_units}"
        raise ValueError(msg)
    if not 0.0 < budget <= 1.0:
        msg = f"budget must be in (0, 1], got {budget}"
        raise ValueError(msg)
    if average_effect == 0.0:
        msg = "average_effect must be non-zero"
        raise ValueError(msg)
    gap = math.sqrt(
        _multiplier(alpha, power)
        * outcome_sd**2
        * _arm_factor(treated_share)
        / (budget * n_units)
    )
    return 1.0 + gap / abs(average_effect)


@dataclass(frozen=True)
class Requirement:
    """One row of :func:`requirement_table`: what a given amount of heterogeneity costs.

    Attributes:
        lift_ratio: How much better the targeted prefix responds than the average.
        units: Total units across both arms needed to detect it.
        multiple_of_effect_detection: How many times the sample needed merely to show the
            intervention works at all. This is the column that surprises people.
    """

    lift_ratio: float
    units: int
    multiple_of_effect_detection: float


@dataclass(frozen=True)
class PowerReport:
    """A full answer to "how big does my pilot have to be".

    Attributes:
        outcome_sd: Standard deviation of the outcome used.
        average_effect: The average effect assumed.
        budget: Share of the population the budget covers.
        treated_share: Share assigned to treatment.
        alpha: Two-sided significance level.
        power: Power everything is quoted at.
        to_detect_an_effect: Units needed to tell the average effect from zero.
        rows: One requirement per lift ratio considered.
    """

    outcome_sd: float
    average_effect: float
    budget: float
    treated_share: float
    alpha: float
    power: float
    to_detect_an_effect: int
    rows: tuple[Requirement, ...]

    def to_markdown(self) -> str:
        """The table, for a README or a terminal.

        Returns:
            A markdown table, one row per lift ratio.
        """
        lines = [
            "| Top group responds | Units needed | Times the sample to detect any effect |",
            "|---|---:|---:|",
        ]
        lines.extend(
            f"| {row.lift_ratio:.2f}x average | {row.units:,} | "
            f"{row.multiple_of_effect_detection:,.0f}x |"
            for row in self.rows
        )
        return "\n".join(lines)

    def summary(self) -> str:
        """The sentences that stop the table being misread.

        Returns:
            Prose to print under the table.
        """
        return (
            f"\nTo detect that the intervention does anything at all: "
            f"{self.to_detect_an_effect:,} units, split {self.treated_share:.0%} treated.\n"
            f"To target a {self.budget:.0%} budget: the table above, and every row of it is a "
            f"floor.\n\n"
            f"Floors, because this prices measuring a ranking and not learning one. The same "
            f"data has to do both, so a real study needs more than these numbers and this "
            f"calculation will not guess how much more.\n"
            f"The spread down the table is the honest uncertainty: how much better the top "
            f"group responds is what the study exists to find out, so it cannot be known "
            f"first, and it enters the arithmetic squared."
        )


def requirement_table(
    *,
    outcome_sd: float,
    average_effect: float,
    budget: float,
    treated_share: float = 0.5,
    lift_ratios: tuple[float, ...] = DEFAULT_LIFT_RATIOS,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> PowerReport:
    """Size a pilot across a range of assumptions about heterogeneity.

    A single sample size from a single guess at the lift ratio is a false precision, because
    that guess is the quantity the study exists to measure. This reports a row per
    assumption so the reader sees how fast the answer moves.

    Args:
        outcome_sd: Standard deviation of the outcome. Use :func:`binary_outcome_sd` for a
            rate.
        average_effect: The average treatment effect expected, in outcome units.
        budget: Share of the population the intervention budget covers.
        treated_share: Share of units assigned to treatment.
        lift_ratios: Heterogeneity assumptions to report.
        alpha: Two-sided significance level.
        power: Power everything is quoted at.

    Returns:
        The finished report.
    """
    floor = units_to_detect_an_effect(
        outcome_sd=outcome_sd,
        average_effect=average_effect,
        treated_share=treated_share,
        alpha=alpha,
        power=power,
    )
    rows = []
    for ratio in lift_ratios:
        units = units_to_rank(
            outcome_sd=outcome_sd,
            average_effect=average_effect,
            lift_ratio=ratio,
            budget=budget,
            treated_share=treated_share,
            alpha=alpha,
            power=power,
        )
        rows.append(
            Requirement(
                lift_ratio=ratio,
                units=units,
                multiple_of_effect_detection=units / floor,
            )
        )
    return PowerReport(
        outcome_sd=outcome_sd,
        average_effect=average_effect,
        budget=budget,
        treated_share=treated_share,
        alpha=alpha,
        power=power,
        to_detect_an_effect=floor,
        rows=tuple(rows),
    )
