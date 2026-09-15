"""How strong an unmeasured confounder would have to be to explain the targeting result away.

Every number this repository reports about a non-randomised dataset rests on an assumption
that cannot be tested: that the covariates carry all of the confounding. The estimators do
not check it, the bootstrap interval does not cover it, and a benchmark that reports only
the interval is quietly telling the reader that the assumption is safe.

The E-value (VanderWeele and Ding, 2017) is the smallest answer to a question a reader can
actually hold in their head: how strongly would some unmeasured thing have to be associated
with *both* the treatment and the outcome, on the risk ratio scale, for the effect we
measured to be entirely its doing? One number, no assumption about the confounder's
distribution, and it is a lower bound, so nothing weaker than it will do the job.

The arithmetic is closed form. For an observed risk ratio ``RR`` above 1,

    E = RR + sqrt(RR * (RR - 1))

and a ratio below 1 is inverted first, because "explaining away" is symmetric about the
null. An RR of 1 gives an E-value of exactly 1, which reads correctly: a result that is
already the null needs no confounding at all to become the null.

## Two E-values, and the second one is the one that matters

Reported for the point estimate and again for the confidence limit nearest the null. The
first says what it would take to move the estimate to no effect; the second says what it
would take to make the result stop excluding no effect, which is the weaker claim and so
always the smaller number. A result whose interval already covers the null has a limit
E-value of exactly 1: nothing has to be explained away, because nothing was established.
:func:`e_value_of` returns 1.0 in that case rather than refusing, since 1 is the true
answer and not a missing value.

## The risk ratio this is computed on

The targeted group's, not the population's, because the targeting decision is the thing
under test (PLAN.md section 4). Take the units a budget would treat, compare the outcome
rate among those of them that were actually treated against those that were not, and that
ratio is what a confounder would have to manufacture.

Binary outcomes give a risk ratio directly. Hillstrom, Criteo and Lenta are all binary and
they are the three the plan names for this measure. Continuous outcomes do not have one, so
:func:`risk_ratio_of` converts through the standardised mean difference using VanderWeele
and Ding's approximation, ``RR ~ exp(0.91 * d)``. That is an approximation and it is
labelled as one on the way out, in :attr:`EValue.scale`, because an E-value quoted from a
converted continuous outcome is a weaker object than one from a real rate and the reader
has to be able to tell which they are holding.

The conversion is exponential in ``d``, which matters more than it looks. It was built for
the modest effect sizes meta-analyses deal in, and a targeted group is selected to be
homogeneous, so its within-group spread is smaller than the population's and ``d`` is
correspondingly larger. ACIC at a 20% budget gives ``d`` of 1.8 and a converted ratio of
5.2, which is already at the edge of what the approximation was meant for. So ``d`` is
reported alongside the E-value in :attr:`EValue.standardised_difference` and named in the
summary, rather than being left inside a number that looks like a risk ratio and is not one.

## What an E-value does not do

It says nothing about whether such a confounder exists. On Hillstrom, Criteo and Lenta the
treatment was randomised, so the honest reading is not "the result survives confounding of
this strength" but "this is the strength of association randomisation is buying us, and
here is the number an observational version of this study would have to argue against".

ACIC and IHDP look like the place to check the device against a known answer, and they are
not, which is worth saying because the mistake is inviting. Their confounding is severe and
it is entirely *measured*: assignment is simulated from the recorded covariates, so the
covariates are sufficient and there is no unmeasured confounding in either. Week 5 measured
that from the other direction, with doubly robust estimation recovering ACIC's true policy
gains to a mean absolute error of 0.11 where the unadjusted comparison is out by 2.8 and of
the wrong sign. So ACIC's large E-value is correct rather than reassuring: it reports that
nothing unmeasured is at work, which is true, and it would report the same on data where
that was false for a reason it cannot see.

The only place these devices can be caught failing is data built to catch them, which is why
``tests/test_negative_control.py`` plants a confounder outside the covariate set and requires
:mod:`itx.sensitivity.negative_control` to find it. That test is the evidence any of this
works; the dataset numbers are not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itx.metrics.bootstrap import DEFAULT_LEVEL, DEFAULT_RESAMPLES, Estimate, bootstrap_ci
from itx.policy.rank_and_cut import rank_and_cut

if TYPE_CHECKING:
    from itx.types import BoolArray, FloatArray, IntArray

#: Coefficient converting a standardised mean difference to an approximate log risk ratio
#: (VanderWeele and Ding 2017, following Chinn 2000). Only used for continuous outcomes.
SMD_TO_LOG_RR = 0.91

#: Below this many units in either arm of the targeted group, the rates being divided are
#: too thin to divide and the risk ratio is reported as undefined rather than as a number.
#: Ten of each arm is the same floor the risk-decile table uses for a band, and it is here
#: for the same reason: a rate from five units is not a rate. It binds in practice. On IHDP
#: a 20% budget leaves five treated units in the targeted group, and the number that came
#: back from them was a standardised difference of 4.3 and an E-value of 102, which is not a
#: finding about confounding but a finding about five units.
MIN_ARM = 10


@dataclass(frozen=True, slots=True)
class EValue:
    """What it would take to explain one targeting result away.

    Attributes:
        risk_ratio: The observed ratio in the targeted group, with its bootstrap interval.
        point: E-value for the point estimate: the confounding strength that moves it to
            no effect.
        limit: E-value for the interval bound nearest the null: the strength that stops the
            result excluding no effect. Never larger than ``point``, and exactly 1.0 when
            the interval already covers the null.
        budget: Share of the population the targeted group is.
        n_targeted: Units in the targeted group.
        scale: ``rate`` when the outcome was binary and the ratio is a real risk ratio,
            ``approximate`` when it was continuous and came through
            :data:`SMD_TO_LOG_RR`.
        standardised_difference: The ``d`` the ratio was converted from on a continuous
            outcome, NaN on a binary one. Reported because the conversion is exponential
            in it and a reader cannot judge the E-value without it.
    """

    risk_ratio: Estimate
    point: float
    limit: float
    budget: float
    n_targeted: int
    scale: str
    standardised_difference: float = math.nan

    @property
    def established(self) -> bool:
        """True when the targeted group's interval excludes no effect."""
        return self.limit > 1.0

    def summary(self) -> str:
        """One paragraph a reader can act on, naming the number and what it means."""
        if not math.isfinite(self.point):
            return (
                f"At a {self.budget:.0%} budget the targeted group is too thin in one arm "
                f"to form a risk ratio, so there is no E-value to report."
            )
        if not self.established:
            return (
                f"At a {self.budget:.0%} budget the targeted group's risk ratio is "
                f"{self.risk_ratio.format(3)}, which covers 1. Nothing has to be explained "
                f"away, so the E-value for the interval is 1.0 and only the point estimate's "
                f"{self.point:.2f} is worth quoting, as what it would take to move an "
                f"estimate that was never established."
            )
        return (
            f"At a {self.budget:.0%} budget the targeted group's risk ratio is "
            f"{self.risk_ratio.format(3)}{self._approximation}. An unmeasured confounder "
            f"would need a risk ratio of at least {self.point:.2f} with both the treatment "
            f"and the outcome to move that to no effect, and at least {self.limit:.2f} to "
            f"stop it excluding no effect. Weaker confounding than that cannot account for "
            f"this result, however it is distributed."
        )

    @property
    def _approximation(self) -> str:
        """The caveat a converted continuous outcome has to carry, and nothing on a rate."""
        if self.scale == "rate":
            return ""
        return (
            f", which is not a rate but a standardised difference of "
            f"{self.standardised_difference:.2f} converted through exp(0.91d). The "
            f"conversion is exponential in d and was built for smaller ones, so read what "
            f"follows as an order of magnitude"
        )


def e_value_of(risk_ratio: float) -> float:
    """The smallest confounder association that could produce this risk ratio on its own.

    Args:
        risk_ratio: An observed ratio. Values below 1 are inverted first, because a
            protective effect and a harmful one of the same magnitude are equally hard to
            manufacture.

    Returns:
        ``RR + sqrt(RR * (RR - 1))`` on the inverted-to-above-1 ratio; 1.0 at the null; NaN
        if the ratio is not a positive finite number.
    """
    if not math.isfinite(risk_ratio) or risk_ratio <= 0.0:
        return math.nan
    above = risk_ratio if risk_ratio >= 1.0 else 1.0 / risk_ratio
    return above + math.sqrt(above * (above - 1.0))


def e_value_of_limit(risk_ratio: float, low: float, high: float) -> float:
    """E-value for whichever interval bound sits nearest the null.

    Args:
        risk_ratio: The point estimate, which decides which side of 1 the result is on.
        low: Lower bound of the interval on the ratio.
        high: Upper bound.

    Returns:
        The E-value of the bound closest to 1 on the estimate's own side, or exactly 1.0
        when the interval covers 1, since a result that does not exclude the null needs no
        confounding to stop excluding it.
    """
    if not math.isfinite(risk_ratio) or not math.isfinite(low) or not math.isfinite(high):
        return math.nan
    if low <= 1.0 <= high:
        return 1.0
    if (risk_ratio > 1.0) != (low > 1.0):
        # A skewed bootstrap on a thin group can put the whole interval on the other side
        # of the null from the point. That is a result that does not exclude the null in
        # any direction the point supports, so it needs no confounding to overturn.
        return 1.0
    return e_value_of(low if risk_ratio > 1.0 else high)


def risk_ratio_of(
    outcome: FloatArray, treatment: IntArray, targeted: BoolArray, *, binary: bool
) -> float:
    """The treated-to-control outcome ratio inside the targeted group.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        targeted: True for the units the policy would treat.
        binary: True when the outcome is 0/1 and the ratio of means is a real risk ratio.
            False converts a standardised mean difference through :data:`SMD_TO_LOG_RR`.

    Returns:
        The ratio, or NaN when either arm of the targeted group holds fewer than
        :data:`MIN_ARM` units, or when a binary outcome's control rate is zero and the
        ratio would be a division by nothing.
    """
    treated = outcome[targeted & (treatment == 1)]
    control = outcome[targeted & (treatment == 0)]
    if treated.size < MIN_ARM or control.size < MIN_ARM:
        return math.nan
    if binary:
        base = float(control.mean())
        if base <= 0.0:
            return math.nan
        return float(treated.mean()) / base
    return math.exp(SMD_TO_LOG_RR * _standardised_difference(treated, control))


def targeting_e_value(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    budget: float,
    binary: bool | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
    tie_seed: int | None = None,
) -> EValue:
    """E-values for what one ranking buys at one budget.

    The targeted group is fixed on the full sample and then carried through the bootstrap
    by row position, so every resample is asking about the same policy rather than about a
    policy the resample re-derived. Re-ranking inside the resample would fold the ranking's
    own sampling variation into a number that is meant to be about confounding.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Targeting scores, higher meaning treat sooner.
        budget: Share of the population to treat.
        binary: Whether the outcome is 0/1. Detected from the data if omitted.
        n_resamples: Bootstrap resamples for the risk ratio's interval.
        level: Nominal coverage.
        seed: Seed for the resampling, and for tie-breaking in the ranking unless
            ``tie_seed`` is given.
        tie_seed: Tie-breaking seed for the ranking, so that the targeted group can be the
            same one every other device and every benchmark table used.

    Returns:
        The E-values with the risk ratio they came from.
    """
    is_binary = _looks_binary(outcome) if binary is None else binary
    targeted = rank_and_cut(scores, budget, seed=seed if tie_seed is None else tie_seed)
    difference = (
        math.nan
        if is_binary
        else _standardised_difference(
            outcome[targeted & (treatment == 1)], outcome[targeted & (treatment == 0)]
        )
    )

    def statistic(index: IntArray) -> float:
        return risk_ratio_of(
            outcome[index], treatment[index], targeted[index], binary=is_binary
        )

    ratio = bootstrap_ci(
        statistic, outcome.size, n_resamples=n_resamples, level=level, seed=seed
    )
    return EValue(
        risk_ratio=ratio,
        point=e_value_of(ratio.value),
        limit=e_value_of_limit(ratio.value, ratio.low, ratio.high),
        budget=budget,
        n_targeted=int(targeted.sum()),
        scale="rate" if is_binary else "approximate",
        standardised_difference=difference,
    )


def _looks_binary(outcome: FloatArray) -> bool:
    """True when every observed outcome is 0 or 1."""
    return bool(np.isin(outcome, (0.0, 1.0)).all())


def _standardised_difference(treated: FloatArray, control: FloatArray) -> float:
    """Cohen's d between the two arms, pooling the variances.

    Args:
        treated: Outcomes of the treated units.
        control: Outcomes of the control units.

    Returns:
        The standardised difference, or NaN when both arms are constant and there is no
        scale to standardise by.
    """
    pooled = math.sqrt((float(treated.var(ddof=1)) + float(control.var(ddof=1))) / 2.0)
    if pooled <= 0.0:
        return math.nan
    return (float(treated.mean()) - float(control.mean())) / pooled
