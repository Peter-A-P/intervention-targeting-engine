"""The one table that says whether the rest of this repository is worth running.

The project's headline finding is that ranking by risk, which is what almost every targeting
system in production actually does, sometimes matches uplift modelling exactly and sometimes
is ten times worse than picking names out of a hat. On Criteo's 14 million randomised rows
the outcome ranking matched every uplift model here; on ACIC it was worse than random by a
factor of ten. That is not a contradiction and it is not a coin flip, and this module is the
cheap measurement that tells the two cases apart in advance (PLAN.md change 32).

## The mechanism

Absolute uplift is baseline risk times the relative effect:

    ``tau(x) = p0(x) * (multiplier(x) - 1)``

where ``p0`` is the probability of the outcome with no intervention and ``multiplier`` is
what the intervention does to it. Ranking by ``p0`` and ranking by ``tau`` give the same list
whenever the spread in ``p0`` across the population dominates the spread in the multiplier.
That is a property of the data, it varies enormously between problems, and it costs one
outcome model to measure.

Cut the population into deciles of predicted risk, and in each one measure what the
intervention actually did. Three numbers come out:

**Risk spread.** How far the baseline rate runs from the top decile to the bottom. Criteo's
spans about 1,500 times, Hillstrom's about 6.

**Multiplier spread.** How far the relative effect moves across those same deciles. Around 2
on Criteo, around 2.8 on Hillstrom.

**Rank correlation.** Whether the deciles with the most risk are the deciles with the most
uplift. Positive and large means the two rankings agree, and the ordinary risk model was
already doing the right thing. Negative means the effect runs against risk, the case where
spending the budget on the highest-risk cases is worse than spending it at random, which is
what ACIC does.

## What this is not

It is not a substitute for the benchmark, and a favourable verdict here is not a measured
result. It reads a decile summary rather than an individual effect, so it can only say
whether the two *rankings* are likely to differ, not by how much a particular estimator will
beat a particular baseline. What it is for is deciding whether to spend the afternoon: a
problem whose effect runs with risk and whose risk spans three orders of magnitude is a
problem where uplift modelling has little room to add anything, and knowing that before
fitting five meta-learners is worth one outcome model.

Every number here carries a bootstrap interval, including the spreads and the correlation,
because a decile table on a small test split can produce a confident-looking ordering out of
nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itx.estimators.baselines import OutcomeRanking
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig
from itx.metrics.bootstrap import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    Estimate,
    bootstrap_vector,
)
from itx.metrics.curves import DEFAULT_TIE_SEED, rank_order

if TYPE_CHECKING:
    from collections.abc import Callable

    from itx.types import FloatArray, IntArray, UpliftDataset

#: Deciles, unless a caller asks for something else. Ten is the convention every risk team
#: already reads, which is the point: this table has to be recognisable to somebody who has
#: never heard of uplift modelling.
DEFAULT_BINS = 10

#: A rank correlation at or above this, with risk spread dominating, is the case where a
#: plain risk model is already close to the right list.
AGREEMENT = 0.6

@dataclass(frozen=True, slots=True)
class Decile:
    """One band of the population, ordered by predicted risk.

    Attributes:
        index: 0 for the highest-risk band.
        n_units: Rows in the band.
        n_treated: How many of them were treated.
        predicted_risk: Mean predicted probability of the outcome without intervention.
        control_rate: Observed outcome rate among the band's control rows.
        treated_rate: Observed outcome rate among the band's treated rows.
        uplift: Treated rate minus control rate, with its interval.
    """

    index: int
    n_units: int
    n_treated: int
    predicted_risk: float
    control_rate: float
    treated_rate: float
    uplift: Estimate

    @property
    def multiplier(self) -> float:
        """Relative effect: what the intervention multiplied the outcome rate by.

        NaN where the band's control rows produced no events at all, which is a real answer
        rather than an infinite one: no relative effect can be formed against a rate of
        zero.
        """
        if self.control_rate == 0.0:
            return float("nan")
        return self.treated_rate / self.control_rate


@dataclass(frozen=True, slots=True)
class RiskDecileTable:
    """The finished diagnostic: the bands, the three summary numbers, and the verdict.

    Attributes:
        dataset: Dataset name.
        n_test: Rows the table was measured on.
        deciles: The bands, highest risk first.
        risk_spread: Highest band's control rate divided by the lowest's, with its interval.
        multiplier_spread: Largest relative effect divided by the smallest, with its
            interval.
        correlation: Rank correlation between a band's risk and its uplift, with its
            interval. Positive means risk ranking and uplift ranking broadly agree.
    """

    dataset: str
    n_test: int
    deciles: tuple[Decile, ...]
    risk_spread: Estimate
    multiplier_spread: Estimate
    correlation: Estimate

    @property
    def risk_dominates(self) -> bool:
        """True when the baseline rate varies more across bands than the relative effect."""
        return (
            np.isfinite(self.risk_spread.value)
            and np.isfinite(self.multiplier_spread.value)
            and self.risk_spread.value > self.multiplier_spread.value
        )

    @property
    def verdict(self) -> str:
        """One sentence on whether uplift modelling has room to add anything here.

        A judgement about the two rankings, not a measured result: see the module docstring
        on what this table cannot tell you.

        Both confident verdicts are gated on the correlation's interval rather than on its
        point estimate, and the difference is not pedantic. A ten-band table on a few
        thousand rows will readily produce a correlation of -0.32 that means nothing, and
        announcing from it that a client's effect runs against their risk is exactly the
        failure this repository was built to demonstrate. When the interval contains zero
        the answer is that this split cannot tell, and that is what it says.
        """
        if self.correlation.high < 0.0:
            return (
                "The effect runs against risk. Spending the budget on the highest-risk "
                "cases will be worse than spending it at random, and uplift modelling is "
                "not an improvement here so much as the difference between helping and "
                "doing harm."
            )
        agrees = self.correlation.low > 0.0 and self.correlation.value >= AGREEMENT
        if agrees and self.risk_dominates:
            return (
                "Risk ranking and uplift ranking broadly agree, because the baseline rate "
                f"varies {self.risk_spread.value:,.0f}x across bands against the relative "
                f"effect's {self.multiplier_spread.value:,.1f}x. An ordinary risk model is "
                "already close to the right list and uplift modelling has little room."
            )
        if self.correlation.excludes_zero:
            return (
                "Risk and uplift rankings agree in part and disagree in part. Uplift "
                "modelling is worth measuring here: how much it buys is what the benchmark "
                "is for."
            )
        return (
            "This split cannot tell risk ranking and uplift ranking apart: the correlation "
            "between a band's risk and its uplift has an interval containing zero. Either "
            "the effect is too small to see at this sample size or it does not vary."
        )

    def to_markdown(self, digits: int = 4) -> str:
        """The table as markdown, one row per band.

        Args:
            digits: Decimal places for the rates.

        Returns:
            A markdown table, highest-risk band first.
        """
        lines = [
            "| Decile | Units | Predicted risk | Control rate | Treated rate | Multiplier "
            "| Uplift (95% CI) |",
            "|---|---|---|---|---|---|---|",
        ]
        for band in self.deciles:
            multiplier = (
                "-" if not np.isfinite(band.multiplier) else f"{band.multiplier:.2f}x"
            )
            lines.append(
                f"| {band.index + 1} | {band.n_units:,} | "
                f"{band.predicted_risk:.{digits}f} | {band.control_rate:.{digits}f} | "
                f"{band.treated_rate:.{digits}f} | {multiplier} | "
                f"{band.uplift.format(digits)} |"
            )
        return "\n".join(lines) + "\n"

    def summary(self) -> str:
        """The three numbers and the verdict, for the command line."""
        return "\n".join(
            [
                f"risk spread across bands:       {self.risk_spread.format(2)}",
                f"multiplier spread across bands: {self.multiplier_spread.format(2)}",
                f"risk-to-uplift correlation:     {self.correlation.format(3)}",
                "",
                self.verdict,
            ]
        )


def risk_scores(
    train: UpliftDataset,
    test: UpliftDataset,
    *,
    config: BaseLearnerConfig = DEFAULT_CONFIG,
    seed: int = 0,
) -> FloatArray:
    """Predicted probability of the outcome without intervention, for the test rows.

    Fitted on the *control* rows only, because that is what a churn score, a fraud score or
    a readmission score actually is: a model of what happens when nobody intervenes. Fitting
    on everybody would mix the treated arm's outcomes into the baseline and flatten exactly
    the spread this table is trying to measure.

    Args:
        train: Rows to fit on. Only its control rows are used.
        test: Rows to score.
        config: Shared LightGBM settings.
        seed: Seed for the model.

    Returns:
        One predicted risk per test row.
    """
    model = OutcomeRanking(config, seed=seed, fit_on="control")
    model.fit(train)
    return model.predict_uplift(test.features)


def risk_deciles(
    train: UpliftDataset,
    test: UpliftDataset,
    *,
    bins: int = DEFAULT_BINS,
    config: BaseLearnerConfig = DEFAULT_CONFIG,
    seed: int = 0,
    tie_seed: int = DEFAULT_TIE_SEED,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
) -> RiskDecileTable:
    """Measure what the intervention did, band by band of predicted risk.

    Args:
        train: Rows to fit the risk model on.
        test: Rows to measure on.
        bins: How many bands to cut the population into.
        config: Shared LightGBM settings.
        seed: Seed for the risk model.
        tie_seed: Seed for tie-breaking the risk ranking, so that a model producing large
            blocks of identical scores still cuts reproducibly.
        n_resamples: Bootstrap resamples for every interval in the table.
        level: Nominal coverage.

    Returns:
        The finished table.

    Raises:
        ValueError: If fewer than two bands are asked for, or the test split has fewer rows
            than bands.
    """
    if bins < 2:
        msg = f"a decile table needs at least two bands, got {bins}"
        raise ValueError(msg)
    if test.n_units < bins:
        msg = f"cannot cut {test.n_units} rows into {bins} bands"
        raise ValueError(msg)

    scores = risk_scores(train, test, config=config, seed=seed)
    # Highest risk first, so band 1 is the group a risk-ranked budget would be spent on.
    band_of = _band_assignment(scores, bins, tie_seed=tie_seed)

    estimates = bootstrap_vector(
        _statistics(test.outcome, test.treatment, band_of, bins),
        test.n_units,
        n_resamples=n_resamples,
        level=level,
        seed=seed,
    )

    full = np.arange(test.n_units)
    point = _statistics(test.outcome, test.treatment, band_of, bins)(full)
    deciles = tuple(
        Decile(
            index=band,
            n_units=int((band_of == band).sum()),
            n_treated=int(((band_of == band) & (test.treatment == 1)).sum()),
            predicted_risk=float(scores[band_of == band].mean()),
            control_rate=point[f"control@{band}"],
            treated_rate=point[f"treated@{band}"],
            uplift=estimates[f"uplift@{band}"],
        )
        for band in range(bins)
    )

    return RiskDecileTable(
        dataset=test.name,
        n_test=test.n_units,
        deciles=deciles,
        risk_spread=estimates["risk_spread"],
        multiplier_spread=estimates["multiplier_spread"],
        correlation=estimates["correlation"],
    )


def _band_assignment(scores: FloatArray, bins: int, *, tie_seed: int) -> IntArray:
    """Which band each unit falls in, highest risk first, bands as equal as they divide.

    Cut on the ranking rather than on the score, so that a model producing large blocks of
    identical probabilities still gives bands of the intended size instead of one band
    holding half the population.
    """
    order = rank_order(scores, seed=tie_seed)
    band_of: IntArray = np.empty(scores.size, dtype=np.int64)
    edges = np.linspace(0, scores.size, bins + 1).astype(int)
    for band in range(bins):
        band_of[order[edges[band] : edges[band + 1]]] = band
    return band_of


def _statistics(
    outcome: FloatArray, treatment: IntArray, band_of: IntArray, bins: int
) -> Callable[[IntArray], dict[str, float]]:
    """Every number in the table, as one function of the resampled row positions.

    One function rather than one per number, because the three summary statistics are
    derived from the per-band rates and have to be computed from the *same* resample as
    them. Bootstrapping the bands and the spreads separately would report a spread whose
    interval did not correspond to any actual redraw of the bands.
    """

    def statistics(index: IntArray) -> dict[str, float]:
        bands = band_of[index]
        treated = treatment[index] == 1
        outcomes = outcome[index]

        values: dict[str, float] = {}
        control_rates: list[float] = []
        multipliers: list[float] = []
        uplifts: list[float] = []
        for band in range(bins):
            here = bands == band
            in_control = here & ~treated
            in_treated = here & treated
            control = float(outcomes[in_control].mean()) if in_control.any() else float("nan")
            treat = float(outcomes[in_treated].mean()) if in_treated.any() else float("nan")
            values[f"control@{band}"] = control
            values[f"treated@{band}"] = treat
            values[f"uplift@{band}"] = treat - control
            control_rates.append(control)
            multipliers.append(treat / control if control > 0.0 else float("nan"))
            uplifts.append(treat - control)

        values["risk_spread"] = _spread(np.array(control_rates))
        values["multiplier_spread"] = _spread(np.array(multipliers))
        values["correlation"] = _rank_correlation(
            np.array(control_rates), np.array(uplifts)
        )
        return values

    return statistics


def _spread(values: FloatArray) -> float:
    """Largest finite value divided by the smallest, or NaN if that cannot be formed."""
    finite = values[np.isfinite(values)]
    if finite.size < 2 or finite.min() <= 0.0:
        return float("nan")
    return float(finite.max() / finite.min())


def _rank_correlation(risk: FloatArray, uplift: FloatArray) -> float:
    """Spearman correlation between a band's baseline risk and its measured uplift.

    Computed here rather than taken from SciPy for one reason: SciPy returns NaN with a
    warning when either input is constant, and a band table where every band has the same
    uplift is a perfectly ordinary thing to measure on a dataset with no heterogeneity. That
    case has an answer, which is zero correlation, and it should not arrive as a warning.
    """
    usable = np.isfinite(risk) & np.isfinite(uplift)
    if usable.sum() < 3:
        return float("nan")
    left = _ranks(risk[usable])
    right = _ranks(uplift[usable])
    if left.std() == 0.0 or right.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _ranks(values: FloatArray) -> FloatArray:
    """Ranks of the values, ties sharing the average rank."""
    order = np.argsort(values, kind="stable")
    ranks: FloatArray = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    # Ties share their average rank, so that a block of equal uplifts does not get an
    # ordering invented for it by the sort.
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks
