"""A question the pipeline already knows the answer to, asked without telling it.

Rosenbaum bounds and E-values both price a hypothetical: they say how strong an unmeasured
confounder would have to be. Neither of them looks at whether this code, on this data, is
already producing effects out of nothing. A negative control does exactly that, and it is
the only one of the three that can fail.

The device is old and simple. Pick something the treatment cannot possibly have changed,
run the whole estimation on it as though it were the outcome, and see what comes back. The
true answer is zero. Anything else is bias: confounding the covariates did not absorb, a
mistake in the adjustment, or a leak.

## What is used as the control outcome

A pre-treatment covariate. These datasets record covariates as they stood before assignment,
so the treatment cannot have moved them, and that makes every column a candidate. The column
is dropped from the feature matrix before the models are fitted, because leaving it in lets
the outcome models predict it perfectly and the doubly robust estimator then returns zero for
a reason that has nothing to do with the data being unconfounded. Dropping it is what makes
this a test rather than a formality, and ``tests/test_negative_control.py`` asserts that a
planted confounder is caught, which it would not be if the column stayed in.

## Which column, and why the choice is made for the caller

:func:`hardest_control` picks the covariate most strongly associated with the real outcome.
That is the column with the best chance of failing, because whatever confounds the real
outcome is most likely to confound its closest neighbour. A negative control chosen to be
easy proves nothing, and a caller left to choose is a caller who can keep choosing until it
passes. The column that was used is reported next to the number so the choice is inspectable
(PLAN.md section 4: one negative control, one number).

## Reading the result

The estimate is the doubly robust average effect of treatment on the control outcome, in
standard deviations of that column so that datasets with different units can sit in one
table, with a bootstrap interval. An interval covering zero is a pass and means only that
this test did not detect bias, which is weaker than a clean bill of health. An interval
excluding zero is a fail and is worth more: the treatment cannot have moved a pre-treatment
covariate, so the pipeline has found an effect that is not there, and every other number it
produces on that dataset inherits the doubt.

ACIC and IHDP look like the place to check this device against a known answer, and they are
not. Their confounding is severe and entirely *measured*: assignment is simulated from the
recorded covariates, so the covariates are sufficient, the adjustment works, and a negative
control that passes there is reporting the truth rather than demonstrating its own
sensitivity. Both do pass, and week 5's doubly robust recovery of ACIC's true policy gains
to a mean absolute error of 0.11 says the same thing from the other side.

So the only evidence this device works is data built to break it, which is why
``tests/test_negative_control.py`` plants a confounder outside the covariate set and requires
the estimate to exclude zero. A method that cannot fail is not being tested, and on these
five datasets it never does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from itx.metrics.bootstrap import DEFAULT_LEVEL, DEFAULT_RESAMPLES, Estimate, bootstrap_ci
from itx.policy.policy_value import dr_gain, fit_nuisances

if TYPE_CHECKING:
    from itx.types import BoolArray, FloatArray, IntArray, UpliftDataset

#: Columns with fewer distinct values than this are skipped when picking a control
#: automatically. A constant column has no variance to standardise by, and a near-constant
#: one produces an estimate whose scale is an artefact of a handful of rows.
MIN_DISTINCT = 5


@dataclass(frozen=True, slots=True)
class NegativeControl:
    """What the pipeline found when it was asked about something it could not have changed.

    Attributes:
        column: The pre-treatment covariate used as the control outcome.
        effect: Doubly robust average effect of treatment on that column, in standard
            deviations of the column, with its bootstrap interval. The true value is zero.
        n_test: Rows the estimate was computed on.
        chosen: How the column was picked, either ``hardest`` or ``named``.
    """

    column: str
    effect: Estimate
    n_test: int
    chosen: str

    @property
    def passes(self) -> bool:
        """True when the interval covers zero, which is the only answer that is not a bug."""
        return not self.effect.excludes_zero

    def summary(self) -> str:
        """One paragraph a reader can act on."""
        where = (
            "the covariate most strongly associated with the real outcome, so the hardest "
            "of them to pass"
            if self.chosen == "hardest"
            else "a covariate chosen by name"
        )
        headline = (
            f"Negative control on {self.column!r}, {where}. The treatment cannot have moved "
            f"a pre-treatment covariate, so the true effect on it is zero. Estimated on "
            f"{self.n_test:,} test rows: {self.effect.format(4)} standard deviations."
        )
        if self.passes:
            return (
                f"{headline} The interval covers zero, so this test did not detect bias. "
                f"That is weaker than a clean bill of health: a negative control can only "
                f"fail, it cannot certify."
            )
        return (
            f"{headline} The interval excludes zero. The pipeline has found an effect that "
            f"cannot exist, so there is bias the covariates did not absorb, and every other "
            f"number on this dataset inherits the doubt."
        )


def hardest_control(data: UpliftDataset) -> str:
    """The covariate with the best chance of failing the test.

    Association with the real outcome is measured by absolute Spearman correlation, which is
    rank based and so does not care whether a column is a count, an indicator or a score, and
    does not assume the relationship is straight.

    Args:
        data: The dataset to choose a column from.

    Returns:
        The column name.

    Raises:
        ValueError: If no column has enough distinct values to serve as a control outcome.
    """
    usable = [
        name
        for name in data.feature_names
        if np.unique(data.features[name].to_numpy()).size >= MIN_DISTINCT
    ]
    if not usable:
        msg = (
            f"{data.name}: no covariate has {MIN_DISTINCT} distinct values, so none can "
            f"serve as a negative control outcome"
        )
        raise ValueError(msg)

    outcome_ranks = _ranks(data.outcome)
    strength = {
        name: abs(_correlation(_ranks(data.features[name].to_numpy()), outcome_ranks))
        for name in usable
    }
    return max(usable, key=lambda name: _nan_to_zero(strength[name]))


def negative_control(
    train: UpliftDataset,
    test: UpliftDataset,
    *,
    column: str | None = None,
    seed: int = 0,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
) -> NegativeControl:
    """Run the estimation on an outcome the treatment cannot have moved.

    Args:
        train: Rows the nuisance models are fitted on.
        test: Rows the effect is estimated on.
        column: Covariate to use as the control outcome; :func:`hardest_control` chooses
            one from ``train`` if omitted.
        seed: Seed for the models and the resampling.
        n_resamples: Bootstrap resamples.
        level: Nominal coverage.

    Returns:
        The estimate, standardised, with the column it came from.

    Raises:
        ValueError: If the named column is not a feature, or if it is constant on the test
            rows and there is no scale to standardise by.
    """
    chosen = "named" if column is not None else "hardest"
    name = hardest_control(train) if column is None else column
    if name not in train.feature_names:
        msg = f"{train.name}: {name!r} is not a feature column"
        raise ValueError(msg)

    spread = float(np.std(test.features[name].to_numpy()))
    if spread <= 0.0:
        msg = f"{test.name}: {name!r} is constant on the test rows, so it has no scale"
        raise ValueError(msg)

    swapped_train = _swap_outcome(train, name)
    swapped_test = _swap_outcome(test, name)
    nuisances = fit_nuisances(swapped_train, swapped_test, seed=seed)
    everyone: BoolArray = np.ones(swapped_test.n_units, dtype=bool)

    def statistic(index: IntArray) -> float:
        return (
            dr_gain(
                swapped_test.outcome[index],
                swapped_test.treatment[index],
                everyone[index],
                nuisances.take(index),
            )
            / spread
        )

    return NegativeControl(
        column=name,
        effect=bootstrap_ci(
            statistic,
            swapped_test.n_units,
            n_resamples=n_resamples,
            level=level,
            seed=seed,
        ),
        n_test=swapped_test.n_units,
        chosen=chosen,
    )


def _swap_outcome(data: UpliftDataset, column: str) -> UpliftDataset:
    """The same dataset with one covariate promoted to outcome and removed from the features.

    Args:
        data: The dataset.
        column: Covariate to promote.

    Returns:
        A dataset whose outcome is that column and whose features no longer contain it. Its
        ``true_effect`` is dropped, because the recorded truth is about the real outcome and
        carrying it forward would let a caller compute PEHE against the wrong thing.
    """
    return replace(
        data,
        name=f"{data.name}:{column}",
        features=data.features.drop(column),
        outcome=data.features[column].to_numpy().astype(np.float64),
        categorical=tuple(name for name in data.categorical if name != column),
        true_effect=None,
    )


def _ranks(values: FloatArray) -> FloatArray:
    """Average ranks, so that a tied column does not get an arbitrary order.

    SciPy's, rather than a hand-rolled one. ``itx.metrics.risk_deciles`` writes its own
    because SciPy's *correlation* warns on a constant input and a flat band table is an
    ordinary thing to measure; that reason does not apply to the ranking itself, which is
    silent, and :func:`_correlation` below handles the constant case without a warning.
    """
    ranked: FloatArray = stats.rankdata(values).astype(np.float64)
    return ranked


def _correlation(left: FloatArray, right: FloatArray) -> float:
    """Pearson correlation of two arrays, NaN when either is constant."""
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _nan_to_zero(value: float) -> float:
    """NaN sorts below every real correlation rather than winning the maximum."""
    return 0.0 if math.isnan(value) else value
