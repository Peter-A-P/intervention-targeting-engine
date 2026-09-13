"""The negative control, on data where it must pass and on data where it must fail.

A test that can only pass is not a test. The centre of this file is
``test_it_catches_a_confounder_the_covariates_cannot_absorb``: a dataset is built where
assignment depends on one covariate and nothing else explains it, and the negative control
has to find the effect that cannot exist. Its companion builds the same dataset with
randomised assignment and requires a pass. If the first ever starts passing, the device has
stopped working and every reassuring number it produces elsewhere is worthless.

The other thing asserted here is that the control column leaves the feature matrix. Left in,
the outcome models predict it exactly, the doubly robust estimate collapses to zero, and the
test passes on every dataset for a reason that has nothing to do with confounding. That
failure mode is silent and total, so it gets its own test.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from itx.sensitivity.negative_control import (
    _swap_outcome,
    hardest_control,
    negative_control,
)
from itx.types import UpliftDataset


def make(n: int, *, confounded: bool, seed: int) -> UpliftDataset:
    """Four covariates, a binary outcome, and assignment that either does or does not lean.

    ``leak`` is the column assignment depends on when confounded. Nothing else in the frame
    predicts it, so no amount of adjustment on the other three can remove the association,
    which is exactly the situation a negative control exists to expose.
    """
    rng = np.random.default_rng(seed)
    leak = rng.normal(size=n)
    other = rng.normal(size=(n, 3))

    probability = 1.0 / (1.0 + np.exp(-1.5 * leak)) if confounded else np.full(n, 0.5)
    treatment = (rng.random(n) < probability).astype(np.int64)

    # The real outcome leans on the leak too, so the hardest-control chooser reaches for it.
    rate = 1.0 / (1.0 + np.exp(-(0.9 * leak + 0.3 * other[:, 0] + 0.4 * treatment)))
    outcome = (rng.random(n) < rate).astype(np.float64)

    features = pl.DataFrame(
        {
            "leak": leak,
            "a": other[:, 0],
            "b": other[:, 1],
            "c": other[:, 2],
        }
    )
    return UpliftDataset(
        name="synthetic", features=features, treatment=treatment, outcome=outcome
    )


def split(data: UpliftDataset):
    """A plain halving; the stratified splitter is tested elsewhere."""
    half = data.n_units // 2
    index = np.arange(data.n_units)
    return data.take(index[:half]), data.take(index[half:])


class TestChoosingTheColumn:
    def test_it_reaches_for_the_column_nearest_the_outcome(self):
        # 'leak' drives the real outcome hardest, so it is the column most likely to fail
        # and therefore the one a caller must not be able to avoid.
        assert hardest_control(make(4000, confounded=True, seed=0)) == "leak"

    def test_a_named_column_is_recorded_as_named(self):
        train, test = split(make(3000, confounded=False, seed=1))
        result = negative_control(train, test, column="b", n_resamples=100, seed=0)
        assert result.column == "b"
        assert result.chosen == "named"
        assert "chosen by name" in result.summary()

    def test_an_unknown_column_is_refused(self):
        train, test = split(make(1000, confounded=False, seed=2))
        with pytest.raises(ValueError, match="not a feature column"):
            negative_control(train, test, column="nope", n_resamples=10)

    def test_a_constant_column_is_refused(self):
        data = make(1000, confounded=False, seed=3)
        flat = UpliftDataset(
            name=data.name,
            features=data.features.with_columns(pl.lit(1.0).alias("a")),
            treatment=data.treatment,
            outcome=data.outcome,
        )
        train, test = split(flat)
        with pytest.raises(ValueError, match="constant on the test rows"):
            negative_control(train, test, column="a", n_resamples=10)


class TestSwappingTheOutcome:
    def test_the_column_leaves_the_feature_matrix(self):
        # The silent failure mode. A control outcome left in the features is predicted
        # perfectly, the doubly robust estimate is zero by construction, and the test
        # passes everywhere for no reason.
        data = make(500, confounded=False, seed=4)
        swapped = _swap_outcome(data, "leak")
        assert "leak" not in swapped.feature_names
        assert swapped.outcome.tolist() == data.features["leak"].to_list()
        assert swapped.n_units == data.n_units

    def test_the_recorded_truth_is_dropped(self):
        # true_effect describes the real outcome. Carrying it onto a swapped dataset would
        # let a caller compute PEHE against a quantity it is not about.
        data = make(500, confounded=False, seed=5)
        with_truth = UpliftDataset(
            name=data.name,
            features=data.features,
            treatment=data.treatment,
            outcome=data.outcome,
            true_effect=np.full(data.n_units, 0.4),
        )
        assert _swap_outcome(with_truth, "a").true_effect is None


class TestWhatItFinds:
    def test_it_passes_when_assignment_was_random(self):
        train, test = split(make(6000, confounded=False, seed=10))
        result = negative_control(train, test, n_resamples=200, seed=0)
        assert result.passes
        assert not result.effect.excludes_zero
        assert abs(result.effect.value) < 0.15  # in standard deviations of the column
        assert "did not detect bias" in result.summary()

    def test_it_catches_a_confounder_the_covariates_cannot_absorb(self):
        # The test that makes the other ones mean something. Assignment leans on 'leak' and
        # the remaining three covariates say nothing about it, so the pipeline reports an
        # effect of treatment on a pre-treatment covariate. That effect cannot exist.
        train, test = split(make(6000, confounded=True, seed=11))
        result = negative_control(train, test, column="leak", n_resamples=200, seed=0)
        assert not result.passes
        assert result.effect.excludes_zero
        assert result.effect.value > 0.5  # the lean is large and in the planted direction
        assert "effect that cannot exist" in result.summary()

    def test_a_covariate_unrelated_to_assignment_still_passes_on_confounded_data(self):
        # Confounding through 'leak' must not make every column fail. 'c' is independent of
        # assignment, so a failure there would mean the estimator is manufacturing effects
        # rather than detecting the planted one.
        train, test = split(make(6000, confounded=True, seed=12))
        result = negative_control(train, test, column="c", n_resamples=200, seed=0)
        assert result.passes

    def test_the_estimate_is_in_standard_deviations(self):
        # Scaling a column by ten must not change the answer, or a table holding dollars
        # next to counts would be reporting units rather than bias.
        data = make(4000, confounded=True, seed=13)
        scaled = UpliftDataset(
            name=data.name,
            features=data.features.with_columns((pl.col("leak") * 10.0).alias("leak")),
            treatment=data.treatment,
            outcome=data.outcome,
        )
        plain = negative_control(*split(data), column="leak", n_resamples=100, seed=0)
        stretched = negative_control(*split(scaled), column="leak", n_resamples=100, seed=0)
        assert stretched.effect.value == pytest.approx(plain.effect.value, rel=0.05)

    def test_it_reports_the_rows_it_used(self):
        train, test = split(make(3000, confounded=False, seed=14))
        result = negative_control(train, test, n_resamples=50, seed=0)
        assert result.n_test == test.n_units
