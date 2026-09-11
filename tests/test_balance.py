"""Covariate balance: the analytic check, and the leak it is supposed to catch.

The interesting test here is not that the arithmetic is right, though that is checked
against a hand-worked case. It is that the diagnostic actually fires on a planted leak and
stays quiet on honest data, because that is the job it was added to do.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from itx.data import synthetic
from itx.metrics.balance import (
    CONVENTIONAL_THRESHOLD,
    standardised_mean_differences,
    worst_imbalance,
)
from itx.types import UpliftDataset


def dataset_with(columns: dict[str, list[float]], treatment: list[int]) -> UpliftDataset:
    """A minimal dataset carrying the given feature columns and arms."""
    return UpliftDataset(
        name="balance-fixture",
        features=pl.DataFrame(columns),
        treatment=np.array(treatment, dtype=np.int64),
        outcome=np.zeros(len(treatment), dtype=np.float64),
    )


class TestTheArithmetic:
    def test_a_hand_worked_case(self):
        # Treated 1, 2, 3: mean 2, sample variance 1. Control 4, 5, 6: mean 5, variance 1.
        # Pooled standard deviation is sqrt((1 + 1) / 2) = 1, so the difference of 3 in the
        # means standardises to exactly 3.
        data = dataset_with(
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
            treatment=[1, 1, 1, 0, 0, 0],
        )
        assert standardised_mean_differences(data)["x"] == pytest.approx(3.0)

    def test_it_is_absolute_so_the_sign_of_the_gap_does_not_matter(self):
        rising = dataset_with({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, [1, 1, 1, 0, 0, 0])
        falling = dataset_with({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, [0, 0, 0, 1, 1, 1])
        assert standardised_mean_differences(rising) == standardised_mean_differences(falling)

    def test_it_is_unitless_so_rescaling_a_column_changes_nothing(self):
        plain = dataset_with({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, [1, 1, 1, 0, 0, 0])
        scaled = dataset_with(
            {"x": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]}, [1, 1, 1, 0, 0, 0]
        )
        assert standardised_mean_differences(plain)["x"] == pytest.approx(
            standardised_mean_differences(scaled)["x"]
        )

    def test_identical_arms_have_no_difference(self):
        data = dataset_with({"x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]}, [1, 1, 1, 0, 0, 0])
        assert standardised_mean_differences(data)["x"] == pytest.approx(0.0)


class TestColumnsItDeclinesToScore:
    def test_a_constant_column_is_omitted_rather_than_dividing_by_zero(self):
        data = dataset_with({"x": [2.0] * 6}, [1, 1, 1, 0, 0, 0])
        assert "x" not in standardised_mean_differences(data)

    def test_a_column_with_too_few_present_rows_in_one_arm_is_omitted(self):
        nan = float("nan")
        data = dataset_with(
            {"x": [1.0, 2.0, 3.0, 4.0, nan, nan]},
            treatment=[1, 1, 1, 0, 0, 0],
        )
        assert "x" not in standardised_mean_differences(data)

    def test_missing_values_are_ignored_not_imputed(self):
        # The present rows are the hand-worked case above, with two NaNs added to each arm.
        # Imputing anything, a zero or a mean, would move the answer away from 3.
        nan = float("nan")
        data = dataset_with(
            {"x": [1.0, 2.0, 3.0, nan, nan, 4.0, 5.0, 6.0, nan, nan]},
            treatment=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        )
        assert standardised_mean_differences(data)["x"] == pytest.approx(3.0)

    def test_one_empty_arm_is_an_error_rather_than_a_silent_nothing(self):
        data = dataset_with({"x": [1.0, 2.0, 3.0]}, [1, 1, 1])
        with pytest.raises(ValueError, match="needs both arms"):
            standardised_mean_differences(data)

    def test_no_comparable_column_is_an_error_from_worst_imbalance(self):
        data = dataset_with({"x": [2.0] * 6}, [1, 1, 1, 0, 0, 0])
        with pytest.raises(ValueError, match="no feature could be compared"):
            worst_imbalance(data)


class TestTheJobItWasAddedToDo:
    """It has to fire on a leak and stay quiet on honest randomised data."""

    def test_randomised_data_is_balanced(self):
        data = synthetic.heterogeneous_effect(20_000, seed=7)
        _, value = worst_imbalance(data)
        assert value < CONVENTIONAL_THRESHOLD

    def test_a_column_that_separates_the_arms_perfectly_is_infinite_not_missing(self):
        # The severest leak: the column *is* the treatment. It has no variance inside
        # either arm, so the pooled standard deviation is zero, and the obvious handling
        # of a zero denominator drops the column and reports nothing. That is silence on
        # exactly the case worth shouting about, so it is reported as infinite instead.
        data = synthetic.heterogeneous_effect(20_000, seed=7)
        leaked = data.features.with_columns(
            pl.Series("exposure", data.treatment.astype(np.float64))
        )
        planted = UpliftDataset(
            name="planted",
            features=leaked,
            treatment=data.treatment,
            outcome=data.outcome,
        )
        name, value = worst_imbalance(planted)
        assert name == "exposure"
        assert value == float("inf")

    def test_a_partially_separating_column_is_caught_too(self):
        # Criteo's actual `exposure`: never set in control, set for some of the treated.
        # This one has variance inside the treated arm, so it standardises finitely.
        data = synthetic.heterogeneous_effect(20_000, seed=7)
        rng = np.random.default_rng(0)
        exposed = data.treatment * rng.binomial(1, 0.2, size=data.n_units)
        planted = UpliftDataset(
            name="planted",
            features=data.features.with_columns(
                pl.Series("exposure", exposed.astype(np.float64))
            ),
            treatment=data.treatment,
            outcome=data.outcome,
        )
        name, value = worst_imbalance(planted)
        assert name == "exposure"
        assert np.isfinite(value)
        assert value > CONVENTIONAL_THRESHOLD

    def test_a_partial_leak_is_caught_too(self):
        # Lenta's defect rather than Criteo's: a column present in both arms, but shifted.
        # This one is a tenth of a standard deviation, half the size of the real one.
        data = synthetic.heterogeneous_effect(20_000, seed=7)
        rng = np.random.default_rng(0)
        contaminated = rng.normal(size=data.n_units) + 0.1 * data.treatment
        planted = UpliftDataset(
            name="planted",
            features=data.features.with_columns(pl.Series("response_sms", contaminated)),
            treatment=data.treatment,
            outcome=data.outcome,
        )
        name, _ = worst_imbalance(planted)
        assert name == "response_sms"

    def test_a_confounded_dataset_is_imbalanced_by_construction(self):
        # The other reading of the same number: here imbalance is the confounding, not a
        # leak, and it is exactly what the dataset was generated to have.
        data = synthetic.confounded(20_000, seed=7)
        _, value = worst_imbalance(data)
        assert value > CONVENTIONAL_THRESHOLD
