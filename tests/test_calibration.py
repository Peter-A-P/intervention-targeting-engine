"""Calibration: is the predicted uplift the right size, not just in the right order.

The analytic cases here are built so that the answer is known before the code runs: a
dataset whose realised uplift is exactly the prediction, one where it is exactly half, and
one where it is the prediction plus a constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from itx.data.splits import stratified_split
from itx.estimators.s_learner import SLearner
from itx.metrics.calibration import (
    calibration_error,
    calibration_slope,
    calibration_table,
    usable_bins,
)
from itx.types import UpliftDataset

import polars as pl  # isort: skip


def dataset_with_uplift(
    scores: np.ndarray, realised_multiplier: float, offset: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build outcomes whose realised uplift is a known function of the score.

    Every unit appears twice, once treated and once control, with the control outcome fixed
    at zero and the treated outcome set to the intended uplift. The realised uplift inside
    any bin is then exactly the mean intended uplift of that bin, with no sampling noise to
    argue about.

    Args:
        scores: Predicted uplift per unit.
        realised_multiplier: What the realised uplift is, as a multiple of the prediction.
        offset: A constant added to the realised uplift.
        seed: Unused, kept so callers read consistently.

    Returns:
        Outcome, treatment and score arrays, each of twice the length of ``scores``.
    """
    del seed
    intended = scores * realised_multiplier + offset
    outcome = np.concatenate([intended, np.zeros_like(intended)])
    treatment = np.concatenate(
        [np.ones_like(scores, dtype=np.int64), np.zeros_like(scores, dtype=np.int64)]
    )
    doubled = np.concatenate([scores, scores])
    return outcome, treatment, doubled


class TestCalibrationTable:
    def test_bins_are_ascending_in_predicted_uplift(self):
        scores = np.linspace(0.0, 1.0, 400)
        outcome, treatment, doubled = dataset_with_uplift(scores, 1.0)
        table = calibration_table(outcome, treatment, doubled, n_bins=5)
        assert np.all(np.diff(table.predicted) > 0)

    def test_a_perfectly_calibrated_model_sits_on_the_diagonal(self):
        scores = np.linspace(0.0, 1.0, 400)
        outcome, treatment, doubled = dataset_with_uplift(scores, 1.0)
        table = calibration_table(outcome, treatment, doubled, n_bins=5)
        assert table.realised == pytest.approx(table.predicted, abs=1e-9)

    def test_bins_hold_equal_numbers_of_units(self):
        scores = np.linspace(0.0, 1.0, 500)
        outcome, treatment, doubled = dataset_with_uplift(scores, 1.0)
        table = calibration_table(outcome, treatment, doubled, n_bins=10)
        assert set(table.n_units.tolist()) == {100}

    def test_a_bin_missing_an_arm_reports_nothing_rather_than_zero(self):
        # Treated units get the highest scores, so the top bin has no control unit and the
        # bottom none treated. Substituting zero there would drag every summary toward zero.
        outcome = np.array([1.0, 1.0, 0.0, 0.0])
        treatment = np.array([1, 1, 0, 0])
        scores = np.array([4.0, 3.0, 2.0, 1.0])
        table = calibration_table(outcome, treatment, scores, n_bins=2)
        assert np.isnan(table.realised).all()
        assert not table.usable.any()
        assert "no bin had both arms" in table.describe()

    def test_it_describes_the_worst_gap(self):
        scores = np.linspace(0.0, 1.0, 400)
        outcome, treatment, doubled = dataset_with_uplift(scores, 1.0, offset=0.25)
        table = calibration_table(outcome, treatment, doubled, n_bins=4)
        assert "largest gap 0.25" in table.describe()

    @pytest.mark.parametrize("n_bins", [0, 1])
    def test_too_few_bins_is_an_error(self, n_bins):
        with pytest.raises(ValueError, match="at least 2"):
            calibration_table(
                np.zeros(10), np.zeros(10, dtype=np.int64), np.zeros(10), n_bins=n_bins
            )

    def test_more_bins_than_units_is_an_error(self):
        with pytest.raises(ValueError, match="cannot exceed the sample size"):
            calibration_table(np.zeros(4), np.array([0, 1, 0, 1]), np.zeros(4), n_bins=10)


class TestCalibrationSlope:
    def test_a_perfect_model_scores_one(self):
        scores = np.linspace(0.0, 1.0, 600)
        outcome, treatment, doubled = dataset_with_uplift(scores, 1.0)
        # A noiseless construction gives every bin a zero standard error, so add a whisper
        # of noise to make the weighting well defined.
        rng = np.random.default_rng(0)
        outcome = outcome + rng.normal(scale=0.01, size=outcome.size)
        assert calibration_slope(outcome, treatment, doubled, n_bins=6) == pytest.approx(
            1.0, abs=0.05
        )

    def test_predictions_twice_too_large_score_a_half(self):
        scores = np.linspace(0.0, 1.0, 600)
        outcome, treatment, doubled = dataset_with_uplift(scores, 0.5)
        rng = np.random.default_rng(1)
        outcome = outcome + rng.normal(scale=0.01, size=outcome.size)
        assert calibration_slope(outcome, treatment, doubled, n_bins=6) == pytest.approx(
            0.5, abs=0.05
        )

    def test_a_constant_offset_leaves_the_slope_alone(self):
        scores = np.linspace(0.0, 1.0, 600)
        outcome, treatment, doubled = dataset_with_uplift(scores, 1.0, offset=0.3)
        rng = np.random.default_rng(2)
        outcome = outcome + rng.normal(scale=0.01, size=outcome.size)
        assert calibration_slope(outcome, treatment, doubled, n_bins=6) == pytest.approx(
            1.0, abs=0.05
        )

    def test_it_is_undefined_when_every_bin_predicts_the_same(self):
        outcome, treatment, doubled = dataset_with_uplift(np.full(400, 0.5), 1.0)
        assert np.isnan(calibration_slope(outcome, treatment, doubled, n_bins=4))


class TestCalibrationError:
    def test_a_perfect_model_scores_zero(self):
        scores = np.linspace(0.0, 1.0, 400)
        outcome, treatment, doubled = dataset_with_uplift(scores, 1.0)
        assert calibration_error(outcome, treatment, doubled, n_bins=5) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_a_constant_offset_is_exactly_that_offset(self):
        # The complement to the slope: this one does see an offset, and reports it in the
        # outcome's own units.
        scores = np.linspace(0.0, 1.0, 400)
        outcome, treatment, doubled = dataset_with_uplift(scores, 1.0, offset=0.2)
        assert calibration_error(outcome, treatment, doubled, n_bins=5) == pytest.approx(
            0.2, abs=1e-9
        )

    def test_it_is_undefined_when_no_bin_has_both_arms(self):
        outcome = np.array([1.0, 1.0, 0.0, 0.0])
        treatment = np.array([1, 1, 0, 0])
        scores = np.array([4.0, 3.0, 2.0, 1.0])
        assert np.isnan(calibration_error(outcome, treatment, scores, n_bins=2))


class TestOnFittedEstimators:
    def test_a_fitted_estimator_is_roughly_calibrated_on_randomised_data(self, binary_data):
        split = stratified_split(binary_data, 11)
        predicted = SLearner(seed=11).fit(split.train).predict_uplift(split.test.features)
        slope = calibration_slope(split.test.outcome, split.test.treatment, predicted, n_bins=5)
        # A wide band: five bins of a 2,400-row split is a noisy thing to fit a line to.
        assert 0.3 < slope < 2.0

    def test_inflating_every_prediction_halves_the_slope(self, binary_data):
        split = stratified_split(binary_data, 11)
        predicted = SLearner(seed=11).fit(split.train).predict_uplift(split.test.features)
        honest = calibration_slope(
            split.test.outcome, split.test.treatment, predicted, n_bins=5
        )
        inflated = calibration_slope(
            split.test.outcome, split.test.treatment, predicted * 2.0, n_bins=5
        )
        assert inflated == pytest.approx(honest / 2.0, rel=0.01)

    def test_scaling_predictions_leaves_the_ranking_metrics_alone(self, binary_data):
        # The reason calibration is reported at all: every other metric is blind to this.
        from itx.metrics.qini import qini_coefficient

        split = stratified_split(binary_data, 11)
        predicted = SLearner(seed=11).fit(split.train).predict_uplift(split.test.features)
        honest = qini_coefficient(split.test.outcome, split.test.treatment, predicted)
        inflated = qini_coefficient(split.test.outcome, split.test.treatment, predicted * 10.0)
        assert inflated == pytest.approx(honest)


class TestPlot:
    def test_it_writes_a_calibration_figure(self, tmp_path, binary_data):
        from itx.bench.plots import plot_calibration
        from itx.bench.runner import evaluate

        split = stratified_split(binary_data, 11)
        row = evaluate(SLearner(seed=11), split, n_resamples=10)
        path = plot_calibration([row], split.test, tmp_path / "calibration.png")
        assert path.exists()
        assert path.stat().st_size > 1_000

    def test_plotting_nothing_is_an_error(self, tmp_path, binary_data):
        from itx.bench.plots import plot_calibration

        with pytest.raises(ValueError, match="no benchmark rows"):
            plot_calibration([], binary_data, tmp_path / "calibration.png")


def test_the_synthetic_helper_builds_what_it_claims():
    # The fixture above is doing real work, so it gets its own check.
    scores = np.array([0.2, 0.8])
    outcome, treatment, doubled = dataset_with_uplift(scores, 1.0)
    frame = pl.DataFrame({"score": doubled, "y": outcome, "t": treatment})
    treated = frame.filter(pl.col("t") == 1)
    assert treated["y"].to_list() == pytest.approx([0.2, 0.8])
    assert (
        UpliftDataset(
            name="check",
            features=pl.DataFrame({"score": doubled}),
            treatment=treatment,
            outcome=outcome,
        ).n_units
        == 4
    )


class TestAdaptiveBinning:
    """Ten bins is a convention, not a law, and a small treated arm cannot support it."""

    def test_a_balanced_sample_gets_what_it_asked_for(self):
        treatment = np.tile([0, 1], 500)
        assert usable_bins(treatment) == 10

    def test_a_thin_treated_arm_caps_the_bin_count(self):
        # 150 units with 28 treated, which is IHDP's test split: ten bins would leave under
        # three treated units in each, and most bins would report nothing at all.
        treatment = np.concatenate([np.ones(28, dtype=np.int64), np.zeros(122, dtype=np.int64)])
        assert usable_bins(treatment) == 2

    def test_it_never_returns_fewer_than_two(self):
        treatment = np.concatenate([np.ones(3, dtype=np.int64), np.zeros(50, dtype=np.int64)])
        assert usable_bins(treatment) == 2

    def test_the_table_applies_the_cap(self):
        rng = np.random.default_rng(0)
        treatment = np.concatenate([np.ones(28, dtype=np.int64), np.zeros(122, dtype=np.int64)])
        outcome = rng.normal(size=150)
        scores = rng.normal(size=150)
        table = calibration_table(outcome, treatment, scores, n_bins=10)
        assert table.predicted.size == 2

    def test_an_undefined_estimate_renders_as_a_dash_not_as_nan(self):
        # A NaN point estimate beside a finite-looking interval invites a reader to believe
        # the interval means something.
        from itx.metrics.bootstrap import Estimate

        assert Estimate(float("nan"), -0.5, 0.5).format() == "-"
        assert Estimate(0.25, 0.1, 0.4).format(2) == "0.25 (0.10, 0.40)"
