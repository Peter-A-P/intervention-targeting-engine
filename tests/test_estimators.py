"""Every estimator has to recover an effect somebody wrote down.

The bar (PLAN.md section 5) is that an estimator recovers a known constant effect and a
known heterogeneous effect on synthetic data within tolerance. That is a test that can
actually fail: an estimator which ranks plausibly but is systematically wrong about the
size of the effect passes a Qini comparison and fails here.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from itx.data import synthetic
from itx.data.splits import stratified_split
from itx.estimators.base import DegenerateFitWarning, NotFittedError, UpliftEstimator
from itx.estimators.baselines import OutcomeRanking, RandomRanking
from itx.estimators.lightgbm_base import DEFAULT_CONFIG
from itx.estimators.s_learner import SLearner
from itx.metrics.ground_truth import ate_error, pehe

ESTIMATOR_FACTORIES = [
    pytest.param(lambda: SLearner(seed=0), id="s-learner"),
    pytest.param(lambda: OutcomeRanking(seed=0), id="outcome-ranking"),
    pytest.param(lambda: RandomRanking(seed=0), id="random"),
]


@pytest.mark.parametrize("factory", ESTIMATOR_FACTORIES)
def test_every_estimator_satisfies_the_protocol(factory, constant_data):
    estimator = factory()
    assert isinstance(estimator, UpliftEstimator)
    fitted = estimator.fit(constant_data)
    assert fitted is estimator
    scores = estimator.predict_uplift(constant_data.features)
    assert scores.shape == (constant_data.n_units,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("factory", ESTIMATOR_FACTORIES)
def test_predicting_before_fitting_is_an_error(factory, constant_data):
    with pytest.raises(NotFittedError):
        factory().predict_uplift(constant_data.features)


@pytest.mark.parametrize("factory", ESTIMATOR_FACTORIES)
def test_scoring_a_different_feature_matrix_is_an_error(factory, constant_data):
    estimator = factory().fit(constant_data)
    renamed = constant_data.features.rename({"x0": "z0"})
    with pytest.raises(ValueError, match="feature columns"):
        estimator.predict_uplift(renamed)


@pytest.mark.parametrize("factory", ESTIMATOR_FACTORIES)
def test_fitting_twice_on_the_same_data_gives_the_same_scores(factory, constant_data):
    first = factory().fit(constant_data).predict_uplift(constant_data.features)
    second = factory().fit(constant_data).predict_uplift(constant_data.features)
    assert first == pytest.approx(second)


def test_s_learner_recovers_a_known_constant_effect(constant_data):
    split = stratified_split(constant_data, 11)
    estimator = SLearner(seed=11).fit(split.train)
    predicted = estimator.predict_uplift(split.test.features)
    # The true effect is 1.0 for everyone, so the average has to land on it and the spread
    # around it has to be small: there is no heterogeneity to find.
    assert float(predicted.mean()) == pytest.approx(1.0, abs=0.1)
    assert float(predicted.std()) < 0.35
    assert ate_error(predicted, split.test.require_true_effect()) < 0.1


def test_s_learner_recovers_a_known_heterogeneous_effect(heterogeneous_data):
    split = stratified_split(heterogeneous_data, 11)
    estimator = SLearner(seed=11).fit(split.train)
    predicted = estimator.predict_uplift(split.test.features)
    truth = split.test.require_true_effect()

    assert ate_error(predicted, truth) < 0.15
    # Ranking is the thing this package exists to get right, so the ordering matters more
    # than the level: Spearman against the truth, well clear of chance.
    ranks = np.argsort(np.argsort(predicted))
    true_ranks = np.argsort(np.argsort(truth))
    correlation = np.corrcoef(ranks, true_ranks)[0, 1]
    assert correlation > 0.6
    # And it has to beat the trivial estimator that predicts the average effect everywhere.
    assert pehe(predicted, truth) < pehe(np.full_like(truth, truth.mean()), truth)


def test_s_learner_recovers_the_effect_on_a_binary_outcome(binary_data):
    split = stratified_split(binary_data, 11)
    predicted = SLearner(seed=11).fit(split.train).predict_uplift(split.test.features)
    assert ate_error(predicted, split.test.require_true_effect()) < 0.02


def test_an_over_regularised_s_learner_warns_instead_of_returning_a_silent_zero():
    # The failure this guard exists for: a leaf size larger than the training set leaves
    # LightGBM unable to split at all, and the estimator returns exactly zero uplift for
    # everyone while reporting a PEHE identical to predicting nothing.
    small = synthetic.heterogeneous_effect(200, seed=5)
    config = DEFAULT_CONFIG.with_(min_child_samples=500)
    with pytest.warns(DegenerateFitWarning, match="zero for every unit"):
        estimator = SLearner(config, seed=0).fit(small)
    assert estimator.predict_uplift(small.features) == pytest.approx(np.zeros(small.n_units))


def test_the_s_learner_reports_how_often_it_split_on_the_treatment(heterogeneous_data):
    estimator = SLearner(seed=0).fit(heterogeneous_data)
    share = estimator.treatment_split_share()
    assert 0.0 < share < 1.0


def test_the_split_share_needs_a_fitted_model():
    with pytest.raises(RuntimeError, match="fit before"):
        SLearner().treatment_split_share()


def test_the_random_baseline_is_uncorrelated_with_the_truth(heterogeneous_data):
    scores = (
        RandomRanking(seed=7)
        .fit(heterogeneous_data)
        .predict_uplift(heterogeneous_data.features)
    )
    correlation = np.corrcoef(scores, heterogeneous_data.require_true_effect())[0, 1]
    assert abs(correlation) < 0.05


def test_the_random_baseline_repeats_for_a_given_seed(heterogeneous_data):
    first = (
        RandomRanking(seed=7)
        .fit(heterogeneous_data)
        .predict_uplift(heterogeneous_data.features)
    )
    second = (
        RandomRanking(seed=7)
        .fit(heterogeneous_data)
        .predict_uplift(heterogeneous_data.features)
    )
    assert first == pytest.approx(second)
    other = (
        RandomRanking(seed=8)
        .fit(heterogeneous_data)
        .predict_uplift(heterogeneous_data.features)
    )
    assert not np.allclose(first, other)


def test_the_outcome_ranking_predicts_outcomes_rather_than_effects(binary_data):
    # The whole point of the baseline: its scores track the outcome level, not the effect.
    # On this generator those are driven by different covariates, so the two orderings
    # disagree, and a reader who mistakes one for the other targets the wrong people.
    scores = OutcomeRanking(seed=0).fit(binary_data).predict_uplift(binary_data.features)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    with_effect = np.corrcoef(scores, binary_data.require_true_effect())[0, 1]
    with_outcome_driver = np.corrcoef(scores, binary_data.features["x2"].to_numpy())[0, 1]
    assert abs(with_outcome_driver) > abs(with_effect)


@pytest.mark.parametrize("fit_on", ["all", "treated", "control"])
def test_the_outcome_ranking_can_be_fitted_on_either_arm(fit_on, binary_data):
    scores = (
        OutcomeRanking(seed=0, fit_on=fit_on)
        .fit(binary_data)
        .predict_uplift(binary_data.features)
    )
    assert np.isfinite(scores).all()


def test_the_outcome_ranking_refuses_an_empty_arm(binary_data):
    control_only = binary_data.take(np.flatnonzero(binary_data.treatment == 0))
    with pytest.raises(ValueError, match="no treated rows"):
        OutcomeRanking(seed=0, fit_on="treated").fit(control_only)


def test_a_constant_outcome_does_not_crash_the_learner():
    # Possible on a thin split of a rare-event dataset: the classifier cannot be fitted,
    # and the benchmark row should still complete rather than taking down the run.
    data = synthetic.binary_outcome(400, seed=6)
    all_zero = pl.DataFrame(data.features)
    from itx.types import UpliftDataset

    flat = UpliftDataset(
        name="flat",
        features=all_zero,
        treatment=data.treatment,
        outcome=np.zeros(data.n_units),
    )
    with pytest.warns(DegenerateFitWarning):
        scores = SLearner(seed=0).fit(flat).predict_uplift(flat.features)
    assert scores == pytest.approx(np.zeros(data.n_units))


def test_policy_treats_the_requested_share_of_the_population(heterogeneous_data):
    estimator = SLearner(seed=0).fit(heterogeneous_data)
    mask = estimator.policy(heterogeneous_data.features, 0.2)
    assert mask.sum() == pytest.approx(0.2 * heterogeneous_data.n_units, rel=0.01)


def test_policy_treats_the_units_with_the_highest_predicted_uplift(heterogeneous_data):
    estimator = SLearner(seed=0).fit(heterogeneous_data)
    scores = estimator.predict_uplift(heterogeneous_data.features)
    mask = estimator.policy(heterogeneous_data.features, 0.1)
    assert scores[mask].min() >= scores[~mask].max()


def test_the_baselines_declare_that_they_do_not_estimate_an_effect():
    # PEHE compares a predicted effect with a true effect. A risk score is on the outcome's
    # scale, not the effect's, so scoring one against the truth reports a units mismatch
    # dressed up as an error. The benchmark reads this flag and leaves those cells empty.
    assert SLearner().estimates_effect
    assert not OutcomeRanking().estimates_effect
    assert not RandomRanking().estimates_effect
