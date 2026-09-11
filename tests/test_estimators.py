"""Every estimator has to recover an effect somebody wrote down.

The bar (PLAN.md section 5) is that an estimator recovers a known constant effect and a
known heterogeneous effect on synthetic data within tolerance. That is a test that can
actually fail: an estimator which ranks plausibly but is systematically wrong about the
size of the effect passes a Qini comparison and fails here.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from itx.data import synthetic
from itx.data.splits import stratified_split
from itx.estimators.base import DegenerateFitWarning, NotFittedError, UpliftEstimator
from itx.estimators.baselines import OutcomeRanking, RandomRanking
from itx.estimators.dr_learner import DRLearner
from itx.estimators.lightgbm_base import DEFAULT_CONFIG
from itx.estimators.propensity import PropensityModel
from itx.estimators.r_learner import RLearner
from itx.estimators.s_learner import SLearner
from itx.estimators.sklearn_bridge import LightGBMClassifier, LightGBMRegressor
from itx.estimators.t_learner import TLearner
from itx.estimators.x_learner import XLearner
from itx.metrics.ground_truth import ate_error, pehe
from itx.metrics.qini import ate
from itx.types import UpliftDataset

ESTIMATOR_FACTORIES = [
    pytest.param(lambda: SLearner(seed=0), id="s-learner"),
    pytest.param(lambda: TLearner(seed=0), id="t-learner"),
    pytest.param(lambda: XLearner(seed=0), id="x-learner"),
    pytest.param(lambda: DRLearner(seed=0), id="dr-learner"),
    pytest.param(lambda: RLearner(seed=0), id="r-learner"),
    pytest.param(lambda: OutcomeRanking(seed=0), id="outcome-ranking"),
    pytest.param(lambda: RandomRanking(seed=0), id="random"),
]

#: The estimators that claim to estimate an effect, and are therefore held to the
#: recovery bar: a known constant effect and a known heterogeneous one, within tolerance.
EFFECT_ESTIMATORS = [
    pytest.param(SLearner, id="s-learner"),
    pytest.param(TLearner, id="t-learner"),
    pytest.param(XLearner, id="x-learner"),
    pytest.param(DRLearner, id="dr-learner"),
    pytest.param(RLearner, id="r-learner"),
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


@pytest.mark.parametrize("estimator_class", EFFECT_ESTIMATORS)
def test_every_estimator_recovers_a_known_constant_effect(estimator_class, constant_data):
    split = stratified_split(constant_data, 11)
    predicted = estimator_class(seed=11).fit(split.train).predict_uplift(split.test.features)
    # The true effect is 1.0 for everyone. An estimator that cannot land on that has a bug
    # rather than a disagreement: there is no heterogeneity to get wrong.
    assert float(predicted.mean()) == pytest.approx(1.0, abs=0.15)
    assert ate_error(predicted, split.test.require_true_effect()) < 0.15


@pytest.mark.parametrize("estimator_class", EFFECT_ESTIMATORS)
def test_every_estimator_recovers_a_known_heterogeneous_effect(
    estimator_class, heterogeneous_data
):
    split = stratified_split(heterogeneous_data, 11)
    predicted = estimator_class(seed=11).fit(split.train).predict_uplift(split.test.features)
    truth = split.test.require_true_effect()

    assert ate_error(predicted, truth) < 0.2
    # Ranking is what this package exists to get right, so the ordering matters more than
    # the level: Spearman against the truth, well clear of chance.
    ranks = np.argsort(np.argsort(predicted))
    true_ranks = np.argsort(np.argsort(truth))
    assert np.corrcoef(ranks, true_ranks)[0, 1] > 0.6
    # And it has to beat the trivial estimator that predicts the average effect everywhere.
    assert pehe(predicted, truth) < pehe(np.full_like(truth, truth.mean()), truth)


@pytest.mark.parametrize("estimator_class", EFFECT_ESTIMATORS)
def test_every_estimator_recovers_the_effect_on_a_binary_outcome(estimator_class, binary_data):
    split = stratified_split(binary_data, 11)
    predicted = estimator_class(seed=11).fit(split.train).predict_uplift(split.test.features)
    assert ate_error(predicted, split.test.require_true_effect()) < 0.03


@pytest.mark.parametrize("estimator_class", EFFECT_ESTIMATORS)
def test_every_estimator_survives_a_badly_unbalanced_treated_arm(estimator_class):
    # The normal case in every application this is aimed at: a small treated group. The
    # estimates get noisier and that is legitimate; returning nothing, crashing, or losing
    # the average effect entirely is not.
    data = synthetic.heterogeneous_effect(6_000, propensity=0.08, seed=9)
    split = stratified_split(data, 11)
    predicted = estimator_class(seed=11).fit(split.train).predict_uplift(split.test.features)
    truth = split.test.require_true_effect()
    assert np.isfinite(predicted).all()
    assert ate_error(predicted, truth) < 0.4


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


class TestTLearner:
    def test_it_fits_one_model_per_arm(self, heterogeneous_data):
        estimator = TLearner(seed=0).fit(heterogeneous_data)
        treated, control = estimator.arm_sizes()
        assert treated == int((heterogeneous_data.treatment == 1).sum())
        assert control == int((heterogeneous_data.treatment == 0).sum())

    def test_the_arm_sizes_need_a_fitted_model(self):
        with pytest.raises(RuntimeError, match="fit before"):
            TLearner().arm_sizes()

    def test_an_empty_arm_is_an_error(self, heterogeneous_data):
        control_only = heterogeneous_data.take(
            np.flatnonzero(heterogeneous_data.treatment == 0)
        )
        with pytest.raises(ValueError, match="no treated rows"):
            TLearner(seed=0).fit(control_only)

    def test_it_cannot_regularise_the_treatment_away(self):
        # The S-learner's characteristic failure is structurally impossible here: the arms
        # are fitted on disjoint rows, so no leaf-size setting can collapse the two models
        # into one. A leaf size above half the row count makes any split impossible, which
        # silences the S-learner completely; the T-learner still answers, because each of
        # its models only has to describe one arm rather than the difference between two.
        small = synthetic.heterogeneous_effect(400, seed=5)
        config = DEFAULT_CONFIG.with_(min_child_samples=250)
        with pytest.warns(DegenerateFitWarning):
            SLearner(config, seed=0).fit(small)
        predicted = TLearner(config, seed=0).fit(small).predict_uplift(small.features)
        assert not np.allclose(predicted, 0.0)


class TestXLearner:
    def test_it_uses_a_known_propensity_rather_than_estimating_one(self, binary_data):
        estimator = XLearner(seed=0).fit(binary_data)
        assert estimator.propensity_fit is not None
        assert estimator.propensity_fit.known
        assert estimator.propensity_fit.n_clipped == 0

    def test_it_estimates_the_propensity_when_the_dataset_has_none(self, binary_data):
        stripped = binary_data.take(np.arange(binary_data.n_units))
        object.__setattr__(stripped, "propensity", None)
        estimator = XLearner(seed=0).fit(stripped)
        assert estimator.propensity_fit is not None
        assert not estimator.propensity_fit.known

    def test_the_imputed_effects_average_to_about_the_true_effect(self, heterogeneous_data):
        # Stage 2 is only as good as these. If the imputed effects are already wrong, the
        # combination rule cannot rescue them, so this is where to look first.
        estimator = XLearner(seed=0).fit(heterogeneous_data)
        treated, control = estimator.stage_two_targets(heterogeneous_data)
        truth = heterogeneous_data.require_true_effect().mean()
        assert float(treated.mean()) == pytest.approx(truth, abs=0.2)
        assert float(control.mean()) == pytest.approx(truth, abs=0.2)

    def test_the_stage_two_targets_need_a_fitted_model(self, heterogeneous_data):
        with pytest.raises(RuntimeError, match="fit before"):
            XLearner().stage_two_targets(heterogeneous_data)

    def test_an_empty_arm_is_an_error(self, heterogeneous_data):
        treated_only = heterogeneous_data.take(
            np.flatnonzero(heterogeneous_data.treatment == 1)
        )
        with pytest.raises(ValueError, match="no control rows"):
            XLearner(seed=0).fit(treated_only)


class TestPropensity:
    def test_a_known_propensity_is_adopted_unchanged(self, binary_data):
        fit = PropensityModel(seed=0).fit(binary_data)
        assert fit.known
        assert fit.values == pytest.approx(binary_data.propensity)
        assert "known by design" in fit.describe()

    def test_an_estimated_propensity_recovers_a_constant_assignment(self, binary_data):
        stripped = binary_data.take(np.arange(binary_data.n_units))
        object.__setattr__(stripped, "propensity", None)
        fit = PropensityModel(seed=0).fit(stripped)
        assert not fit.known
        # Assignment really was a coin flip, so a model of it should find nothing.
        assert float(fit.values.mean()) == pytest.approx(0.5, abs=0.05)

    def test_clipping_is_counted_rather_than_done_quietly(self):
        # Assignment determined by a covariate: perfect separation, no overlap, and every
        # inverse weight unbounded. Clipping keeps the arithmetic finite; the count is what
        # tells a reader the comparison is resting on the bound.
        data = separated_dataset(2_000, seed=0)
        fit = PropensityModel(seed=0).fit(data)
        assert fit.n_clipped > 0
        assert fit.has_overlap_problem
        assert "clipped" in fit.describe()

    def test_clipped_values_stay_inside_the_bound(self):
        data = separated_dataset(1_000, seed=1)
        model = PropensityModel(seed=0, clip=0.05)
        model.fit(data)
        values = model.predict(data.features)
        assert values.min() >= 0.05
        assert values.max() <= 0.95

    def test_predicting_before_fitting_is_an_error(self, binary_data):
        with pytest.raises(RuntimeError, match="fit before"):
            PropensityModel(seed=0).predict(binary_data.features)

    def test_a_known_constant_propensity_applies_to_any_number_of_rows(self, binary_data):
        model = PropensityModel(seed=0)
        model.fit(binary_data)
        subset = binary_data.take(np.arange(50))
        assert model.predict(subset.features) == pytest.approx(np.full(50, 0.5))

    def test_the_known_propensity_can_be_ignored_on_purpose(self, binary_data):
        fit = PropensityModel(seed=0, use_known=False).fit(binary_data)
        assert not fit.known


def separated_dataset(n_units: int, seed: int) -> UpliftDataset:
    """Treatment decided by a covariate: perfect separation and no overlap at all."""
    rng = np.random.default_rng(seed)
    driver = rng.normal(size=n_units)
    return UpliftDataset(
        name="separated",
        features=pl.DataFrame({"x0": driver, "x1": rng.normal(size=n_units)}),
        treatment=(driver > 0).astype(np.int64),
        outcome=rng.normal(size=n_units),
    )


class TestWhichLearnerWinsWhen:
    """The claim in docs/estimators.md, as a test that can fail.

    Meta-learners are not ranked once and for all: which one wins depends on whether the
    treatment effect is simpler or more complicated than the baseline it sits on. The two
    synthetic generators are built to be the two regimes, so the claim is checkable rather
    than quoted from a paper.
    """

    def test_the_s_learner_wins_when_the_effect_is_simpler_than_the_baseline(self):
        # heterogeneous_effect: nonlinear baseline in four covariates, near-linear effect
        # in two. One shared model carries the hard part and a few treatment splits carry
        # the rest, which is what the S-learner is built for.
        data = synthetic.heterogeneous_effect(6_000, seed=4)
        split = stratified_split(data, 11)
        truth = split.test.require_true_effect()
        scores = {
            name: pehe(
                estimator(seed=11).fit(split.train).predict_uplift(split.test.features), truth
            )
            for name, estimator in (("s", SLearner), ("t", TLearner), ("x", XLearner))
        }
        assert scores["s"] < scores["t"]

    def test_the_s_learner_loses_when_the_effect_is_harder_than_the_baseline(self):
        # complex_effect: near-linear baseline in one covariate, an interaction gated by a
        # hinge for the effect. Representing that inside a shared model costs many more
        # splits than fitting the two arms separately, and the ordering reverses.
        data = synthetic.complex_effect(6_000, seed=4)
        split = stratified_split(data, 11)
        truth = split.test.require_true_effect()
        scores = {
            name: pehe(
                estimator(seed=11).fit(split.train).predict_uplift(split.test.features), truth
            )
            for name, estimator in (("s", SLearner), ("t", TLearner), ("x", XLearner))
        }
        assert scores["t"] < scores["s"]
        assert scores["x"] < scores["s"]


class TestComplexGenerator:
    def test_the_published_effect_function_is_what_the_generator_used(self):
        data = synthetic.complex_effect(2_000, seed=0)
        covariates = data.features.to_numpy()
        assert data.require_true_effect() == pytest.approx(
            synthetic.true_complex_effect(covariates)
        )

    def test_the_effect_is_genuinely_heterogeneous(self):
        effect = synthetic.complex_effect(4_000, seed=0).require_true_effect()
        assert effect.std() > 1.0
        assert effect.min() < 0.0 < effect.max()


class TestConfoundedData:
    """The generator that exists so the confounding machinery has something to fail on."""

    def test_the_naive_comparison_gets_the_sign_wrong(self):
        # If this stops being true the generator has stopped confounding anything and every
        # test below it is passing for the wrong reason.
        data = synthetic.confounded(6_000, seed=0)
        assert data.require_true_effect().mean() > 0.2
        assert ate(data.outcome, data.treatment) < 0.0

    def test_strength_zero_is_a_coin_flip(self):
        data = synthetic.confounded(4_000, strength=0.0, seed=0)
        assert data.propensity is not None
        assert data.propensity == pytest.approx(np.full(data.n_units, 0.5))

    def test_the_true_propensity_is_recorded(self):
        data = synthetic.confounded(2_000, seed=0)
        assert data.propensity is not None
        assert 0.0 < data.propensity.min() < 0.1
        assert 0.9 < data.propensity.max() < 1.0


class TestCrossFittedPropensity:
    """The week 3 finding, as a test.

    An in-sample propensity is not a slightly worse propensity. It breaks the orthogonality
    the R-learner is built on, and the failure is invisible in every diagnostic except a
    comparison against a known truth.
    """

    def test_out_of_fold_is_the_default(self):
        data = synthetic.confounded(2_000, seed=0)
        fit = PropensityModel(seed=0, use_known=False).fit(data)
        assert fit.out_of_fold
        assert "out-of-fold" in fit.describe()

    def test_it_can_be_turned_off_so_the_cost_can_be_measured(self):
        data = synthetic.confounded(2_000, seed=0)
        fit = PropensityModel(seed=0, use_known=False, cross_fit=False).fit(data)
        assert not fit.out_of_fold
        assert "in-sample" in fit.describe()

    def test_an_in_sample_propensity_is_closer_to_the_realised_treatment(self):
        # The mechanism: fitted on its own rows, the model is pulled toward each unit's
        # actual treatment, so the residual it leaves behind is too small to be a residual.
        data = synthetic.confounded(4_000, seed=1)
        in_sample = PropensityModel(seed=0, use_known=False, cross_fit=False).fit(data).values
        out_of_fold = PropensityModel(seed=0, use_known=False).fit(data).values
        treatment = data.treatment.astype(float)
        assert np.abs(treatment - in_sample).mean() < np.abs(treatment - out_of_fold).mean()

    def test_the_out_of_fold_propensity_tracks_the_true_one_better(self):
        data = synthetic.confounded(4_000, seed=1)
        truth = data.propensity
        assert truth is not None
        in_sample = PropensityModel(seed=0, use_known=False, cross_fit=False).fit(data).values
        out_of_fold = PropensityModel(seed=0, use_known=False).fit(data).values
        assert np.abs(out_of_fold - truth).mean() < np.abs(in_sample - truth).mean()

    def test_the_r_learner_is_far_better_with_an_out_of_fold_propensity(self):
        data = synthetic.confounded(6_000, seed=2)
        split = stratified_split(data, 11)
        truth = split.test.require_true_effect()

        scores = {}
        for cross_fit in (False, True):
            estimator = RLearner(seed=11, use_known_propensity=False)
            estimator._propensity.cross_fit = cross_fit
            predicted = estimator.fit(split.train).predict_uplift(split.test.features)
            scores[cross_fit] = pehe(predicted, truth)
        assert scores[True] < scores[False]

    def test_folds_shrink_rather_than_fail_when_an_arm_is_thin(self):
        data = synthetic.confounded(600, strength=0.0, seed=3)
        thin = data.take(
            np.concatenate(
                [
                    np.flatnonzero(data.treatment == 1)[:3],
                    np.flatnonzero(data.treatment == 0),
                ]
            )
        )
        fit = PropensityModel(seed=0, use_known=False, folds=5).fit(thin)
        assert np.isfinite(fit.values).all()


class TestDRLearner:
    def test_it_reports_the_overlap_it_is_working_with(self):
        # EconML fits its own propensity whatever the dataset knows, so the reported
        # overlap has to describe an estimated propensity, not a design one.
        data = synthetic.confounded(3_000, seed=0)
        estimator = DRLearner(seed=0).fit(data)
        assert estimator.propensity_fit is not None
        assert not estimator.propensity_fit.known
        assert estimator.propensity_fit.out_of_fold

    def test_it_handles_confounding_the_naive_comparison_cannot(self):
        data = synthetic.confounded(8_000, seed=4)
        split = stratified_split(data, 11)
        truth = split.test.require_true_effect()
        predicted = DRLearner(seed=11).fit(split.train).predict_uplift(split.test.features)
        # The naive difference in arm means has the wrong sign here; a doubly robust
        # estimate should land near the truth instead.
        assert ate(split.test.outcome, split.test.treatment) < 0.0
        assert ate_error(predicted, truth) < 0.25

    def test_folds_shrink_rather_than_fail_when_an_arm_is_thin(self):
        data = synthetic.heterogeneous_effect(500, propensity=0.5, seed=5)
        thin = data.take(
            np.concatenate(
                [
                    np.flatnonzero(data.treatment == 1)[:3],
                    np.flatnonzero(data.treatment == 0),
                ]
            )
        )
        predicted = DRLearner(seed=0, folds=5).fit(thin).predict_uplift(thin.features)
        assert np.isfinite(predicted).all()


class TestRLearner:
    def test_it_uses_a_known_propensity_when_the_dataset_has_one(self, binary_data):
        estimator = RLearner(seed=0).fit(binary_data)
        assert estimator.propensity_fit is not None
        assert estimator.propensity_fit.known

    def test_it_handles_confounding_the_naive_comparison_cannot(self):
        data = synthetic.confounded(8_000, seed=4)
        split = stratified_split(data, 11)
        truth = split.test.require_true_effect()
        predicted = (
            RLearner(seed=11, use_known_propensity=False)
            .fit(split.train)
            .predict_uplift(split.test.features)
        )
        assert ate(split.test.outcome, split.test.treatment) < 0.0
        assert ate_error(predicted, truth) < 0.3


class TestSklearnBridge:
    """The wrapped libraries clone whatever they are handed, so it has to survive cloning."""

    def test_cloning_preserves_the_configuration(self):
        from sklearn.base import clone

        original = LightGBMRegressor(
            DEFAULT_CONFIG.with_(num_leaves=7), seed=3, categorical=(1,)
        )
        copy = clone(original)
        assert copy.config.num_leaves == 7
        assert copy.seed == 3
        assert copy.categorical == (1,)

    def test_the_regressor_fits_and_predicts(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 3))
        y = x[:, 0] * 2.0 + rng.normal(scale=0.1, size=500)
        model = LightGBMRegressor(seed=0).fit(x, y)
        assert np.corrcoef(model.predict(x), y)[0, 1] > 0.9

    def test_the_classifier_returns_probabilities_for_both_classes(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(500, 3))
        y = (x[:, 0] > 0).astype(int)
        model = LightGBMClassifier(seed=0).fit(x, y)
        probabilities = model.predict_proba(x)
        assert probabilities.shape == (500, 2)
        assert probabilities.sum(axis=1) == pytest.approx(np.ones(500))

    def test_a_single_class_target_does_not_crash_a_fold(self):
        # Possible inside a cross-fitting fold on a rare-event dataset.
        rng = np.random.default_rng(2)
        x = rng.normal(size=(50, 3))
        model = LightGBMClassifier(seed=0).fit(x, np.zeros(50, dtype=int))
        assert model.predict_proba(x).shape == (50, 1)
        assert model.predict(x) == pytest.approx(np.zeros(50))

    def test_the_categorical_columns_reach_lightgbm(self):
        # Two models on the same data, one told a column is categorical and one not. If the
        # declaration were being dropped they would be identical.
        rng = np.random.default_rng(3)
        codes = rng.integers(0, 5, size=800).astype(float)
        x = np.column_stack([codes, rng.normal(size=800)])
        y = np.array([0.0, 3.0, 1.0, 4.0, 2.0])[codes.astype(int)] + rng.normal(
            scale=0.1, size=800
        )
        plain = LightGBMRegressor(seed=0).fit(x, y).predict(x)
        declared = LightGBMRegressor(seed=0, categorical=(0,)).fit(x, y).predict(x)
        assert not np.allclose(plain, declared)


class TestMissingValues:
    """Every estimator has to survive a feature matrix with holes in it.

    Lenta has missing values in 150 of its 191 columns, and imputing them would put a
    modelling choice made in a loader into every estimator's input. LightGBM routes NaN
    down its own branch at every split, so nothing needs imputing, but only five of the six
    estimators here get to use that: EconML validates its inputs with scikit-learn's
    finiteness check before any model runs, so the DR-learner refused the matrix outright
    until it was told not to (PLAN.md change 29). This is the test that would have caught
    it, and the one that catches it coming back.
    """

    @pytest.fixture(scope="class")
    def holey(self):
        """A randomised dataset with a true effect of 0.5 and a quarter of its cells gone."""
        rng = np.random.default_rng(0)
        n = 6_000
        columns = {}
        for index in range(4):
            values = rng.normal(size=n)
            values[rng.random(n) < 0.25] = np.nan
            columns[f"x{index}"] = values
        treatment = rng.binomial(1, 0.5, n).astype(np.int64)
        return UpliftDataset(
            name="holey",
            features=pl.DataFrame(columns),
            treatment=treatment,
            outcome=(rng.normal(size=n) + 0.5 * treatment).astype(np.float64),
        )

    def test_the_fixture_really_does_have_holes(self, holey):
        assert np.isnan(holey.features.to_numpy()).any()

    @pytest.mark.parametrize("estimator_class", EFFECT_ESTIMATORS)
    def test_every_estimator_recovers_the_effect_through_the_holes(
        self, estimator_class, holey
    ):
        estimator = estimator_class(DEFAULT_CONFIG.with_(n_estimators=100), seed=0)
        estimator.fit(holey)
        predicted = estimator.predict_uplift(holey.features)
        assert np.isfinite(predicted).all()
        assert predicted.mean() == pytest.approx(0.5, abs=0.1)

    def test_no_warning_escapes_the_dr_learner(self, holey):
        # EconML warns on every fold and again on every effect() call. It is filtered at
        # the call site, narrowly; a warning leaking out here means the filter has drifted
        # away from the call, or that EconML has something new to say and is being muffled.
        estimator = DRLearner(DEFAULT_CONFIG.with_(n_estimators=100), seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            estimator.fit(holey)
            estimator.predict_uplift(holey.features)


@pytest.mark.parametrize("estimator_class", [SLearner, TLearner, XLearner])
def test_the_thread_count_does_not_change_the_answer(estimator_class, binary_data):
    """A result must not depend on how many cores the machine running it happened to have.

    ``deterministic`` and ``force_row_wise`` are set in ``BaseLearnerConfig`` for exactly
    this reason, and without them LightGBM's histogram construction can accumulate floating
    point sums in a thread-dependent order. This is the test that says the setting is doing
    its job, and it is why raising ``n_jobs`` from 4 to 8 (PLAN.md change 24) left every
    committed results table untouched and moved only ``fit_seconds``.

    Only the three hand-written learners are checked. The DR and R learners run inside
    EconML and CausalML, whose own parallelism this package does not control, so the
    guarantee here would be a claim about someone else's code.
    """
    split = stratified_split(binary_data, 11)
    predictions = []
    for jobs in (1, 4, 8):
        estimator = estimator_class(DEFAULT_CONFIG.with_(n_jobs=jobs), seed=0)
        estimator.fit(split.train)
        predictions.append(estimator.predict_uplift(split.test.features))

    assert np.array_equal(predictions[0], predictions[1])
    assert np.array_equal(predictions[0], predictions[2])
