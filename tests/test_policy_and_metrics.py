"""Rank-and-cut, the ground-truth metrics, and the random-targeting reference."""

from __future__ import annotations

import numpy as np
import pytest

from itx.metrics.baselines import random_ranking_reference
from itx.metrics.ground_truth import ate_error, pehe
from itx.metrics.qini import ate, uplift_at_k
from itx.policy.rank_and_cut import n_targeted, rank_and_cut


class TestRankAndCut:
    def test_it_treats_the_requested_share(self):
        scores = np.linspace(0.0, 1.0, 1_000)
        assert rank_and_cut(scores, 0.25).sum() == 250

    def test_it_treats_the_highest_scoring_units(self):
        scores = np.array([0.1, 0.9, 0.4, 0.7])
        mask = rank_and_cut(scores, 0.5)
        assert mask.tolist() == [False, True, False, True]

    def test_a_full_budget_treats_everybody(self):
        scores = np.linspace(0.0, 1.0, 50)
        assert rank_and_cut(scores, 1.0).all()

    def test_a_tiny_budget_still_treats_one_unit(self):
        # A budget holder who says "the top one percent" of fifteen people means somebody,
        # not nobody.
        assert n_targeted(15, 0.01) == 1
        assert rank_and_cut(np.linspace(0, 1, 15), 0.01).sum() == 1

    def test_the_cut_rounds_up(self):
        assert n_targeted(15, 0.1) == 2

    @pytest.mark.parametrize("budget", [0.0, -0.1, 1.5])
    def test_an_impossible_budget_is_rejected(self, budget):
        with pytest.raises(ValueError, match="budget must be"):
            rank_and_cut(np.zeros(10), budget)

    def test_positive_only_declines_to_spend_on_predicted_harm(self):
        scores = np.array([0.5, 0.2, -0.1, -0.4])
        assert rank_and_cut(scores, 1.0).sum() == 4
        assert rank_and_cut(scores, 1.0, positive_only=True).tolist() == [
            True,
            True,
            False,
            False,
        ]

    def test_ties_are_broken_the_same_way_the_metrics_break_them(self):
        # A curve at budget b and the treated set at budget b have to describe the same
        # people, or the reported policy value belongs to a different policy.
        tied = np.zeros(100)
        first = rank_and_cut(tied, 0.3, seed=5)
        second = rank_and_cut(tied, 0.3, seed=5)
        assert first.tolist() == second.tolist()
        assert rank_and_cut(tied, 0.3, seed=6).tolist() != first.tolist()


class TestGroundTruth:
    def test_a_perfect_prediction_scores_zero(self):
        truth = np.array([1.0, 2.0, 3.0])
        assert pehe(truth, truth) == pytest.approx(0.0)
        assert ate_error(truth, truth) == pytest.approx(0.0)

    def test_pehe_is_the_root_mean_squared_error(self):
        predicted = np.array([1.0, 2.0, 3.0])
        truth = np.array([2.0, 2.0, 5.0])
        # Errors are -1, 0, -2; mean square 5/3; root is sqrt(5/3).
        assert pehe(predicted, truth) == pytest.approx(np.sqrt(5 / 3))

    def test_ate_error_ignores_errors_that_cancel(self):
        # The point of reporting both: this estimator has the population average exactly
        # right and every individual wrong.
        predicted = np.array([0.0, 4.0])
        truth = np.array([4.0, 0.0])
        assert ate_error(predicted, truth) == pytest.approx(0.0)
        assert pehe(predicted, truth) == pytest.approx(4.0)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            pehe(np.zeros(3), np.zeros(4))

    def test_an_empty_sample_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ate_error(np.array([]), np.array([]))


class TestRandomReference:
    def test_the_average_random_qini_is_about_zero(self, binary_data):
        from itx.metrics.qini import qini_coefficient

        estimate = random_ranking_reference(
            lambda scores: qini_coefficient(binary_data.outcome, binary_data.treatment, scores),
            binary_data.n_units,
            n_rankings=60,
            seed=0,
        )
        assert estimate.value == pytest.approx(0.0, abs=2e-3)
        assert estimate.low < 0.0 < estimate.high

    def test_the_average_random_uplift_is_about_the_population_effect(self, binary_data):
        # Targeting at random buys the average treatment effect, by definition. If this
        # drifts, the uplift-at-k metric is measuring something other than it claims.
        population = ate(binary_data.outcome, binary_data.treatment)
        estimate = random_ranking_reference(
            lambda scores: uplift_at_k(binary_data.outcome, binary_data.treatment, scores, 0.3),
            binary_data.n_units,
            n_rankings=60,
            seed=0,
        )
        assert estimate.value == pytest.approx(population, abs=0.01)


class TestAte:
    def test_it_is_the_difference_of_arm_means(self):
        outcome = np.array([1.0, 3.0, 0.0, 2.0])
        treatment = np.array([1, 1, 0, 0])
        assert ate(outcome, treatment) == pytest.approx(1.0)

    def test_an_empty_arm_is_an_error(self):
        with pytest.raises(ValueError, match="empty arm"):
            ate(np.array([1.0, 2.0]), np.array([1, 1]))

    def test_a_randomised_sample_recovers_the_true_average_effect(self, binary_data):
        assert ate(binary_data.outcome, binary_data.treatment) == pytest.approx(
            binary_data.require_true_effect().mean(), abs=0.015
        )


class TestUpliftAtK:
    def test_a_full_budget_equals_the_population_effect(self, binary_data):
        rng = np.random.default_rng(0)
        scores = rng.random(binary_data.n_units)
        assert uplift_at_k(
            binary_data.outcome, binary_data.treatment, scores, 1.0
        ) == pytest.approx(ate(binary_data.outcome, binary_data.treatment))

    def test_a_prefix_holding_one_arm_only_says_so_rather_than_guessing(self):
        outcome = np.array([1.0, 0.0, 1.0, 0.0])
        treatment = np.array([1, 1, 0, 0])
        scores = np.array([4.0, 3.0, 2.0, 1.0])
        assert np.isnan(uplift_at_k(outcome, treatment, scores, 0.25))

    @pytest.mark.parametrize("k", [0.0, -0.2, 1.2])
    def test_an_impossible_budget_is_rejected(self, k):
        with pytest.raises(ValueError, match="k must be"):
            uplift_at_k(np.zeros(4), np.array([0, 1, 0, 1]), np.zeros(4), k)
