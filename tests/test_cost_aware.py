"""The cost-aware knapsack, against brute force and against the headcount rule.

The two properties worth protecting. When every intervention costs the same, this has to
agree exactly with ``rank_and_cut``, or the two rules would be quietly disagreeing about who
to treat on the datasets where cost is uniform. And on instances small enough to enumerate,
the greedy answer is checked against the true optimum rather than against itself, including
one instance where it is deliberately beaten, so the reported optimality gap is shown to be
doing its job rather than always printing zero.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from itx.policy.cost_aware import (
    Allocation,
    budget_for_share,
    cost_aware_policy,
    fractional_bound,
)
from itx.policy.rank_and_cut import rank_and_cut


def brute_force(scores, costs, budget) -> float:
    """The best gain any affordable subset achieves. Exponential, so tiny inputs only."""
    units = range(len(scores))
    best = 0.0
    for size in range(len(scores) + 1):
        for chosen in combinations(units, size):
            picked = list(chosen)
            if float(costs[picked].sum()) <= budget:
                best = max(best, float(scores[picked].sum()))
    return best


def random_instance(n: int, seed: int):
    """Positive costs, mixed-sign scores, and a budget covering about a third of the bill."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.05, 0.1, n)
    costs = rng.uniform(0.5, 5.0, n)
    return scores, costs, float(costs.sum()) * 0.3


class TestAgainstTheHeadcountRule:
    def test_equal_costs_reduce_to_rank_and_cut(self):
        # If these two disagreed, a dataset with uniform cost would have two different
        # answers to the same question depending on which module was asked.
        rng = np.random.default_rng(4)
        scores = rng.normal(1.0, 0.3, 500)  # all positive, so nothing is declined on merit
        costs = np.full(500, 2.0)
        for share in (0.1, 0.25, 0.5):
            allocation = cost_aware_policy(scores, costs, budget_for_share(costs, share))
            assert allocation.treat.tolist() == rank_and_cut(scores, share).tolist()

    def test_it_reorders_the_list_when_costs_differ(self):
        # The expensive unit has the larger effect and the smaller effect per dollar, so
        # the uplift ranking and the cost-aware ranking put different people first.
        scores = np.array([1.0, 0.5])
        costs = np.array([10.0, 1.0])
        assert rank_and_cut(scores, 0.5).tolist() == [True, False]
        allocation = cost_aware_policy(scores, costs, 1.0)
        assert allocation.treat.tolist() == [False, True]

    def test_a_budget_for_a_share_is_that_share_of_the_bill(self):
        costs = np.array([1.0, 2.0, 3.0, 4.0])
        assert budget_for_share(costs, 0.5) == pytest.approx(5.0)

    @pytest.mark.parametrize("share", [0.0, -0.2, 1.5])
    def test_an_impossible_share_is_rejected(self, share):
        with pytest.raises(ValueError, match="share must be"):
            budget_for_share(np.ones(4), share)


class TestOptimality:
    @pytest.mark.parametrize("seed", range(8))
    def test_it_is_exactly_optimal_when_every_unit_costs_the_same(self, seed):
        # With uniform costs the ratio order is the score order and the budget is a
        # headcount, so greedy is the optimum rather than an approximation of it.
        rng = np.random.default_rng(seed)
        scores = rng.normal(0.05, 0.1, 14)
        costs = np.full(14, 3.0)
        budget = 15.0
        allocation = cost_aware_policy(scores, costs, budget)
        assert allocation.predicted_gain == pytest.approx(
            brute_force(scores, costs, budget), rel=1e-9
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_the_true_optimum_sits_between_the_answer_and_the_bound(self, seed):
        # The property that makes the reported gap trustworthy, checked against the real
        # optimum rather than against the estimator's own arithmetic. Greedy can fall short
        # on instances this small, and the bound is never allowed to.
        scores, costs, budget = random_instance(14, seed)
        allocation = cost_aware_policy(scores, costs, budget)
        optimum = brute_force(scores, costs, budget)
        assert allocation.predicted_gain <= optimum + 1e-12
        assert optimum <= allocation.upper_bound + 1e-12

    def test_greedy_can_be_beaten_and_says_so(self):
        # One unit costs five eighths of the whole budget, which is the case the module
        # docstring warns about. Greedy takes it for its ratio and then cannot afford the
        # pair that would have been better.
        scores = np.array([10.0, 7.0, 7.0])
        costs = np.array([5.0, 4.0, 4.0])
        allocation = cost_aware_policy(scores, costs, 8.0)

        assert allocation.predicted_gain == pytest.approx(10.0)
        assert brute_force(scores, costs, 8.0) == pytest.approx(14.0)
        # The gap is measured against the fractional relaxation, 10 + (3/4) * 7, so it
        # over-states the true shortfall and never hides it.
        assert allocation.upper_bound == pytest.approx(15.25)
        assert allocation.optimality_gap > 0.3

    @pytest.mark.parametrize("seed", range(6))
    def test_the_bound_is_never_below_what_was_achieved(self, seed):
        scores, costs, budget = random_instance(200, seed)
        allocation = cost_aware_policy(scores, costs, budget)
        assert allocation.upper_bound >= allocation.predicted_gain - 1e-12
        assert 0.0 <= allocation.optimality_gap < 1.0

    def test_the_gap_closes_as_units_get_small_against_the_budget(self):
        # The practical claim the module makes: greedy is a real approximation on ten
        # units and is indistinguishable from optimal on ten thousand.
        scores, costs, budget = random_instance(10_000, 3)
        allocation = cost_aware_policy(scores, costs, budget)
        assert allocation.optimality_gap < 1e-4

    def test_a_budget_that_exactly_fits_a_prefix_is_provably_optimal(self):
        scores = np.array([3.0, 2.0, 1.0])
        costs = np.array([1.0, 1.0, 1.0])
        allocation = cost_aware_policy(scores, costs, 2.0)
        assert allocation.predicted_gain == pytest.approx(5.0)
        assert allocation.optimality_gap == pytest.approx(0.0)


class TestSpending:
    def test_it_never_exceeds_the_budget(self):
        for seed in range(5):
            scores, costs, budget = random_instance(400, seed)
            allocation = cost_aware_policy(scores, costs, budget)
            assert allocation.spend <= budget + 1e-12
            assert float(costs[allocation.treat].sum()) == pytest.approx(allocation.spend)

    def test_it_never_buys_predicted_harm(self):
        # Under a money budget there is nothing to trade off: declining to spend is free.
        scores = np.array([1.0, -1.0, -2.0])
        costs = np.array([1.0, 1.0, 1.0])
        allocation = cost_aware_policy(scores, costs, 100.0)
        assert allocation.treat.tolist() == [True, False, False]
        assert allocation.unspent == pytest.approx(99.0)

    def test_a_budget_of_nothing_treats_nobody(self):
        allocation = cost_aware_policy(np.ones(5), np.ones(5), 0.0)
        assert allocation.n_treated == 0
        assert allocation.predicted_gain == pytest.approx(0.0)
        assert allocation.optimality_gap == pytest.approx(0.0)

    def test_a_budget_larger_than_the_bill_treats_everyone_worth_treating(self):
        scores = np.array([1.0, 2.0, -1.0])
        costs = np.array([1.0, 1.0, 1.0])
        allocation = cost_aware_policy(scores, costs, 999.0)
        assert allocation.treat.tolist() == [True, True, False]

    def test_it_keeps_filling_past_a_unit_it_cannot_afford(self):
        # Stopping at the first unit that does not fit would leave the cheap, still
        # worthwhile unit behind for no reason.
        scores = np.array([10.0, 1.0])
        costs = np.array([100.0, 1.0])
        allocation = cost_aware_policy(scores, costs, 1.0)
        assert allocation.treat.tolist() == [False, True]


class TestAllocationReporting:
    def test_it_describes_itself(self):
        scores, costs, budget = random_instance(100, 1)
        line = cost_aware_policy(scores, costs, budget).describe()
        assert "of 100 units" in line
        assert "best possible" in line

    def test_a_bound_of_zero_reports_no_gap(self):
        allocation = Allocation(
            treat=np.zeros(3, dtype=bool),
            spend=0.0,
            budget=1.0,
            predicted_gain=0.0,
            upper_bound=0.0,
        )
        assert allocation.optimality_gap == pytest.approx(0.0)
        assert allocation.n_treated == 0


class TestRefusals:
    def test_a_cost_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="strictly positive"):
            cost_aware_policy(np.ones(3), np.array([1.0, 0.0, 1.0]), 2.0)

    def test_a_negative_cost_is_refused(self):
        with pytest.raises(ValueError, match="strictly positive"):
            cost_aware_policy(np.ones(3), np.array([1.0, -2.0, 1.0]), 2.0)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same length"):
            cost_aware_policy(np.ones(3), np.ones(4), 2.0)

    def test_a_negative_budget_is_refused(self):
        with pytest.raises(ValueError, match="must not be negative"):
            cost_aware_policy(np.ones(3), np.ones(3), -1.0)

    def test_an_empty_population_is_refused(self):
        with pytest.raises(ValueError, match="empty population"):
            cost_aware_policy(np.zeros(0), np.zeros(0), 1.0)

    def test_the_bound_refuses_the_same_inputs(self):
        with pytest.raises(ValueError, match="strictly positive"):
            fractional_bound(np.ones(3), np.zeros(3), 2.0)
