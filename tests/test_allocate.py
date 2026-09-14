"""Spending analyst hours four ways, on data where the answer is written down.

What has to hold: no queue spends more than it was given, the knapsack and the cost-blind
queue agree exactly when every review costs the same (or the two rules would quietly disagree
on the five public datasets where cost is uniform), the knapsack pulls ahead when costs vary
and the model knows it, the true value is the sum of the written effects over the chosen
units, and the doubly robust estimate lands near that truth on a randomised design. The
oracle ranking has to beat everything, because if it did not the scoring would be wrong
rather than the ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from itx.bench.allocate import MINUTES_PER_HOUR, Queue, compare_queues, to_markdown
from itx.bench.seeds import TIE_SEED
from itx.estimators.propensity import DEFAULT_CLIP, PropensityFit
from itx.policy.policy_value import Nuisances
from itx.types import FloatArray, IntArray

Population = tuple[FloatArray, IntArray, FloatArray, FloatArray, Nuisances]


def fraud_shaped(n: int = 6_000, seed: int = 0) -> Population:
    """A minority with a large positive effect, everyone else with a small negative one."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < 0.15
    amount = rng.exponential(80.0, n) + 5.0
    truth = np.where(is_fraud, 0.6 * amount, -0.02 * amount)
    treatment = (rng.random(n) < 0.5).astype(np.int64)
    baseline = np.where(is_fraud, -amount, 0.03 * amount)
    outcome = baseline + treatment * truth + rng.normal(0.0, 1.0, n)
    costs = rng.uniform(9.0, 18.0, n)
    # Nuisances assembled from the truth, so the test is about the allocation arithmetic
    # rather than about a LightGBM fit.
    propensity = np.full(n, 0.5)
    nuisances = Nuisances(
        propensity=propensity,
        mu0=baseline,
        mu1=baseline + truth,
        propensity_fit=PropensityFit(
            values=propensity,
            known=True,
            clip=DEFAULT_CLIP,
            n_clipped=0,
            raw_min=0.5,
            raw_max=0.5,
        ),
    )
    return outcome, treatment, truth, costs, nuisances


@pytest.fixture(scope="module")
def population() -> Population:
    return fraud_shaped()


def run(
    population: Population,
    scores: dict[str, FloatArray],
    hours: float,
    knapsack_for: tuple[str, ...] = (),
) -> list[Queue]:
    outcome, treatment, truth, costs, nuisances = population
    return compare_queues(
        scores,
        costs,
        hours,
        outcome=outcome,
        treatment=treatment,
        nuisances=nuisances,
        truth=truth,
        knapsack_for=knapsack_for,
        seed=TIE_SEED,
    )


class TestTheBudgetIsRespected:
    def test_no_queue_spends_more_than_it_was_given(self, population):
        _, _, truth, _, _ = population
        rng = np.random.default_rng(1)
        queues = run(
            population,
            {"oracle": truth, "random": rng.normal(size=truth.size)},
            hours=100.0,
            knapsack_for=("oracle",),
        )
        for queue in queues:
            assert queue.minutes <= 100.0 * MINUTES_PER_HOUR + 1e-9

    def test_the_cost_blind_queue_never_buys_predicted_harm(self, population):
        # Most of a fraud-shaped population has a negative predicted effect. A queue that
        # kept spending into it because budget was left would be paying to do damage.
        _, _, truth, costs, _ = population
        (queue,) = run(population, {"oracle": truth}, hours=1e6)
        assert queue.n_reviewed == int((truth > 0).sum())
        assert queue.true_value == pytest.approx(truth[truth > 0].sum())
        assert queue.minutes == pytest.approx(costs[truth > 0].sum())


class TestKnapsackAgainstRankAndCut:
    def test_they_agree_exactly_when_every_review_costs_the_same(self, population):
        # If these differed, the five public datasets (uniform cost) would have two
        # different answers to the same question depending on which rule was asked.
        outcome, treatment, truth, _, nuisances = population
        flat = np.full(truth.size, 12.0)
        queues = compare_queues(
            {"knapsack": truth, "rank-and-cut": truth},
            flat,
            200.0,
            outcome=outcome,
            treatment=treatment,
            nuisances=nuisances,
            truth=truth,
            knapsack_for=("knapsack",),
            seed=TIE_SEED,
        )
        assert queues[0].n_reviewed == queues[1].n_reviewed
        assert queues[0].true_value == pytest.approx(queues[1].true_value)

    def test_the_knapsack_wins_when_costs_vary_and_the_score_is_right(self, population):
        _, _, truth, _, _ = population
        queues = run(
            population,
            {"knapsack": truth, "rank-and-cut": truth},
            hours=150.0,
            knapsack_for=("knapsack",),
        )
        knapsack, plain = queues
        assert knapsack.true_value >= plain.true_value
        assert knapsack.dollars_per_hour >= plain.dollars_per_hour


class TestWhatTheNumbersMean:
    def test_the_true_value_is_the_sum_over_the_chosen_units(self, population):
        _, _, truth, costs, _ = population
        (queue,) = run(population, {"oracle": truth}, hours=80.0, knapsack_for=("oracle",))
        # Reconstruct which units a knapsack at that budget picks and sum their truth.
        from itx.policy.cost_aware import cost_aware_policy

        treat = cost_aware_policy(truth, costs, 80.0 * MINUTES_PER_HOUR, seed=TIE_SEED).treat
        assert queue.true_value == pytest.approx(truth[treat].sum())
        assert queue.n_reviewed == int(treat.sum())

    def test_the_doubly_robust_estimate_lands_near_the_truth_on_a_randomised_design(
        self, population
    ):
        # The DR estimate is what this package would report on real data. With correct
        # nuisances and a fair coin it has to sit near the written truth, or the case's
        # side-by-side comparison would be comparing an estimate against a number it
        # cannot approach even in principle.
        _, _, truth, _, _ = population
        (queue,) = run(population, {"oracle": truth}, hours=150.0, knapsack_for=("oracle",))
        assert queue.dr_value == pytest.approx(queue.true_value, rel=0.15)

    def test_the_oracle_beats_a_noisy_ranking_and_random(self, population):
        _, _, truth, _, _ = population
        rng = np.random.default_rng(2)
        queues = run(
            population,
            {
                "oracle": truth,
                "noisy": truth + rng.normal(0.0, truth.std(), truth.size),
                "random": rng.normal(size=truth.size),
            },
            hours=150.0,
            knapsack_for=("oracle", "noisy"),
        )
        oracle, noisy, random = queues
        assert oracle.true_value > noisy.true_value > random.true_value

    def test_dollars_per_hour_is_undefined_for_an_empty_queue(self):
        empty = Queue(name="none", n_reviewed=0, minutes=0.0, dr_value=0.0, true_value=0.0)
        assert np.isnan(empty.dollars_per_hour)


class TestTheTable:
    def test_it_renders_every_queue_in_order(self, population):
        _, _, truth, _, _ = population
        rng = np.random.default_rng(3)
        queues = run(
            population, {"first": truth, "second": rng.normal(size=truth.size)}, hours=50.0
        )
        table = to_markdown(queues, 50.0)
        lines = table.splitlines()
        assert "50 analyst hours" in lines[0]
        assert lines[2].startswith("| `first` |")
        assert lines[3].startswith("| `second` |")
