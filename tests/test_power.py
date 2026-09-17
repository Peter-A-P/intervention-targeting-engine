"""Tests for the sample-size calculation.

The arithmetic is standard and the analytic checks below pin it. The test that earns its
place is :class:`TestItRetrodictsTheSixDatasets`, which asks the calculation to predict,
from four summary numbers per dataset, which of this repository's six benchmarks could be
ranked on and which could not. It gets all six right, including both failures. That is the
only evidence offered that the formula describes this problem rather than merely being
algebra, and it is why the module ships.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from itx.metrics.power import (
    binary_outcome_sd,
    detectable_lift,
    requirement_table,
    units_to_detect_an_effect,
    units_to_rank,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"

#: Per dataset: test rows, outcome standard deviation, average effect and treated share, all
#: measured on the committed seed 11 split by `scratchpad` probe on 2026-09-17. Four numbers
#: is the whole input: no model, no features, no ranking.
MEASURED = {
    "hillstrom": (8_539, 0.335001, 0.045394, 0.5010),
    "ihdp": (150, 2.127207, 3.641987, 0.1867),
    "acic": (961, 5.110369, 3.606592, 0.1790),
    "lenta": (137_406, 0.310694, 0.007549, 0.7509),
    "criteo": (279_592, 0.211617, 0.010343, 0.8500),
    "ieee-fraud": (118_108, 47.684558, 2.303739, 0.4991),
}
BUDGET = 0.20


class TestTheBinaryStandardDeviation:
    def test_it_is_the_bernoulli_one(self):
        assert binary_outcome_sd(0.5) == pytest.approx(0.5)
        assert binary_outcome_sd(0.1) == pytest.approx(math.sqrt(0.09))

    def test_it_is_largest_at_a_half(self):
        assert binary_outcome_sd(0.5) > binary_outcome_sd(0.3) > binary_outcome_sd(0.05)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_a_rate_outside_the_unit_interval_is_refused(self, bad):
        with pytest.raises(ValueError, match="base rate"):
            binary_outcome_sd(bad)


class TestDetectingAnEffect:
    def test_it_matches_the_textbook_balanced_formula(self):
        """The textbook balanced formula.

        n = 4 sigma^2 (z_a + z_b)^2 / d^2, at a 50/50 split, alpha 0.05 and power 0.8.
        """
        sd, effect = 1.0, 0.1
        expected = 4.0 * (1.959963985 + 0.841621234) ** 2 * sd**2 / effect**2
        got = units_to_detect_an_effect(outcome_sd=sd, average_effect=effect)
        assert got == pytest.approx(math.ceil(expected))

    def test_halving_the_effect_quadruples_the_sample(self):
        big = units_to_detect_an_effect(outcome_sd=1.0, average_effect=0.2)
        small = units_to_detect_an_effect(outcome_sd=1.0, average_effect=0.1)
        assert small / big == pytest.approx(4.0, rel=0.001)

    def test_an_unbalanced_split_costs_more_than_a_balanced_one(self):
        balanced = units_to_detect_an_effect(
            outcome_sd=1.0, average_effect=0.1, treated_share=0.5
        )
        lopsided = units_to_detect_an_effect(
            outcome_sd=1.0, average_effect=0.1, treated_share=0.1
        )
        assert lopsided > balanced

    def test_the_sign_of_the_effect_does_not_matter(self):
        up = units_to_detect_an_effect(outcome_sd=1.0, average_effect=0.1)
        down = units_to_detect_an_effect(outcome_sd=1.0, average_effect=-0.1)
        assert up == down

    def test_an_effect_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="non-zero"):
            units_to_detect_an_effect(outcome_sd=1.0, average_effect=0.0)


class TestRanking:
    def test_ranking_costs_more_than_detecting_an_effect(self):
        """The claim the module exists to make, at the settings it makes it for."""
        floor = units_to_detect_an_effect(outcome_sd=1.0, average_effect=0.1)
        ranking = units_to_rank(outcome_sd=1.0, average_effect=0.1, lift_ratio=2.0, budget=0.2)
        assert ranking / floor == pytest.approx(5.0, rel=0.01)

    def test_the_ratio_is_one_over_budget_times_lift_gap_squared(self):
        floor = units_to_detect_an_effect(outcome_sd=1.0, average_effect=0.1)
        ranking = units_to_rank(outcome_sd=1.0, average_effect=0.1, lift_ratio=1.5, budget=0.1)
        assert ranking / floor == pytest.approx(1.0 / (0.1 * 0.5**2), rel=0.01)

    def test_a_tighter_budget_needs_more_data(self):
        wide = units_to_rank(outcome_sd=1.0, average_effect=0.1, lift_ratio=2.0, budget=0.5)
        tight = units_to_rank(outcome_sd=1.0, average_effect=0.1, lift_ratio=2.0, budget=0.05)
        assert tight > wide

    def test_weaker_heterogeneity_needs_more_data(self):
        strong = units_to_rank(outcome_sd=1.0, average_effect=0.1, lift_ratio=3.0, budget=0.2)
        weak = units_to_rank(outcome_sd=1.0, average_effect=0.1, lift_ratio=1.1, budget=0.2)
        assert weak > strong

    def test_a_prefix_that_responds_like_the_population_is_refused(self):
        """Random targeting is not distinguishable from itself at any sample size."""
        with pytest.raises(ValueError, match="lift_ratio must be above 1"):
            units_to_rank(outcome_sd=1.0, average_effect=0.1, lift_ratio=1.0, budget=0.2)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_a_budget_outside_the_unit_interval_is_refused(self, bad):
        with pytest.raises(ValueError, match="budget"):
            units_to_rank(outcome_sd=1.0, average_effect=0.1, lift_ratio=2.0, budget=bad)


class TestDetectableLift:
    def test_it_inverts_units_to_rank(self):
        units = units_to_rank(outcome_sd=1.0, average_effect=0.1, lift_ratio=2.0, budget=0.2)
        back = detectable_lift(n_units=units, outcome_sd=1.0, average_effect=0.1, budget=0.2)
        assert back == pytest.approx(2.0, rel=0.001)

    def test_more_data_detects_weaker_heterogeneity(self):
        small = detectable_lift(n_units=1_000, outcome_sd=1.0, average_effect=0.1, budget=0.2)
        large = detectable_lift(n_units=100_000, outcome_sd=1.0, average_effect=0.1, budget=0.2)
        assert large < small
        assert large > 1.0


class TestTheReport:
    def test_every_row_is_above_the_floor_and_they_fall_with_heterogeneity(self):
        report = requirement_table(
            outcome_sd=binary_outcome_sd(0.1), average_effect=0.01, budget=0.2
        )
        assert all(row.units > report.to_detect_an_effect for row in report.rows)
        assert all(row.multiple_of_effect_detection > 1.0 for row in report.rows)
        units = [row.units for row in report.rows]
        assert units == sorted(units, reverse=True)

    def test_the_summary_says_the_numbers_are_floors(self):
        report = requirement_table(outcome_sd=0.3, average_effect=0.01, budget=0.2)
        assert "floor" in report.summary()

    def test_the_markdown_has_a_row_per_lift_ratio(self):
        report = requirement_table(
            outcome_sd=0.3, average_effect=0.01, budget=0.2, lift_ratios=(1.5, 2.0)
        )
        body = [
            line
            for line in report.to_markdown().splitlines()
            if line.startswith("| 1") or line.startswith("| 2")
        ]
        assert len(body) == 2


class TestItRetrodictsTheSixDatasets:
    """Four summary numbers per dataset, and it gets every verdict right.

    For each benchmark: the lift the study could have detected, against the lift the best
    estimator actually found. The calculation sees no features, no model and no ranking,
    only the test size, the outcome spread, the average effect and the treated share. If it
    says a dataset could be ranked on, the measured uplift in the top 20% should be clear of
    the average effect; if it says it could not, it should not be.
    """

    def best_uplift_at_20(self, dataset: str) -> tuple[float, float]:
        rows = json.loads((RESULTS / f"{dataset}.json").read_text(encoding="utf-8"))
        best: tuple[float, float] | None = None
        for row in rows:
            if row["estimator"] in ("outcome-ranking", "random-200") or row["seed"] != 11:
                continue
            metric = row["metrics"]["uplift@20%"]
            if best is None or metric["value"] > best[0]:
                best = (metric["value"], metric["low"])
        assert best is not None, dataset
        return best

    @pytest.mark.parametrize("dataset", sorted(MEASURED))
    def test_the_verdict_matches_what_was_measured(self, dataset):
        n_units, outcome_sd, average_effect, treated_share = MEASURED[dataset]
        needed = detectable_lift(
            n_units=n_units,
            outcome_sd=outcome_sd,
            average_effect=average_effect,
            budget=BUDGET,
            treated_share=treated_share,
        )
        value, low = self.best_uplift_at_20(dataset)
        predicted_rankable = (value / average_effect) >= needed
        # "Separates" means the top 20% is clear of what treating everybody would buy, which
        # is the interval's lower end sitting above the average effect.
        measured_rankable = low > average_effect
        assert predicted_rankable == measured_rankable, (
            f"{dataset}: needed {needed:.2f}x, observed {value / average_effect:.2f}x, "
            f"predicted rankable={predicted_rankable}, measured={measured_rankable}"
        )

    def test_it_is_right_about_both_of_the_failures(self):
        """The two that cannot be ranked on are the ones the calculation has to catch."""
        for dataset in ("lenta", "ihdp"):
            n_units, outcome_sd, average_effect, treated_share = MEASURED[dataset]
            needed = detectable_lift(
                n_units=n_units,
                outcome_sd=outcome_sd,
                average_effect=average_effect,
                budget=BUDGET,
                treated_share=treated_share,
            )
            value, _ = self.best_uplift_at_20(dataset)
            assert value / average_effect < needed, dataset

    def test_lenta_needs_more_lift_than_hillstrom_despite_being_sixteen_times_larger(self):
        """Size alone does not decide it, which is the point of the whole module."""
        needed = {}
        for dataset in ("lenta", "hillstrom"):
            n_units, outcome_sd, average_effect, treated_share = MEASURED[dataset]
            needed[dataset] = detectable_lift(
                n_units=n_units,
                outcome_sd=outcome_sd,
                average_effect=average_effect,
                budget=BUDGET,
                treated_share=treated_share,
            )
        assert MEASURED["lenta"][0] > 16 * MEASURED["hillstrom"][0]
        assert needed["lenta"] > needed["hillstrom"]
