"""Rosenbaum bounds, against scipy, against a binomial, and against brute-force matching.

Four things are worth protecting here, and each is checked against something outside this
module rather than against itself.

At Gamma of 1 the bound is just the one-sided signed-rank test, so it is checked against
scipy's, which would catch a wrong variance or a wrong tail. On a binary outcome the whole
expression is supposed to collapse to McNemar's test, so it is checked against a binomial
computed directly from the discordant counts; the module's claim that one implementation
serves both outcome types rests on that reduction and the datasets this gets reported on are
all binary. The matcher's fast free-list is checked against a plain quadratic
nearest-neighbour matcher on small inputs, because a union-find with a compression bug
returns plausible matches rather than obvious nonsense. And the breaking point is checked to
bracket the crossing, so a rounding error cannot report a Gamma the bound is not significant
at.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from itx.sensitivity.rosenbaum import (
    DEFAULT_ALPHA,
    GAMMA_STEP,
    MAX_GAMMA,
    MIN_PAIRS,
    RosenbaumBound,
    bound_p_value,
    breaking_point,
    match_pairs,
    matching_key,
    signed_rank,
    targeting_rosenbaum,
)


def brute_force_match(propensity, treatment, order, caliper: float) -> list[tuple[int, int]]:
    """Greedy nearest neighbour without replacement, written the slow obvious way."""
    treated = np.flatnonzero(treatment == 1)
    control = list(np.flatnonzero(treatment == 0))
    pairs = []
    for position in order:
        unit = int(treated[position])
        if not control:
            break
        gaps = [abs(propensity[c] - propensity[unit]) for c in control]
        best = int(np.argmin(gaps))
        if gaps[best] > caliper:
            continue
        pairs.append((unit, int(control.pop(best))))
    return sorted(pairs)


def binary_pairs(n_positive: int, n_negative: int, n_tied: int):
    """Discordant pairs favouring treatment, discordant against, and uninformative ones."""
    return np.concatenate([np.ones(n_positive), -np.ones(n_negative), np.zeros(n_tied)]).astype(
        np.float64
    )


class TestTheStatisticItself:
    def test_at_gamma_one_it_is_the_signed_rank_test(self):
        # Distinct absolute differences, so there are no ties to argue about and scipy's
        # normal approximation is computing exactly the same quantity.
        rng = np.random.default_rng(0)
        difference = rng.normal(0.4, 1.0, 200)
        assert np.unique(np.abs(difference)).size == difference.size

        mine = bound_p_value(difference, 1.0)
        theirs = stats.wilcoxon(
            difference, alternative="greater", method="approx", correction=False
        ).pvalue
        assert mine == pytest.approx(float(theirs), rel=1e-10)

    def test_zeros_are_dropped_and_do_not_move_the_answer(self):
        rng = np.random.default_rng(1)
        difference = rng.normal(0.3, 1.0, 150)
        padded = np.concatenate([difference, np.zeros(80)])
        assert bound_p_value(padded, 1.0) == pytest.approx(bound_p_value(difference, 1.0))

    def test_ties_get_average_ranks(self):
        _, ranks = signed_rank(np.array([1.0, -1.0, 1.0, 2.0]))
        assert sorted(ranks.tolist()) == [2.0, 2.0, 2.0, 4.0]

    def test_the_statistic_counts_only_the_positive_side(self):
        statistic, ranks = signed_rank(np.array([3.0, -1.0, 2.0]))
        # Ranks of |d| are 1 for the -1, 2 for the 2, 3 for the 3. Positives are 3 and 2.
        assert ranks.tolist() == [3.0, 1.0, 2.0]
        assert statistic == 5.0

    def test_too_few_pairs_and_too_few_discordant_are_reported_differently(self):
        # A continuous outcome has no exact ties, so every pair is "discordant" and the
        # real problem is the pair count. Telling a reader that concordant pairs carry no
        # information, when there are none, points them at the wrong thing.
        # The two members of each pair share a propensity, so the caliper never bites and
        # the matcher forms exactly five pairs.
        pairs = 5
        propensity = np.repeat(np.linspace(0.2, 0.8, pairs), 2)
        treatment = np.array([1, 0] * pairs, dtype=np.int64)
        outcome = np.array([1.5, 0.0] * pairs)

        thin = targeting_rosenbaum(
            propensity,
            treatment,
            outcome,
            np.ones(2 * pairs, dtype=bool),
            budget=1.0,
            seed=0,
        )
        assert thin.n_pairs == pairs
        assert thin.n_discordant == pairs
        assert "only 5 pairs could be matched" in thin.summary()
        assert "Concordant pairs" not in thin.summary()

        # The binary case, where the pair count is ample and the informative count is not.
        blunt = RosenbaumBound(
            gamma=math.nan,
            p_value=math.nan,
            n_pairs=900,
            n_discordant=4,
            matched_share=0.9,
            budget=0.2,
            alpha=DEFAULT_ALPHA,
        )
        assert "Concordant pairs carry no information" in blunt.summary()

    def test_a_sample_with_too_few_informative_pairs_says_nothing(self):
        thin = binary_pairs(MIN_PAIRS - 1, 0, 500)
        assert math.isnan(bound_p_value(thin, 1.0))
        assert math.isnan(breaking_point(thin)[0])

    def test_gamma_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            bound_p_value(binary_pairs(40, 10, 0), 0.5)


class TestTheBinaryReduction:
    @pytest.mark.parametrize(
        ("positive", "negative"), [(40, 10), (60, 40), (100, 30), (25, 24)]
    )
    @pytest.mark.parametrize("gamma", [1.0, 1.5, 2.0, 4.0])
    def test_it_becomes_mcnemar(self, positive, negative, gamma):
        # With every difference equal to plus or minus one, every rank is the same average
        # rank and the bound has to reduce to the normal approximation of a binomial on the
        # discordant pairs. This is the claim that lets one implementation serve both a
        # continuous and a binary outcome, so it is asserted rather than reasoned about.
        difference = binary_pairs(positive, negative, 300)
        discordant = positive + negative
        probability = gamma / (1.0 + gamma)
        deviate = (positive - discordant * probability) / math.sqrt(
            discordant * probability * (1.0 - probability)
        )
        expected = float(stats.norm.sf(deviate))
        assert bound_p_value(difference, gamma) == pytest.approx(expected, rel=1e-10)

    def test_concordant_pairs_carry_no_information(self):
        # Padding a result with pairs where both members did the same thing must not make it
        # look stronger or weaker. If it did, the reported Gamma would depend on the
        # outcome's base rate rather than on the effect.
        lean = binary_pairs(40, 10, 0)
        padded = binary_pairs(40, 10, 5000)
        assert bound_p_value(padded, 2.0) == pytest.approx(bound_p_value(lean, 2.0))


class TestTheBound:
    def test_the_p_value_rises_with_gamma(self):
        difference = binary_pairs(80, 20, 200)
        values = [bound_p_value(difference, g) for g in (1.0, 1.2, 1.5, 2.0, 3.0, 6.0)]
        assert values == sorted(values)

    def test_it_brackets_the_crossing(self):
        difference = binary_pairs(80, 20, 200)
        gamma, censored = breaking_point(difference)
        assert not censored
        assert bound_p_value(difference, gamma) <= DEFAULT_ALPHA
        assert bound_p_value(difference, gamma + GAMMA_STEP) > DEFAULT_ALPHA

    def test_a_result_that_was_never_significant_survives_no_bias_at_all(self):
        difference = binary_pairs(50, 50, 200)  # dead even
        gamma, censored = breaking_point(difference)
        assert gamma == 1.0
        assert not censored
        assert bound_p_value(difference, 1.0) > DEFAULT_ALPHA

    def test_an_overwhelming_result_is_reported_as_censored(self):
        difference = binary_pairs(5000, 100, 0)
        gamma, censored = breaking_point(difference)
        assert censored
        assert gamma == MAX_GAMMA

    def test_an_effect_in_the_other_direction_survives_nothing(self):
        # The test is one-sided for a positive difference, deliberately and not adaptively:
        # picking the direction after seeing the data would inflate the error rate. A result
        # pointing the other way has to come back as Gamma 1.0 rather than as a large number
        # for a finding the test was not asking about.
        against = binary_pairs(20, 80, 0)
        assert bound_p_value(against, 1.0) > 0.9
        gamma, censored = breaking_point(against)
        assert gamma == 1.0
        assert not censored

    def test_a_stronger_effect_survives_more_bias(self):
        weak, _ = breaking_point(binary_pairs(60, 40, 0))
        strong, _ = breaking_point(binary_pairs(80, 20, 0))
        assert strong > weak


class TestTheMatcher:
    def test_it_agrees_with_the_slow_obvious_matcher(self):
        # The free list is a union-find with path compression, and a bug there returns
        # plausible-looking pairs rather than an error, so it is checked against a
        # quadratic matcher that cannot be clever enough to be wrong.
        rng = np.random.default_rng(7)
        for seed in range(5):
            propensity = rng.uniform(0.05, 0.95, 120)
            treatment = (rng.random(120) < 0.5).astype(np.int64)
            outcome = rng.normal(size=120)

            pairs = match_pairs(propensity, treatment, outcome, caliper_sds=math.inf, seed=seed)
            order = np.random.default_rng(seed).permutation(int((treatment == 1).sum()))
            expected = brute_force_match(propensity, treatment, order, math.inf)
            assert (
                list(zip(pairs.treated.tolist(), pairs.control.tolist(), strict=True))
                == expected
            )

    def test_it_agrees_with_the_slow_matcher_inside_a_caliper(self):
        rng = np.random.default_rng(9)
        propensity = rng.uniform(0.0, 1.0, 200)
        treatment = (rng.random(200) < 0.5).astype(np.int64)
        outcome = rng.normal(size=200)

        caliper_sds = 0.05
        pairs = match_pairs(propensity, treatment, outcome, caliper_sds=caliper_sds, seed=3)
        caliper = caliper_sds * float(np.std(propensity))
        order = np.random.default_rng(3).permutation(int((treatment == 1).sum()))
        expected = brute_force_match(propensity, treatment, order, caliper)
        assert (
            list(zip(pairs.treated.tolist(), pairs.control.tolist(), strict=True)) == expected
        )
        assert pairs.n_pairs < int((treatment == 1).sum())  # the caliper really bit

    def test_no_control_is_used_twice(self):
        rng = np.random.default_rng(4)
        propensity = rng.uniform(0.1, 0.9, 2000)
        treatment = (rng.random(2000) < 0.7).astype(np.int64)  # treated outnumber controls
        pairs = match_pairs(
            propensity, treatment, rng.normal(size=2000), caliper_sds=math.inf, seed=0
        )
        assert np.unique(pairs.control).size == pairs.n_pairs
        assert pairs.n_pairs == int((treatment == 0).sum())  # every control got used

    def test_it_reports_who_it_could_not_match(self):
        rng = np.random.default_rng(5)
        propensity = np.concatenate([np.full(100, 0.9), np.full(100, 0.1)])
        treatment = np.array([1] * 100 + [0] * 100, dtype=np.int64)
        pairs = match_pairs(
            propensity, treatment, rng.normal(size=200), caliper_sds=0.01, seed=0
        )
        # Every treated unit sits 0.8 away from every control, far outside the caliper.
        assert pairs.n_pairs == 0
        assert pairs.n_eligible_treated == 100
        assert pairs.matched_share == 0.0

    def test_only_eligible_units_are_matched(self):
        rng = np.random.default_rng(6)
        propensity = rng.uniform(0.2, 0.8, 400)
        treatment = (rng.random(400) < 0.5).astype(np.int64)
        eligible = np.zeros(400, dtype=bool)
        eligible[:100] = True

        pairs = match_pairs(
            propensity,
            treatment,
            rng.normal(size=400),
            eligible=eligible,
            caliper_sds=math.inf,
            seed=0,
        )
        assert pairs.treated.max() < 100
        assert pairs.control.max() < 100

    def test_the_difference_is_treated_minus_control(self):
        propensity = np.array([0.5, 0.5])
        treatment = np.array([1, 0], dtype=np.int64)
        outcome = np.array([3.0, 1.0])
        pairs = match_pairs(propensity, treatment, outcome, caliper_sds=math.inf, seed=0)
        assert pairs.difference.tolist() == [2.0]

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="must align"):
            match_pairs(np.zeros(5), np.zeros(4, dtype=np.int64), np.zeros(5))


class TestTheWholeThingOnKnownData:
    def make(self, n: int, effect: float, seed: int):
        """Randomised assignment, so the true Gamma is 1 and any effect is real."""
        rng = np.random.default_rng(seed)
        propensity = np.full(n, 0.5)
        treatment = (rng.random(n) < 0.5).astype(np.int64)
        base = 0.2
        rate = base + effect * treatment
        outcome = (rng.random(n) < rate).astype(np.float64)
        return propensity, treatment, outcome

    def test_a_real_effect_survives_some_hidden_bias(self):
        propensity, treatment, outcome = self.make(20000, effect=0.10, seed=1)
        result = targeting_rosenbaum(
            propensity,
            treatment,
            outcome,
            np.ones(20000, dtype=bool),
            budget=1.0,
            seed=0,
        )
        assert result.p_value < 1e-6
        assert result.survives_any_bias
        assert result.gamma > 1.2
        assert "Gamma" in result.summary()

    def test_no_effect_survives_none(self):
        propensity, treatment, outcome = self.make(20000, effect=0.0, seed=2)
        result = targeting_rosenbaum(
            propensity,
            treatment,
            outcome,
            np.ones(20000, dtype=bool),
            budget=1.0,
            seed=0,
        )
        assert not result.survives_any_bias
        assert result.gamma == 1.0
        assert "nothing for a sensitivity analysis to protect" in result.summary()

    def test_a_bigger_effect_buys_a_bigger_gamma(self):
        gammas = []
        for effect in (0.03, 0.08, 0.15):
            propensity, treatment, outcome = self.make(20000, effect=effect, seed=3)
            gammas.append(
                targeting_rosenbaum(
                    propensity,
                    treatment,
                    outcome,
                    np.ones(20000, dtype=bool),
                    budget=1.0,
                    seed=0,
                ).gamma
            )
        assert gammas == sorted(gammas)

    def test_the_targeted_group_is_the_one_measured(self):
        propensity, treatment, outcome = self.make(20000, effect=0.10, seed=4)
        targeted = np.zeros(20000, dtype=bool)
        targeted[:4000] = True
        result = targeting_rosenbaum(
            propensity, treatment, outcome, targeted, budget=0.2, seed=0
        )
        assert result.n_pairs <= 2000  # at most one pair per treated unit in the group
        assert result.budget == 0.2


class TestWhichKeyItMatchesOn:
    def test_an_observational_propensity_is_used_directly(self):
        propensity = np.linspace(0.1, 0.9, 100)
        key, named = matching_key(propensity, np.zeros(100))
        assert named == "propensity"
        assert key.tolist() == propensity.tolist()

    def test_a_constant_propensity_falls_back_to_the_prognostic_score(self):
        # The failure this exists to stop. Hillstrom's design propensity is exactly 0.5 for
        # every row, so every unit is equidistant from every other and propensity matching
        # degenerates into arbitrary pairing while still calling itself matching.
        prognostic = np.linspace(0.0, 1.0, 100)
        key, named = matching_key(np.full(100, 0.5), prognostic)
        assert named == "prognostic score"
        assert key.tolist() == prognostic.tolist()

    def test_with_nothing_that_varies_it_says_so(self):
        _, named = matching_key(np.full(50, 0.5), np.full(50, 2.0))
        assert named == "nothing"

    def test_with_no_prognostic_score_offered_it_says_so(self):
        _, named = matching_key(np.full(50, 0.5))
        assert named == "nothing"

    def test_the_key_reaches_the_result(self):
        rng = np.random.default_rng(2)
        n = 4000
        treatment = (rng.random(n) < 0.5).astype(np.int64)
        outcome = (rng.random(n) < 0.2 + 0.1 * treatment).astype(np.float64)
        result = targeting_rosenbaum(
            np.full(n, 0.5),
            treatment,
            outcome,
            np.ones(n, dtype=bool),
            budget=1.0,
            prognostic=rng.normal(size=n),
            seed=0,
        )
        assert result.matched_on == "prognostic score"
        assert "matched on the prognostic score" in result.summary()
