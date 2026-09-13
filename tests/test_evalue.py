"""The E-value, against its closed form and against data where the answer is known.

Three properties worth protecting. The formula is checked against values published with
VanderWeele and Ding's paper rather than against itself, so a sign error in the square root
cannot pass. The symmetry about the null is checked directly, because a protective effect
and a harmful one of equal magnitude must cost the same to explain away and it is easy to
write an inversion that does not. And the limit E-value is checked to collapse to exactly
1.0 when the interval covers the null, which is the case a reader is most likely to
misread and the case where quoting the point estimate alone would overstate the result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from itx.sensitivity.evalue import (
    MIN_ARM,
    SMD_TO_LOG_RR,
    e_value_of,
    e_value_of_limit,
    risk_ratio_of,
    targeting_e_value,
)


def two_arm_sample(n: int, treated_rate: float, control_rate: float, seed: int = 0):
    """Half treated, half control, with outcome rates set exactly rather than drawn."""
    half = n // 2
    treatment = np.array([1] * half + [0] * half, dtype=np.int64)
    outcome = np.concatenate(
        [
            _exact_rate(half, treated_rate),
            _exact_rate(half, control_rate),
        ]
    )
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n)
    return outcome, treatment, scores


def _exact_rate(n: int, rate: float):
    """``n`` outcomes of which exactly ``round(n * rate)`` are ones."""
    ones = round(n * rate)
    return np.concatenate([np.ones(ones), np.zeros(n - ones)])


class TestTheClosedForm:
    @pytest.mark.parametrize(
        ("risk_ratio", "expected"),
        [
            (1.0, 1.0),  # the null needs no confounding at all
            (2.0, 3.4142135623730951),  # 2 + sqrt(2)
            (3.0, 5.449489742783178),  # 3 + sqrt(6)
            (1.5, 2.366025403784439),  # 1.5 + sqrt(0.75)
        ],
    )
    def test_it_matches_the_published_formula(self, risk_ratio, expected):
        assert e_value_of(risk_ratio) == pytest.approx(expected, rel=1e-12)

    def test_it_is_symmetric_about_the_null(self):
        # A halving and a doubling are the same claim seen from the two arms, so they must
        # cost the same to explain away. An inversion written the wrong way round would
        # return 1.0 for every protective effect and nobody would notice.
        for ratio in (1.25, 1.5, 2.0, 4.0, 10.0):
            assert e_value_of(ratio) == pytest.approx(e_value_of(1.0 / ratio), rel=1e-12)

    def test_it_grows_with_the_effect(self):
        values = [e_value_of(r) for r in (1.0, 1.2, 1.5, 2.0, 5.0)]
        assert values == sorted(values)
        assert values[0] == 1.0

    def test_it_always_exceeds_the_risk_ratio_away_from_the_null(self):
        # The E-value is on the confounder-association scale, where a confounder has to be
        # associated with both treatment and outcome, so it is necessarily the larger
        # number. A reader who compares it against the risk ratio directly is being told
        # something true by that ordering.
        for ratio in (1.1, 1.5, 3.0, 8.0):
            assert e_value_of(ratio) > ratio

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_it_refuses_a_ratio_that_is_not_one(self, bad):
        assert math.isnan(e_value_of(bad))


class TestTheIntervalLimit:
    def test_an_interval_covering_the_null_costs_nothing_to_explain_away(self):
        # The case worth being loud about. Quoting only the point estimate's E-value for a
        # result whose interval covers 1 tells the reader something was established.
        assert e_value_of_limit(1.8, 0.9, 3.2) == 1.0

    def test_it_reads_the_bound_nearest_the_null(self):
        assert e_value_of_limit(2.0, 1.4, 3.0) == pytest.approx(e_value_of(1.4))
        assert e_value_of_limit(0.5, 0.3, 0.7) == pytest.approx(e_value_of(0.7))

    def test_it_never_exceeds_the_point_estimate(self):
        for ratio, low, high in [(2.0, 1.4, 3.0), (0.5, 0.3, 0.7), (4.0, 3.9, 4.1)]:
            assert e_value_of_limit(ratio, low, high) <= e_value_of(ratio) + 1e-12


class TestTheRiskRatioInTheTargetedGroup:
    def test_it_divides_the_rates_it_says_it_does(self):
        outcome, treatment, _ = two_arm_sample(400, treated_rate=0.3, control_rate=0.2)
        targeted = np.ones(400, dtype=bool)
        assert risk_ratio_of(outcome, treatment, targeted, binary=True) == pytest.approx(1.5)

    def test_it_reads_only_the_targeted_units(self):
        # Half the population responds and half does not. Targeting the half that does has
        # to report that half's ratio of 2.0, not the population's diluted 1.5. The rates
        # are laid out by hand rather than drawn, so the expected numbers are exact.
        responders = np.concatenate(
            [_exact_rate(100, 0.4), _exact_rate(100, 0.2)]  # treated, then control
        )
        rest = np.concatenate([_exact_rate(100, 0.2), _exact_rate(100, 0.2)])
        outcome = np.concatenate([responders, rest])
        treatment = np.array(([1] * 100 + [0] * 100) * 2, dtype=np.int64)

        targeted = np.zeros(400, dtype=bool)
        targeted[:200] = True

        assert risk_ratio_of(outcome, treatment, targeted, binary=True) == pytest.approx(2.0)
        assert risk_ratio_of(
            outcome, treatment, np.ones(400, dtype=bool), binary=True
        ) == pytest.approx(1.5)

    def test_a_group_too_thin_in_one_arm_reports_nothing(self):
        outcome, treatment, _ = two_arm_sample(400, treated_rate=0.3, control_rate=0.2)
        thin = np.zeros(400, dtype=bool)
        thin[:200] = True  # every treated unit, no controls at all
        assert math.isnan(risk_ratio_of(outcome, treatment, thin, binary=True))

        barely = np.zeros(400, dtype=bool)
        barely[:50] = True  # plenty of treated units
        barely[200 : 200 + MIN_ARM - 1] = True  # one control short of the floor
        assert math.isnan(risk_ratio_of(outcome, treatment, barely, binary=True))

        enough = barely.copy()
        enough[200 + MIN_ARM - 1] = True  # the floor exactly
        assert math.isfinite(risk_ratio_of(outcome, treatment, enough, binary=True))

    def test_a_zero_control_rate_is_undefined_rather_than_infinite(self):
        outcome, treatment, _ = two_arm_sample(400, treated_rate=0.3, control_rate=0.0)
        assert math.isnan(
            risk_ratio_of(outcome, treatment, np.ones(400, dtype=bool), binary=True)
        )

    def test_a_continuous_outcome_goes_through_the_approximation(self):
        rng = np.random.default_rng(11)
        treatment = np.array([1] * 500 + [0] * 500, dtype=np.int64)
        outcome = np.concatenate([rng.normal(1.0, 1.0, 500), rng.normal(0.0, 1.0, 500)])
        targeted = np.ones(1000, dtype=bool)

        ratio = risk_ratio_of(outcome, treatment, targeted, binary=False)
        # A one standard deviation shift is d ~ 1, so the converted ratio is about e^0.91.
        assert ratio == pytest.approx(math.exp(SMD_TO_LOG_RR), rel=0.1)


class TestTheReportedEValue:
    def test_a_real_effect_needs_real_confounding(self):
        outcome, treatment, scores = two_arm_sample(2000, treated_rate=0.30, control_rate=0.15)
        result = targeting_e_value(
            outcome, treatment, scores, budget=1.0, n_resamples=200, seed=3
        )
        assert result.risk_ratio.value == pytest.approx(2.0, rel=0.01)
        assert result.point == pytest.approx(e_value_of(result.risk_ratio.value))
        assert result.established
        assert 1.0 < result.limit < result.point
        assert result.scale == "rate"

    def test_no_effect_leaves_nothing_to_explain_away(self):
        outcome, treatment, scores = two_arm_sample(2000, treated_rate=0.2, control_rate=0.2)
        result = targeting_e_value(
            outcome, treatment, scores, budget=1.0, n_resamples=200, seed=3
        )
        assert result.risk_ratio.value == pytest.approx(1.0, abs=1e-9)
        assert result.point == pytest.approx(1.0)
        assert result.limit == 1.0
        assert not result.established
        assert "covers 1" in result.summary()

    def test_the_budget_decides_who_is_in_the_group(self):
        outcome, treatment, scores = two_arm_sample(2000, treated_rate=0.3, control_rate=0.2)
        for budget in (0.1, 0.5, 1.0):
            result = targeting_e_value(
                outcome, treatment, scores, budget=budget, n_resamples=50, seed=1
            )
            assert result.n_targeted == round(2000 * budget)
            assert result.budget == budget

    def test_the_policy_is_fixed_before_the_bootstrap_runs(self):
        # The interval is meant to carry the sampling variation of the *rates*, not of the
        # ranking. Re-ranking inside each resample would widen it with a source of noise
        # that has nothing to do with confounding, so the targeted set is chosen once and
        # then indexed. Two seeds that differ only in the resampling must agree on who was
        # targeted, which shows up as the same point estimate.
        outcome, treatment, scores = two_arm_sample(2000, treated_rate=0.3, control_rate=0.2)
        first = targeting_e_value(
            outcome, treatment, scores, budget=0.3, n_resamples=50, seed=1
        )
        second = targeting_e_value(
            outcome, treatment, scores, budget=0.3, n_resamples=80, seed=1
        )
        assert first.risk_ratio.value == second.risk_ratio.value
        assert first.n_targeted == second.n_targeted

    def test_a_continuous_outcome_reports_the_difference_it_converted(self):
        # The conversion is exponential in d, so the E-value cannot be read without it.
        rng = np.random.default_rng(21)
        treatment = np.array([1] * 500 + [0] * 500, dtype=np.int64)
        outcome = np.concatenate([rng.normal(1.0, 1.0, 500), rng.normal(0.0, 1.0, 500)])
        result = targeting_e_value(
            outcome, treatment, rng.normal(size=1000), budget=1.0, n_resamples=100, seed=2
        )
        assert result.standardised_difference == pytest.approx(1.0, abs=0.15)
        assert "standardised difference" in result.summary()
        assert "order of magnitude" in result.summary()

    def test_a_binary_outcome_carries_no_such_caveat(self):
        outcome, treatment, scores = two_arm_sample(2000, treated_rate=0.30, control_rate=0.15)
        result = targeting_e_value(
            outcome, treatment, scores, budget=1.0, n_resamples=100, seed=3
        )
        assert math.isnan(result.standardised_difference)
        assert "standardised difference" not in result.summary()

    def test_a_continuous_outcome_is_labelled_as_approximate(self):
        rng = np.random.default_rng(5)
        treatment = np.array([1] * 500 + [0] * 500, dtype=np.int64)
        outcome = np.concatenate([rng.normal(1.0, 1.0, 500), rng.normal(0.0, 1.0, 500)])
        result = targeting_e_value(
            outcome, treatment, rng.normal(size=1000), budget=1.0, n_resamples=100, seed=2
        )
        assert result.scale == "approximate"
