"""The risk-decile diagnostic, on data built so that the right answer is known in advance.

The table's job is to decide whether ranking by risk approximates ranking by uplift, so the
tests build three populations where that question has a known answer and check that the
verdict is the right one: an effect proportional to risk, an effect running against risk,
and no effect at all. A diagnostic that cannot tell those three apart is worse than no
diagnostic, because it would be consulted.

The fourth test is the one that matters most, and it checks a refusal rather than an answer.
It takes the same against-risk population, weakens the effect to a twentieth of its size, and
requires the table to say it cannot tell. Announcing a direction from ten noisy bands is
precisely the mistake this repository was built to demonstrate, so a diagnostic that made it
would be worse than none.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from itx.estimators.baselines import OutcomeRanking
from itx.metrics.bootstrap import Estimate
from itx.metrics.risk_deciles import (
    Decile,
    RiskDecileTable,
    _rank_correlation,
    _statistics,
    risk_deciles,
    risk_scores,
)
from itx.types import UpliftDataset


def population(
    n: int, *, effect: str, seed: int = 0, share: float = 0.5, strength: float = 1.0
) -> UpliftDataset:
    """A binary-outcome population whose risk and effect are related as asked.

    ``driver`` sets the baseline risk. The three effect shapes are the three cases the
    diagnostic exists to distinguish, and ``strength`` scales the effect down so that a
    shape the table would resolve easily at full size can be made too faint to resolve.
    """
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.3, n)
    baseline = 0.02 + 0.6 * driver

    if effect == "with-risk":
        # A constant relative effect, so absolute uplift is proportional to risk.
        uplift = strength * 0.4 * baseline
    elif effect == "against-risk":
        # The intervention helps the people least likely to have the outcome anyway.
        uplift = strength * 0.3 * (1.0 - driver)
    elif effect == "none":
        uplift = np.zeros(n)
    else:  # pragma: no cover - guarded by the callers
        raise ValueError(effect)

    treatment = (rng.random(n) < share).astype(np.int64)
    probability = np.clip(baseline + treatment * uplift, 0.0, 1.0)
    outcome = (rng.random(n) < probability).astype(np.float64)
    return UpliftDataset(
        name=f"synthetic-{effect}",
        features=pl.DataFrame({"driver": driver, "noise": noise}),
        treatment=treatment,
        outcome=outcome,
        propensity=np.full(n, share),
        true_effect=uplift,
    )


def diagnose(
    effect: str, n: int = 30_000, *, seed: int = 0, strength: float = 1.0, **kwargs
) -> RiskDecileTable:
    """Fit the diagnostic on one half of a population and measure it on the other."""
    data = population(n, effect=effect, seed=seed, strength=strength)
    half = n // 2
    train = data.take(np.arange(half))
    test = data.take(np.arange(half, n), name=data.name)
    return risk_deciles(train, test, n_resamples=200, seed=seed, **kwargs)


class TestTheThreeCases:
    def test_an_effect_proportional_to_risk_is_called_agreement(self):
        table = diagnose("with-risk")
        assert table.correlation.low > 0.0
        assert table.risk_dominates
        assert "broadly agree" in table.verdict

    def test_an_effect_running_against_risk_is_called_opposition(self):
        table = diagnose("against-risk")
        assert table.correlation.high < 0.0
        assert "runs against risk" in table.verdict

    def test_no_effect_at_all_is_called_undecidable(self):
        table = diagnose("none")
        assert not table.correlation.excludes_zero
        assert "cannot tell" in table.verdict

    def test_an_effect_too_faint_to_resolve_is_not_announced(self):
        # The property the whole diagnostic rests on. This population's effect really does
        # run against risk, at a twentieth of the size the test above uses, and the table
        # has to decline to say so rather than report whichever ordering the noise left.
        table = diagnose("against-risk", n=6_000, strength=0.05)
        assert "cannot tell" in table.verdict


def value_population(n: int, *, seed: int = 0) -> UpliftDataset:
    """A dollars-retained population: the intervention removes losses in proportion to risk.

    The binary "with-risk" population, re-expressed as value. Each event is a loss of one
    dollar; the intervention prevents events in proportion to the baseline rate; and the
    outcome recorded is minus the loss. So the units a risk model should queue first are the
    ones with the lowest predicted outcome, uplift in dollars is positive and proportional
    to risk, and the right verdict is that the two rankings agree. Read the other way, as
    every dataset before the fraud case was, the same numbers say the effect runs against
    risk, which is the mistake the flag exists to prevent.
    """
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.3, n)
    baseline = 0.02 + 0.6 * driver
    prevented = 0.4 * baseline
    treatment = (rng.random(n) < 0.5).astype(np.int64)
    probability = np.clip(baseline - treatment * prevented, 0.0, 1.0)
    loss = (rng.random(n) < probability).astype(np.float64)
    return UpliftDataset(
        name="synthetic-value",
        features=pl.DataFrame({"driver": driver, "noise": noise}),
        treatment=treatment,
        outcome=-loss,
        propensity=np.full(n, 0.5),
        true_effect=prevented,
        risk_is_low_outcome=True,
    )


class TestAValueOutcome:
    def diagnose(self, data: UpliftDataset) -> RiskDecileTable:
        half = data.n_units // 2
        train = data.take(np.arange(half))
        test = data.take(np.arange(half, data.n_units), name=data.name)
        return risk_deciles(train, test, n_resamples=200, seed=0)

    def test_the_flag_survives_a_split(self):
        data = value_population(200)
        assert data.take(np.arange(50)).risk_is_low_outcome

    def test_risk_points_at_the_lowest_outcome_and_the_verdict_follows(self):
        table = self.diagnose(value_population(30_000))
        # Band 1 is the riskiest: the most negative predicted outcome, so the largest
        # predicted risk, and the largest measured uplift in dollars.
        risks = [band.predicted_risk for band in table.deciles]
        assert risks == sorted(risks, reverse=True)
        assert table.deciles[0].control_rate < table.deciles[-1].control_rate
        assert table.deciles[0].uplift.value > table.deciles[-1].uplift.value
        assert table.correlation.low > 0.0
        assert "runs against risk" not in table.verdict

    def test_read_the_old_way_the_same_data_gives_the_opposite_verdict(self):
        # The defect this guards against, kept as a test rather than a memory.
        from dataclasses import replace

        table = self.diagnose(replace(value_population(30_000), risk_is_low_outcome=False))
        assert table.correlation.high < 0.0
        assert "runs against risk" in table.verdict

    def test_a_population_that_only_loses_money_has_every_ratio_defined(self):
        table = self.diagnose(value_population(30_000))
        # Every band's control mean is a loss, so every band's risk is positive: the
        # multipliers are all below one (the intervention removes loss), the risk spread is
        # the spread of loss across bands, and the verdict is the same agreement the binary
        # version of this population gets.
        assert all(0.0 < band.multiplier < 1.0 for band in table.deciles)
        assert table.risk_spread.value > 1.0
        assert table.risk_dominates
        assert "broadly agree" in table.verdict

    def test_a_band_that_makes_money_has_no_multiplier(self):
        loses = Decile(0, 10, 5, 0.5, -0.50, -0.25, Estimate(0.25, 0.1, 0.4, 0.95, 1), -1.0)
        earns = Decile(1, 10, 5, -0.5, 0.50, 0.60, Estimate(0.10, 0.0, 0.2, 0.95, 1), -1.0)
        responds = Decile(0, 10, 5, 0.5, 0.50, 0.60, Estimate(0.10, 0.0, 0.2, 0.95, 1))
        never = Decile(1, 10, 5, 0.0, -0.50, -0.25, Estimate(0.25, 0.1, 0.4, 0.95, 1))
        # Review cut the loss to half: a real relative effect, the same number either way up.
        assert loses.multiplier == pytest.approx(0.5)
        assert np.isnan(earns.multiplier)
        assert responds.multiplier == pytest.approx(1.2)
        assert np.isnan(never.multiplier)

    def test_a_spread_across_bands_of_both_signs_is_undefined(self):
        # Two bands: one loses a dollar per unit, one earns a dollar. As risks, +1 and -1,
        # and no ratio can be formed across them. Rendered as a dash, not a number.
        outcome = np.array([-1.0, -1.0, 1.0, 1.0] * 50)
        treatment = np.array([0, 1] * 100, dtype=np.int64)
        band_of = np.array([0, 0, 1, 1] * 50, dtype=np.int64)
        values = _statistics(outcome, treatment, band_of, 2, sign=-1.0)(np.arange(200))
        assert np.isnan(values["risk_spread"])
        assert np.isnan(values["multiplier_spread"])


def churn_population(n: int, *, seed: int = 0) -> UpliftDataset:
    """A churn population: the outcome is bad, and the intervention pushes it down.

    Every dataset in this repository has a good outcome at the high end, so until the CSV
    loader arrived nothing here had a bad one. The same numbers encoded as ``retained``
    rather than ``churned`` are the value population above. Encoded this way, a working
    intervention shows up as a *negative* uplift, and a verdict that assumes bigger is
    better reads it as the effect running against risk, which is the opposite of the truth.
    PLAN.md change 60.
    """
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.3, n)
    baseline = 0.02 + 0.6 * driver
    prevented = 0.4 * baseline
    treatment = (rng.random(n) < 0.5).astype(np.int64)
    probability = np.clip(baseline - treatment * prevented, 0.0, 1.0)
    churn = (rng.random(n) < probability).astype(np.float64)
    return UpliftDataset(
        name="synthetic-churn",
        features=pl.DataFrame({"driver": driver, "noise": noise}),
        treatment=treatment,
        outcome=churn,
        propensity=np.full(n, 0.5),
        true_effect=-prevented,
        risk_is_low_outcome=False,
        higher_outcome_is_better=False,
    )


class TestABadOutcome:
    """Churn, where the intervention works by making the number smaller.

    The project's own headline example is a retention offer, and a retention dataset is
    usually recorded as churn rather than as retention. Nothing exercised that until a
    stranger could hand this package a CSV.
    """

    def table(self, *, higher_outcome_is_better: bool, n: int = 30_000, seed: int = 0):
        data = churn_population(n, seed=seed)
        if higher_outcome_is_better:
            data = UpliftDataset(
                name=data.name,
                features=data.features,
                treatment=data.treatment,
                outcome=data.outcome,
                propensity=data.propensity,
                true_effect=data.true_effect,
                risk_is_low_outcome=data.risk_is_low_outcome,
                higher_outcome_is_better=True,
            )
        half = n // 2
        return risk_deciles(
            data.take(np.arange(half)),
            data.take(np.arange(half, n), name=data.name),
            n_resamples=200,
            seed=seed,
        )

    def test_the_correlation_is_positive_when_the_outcome_is_declared_bad(self):
        """Risk high, benefit high: the two rankings agree and the sign must say so."""
        table = self.table(higher_outcome_is_better=False)
        assert table.correlation.value > 0.0

    def test_the_old_reading_gets_the_sign_exactly_backwards(self):
        """The defect, reproduced: the same data read as though bigger were better."""
        declared = self.table(higher_outcome_is_better=False)
        assumed = self.table(higher_outcome_is_better=True)
        assert assumed.correlation.value == pytest.approx(-declared.correlation.value)

    def test_the_verdict_does_not_announce_harm_where_the_intervention_works(self):
        table = self.table(higher_outcome_is_better=False)
        assert "runs against risk" not in table.verdict

    def test_the_uplifts_stay_in_the_outcomes_own_units(self):
        """The table describes the data; only the verdict knows which way is up."""
        table = self.table(higher_outcome_is_better=False)
        # Treatment prevents churn, so every band's uplift is negative as recorded.
        assert table.deciles[0].uplift.value < 0.0


class TestTheBands:
    def test_the_bands_are_ordered_with_the_riskiest_first(self):
        table = diagnose("with-risk")
        risks = [band.predicted_risk for band in table.deciles]
        assert risks == sorted(risks, reverse=True)
        assert table.deciles[0].index == 0

    def test_the_bands_partition_the_test_split(self):
        table = diagnose("with-risk")
        assert sum(band.n_units for band in table.deciles) == table.n_test
        assert len(table.deciles) == 10

    def test_the_bands_are_as_equal_as_they_divide(self):
        table = diagnose("with-risk", n=30_001)
        sizes = {band.n_units for band in table.deciles}
        assert max(sizes) - min(sizes) <= 1

    def test_a_different_number_of_bands_is_honoured(self):
        table = diagnose("with-risk", bins=4)
        assert len(table.deciles) == 4

    def test_the_measured_risk_tracks_the_predicted_risk(self):
        # If these came apart the model would be ranking on something other than risk, and
        # every number below the table would be describing the wrong bands.
        table = diagnose("with-risk")
        assert table.deciles[0].control_rate > table.deciles[-1].control_rate

    def test_a_constant_relative_effect_shows_as_a_flat_multiplier(self):
        # The generator multiplies the baseline by 1.4 everywhere, so the spread in the
        # multiplier should be small while the spread in risk is large. That gap is exactly
        # what the verdict reads.
        table = diagnose("with-risk")
        assert table.risk_spread.value > 5.0
        assert table.multiplier_spread.value < 2.0


class TestTheRiskModel:
    def test_it_is_fitted_on_the_control_rows_only(self):
        # A risk score is a model of what happens when nobody intervenes. Fitting on
        # everybody mixes the treated arm's elevated outcomes into the baseline, so its
        # predictions sit above the real no-intervention rate.
        data = population(8_000, effect="with-risk", seed=3)
        train = data.take(np.arange(4_000))
        test = data.take(np.arange(4_000, 8_000))

        control_only = risk_scores(train, test, seed=0)
        both_arms = OutcomeRanking(seed=0, fit_on="all")
        both_arms.fit(train)
        mixed = both_arms.predict_uplift(test.features)

        # The control-only fit lands near the real no-intervention rate; the mixed fit sits
        # above it, because it has averaged the treated arm's elevated outcomes in.
        observed_baseline = float(test.outcome[test.treatment == 0].mean())
        assert float(control_only.mean()) == pytest.approx(observed_baseline, abs=0.05)
        assert float(mixed.mean()) > float(control_only.mean())

    def test_it_returns_one_score_per_test_row(self):
        data = population(2_000, effect="none", seed=1)
        train = data.take(np.arange(1_000))
        test = data.take(np.arange(1_000, 2_000))
        assert risk_scores(train, test, seed=0).shape == (1_000,)


class TestRefusals:
    def test_fewer_than_two_bands_is_an_error(self):
        data = population(400, effect="none")
        with pytest.raises(ValueError, match="at least two bands"):
            risk_deciles(data, data, bins=1, n_resamples=10)

    def test_more_bands_than_rows_is_an_error(self):
        data = population(400, effect="none")
        tiny = data.take(np.arange(5))
        with pytest.raises(ValueError, match="cannot cut 5 rows"):
            risk_deciles(data, tiny, bins=10, n_resamples=10)


class TestRankCorrelation:
    def test_a_perfectly_ordered_pair_scores_one(self):
        values = np.arange(10.0)
        assert _rank_correlation(values, values) == pytest.approx(1.0)

    def test_a_reversed_pair_scores_minus_one(self):
        values = np.arange(10.0)
        assert _rank_correlation(values, values[::-1]) == pytest.approx(-1.0)

    def test_it_ignores_the_scale_and_reads_only_the_order(self):
        risk = np.arange(10.0)
        assert _rank_correlation(risk, np.exp(risk)) == pytest.approx(1.0)

    def test_a_constant_input_scores_zero_rather_than_warning(self):
        # SciPy returns NaN with a warning here. A band table where every band has the same
        # uplift is an ordinary thing to measure and its answer is zero, not a warning.
        assert _rank_correlation(np.arange(10.0), np.ones(10)) == pytest.approx(0.0)

    def test_too_few_usable_bands_is_undefined(self):
        assert np.isnan(_rank_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0])))

    def test_bands_that_could_not_be_measured_are_dropped(self):
        risk = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
        uplift = np.array([1.0, 2.0, 3.0, 4.0, 99.0])
        assert _rank_correlation(risk, uplift) == pytest.approx(1.0)


class TestRendering:
    def test_the_markdown_has_a_row_per_band(self):
        table = diagnose("with-risk", bins=5)
        lines = table.to_markdown().strip().splitlines()
        assert len(lines) == 7  # header, rule, five bands
        assert lines[2].startswith("| 1 |")

    def test_the_summary_carries_the_three_numbers_and_the_verdict(self):
        table = diagnose("with-risk")
        summary = table.summary()
        for label in ("risk spread", "multiplier spread", "correlation"):
            assert label in summary
        assert table.verdict in summary

    def test_a_band_with_no_control_events_reports_no_multiplier(self):
        band = RiskDecileTable(
            dataset="x",
            n_test=10,
            deciles=(),
            risk_spread=Estimate(1.0, 1.0, 1.0),
            multiplier_spread=Estimate(1.0, 1.0, 1.0),
            correlation=Estimate(0.0, -1.0, 1.0),
        )
        assert not band.correlation.excludes_zero
        assert "cannot tell" in band.verdict
