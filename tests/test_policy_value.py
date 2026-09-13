"""IPW and doubly robust policy value, against quantities that are known analytically.

Three kinds of check, and the second is the one that matters most.

**Identities.** At a budget of 100% the gain is the average treatment effect, exactly, not
approximately: with a constant propensity the Horvitz-Thompson estimator algebraically
collapses to the difference in arm means. An implementation that gets this wrong is wrong
everywhere, and the test is an equality rather than a tolerance.

**Double robustness.** The estimator's whole claim is that it survives one of its two
nuisance models being wrong. That is checked by breaking each one deliberately on data whose
potential outcomes are written down, and then by breaking both, so the test cannot pass by
the estimator ignoring its nuisances.

**Confounding.** On a dataset where the naive comparison of arms has the wrong sign, both
estimators recover the right one.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from itx.data import synthetic
from itx.data.splits import stratified_split
from itx.estimators.propensity import DEFAULT_CLIP, PropensityFit
from itx.metrics.qini import ate
from itx.policy.policy_value import (
    Nuisances,
    dr_contributions,
    dr_gain,
    dr_value,
    fit_nuisances,
    gain_key,
    ipw_contributions,
    ipw_gain,
    ipw_value,
    policy_metrics,
)
from itx.policy.rank_and_cut import rank_and_cut


def randomised_sample(
    n: int = 5_000, *, share: float = 0.4, effect: float = 2.0, seed: int = 0
):
    """A randomised trial with a constant effect and a known, constant propensity."""
    rng = np.random.default_rng(seed)
    treatment = (rng.random(n) < share).astype(np.int64)
    outcome = rng.normal(size=n) + effect * treatment
    propensity = np.full(n, float(treatment.mean()))
    return outcome, treatment, propensity


def known_potential_outcomes(n: int = 40_000, seed: int = 5):
    """Data whose two potential outcomes and true propensity are both written down.

    The propensity depends on a covariate and stays well inside ``(0, 1)``, so nothing here
    is testing the clipping. The effect depends on the same covariate, so a policy that
    selects on it is selecting on something real.
    """
    rng = np.random.default_rng(seed)
    covariate = rng.uniform(0.0, 1.0, n)
    propensity = 0.2 + 0.6 * covariate
    treatment = (rng.random(n) < propensity).astype(np.int64)
    effect = 1.0 + covariate
    under_control = covariate + rng.normal(0.0, 0.1, n)
    outcome = under_control + treatment * effect
    return {
        "outcome": outcome,
        "treatment": treatment,
        "propensity": propensity,
        "mu0": covariate,
        "mu1": covariate + effect,
        "treat": covariate > 0.5,
        # The sample's own potential-outcome mean under the policy, which is what these
        # estimators target: not the population expectation, the realised one.
        "truth": float((under_control + (covariate > 0.5) * effect).mean()),
    }


def nuisances_of(sample, *, propensity=None, mu0=None, mu1=None) -> Nuisances:
    """Nuisances for a constructed sample, with any of the three replaced by a wrong one.

    Assembled rather than fitted: these tests are about the estimator's arithmetic, and a
    fitted nuisance would put a LightGBM model's error inside a test of whether the
    weighting is right.
    """
    values = sample["propensity"] if propensity is None else propensity
    return Nuisances(
        propensity=values,
        mu0=sample["mu0"] if mu0 is None else mu0,
        mu1=sample["mu1"] if mu1 is None else mu1,
        propensity_fit=PropensityFit(
            values=values,
            known=True,
            clip=DEFAULT_CLIP,
            n_clipped=0,
            raw_min=float(values.min()),
            raw_max=float(values.max()),
        ),
    )


class TestIdentities:
    def test_the_gain_from_treating_everybody_is_the_ate(self):
        # Exact, not approximate: with a constant propensity equal to the treated share,
        # the Horvitz-Thompson estimator collapses to the difference in arm means.
        outcome, treatment, propensity = randomised_sample()
        everybody = np.ones(outcome.size, dtype=bool)
        assert ipw_gain(outcome, treatment, everybody, propensity) == pytest.approx(
            ate(outcome, treatment)
        )

    def test_the_value_of_treating_everybody_is_the_treated_arm_mean(self):
        outcome, treatment, propensity = randomised_sample()
        everybody = np.ones(outcome.size, dtype=bool)
        assert ipw_value(outcome, treatment, everybody, propensity) == pytest.approx(
            outcome[treatment == 1].mean()
        )

    def test_the_value_of_treating_nobody_is_the_control_arm_mean(self):
        outcome, treatment, propensity = randomised_sample()
        nobody = np.zeros(outcome.size, dtype=bool)
        assert ipw_value(outcome, treatment, nobody, propensity) == pytest.approx(
            outcome[treatment == 0].mean()
        )

    def test_treating_nobody_gains_nothing(self):
        outcome, treatment, propensity = randomised_sample()
        nobody = np.zeros(outcome.size, dtype=bool)
        assert ipw_gain(outcome, treatment, nobody, propensity) == pytest.approx(0.0)

    def test_random_targeting_buys_its_share_of_the_ate(self):
        # PLAN.md section 5 lists this as one of the tests that has to exist: a policy that
        # picks at random cannot find heterogeneity, so it buys the budget's share of the
        # average effect and nothing more.
        outcome, treatment, propensity = randomised_sample(n=40_000, seed=4)
        rng = np.random.default_rng(1)
        for budget in (0.1, 0.3, 0.5):
            treat = np.zeros(outcome.size, dtype=bool)
            treat[rng.choice(outcome.size, int(budget * outcome.size), replace=False)] = True
            assert ipw_gain(outcome, treatment, treat, propensity) == pytest.approx(
                budget * ate(outcome, treatment), abs=0.05
            )

    def test_a_policy_that_finds_the_effect_beats_one_that_does_not(self):
        sample = known_potential_outcomes(n=20_000, seed=9)
        targeted = ipw_gain(
            sample["outcome"], sample["treatment"], sample["treat"], sample["propensity"]
        )
        rng = np.random.default_rng(2)
        n = sample["outcome"].size
        random_policy = np.zeros(n, dtype=bool)
        random_policy[rng.choice(n, int(sample["treat"].sum()), replace=False)] = True
        untargeted = ipw_gain(
            sample["outcome"], sample["treatment"], random_policy, sample["propensity"]
        )
        assert targeted > untargeted


class TestDoubleRobustness:
    def test_both_estimators_recover_a_known_policy_value(self):
        sample = known_potential_outcomes()
        nuisances = nuisances_of(sample)
        assert ipw_value(
            sample["outcome"], sample["treatment"], sample["treat"], sample["propensity"]
        ) == pytest.approx(sample["truth"], abs=0.02)
        assert dr_value(
            sample["outcome"], sample["treatment"], sample["treat"], nuisances
        ) == pytest.approx(sample["truth"], abs=0.02)

    def test_it_survives_a_broken_propensity(self):
        # The outcome models are right, the propensity says everybody was a coin flip.
        sample = known_potential_outcomes()
        broken = nuisances_of(sample, propensity=np.full(sample["outcome"].size, 0.5))
        assert dr_value(
            sample["outcome"], sample["treatment"], sample["treat"], broken
        ) == pytest.approx(sample["truth"], abs=0.02)

    def test_it_survives_broken_outcome_models(self):
        # The propensity is right, the outcome models predict zero for everybody. This is
        # the case where the DR estimator reduces to the IPW one.
        sample = known_potential_outcomes()
        zeros = np.zeros(sample["outcome"].size)
        broken = nuisances_of(sample, mu0=zeros, mu1=zeros)
        assert dr_value(
            sample["outcome"], sample["treatment"], sample["treat"], broken
        ) == pytest.approx(sample["truth"], abs=0.02)

    def test_it_does_not_survive_both_being_broken(self):
        # Without this the three tests above would pass on an estimator that quietly
        # ignored its nuisances.
        sample = known_potential_outcomes()
        zeros = np.zeros(sample["outcome"].size)
        broken = nuisances_of(
            sample, propensity=np.full(sample["outcome"].size, 0.5), mu0=zeros, mu1=zeros
        )
        estimated = dr_value(sample["outcome"], sample["treatment"], sample["treat"], broken)
        assert abs(estimated - sample["truth"]) > 0.1

    def test_the_dr_estimate_is_the_tighter_one(self):
        # Both are unbiased here, so the reason to carry outcome models at all is variance.
        # Measured across independent samples rather than asserted: the DR estimator
        # reweights residuals, which are smaller than the outcomes the IPW one reweights.
        ipw_errors, dr_errors = [], []
        for seed in range(12):
            sample = known_potential_outcomes(n=3_000, seed=100 + seed)
            ipw_errors.append(
                ipw_value(
                    sample["outcome"],
                    sample["treatment"],
                    sample["treat"],
                    sample["propensity"],
                )
                - sample["truth"]
            )
            dr_errors.append(
                dr_value(
                    sample["outcome"],
                    sample["treatment"],
                    sample["treat"],
                    nuisances_of(sample),
                )
                - sample["truth"]
            )
        assert float(np.std(dr_errors)) < float(np.std(ipw_errors))


class TestUnderConfounding:
    def test_both_estimators_fix_a_naive_comparison_with_the_wrong_sign(self):
        # The point of the whole module in one test. On this generator the units likeliest
        # to be treated are the ones with the worst outcomes, so comparing the arms
        # directly reports a harmful treatment that is in fact helpful.
        data = synthetic.confounded(8_000, seed=7)
        split = stratified_split(data, 11)
        test = split.test
        nuisances = fit_nuisances(split.train, test, seed=0)
        everybody = np.ones(test.n_units, dtype=bool)
        truth = float(test.require_true_effect().mean())

        assert ate(test.outcome, test.treatment) < 0.0
        assert truth > 0.0
        assert ipw_gain(test.outcome, test.treatment, everybody, nuisances.propensity) > 0.0
        assert dr_gain(test.outcome, test.treatment, everybody, nuisances) > 0.0


class TestNuisances:
    def test_a_known_design_propensity_is_used_rather_than_modelled(self):
        data = synthetic.constant_effect(2_000, effect=1.0, seed=3)
        split = stratified_split(data, 11)
        nuisances = fit_nuisances(split.train, split.test, seed=0)
        assert nuisances.propensity_fit.known
        assert nuisances.propensity == pytest.approx(split.test.propensity)

    def test_an_unknown_propensity_is_fitted_on_the_training_split(self):
        data = synthetic.constant_effect(2_000, effect=1.0, seed=3)
        split = stratified_split(data, 11)
        blind = replace(split.test, propensity=None)
        nuisances = fit_nuisances(split.train, blind, seed=0)
        assert not nuisances.propensity_fit.known
        assert nuisances.n_units == blind.n_units

    def test_they_describe_the_test_rows(self):
        data = synthetic.binary_outcome(3_000, seed=3)
        split = stratified_split(data, 11)
        nuisances = fit_nuisances(split.train, split.test, seed=0)
        assert nuisances.n_units == split.test.n_units
        assert nuisances.mu0.size == split.test.n_units

    def test_a_single_arm_training_split_is_refused(self):
        data = synthetic.constant_effect(500, effect=1.0, seed=3)
        split = stratified_split(data, 11)
        control_only = split.train.take(np.flatnonzero(split.train.treatment == 0))
        with pytest.raises(ValueError, match="no treated rows"):
            fit_nuisances(control_only, split.test, seed=0)

    def test_resampling_keeps_a_unit_with_its_own_probability(self):
        # A bootstrap that resampled outcomes but not propensities would pair each unit's
        # outcome with somebody else's probability of having received it.
        sample = known_potential_outcomes(n=1_000, seed=11)
        nuisances = nuisances_of(sample)
        index = np.array([3, 3, 7, 0], dtype=np.int64)
        taken = nuisances.take(index)
        assert taken.n_units == 4
        assert taken.propensity.tolist() == nuisances.propensity[index].tolist()
        assert taken.mu1.tolist() == nuisances.mu1[index].tolist()

    def test_mismatched_nuisance_arrays_are_refused(self):
        sample = known_potential_outcomes(n=100, seed=11)
        with pytest.raises(ValueError, match="same units"):
            Nuisances(
                propensity=sample["propensity"],
                mu0=sample["mu0"][:50],
                mu1=sample["mu1"],
                propensity_fit=nuisances_of(sample).propensity_fit,
            )


class TestPolicyMetrics:
    def test_it_reports_both_estimators_at_every_budget(self):
        sample = known_potential_outcomes(n=2_000, seed=13)
        metrics = policy_metrics(
            sample["outcome"],
            sample["treatment"],
            sample["mu1"] - sample["mu0"],
            nuisances_of(sample),
            budgets=(0.1, 0.2),
        )
        assert sorted(metrics) == [
            "gain_dr@10%",
            "gain_dr@20%",
            "gain_ipw@10%",
            "gain_ipw@20%",
        ]

    def test_it_agrees_with_the_single_budget_functions(self):
        sample = known_potential_outcomes(n=2_000, seed=13)
        scores = sample["mu1"] - sample["mu0"]
        nuisances = nuisances_of(sample)
        metrics = policy_metrics(
            sample["outcome"], sample["treatment"], scores, nuisances, budgets=(0.25,)
        )
        treat = rank_and_cut(scores, 0.25)
        assert metrics[gain_key("ipw", 0.25)] == pytest.approx(
            ipw_gain(sample["outcome"], sample["treatment"], treat, nuisances.propensity)
        )
        assert metrics[gain_key("dr", 0.25)] == pytest.approx(
            dr_gain(sample["outcome"], sample["treatment"], treat, nuisances)
        )

    def test_a_full_budget_recovers_the_ate(self):
        outcome, treatment, propensity = randomised_sample(n=8_000, seed=6)
        nuisances = Nuisances(
            propensity=propensity,
            mu0=np.zeros(outcome.size),
            mu1=np.zeros(outcome.size),
            propensity_fit=nuisances_of(known_potential_outcomes(n=100, seed=1)).propensity_fit,
        )
        metrics = policy_metrics(
            outcome,
            treatment,
            np.arange(outcome.size, dtype=np.float64),
            nuisances,
            budgets=(1.0,),
        )
        assert metrics[gain_key("ipw", 1.0)] == pytest.approx(ate(outcome, treatment))

    def test_the_budget_prefix_is_the_one_the_curves_use(self):
        # A policy value that belonged to a different set of people than the curve beside
        # it would be very hard to see and completely wrong.
        sample = known_potential_outcomes(n=1_000, seed=17)
        scores = np.zeros(1_000)  # every score tied, so only the tie seed decides
        nuisances = nuisances_of(sample)
        metrics = policy_metrics(
            sample["outcome"], sample["treatment"], scores, nuisances, budgets=(0.3,), seed=5
        )
        treat = rank_and_cut(scores, 0.3, seed=5)
        assert metrics[gain_key("ipw", 0.3)] == pytest.approx(
            ipw_gain(sample["outcome"], sample["treatment"], treat, nuisances.propensity)
        )


class TestContributions:
    """The per-unit decomposition the fast path is built on.

    ``policy_metrics`` does not call the gain functions; it sums a per-unit contribution
    down the ranking. That is a second implementation of the same quantity, so it is checked
    against the first rather than trusted.
    """

    @pytest.mark.parametrize("budget", [0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    def test_the_fast_path_agrees_with_the_definition_at_every_budget(self, budget):
        sample = known_potential_outcomes(n=3_000, seed=21)
        scores = sample["mu1"] - sample["mu0"]
        nuisances = nuisances_of(sample)
        metrics = policy_metrics(
            sample["outcome"], sample["treatment"], scores, nuisances, budgets=(budget,)
        )
        treat = rank_and_cut(scores, budget)
        assert metrics[gain_key("ipw", budget)] == pytest.approx(
            ipw_gain(sample["outcome"], sample["treatment"], treat, nuisances.propensity),
            rel=1e-12,
            abs=1e-12,
        )
        assert metrics[gain_key("dr", budget)] == pytest.approx(
            dr_gain(sample["outcome"], sample["treatment"], treat, nuisances),
            rel=1e-12,
            abs=1e-12,
        )

    def test_a_units_contribution_is_zero_for_the_arm_it_was_not_in(self):
        # Each unit was observed in one arm, so only one of the two terms can be non-zero,
        # which is what makes the weighting an estimate rather than a comparison.
        outcome = np.array([1.0, 1.0])
        treatment = np.array([1, 0], dtype=np.int64)
        propensity = np.array([0.5, 0.5])
        contributions = ipw_contributions(outcome, treatment, propensity)
        assert contributions.tolist() == [2.0, -2.0]

    def test_the_mean_contribution_is_the_gain_from_treating_everybody(self):
        outcome, treatment, propensity = randomised_sample(n=4_000, seed=8)
        contributions = ipw_contributions(outcome, treatment, propensity)
        assert float(contributions.mean()) == pytest.approx(ate(outcome, treatment))

    def test_the_dr_contribution_reduces_to_the_ipw_one_without_outcome_models(self):
        sample = known_potential_outcomes(n=1_000, seed=23)
        zeros = np.zeros(sample["outcome"].size)
        assert dr_contributions(
            sample["outcome"],
            sample["treatment"],
            nuisances_of(sample, mu0=zeros, mu1=zeros),
        ) == pytest.approx(
            ipw_contributions(sample["outcome"], sample["treatment"], sample["propensity"])
        )

    def test_contributions_refuse_an_impossible_propensity(self):
        outcome, treatment, propensity = randomised_sample(n=50)
        propensity = propensity.copy()
        propensity[2] = 1.0
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            ipw_contributions(outcome, treatment, propensity)


class TestRefusals:
    def test_a_propensity_of_zero_is_refused(self):
        outcome, treatment, propensity = randomised_sample(n=100)
        propensity = propensity.copy()
        propensity[0] = 0.0
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            ipw_value(outcome, treatment, np.ones(100, dtype=bool), propensity)

    def test_a_propensity_of_one_is_refused(self):
        outcome, treatment, propensity = randomised_sample(n=100)
        propensity = propensity.copy()
        propensity[5] = 1.0
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            ipw_value(outcome, treatment, np.ones(100, dtype=bool), propensity)

    def test_mismatched_lengths_are_refused(self):
        outcome, treatment, propensity = randomised_sample(n=100)
        with pytest.raises(ValueError, match="same length"):
            ipw_value(outcome, treatment, np.ones(50, dtype=bool), propensity)

    def test_an_empty_sample_is_refused(self):
        empty_float = np.zeros(0)
        with pytest.raises(ValueError, match="empty sample"):
            ipw_value(
                empty_float, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool), empty_float
            )


@pytest.mark.slow
class TestAgainstGroundTruthOnAcic:
    """The estimators, checked against effects that were written down rather than inferred.

    Everywhere else in this file the truth is a quantity the test constructs. ACIC 2016 is
    the real thing: a public benchmark whose individual treatment effects are known because
    the outcomes were simulated from them, and whose covariates and assignment mechanism are
    not simulated at all. So the policy gains the README reports can be compared with what
    the policies actually bought, which is the only check that cannot be gamed by an
    implementation agreeing with itself.

    Two claims rest on this and both are asserted here. The first is the project's headline:
    on ACIC, spending a 10% budget on the highest-risk cases buys a *negative* amount of
    outcome, so the ordinary way of doing this job is worse than doing nothing. The second
    is about which estimator to believe, and it matters because the two disagree by a factor
    of three: 46% of ACIC's test rows sit against the propensity clipping bound, the
    diagnostic says so before any truth is consulted, and the truth then confirms that IPW
    is the one that fell over.
    """

    @pytest.fixture(scope="class")
    def measured(
        self,
    ) -> tuple[dict[tuple[str, float], float], dict[tuple[str, float, str], float], list[bool]]:
        """True, DR and IPW policy gains per estimator and budget, averaged over the seeds.

        Class-scoped, because it is twenty-five refits of ACIC and the three tests below all
        read the same numbers.
        """
        from itx.bench.runner import nuisances_for, refit_seed
        from itx.bench.seeds import SEEDS, TIE_SEED
        from itx.bench.table import read_json

        results = Path("results/acic.json")
        if not results.is_file():  # pragma: no cover - needs a finished benchmark
            pytest.skip("needs results/acic.json from a finished benchmark run")

        rows = read_json(results)
        budgets = (0.1, 0.2, 0.3)
        truth: dict[tuple[str, float], list[float]] = {}
        estimated: dict[tuple[str, float, str], list[float]] = {}
        overlap: list[bool] = []

        for seed in SEEDS:
            selections = {
                row.estimator: row.selection
                for row in rows
                if row.seed == seed and row.selection is not None
            }
            refitted, split = refit_seed("acic", seed, selections=selections)
            effects = split.test.require_true_effect()
            nuisances = nuisances_for(split)
            overlap.append(nuisances.propensity_fit.has_overlap_problem)

            for row in refitted:
                metrics = policy_metrics(
                    split.test.outcome,
                    split.test.treatment,
                    row.scores,
                    nuisances,
                    budgets=budgets,
                    seed=TIE_SEED,
                )
                for budget in budgets:
                    treated = rank_and_cut(row.scores, budget, seed=TIE_SEED)
                    truth.setdefault((row.estimator, budget), []).append(
                        float((effects * treated).mean())
                    )
                    for name in ("ipw", "dr"):
                        estimated.setdefault((row.estimator, budget, name), []).append(
                            metrics[gain_key(name, budget)]
                        )

        return (
            {key: float(np.mean(values)) for key, values in truth.items()},
            {key: float(np.mean(values)) for key, values in estimated.items()},
            overlap,
        )

    def test_the_risk_ranking_really_does_buy_negative_outcome(self, measured):
        true_gain, estimated, _ = measured
        # Not "worse than random", which the ranking table could already suggest and whose
        # interval there contains zero. Worse than treating nobody at all, against effects
        # that were written down before any model saw them.
        assert true_gain[("outcome-ranking", 0.1)] < 0.0
        assert estimated[("outcome-ranking", 0.1, "dr")] < 0.0
        for estimator in ("s-learner", "t-learner", "x-learner", "dr-learner", "r-learner"):
            assert true_gain[(estimator, 0.1)] > 0.0
            assert true_gain[(estimator, 0.1)] > true_gain[("outcome-ranking", 0.1)]

    def test_the_doubly_robust_estimate_is_the_one_to_believe_here(self, measured):
        true_gain, estimated, overlap = measured
        errors = {
            name: float(
                np.mean(
                    [
                        abs(estimated[(estimator, budget, name)] - value)
                        for (estimator, budget), value in true_gain.items()
                    ]
                )
            )
            for name in ("ipw", "dr")
        }
        # A factor of three on the levels, and far more than that on the error.
        assert errors["dr"] < errors["ipw"] / 5.0
        # And the propensity diagnostic said which one would fall over, without the truth.
        assert all(overlap)

    def test_every_doubly_robust_estimate_lands_near_the_truth(self, measured):
        true_gain, estimated, _ = measured
        for (estimator, budget), value in true_gain.items():
            assert estimated[(estimator, budget, "dr")] == pytest.approx(value, abs=0.3), (
                f"{estimator} at {budget:.0%}"
            )
