"""What a policy is actually worth: the outcome it buys, not the curve it draws.

A Qini coefficient says a ranking is better than another ranking. It does not say what
happens if you deploy it, and the two questions come apart often enough that this project
exists. The quantity a budget holder is paying for is the expected outcome under the rule
"treat the top b%, leave everybody else alone", and this module estimates it from a test
split on which treatment was assigned by the experiment rather than by the policy.

## The estimand, and why the gain rather than the level

The textbook policy value is ``V(pi) = E[Y(pi(X))]``: the average outcome if everybody were
assigned the arm the policy picks for them. PLAN.md section 4 asks for that, estimated two
ways. What goes in the results table is the difference

    ``G(pi) = V(pi) - V(treat nobody)``

which is the extra outcome per head of the whole population that the policy buys against
doing nothing. Both are here and :func:`ipw_value` reports the level, but the level is the
wrong thing to put in a table of estimators, for a reason that is about variance rather
than taste. ``V(pi)`` is dominated by the baseline outcome rate, which every policy on the
dataset shares: on Hillstrom the level is around 0.15 and the differences between policies
are around 0.01, so bootstrap intervals on the levels overlap almost completely even where
the differences are firm. In the difference, every unit the policy leaves alone cancels
exactly, and what is left is an estimate over the targeted set only. Same information,
readable intervals (PLAN.md change 36).

``G`` is on the scale of the outcome per unit of population, so on a binary outcome a gain
of 0.017 means seventeen extra events per thousand people in the population. At a budget of
100% it is the average treatment effect, which is the analytic check the tests make.

## Two estimators, and what each one is resting on

**IPW** reweights the units whose observed arm happens to match what the policy would have
done, by the probability of that arm. Unbiased whenever the propensity is right, which on a
randomised dataset is a design constant and not a modelling assumption. Its weakness is
variance: a unit with a propensity of 0.02 carries fifty times the weight of an average one,
and on an observational dataset a handful of rows can carry the estimate. A known design
propensity is necessary for believing it and, as Criteo turned out to show, not sufficient.

**DR** adds outcome models for the two arms and reweights only their residuals. It is
consistent if *either* the propensity or the outcome models are right, which is the property
the tests check directly by breaking one at a time. It is also the lower-variance of the two
when the outcome models are any good, because the residuals are smaller than the outcomes.
This is the estimator to believe on ACIC and IHDP, where assignment is not randomised.

Both are reported, always, and a disagreement between them is information rather than an
embarrassment: it says the outcome models and the propensity model are telling different
stories about the same units.

**Which to read is decided by the clipped share, not by the dataset's reputation.** It is
tempting to sort the datasets into "randomised, trust IPW" and "observational, trust DR" and
stop there, and that sorting is wrong twice over. Lenta was randomised but does not publish
its assignment probability, so a model estimates a number that is in truth a constant, and
whether that model stayed away from the bounds is a question rather than an assumption. ACIC
and IHDP are the reverse case made concrete: 46% and 17.3% of their test rows sit against the
0.01 bound, each carrying an inverse weight of 100, and their IPW gains are wrong by a factor
of three and unusable respectively. :attr:`PropensityFit.clipped_share` is the number that
sorts those cases, it costs nothing, and it is available before any policy value is computed.

**And the clipped share is not sufficient either.** On Criteo nothing is clipped, the
propensity is a design constant, and the IPW gain is still half again the doubly robust one,
consistently across every seed. The cause is that the design is lopsided: 85% treated, so each
control unit is weighted by 1/0.15, and the top 10% of a fitted ranking turns out to hold 86.7%
treated rather than 85%. Criteo's arms are not quite balanced, by far too little to fail a
balance test and by more than enough for a model ranking on those covariates to concentrate
treated units at the top. A weight of 6.67 turns 1.7 points of concentration into 50% of the
answer (PLAN.md change 43).

The general statement is that Horvitz-Thompson weighting is fragile whenever an arm is small,
because a *selected* subset need not carry the population's treated share and the small arm's
weight multiplies the difference. The cheap check is to compare the realised treated share
inside the targeted prefix with the design propensity. The doubly robust estimator does not
have this problem, because its outcome models carry the prediction and the weights touch only
the residuals.

## The nuisance models are fitted once per split, not once per estimator

``mu0``, ``mu1`` and the propensity are properties of the *split*, not of the estimator
being evaluated. Fitting them separately inside each estimator's row, or reusing an
estimator's own internal models, would mean each estimator was scored by a different
referee, and an estimator whose nuisance models happened to be optimistic would be rewarded
for it. :func:`fit_nuisances` is therefore called once per split and the same arrays are
handed to every row in the table, including the two baselines.

They are fitted on the training split and applied to the test split, so no cross-fitting is
needed: the rows being scored were never seen by the models scoring them. That is the one
place where evaluating on held-out data makes a nuisance model easier rather than harder.

## How this relates to uplift at k, which is next to it in the table

They are different estimators of closely related quantities and it is worth saying which is
which. ``uplift@k`` compares the arms *inside* the targeted prefix, dividing by the arm
counts in that prefix, which makes it a ratio estimator of the Hajek kind. The IPW gain
divides by the arm probabilities in the *population*, which makes it Horvitz-Thompson. They
agree exactly when the prefix carries the population's treated share and diverge in
proportion to how far it does not, multiplied by the smaller arm's weight.

That divergence was described here as sampling noise in the prefix's share until Criteo
showed it is not always noise. A ranking selects the prefix on covariates, so any association
between covariates and assignment concentrates one arm there systematically rather than
randomly, and 1.7 points of concentration against a control weight of 6.67 moved the estimate
by half (change 43). Each is the honest answer to a slightly different question: the
prefix version is what a practitioner can compute without knowing a propensity at all, and it
is kept because it is what gets quoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig, OutcomeLearner
from itx.estimators.propensity import DEFAULT_CLIP, PropensityFit, PropensityModel
from itx.metrics.curves import DEFAULT_TIE_SEED, rank_order
from itx.policy.rank_and_cut import n_targeted

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itx.types import BoolArray, FloatArray, IntArray, UpliftDataset


@dataclass(frozen=True, slots=True)
class Nuisances:
    """The three arrays a policy-value estimator needs about the rows it is scoring.

    Aligned with the test split, row for row, and produced by models that were fitted on
    the training split and have never seen these rows.

    Attributes:
        propensity: Probability of treatment per unit, already clipped away from 0 and 1.
        mu0: Predicted outcome under control per unit.
        mu1: Predicted outcome under treatment per unit.
        propensity_fit: What the propensity cost to obtain, for the diagnostics. A known
            design propensity says so here rather than being indistinguishable from a
            fitted one.
    """

    propensity: FloatArray
    mu0: FloatArray
    mu1: FloatArray
    propensity_fit: PropensityFit

    def __post_init__(self) -> None:
        """Reject arrays that cannot describe the same units."""
        sizes = {self.propensity.size, self.mu0.size, self.mu1.size}
        if len(sizes) != 1:
            msg = (
                "nuisances must describe the same units; got propensity "
                f"{self.propensity.size}, mu0 {self.mu0.size}, mu1 {self.mu1.size}"
            )
            raise ValueError(msg)

    @property
    def n_units(self) -> int:
        """Number of units these nuisances describe."""
        return int(self.propensity.size)

    def take(self, index: IntArray) -> Nuisances:
        """The nuisances for a subset or a resample of the rows.

        The bootstrap resamples row positions, and a policy value computed on resampled
        outcomes with unresampled propensities would be pairing each unit's outcome with
        somebody else's probability of having received it. This keeps them together.

        Args:
            index: Row positions, which may repeat.

        Returns:
            The same nuisances, restricted to those rows.
        """
        return Nuisances(
            propensity=self.propensity[index],
            mu0=self.mu0[index],
            mu1=self.mu1[index],
            propensity_fit=self.propensity_fit,
        )


def fit_nuisances(
    train: UpliftDataset,
    test: UpliftDataset,
    *,
    config: BaseLearnerConfig = DEFAULT_CONFIG,
    seed: int = 0,
    clip: float = DEFAULT_CLIP,
) -> Nuisances:
    """Fit the outcome and propensity models on train, and apply them to test.

    The outcome models are a plain pair, one per arm, which is the T-learner's fit. It is
    deliberately not the T-learner object: the same arrays score every estimator in the
    table including the T-learner itself, and an estimator must not be allowed to supply
    its own referee.

    Args:
        train: Rows to fit the nuisance models on.
        test: Rows to predict for. Its design propensity is used where it has one.
        config: Shared LightGBM settings, the same ones the estimators use.
        seed: Seed for the models.
        clip: Propensities are held inside ``[clip, 1 - clip]``.

    Returns:
        Nuisances aligned with the test rows.

    Raises:
        ValueError: If either arm of the training split is empty, which leaves one of the
            two outcome models with nothing to fit on.
    """
    binary = train.outcome_is_binary
    categorical = [train.features.columns.index(column) for column in train.categorical]
    test_matrix = test.features.to_numpy().astype(np.float64, copy=False)
    predictions: dict[int, FloatArray] = {}
    for arm in (0, 1):
        rows = np.flatnonzero(train.treatment == arm)
        if rows.size == 0:
            label = "treated" if arm == 1 else "control"
            msg = f"policy value: no {label} rows in the training split to fit a nuisance on"
            raise ValueError(msg)
        learner = OutcomeLearner(config, binary=binary, seed=seed + arm)
        frame = train.features[rows]
        learner.fit(
            frame.to_numpy().astype(np.float64, copy=False),
            train.outcome[rows],
            categorical=categorical,
            feature_names=frame.columns,
        )
        predictions[arm] = learner.predict(test_matrix)

    fit = _test_propensity(train, test, config=config, seed=seed, clip=clip)
    return Nuisances(
        propensity=fit.values,
        mu0=predictions[0],
        mu1=predictions[1],
        propensity_fit=fit,
    )


def _test_propensity(
    train: UpliftDataset,
    test: UpliftDataset,
    *,
    config: BaseLearnerConfig,
    seed: int,
    clip: float,
) -> PropensityFit:
    """Probability of treatment for the test rows, known by design or fitted on train.

    A test row's propensity is a property of that row, so a dataset that records one is
    believed rather than modelled: fitting a model to recover a design constant adds
    variance and can only make the estimate worse (see :mod:`itx.estimators.propensity`).
    Cross-fitting does not arise here, because the model never sees the rows it scores.
    """
    if test.propensity is not None:
        raw = test.propensity
    else:
        model = PropensityModel(config, seed=seed, clip=clip, use_known=False, cross_fit=False)
        model.fit(train)
        # Unclipped on purpose: the clipped output cannot tell a propensity of 0.01 from one
        # of 0.0001, so counting clipped units from it would always count zero and the
        # overlap problem would be invisible in the one place it has to be visible.
        raw = model.predict_unclipped(test.features)
    clipped: FloatArray = np.clip(raw, clip, 1.0 - clip)
    return PropensityFit(
        values=clipped,
        known=test.propensity is not None,
        # These rows were held out of the fit, so by the field's own meaning nothing here
        # was predicted by a model that had seen it. The cross-fitting the training rows
        # need does not arise.
        out_of_fold=test.propensity is None,
        clip=clip,
        n_clipped=int(np.sum(raw != clipped)),
        raw_min=float(raw.min()) if raw.size else float("nan"),
        raw_max=float(raw.max()) if raw.size else float("nan"),
    )


def ipw_value(
    outcome: FloatArray,
    treatment: IntArray,
    treat: BoolArray,
    propensity: FloatArray,
) -> float:
    """Expected outcome under a policy, by inverse-probability weighting.

    Every unit whose observed arm is the one the policy would have chosen contributes its
    outcome, weighted by one over the probability it had of being in that arm. Units whose
    observed arm disagrees with the policy contribute nothing, which is why a policy that
    disagrees with the experiment almost everywhere is estimated from almost nothing.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        treat: The policy: True for units it would treat.
        propensity: Probability of treatment per unit, strictly inside ``(0, 1)``.

    Returns:
        The estimated average outcome under the policy.

    Raises:
        ValueError: If the inputs are different lengths or a propensity is not a
            probability strictly between 0 and 1.
    """
    _check(outcome, treatment, treat, propensity)
    weighted = np.where(
        _agrees(treatment, treat), outcome / _assigned_probability(treat, propensity), 0.0
    )
    return float(weighted.mean())


def dr_value(
    outcome: FloatArray,
    treatment: IntArray,
    treat: BoolArray,
    nuisances: Nuisances,
) -> float:
    """Expected outcome under a policy, doubly robust.

    The outcome models predict what the policy would buy for everybody, and the weighting
    corrects that prediction using only the residuals of the units whose observed arm
    agrees with the policy. If the outcome models are right the correction has mean zero;
    if the propensity is right the correction removes the models' bias. Either one being
    right is enough, which is the property the tests check by breaking one at a time.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        treat: The policy: True for units it would treat.
        nuisances: Propensity and the two outcome models, on these same rows.

    Returns:
        The estimated average outcome under the policy.

    Raises:
        ValueError: If the inputs are different lengths or a propensity is not a
            probability strictly between 0 and 1.
    """
    # The nuisance arrays are all the same length by construction, so checking the
    # propensity against the sample checks all three.
    _check(outcome, treatment, treat, nuisances.propensity)
    treated = treatment == 1
    under_policy = np.where(treat, nuisances.mu1, nuisances.mu0)
    under_observed = np.where(treated, nuisances.mu1, nuisances.mu0)
    residual = np.where(
        _agrees(treatment, treat),
        (outcome - under_observed) / _assigned_probability(treat, nuisances.propensity),
        0.0,
    )
    return float((under_policy + residual).mean())


def ipw_gain(
    outcome: FloatArray,
    treatment: IntArray,
    treat: BoolArray,
    propensity: FloatArray,
) -> float:
    """Extra outcome per head of the population from running a policy rather than nothing.

    The difference of two :func:`ipw_value` calls. Every unit the policy leaves alone
    cancels exactly between them, so what survives is an estimate over the targeted set,
    and at a budget of 100% it is the Horvitz-Thompson average treatment effect.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        treat: The policy: True for units it would treat.
        propensity: Probability of treatment per unit.

    Returns:
        The gain against treating nobody, on the scale of the outcome per unit of
        population.
    """
    nobody: BoolArray = np.zeros(outcome.size, dtype=bool)
    return ipw_value(outcome, treatment, treat, propensity) - ipw_value(
        outcome, treatment, nobody, propensity
    )


def dr_gain(
    outcome: FloatArray,
    treatment: IntArray,
    treat: BoolArray,
    nuisances: Nuisances,
) -> float:
    """Extra outcome per head of the population, doubly robust.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        treat: The policy: True for units it would treat.
        nuisances: Propensity and the two outcome models, on these same rows.

    Returns:
        The gain against treating nobody.
    """
    nobody: BoolArray = np.zeros(outcome.size, dtype=bool)
    return dr_value(outcome, treatment, treat, nuisances) - dr_value(
        outcome, treatment, nobody, nuisances
    )


def ipw_contributions(
    outcome: FloatArray, treatment: IntArray, propensity: FloatArray
) -> FloatArray:
    """What treating each unit rather than leaving it alone adds to the IPW gain.

    The gain of any policy is the mean of these over the whole population, counting only
    the units the policy treats. That is what makes every budget cost one cumulative sum
    instead of a fresh pass (see :func:`policy_metrics`), and it is why the units a policy
    leaves alone cancel out of the gain exactly rather than approximately.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        propensity: Probability of treatment per unit.

    Returns:
        One contribution per unit. At most one of the two terms is ever non-zero for a
        given unit, because a unit was observed in one arm.

    Raises:
        ValueError: If the inputs are different lengths or a propensity is not a
            probability strictly between 0 and 1.
    """
    _check(outcome, treatment, np.zeros(outcome.size, dtype=bool), propensity)
    treated = treatment == 1
    under_treatment = np.where(treated, outcome / propensity, 0.0)
    under_control = np.where(~treated, outcome / (1.0 - propensity), 0.0)
    contributions: FloatArray = under_treatment - under_control
    return contributions


def dr_contributions(
    outcome: FloatArray, treatment: IntArray, nuisances: Nuisances
) -> FloatArray:
    """What treating each unit rather than leaving it alone adds to the doubly robust gain.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        nuisances: Propensity and the two outcome models, on these same rows.

    Returns:
        One contribution per unit: the outcome models' predicted difference, corrected by
        the residual of whichever arm the unit was actually observed in.

    Raises:
        ValueError: If the inputs are different lengths or a propensity is not a
            probability strictly between 0 and 1.
    """
    propensity = nuisances.propensity
    _check(outcome, treatment, np.zeros(outcome.size, dtype=bool), propensity)
    treated = treatment == 1
    under_treatment = nuisances.mu1 + np.where(
        treated, (outcome - nuisances.mu1) / propensity, 0.0
    )
    under_control = nuisances.mu0 + np.where(
        ~treated, (outcome - nuisances.mu0) / (1.0 - propensity), 0.0
    )
    contributions: FloatArray = under_treatment - under_control
    return contributions


def policy_metrics(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    nuisances: Nuisances,
    *,
    budgets: Sequence[float],
    seed: int = DEFAULT_TIE_SEED,
) -> dict[str, float]:
    """Both policy gains at every budget, from one sorted pass over the ranking.

    The arithmetic here is the same as calling :func:`ipw_gain` and :func:`dr_gain` once
    per budget, and the tests assert that it agrees with them to floating point. What it
    avoids is the cost. Each gain is the mean of a per-unit contribution over the units a
    policy treats, and the policies at 10%, 20% and 30% are nested prefixes of one ranking,
    so a single sort and a single cumulative sum answer every budget at once. Measured on
    137,000 rows at three budgets, against the 70 ms the ranking metrics cost on the same
    sample: calling the gain functions once per budget costs 98 ms and this costs 32 ms, so
    a bootstrap resample grows by 46% rather than 141%. Over a thousand resamples, six
    estimators and five seeds that is about two hours of Lenta.

    The treated set at budget b is the same set of people the curve at budget b describes,
    because both go through :func:`itx.metrics.curves.rank_order` with the same tie seed. A
    policy value that belonged to a slightly different set of people than the curve next to
    it would be a very hard bug to see.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Targeting scores, higher meaning treat sooner.
        nuisances: Propensity and outcome models for these rows.
        budgets: Shares of the population to report at.
        seed: Seed for tie-breaking.

    Returns:
        A ``gain_ipw@k`` and a ``gain_dr@k`` entry per budget.
    """
    n_units = outcome.size
    order = rank_order(scores, seed=seed)
    cumulative = {
        "ipw": np.cumsum(ipw_contributions(outcome, treatment, nuisances.propensity)[order]),
        "dr": np.cumsum(dr_contributions(outcome, treatment, nuisances)[order]),
    }

    metrics: dict[str, float] = {}
    for budget in budgets:
        cut = n_targeted(n_units, budget)
        for estimator, running in cumulative.items():
            metrics[gain_key(estimator, budget)] = float(running[cut - 1]) / n_units
    return metrics


def gain_key(estimator: str, budget: float) -> str:
    """Metric name for one estimator at one budget, for example ``gain_ipw@20%``.

    Args:
        estimator: ``ipw`` or ``dr``.
        budget: Share of the population.

    Returns:
        The metric name used in results files and tables.
    """
    return f"gain_{estimator}@{budget:.0%}"


def _agrees(treatment: IntArray, treat: BoolArray) -> BoolArray:
    """True where the arm a unit was observed in is the arm the policy would assign it."""
    agrees: BoolArray = (treatment == 1) == treat
    return agrees


def _assigned_probability(treat: BoolArray, propensity: FloatArray) -> FloatArray:
    """Probability each unit had of ending up in the arm the policy assigns it."""
    assigned: FloatArray = np.where(treat, propensity, 1.0 - propensity)
    return assigned


def _check(
    outcome: FloatArray,
    treatment: IntArray,
    treat: BoolArray,
    propensity: FloatArray,
) -> None:
    """Reject inputs that cannot describe the same units, or a propensity that is not one.

    A propensity of exactly 0 or 1 is rejected rather than clipped here. Clipping is a
    decision about how much extrapolation to tolerate and it belongs with the model that
    produced the number (:mod:`itx.estimators.propensity`), not silently inside a division.
    """
    sizes = {outcome.size, treatment.size, treat.size, propensity.size}
    if len(sizes) != 1:
        msg = (
            "outcome, treatment, policy and propensity must be the same length; got "
            f"{outcome.size}, {treatment.size}, {treat.size}, {propensity.size}"
        )
        raise ValueError(msg)
    if outcome.size == 0:
        msg = "cannot estimate a policy value on an empty sample"
        raise ValueError(msg)
    if not ((propensity > 0.0) & (propensity < 1.0)).all():
        msg = (
            "every propensity must be strictly between 0 and 1; a unit that could not have "
            f"been in one of the arms carries an infinite weight. Range is "
            f"{propensity.min():.4g} to {propensity.max():.4g}"
        )
        raise ValueError(msg)
