"""Synthetic data with a treatment effect the test suite already knows.

Every estimator is required to recover a known constant effect and a known heterogeneous
effect within tolerance (PLAN.md section 5). That is only possible against a generator
whose effect function is written down here rather than inferred, so this module is the
reference the estimator tests measure against, not a convenience.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from itx.types import FloatArray, IntArray, UpliftDataset

FEATURES = ("x0", "x1", "x2", "x3", "x4")


def constant_effect(
    n_units: int = 8_000,
    *,
    effect: float = 0.15,
    propensity: float = 0.5,
    seed: int = 0,
    noise: float = 0.5,
) -> UpliftDataset:
    """A continuous outcome whose treatment effect is the same for everyone.

    An estimator that cannot return a flat ``effect`` here has a bug rather than a
    disagreement: there is no heterogeneity to get wrong.

    Args:
        n_units: Rows to generate.
        effect: The true effect, identical for every unit.
        propensity: Probability of treatment, constant, so assignment is randomised.
        seed: Generator seed.
        noise: Standard deviation of the outcome noise.

    Returns:
        The dataset, with ``true_effect`` filled in and ``propensity`` known.
    """
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n_units, len(FEATURES)))
    treatment = rng.binomial(1, propensity, size=n_units).astype(np.int64)
    effects = np.full(n_units, effect)
    noise_draw = rng.normal(scale=noise, size=n_units)
    outcome = _baseline(covariates) + treatment * effects + noise_draw
    return _pack("synthetic-constant", covariates, treatment, outcome, effects, propensity)


def heterogeneous_effect(
    n_units: int = 8_000,
    *,
    propensity: float = 0.5,
    seed: int = 0,
    noise: float = 0.5,
) -> UpliftDataset:
    """A continuous outcome whose effect depends on two of the five covariates.

    The effect is strong for some units, near zero for others and negative for a third
    group, so a ranking has something real to find and a sleeping-dog region exists. The
    other three covariates do not enter the effect at all, and an estimator that ranks on
    them is overfitting.

    Args:
        n_units: Rows to generate.
        propensity: Probability of treatment, constant.
        seed: Generator seed.
        noise: Standard deviation of the outcome noise.

    Returns:
        The dataset, with ``true_effect`` filled in and ``propensity`` known.
    """
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n_units, len(FEATURES)))
    treatment = rng.binomial(1, propensity, size=n_units).astype(np.int64)
    effects = true_heterogeneous_effect(covariates)
    noise_draw = rng.normal(scale=noise, size=n_units)
    outcome = _baseline(covariates) + treatment * effects + noise_draw
    return _pack("synthetic-heterogeneous", covariates, treatment, outcome, effects, propensity)


def true_heterogeneous_effect(covariates: FloatArray) -> FloatArray:
    """The effect function :func:`heterogeneous_effect` uses, published so tests can check it.

    Args:
        covariates: An ``(n, 5)`` covariate matrix.

    Returns:
        The true per-unit effect: linear in the first covariate, hinged on the second.
    """
    hinge: FloatArray = covariates[:, 1] * (covariates[:, 1] > 0)
    return 0.5 * covariates[:, 0] + hinge


def binary_outcome(
    n_units: int = 20_000,
    *,
    propensity: float = 0.5,
    seed: int = 0,
) -> UpliftDataset:
    """A binary outcome with a known per-unit uplift, for the Qini and policy-value checks.

    Uplift on a probability scale is bounded by the baseline rate, so the generator works
    on probabilities directly: a baseline probability from the covariates, and a treated
    probability clipped into ``[0, 1]``. ``true_effect`` is the difference of the two
    probabilities, which is exactly what a Qini curve is trying to rank on. The offset
    makes part of the population genuine sleeping dogs, so a policy that treats everybody
    is measurably worse than one that treats the top of the list.

    Args:
        n_units: Rows to generate.
        propensity: Probability of treatment, constant.
        seed: Generator seed.

    Returns:
        The dataset, with ``true_effect`` filled in and ``propensity`` known.
    """
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n_units, len(FEATURES)))
    treatment = rng.binomial(1, propensity, size=n_units).astype(np.int64)
    control_probability = _sigmoid(-1.0 + 0.6 * covariates[:, 2] - 0.4 * covariates[:, 3])
    treated_probability = np.clip(
        control_probability + 0.25 * _sigmoid(2.0 * covariates[:, 0]) - 0.08, 0.0, 1.0
    )
    probability = np.where(treatment == 1, treated_probability, control_probability)
    outcome = rng.binomial(1, probability).astype(np.float64)
    return _pack(
        "synthetic-binary",
        covariates,
        treatment,
        outcome,
        treated_probability - control_probability,
        propensity,
    )


def _baseline(covariates: FloatArray) -> FloatArray:
    """Outcome level in the absence of treatment: nonlinear, so a learner has work to do."""
    baseline: FloatArray = (
        covariates[:, 0]
        + 0.5 * covariates[:, 1] ** 2
        - 0.75 * covariates[:, 2]
        + 0.25 * covariates[:, 3] * covariates[:, 4]
    )
    return baseline


def _sigmoid(values: FloatArray) -> FloatArray:
    """Logistic function."""
    transformed: FloatArray = 1.0 / (1.0 + np.exp(-values))
    return transformed


def _pack(
    name: str,
    covariates: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    effects: FloatArray,
    propensity: float,
) -> UpliftDataset:
    """Assemble the generated pieces into a dataset."""
    features = pl.DataFrame(
        {column: covariates[:, index] for index, column in enumerate(FEATURES)}
    )
    return UpliftDataset(
        name=name,
        features=features,
        treatment=treatment,
        outcome=outcome,
        true_effect=effects,
        propensity=np.full(covariates.shape[0], propensity),
    )


def complex_effect(
    n_units: int = 8_000,
    *,
    propensity: float = 0.5,
    seed: int = 0,
    noise: float = 0.5,
) -> UpliftDataset:
    """A simple baseline with a treatment effect more complicated than it.

    This generator exists to answer a question the other two cannot, and the reason is a
    limitation of theirs worth stating. In :func:`heterogeneous_effect` the baseline is a
    nonlinear function of four covariates and the effect is nearly linear in two, so the
    effect is the *simpler* of the two surfaces. That is exactly the regime an S-learner is
    built for: one shared model carries the complicated part, and a few splits on the
    treatment indicator carry the rest. Benchmarking meta-learners only on data shaped that
    way does not compare them, it tells the S-learner it was right.

    Here the arrangement is reversed. The baseline is one covariate, near-linear. The effect
    is an interaction between two others crossed with a hinge on a third, so representing it
    inside a shared model costs many more splits than representing it as the difference of
    two arm models. This is the regime the literature says the T-learner should win, and
    the benchmark runs both so the claim can be checked rather than repeated.

    Args:
        n_units: Rows to generate.
        propensity: Probability of treatment, constant.
        seed: Generator seed.
        noise: Standard deviation of the outcome noise.

    Returns:
        The dataset, with ``true_effect`` filled in and ``propensity`` known.
    """
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n_units, len(FEATURES)))
    treatment = rng.binomial(1, propensity, size=n_units).astype(np.int64)
    effects = true_complex_effect(covariates)
    baseline = 0.5 * covariates[:, 2]
    noise_draw = rng.normal(scale=noise, size=n_units)
    outcome = baseline + treatment * effects + noise_draw
    return _pack("synthetic-complex", covariates, treatment, outcome, effects, propensity)


def true_complex_effect(covariates: FloatArray) -> FloatArray:
    """The effect function used by :func:`complex_effect`, published so tests can check it.

    Args:
        covariates: An ``(n, 5)`` covariate matrix.

    Returns:
        The true per-unit effect: an interaction, gated by a hinge on a third covariate.
    """
    hinge: FloatArray = (covariates[:, 3] > 0.5).astype(np.float64)
    effect: FloatArray = covariates[:, 0] * covariates[:, 1] + 1.5 * hinge * covariates[:, 4]
    return effect
