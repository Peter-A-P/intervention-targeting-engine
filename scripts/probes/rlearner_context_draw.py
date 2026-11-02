"""Why the R-learner on TabFM gives different answers on a CPU and a GPU, on ACIC.

The evidence behind "What ACIC says" in docs/estimators.md. Fits the R-learner on TabFM
(CPU, float32, the committed settings) for one split seed, capturing the inputs of
CausalML's final stage: the matrix, the target (y - m) / (t - e) and the weight (t - e)^2.
The final stage is then refitted from those inputs under each variant below, and each
variant's PEHE on the test rows is printed. Every perturbation multiplies by (1 + 0.004z),
z standard normal, because 0.004 is bfloat16's rounding unit (2^-8): the size of difference
a GPU run makes.

- unchanged: must reproduce the full fit, which shows the CPU run is deterministic;
- targets perturbed, then also winsorised at their 1st and 99th percentiles;
- the outcome model's prediction m perturbed before the division;
- the propensity e perturbed, which also changes the weights;
- nothing perturbed, only the seed of TabFM's weighted context draw changed.

Run from the repository root with the 'learners' and 'foundation' extras installed:

    uv run python scripts/probes/rlearner_context_draw.py 53

About 55 minutes on an 8-thread laptop CPU: one full fit, then thirteen final-stage refits.
"""

from __future__ import annotations

import copy
import sys
from typing import Any

import numpy as np
from causalml.inference import meta

from itx.data.acic import load_acic
from itx.data.splits import stratified_split
from itx.estimators.r_learner import RLearner
from itx.estimators.tabfm_base import DEFAULT_TABFM
from itx.metrics.ground_truth import pehe

#: bfloat16's rounding unit, the relative size of every perturbation.
ROUNDING = 0.004

captured: dict[str, Any] = {}


class Capture:
    """Stands in for CausalML's final-stage model and keeps a copy of what it was given."""

    def __init__(self, inner: Any) -> None:  # noqa: ANN401 - any scikit-learn regressor
        self.inner = inner

    def fit(self, matrix: Any, target: Any, sample_weight: Any = None) -> Capture:  # noqa: ANN401
        """Record the inputs and an unfitted copy of the model, then fit it."""
        captured["model"] = copy.deepcopy(self.inner)
        captured["matrix"], captured["target"] = matrix, np.asarray(target)
        captured["weight"] = np.asarray(sample_weight)
        self.inner.fit(matrix, target, sample_weight=sample_weight)
        return self

    def predict(self, matrix: Any) -> Any:  # noqa: ANN401
        """Record the test matrix and predict."""
        captured["test_matrix"] = matrix
        return self.inner.predict(matrix)


class Hooked(meta.BaseRRegressor):  # type: ignore[misc,no-any-unimported]
    """CausalML's R-learner with :class:`Capture` as its final-stage model."""

    def __init__(self, learner: Any = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(
            learner=learner, effect_learner=Capture(copy.deepcopy(learner)), **kwargs
        )


def refit(target: np.ndarray, weight: np.ndarray, truth: np.ndarray, seed: int | None) -> float:
    """PEHE of the final stage refitted on these targets and weights.

    Args:
        target: Final-stage regression targets.
        weight: Final-stage sample weights, which TabFM turns into a weighted context draw.
        truth: True effect on the test rows.
        seed: A new seed for the model's context draw, or None to keep the original.

    Returns:
        PEHE on the test rows.
    """
    model = copy.deepcopy(captured["model"])
    if seed is not None:
        model.seed = seed
    model.fit(captured["matrix"], target, sample_weight=weight)
    predicted = np.asarray(model.predict(captured["test_matrix"])).reshape(-1)
    return pehe(predicted, truth)


def main(seed: int) -> None:
    """Fit once, capture the final stage, and refit it under every variant."""
    meta.BaseRRegressor = Hooked
    split = stratified_split(load_acic(), seed)
    truth = split.test.require_true_effect()
    model = RLearner(DEFAULT_TABFM, seed=seed).fit(split.train)
    full = pehe(model.predict_uplift(split.test.features), truth)
    print(f"seed {seed}, full fit: PEHE {full:.4f}")
    if model.propensity_fit is None:  # pragma: no cover - set by fit
        raise RuntimeError("the R-learner did not record its propensity")

    target, weight = captured["target"], captured["weight"]
    t = np.asarray(split.train.treatment, dtype=np.float64)
    y = np.asarray(split.train.outcome, dtype=np.float64)
    e = model.propensity_fit.values
    m = y - target * (t - e)
    low, high = np.quantile(target, [0.01, 0.99])
    print(
        f"treatment residual |t - e| at the 1st, 10th and 50th percentiles: "
        f"{np.quantile(np.sqrt(weight), [0.01, 0.1, 0.5]).round(3).tolist()}; "
        f"|target| median {np.median(np.abs(target)):.1f}, max {np.abs(target).max():.1f}"
    )

    def noise(draw: int) -> np.ndarray:
        factor: np.ndarray = 1 + ROUNDING * np.random.default_rng(draw).standard_normal(
            target.shape
        )
        return factor

    variants: dict[str, tuple[np.ndarray, np.ndarray, int | None]] = {
        "unchanged": (target, weight, None)
    }
    for draw in (1, 2):
        variants[f"targets perturbed, draw {draw}"] = (target * noise(draw), weight, None)
        variants[f"targets perturbed and winsorised, draw {draw}"] = (
            np.clip(target * noise(draw), low, high),
            weight,
            None,
        )
        variants[f"m perturbed, draw {draw}"] = ((y - m * noise(draw)) / (t - e), weight, None)
    for draw in (1, 2, 3):
        shifted = np.clip(e * noise(draw), 0.01, 0.99)
        variants[f"e perturbed, draw {draw}"] = (
            (y - m) / (t - shifted),
            (t - shifted) ** 2,
            None,
        )
    for new_seed in (1001, 1002, 1003):
        variants[f"only the context draw's seed, {new_seed}"] = (target, weight, new_seed)
    for name, (variant_target, variant_weight, draw_seed) in variants.items():
        print(
            f"  {name}: PEHE {refit(variant_target, variant_weight, truth, draw_seed):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 53)
