"""The two rankings every real estimator is shown next to.

Neither of these estimates a treatment effect. They are here because a Qini coefficient on
its own does not say whether a model found anything: it has to be read against what you
would have got by picking at random, and against what you would have got by doing the
normal thing, which is ranking people by how likely the outcome is. PLAN.md makes showing
both mandatory, and the outcome ranking is the project's headline trap.

Both implement the same protocol as the real estimators and go through the same metric
code, so nothing about the comparison is special-cased.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from itx.estimators.base import BaseUpliftEstimator
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig, OutcomeLearner

if TYPE_CHECKING:
    import polars as pl

    from itx.types import FloatArray, UpliftDataset

FitOn = Literal["all", "treated", "control"]


class RandomRanking(BaseUpliftEstimator):
    """Scores every unit at random. The floor any estimator has to clear.

    The scores are drawn from the seed rather than from the model, so two calls with the
    same feature matrix give the same ranking and a benchmark row is reproducible.
    """

    name = "random"
    estimates_effect = False

    def __init__(self, *, seed: int = 0) -> None:
        """Build the baseline.

        Args:
            seed: Seed for the score draw.
        """
        super().__init__()
        self.seed = seed

    def _fit(self, data: UpliftDataset) -> None:
        """Nothing to fit."""

    def _predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """Uniform random scores, reproducible for a given seed and row count."""
        rng = np.random.default_rng(self.seed)
        scores: FloatArray = rng.random(features.height)
        return scores


class OutcomeRanking(BaseUpliftEstimator):
    """Ranks by predicted outcome, ignoring the treatment entirely. The trap.

    This is what most targeting in the wild actually does: model who is likely to churn,
    respond, default or be fraudulent, and spend the budget from the top of that list. It
    answers a different question from the one the budget holder is paying for, and it can
    still score respectably on a Qini curve, which is why the results table reports
    realised policy value next to the curve.

    ``predict_uplift`` here returns a predicted outcome, not an effect. The abuse of the
    name is deliberate: this baseline has to travel through exactly the same ranking,
    metric and policy code as a real estimator, or the comparison would not be honest.
    """

    name = "outcome-ranking"
    #: The scores are predicted outcomes, so PEHE and ATE error are not computed for this
    #: baseline: they would compare an outcome level with an effect and report a number
    #: that looks like an error and is really a units mismatch. The trap this baseline
    #: exists to show is visible in the ranking metrics, where it belongs.
    estimates_effect = False

    def __init__(
        self,
        config: BaseLearnerConfig = DEFAULT_CONFIG,
        *,
        seed: int = 0,
        fit_on: FitOn = "all",
    ) -> None:
        """Build an unfitted outcome ranker.

        Args:
            config: Shared LightGBM settings, the same ones the real estimators use.
            seed: Seed for the outcome model.
            fit_on: Which rows to fit on. ``all`` ignores the treatment column, the
                commonest version of the mistake. ``treated`` is the marketing "response
                model", fitted only on people who got the campaign. ``control`` is the
                risk model fitted on the untreated, which is what a churn or fraud score
                usually is.
        """
        super().__init__()
        self.config = config
        self.seed = seed
        self.fit_on = fit_on
        self._learner: OutcomeLearner | None = None

    def _fit(self, data: UpliftDataset) -> None:
        """Fit a plain outcome model on the selected rows."""
        rows = _rows_for(data, self.fit_on)
        frame = data.features[rows]
        self._learner = OutcomeLearner(
            self.config, binary=data.outcome_is_binary, seed=self.seed
        )
        self._learner.fit(
            self._matrix(frame),
            data.outcome[rows],
            categorical=self._categorical_indices(frame),
            feature_names=frame.columns,
        )

    def _predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """The predicted outcome, used as a targeting score."""
        if self._learner is None:  # pragma: no cover - guarded by _check_features
            msg = "outcome-ranking: fit before predicting"
            raise RuntimeError(msg)
        return self._learner.predict(self._matrix(features))


def _rows_for(data: UpliftDataset, fit_on: FitOn) -> np.ndarray:
    """Row positions the outcome model is fitted on."""
    if fit_on == "all":
        return np.arange(data.n_units)
    wanted = 1 if fit_on == "treated" else 0
    rows = np.flatnonzero(data.treatment == wanted)
    if rows.size == 0:
        msg = f"outcome-ranking: no {fit_on} rows to fit on"
        raise ValueError(msg)
    return rows
