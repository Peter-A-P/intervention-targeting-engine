"""S-learner: one model, treatment as a feature.

The simplest thing that could work. Fit a single outcome model on the features plus the
treatment indicator, then predict each unit twice, once with the indicator set to 1 and
once to 0, and take the difference.

Its weakness is structural and worth stating up front, because PLAN.md section 9 expects
to be able to show it: the treatment indicator is one column among many, so a regularised
tree ensemble that finds the other columns more predictive will split on the indicator
rarely or never. When that happens the two predictions are nearly identical and the
predicted uplift collapses toward zero everywhere. That is not a tuning failure, it is
what a single shared model does to a small effect, and it is why the T-learner exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from itx.estimators.base import BaseUpliftEstimator, add_treatment_column
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig, OutcomeLearner

if TYPE_CHECKING:
    import polars as pl

    from itx.types import FloatArray, UpliftDataset

TREATMENT_COLUMN = "__treatment"


class SLearner(BaseUpliftEstimator):
    """Single-model meta-learner with the treatment indicator as a feature."""

    name = "s-learner"

    def __init__(self, config: BaseLearnerConfig = DEFAULT_CONFIG, *, seed: int = 0) -> None:
        """Build an unfitted S-learner.

        Args:
            config: Shared LightGBM settings.
            seed: Seed for the single outcome model.
        """
        super().__init__()
        self.config = config
        self.seed = seed
        self._learner: OutcomeLearner | None = None

    def _fit(self, data: UpliftDataset) -> None:
        """Fit one model on features plus treatment."""
        self._learner = OutcomeLearner(
            self.config, binary=data.outcome_is_binary, seed=self.seed
        )
        frame = add_treatment_column(data.features, data.treatment, TREATMENT_COLUMN)
        self._learner.fit(
            self._matrix(frame),
            data.outcome,
            categorical=self._categorical_indices(frame),
            feature_names=frame.columns,
        )

    def _predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """Predict with the indicator on and off, and difference the two."""
        if self._learner is None:  # pragma: no cover - guarded by _check_features
            msg = "s-learner: fit before predicting"
            raise RuntimeError(msg)
        ones = np.ones(features.height, dtype=np.float64)
        treated = self._matrix(add_treatment_column(features, ones, TREATMENT_COLUMN))
        control = self._matrix(add_treatment_column(features, ones * 0.0, TREATMENT_COLUMN))
        uplift: FloatArray = self._learner.predict(treated) - self._learner.predict(control)
        return uplift

    def treatment_split_share(self) -> float:
        """Share of the fitted trees' splits that used the treatment indicator.

        The diagnostic for this estimator's characteristic failure: a value at or near
        zero means the model never looked at the treatment and the predicted uplift is an
        artefact rather than an effect.

        Returns:
            Splits on the treatment column as a fraction of all splits, 0 if none.

        Raises:
            RuntimeError: If called before fitting.
        """
        if self._learner is None:
            msg = "s-learner: fit before asking for the split share"
            raise RuntimeError(msg)
        importance = self._learner.booster_.feature_importance(importance_type="split")
        total = float(np.sum(importance))
        if total == 0.0:
            return 0.0
        index = self._learner.feature_names.index(TREATMENT_COLUMN)
        return float(importance[index]) / total
