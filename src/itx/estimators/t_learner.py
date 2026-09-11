"""T-learner: one model per arm, differenced.

Fit a model of the outcome on the treated rows, another on the control rows, and predict
the difference. It is the obvious fix for the S-learner's characteristic failure: the two
arms cannot share a model, so the treatment can no longer be regularised away. Nothing in
this estimator is able to ignore the treatment, because the treatment decided which rows
each model saw.

It buys that with the opposite problem. The two models are fitted independently on
disjoint rows, so each brings its own error and the difference carries both. When one arm
is much smaller than the other, which is the normal case in every application this package
is aimed at, the model for the small arm is fitted on few rows, and its noise appears in
the predicted uplift as heterogeneity that is not there. The S-learner shrinks a real
effect toward zero; the T-learner invents structure in the noise. Neither is safe, which is
why the X-learner exists and why both are in the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from itx.estimators.base import BaseUpliftEstimator
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig, OutcomeLearner

if TYPE_CHECKING:
    import polars as pl

    from itx.types import FloatArray, UpliftDataset


class TLearner(BaseUpliftEstimator):
    """Two-model meta-learner: one outcome model per treatment arm."""

    name = "t-learner"

    def __init__(self, config: BaseLearnerConfig = DEFAULT_CONFIG, *, seed: int = 0) -> None:
        """Build an unfitted T-learner.

        Args:
            config: Shared LightGBM settings.
            seed: Seed for the arm models. The two get different seeds, derived from this
                one, so their errors are not identical by construction: two models fitted
                with the same seed on disjoint rows would still share the same column and
                row subsampling pattern, and differencing them would cancel part of the
                noise in a way that flatters the estimator.
        """
        super().__init__()
        self.config = config
        self.seed = seed
        self._treated: OutcomeLearner | None = None
        self._control: OutcomeLearner | None = None

    def _fit(self, data: UpliftDataset) -> None:
        """Fit one model on the treated rows and one on the control rows."""
        binary = data.outcome_is_binary
        categorical = self._categorical_indices(data.features)
        names = data.features.columns

        for label, wanted, offset in (("treated", 1, 0), ("control", 0, 1)):
            rows = np.flatnonzero(data.treatment == wanted)
            if rows.size == 0:
                msg = f"t-learner: no {label} rows to fit on"
                raise ValueError(msg)
            learner = OutcomeLearner(self.config, binary=binary, seed=self.seed + offset)
            learner.fit(
                self._matrix(data.features[rows]),
                data.outcome[rows],
                categorical=categorical,
                feature_names=names,
            )
            if label == "treated":
                self._treated = learner
            else:
                self._control = learner

    def _predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """The difference between the two arms' predicted outcomes."""
        if self._treated is None or self._control is None:  # pragma: no cover
            msg = "t-learner: fit before predicting"
            raise RuntimeError(msg)
        matrix = self._matrix(features)
        uplift: FloatArray = self._treated.predict(matrix) - self._control.predict(matrix)
        return uplift

    def arm_sizes(self) -> tuple[int, int]:
        """Rows each arm's model was fitted on, treated first.

        The diagnostic for this estimator's characteristic failure. When the two are very
        unequal, the predicted heterogeneity is partly the small arm's sampling noise, and
        no amount of tuning removes that: it is what fitting the arms separately costs.

        Returns:
            Treated and control row counts.

        Raises:
            RuntimeError: If called before fitting.
        """
        if self._treated is None or self._control is None:
            msg = "t-learner: fit before asking for the arm sizes"
            raise RuntimeError(msg)
        return self._treated.n_rows, self._control.n_rows
