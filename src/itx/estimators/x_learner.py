"""X-learner: impute the missing half of the data, then model the effect directly.

Künzel, Sekhon, Bickel and Yu (2019). It is the first estimator in this package that is
built for the situation every application here actually has: one arm much smaller than the
other.

Three stages.

1. **Outcome models per arm**, exactly the T-learner's two models: ``mu1`` on the treated
   rows, ``mu0`` on the control rows.
2. **Imputed effects.** For a treated unit the observed outcome is its treated outcome, so
   its effect is imputed as ``y - mu0(x)``: what happened, minus what the control model
   says would have happened. For a control unit it is ``mu1(x) - y``. Each unit now has
   one number that is an estimate of its own effect, so a model can be fitted to the
   effect itself rather than to the outcome. Two are: ``tau1`` on the treated rows, ``tau0``
   on the control rows.
3. **Combine, weighted by the propensity.** ``tau(x) = e(x) * tau0(x) + (1 - e(x)) *
   tau1(x)``.

That weighting is the whole idea and it is worth being explicit about, because it looks
backwards at first glance. The model fitted on the *treated* rows gets the weight
``1 - e(x)``, which is small exactly where treatment was common. It is right: where almost
everyone was treated there are few control units, so ``mu0`` is the weak model, so
``tau1``, which leans on ``mu0``, is the noisy one, and it should be down-weighted in
favour of ``tau0``, which leans on the well-estimated ``mu1``. Each imputed effect is
trusted in proportion to how well-estimated the counterfactual it borrowed actually is.

Where it breaks: the imputation inherits every bias in the arm models. If ``mu0`` is
systematically wrong in some region of the covariate space, every treated unit there gets
an imputed effect that is wrong by the same amount, and stage 2 fits a smooth, confident
model to that error. The X-learner does not detect this; it is only ever as good as the
counterfactual it borrowed. It also does nothing about confounding on its own, which is
what the DR and R learners are for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from itx.estimators.base import BaseUpliftEstimator
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig, OutcomeLearner
from itx.estimators.propensity import DEFAULT_CLIP, PropensityFit, PropensityModel

if TYPE_CHECKING:
    import polars as pl

    from itx.types import FloatArray, UpliftDataset


class XLearner(BaseUpliftEstimator):
    """Three-stage meta-learner with propensity-weighted imputed effects."""

    name = "x-learner"

    def __init__(
        self,
        config: BaseLearnerConfig = DEFAULT_CONFIG,
        *,
        seed: int = 0,
        clip: float = DEFAULT_CLIP,
        use_known_propensity: bool = True,
    ) -> None:
        """Build an unfitted X-learner.

        Args:
            config: Shared LightGBM settings.
            seed: Base seed. The five models inside get distinct seeds derived from it.
            clip: Propensity clipping bound.
            use_known_propensity: Use the dataset's design propensity when it has one.
        """
        super().__init__()
        self.config = config
        self.seed = seed
        self._propensity = PropensityModel(
            config, seed=seed + 4, clip=clip, use_known=use_known_propensity
        )
        self._mu1: OutcomeLearner | None = None
        self._mu0: OutcomeLearner | None = None
        self._tau1: OutcomeLearner | None = None
        self._tau0: OutcomeLearner | None = None
        self.propensity_fit: PropensityFit | None = None

    def _fit(self, data: UpliftDataset) -> None:
        """Fit the two arm models, the two effect models, and the propensity."""
        treated = np.flatnonzero(data.treatment == 1)
        control = np.flatnonzero(data.treatment == 0)
        for label, rows in (("treated", treated), ("control", control)):
            if rows.size == 0:
                msg = f"x-learner: no {label} rows to fit on"
                raise ValueError(msg)

        binary = data.outcome_is_binary
        categorical = self._categorical_indices(data.features)
        names = data.features.columns
        matrix = self._matrix(data.features)

        # Stage 1: the outcome in each arm.
        self._mu1 = self._fit_learner(
            matrix[treated], data.outcome[treated], binary, categorical, names, self.seed
        )
        self._mu0 = self._fit_learner(
            matrix[control], data.outcome[control], binary, categorical, names, self.seed + 1
        )

        # Stage 2: the imputed effect, regressed on the covariates. These targets are
        # differences and can be negative even when the outcome is binary, so the effect
        # models are always regressions regardless of the outcome type.
        imputed_treated = data.outcome[treated] - self._mu0.predict(matrix[treated])
        imputed_control = self._mu1.predict(matrix[control]) - data.outcome[control]
        self._tau1 = self._fit_learner(
            matrix[treated], imputed_treated, False, categorical, names, self.seed + 2
        )
        self._tau0 = self._fit_learner(
            matrix[control], imputed_control, False, categorical, names, self.seed + 3
        )

        self.propensity_fit = self._propensity.fit(data)

    def _predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """Combine the two effect models, weighted by the propensity."""
        if self._tau1 is None or self._tau0 is None:  # pragma: no cover
            msg = "x-learner: fit before predicting"
            raise RuntimeError(msg)
        matrix = self._matrix(features)
        weight = self._propensity.predict(features)
        uplift: FloatArray = weight * self._tau0.predict(matrix) + (1.0 - weight) * (
            self._tau1.predict(matrix)
        )
        return uplift

    def _fit_learner(
        self,
        matrix: FloatArray,
        target: FloatArray,
        binary: bool,
        categorical: list[int],
        names: list[str],
        seed: int,
    ) -> OutcomeLearner:
        """Fit one of the five internal models."""
        learner = OutcomeLearner(self.config, binary=binary, seed=seed)
        learner.fit(matrix, target, categorical=categorical, feature_names=names)
        return learner

    def stage_two_targets(self, data: UpliftDataset) -> tuple[FloatArray, FloatArray]:
        """The imputed effects this estimator fitted its second stage on, treated first.

        Exposed because it is the part worth looking at when the X-learner disagrees with
        the T-learner: these are the numbers stage 2 believes, and if they are implausible
        the arm models are wrong rather than the combination rule.

        Args:
            data: The dataset to impute for.

        Returns:
            Imputed effects for the treated rows and for the control rows.

        Raises:
            RuntimeError: If called before fitting.
        """
        if self._mu1 is None or self._mu0 is None:
            msg = "x-learner: fit before asking for the stage two targets"
            raise RuntimeError(msg)
        matrix = self._matrix(data.features)
        treated = np.flatnonzero(data.treatment == 1)
        control = np.flatnonzero(data.treatment == 0)
        return (
            data.outcome[treated] - self._mu0.predict(matrix[treated]),
            self._mu1.predict(matrix[control]) - data.outcome[control],
        )
