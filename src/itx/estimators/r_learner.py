"""R-learner: Robinson's residual-on-residual trick, wrapped from CausalML.

Nie and Wager (2021), built on Robinson (1988). The idea is the most elegant one in the
meta-learner family and it is worth stating properly, because the name makes it sound like
a variant of the others and it is not.

Fit two nuisance models: ``m(x)``, the expected outcome ignoring treatment, and ``e(x)``,
the propensity. Then form two residuals per unit, ``y - m(x)`` and ``t - e(x)``, and
estimate the effect by regressing the first on the second, weighted by the square of the
treatment residual. Everything the covariates can explain about the outcome, and everything
they can explain about who got treated, has been subtracted out before the effect is looked
at; what is left is the part of the outcome that moves with the part of the treatment that
was not predictable. The confounding is removed by partialling out rather than by modelling
around it.

The weighting is the part people skip. A unit whose treatment was almost perfectly
predictable has a treatment residual near zero, and it contributes almost nothing to the
estimate, which is right: it carries almost no information about what treatment does,
because it was never really in question. The R-learner does not need to clip such a unit,
it simply stops listening to it, and that is a cleaner answer to poor overlap than a bound.

Where it breaks: it is the most sensitive of the five to a badly fitted ``m(x)``. If the
outcome model leaves structure in its residuals, that structure is exactly what the second
stage attributes to the treatment. It also gives up on the small-sample regime that the
X-learner was built for, because the residuals themselves have to be estimated well before
anything downstream is meaningful, and on a few hundred rows they are not.

Wrapped rather than reimplemented for the same reason as the DR-learner (PLAN.md section 2):
the cross-fitting details are where the mistakes live.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from itx.estimators.base import BaseUpliftEstimator
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig
from itx.estimators.propensity import DEFAULT_CLIP, PropensityFit, PropensityModel
from itx.estimators.sklearn_bridge import LightGBMRegressor

if TYPE_CHECKING:
    import polars as pl

    from itx.types import FloatArray, UpliftDataset

DEFAULT_FOLDS = 5

#: CausalML labels arms by name and needs to be told which name is the control.
CONTROL_NAME = "control"
TREATMENT_NAME = "treatment"

INSTALL_HINT = (
    "the R-learner needs CausalML, which is an optional dependency: install it with "
    "'uv sync --extra learners'"
)


class RLearner(BaseUpliftEstimator):
    """Residual-on-residual meta-learner.

    Wraps ``causalml.inference.meta.BaseRRegressor``.
    """

    name = "r-learner"

    def __init__(
        self,
        config: BaseLearnerConfig = DEFAULT_CONFIG,
        *,
        seed: int = 0,
        folds: int = DEFAULT_FOLDS,
        clip: float = DEFAULT_CLIP,
        use_known_propensity: bool = True,
    ) -> None:
        """Build an unfitted R-learner.

        Args:
            config: Shared LightGBM settings.
            seed: Base seed; the internal models get distinct seeds derived from it.
            folds: Cross-fitting folds.
            clip: Propensity clipping bound for the propensity this wrapper supplies.
            use_known_propensity: Hand CausalML the dataset's design propensity when it has
                one. On a randomised experiment that is a constant, and estimating it can
                only add variance to the treatment residual.
        """
        super().__init__()
        self.config = config
        self.seed = seed
        self.folds = folds
        self.clip = clip
        self.use_known_propensity = use_known_propensity
        self._model: Any = None
        self._propensity = PropensityModel(
            config, seed=seed + 1, clip=clip, use_known=use_known_propensity
        )
        self.propensity_fit: PropensityFit | None = None

    def _fit(self, data: UpliftDataset) -> None:
        """Fit the propensity, then hand it and the data to CausalML."""
        try:
            from causalml.inference.meta import BaseRRegressor
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise ImportError(INSTALL_HINT) from error

        categorical = self._categorical_indices(data.features)
        self.propensity_fit = self._propensity.fit(data)

        self._model = BaseRRegressor(
            learner=LightGBMRegressor(self.config, self.seed, categorical),
            n_fold=self._usable_folds(data),
            random_state=self.seed,
            control_name=CONTROL_NAME,
        )
        self._model.fit(
            X=self._matrix(data.features),
            treatment=self._arm_labels(data),
            y=data.outcome,
            p=self.propensity_fit.values,
        )

    def _predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """The estimated effect of moving from control to treatment."""
        if self._model is None:  # pragma: no cover - guarded by _check_features
            msg = "r-learner: fit before predicting"
            raise RuntimeError(msg)
        effect: FloatArray = np.asarray(
            self._model.predict(self._matrix(features)), dtype=np.float64
        ).reshape(-1)
        return effect

    @staticmethod
    def _arm_labels(data: UpliftDataset) -> np.ndarray:
        """CausalML wants arm names rather than a 0/1 indicator."""
        return np.where(data.treatment == 1, TREATMENT_NAME, CONTROL_NAME)

    def _usable_folds(self, data: UpliftDataset) -> int:
        """Folds, reduced when an arm is too thin to appear in every one of them."""
        smallest_arm = min(int(data.treatment.sum()), int((1 - data.treatment).sum()))
        return max(2, min(self.folds, smallest_arm))
