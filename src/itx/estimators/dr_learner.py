"""DR-learner: the doubly robust one, wrapped from EconML.

The first estimator in this package with a real theoretical guarantee attached, and the
guarantee is worth stating precisely because it is routinely oversold.

It builds a pseudo-outcome for every unit that combines an outcome model with an
inverse-propensity correction, then regresses that pseudo-outcome on the covariates. The
doubly robust property is that the estimate stays consistent if **either** the outcome model
**or** the propensity model is right, not necessarily both. That is genuinely useful: it is
two chances instead of one.

What it is not is protection against both being wrong, which is the usual situation with
observational data and unmeasured confounding, and it is not a reason to skip the
sensitivity analysis in week 6. It also does nothing about a propensity that is merely
extreme rather than wrong: the correction divides by it, so a unit with a fitted probability
of 0.005 arrives carrying two hundred times the weight of a typical row, and the estimate
becomes a report on that one unit. EconML clips at ``min_propensity`` for exactly this
reason, and this wrapper reports the overlap separately (see :mod:`itx.estimators.propensity`)
so that the clipping shows up as a number rather than as silence.

Cross-fitting is on by default: the nuisance models are fitted on folds that exclude the
rows they later score, so the pseudo-outcome for a unit does not depend on a model that saw
that unit's own outcome. Without it, the regression in the final stage is fitting partly to
the nuisance models' overfitting, and the estimate is optimistic in a way no held-out test
split can detect, because the damage happened during fitting.

Missing values need a flag, and the flag is not the default. Of the six estimators here,
this is the only one that refuses a feature matrix containing NaN: EconML validates its
inputs with scikit-learn's finiteness check before any model sees them, so the fact that
LightGBM handles NaN natively never gets a chance to matter. Lenta has missing values in 150
of its 191 columns, which made this the difference between a results row and a blank one.
``allow_missing=True`` turns the check off for both X and W, and EconML then warns, once per
fold, that "causal identification strategy can be erroneous in the presence of missing
values". That warning is correct and worth reading rather than silencing on principle: if
whether a covariate is observed depends on the treatment, or on something that also drives
the outcome, then the missingness is itself a confounder and no amount of doubly robust
machinery fixes it. On a randomised dataset it does not bite, because assignment is
independent of the covariates and of their missingness pattern by construction, and Lenta and
Criteo are the only datasets here with missing values. The warning is suppressed at the call
site rather than globally, and this paragraph is the reason it is safe to do so.

Why wrapped rather than reimplemented: PLAN.md section 2. The S, T and X learners are
written out here because their mechanics are the thing worth seeing. The DR and R learners
have subtle cross-fitting and trimming details where a reference implementation that many
people have already stress-tested is worth more than a hand-rolled one.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from itx.estimators.base import BaseUpliftEstimator
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig
from itx.estimators.propensity import DEFAULT_CLIP, PropensityFit, PropensityModel
from itx.estimators.sklearn_bridge import base_learners

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl

    from itx.types import FloatArray, UpliftDataset

#: Cross-fitting folds. Two is EconML's default and is too few on a small dataset: each
#: nuisance model then sees only half the training rows. Five is the usual compromise.
DEFAULT_FOLDS = 5

INSTALL_HINT = (
    "the DR-learner needs EconML, which is an optional dependency: install it with "
    "'uv sync --extra learners'"
)


class DRLearner(BaseUpliftEstimator):
    """Doubly robust meta-learner, wrapping ``econml.dr.DRLearner``."""

    name = "dr-learner"

    def __init__(
        self,
        config: BaseLearnerConfig = DEFAULT_CONFIG,
        *,
        seed: int = 0,
        folds: int = DEFAULT_FOLDS,
        clip: float = DEFAULT_CLIP,
    ) -> None:
        """Build an unfitted DR-learner.

        Args:
            config: Shared LightGBM settings, used for all three nuisance models.
            seed: Base seed; the internal models get distinct seeds derived from it.
            folds: Cross-fitting folds.
            clip: Smallest propensity the correction is allowed to divide by. Passed to
                EconML as ``min_propensity`` and used for this wrapper's own overlap report,
                so both numbers come from the same bound.
        """
        super().__init__()
        self.config = config
        self.seed = seed
        self.folds = folds
        self.clip = clip
        self._model: Any = None
        self.propensity_fit: PropensityFit | None = None

    def _fit(self, data: UpliftDataset) -> None:
        """Fit EconML's DR-learner, and separately report the overlap it is working with."""
        try:
            from econml.dr import DRLearner as EconMLDRLearner
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise ImportError(INSTALL_HINT) from error

        categorical = self._categorical_indices(data.features)
        models = base_learners(self.config, self.seed, categorical)
        self._model = EconMLDRLearner(
            model_propensity=models["propensity"],
            model_regression=models["regression"],
            model_final=models["final"],
            cv=self._usable_folds(data),
            min_propensity=self.clip,
            random_state=self.seed,
            # See the module docstring. Without this, a NaN anywhere in the matrix is a
            # hard failure before any model runs, and LightGBM's own NaN handling never
            # gets used. The accompanying EconML warning is real but does not apply to a
            # randomised design, and it fires once per fold on every one of the hundreds of
            # fits a benchmark runs, so it is filtered here and explained there.
            allow_missing=True,
        )
        with _quiet_missing_value_warning():
            self._model.fit(data.outcome, data.treatment, X=self._matrix(data.features))

        # EconML clips internally and says nothing about it, so the overlap is measured
        # here on the same bound and reported alongside the fit. use_known is off on
        # purpose: EconML always fits its own propensity model, so a report based on a
        # design propensity the dataset happens to know would describe a fit that did not
        # happen. The number is meant to say what this estimator is actually working with.
        self.propensity_fit = PropensityModel(
            self.config, seed=self.seed + 1, clip=self.clip, use_known=False
        ).fit(data)

    def _predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """The estimated effect of moving from control to treatment."""
        if self._model is None:  # pragma: no cover - guarded by _check_features
            msg = "dr-learner: fit before predicting"
            raise RuntimeError(msg)
        with _quiet_missing_value_warning():
            effect: FloatArray = np.asarray(
                self._model.effect(self._matrix(features)), dtype=np.float64
            ).reshape(-1)
        return effect

    def _usable_folds(self, data: UpliftDataset) -> int:
        """Folds, reduced when an arm is too thin to appear in every one of them.

        Cross-fitting needs both arms present in every training fold. On a small dataset
        with a rare treatment, asking for five folds can leave a fold with no treated rows
        at all, and the failure surfaces from inside a library as something unhelpful.
        """
        smallest_arm = min(int(data.treatment.sum()), int((1 - data.treatment).sum()))
        return max(2, min(self.folds, smallest_arm))


@contextmanager
def _quiet_missing_value_warning() -> Iterator[None]:
    """Silence EconML's missing-value warning for the duration of one call.

    EconML re-validates its inputs on every ``fit`` fold and again on every ``effect``
    call, so the warning fires several times per fit and once per prediction: on a
    benchmark that fits hundreds of models it is thousands of identical lines.

    It is filtered here, narrowly, rather than globally, and the module docstring carries
    the argument for why it does not apply to the randomised datasets this package ships.
    Anything else EconML has to say still comes through.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Input contains NaN.*",
            category=UserWarning,
        )
        yield
