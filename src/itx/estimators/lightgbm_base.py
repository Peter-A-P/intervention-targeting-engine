"""The base learner, configured once.

Every meta-learner in this package is built out of ordinary outcome models, and within one
run they are all the same one: LightGBM, same hyperparameters, same seed discipline (PLAN.md
section 5). If the S-learner and the X-learner used different engines, the results table
would be comparing engines and the estimator column would mean nothing.

A second base exists, TabFM (:mod:`itx.estimators.tabfm_base`), and it is a whole second
column rather than a mix: an estimator handed a :class:`TabFMConfig` builds every one of its
internal models on TabFM, and :class:`OutcomeLearner` is where that choice is made. Nothing
outside the base-learner comparison uses it (PLAN.md change 65).

The settings are deliberately conservative rather than tuned. ``deterministic`` and
``force_row_wise`` are on because the project's claim is that a stranger can rerun the
benchmark and get the table back; a two percent speed-up is not worth a table that moves
between machines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from itx.estimators.tabfm_base import TabFMConfig, TabFMLearner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itx.types import FloatArray


@dataclass(frozen=True, slots=True)
class BaseLearnerConfig:
    """LightGBM settings shared by every estimator.

    Attributes:
        n_estimators: Boosting rounds. Fixed rather than early-stopped, so that two
            estimators fitted on the same split see the same amount of model capacity.
        learning_rate: Shrinkage per round.
        num_leaves: Tree width.
        max_depth: Tree depth, -1 for unlimited.
        min_child_samples: Minimum rows in a leaf. Left at LightGBM's own default, and
            the reason is a finding rather than a preference. Uplift is a difference of two
            noisy quantities, so thick leaves look like the safe choice, and 100 was the
            first value tried. On Hillstrom's 25,000 training rows it cost a little Qini.
            On IHDP's 448 it disabled the model completely: no split was ever made, the
            predicted uplift was exactly zero everywhere, and the PEHE came out identical
            to predicting nothing at all, with no error raised. One number that is merely
            cautious on a large dataset is fatal on a small one, which is why this belongs
            in the validation grid per dataset rather than in a constant.
        subsample: Row subsampling fraction, applied every ``subsample_freq`` rounds.
        subsample_freq: How often to resample rows.
        colsample_bytree: Column subsampling fraction.
        reg_lambda: L2 penalty on leaf weights.
        n_jobs: Threads. Eight, on a twelve-core laptop, leaving room for the machine to
            stay usable. It is not a modelling parameter and does not move a result:
            ``deterministic`` and ``force_row_wise`` below exist so that the same fit gives
            the same answer whatever the thread count. It does move ``fit_seconds``, so
            timings recorded before change 24 in PLAN.md are not comparable with later ones.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    n_jobs: int = 8

    def with_(self, **changes: Any) -> BaseLearnerConfig:  # noqa: ANN401
        """Return a copy with some settings replaced, for the validation grid.

        Args:
            **changes: Field names and values to override.

        Returns:
            A new config.
        """
        return replace(self, **changes)

    def to_kwargs(self, seed: int) -> dict[str, Any]:
        """LightGBM keyword arguments for a given seed.

        Args:
            seed: Seed for every source of randomness in the fit.

        Returns:
            Keyword arguments to pass to the LightGBM sklearn estimator.
        """
        return {
            **asdict(self),
            "random_state": seed,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }


DEFAULT_CONFIG = BaseLearnerConfig()

#: Either base learner's configuration. The type of the config is the choice of engine.
type LearnerConfig = BaseLearnerConfig | TabFMConfig


def base_name(config: LearnerConfig) -> str:
    """The engine a configuration builds on, as it is written in tables and file names.

    Args:
        config: A base-learner configuration.

    Returns:
        ``"lightgbm"`` or ``"tabfm"``.
    """
    return "tabfm" if isinstance(config, TabFMConfig) else "lightgbm"


def config_label(config: LearnerConfig) -> str:
    """A short description of a configuration, for tables and logs.

    Args:
        config: A base-learner configuration.

    Returns:
        ``"min_child_samples=20, num_leaves=31"`` for LightGBM, or the TabFM context cap and
        ensemble size.
    """
    if isinstance(config, TabFMConfig):
        return f"tabfm, context_rows={config.context_rows}, ensemble={config.ensemble}"
    return f"min_child_samples={config.min_child_samples}, num_leaves={config.num_leaves}"


def estimator_name(name: str, config: LearnerConfig) -> str:
    """An estimator's row name: unchanged on LightGBM, suffixed on any other base.

    Unchanged on LightGBM so that every results file written before the second base existed
    still names its rows correctly.

    Args:
        name: The estimator's own name, such as ``"s-learner"``.
        config: The base it was built on.

    Returns:
        ``"s-learner"`` or ``"s-learner-tabfm"``.
    """
    base = base_name(config)
    return name if base == "lightgbm" else f"{name}-{base}"


class OutcomeLearner:
    """One model of the outcome, regression or probability, on whichever base it is given.

    Binary outcomes are modelled as probabilities and continuous ones by regression, so a
    predicted uplift is always on the same scale as the outcome itself: a change in the
    probability of a visit, or a change in a spend amount.
    """

    def __init__(
        self,
        config: LearnerConfig = DEFAULT_CONFIG,
        *,
        binary: bool,
        seed: int = 0,
    ) -> None:
        """Build an unfitted learner.

        Args:
            config: Shared base-learner settings. A :class:`TabFMConfig` makes this a
                TabFM model; anything else is LightGBM.
            binary: True to model a probability, False to regress.
            seed: Seed for this model. Distinct models inside one estimator get distinct
                seeds so their errors are not identical by construction.
        """
        self.config = config
        self.binary = binary
        self.seed = seed
        self._model: LGBMClassifier | LGBMRegressor | None = None
        self._tabfm: TabFMLearner | None = None
        if isinstance(config, BaseLearnerConfig):
            kwargs = config.to_kwargs(seed)
            self._model = LGBMClassifier(**kwargs) if binary else LGBMRegressor(**kwargs)
        self._single_class: float | None = None
        self._fitted = False
        self.feature_names: tuple[str, ...] = ()
        self.n_rows = 0

    def fit(
        self,
        matrix: FloatArray,
        target: FloatArray,
        *,
        categorical: Sequence[int] = (),
        feature_names: Sequence[str] | None = None,
    ) -> None:
        """Fit the model.

        A target with only one distinct value is possible on a small split of a
        rare-event dataset; LightGBM's classifier will not fit one, so the constant is
        remembered and returned directly rather than failing a whole benchmark row.

        Args:
            matrix: Feature matrix.
            target: Outcome values.
            categorical: Column positions holding integer category codes.
            feature_names: Column names. Kept on this object for diagnostics rather than
                handed to LightGBM: naming the columns of a numpy matrix makes scikit-learn
                warn on every later prediction from an unnamed array, and the warning is
                noise on a benchmark that fits hundreds of models.
        """
        self.n_rows = int(matrix.shape[0])
        self.feature_names = tuple(feature_names) if feature_names is not None else ()
        if self.binary and len(np.unique(target)) < 2:
            self._single_class = float(target[0]) if target.size else 0.0
            return
        if isinstance(self.config, TabFMConfig):
            self._tabfm = TabFMLearner(
                self.config, binary=self.binary, seed=self.seed, categorical=categorical
            )
            self._tabfm.fit(matrix, target)
        elif self._model is not None:
            self._model.fit(matrix, target, categorical_feature=list(categorical))
        self._fitted = True

    def predict(self, matrix: FloatArray) -> FloatArray:
        """Predict the outcome, as a probability for binary outcomes.

        Args:
            matrix: Feature matrix.

        Returns:
            One prediction per row.
        """
        if self._single_class is not None:
            return np.full(matrix.shape[0], self._single_class, dtype=np.float64)
        if self._tabfm is not None:
            return self._tabfm.predict(matrix)
        if self._model is None:
            msg = "outcome learner: fit before predicting"
            raise RuntimeError(msg)
        if self.binary:
            probabilities: FloatArray = np.asarray(
                self._model.predict_proba(matrix), dtype=np.float64
            )[:, 1]
            return probabilities
        predictions: FloatArray = np.asarray(self._model.predict(matrix), dtype=np.float64)
        return predictions

    @property
    def booster_(self) -> Any:  # noqa: ANN401
        """The underlying LightGBM booster, for feature-importance inspection.

        Raises:
            TypeError: On a TabFM learner, which has no trees to inspect.
        """
        if self._model is None:
            msg = "only a LightGBM learner has a booster; TabFM has no trees to inspect"
            raise TypeError(msg)
        return self._model.booster_
