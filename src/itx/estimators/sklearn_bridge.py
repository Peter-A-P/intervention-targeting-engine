"""The base learner wearing a scikit-learn face, so EconML and CausalML get the same one.

Written for LightGBM, and the argument below is about LightGBM. The TabFM classes at the end
exist so that the DR and R learners can be built on the second base too, through
:func:`base_learners` and :func:`regressor_for`, which pick the engine from the type of the
configuration exactly as :class:`itx.estimators.lightgbm_base.OutcomeLearner` does.

The repository rule is that the base learner is LightGBM everywhere, so that differences in
the results table are differences between estimators rather than between their engines
(PLAN.md section 5). The DR and R learners are wrapped from EconML and CausalML, and both
libraries build their own models by cloning an estimator the caller hands in. Handing them a
bare ``LGBMRegressor`` would satisfy the rule only halfway: the model would be the same, but
the integer-coded categorical columns would arrive as plain numbers, because
``categorical_feature`` is a ``fit`` argument and nothing in either library knows to pass it.

Hillstrom's ``zip_code`` and ``channel`` would then be read as ordered quantities by the DR
and R learners and as categories by the S, T and X learners, and the estimator column in
the table would quietly be carrying an encoding difference. These two classes close that
gap: ordinary scikit-learn estimators that remember which columns are categories and
declare them on every fit, including every fit inside a cross-fitting loop.

They are deliberately thin. ``BaseEstimator`` reads the constructor signature to implement
``get_params`` and ``set_params``, which is what makes ``sklearn.clone`` work, so every
setting has to be a plain named argument stored unchanged on the instance. Doing anything
else in ``__init__`` is the classic way to produce an estimator that silently loses its
configuration the first time a library clones it.

The ``type: ignore`` comments below are the only ones in the package. scikit-learn does
not ship a ``py.typed`` marker, so under ``disallow_any_unimported`` mypy treats its base
classes as untyped and refuses the subclassing. The alternative, reimplementing
``get_params`` and ``set_params`` by hand to avoid inheriting, would trade two suppressions
for a real risk: scikit-learn decides what an estimator is from those base classes, and a
hand-rolled imitation is one release away from being wrong in a way nothing here would catch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig, LearnerConfig
from itx.estimators.tabfm_base import DEFAULT_TABFM, TabFMConfig, TabFMLearner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itx.types import FloatArray


class LightGBMRegressor(RegressorMixin, BaseEstimator):  # type: ignore[misc,no-any-unimported]
    """A LightGBM regressor that declares its categorical columns on every fit."""

    def __init__(
        self,
        config: BaseLearnerConfig = DEFAULT_CONFIG,
        seed: int = 0,
        categorical: Sequence[int] = (),
    ) -> None:
        """Store the settings unchanged, as scikit-learn requires.

        Args:
            config: Shared LightGBM settings.
            seed: Seed for this model.
            categorical: Column positions holding integer category codes.
        """
        self.config = config
        self.seed = seed
        self.categorical = categorical

    def fit(
        self, X: FloatArray, y: FloatArray, sample_weight: FloatArray | None = None
    ) -> Self:
        """Fit the underlying model.

        Args:
            X: Feature matrix.
            y: Target.
            sample_weight: Optional per-row weights, which EconML uses.

        Returns:
            Self, fitted.
        """
        self.model_ = LGBMRegressor(**self.config.to_kwargs(self.seed))
        self.model_.fit(
            X, y, sample_weight=sample_weight, categorical_feature=list(self.categorical)
        )
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        """Predict the target.

        Args:
            X: Feature matrix.

        Returns:
            One prediction per row.
        """
        predictions: FloatArray = np.asarray(self.model_.predict(X), dtype=np.float64)
        return predictions


class LightGBMClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc,no-any-unimported]
    """A LightGBM classifier that declares its categorical columns on every fit.

    Used for the propensity model inside the DR learner, and for binary outcomes.
    """

    def __init__(
        self,
        config: BaseLearnerConfig = DEFAULT_CONFIG,
        seed: int = 0,
        categorical: Sequence[int] = (),
    ) -> None:
        """Store the settings unchanged, as scikit-learn requires.

        Args:
            config: Shared LightGBM settings.
            seed: Seed for this model.
            categorical: Column positions holding integer category codes.
        """
        self.config = config
        self.seed = seed
        self.categorical = categorical

    def fit(
        self, X: FloatArray, y: FloatArray, sample_weight: FloatArray | None = None
    ) -> Self:
        """Fit the underlying model.

        A target with a single distinct value is possible inside a cross-fitting fold on a
        rare-event dataset. LightGBM will not fit one, so the constant is remembered and
        returned as a degenerate probability rather than failing the whole outer fit.

        Args:
            X: Feature matrix.
            y: Class labels.
            sample_weight: Optional per-row weights.

        Returns:
            Self, fitted.
        """
        self.classes_ = np.unique(y)
        self.single_class_ = self.classes_.size < 2
        if self.single_class_:
            return self
        self.model_ = LGBMClassifier(**self.config.to_kwargs(self.seed))
        self.model_.fit(
            X, y, sample_weight=sample_weight, categorical_feature=list(self.categorical)
        )
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            An ``(n, n_classes)`` array of probabilities.
        """
        rows = int(np.asarray(X).shape[0])
        if self.single_class_:
            return np.ones((rows, 1), dtype=np.float64)
        probabilities: FloatArray = np.asarray(self.model_.predict_proba(X), dtype=np.float64)
        return probabilities

    def predict(self, X: FloatArray) -> FloatArray:
        """Most likely class per row.

        Args:
            X: Feature matrix.

        Returns:
            One label per row.
        """
        rows = int(np.asarray(X).shape[0])
        if self.single_class_:
            return np.full(rows, self.classes_[0])
        labels: FloatArray = np.asarray(self.model_.predict(X))
        return labels


class TabFMRegressor(RegressorMixin, BaseEstimator):  # type: ignore[misc,no-any-unimported]
    """TabFM wearing the same scikit-learn face, for the second base learner.

    Sample weights are accepted because CausalML's R-learner passes them to its final
    stage; TabFM has no weight argument, so they are honoured by resampling the context
    (:func:`itx.estimators.tabfm_base.context_rows`).
    """

    def __init__(
        self,
        config: TabFMConfig = DEFAULT_TABFM,
        seed: int = 0,
        categorical: Sequence[int] = (),
    ) -> None:
        """Store the settings unchanged, as scikit-learn requires.

        Args:
            config: Shared TabFM settings.
            seed: Seed for this model.
            categorical: Column positions holding integer category codes.
        """
        self.config = config
        self.seed = seed
        self.categorical = categorical

    def fit(
        self, X: FloatArray, y: FloatArray, sample_weight: FloatArray | None = None
    ) -> Self:
        """Hand the context to TabFM.

        Args:
            X: Feature matrix.
            y: Target.
            sample_weight: Optional per-row weights, honoured by resampling.

        Returns:
            Self, fitted.
        """
        self.model_ = TabFMLearner(
            self.config, binary=False, seed=self.seed, categorical=self.categorical
        )
        self.model_.fit(
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
        )
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        """Predict the target.

        Args:
            X: Feature matrix.

        Returns:
            One prediction per row.
        """
        return self.model_.predict(np.asarray(X, dtype=np.float64))


class TabFMClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc,no-any-unimported]
    """TabFM as a binary classifier, for the propensity models on the second base."""

    def __init__(
        self,
        config: TabFMConfig = DEFAULT_TABFM,
        seed: int = 0,
        categorical: Sequence[int] = (),
    ) -> None:
        """Store the settings unchanged, as scikit-learn requires.

        Args:
            config: Shared TabFM settings.
            seed: Seed for this model.
            categorical: Column positions holding integer category codes.
        """
        self.config = config
        self.seed = seed
        self.categorical = categorical

    def fit(
        self, X: FloatArray, y: FloatArray, sample_weight: FloatArray | None = None
    ) -> Self:
        """Hand the context to TabFM.

        Args:
            X: Feature matrix.
            y: Class labels, 0 and 1.
            sample_weight: Optional per-row weights, honoured by resampling.

        Returns:
            Self, fitted.

        Raises:
            ValueError: If the labels are not binary.
        """
        self.classes_ = np.unique(y)
        if not set(self.classes_.tolist()) <= {0, 1}:
            msg = f"TabFM bridge classifier is binary only, got classes {self.classes_}"
            raise ValueError(msg)
        self.model_ = TabFMLearner(
            self.config, binary=True, seed=self.seed, categorical=self.categorical
        )
        self.model_.fit(
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
        )
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Class probabilities, one column per class seen in training.

        Args:
            X: Feature matrix.

        Returns:
            An ``(n, n_classes)`` array of probabilities.
        """
        positive = self.model_.predict(np.asarray(X, dtype=np.float64))
        if self.classes_.size < 2:
            return np.ones((positive.shape[0], 1), dtype=np.float64)
        probabilities: FloatArray = np.column_stack([1.0 - positive, positive])
        return probabilities

    def predict(self, X: FloatArray) -> FloatArray:
        """Most likely class per row.

        Args:
            X: Feature matrix.

        Returns:
            One label per row.
        """
        probabilities = self.predict_proba(X)
        labels: FloatArray = self.classes_[np.argmax(probabilities, axis=1)]
        return labels


def regressor_for(config: LearnerConfig, seed: int, categorical: Sequence[int]) -> Any:  # noqa: ANN401
    """A scikit-learn regressor on whichever base the configuration names.

    Args:
        config: Base-learner settings.
        seed: Seed for the model.
        categorical: Column positions holding integer category codes.

    Returns:
        An unfitted regressor.
    """
    if isinstance(config, TabFMConfig):
        return TabFMRegressor(config, seed, categorical)
    return LightGBMRegressor(config, seed, categorical)


def classifier_for(config: LearnerConfig, seed: int, categorical: Sequence[int]) -> Any:  # noqa: ANN401
    """A scikit-learn binary classifier on whichever base the configuration names.

    Args:
        config: Base-learner settings.
        seed: Seed for the model.
        categorical: Column positions holding integer category codes.

    Returns:
        An unfitted classifier.
    """
    if isinstance(config, TabFMConfig):
        return TabFMClassifier(config, seed, categorical)
    return LightGBMClassifier(config, seed, categorical)


def base_learners(
    config: LearnerConfig,
    seed: int,
    categorical: Sequence[int],
) -> dict[str, Any]:
    """The three models a wrapped estimator usually needs, built from one configuration.

    Args:
        config: Shared base-learner settings; the type picks the engine for all three.
        seed: Base seed; the three models get distinct seeds derived from it.
        categorical: Column positions holding integer category codes.

    Returns:
        A mapping with ``regression``, ``propensity`` and ``final`` models.
    """
    return {
        "regression": regressor_for(config, seed, categorical),
        "propensity": classifier_for(config, seed + 1, categorical),
        "final": regressor_for(config, seed + 2, categorical),
    }
