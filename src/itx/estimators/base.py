"""The one protocol every estimator implements, and the boundary to the base learner.

Every estimator in this package, including the two baselines that exist to be beaten,
answers the same three questions: fit on a dataset, score a feature matrix by predicted
uplift, and turn that score into a treat/do-not-treat decision at a budget. One protocol
is what makes the results table a comparison rather than a list.

PLAN.md section 2 wrote this protocol as ``fit(X, treatment, outcome)``. It takes an
:class:`~itx.types.UpliftDataset` instead, because which feature columns are
integer-encoded categories has to travel with the matrix: with three loose arguments,
every estimator re-derives that from the dtypes and the LightGBM calls drift apart. The
plan was updated in the commit that added this file.
"""

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from itx.types import BoolArray, FloatArray, UpliftDataset


@runtime_checkable
class UpliftEstimator(Protocol):
    """What a metric, a policy or a benchmark row is allowed to assume about an estimator."""

    #: Short identifier used in result tables and file names.
    name: str

    #: Whether ``predict_uplift`` returns an estimate of the treatment effect, on the
    #: scale of the outcome, or merely a score to rank by. The baselines return a score:
    #: an outcome probability, or a random number. It matters because PEHE and ATE error
    #: compare a predicted effect with a true effect, and running a risk score through
    #: them produces a number with no meaning rather than a bad score.
    estimates_effect: bool

    def fit(self, data: UpliftDataset) -> Self:
        """Fit on a dataset and return self."""
        ...

    def predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """Predicted effect of treatment on the outcome, one number per row."""
        ...

    def policy(self, features: pl.DataFrame, budget: float) -> BoolArray:
        """Who to treat when the budget covers ``budget`` of the population."""
        ...


class NotFittedError(RuntimeError):
    """An estimator was asked to predict before it was fitted."""


class DegenerateFitError(RuntimeError):
    """An estimator produced non-finite scores on the split it is being measured on.

    A warning was not enough. Change 51 found Dragonnet returning all-NaN scores on Lenta,
    which sorted to one end, made the ranking row order, and reported numbers that matched
    random targeting for a week. The benchmark now stops on the first non-finite score
    rather than writing a row that looks like a measurement (PLAN.md change 56). Fits are
    checkpointed, so stopping costs the one fit.
    """


class DegenerateFitWarning(UserWarning):
    """A fitted estimator predicts no uplift at all, for anyone.

    Worth a warning rather than a silent row in the table. A model that returns a constant
    zero is not a model that found no effect: it is usually a model that was never able to
    look, and it will report a PEHE identical to predicting nothing while a reader assumes
    an estimator ran. It happens when regularisation is too strong for the training set, so
    the tree never splits on the treatment at all.
    """


class BaseUpliftEstimator(abc.ABC):
    """Shared machinery: fitted-state tracking, feature-order checks, rank-and-cut policy.

    Subclasses implement :meth:`_fit` and :meth:`_predict_uplift` and get the rest.
    """

    name: str = "base"
    estimates_effect: bool = True

    def __init__(self) -> None:
        self._feature_names: tuple[str, ...] | None = None
        self._categorical: tuple[str, ...] = ()
        self._outcome_is_binary: bool = False

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has completed."""
        return self._feature_names is not None

    def fit(self, data: UpliftDataset) -> Self:
        """Fit on a dataset.

        Args:
            data: Features, binary treatment and observed outcome.

        Returns:
            Self, fitted.
        """
        self._feature_names = tuple(data.feature_names)
        self._categorical = data.categorical
        self._outcome_is_binary = data.outcome_is_binary
        self._fit(data)
        self._warn_if_degenerate(data)
        return self

    def predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """Predict the per-unit effect of treatment.

        Args:
            features: A feature matrix with the same columns, in the same order, as the
                one this estimator was fitted on.

        Returns:
            One predicted effect per row.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            ValueError: If the columns do not match the fitted ones.
        """
        self._check_features(features)
        return self._predict_uplift(features)

    def policy(self, features: pl.DataFrame, budget: float) -> BoolArray:
        """Decide who to treat under a budget, by rank and cut.

        Args:
            features: Feature matrix to decide over.
            budget: Share of the population the budget covers, in ``(0, 1]``.

        Returns:
            A boolean mask, True for the units to treat.
        """
        from itx.policy.rank_and_cut import rank_and_cut

        return rank_and_cut(self.predict_uplift(features), budget)

    @abc.abstractmethod
    def _fit(self, data: UpliftDataset) -> None:
        """Estimator-specific fitting."""

    @abc.abstractmethod
    def _predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """Estimator-specific prediction, called only after the feature check."""

    def _warn_if_degenerate(self, data: UpliftDataset) -> None:
        """Warn when the fitted model predicts nothing usable for anyone.

        Two separate failures, and the second was added in week 6 after it cost a result.
        A model that returns a constant zero was the case this was built for. A model that
        returns NaN is worse and slipped past it, because ``np.allclose(nan, 0.0)`` is
        False: Dragonnet on Lenta produced an all-NaN prediction, ``rank_order`` fell back
        to row order, and a broken fit reported a Qini and a policy gain indistinguishable
        from random targeting rather than reporting nothing. Plausible numbers from a dead
        model are worse than an error, so this checks for them by name.
        """
        predictions = self._predict_uplift(data.features)
        if predictions.size and not np.isfinite(predictions).all():
            warnings.warn(
                f"{self.name}: predicted uplift is not finite for "
                f"{int((~np.isfinite(predictions)).sum()):,} of {predictions.size:,} units "
                f"on {data.n_units:,} training rows. Every metric computed from this ranking "
                f"is meaningless: a non-finite score sorts to one end, so the ranking becomes "
                f"the order the rows happened to arrive in. Check the feature matrix for "
                f"missing values, which the tree learners accept and a network does not.",
                DegenerateFitWarning,
                stacklevel=3,
            )
            return
        if predictions.size and np.allclose(predictions, 0.0):
            warnings.warn(
                f"{self.name}: predicted uplift is zero for every unit on "
                f"{data.n_units:,} training rows. The model never separated the arms; "
                f"check min_child_samples and the size of the treated group before "
                f"reading anything into its metrics.",
                DegenerateFitWarning,
                stacklevel=3,
            )

    def _check_features(self, features: pl.DataFrame) -> None:
        """Fail loudly on a column mismatch rather than silently scoring the wrong matrix."""
        if self._feature_names is None:
            msg = f"{self.name}: fit before predicting"
            raise NotFittedError(msg)
        if tuple(features.columns) != self._feature_names:
            msg = (
                f"{self.name}: feature columns do not match the fitted ones.\n"
                f"  fitted:   {list(self._feature_names)}\n"
                f"  received: {features.columns}"
            )
            raise ValueError(msg)

    def _matrix(self, features: pl.DataFrame) -> FloatArray:
        """Cross into numpy for LightGBM.

        The frame is already all-numeric, with categories held as integer codes, so this
        is a view-shaped conversion rather than an encoding step. Codes plus an explicit
        list of which columns are categorical is the one encoding that cannot shift when a
        category happens to be absent from a prediction batch, which is why the loaders
        encode rather than leaving strings for the boundary to guess at.
        """
        matrix: FloatArray = features.to_numpy().astype(np.float64, copy=False)
        return matrix

    def _categorical_indices(self, features: pl.DataFrame) -> list[int]:
        """Positions of the categorical columns, which is what LightGBM wants on an array."""
        return [features.columns.index(column) for column in self._categorical]


def add_treatment_column(
    features: pl.DataFrame, treatment: np.ndarray, column: str
) -> pl.DataFrame:
    """Append the treatment indicator as a feature, for the learners that need it inline.

    Args:
        features: The feature matrix.
        treatment: Values for the appended column, one per row.
        column: Name for the appended column.

    Returns:
        A new frame with the column appended at the end.

    Raises:
        ValueError: If the name is already taken.
    """
    if column in features.columns:
        msg = f"cannot append treatment column {column!r}: name already in the feature matrix"
        raise ValueError(msg)
    return features.with_columns(pl.Series(column, treatment.astype(np.float64)))
