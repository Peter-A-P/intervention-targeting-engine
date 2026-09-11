"""The probability of being treated, given the covariates.

Three estimators need it and each needs it for a different reason: the X-learner weights
its two imputed effects by it, the DR and R learners divide by it, and the policy-value
estimators in week 5 reweight the test split with it. It is worth one careful
implementation rather than three casual ones, because everything that divides by a
propensity is one badly calibrated model away from an enormous variance.

Two safeguards are built in and both are stated plainly in the results.

**Known propensities are used, not estimated.** On a randomised experiment the probability
of treatment is a design constant. Fitting a model to recover a number that is already
known adds variance and can only make things worse, so ``UpliftDataset.propensity`` wins
whenever it is present.

**Estimates are clipped, and the clipping is counted.** A fitted propensity near 0 or 1
means a unit that essentially could not have been in the arm it is being compared against,
and any estimator that divides by it will produce a huge weight from a single row. The
standard fix is to clip into ``[eps, 1 - eps]``. The standard mistake is to do it quietly:
clipping does not repair the overlap problem, it hides it, so the number of clipped units
is recorded and a fit that has to clip a meaningful share of the sample says so.

**Training-row propensities are out-of-fold, and this is not a refinement.** A model asked
to predict the treatment of the rows it was fitted on has partly memorised them, so its
prediction for a unit is pulled toward that unit's own realised treatment. The residual
``t - e(x)`` then no longer has conditional mean zero, and every estimator that treats it
as a residual, the R-learner most of all, is built on something that is not one. This is
the orthogonality condition that double machine learning exists to protect, and breaking it
does not degrade an estimate gracefully.

On ACIC 2016 the R-learner's PEHE is 23.95 with an in-sample propensity and 1.20 with an
out-of-fold one, on a true effect whose standard deviation is 3.85. The same twenty-fold
gap appears on every seed.

The part worth remembering is what the propensity diagnostics said while that was going on.
The share of units against the clipping bound moved from 48.2% to 47.8%. The median
treatment residual moved from 0.014 to 0.025. Neither number tells you that one of the two
fits is unusable, because the damage is in the correlation between the residual and the
unit's own outcome rather than in any marginal summary of the propensity. It was visible
only against a known truth, which is the argument for keeping the simulated datasets in the
benchmark, stated as concretely as it can be.

A separate model fitted on everything is kept for scoring rows the estimator never saw,
where the problem does not arise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import StratifiedKFold

from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig, OutcomeLearner

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from itx.types import FloatArray, IntArray, UpliftDataset

#: Default clipping bound. Below this a single unit's inverse weight exceeds 100, which is
#: enough for one row to dominate an average over thousands.
DEFAULT_CLIP = 0.01

#: A fit that clips more than this share of the sample is reporting a real overlap
#: problem rather than tidying up a few tail cases.
OVERLAP_WARNING_SHARE = 0.01

#: Cross-fitting folds for the training-row propensities.
DEFAULT_FOLDS = 5


@dataclass(frozen=True, slots=True)
class PropensityFit:
    """Fitted or known treatment probabilities, and what had to be done to them.

    Attributes:
        values: Probability of treatment per unit, already clipped.
        known: True when these came from the dataset's design rather than from a model.
        clip: The bound applied.
        n_clipped: How many units hit the bound.
        raw_min: Smallest probability before clipping.
        raw_max: Largest probability before clipping.
        out_of_fold: True when the values were predicted by models that had not seen the
            row they are predicting. False for a known propensity, where it does not apply.
    """

    values: FloatArray
    known: bool
    clip: float
    n_clipped: int
    raw_min: float
    raw_max: float
    out_of_fold: bool = False

    @property
    def clipped_share(self) -> float:
        """Share of units whose probability hit the bound."""
        return self.n_clipped / self.values.size if self.values.size else 0.0

    @property
    def has_overlap_problem(self) -> bool:
        """True when enough units were clipped that the comparison is resting on the bound."""
        return self.clipped_share > OVERLAP_WARNING_SHARE

    def describe(self) -> str:
        """One line for logs and dataset cards."""
        if self.known:
            return (
                f"propensity known by design, "
                f"{self.values.min():.3f} to {self.values.max():.3f}"
            )
        fitting = "out-of-fold" if self.out_of_fold else "in-sample"
        return (
            f"propensity estimated {fitting}, raw range {self.raw_min:.4f} to "
            f"{self.raw_max:.4f}, {self.n_clipped:,} of {self.values.size:,} clipped at "
            f"{self.clip}"
        )


class PropensityModel:
    """A LightGBM classifier for the treatment indicator.

    Short-circuited when the dataset already knows its propensities by design.
    """

    def __init__(
        self,
        config: BaseLearnerConfig = DEFAULT_CONFIG,
        *,
        seed: int = 0,
        clip: float = DEFAULT_CLIP,
        use_known: bool = True,
        folds: int = DEFAULT_FOLDS,
        cross_fit: bool = True,
    ) -> None:
        """Build an unfitted propensity model.

        Args:
            config: Shared LightGBM settings, the same ones the outcome models use.
            seed: Seed for the classifier.
            clip: Probabilities are held inside ``[clip, 1 - clip]``.
            use_known: Use ``UpliftDataset.propensity`` when the dataset has it. Turning
                this off is useful for exactly one thing: showing on a randomised dataset
                how much an estimated propensity costs against the known one.
            folds: Cross-fitting folds for the training-row propensities.
            cross_fit: Predict the training rows out of fold. Turning it off is what the
                literature calls the in-sample propensity, and it exists here so the cost
                of the shortcut can be measured rather than asserted.
        """
        self.config = config
        self.seed = seed
        self.clip = clip
        self.use_known = use_known
        self.folds = folds
        self.cross_fit = cross_fit
        self._model: OutcomeLearner | None = None
        self._known: FloatArray | None = None
        self._categorical: Sequence[int] = ()

    def fit(self, data: UpliftDataset) -> PropensityFit:
        """Fit the model, or adopt the dataset's known propensities.

        Args:
            data: The dataset to fit on.

        Returns:
            The fit, including what was clipped.
        """
        if self.use_known and data.propensity is not None:
            self._known = data.propensity
            return self._package(data.propensity, known=True)

        self._categorical = [data.features.columns.index(column) for column in data.categorical]
        matrix = data.features.to_numpy().astype(np.float64, copy=False)
        target = data.treatment.astype(np.float64)

        # The model kept for scoring unseen rows is fitted on everything.
        self._model = OutcomeLearner(self.config, binary=True, seed=self.seed)
        self._model.fit(
            matrix,
            target,
            categorical=self._categorical,
            feature_names=data.features.columns,
        )

        if not self.cross_fit:
            return self._package(self._raw(data.features), known=False, out_of_fold=False)
        return self._package(self._out_of_fold(matrix, target), known=False, out_of_fold=True)

    def predict(self, features: pl.DataFrame) -> FloatArray:
        """Clipped probability of treatment for each row.

        Args:
            features: Feature matrix.

        Returns:
            One probability per row.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._known is not None:
            if self._known.size == features.height:
                return np.clip(self._known, self.clip, 1.0 - self.clip)
            # A known constant propensity applies to any number of rows; a known per-unit
            # one does not, and silently recycling it would be a bug worth catching.
            unique = np.unique(self._known)
            if unique.size == 1:
                return np.full(features.height, float(unique[0]))
            msg = (
                "propensity: this dataset's known propensities vary by unit and cannot be "
                "applied to a different set of rows"
            )
            raise RuntimeError(msg)
        if self._model is None:
            msg = "propensity: fit before predicting"
            raise RuntimeError(msg)
        return np.clip(self._raw(features), self.clip, 1.0 - self.clip)

    def _out_of_fold(self, matrix: FloatArray, target: FloatArray) -> FloatArray:
        """Predict every training row from a model that was not shown it.

        Folds are stratified on the treatment, because a fold with no treated rows cannot
        produce a propensity at all, and on a dataset with a rare treatment an unstratified
        split produces one sooner or later.
        """
        folds = self._usable_folds(target)
        if folds < 2:  # pragma: no cover - a single-arm dataset fails earlier than this
            return self._raw_matrix(matrix)

        predictions = np.empty(matrix.shape[0], dtype=np.float64)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        for fold, (train_rows, held_out) in enumerate(splitter.split(matrix, target)):
            learner = OutcomeLearner(self.config, binary=True, seed=self.seed + fold + 1)
            learner.fit(matrix[train_rows], target[train_rows], categorical=self._categorical)
            predictions[held_out] = learner.predict(matrix[held_out])
        return predictions

    def _usable_folds(self, target: FloatArray) -> int:
        """Folds, reduced when an arm is too thin to appear in all of them."""
        smallest_arm = int(min(target.sum(), target.size - target.sum()))
        return max(2, min(self.folds, smallest_arm))

    def _raw_matrix(self, matrix: FloatArray) -> FloatArray:
        """Unclipped model output for an already-converted matrix."""
        if self._model is None:  # pragma: no cover - guarded by callers
            msg = "propensity: fit before predicting"
            raise RuntimeError(msg)
        return self._model.predict(matrix)

    def _raw(self, features: pl.DataFrame) -> FloatArray:
        """Unclipped model output."""
        if self._model is None:  # pragma: no cover - guarded by callers
            msg = "propensity: fit before predicting"
            raise RuntimeError(msg)
        return self._model.predict(features.to_numpy().astype(np.float64, copy=False))

    def _package(
        self, raw: FloatArray, *, known: bool, out_of_fold: bool = False
    ) -> PropensityFit:
        """Clip, count what was clipped, and record the raw range."""
        clipped = np.clip(raw, self.clip, 1.0 - self.clip)
        return PropensityFit(
            values=clipped,
            known=known,
            out_of_fold=out_of_fold,
            clip=self.clip,
            n_clipped=int(np.sum(raw != clipped)),
            raw_min=float(raw.min()) if raw.size else float("nan"),
            raw_max=float(raw.max()) if raw.size else float("nan"),
        )


def treatment_share(treatment: IntArray) -> float:
    """Share of units that were treated.

    The crudest possible propensity, and the right one when assignment was randomised with
    a single constant probability. Used as a fallback so that an estimator is never left
    dividing by a number that came from nowhere.

    Args:
        treatment: Binary treatment indicator.

    Returns:
        The treated share.
    """
    return float(np.mean(treatment))
