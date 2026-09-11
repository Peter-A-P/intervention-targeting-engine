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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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
    """

    values: FloatArray
    known: bool
    clip: float
    n_clipped: int
    raw_min: float
    raw_max: float

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
        return (
            f"propensity estimated, raw range {self.raw_min:.4f} to {self.raw_max:.4f}, "
            f"{self.n_clipped:,} of {self.values.size:,} clipped at {self.clip}"
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
    ) -> None:
        """Build an unfitted propensity model.

        Args:
            config: Shared LightGBM settings, the same ones the outcome models use.
            seed: Seed for the classifier.
            clip: Probabilities are held inside ``[clip, 1 - clip]``.
            use_known: Use ``UpliftDataset.propensity`` when the dataset has it. Turning
                this off is useful for exactly one thing: showing on a randomised dataset
                how much an estimated propensity costs against the known one.
        """
        self.config = config
        self.seed = seed
        self.clip = clip
        self.use_known = use_known
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
        self._model = OutcomeLearner(self.config, binary=True, seed=self.seed)
        self._model.fit(
            data.features.to_numpy().astype(np.float64, copy=False),
            data.treatment.astype(np.float64),
            categorical=self._categorical,
            feature_names=data.features.columns,
        )
        return self._package(self._raw(data.features), known=False)

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

    def _raw(self, features: pl.DataFrame) -> FloatArray:
        """Unclipped model output."""
        if self._model is None:  # pragma: no cover - guarded by callers
            msg = "propensity: fit before predicting"
            raise RuntimeError(msg)
        return self._model.predict(features.to_numpy().astype(np.float64, copy=False))

    def _package(self, raw: FloatArray, *, known: bool) -> PropensityFit:
        """Clip, count what was clipped, and record the raw range."""
        clipped = np.clip(raw, self.clip, 1.0 - self.clip)
        return PropensityFit(
            values=clipped,
            known=known,
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
