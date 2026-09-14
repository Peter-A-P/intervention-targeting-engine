"""Shared types: the dataset container every loader returns and every estimator consumes.

Polars is the internal data format (PLAN.md section 5). Treatment and outcome are numpy
arrays because every estimator, metric and bootstrap in this package indexes them
positionally; keeping them out of the frame makes it impossible to leak either one into
the feature matrix by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class UpliftDataset:
    """One dataset ready for fitting: features, binary treatment, outcome.

    Attributes:
        name: Short identifier, used in result tables and file names.
        features: Numeric feature matrix. Categorical columns are integer-encoded and
            named in ``categorical``; no treatment or outcome column is ever present.
        treatment: Binary treatment indicator, 0 or 1, one per row.
        outcome: Observed outcome, one per row. Binary outcomes are 0.0 or 1.0.
        categorical: Names of the columns in ``features`` that are integer-encoded
            categories rather than numbers, passed through to LightGBM.
        propensity: Per-unit probability of treatment where it is known by design (a
            randomised experiment), otherwise None and it has to be estimated.
        true_effect: Per-unit treatment effect where the data is simulated and the truth
            was written down, otherwise None. Only the semi-synthetic sets have it.
        replicate: Index of the replicate for datasets that ship many (IHDP, ACIC).
        risk_is_low_outcome: Which way a risk model points. False, the default, means the
            units a risk score puts first are the ones with the *highest* predicted
            outcome: likely to churn, respond, visit, be readmitted. True means they are the
            ones with the *lowest*, which is the case when the outcome is a value the harm
            reduces, such as dollars retained on a transaction that may be fraudulent. The
            outcome-ranking baseline and the risk-decile diagnostic read this so that "rank
            by risk" means what a team running that queue would mean by it, on every
            dataset, without the sign being fixed by hand at each call site.
    """

    name: str
    features: pl.DataFrame
    treatment: IntArray
    outcome: FloatArray
    categorical: tuple[str, ...] = ()
    propensity: FloatArray | None = None
    true_effect: FloatArray | None = None
    replicate: int | None = None
    risk_is_low_outcome: bool = False

    def __post_init__(self) -> None:
        """Reject a malformed dataset at construction rather than mid-benchmark."""
        n = self.features.height
        for label, array in (("treatment", self.treatment), ("outcome", self.outcome)):
            if array.shape != (n,):
                msg = f"{self.name}: {label} has shape {array.shape}, expected ({n},)"
                raise ValueError(msg)
        if not np.isin(self.treatment, (0, 1)).all():
            msg = f"{self.name}: treatment must be binary 0/1"
            raise ValueError(msg)
        missing = set(self.categorical) - set(self.features.columns)
        if missing:
            msg = f"{self.name}: categorical columns not in features: {sorted(missing)}"
            raise ValueError(msg)
        for optional_name, optional in (
            ("propensity", self.propensity),
            ("true_effect", self.true_effect),
        ):
            if optional is not None and optional.shape != (n,):
                msg = (
                    f"{self.name}: {optional_name} has shape {optional.shape}, expected ({n},)"
                )
                raise ValueError(msg)

    @property
    def n_units(self) -> int:
        """Number of rows."""
        return self.features.height

    @property
    def n_treated(self) -> int:
        """Number of treated rows."""
        return int(self.treatment.sum())

    @property
    def feature_names(self) -> list[str]:
        """Feature column names, in matrix order."""
        return self.features.columns

    @property
    def outcome_is_binary(self) -> bool:
        """True when the outcome only ever takes the values 0 and 1."""
        return bool(np.isin(self.outcome, (0.0, 1.0)).all())

    def require_true_effect(self) -> FloatArray:
        """The true per-unit effect, or an error saying this dataset does not have one.

        PEHE and ATE error are only defined against a known truth, and a caller that
        reaches for one on a randomised dataset has made a category mistake rather than
        hit a missing value. Failing here beats threading an optional array through every
        metric and quietly returning nothing.

        Returns:
            The true per-unit effect.

        Raises:
            ValueError: If this dataset has no ground truth.
        """
        if self.true_effect is None:
            msg = (
                f"{self.name}: no ground truth. The individual effect is not recorded in "
                f"this dataset and cannot be; only the simulated sets have it."
            )
            raise ValueError(msg)
        return self.true_effect

    def take(self, index: IntArray, *, name: str | None = None) -> UpliftDataset:
        """Return the subset at ``index``, carrying every optional array along with it.

        Args:
            index: Row positions to keep, in the order they should appear.
            name: Name for the subset; defaults to the parent's name.

        Returns:
            A new dataset holding only the selected rows.
        """
        return replace(
            self,
            name=self.name if name is None else name,
            features=self.features[index],
            treatment=self.treatment[index],
            outcome=self.outcome[index],
            propensity=None if self.propensity is None else self.propensity[index],
            true_effect=None if self.true_effect is None else self.true_effect[index],
        )

    def matrix(self) -> pl.DataFrame:
        """The feature matrix on its own, for handing to a base learner."""
        return self.features

    def describe(self) -> str:
        """One line for logs: size, treated share, outcome mean by arm."""
        treated = self.treatment == 1
        return (
            f"{self.name}: {self.n_units:,} units, "
            f"{treated.mean():.1%} treated, "
            f"outcome mean {self.outcome[treated].mean():.4f} treated / "
            f"{self.outcome[~treated].mean():.4f} control"
        )


@dataclass(frozen=True, slots=True)
class Split:
    """A fixed train/validation/test partition of one dataset.

    Attributes:
        train: Fitting rows.
        validation: Hyperparameter selection rows. No metric is ever reported from these.
        test: Held-out rows. Every number in the results table comes from here.
        seed: The seed that produced the partition, committed with the results.
    """

    train: UpliftDataset
    validation: UpliftDataset
    test: UpliftDataset
    seed: int

    def describe(self) -> str:
        """One line for logs: the three sizes and the seed."""
        return (
            f"{self.train.name} seed {self.seed}: "
            f"{self.train.n_units:,} train / {self.validation.n_units:,} val / "
            f"{self.test.n_units:,} test"
        )
