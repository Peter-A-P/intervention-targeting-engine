"""Load somebody else's data, and refuse it when it cannot answer the question.

Every other loader in this package reads one known file and knows what is in it. This one
reads a file it has never seen, named by a stranger on a command line, and its job is
therefore mostly refusal. The failure this file exists to prevent is not a crash. It is a
neat table of Qini numbers computed on data that could never have supported one, screenshotted
away from every caveat, and used to move a budget.

## What is checked, and why each one

Structural, because without them there is nothing to compute:

- the named columns exist, and the treatment and outcome are not among the features
- the treatment is binary, with both arms present
- the outcome is numeric and not constant
- there are enough rows, and enough of both arms, for the requested number of bands

Substantive, because with them the computation runs and lies:

- **The design has to be declared and there is no default.** A randomised design makes the
  per-band arm difference an unbiased estimate of what the intervention did. An observational
  one does not: the difference inside a band then mixes the effect with whatever made those
  people likelier to be treated, and no amount of data fixes it. The diagnostic cannot detect
  this, so the caller states it and the statement is printed above every table.
- **Which direction risk points has to be declared** when the outcome is a value rather than
  an event. On a churn flag, risk is a high predicted outcome. On dollars retained it is a low
  one, and a tool that assumes the first will confidently rank the most profitable customers
  as the most at risk. That is not hypothetical: it shipped here, and PLAN.md change 54 is
  what it cost.
- **Whether a bigger number is a better one has to be declared**, separately, and it is not
  the same question. A churn dataset has risk at the high end and benefit at the low end, so
  an intervention that works shows up as a negative uplift. Every dataset in this repository
  has a good outcome at the high end, so nothing exercised the other case until this loader
  existed: the first synthetic churn file put through it got the opposite verdict from the
  same data encoded as retention. PLAN.md change 60.

## What is not checked, and cannot be

Whether the feature columns are pre-treatment. A column recorded after the intervention
carries part of the answer, and a model handed it will look excellent and mean nothing. From
a CSV of unknown provenance this is undecidable: ``itx`` can measure that a column is
suspiciously well balanced or suspiciously predictive, which
:mod:`itx.metrics.balance` does, but the question "was this measured before the offer went
out" is a fact about the world and the caller is the only one who knows it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from itx.types import UpliftDataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from itx.types import FloatArray

Design = Literal["randomised", "observational"]

#: Below this many rows in the smaller arm of a band, a band's difference is noise dressed as
#: a number. Ten bands of thirty a side is the smallest thing worth printing, and even that is
#: generous.
MIN_PER_ARM_PER_BAND = 30

OBSERVATIONAL_WARNING = (
    "This data is declared observational. Each band's number is the plain difference between "
    "the treated and untreated people in it, so it mixes what the intervention did with "
    "whatever made those people likelier to be treated in the first place. Nothing here "
    "corrects for that and nothing here can detect it. Read the table as a description of "
    "the data, not as an effect, unless you can argue that treatment was as good as random "
    "given the features."
)


@dataclass(frozen=True)
class ColumnSpec:
    """Which column is which, as the caller declares it.

    Attributes:
        treatment: Column holding the 0/1 intervention indicator.
        outcome: Column holding the observed outcome.
        features: Columns to model on. Every one must be pre-treatment, which this package
            cannot verify.
        categorical: Which of the features are categories rather than numbers.
        design: ``"randomised"`` if assignment was random, otherwise ``"observational"``.
        higher_outcome_is_better: Whether a larger outcome is the good end. True for a
            response or dollars retained, False for churn or a readmission. There is no
            default because the same file encoded as ``churned`` and as ``retained`` gets
            opposite verdicts, and only the caller knows which way theirs runs.
        risk_is_low_outcome: True when a *low* predicted outcome is the risky end, as on a
            dollars-retained outcome.
        propensity: Column holding the known probability of treatment, where the design
            fixes it.
    """

    treatment: str
    outcome: str
    features: tuple[str, ...]
    design: Design
    higher_outcome_is_better: bool
    categorical: tuple[str, ...] = ()
    risk_is_low_outcome: bool = False
    propensity: str | None = None


class UnusableDataError(ValueError):
    """The file cannot answer the question, and saying so is the whole point.

    Separate from :class:`ValueError` so a caller can tell "this data will not do" from a
    programming mistake, and so the CLI can print the reason without a traceback.
    """


def _encode_category(name: str) -> pl.Expr:
    """Integer-encode one column by sorted level, reserving 0 for missing.

    The code is the level's position in the sorted list of levels, so the same value encodes
    to the same integer in every process. Codes assigned by first appearance do not, which
    PLAN.md change 58 records the cost of.

    Args:
        name: Column name.

    Returns:
        An expression producing the encoded column.
    """
    return pl.col(name).rank("dense").cast(pl.Float64).fill_null(0.0).alias(name)


def _check_columns(frame: pl.DataFrame, spec: ColumnSpec) -> None:
    """Raise unless every declared column is present and nothing is used twice.

    Args:
        frame: The loaded file.
        spec: The caller's declaration.

    Raises:
        UnusableDataError: If a column is missing, duplicated across roles, or if no
            features remain.
    """
    present = set(frame.columns)
    wanted = {spec.treatment, spec.outcome, *spec.features}
    if spec.propensity is not None:
        wanted.add(spec.propensity)
    missing = sorted(wanted - present)
    if missing:
        msg = (
            f"columns not in the file: {', '.join(missing)}. It has: {', '.join(frame.columns)}"
        )
        raise UnusableDataError(msg)

    overlap = sorted({spec.treatment, spec.outcome} & set(spec.features))
    if overlap:
        msg = (
            f"{', '.join(overlap)} is both a feature and the treatment or outcome. A model "
            f"handed the answer will look excellent and mean nothing."
        )
        raise UnusableDataError(msg)

    unknown = sorted(set(spec.categorical) - set(spec.features))
    if unknown:
        msg = f"declared categorical but not a feature: {', '.join(unknown)}"
        raise UnusableDataError(msg)

    if not spec.features:
        msg = "no feature columns given; there is nothing to model on"
        raise UnusableDataError(msg)


def _check_treatment(frame: pl.DataFrame, spec: ColumnSpec) -> None:
    """Raise unless the treatment is binary with both arms populated.

    Args:
        frame: The loaded file.
        spec: The caller's declaration.

    Raises:
        UnusableDataError: If the column is not 0/1, or either arm is empty.
    """
    values = frame[spec.treatment].drop_nulls().unique().to_list()
    unexpected = sorted(str(v) for v in values if v not in (0, 1, 0.0, 1.0, True, False))
    if unexpected:
        msg = (
            f"{spec.treatment} must be 0 or 1, found {', '.join(unexpected[:6])}. This package "
            f"handles one binary intervention, not doses or several arms."
        )
        raise UnusableDataError(msg)
    treated = int(frame[spec.treatment].cast(pl.Int64).sum())
    total = frame.height
    if treated == 0 or treated == total:
        which = "treated" if treated == 0 else "untreated"
        msg = (
            f"every row is {'untreated' if treated == 0 else 'treated'}; there are no {which} "
            f"rows to compare against. Nothing can be learned about an intervention from a "
            f"population that all got it, or none did."
        )
        raise UnusableDataError(msg)


def _check_size(frame: pl.DataFrame, spec: ColumnSpec, bins: int) -> None:
    """Raise when the file is too small for the requested bands to mean anything.

    Args:
        frame: The loaded file.
        spec: The caller's declaration.
        bins: Bands the diagnostic will cut the population into.

    Raises:
        UnusableDataError: If a band would hold fewer than :data:`MIN_PER_ARM_PER_BAND` per arm.
    """
    treated = int(frame[spec.treatment].cast(pl.Int64).sum())
    smaller = min(treated, frame.height - treated)
    per_band = smaller / bins
    if per_band < MIN_PER_ARM_PER_BAND:
        msg = (
            f"{frame.height:,} rows with {smaller:,} in the smaller arm is about "
            f"{per_band:.0f} per arm per band across {bins} bands, under the "
            f"{MIN_PER_ARM_PER_BAND} this will report on. Use fewer bands, or collect "
            f"more data: 'itx power' will say how much more."
        )
        raise UnusableDataError(msg)


def load_csv(
    path: Path | str,
    spec: ColumnSpec,
    *,
    bins: int = 10,
    name: str | None = None,
) -> UpliftDataset:
    """Read a caller's CSV into a dataset, or refuse it with a reason.

    Args:
        path: The file to read.
        spec: Which column is which, and what the caller asserts about the design.
        bins: Bands the diagnostic will use, checked against the arm sizes here so the
            refusal arrives before any fitting.
        name: Dataset name for tables; the file's stem if omitted.

    Returns:
        The dataset, ready for :func:`itx.metrics.risk_deciles.risk_deciles`.

    Raises:
        UnusableDataError: If the file cannot support the question being asked of it.

    Warns:
        UserWarning: When the design is declared observational, because every number that
            follows means something weaker than it appears to.
    """
    path = Path(path)
    if not path.exists():
        msg = f"no such file: {path}"
        raise UnusableDataError(msg)

    frame = pl.read_csv(path, infer_schema_length=10_000)
    if frame.height == 0:
        msg = f"{path} has no rows"
        raise UnusableDataError(msg)

    _check_columns(frame, spec)
    _check_treatment(frame, spec)
    _check_size(frame, spec, bins)

    outcome_column = frame[spec.outcome]
    if not outcome_column.dtype.is_numeric():
        msg = (
            f"{spec.outcome} is {outcome_column.dtype}, not a number. Encode the outcome "
            f"first: 0/1 for an event, a number for a value."
        )
        raise UnusableDataError(msg)
    if outcome_column.drop_nulls().n_unique() < 2:
        msg = f"{spec.outcome} takes one value; there is no outcome to explain"
        raise UnusableDataError(msg)
    if outcome_column.null_count():
        msg = (
            f"{spec.outcome} has {outcome_column.null_count():,} missing values. An outcome "
            f"cannot be imputed here: a row whose outcome is unknown carries no information "
            f"about the effect and has to be dropped by the caller, deliberately."
        )
        raise UnusableDataError(msg)

    if spec.design == "observational":
        warnings.warn(OBSERVATIONAL_WARNING, UserWarning, stacklevel=2)

    categorical = tuple(c for c in spec.features if c in set(spec.categorical))
    features = frame.select(
        [
            _encode_category(column)
            if column in set(categorical)
            else pl.col(column).cast(pl.Float64)
            for column in spec.features
        ]
    )

    propensity: FloatArray | None = None
    if spec.propensity is not None:
        propensity = frame[spec.propensity].to_numpy().astype(np.float64)
        if not ((propensity > 0.0) & (propensity < 1.0)).all():
            msg = (
                f"{spec.propensity} must be strictly between 0 and 1. A unit with a "
                f"propensity of 0 or 1 could never have been in the other arm."
            )
            raise UnusableDataError(msg)

    return UpliftDataset(
        name=name or path.stem,
        features=features,
        treatment=frame[spec.treatment].cast(pl.Int64).to_numpy().astype(np.int64),
        outcome=outcome_column.cast(pl.Float64).to_numpy().astype(np.float64),
        categorical=categorical,
        propensity=propensity,
        risk_is_low_outcome=spec.risk_is_low_outcome,
        higher_outcome_is_better=spec.higher_outcome_is_better,
    )
