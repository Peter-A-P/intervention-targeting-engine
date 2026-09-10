"""Hillstrom MineThatData: 64,000 customers, a randomised three-arm email experiment.

Two of the three arms are kept, so the treatment is binary (PLAN.md section 3). Assignment
was random and equal across the three arms, so the propensity among any two of them is
exactly 0.5 and is recorded as known rather than estimated. Card: docs/data/hillstrom.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from itx.data.download import fetch
from itx.types import UpliftDataset

Arm = Literal["womens", "mens"]
Outcome = Literal["visit", "conversion", "spend"]

ARM_SEGMENT: dict[Arm, str] = {"womens": "Womens E-Mail", "mens": "Mens E-Mail"}
CONTROL_SEGMENT = "No E-Mail"

#: Ordered categories, encoded as the integers LightGBM is told are categorical.
ZIP_CODES = ("Rural", "Surburban", "Urban")  # the source file's spelling of "Suburban"
CHANNELS = ("Multichannel", "Phone", "Web")

CATEGORICAL = ("zip_code", "channel")
NUMERIC = ("recency", "history", "history_segment", "mens", "womens", "newbie")
FEATURES = (*NUMERIC, *CATEGORICAL)

KNOWN_PROPENSITY = 0.5

#: The full schema, declared rather than inferred. Polars infers from the first rows and
#: guesses integer for ``spend``, which is 0 for the first few thousand customers and then
#: 29.99; an inferred schema makes a loader that depends on row order.
SCHEMA: dict[str, pl.DataType] = {
    "recency": pl.Int64(),
    "history_segment": pl.String(),
    "history": pl.Float64(),
    "mens": pl.Int64(),
    "womens": pl.Int64(),
    "zip_code": pl.String(),
    "newbie": pl.Int64(),
    "channel": pl.String(),
    "segment": pl.String(),
    "visit": pl.Int64(),
    "conversion": pl.Int64(),
    "spend": pl.Float64(),
}


def load_hillstrom(
    *,
    outcome: Outcome = "visit",
    arm: Arm = "womens",
    path: Path | None = None,
) -> UpliftDataset:
    """Load Hillstrom as a binary-treatment uplift dataset.

    Args:
        outcome: Which recorded outcome to model. ``visit`` is the headline one: it is the
            only one with enough events for a stable Qini at 64k rows.
        arm: Which email arm is the treatment. The other arm is dropped entirely rather
            than folded into the control, so the control really is "no email".
        path: Read from this file instead of the download cache. For tests.

    Returns:
        The dataset, with ``propensity`` set to the design value of 0.5.
    """
    source_path = path if path is not None else fetch("hillstrom", quiet=True)
    frame = pl.read_csv(source_path, schema=SCHEMA)

    treated_segment = ARM_SEGMENT[arm]
    frame = frame.filter(pl.col("segment").is_in([treated_segment, CONTROL_SEGMENT]))

    treatment = (frame["segment"] == treated_segment).cast(pl.Int64).to_numpy()
    outcome_values = frame[outcome].cast(pl.Float64).to_numpy()

    features = frame.select(
        pl.col("recency").cast(pl.Float64),
        pl.col("history").cast(pl.Float64),
        _history_segment_ordinal().alias("history_segment"),
        pl.col("mens").cast(pl.Float64),
        pl.col("womens").cast(pl.Float64),
        pl.col("newbie").cast(pl.Float64),
        _encode("zip_code", ZIP_CODES),
        _encode("channel", CHANNELS),
    )

    return UpliftDataset(
        name=f"hillstrom-{arm}-{outcome}",
        features=features,
        treatment=treatment,
        outcome=outcome_values,
        categorical=CATEGORICAL,
        propensity=np.full(frame.height, KNOWN_PROPENSITY),
    )


def _history_segment_ordinal() -> pl.Expr:
    """Turn ``"2) $100 - $200"`` into 2.

    The segment is an ordered binning of ``history``, so it stays numeric rather than
    categorical: the order carries information a one-of-seven encoding would throw away.
    """
    return pl.col("history_segment").str.extract(r"^(\d)", 1).cast(pl.Float64)


def _encode(column: str, categories: tuple[str, ...]) -> pl.Expr:
    """Map a string column to the index of its category, as a float LightGBM reads as a code."""
    mapping = {value: float(index) for index, value in enumerate(categories)}
    return pl.col(column).replace_strict(mapping, return_dtype=pl.Float64).alias(column)
