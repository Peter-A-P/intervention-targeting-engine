"""Lenta: 687,029 grocery customers, an SMS campaign, and 194 mostly undocumented columns.

Distributed by the scikit-uplift project, which fetches it from the same public bucket this
loader does. Card: docs/data/lenta.md.

It is the widest dataset in the project and the messiest, and both of those are why it is
worth carrying. Hillstrom has eight tidy features and Criteo has twelve anonymous ones; this
has 194 with names like ``k_var_disc_share_15d_g34``, no data dictionary, and missing values
in 153 of them. That is the shape of a real customer table, and an estimator that only works
on the tidy ones is not much use.

**No licence is stated, anywhere.** Not by the publisher, not in scikit-uplift's code, not in
its documentation. This repository downloads the file from the publisher's own bucket at run
time and redistributes nothing, which is exactly what scikit-uplift does, but a reader who
wants to use the data for anything beyond reproducing this benchmark has no licence to rely
on and should say so out loud rather than assume. PLAN.md section 3 said "check licence in
the package"; the check was done and the answer is that there is not one (change 25).

**Two columns are leaks, and the balance table is what proved it.** ``response_sms`` and
``response_viber`` sit in the feature block with names that could plausibly mean "responded
to some earlier campaign". They do not. In a randomised trial a pre-treatment covariate has
the same mean in both arms, and across the other 192 numeric columns the median standardised
mean difference is 0.011 and the worst is 0.025. ``response_sms`` is 0.198 and
``response_viber`` is 0.068: eighteen and six times the median, in the only two columns whose
names suggest they were recorded after the campaign went out. They are responses to the
campaign's own delivery channels. Both are dropped, and :mod:`itx.metrics.balance` holds the
diagnostic that found them so the same check runs against every dataset rather than living
in a notebook someone ran once.

**The arms are 75/25 and randomisation is assumed, not documented.** Nothing states the
design probability, so unlike Hillstrom's 0.5 and Criteo's 0.85 the propensity here is left
to be estimated rather than declared known. The covariate balance above is the evidence that
it was randomised, and estimating a propensity that is in truth constant costs a little
precision and no correctness, which is the right way round for an assumption this thin.

**Missing values are left missing.** LightGBM routes NaN down its own branch at every split,
so no imputation is needed and none is done. The alternative, filling with a median or a
zero, would put a modelling choice made here into every estimator's input and make the
estimator column report partly on that choice. One column, ``k_var_sku_price_15d_g49``, is
absent for 72% of rows and is kept on the same reasoning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from itx.data.download import fetch, require_rows
from itx.types import UpliftDataset

Outcome = Literal["response_att"]

#: The treatment column, and the value that means treated.
GROUP_COLUMN = "group"
TREATED_GROUP = "test"
CONTROL_GROUP = "control"

#: The recorded outcome: the customer visited a store during the campaign window.
OUTCOME_COLUMN = "response_att"

#: Recorded after the campaign went out, and therefore not covariates. See the module
#: docstring: the standardised mean difference between arms is 0.198 and 0.068 against a
#: median of 0.011 across every other column.
POST_TREATMENT: tuple[str, ...] = ("response_sms", "response_viber")

#: Gender, the one genuinely categorical column. The labels are Cyrillic in the source
#: file and are written here as escapes rather than as characters, because a Cyrillic
#: capital EM renders identically to a Latin M and a reader has no way to tell which one a
#: literal holds. Zhe is female, Em is male, and the third label reads "not determined".
GENDER_COLUMN = "gender"
GENDER_FEMALE = "\u0416"
GENDER_MALE = "\u041c"
GENDER_UNKNOWN = "\u041d\u0435 \u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d"

#: Codes handed to LightGBM as categories. The file's explicit "not determined" level
#: (1,090 rows) and its 8,581 nulls are folded into one unknown code: they mean the same
#: thing to a model, and 1,090 rows cannot support a level of their own on a split.
GENDER_CODES: dict[str, int] = {GENDER_FEMALE: 0, GENDER_MALE: 1, GENDER_UNKNOWN: 2}
GENDER_UNKNOWN_CODE = 2

#: Integer-coded categories, named for LightGBM. ``main_format`` is already 0/1 in the
#: file and binary, so its coding is its own and nothing has to be mapped.
CATEGORICAL: tuple[str, ...] = ("gender", "main_format")

#: Columns that are not features: the treatment, the outcome, and the two leaks.
NOT_FEATURES: frozenset[str] = frozenset({GROUP_COLUMN, OUTCOME_COLUMN, *POST_TREATMENT})

#: Rows in the published file, checked on load so that a re-release is a failure rather
#: than a quietly different result.
EXPECTED_ROWS = 687_029


def load_lenta(*, path: Path | None = None) -> UpliftDataset:
    """Load Lenta as a binary-treatment uplift dataset.

    Args:
        path: Read from this file instead of the download cache. For tests.

    Returns:
        The dataset. ``propensity`` is None: the design probability is not documented, so
        it is estimated rather than asserted.

    Raises:
        ValueError: If the file does not hold the expected number of rows, or if the
            treatment column holds a value that is neither arm.
    """
    source_path = path if path is not None else fetch("lenta", quiet=True)
    frame = pl.read_csv(source_path, infer_schema_length=None)

    if path is None:
        require_rows("lenta", frame.height, EXPECTED_ROWS)

    arms = set(frame[GROUP_COLUMN].unique().to_list())
    unexpected = arms - {TREATED_GROUP, CONTROL_GROUP}
    if unexpected:
        msg = f"lenta: unexpected values in {GROUP_COLUMN!r}: {sorted(unexpected)}"
        raise ValueError(msg)

    treatment = (frame[GROUP_COLUMN] == TREATED_GROUP).cast(pl.Int64).to_numpy()
    outcome = frame[OUTCOME_COLUMN].cast(pl.Float64).to_numpy()

    return UpliftDataset(
        name="lenta",
        features=frame.select(_feature_expressions(frame.columns)),
        treatment=treatment,
        outcome=outcome,
        categorical=CATEGORICAL,
    )


def feature_names(columns: list[str]) -> list[str]:
    """The feature columns, in file order, with the treatment, outcome and leaks removed.

    Args:
        columns: Every column in the raw file.

    Returns:
        The names that become the feature matrix.
    """
    return [name for name in columns if name not in NOT_FEATURES]


def _feature_expressions(columns: list[str]) -> list[pl.Expr]:
    """Build the feature matrix: gender coded, everything else cast to float."""
    expressions: list[pl.Expr] = []
    for name in feature_names(columns):
        if name == GENDER_COLUMN:
            expressions.append(_encode_gender())
        else:
            expressions.append(pl.col(name).cast(pl.Float64))
    return expressions


def _encode_gender() -> pl.Expr:
    """Map the Cyrillic gender labels to integer codes, nulls to the unknown code."""
    mapping = {label: float(code) for label, code in GENDER_CODES.items()}
    return (
        pl.col(GENDER_COLUMN)
        .replace_strict(mapping, default=float(GENDER_UNKNOWN_CODE), return_dtype=pl.Float64)
        .fill_null(float(GENDER_UNKNOWN_CODE))
        .alias(GENDER_COLUMN)
    )
