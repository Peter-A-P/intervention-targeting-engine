"""Covariate balance between the arms, and the leak it catches.

In a randomised experiment every pre-treatment covariate has the same distribution in both
arms, up to sampling noise. So a covariate that does not is either a coincidence or not
pre-treatment, and on a dataset with hundreds of thousands of rows the sampling noise is
small enough that the second explanation is usually the right one.

That makes the standardised mean difference a leak detector, not just a table for the
appendix. It is the cheapest check in this package: two means, two standard deviations, no
model, one pass over the data.

It earned its place. Lenta ships 194 columns and no documentation of which are measured
before the campaign, and two of them are named ``response_sms`` and ``response_viber``. The
balance table settles what reading the names cannot::

    response_sms                   0.198
    response_viber                 0.068
    k_var_count_per_cheque_1m_g34  0.025
    ...
    median over 192 columns        0.011

One column is eighteen times the median and three times the next worst, in a trial where
everything else agrees to two decimal places. It is a response to the campaign, so it is a
consequence of the treatment and cannot be a feature; an estimator handed it would be told
part of the answer. Criteo's ``exposure`` is the same defect in a more obvious form, where
the column is simply zero for every control row.

The usual reading of this statistic is the other way round, as a check on a matching or
weighting procedure in an observational study, where imbalance means confounding rather
than leakage. Both readings are here: :func:`worst_imbalance` on a randomised dataset is
asking "did something post-treatment get in", and on a confounded one it is asking "how
much overlap is there to work with". The number is the same; what it licenses is not.

The 0.1 threshold below is convention, from the matching literature, and convention is all
it is. It is used here as a place to put a test's tripwire, not as a decision rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from itx.types import BoolArray, FloatArray, UpliftDataset

#: The customary line between "balanced" and "not", from the matching literature. A
#: convention, and used here only as somewhere to put a tripwire.
CONVENTIONAL_THRESHOLD = 0.1


def standardised_mean_differences(dataset: UpliftDataset) -> dict[str, float]:
    """Absolute standardised mean difference between arms, per feature.

    The statistic is the difference in arm means divided by the pooled within-arm standard
    deviation, so it is unitless and comparable across columns of different scales.

    Missing values are ignored per column rather than imputed. A column that is missing
    for most rows is still worth comparing on the rows that have it, and an imputation
    rule chosen here would put its own signal into a diagnostic whose whole job is to
    detect signal that should not be there.

    Args:
        dataset: The dataset to examine. Only ``features`` and ``treatment`` are read.

    Returns:
        One absolute standardised mean difference per feature column, by name. A column
        with no usable rows in one arm is omitted. A column that is constant within each
        arm but takes a different constant in each is reported as infinite: see
        :func:`_one_column`, which is where that case nearly went missing.

    Raises:
        ValueError: If either arm is empty, which makes the comparison undefined.
    """
    treated = dataset.treatment == 1
    control = ~treated
    if not treated.any() or not control.any():
        msg = f"{dataset.name}: balance needs both arms, got {int(treated.sum())} treated"
        raise ValueError(msg)

    differences: dict[str, float] = {}
    for name in dataset.features.columns:
        column = dataset.features[name].cast(float).to_numpy()
        value = _one_column(column, treated=treated, control=control)
        if value is not None:
            differences[name] = value
    return differences


def worst_imbalance(dataset: UpliftDataset) -> tuple[str, float]:
    """The single most imbalanced feature and its standardised mean difference.

    Args:
        dataset: The dataset to examine.

    Returns:
        The column name and its absolute standardised mean difference.

    Raises:
        ValueError: If no column can be compared at all.
    """
    differences = standardised_mean_differences(dataset)
    if not differences:
        msg = f"{dataset.name}: no feature could be compared between arms"
        raise ValueError(msg)
    name = max(differences, key=lambda key: differences[key])
    return name, differences[name]


def _one_column(
    column: FloatArray,
    *,
    treated: BoolArray,
    control: BoolArray,
) -> float | None:
    """The standardised mean difference for one column, or None if it has none.

    The zero-denominator case is the one worth spelling out, because the obvious handling
    of it is wrong and this function shipped wrong for an hour. A column with no variance
    inside either arm cannot be divided by its pooled standard deviation. If the two arms
    also share the same constant the column is simply constant and there is nothing to
    report, so it is dropped.

    But if the constants differ, the column separates the arms perfectly: knowing it tells
    you the treatment exactly. That is the most severe leak there is, and dropping it would
    mean the detector stayed silent on precisely the case it exists to catch. It is
    reported as infinite instead, which sorts to the top of any ranking of imbalance and
    cannot be mistaken for a small number.
    """
    present = ~np.isnan(column)
    left = column[treated & present]
    right = column[control & present]
    if left.size < 2 or right.size < 2:
        return None
    difference = abs(float(left.mean()) - float(right.mean()))
    pooled = float(np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0))
    if pooled == 0.0:
        return None if difference == 0.0 else float("inf")
    return difference / pooled
