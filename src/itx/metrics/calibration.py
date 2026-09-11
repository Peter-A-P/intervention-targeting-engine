"""Is the predicted uplift the size it claims to be, not just in the right order?

Every other metric in this package scores a ranking. Qini, AUUC and uplift at k would all
be unchanged if every predicted effect were multiplied by ten, because none of them looks
at the magnitude. That is fine for choosing who to treat and useless for the question a
budget holder asks next, which is what the intervention is worth: a model that ranks
perfectly and predicts effects three times too large will forecast three times the return
and be wrong about whether the programme was worth running at all.

The check is the standard one. Sort by predicted uplift, cut into equal-sized bins, and in
each bin compare the average prediction with the uplift actually realised there, estimated
as the difference in outcome between the treated and control units that fall in the bin.
A calibrated estimator puts the points on the diagonal.

Two summary numbers, because they fail differently:

``calibration_slope``
    The slope of realised uplift regressed on predicted uplift across the bins, weighted by
    bin precision. One is perfect. Below one means the predictions are too spread out, the
    usual direction: the model is confident about heterogeneity that is partly noise. Above
    one means they are too compressed, which is what regularisation does to a real effect.
    The slope is about the shape of the relationship and ignores a constant offset.

``calibration_error``
    Mean absolute difference between predicted and realised uplift across bins, in the
    outcome's units. Zero is perfect. This one does see a constant offset, so an estimator
    that gets every effect right except for adding 0.1 to all of them scores a perfect slope
    and a calibration error of 0.1.

Both are noisy on a small test split, and neither is reported without an interval.

The bin count matters and cannot be a constant. Too few and the plot hides the
miscalibration; too many and each bin's realised uplift is a difference between two tiny
averages, which is noise wearing a decimal point. Ten is the convention and it is the
default here, but ten bins of IHDP's 150-row test split leaves under three treated units per
bin, and most bins then contain no treated unit at all and report nothing. So the requested
count is capped by the smaller arm, at :data:`MIN_PER_ARM` units of each arm per bin, and
the cap is reported rather than applied silently. On IHDP that means two bins and a
calibration estimate too weak to lean on, which is the honest description of what a
150-row test split with 28 treated units can support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itx.metrics.curves import DEFAULT_TIE_SEED, rank_order

if TYPE_CHECKING:
    from itx.types import FloatArray, IntArray

DEFAULT_BINS = 10

#: Fewest units of each arm a bin needs before its realised uplift is worth computing. Below
#: this a bin's estimate is a difference between two averages of a handful of rows.
MIN_PER_ARM = 10


def usable_bins(
    treatment: IntArray, *, requested: int = DEFAULT_BINS, min_per_arm: int = MIN_PER_ARM
) -> int:
    """How many bins this sample can actually support.

    Args:
        treatment: Binary treatment indicator per unit.
        requested: The bin count that was asked for.
        min_per_arm: Fewest units of each arm a bin needs.

    Returns:
        The requested count, capped so that a typical bin holds ``min_per_arm`` units of
        each arm, and never below 2.
    """
    treated = int(treatment.sum())
    control = int(treatment.size - treated)
    affordable = min(treated // min_per_arm, control // min_per_arm)
    return max(2, min(requested, affordable))


@dataclass(frozen=True, slots=True)
class CalibrationTable:
    """Predicted against realised uplift, one row per bin.

    Attributes:
        predicted: Mean predicted uplift in each bin, ascending.
        realised: Uplift actually observed in each bin, treated mean minus control mean.
        standard_error: Standard error of the realised uplift in each bin.
        n_units: Units in each bin.
        n_treated: Treated units in each bin.
    """

    predicted: FloatArray
    realised: FloatArray
    standard_error: FloatArray
    n_units: IntArray
    n_treated: IntArray

    @property
    def usable(self) -> np.ndarray:
        """Bins where both arms are present and the realised uplift is defined."""
        return np.isfinite(self.realised)

    def describe(self) -> str:
        """One line: how many bins were usable, and the worst gap among them."""
        usable = self.usable
        if not usable.any():
            return "calibration: no bin had both arms"
        gap = np.abs(self.predicted[usable] - self.realised[usable])
        return (
            f"calibration: {int(usable.sum())} of {self.predicted.size} bins usable, "
            f"largest gap {gap.max():.4f}"
        )


def calibration_table(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    n_bins: int = DEFAULT_BINS,
    seed: int = DEFAULT_TIE_SEED,
) -> CalibrationTable:
    """Bin by predicted uplift and measure the uplift realised in each bin.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Predicted uplift per unit.
        n_bins: Number of equal-sized bins.
        seed: Seed for tie-breaking, shared with the ranking metrics so a bin holds the
            same people a curve at that point would.

    Returns:
        The table, ascending in predicted uplift. A bin missing one arm has NaN for its
        realised uplift rather than a substituted zero: the sample cannot say what happened
        there, and pretending otherwise would drag a summary toward zero.

    Raises:
        ValueError: If ``n_bins`` is less than 2 or exceeds the sample size.
    """
    if n_bins < 2:
        msg = f"n_bins must be at least 2, got {n_bins}"
        raise ValueError(msg)
    if n_bins > outcome.size:
        msg = f"n_bins ({n_bins}) cannot exceed the sample size ({outcome.size})"
        raise ValueError(msg)
    n_bins = usable_bins(treatment, requested=n_bins)

    # Ascending, so the plot reads left to right from least to most predicted uplift.
    order = rank_order(scores, seed=seed)[::-1]
    bins = np.array_split(order, n_bins)

    predicted = np.empty(n_bins, dtype=np.float64)
    realised = np.empty(n_bins, dtype=np.float64)
    standard_error = np.empty(n_bins, dtype=np.float64)
    counts = np.empty(n_bins, dtype=np.int64)
    treated_counts = np.empty(n_bins, dtype=np.int64)

    for index, rows in enumerate(bins):
        treated = treatment[rows] == 1
        predicted[index] = scores[rows].mean()
        counts[index] = rows.size
        treated_counts[index] = int(treated.sum())
        if not treated.any() or treated.all():
            realised[index] = np.nan
            standard_error[index] = np.nan
            continue
        treated_outcome = outcome[rows][treated]
        control_outcome = outcome[rows][~treated]
        realised[index] = treated_outcome.mean() - control_outcome.mean()
        standard_error[index] = np.sqrt(
            treated_outcome.var(ddof=1) / treated_outcome.size
            + control_outcome.var(ddof=1) / control_outcome.size
        )

    return CalibrationTable(
        predicted=predicted,
        realised=realised,
        standard_error=standard_error,
        n_units=counts,
        n_treated=treated_counts,
    )


def calibration_slope(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    n_bins: int = DEFAULT_BINS,
    seed: int = DEFAULT_TIE_SEED,
) -> float:
    """Slope of realised uplift on predicted uplift across bins. One is perfect.

    Bins are weighted by the inverse variance of their realised uplift, so a bin whose
    estimate is barely distinguishable from noise does not get the same say as a precise
    one. A bin with a zero standard error, which happens when an arm has a single unit, is
    dropped rather than given infinite weight.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Predicted uplift per unit.
        n_bins: Number of equal-sized bins.
        seed: Seed for tie-breaking.

    Returns:
        The weighted least-squares slope, or NaN when fewer than two bins are usable or
        every bin has the same prediction.
    """
    table = calibration_table(outcome, treatment, scores, n_bins=n_bins, seed=seed)
    usable = table.usable & (table.standard_error > 0.0)
    if usable.sum() < 2:
        return float("nan")

    x = table.predicted[usable]
    y = table.realised[usable]
    weight = 1.0 / table.standard_error[usable] ** 2

    mean_x = np.average(x, weights=weight)
    variance_x = np.average((x - mean_x) ** 2, weights=weight)
    if variance_x == 0.0:
        return float("nan")
    covariance = np.average((x - mean_x) * (y - np.average(y, weights=weight)), weights=weight)
    return float(covariance / variance_x)


def calibration_error(
    outcome: FloatArray,
    treatment: IntArray,
    scores: FloatArray,
    *,
    n_bins: int = DEFAULT_BINS,
    seed: int = DEFAULT_TIE_SEED,
) -> float:
    """Mean absolute gap between predicted and realised uplift across bins. Zero is perfect.

    Args:
        outcome: Observed outcome per unit.
        treatment: Binary treatment indicator per unit.
        scores: Predicted uplift per unit.
        n_bins: Number of equal-sized bins.
        seed: Seed for tie-breaking.

    Returns:
        The mean absolute gap in the outcome's units, or NaN when no bin is usable.
    """
    table = calibration_table(outcome, treatment, scores, n_bins=n_bins, seed=seed)
    usable = table.usable
    if not usable.any():
        return float("nan")
    return float(np.mean(np.abs(table.predicted[usable] - table.realised[usable])))
