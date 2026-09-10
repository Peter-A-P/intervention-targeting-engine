"""Metrics that need the truth, and therefore only work on simulated data.

PEHE and absolute ATE error are the only measurements in this package that compare a
predicted effect with the effect itself. Everywhere else the individual effect is missing
by construction and the best available check is whether a ranking buys outcome, which a
self-consistent but wrong estimator can pass. That is the whole reason the benchmark
carries IHDP and ACIC alongside the three randomised sets (PLAN.md section 1).

Neither metric means anything on a dataset where ``true_effect`` is None, and both raise
rather than quietly substituting an estimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from itx.types import FloatArray


def pehe(predicted: FloatArray, true_effect: FloatArray) -> float:
    """Root mean squared error of the individual treatment effect.

    The standard IHDP and ACIC score, usually written as the square root of the precision
    in estimating heterogeneous effects. It is an error, so lower is better, and it is on
    the scale of the outcome.

    Args:
        predicted: Predicted per-unit effect.
        true_effect: True per-unit effect.

    Returns:
        The root mean squared difference.

    Raises:
        ValueError: If the two arrays are different lengths or empty.
    """
    _check(predicted, true_effect)
    return float(np.sqrt(np.mean((predicted - true_effect) ** 2)))


def ate_error(predicted: FloatArray, true_effect: FloatArray) -> float:
    """Absolute error of the average treatment effect.

    Reported next to PEHE because they fail differently and a reader needs both: an
    estimator can get the population average exactly right while ranking individuals no
    better than a coin, which is the difference between a number for a report and a
    targeting decision.

    Args:
        predicted: Predicted per-unit effect.
        true_effect: True per-unit effect.

    Returns:
        The absolute difference of the two means.

    Raises:
        ValueError: If the two arrays are different lengths or empty.
    """
    _check(predicted, true_effect)
    return float(abs(predicted.mean() - true_effect.mean()))


def _check(predicted: FloatArray, true_effect: FloatArray) -> None:
    """Reject inputs that cannot be compared."""
    if predicted.shape != true_effect.shape:
        msg = f"shape mismatch: predicted {predicted.shape}, true {true_effect.shape}"
        raise ValueError(msg)
    if predicted.size == 0:
        msg = "cannot score an empty sample"
        raise ValueError(msg)
