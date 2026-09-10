"""Turning a ranking into an allocation: treat the top of the list until the budget runs out.

A ranked list is not yet a decision. This is the simplest rule that turns one into a
decision, and it is the right rule when every intervention costs the same. The cost-aware
variant, where a fraud review costs an analyst an hour and a text message costs a fraction
of a cent, arrives in week 5 (PLAN.md section 6).

``positive_only`` is off by default, and the reason is worth stating. Treating a unit whose
predicted effect is negative is never right on its own terms: it spends money to make the
outcome worse. But turning those units off changes how much of the budget is spent, and
two policies that spend different amounts are not comparable at a fixed budget. The default
therefore treats exactly the requested share of the population, which is what the results
table compares, and the flag is there for the separate question of what the best policy
would do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from itx.metrics.curves import DEFAULT_TIE_SEED, rank_order

if TYPE_CHECKING:
    from itx.types import BoolArray, FloatArray


def n_targeted(n_units: int, budget: float) -> int:
    """How many units a budget of ``budget`` of the population covers.

    Rounds up, so a budget of 10% of 15 units treats 2 rather than 1: a budget holder who
    says "the top ten percent" is describing a cutoff, not a maximum spend.

    Args:
        n_units: Population size.
        budget: Share of the population, in ``(0, 1]``.

    Returns:
        The number of units to treat, at least 1.

    Raises:
        ValueError: If the budget is outside ``(0, 1]``.
    """
    if not 0.0 < budget <= 1.0:
        msg = f"budget must be a share of the population in (0, 1], got {budget}"
        raise ValueError(msg)
    return max(1, int(np.ceil(budget * n_units)))


def rank_and_cut(
    scores: FloatArray,
    budget: float,
    *,
    positive_only: bool = False,
    seed: int = DEFAULT_TIE_SEED,
) -> BoolArray:
    """Treat the highest-scoring units the budget covers.

    Args:
        scores: Predicted uplift per unit, higher meaning treat sooner.
        budget: Share of the population the budget covers, in ``(0, 1]``.
        positive_only: Also refuse to treat units whose predicted effect is not positive,
            leaving part of the budget unspent.
        seed: Seed for tie-breaking, shared with the metrics so a curve and a policy at
            the same budget always describe the same set of people.

    Returns:
        A boolean mask, True for the units to treat.
    """
    order = rank_order(scores, seed=seed)
    chosen = order[: n_targeted(scores.size, budget)]
    mask: BoolArray = np.zeros(scores.size, dtype=bool)
    mask[chosen] = True
    if positive_only:
        mask &= scores > 0.0
    return mask
