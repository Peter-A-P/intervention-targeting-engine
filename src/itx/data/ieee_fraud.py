"""The fraud worked case: real features, invented effect, and it says so before anything else.

**Semi-synthetic. The features are real and the treatment effect is invented.**

That sentence is the first one on purpose, because everything below is only worth reading if
nobody mistakes it for a measured effect of fraud review. IEEE-CIS Fraud Detection supplies
590,540 real transactions with real features and a real ``isFraud`` label. The treatment,
"this transaction was sent to a human analyst", was never recorded in it, so it is simulated
here, and so is what the analyst then does. The purpose is to show the allocation logic on a
fraud-shaped problem (PLAN.md section 3), not to claim anything about fraud review.

Everything invented is in one function, :func:`simulate`, with its constants named below so
the effect function is published rather than described. Card: docs/data/ieee-fraud.md.

## What the simulation says happens

A transaction is worth ``amount``. Left alone, a fraudulent one is charged back and the
business loses that amount; a legitimate one completes and the business keeps a margin on it.
Sent to review, two things can happen that would not otherwise:

* a fraudulent transaction is **caught**, with probability :func:`catch_probability`, and the
  loss is avoided;
* a legitimate transaction is **wrongly declined**, with probability
  :func:`decline_probability`, and the margin on a real sale is lost.

So the effect of reviewing is ``+amount * catch`` on a fraudulent transaction and
``-MARGIN * amount * decline`` on a legitimate one. Both are known per unit, because the
simulation wrote them, which is why this dataset carries ``true_effect`` and can be scored on
PEHE like the two semi-synthetic benchmark sets.

## The one design decision that matters

Review is hardest on exactly the transactions that look riskiest. :func:`catch_probability`
falls as a transaction's fraud signal rises, because the fraud a simple rule would have
stopped is not the fraud that reaches an analyst's queue: what gets there is the practised
kind, and a human staring at it has a worse chance than the base rate suggests.

That single choice is what makes this a worked case for this repository rather than a
demonstration that expensive things are worth doing. It puts the highest-risk transactions in
the lost-causes quadrant, where review buys least, and it means **ranking the queue by fraud
risk is not ranking it by what review is worth**. That is the trap the whole project is about,
arriving on a problem shaped like the ones it is aimed at. It is an assumption, it is stated
here, and a reader who thinks real review works the other way can change one constant and
rerun.

## Assignment is randomised, deliberately

Half the transactions are reviewed, by a fair coin that depends on nothing. Real review
queues are not randomised and that is what makes real fraud data hard, but this case is about
allocation under a budget rather than about identification, and the repository already has
two datasets with real confounding to make the other point. So the propensity is known and
exactly 0.5, and any gap between a ranking and the truth here is the ranking's fault rather
than the design's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from itx.data.download import fetch
from itx.types import UpliftDataset

if TYPE_CHECKING:
    from pathlib import Path

    from itx.types import FloatArray, IntArray

#: Share of transactions sent to review. A fair coin, so the propensity is known.
KNOWN_PROPENSITY = 0.5

#: What the business keeps on a completed legitimate transaction, as a share of its value.
#: Three percent is a card-economics figure rather than a measured one, and it sets how much
#: a wrongly declined sale costs relative to a caught fraud. Raising it makes false positives
#: hurt more and pulls the optimal queue away from anything that resembles a risk ranking.
MARGIN = 0.03

#: Chance an analyst catches a fraudulent transaction with no fraud signal at all, and the
#: amount that chance falls by across the full range of the signal. 0.85 down to 0.30: review
#: is good at obvious fraud and poor at practised fraud. See the module docstring; this pair
#: is the assumption the whole case rests on.
CATCH_AT_ZERO_SIGNAL = 0.85
CATCH_FALL = 0.55

#: Chance an analyst wrongly declines a legitimate transaction, floor and rise, against the
#: same signal. A legitimate transaction that looks like fraud is the one that gets declined.
DECLINE_AT_ZERO_SIGNAL = 0.01
DECLINE_RISE = 0.14

#: Analyst minutes a review takes: a fixed cost plus a variable one that rises with how much
#: the transaction's identity fields are missing. A queue where every review costs the same
#: is a queue where the cost-aware knapsack reduces to rank-and-cut, and this case exists
#: partly to exercise the knapsack (PLAN.md section 7).
REVIEW_MINUTES_BASE = 8.0
REVIEW_MINUTES_PER_MISSING = 22.0

#: Columns kept as features. The V1-V339 block is dropped: it is 339 anonymised engineered
#: columns that trebles the fit time and adds nothing a reader of this case can interpret.
#: What is left is the part a fraud analyst would recognise.
NUMERIC: tuple[str, ...] = (
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    *(f"C{i}" for i in range(1, 15)),
    *(f"D{i}" for i in range(1, 16)),
)

#: Kept as categories, integer-encoded with a code reserved for missing.
CATEGORICAL: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
)

FEATURES: tuple[str, ...] = (*NUMERIC, *CATEGORICAL)

#: Columns read from the file but never offered to a model. ``isFraud`` decides the
#: counterfactual, so a model given it would be reading the answer.
LABEL = "isFraud"

#: Feature column holding what a review of that transaction costs, in analyst minutes. It is
#: a feature because it is observable before the decision is made, and it is the column the
#: cost-aware allocation spends against.
COST_COLUMN = "review_minutes"


def load_ieee_fraud(
    *,
    fraction: float = 1.0,
    seed: int = 20260914,
    path: Path | None = None,
) -> UpliftDataset:
    """Load the fraud worked case: real features, simulated review and simulated effect.

    Args:
        fraction: Share of rows to keep, stratified on the fraud label so the 3.50% base
            rate is preserved exactly. 1.0 keeps all 590,540.
        seed: Seed for the subsample, the review coin and the simulated outcomes. Fixed by
            default so the case is the same case on every machine.
        path: Read from this file instead of the download cache. For tests.

    Returns:
        The dataset, carrying the known propensity of 0.5 and the ``true_effect`` the
        simulation wrote, in dollars of retained value per transaction.
    """
    source = path if path is not None else fetch("ieee-fraud-train", quiet=True)
    frame = pl.read_csv(
        source, columns=[LABEL, *FEATURES], infer_schema_length=20_000, null_values=["NotFound"]
    )
    if fraction < 1.0:
        frame = _stratified_subsample(frame, fraction, seed)

    fraud = frame[LABEL].cast(pl.Int64).to_numpy()
    amount = frame["TransactionAmt"].cast(pl.Float64).to_numpy()
    signal = _fraud_signal(frame)
    missing = _missing_share(frame)

    treatment, outcome, effect = simulate(fraud, amount, signal, seed=seed)

    features = frame.select(
        *(pl.col(name).cast(pl.Float64) for name in NUMERIC),
        *(_encode(name) for name in CATEGORICAL),
        pl.Series(COST_COLUMN, REVIEW_MINUTES_BASE + REVIEW_MINUTES_PER_MISSING * missing),
    )
    return UpliftDataset(
        name="ieee-fraud",
        features=features,
        treatment=treatment,
        outcome=outcome,
        categorical=CATEGORICAL,
        propensity=np.full(frame.height, KNOWN_PROPENSITY),
        true_effect=effect,
    )


def simulate(
    fraud: IntArray, amount: FloatArray, signal: FloatArray, *, seed: int
) -> tuple[IntArray, FloatArray, FloatArray]:
    """The whole invention, in one place: who was reviewed, what happened, what review did.

    Args:
        fraud: The real ``isFraud`` label, 1 for a fraudulent transaction.
        amount: Transaction value in dollars.
        signal: Fraud signal per transaction, in ``[0, 1]``; see :func:`_fraud_signal`.
        seed: Seed for the review coin and the two outcome draws.

    Returns:
        The review indicator, the realised outcome in dollars of retained value, and the
        true per-unit effect of review, which is what PEHE is scored against.
    """
    rng = np.random.default_rng(seed)
    reviewed: IntArray = (rng.random(fraud.size) < KNOWN_PROPENSITY).astype(np.int64)

    caught = rng.random(fraud.size) < catch_probability(signal)
    declined = rng.random(fraud.size) < decline_probability(signal)
    is_fraud = fraud == 1

    # Left alone: a fraud is charged back, a legitimate sale earns its margin.
    untreated = np.where(is_fraud, -amount, MARGIN * amount)
    # Reviewed: a caught fraud costs nothing, a wrongly declined sale earns nothing.
    treated = np.where(
        is_fraud,
        np.where(caught, 0.0, -amount),
        np.where(declined, 0.0, MARGIN * amount),
    )
    outcome: FloatArray = np.where(reviewed == 1, treated, untreated)

    # The expected effect rather than the realised one, which is the quantity an estimator
    # is trying to recover: a per-unit effect defined by one coin flip would be unlearnable
    # and PEHE against it would measure the coin.
    effect: FloatArray = np.where(
        is_fraud,
        amount * catch_probability(signal),
        -MARGIN * amount * decline_probability(signal),
    )
    return reviewed, outcome, effect


def catch_probability(signal: FloatArray) -> FloatArray:
    """How often review catches a fraudulent transaction, falling as it looks more like fraud.

    The assumption the case rests on, stated in the module docstring: obvious fraud is
    stopped before it reaches a queue, so what an analyst sees is the practised kind.

    Args:
        signal: Fraud signal in ``[0, 1]``.

    Returns:
        A probability per transaction.
    """
    caught: FloatArray = CATCH_AT_ZERO_SIGNAL - CATCH_FALL * signal
    return caught


def decline_probability(signal: FloatArray) -> FloatArray:
    """How often review wrongly declines a legitimate transaction, rising with the signal.

    Args:
        signal: Fraud signal in ``[0, 1]``.

    Returns:
        A probability per transaction.
    """
    declined: FloatArray = DECLINE_AT_ZERO_SIGNAL + DECLINE_RISE * signal
    return declined


def _fraud_signal(frame: pl.DataFrame) -> FloatArray:
    """How much a transaction looks like fraud, in ``[0, 1]``, from observable columns only.

    Built from three real columns rather than from the label, so that a model fitted on the
    features can see what drives the effect. It is deliberately crude: this is the
    simulation's own notion of "looks risky", not a fraud model, and a good fraud model on
    this data would beat it easily.

    Args:
        frame: The loaded columns.

    Returns:
        The signal per row.
    """
    parts = []
    for name, invert in (("TransactionAmt", False), ("C1", False), ("D1", True)):
        values = frame[name].cast(pl.Float64).to_numpy()
        ranked = _rank_uniform(values)
        parts.append(1.0 - ranked if invert else ranked)
    signal: FloatArray = np.clip(np.mean(parts, axis=0), 0.0, 1.0)
    return signal


def _rank_uniform(values: FloatArray) -> FloatArray:
    """Column ranks mapped onto ``[0, 1]``, with missing values sent to the middle.

    Args:
        values: One column, possibly holding NaN.

    Returns:
        The rank of each value as a share of the column, 0.5 where it was missing.
    """
    present = np.isfinite(values)
    out = np.full(values.size, 0.5, dtype=np.float64)
    if not present.any():
        return out
    order = np.argsort(values[present], kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    out[present] = ranks / max(order.size - 1, 1)
    return out


def _missing_share(frame: pl.DataFrame) -> FloatArray:
    """Share of a transaction's numeric fields that are missing, which sets its review cost.

    Args:
        frame: The loaded columns.

    Returns:
        A value in ``[0, 1]`` per row.
    """
    block = frame.select(*(pl.col(name).cast(pl.Float64) for name in NUMERIC)).to_numpy()
    share: FloatArray = np.isnan(block).mean(axis=1)
    return share


def _encode(name: str) -> pl.Expr:
    """Integer-encode one string column, reserving 0 for missing.

    Codes come from the sorted distinct values, so they do not depend on row order and a
    subsample encodes the same category to the same integer as the full file.

    Args:
        name: Column name.

    Returns:
        An expression producing the encoded column.
    """
    codes = pl.col(name).cast(pl.Categorical).to_physical().cast(pl.Float64)
    return (codes.fill_null(-1.0) + 1.0).alias(name)


def _stratified_subsample(frame: pl.DataFrame, fraction: float, seed: int) -> pl.DataFrame:
    """Keep a share of the rows with the fraud rate preserved exactly.

    Args:
        frame: The loaded columns.
        fraction: Share to keep, in ``(0, 1]``.
        seed: Seed for the draw.

    Returns:
        The subsample, in the original row order.

    Raises:
        ValueError: If the fraction is not in ``(0, 1]``.
    """
    if not 0.0 < fraction <= 1.0:
        msg = f"fraction must be in (0, 1], got {fraction}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    label = frame[LABEL].to_numpy()
    keep: list[IntArray] = []
    for arm in (0, 1):
        rows = np.flatnonzero(label == arm)
        take = max(1, round(rows.size * fraction))
        keep.append(rng.choice(rows, size=take, replace=False))
    return frame[np.sort(np.concatenate(keep))]
