"""Precompute what the static demo page needs, so the page itself does no modelling.

PLAN.md section 7: the demo is static because the budget is CA$25 and a static page is a
URL a hiring manager can click. That constraint decides the whole design. Every number the
slider can display has to exist in a JSON file before anyone opens the page, because there
is no backend to ask.

What the slider moves is the budget, and what changes with it is the answer to "who do we
treat and what does it buy". Both are already computed by this package at any budget, so
this module is arithmetic and packaging rather than new modelling:
:func:`itx.policy.policy_value.policy_metrics` reads every budget from one sorted pass, and
:func:`itx.metrics.bootstrap.bootstrap_vector` puts an interval on all of them from one set
of resamples.

## What is in the file, and what is deliberately not

Per dataset: the realised policy gain at every budget point on a fixed grid, doubly robust
and IPW, each with a bootstrap interval, for every estimator in the results file, plus the
expected curve under random targeting. Then the top of each estimator's ranking, as row
numbers with their predicted uplift, so the page can show who is actually in the treated
set rather than only how much it buys.

Not in the file: anything identifying. These are public datasets and the demo ships row
numbers, never the feature values, so there is nothing to re-identify even in principle
(PLAN.md section 7). And not the whole ranking either. Criteo's test split is 279,592 rows
and shipping it would be a 20 MB page load to show a list nobody scrolls, so the file
carries :data:`TOP_UNITS` per estimator and the page says how many more it is not showing.

## Why the random baseline is an expectation rather than a draw

The benchmark's random baseline is the mean of 200 random rankings, which is the right
thing there because it is being compared against with an interval. Here the curve is drawn
at fifty budgets and a reader is looking at its shape, so it is computed exactly instead:
the doubly robust gain from treating a random share ``b`` of the population is ``b`` times
the gain from treating all of it, because a random subset carries the population's average
contribution. That is the same quantity the 200 draws estimate, without the wobble that
would invite someone to read the noise as structure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from itx.bench.runner import nuisances_for
from itx.metrics.bootstrap import DEFAULT_LEVEL, bootstrap_vector
from itx.metrics.curves import rank_order
from itx.policy.policy_value import dr_contributions, gain_key, policy_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from itx.bench.runner import BenchmarkRow
    from itx.policy.policy_value import Nuisances
    from itx.types import FloatArray, IntArray, Split, UpliftDataset

#: Budget points the curve is drawn at, as shares of the population. Fifty steps of two
#: percent: fine enough that the line looks continuous at the width a browser draws it,
#: coarse enough that the file stays small and the bootstrap stays quick. The three budgets
#: the results table reports, 10%, 20% and 30%, all land exactly on this grid, so a reader
#: can check the demo against the README rather than having to interpolate.
DEMO_BUDGETS: tuple[float, ...] = tuple(round(0.02 * step, 2) for step in range(1, 51))

#: How many of each ranking's top units travel with the file. Enough to scroll, small
#: enough that five datasets together stay under a megabyte.
TOP_UNITS = 200

#: Resamples behind the interval band. Fewer than the benchmark's thousand, because the
#: band is drawn at fifty budgets for seven estimators and the page is illustrating a
#: shape rather than settling a question. The README carries the numbers that settle it.
DEMO_RESAMPLES = 200


@dataclass(frozen=True, slots=True)
class Unit:
    """One row of a ranking, as the page will show it.

    Attributes:
        row: Position in the test split. A row number, never an identifier: the demo ships
            no feature values at all.
        uplift: The estimator's predicted effect for that row.
        cost: What treating it costs. Uniform at 1.0 on all five public datasets, so the
            cost-aware allocation of :mod:`itx.policy.cost_aware` reduces to rank-and-cut
            here, which is asserted in its tests. It is carried anyway because the field
            is what the fraud worked case varies, and a schema that gains a column later
            is worse than one that always had it.
    """

    row: int
    uplift: float
    cost: float


def build_payload(
    dataset: str,
    rows: Sequence[BenchmarkRow],
    split: Split,
    *,
    budgets: Sequence[float] = DEMO_BUDGETS,
    n_resamples: int = DEMO_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> dict[str, Any]:
    """Everything the page needs for one dataset, ready to serialise.

    Args:
        dataset: Dataset key, used as the file name and shown on the page.
        rows: Refitted rows carrying per-unit scores, from
            :func:`itx.bench.runner.refit_seed`.
        split: The partition those rows were fitted on.
        budgets: Budget points to evaluate at.
        n_resamples: Bootstrap resamples behind the band.
        level: Nominal coverage.
        seed: Seed for the resampling and for tie-breaking.

    Returns:
        A JSON-serialisable mapping.

    Raises:
        ValueError: If no rows carry scores, which means the caller passed results read
            back from disk rather than a refit.
    """
    scored = [row for row in rows if row.scores.size]
    if not scored:
        msg = (
            f"{dataset}: no rows carry per-unit scores. The demo needs a refit "
            f"(itx.bench.runner.refit_seed), not a results file, which does not store them."
        )
        raise ValueError(msg)

    test = split.test
    nuisances = nuisances_for(split)
    estimators: dict[str, Any] = {}
    for row in scored:
        estimators[row.estimator] = _curve(
            test,
            row.scores,
            nuisances=nuisances,
            budgets=budgets,
            n_resamples=n_resamples,
            level=level,
            seed=seed,
        ) | {"top": [asdict(unit) for unit in _top_units(row.scores, seed=seed)]}

    return {
        "dataset": dataset,
        "n_test": int(test.n_units),
        "n_treated": int(test.n_treated),
        "outcome_is_binary": bool(test.outcome_is_binary),
        "seed": int(split.seed),
        "budgets": [float(b) for b in budgets],
        "level": float(level),
        "n_resamples": int(n_resamples),
        "top_units": TOP_UNITS,
        "random": _random_curve(test, nuisances, budgets=budgets),
        "estimators": estimators,
    }


#: Name of the file the page loads first, to learn which datasets exist.
INDEX_NAME = "index.json"


def write_payloads(payloads: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Write one JSON file per dataset, then rebuild the index from everything present.

    The index is built from the directory rather than from ``payloads``, and that is a fix
    rather than a preference. Rebuilding one dataset is the normal thing to want, and an
    index written from the payloads just built would silently drop every dataset that was
    not in the batch: the files stay on disk, the page stops offering them, and nothing
    errors. That happened while building this demo in two batches, and the page came out
    listing three of the five datasets sitting next to it.

    Args:
        payloads: Results of :func:`build_payload`.
        out_dir: Directory to write into; created if missing.

    Returns:
        The paths written, index last.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for payload in payloads:
        path = out_dir / f"{payload['dataset']}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        written.append(path)

    index = out_dir / INDEX_NAME
    index.write_text(
        json.dumps({"datasets": _catalogue(out_dir)}, separators=(",", ":")),
        encoding="utf-8",
    )
    written.append(index)
    return written


def _catalogue(out_dir: Path) -> list[dict[str, Any]]:
    """What the index says about every payload file in a directory, in name order.

    Args:
        out_dir: Directory holding the payloads.

    Returns:
        One entry per dataset, carrying only what the page needs before it has loaded the
        dataset itself: the key to fetch, and enough to label the picker.
    """
    entries: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == INDEX_NAME:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "key": payload["dataset"],
                "n_test": payload["n_test"],
                "outcome_is_binary": payload["outcome_is_binary"],
            }
        )
    return entries


def _curve(
    test: UpliftDataset,
    scores: FloatArray,
    *,
    nuisances: Nuisances,
    budgets: Sequence[float],
    n_resamples: int,
    level: float,
    seed: int,
) -> dict[str, Any]:
    """One estimator's gain curve, with a band, at every budget.

    All budgets and both estimators come out of one set of resamples, which is what
    :func:`itx.metrics.bootstrap.bootstrap_vector` is for: the expensive part is sorting
    the ranking, and it is sorted once per resample rather than once per budget.

    Args:
        test: The test split.
        scores: Predicted uplift per test row.
        nuisances: Nuisances aligned with the test rows.
        budgets: Budget points.
        n_resamples: Bootstrap resamples.
        level: Nominal coverage.
        seed: Seed for resampling and tie-breaking.

    Returns:
        Point estimates and bounds per estimator, each a list aligned with ``budgets``.
    """

    def statistics(index: IntArray) -> dict[str, float]:
        return policy_metrics(
            test.outcome[index],
            test.treatment[index],
            scores[index],
            nuisances.take(index),
            budgets=budgets,
            seed=seed,
        )

    estimates = bootstrap_vector(
        statistics, test.n_units, n_resamples=n_resamples, level=level, seed=seed
    )
    curve: dict[str, Any] = {}
    for name in ("dr", "ipw"):
        chosen = [estimates[gain_key(name, budget)] for budget in budgets]
        curve[name] = {
            "value": [round(e.value, 6) for e in chosen],
            "low": [round(e.low, 6) for e in chosen],
            "high": [round(e.high, 6) for e in chosen],
        }
    return curve


def _random_curve(
    test: UpliftDataset, nuisances: Nuisances, *, budgets: Sequence[float]
) -> dict[str, list[float]]:
    """The doubly robust gain expected from treating a random share of the population.

    Exact rather than sampled: a random subset carries the population's average per-unit
    contribution, so the gain at share ``b`` is ``b`` times the gain from treating
    everybody. See the module docstring for why the benchmark's 200 draws are not reused.

    Args:
        test: The test split.
        nuisances: Nuisances aligned with it.
        budgets: Budget points.

    Returns:
        The curve, aligned with ``budgets``.
    """
    everybody = float(dr_contributions(test.outcome, test.treatment, nuisances).mean())
    return {"dr": [round(everybody * float(budget), 6) for budget in budgets]}


def _top_units(scores: FloatArray, *, seed: int, limit: int = TOP_UNITS) -> list[Unit]:
    """The head of one ranking, in the order the policy would treat them.

    Ordered by :func:`itx.metrics.curves.rank_order` rather than by a fresh sort, so the
    people the page lists are exactly the people the curve beside it is pricing. Two
    different orderings of the same scores would be a very hard thing to notice.

    Args:
        scores: Predicted uplift per test row.
        seed: Tie-breaking seed, matching the one the curve used.
        limit: How many to keep.

    Returns:
        Up to ``limit`` units, best first.
    """
    order = rank_order(scores, seed=seed)[:limit]
    return [
        Unit(row=int(position), uplift=round(float(scores[position]), 6), cost=1.0)
        for position in order
    ]
