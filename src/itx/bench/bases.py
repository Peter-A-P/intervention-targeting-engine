"""Two base learners under the same five meta-learners, compared on the same rows.

The main benchmark holds the base learner fixed so that the estimator column means
something. This module asks the question that design deliberately leaves open: if the
engine under every meta-learner is changed, does the answer change? Two results come out of
it, and ``itx bases`` writes both:

1. **Does the base change the ranking?** Qini, AUUC, the realised value a 10% budget buys
   and, where the dataset knows the true effect, PEHE, for each meta-learner on each base,
   with the TabFM minus LightGBM difference as its own bootstrap interval.
2. **What was the tuning grid buying?** LightGBM pays for a validation-grid search before
   its final fit; TabFM has nothing to search and pays at inference instead. Both bills are
   recorded in seconds, per fit.

**Why the difference has its own interval.** The two bases are scored on the same test rows
of the same split, so the right comparison is paired: each bootstrap resample draws one set
of rows and scores both rankings on it, and the difference is taken inside the resample.
Two separate intervals that overlap say nothing about whether the paired difference is zero,
and with the test splits this benchmark uses they usually overlap.

**Why the LightGBM side is refitted rather than read back.** The results files keep metrics
but not per-unit scores, and a paired interval needs the scores. Refitting is cheap next to
TabFM, and it is a check as well as a cost: the LightGBM numbers here come out of the same
seeds, grid and selection rule as the main table, so they must match it, and
``tests/test_bases.py`` asserts that they do.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itx.bench.grid import select_config
from itx.bench.runner import (
    DATASETS,
    ESTIMATORS,
    _duration,
    _statistics,
    nuisances_for,
)
from itx.bench.seeds import SEEDS, TIE_SEED, bootstrap_seed_for
from itx.data.splits import stratified_split
from itx.estimators.base import BaseUpliftEstimator, DegenerateFitError
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, LearnerConfig, config_label
from itx.estimators.tabfm_base import DEFAULT_TABFM, TabFMConfig, pretrained
from itx.metrics.bootstrap import DEFAULT_LEVEL, DEFAULT_RESAMPLES, Estimate, bootstrap_vector

if TYPE_CHECKING:
    from pathlib import Path

    from itx.policy.policy_value import Nuisances
    from itx.types import FloatArray, Split

#: The five meta-learners the second base goes under. Dragonnet and the baselines are not
#: here: Dragonnet has no base learner to swap, and a baseline built on a different engine
#: would no longer be the baseline the main table shows.
META_LEARNERS: tuple[str, ...] = (
    "s-learner",
    "t-learner",
    "x-learner",
    "dr-learner",
    "r-learner",
)

#: The budget the realised value is read at.
BUDGET = 0.1

#: Metrics compared, in table order. PEHE is present only where the dataset has a truth.
COMPARED: tuple[str, ...] = ("qini", "auuc", f"gain_dr@{BUDGET:.0%}", "pehe")

#: Which way is better for each metric, so a table can say who won without a reader having
#: to remember that PEHE is an error.
HIGHER_IS_BETTER: dict[str, bool] = {
    "qini": True,
    "auuc": True,
    f"gain_dr@{BUDGET:.0%}": True,
    "pehe": False,
}

BASES: tuple[str, ...] = ("lightgbm", "tabfm")


@dataclass(frozen=True, slots=True)
class BaseComparisonRow:
    """One meta-learner on one split, fitted on both bases and scored on the same rows.

    Attributes:
        dataset: Dataset name.
        learner: Meta-learner name, without a base suffix.
        seed: The split seed.
        context_rows: TabFM's context cap for this row.
        ensemble: TabFM's ensemble size for this row.
        n_train: Training rows both bases were fitted on.
        n_test: Test rows both were scored on.
        metrics: ``lightgbm:<metric>``, ``tabfm:<metric>`` and ``delta:<metric>`` (TabFM
            minus LightGBM), each with its paired bootstrap interval.
        lightgbm_config: The configuration LightGBM's grid selected.
        lightgbm_select_seconds: Wall-clock of the grid search.
        lightgbm_fit_seconds: Wall-clock of the final fit.
        lightgbm_predict_seconds: Wall-clock of predicting the test rows.
        tabfm_fit_seconds: Wall-clock of TabFM's fit, which for most meta-learners includes
            predicting training rows, because their later stages are fitted on earlier
            stages' predictions.
        tabfm_predict_seconds: Wall-clock of predicting the test rows.
    """

    dataset: str
    learner: str
    seed: int
    context_rows: int
    ensemble: int
    n_train: int
    n_test: int
    metrics: dict[str, Estimate]
    lightgbm_config: str
    lightgbm_select_seconds: float
    lightgbm_fit_seconds: float
    lightgbm_predict_seconds: float
    tabfm_fit_seconds: float
    tabfm_predict_seconds: float

    @property
    def key(self) -> tuple[str, int, int, int]:
        """What identifies a row for resuming and merging."""
        return (self.learner, self.seed, self.context_rows, self.ensemble)

    @property
    def lightgbm_seconds(self) -> float:
        """LightGBM's whole bill: search, final fit and prediction."""
        return (
            self.lightgbm_select_seconds
            + self.lightgbm_fit_seconds
            + self.lightgbm_predict_seconds
        )

    @property
    def tabfm_seconds(self) -> float:
        """TabFM's whole bill: there is no search, so fit and prediction."""
        return self.tabfm_fit_seconds + self.tabfm_predict_seconds


def compare(
    dataset: str,
    learner: str,
    split: Split,
    *,
    tabfm: TabFMConfig = DEFAULT_TABFM,
    nuisances: Nuisances | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    tie_seed: int = TIE_SEED,
    tune: bool = True,
) -> BaseComparisonRow:
    """Fit one meta-learner on both bases and score both on the same test rows.

    Args:
        dataset: Dataset name, for the row.
        learner: A meta-learner from :data:`META_LEARNERS`.
        split: The partition.
        tabfm: TabFM settings.
        nuisances: The policy-value referee for this split, fitted here if omitted.
        n_resamples: Bootstrap resamples for the paired intervals.
        level: Nominal coverage.
        tie_seed: Seed for tie-breaking inside rankings.
        tune: Run LightGBM's validation grid, as the main benchmark does. Off only in the
            fast tests.

    Returns:
        The finished row.

    Raises:
        KeyError: If the learner is not one of the five.
    """
    if learner not in META_LEARNERS:
        msg = f"unknown meta-learner {learner!r}; known: {', '.join(META_LEARNERS)}"
        raise KeyError(msg)
    factory = ESTIMATORS[learner]
    seed = split.seed
    test = split.test
    referee = nuisances if nuisances is not None else nuisances_for(split)

    started = time.perf_counter()
    selection_config: LearnerConfig = (
        select_config(factory, split, seed=seed).config if tune else DEFAULT_CONFIG
    )
    select_seconds = time.perf_counter() - started
    lightgbm_scores, lightgbm_fit, lightgbm_predict = _fit_and_score(
        factory(selection_config, seed), split
    )
    tabfm_scores, tabfm_fit, tabfm_predict = _fit_and_score(factory(tabfm, seed), split)

    statistics = {
        base: _statistics(
            test,
            scores,
            referee,
            budgets=(BUDGET,),
            tie_seed=tie_seed,
            with_ground_truth=True,
            with_calibration=False,
        )
        for base, scores in (("lightgbm", lightgbm_scores), ("tabfm", tabfm_scores))
    }

    def paired(index: np.ndarray) -> dict[str, float]:
        values = {base: function(index) for base, function in statistics.items()}
        out: dict[str, float] = {}
        for metric in COMPARED:
            if metric not in values["lightgbm"]:
                continue
            out[f"lightgbm:{metric}"] = values["lightgbm"][metric]
            out[f"tabfm:{metric}"] = values["tabfm"][metric]
            out[f"delta:{metric}"] = values["tabfm"][metric] - values["lightgbm"][metric]
        return out

    metrics = bootstrap_vector(
        paired,
        test.n_units,
        n_resamples=n_resamples,
        level=level,
        seed=bootstrap_seed_for(seed),
    )
    return BaseComparisonRow(
        dataset=dataset,
        learner=learner,
        seed=seed,
        context_rows=tabfm.context_rows,
        ensemble=tabfm.ensemble,
        n_train=split.train.n_units,
        n_test=test.n_units,
        metrics=metrics,
        lightgbm_config=config_label(selection_config),
        lightgbm_select_seconds=select_seconds,
        lightgbm_fit_seconds=lightgbm_fit,
        lightgbm_predict_seconds=lightgbm_predict,
        tabfm_fit_seconds=tabfm_fit,
        tabfm_predict_seconds=tabfm_predict,
    )


def _fit_and_score(
    estimator: BaseUpliftEstimator, split: Split
) -> tuple[FloatArray, float, float]:
    """Fit on train, predict test, and time both halves separately."""
    started = time.perf_counter()
    estimator.fit(split.train)
    fitted = time.perf_counter()
    scores = estimator.predict_uplift(split.test.features)
    predicted = time.perf_counter()
    if not np.isfinite(scores).all():
        msg = f"{estimator.name} on {split.test.name} seed {split.seed}: non-finite scores"
        raise DegenerateFitError(msg)
    return scores, fitted - started, predicted - fitted


def run_bases(
    dataset: str,
    *,
    learners: Sequence[str] = META_LEARNERS,
    seeds: Sequence[int] = SEEDS,
    tabfm: TabFMConfig = DEFAULT_TABFM,
    n_resamples: int = DEFAULT_RESAMPLES,
    tune: bool = True,
    progress: Callable[[str], None] | None = None,
    completed: Sequence[BaseComparisonRow] = (),
    on_row: Callable[[BaseComparisonRow], None] | None = None,
    warm: bool = True,
) -> list[BaseComparisonRow]:
    """Run the comparison on one dataset, across learners and seeds.

    Args:
        dataset: A key of :data:`itx.bench.runner.DATASETS`.
        learners: Meta-learners to compare.
        seeds: Split seeds; the committed five by default.
        tabfm: TabFM settings, including the context cap.
        n_resamples: Bootstrap resamples per row.
        tune: Run LightGBM's validation grid.
        progress: Called with one line per finished row.
        completed: Rows from an interrupted run, reused rather than refitted.
        on_row: Called with each finished row, so the caller can checkpoint it. TabFM on a
            CPU is slow enough that losing a night's work to the scheduled reboot is the
            expected failure, not an edge case.
        warm: Load both TabFM checkpoints before the first timed fit.

    Returns:
        Every row, seed then learner.

    Raises:
        KeyError: If the dataset is not registered.
    """
    if dataset not in DATASETS:
        msg = f"unknown dataset {dataset!r}; known: {', '.join(sorted(DATASETS))}"
        raise KeyError(msg)
    data = DATASETS[dataset]()
    already = {row.key: row for row in completed}
    if warm:
        # Loading and verifying the two 6.5 GB checkpoints takes a minute, and charging it
        # to whichever fit happened to come first would put it in that row's timing.
        for model_type in ("classification", "regression"):
            pretrained(model_type, float32=tabfm.float32)
    started = time.perf_counter()
    rows: list[BaseComparisonRow] = []
    total = len(seeds) * len(learners)
    done = 0
    for seed in seeds:
        split: Split | None = None
        nuisances: Nuisances | None = None
        for learner in learners:
            done += 1
            finished = already.get((learner, seed, tabfm.context_rows, tabfm.ensemble))
            if finished is not None:
                rows.append(finished)
                if progress is not None:
                    progress(f"  [{done}/{total}] seed {seed} {learner}: from checkpoint")
                continue
            if split is None:
                split = stratified_split(data, seed)
            if nuisances is None:
                nuisances = nuisances_for(split)
            row = compare(
                dataset,
                learner,
                split,
                tabfm=tabfm,
                nuisances=nuisances,
                n_resamples=n_resamples,
                tune=tune,
            )
            rows.append(row)
            if on_row is not None:
                on_row(row)
            if progress is not None:
                progress(
                    f"  [{done}/{total}] seed {seed} {learner}: lightgbm "
                    f"{_duration(row.lightgbm_seconds)}, tabfm {_duration(row.tabfm_seconds)}, "
                    f"elapsed {_duration(time.perf_counter() - started)}"
                )
    return rows


def write_json(rows: Sequence[BaseComparisonRow], path: Path) -> None:
    """Write every per-seed number, so the tables can be redrawn without refitting.

    Args:
        rows: Rows to write.
        path: Destination file. Parent directories are created.
    """
    payload = [
        {
            "dataset": row.dataset,
            "learner": row.learner,
            "seed": row.seed,
            "context_rows": row.context_rows,
            "ensemble": row.ensemble,
            "n_train": row.n_train,
            "n_test": row.n_test,
            "lightgbm_config": row.lightgbm_config,
            "seconds": {
                "lightgbm_select": round(row.lightgbm_select_seconds, 3),
                "lightgbm_fit": round(row.lightgbm_fit_seconds, 3),
                "lightgbm_predict": round(row.lightgbm_predict_seconds, 3),
                "tabfm_fit": round(row.tabfm_fit_seconds, 3),
                "tabfm_predict": round(row.tabfm_predict_seconds, 3),
            },
            "metrics": {
                name: {
                    "value": estimate.value,
                    "low": estimate.low,
                    "high": estimate.high,
                    "level": estimate.level,
                    "n_resamples": estimate.n_resamples,
                }
                for name, estimate in row.metrics.items()
            },
        }
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> list[BaseComparisonRow]:
    """Rebuild rows from a file written by :func:`write_json`.

    Args:
        path: The file.

    Returns:
        The rows, in the order they were written.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        BaseComparisonRow(
            dataset=record["dataset"],
            learner=record["learner"],
            seed=record["seed"],
            context_rows=record["context_rows"],
            ensemble=record["ensemble"],
            n_train=record["n_train"],
            n_test=record["n_test"],
            metrics={name: Estimate(**metric) for name, metric in record["metrics"].items()},
            lightgbm_config=record["lightgbm_config"],
            lightgbm_select_seconds=record["seconds"]["lightgbm_select"],
            lightgbm_fit_seconds=record["seconds"]["lightgbm_fit"],
            lightgbm_predict_seconds=record["seconds"]["lightgbm_predict"],
            tabfm_fit_seconds=record["seconds"]["tabfm_fit"],
            tabfm_predict_seconds=record["seconds"]["tabfm_predict"],
        )
        for record in payload
    ]


def merge(
    existing: Sequence[BaseComparisonRow], new: Sequence[BaseComparisonRow]
) -> list[BaseComparisonRow]:
    """Combine a results file with a fresh run, the fresh row winning on the same key.

    A context-size sweep is several runs into one file, so a run must not erase the rows
    another context size wrote.

    Args:
        existing: Rows already on disk.
        new: Rows from this run.

    Returns:
        The union, ordered by context size, then seed, then learner order.
    """
    by_key = {row.key: row for row in existing}
    by_key.update({row.key: row for row in new})
    order = {name: index for index, name in enumerate(META_LEARNERS)}
    return sorted(
        by_key.values(),
        key=lambda row: (row.context_rows, row.ensemble, row.seed, order.get(row.learner, 99)),
    )


def _pooled(estimates: Sequence[Estimate]) -> Estimate | None:
    """Mean across seeds with the averaged interval, as the main table pools."""
    finite = [e for e in estimates if math.isfinite(e.value)]
    if not finite:
        return None
    return Estimate(
        value=float(np.mean([e.value for e in finite])),
        low=float(np.mean([e.low for e in finite])),
        high=float(np.mean([e.high for e in finite])),
        level=finite[0].level,
        n_resamples=sum(e.n_resamples for e in finite),
    )


def _label(metric: str) -> str:
    """Column label for a metric."""
    return {"qini": "Qini", "auuc": "AUUC", "pehe": "PEHE"}.get(
        metric, f"Value at a {BUDGET:.0%} budget (DR)"
    )


def _group(rows: Sequence[BaseComparisonRow]) -> dict[str, list[BaseComparisonRow]]:
    """Rows by learner, in learner order."""
    grouped: dict[str, list[BaseComparisonRow]] = {}
    for name in META_LEARNERS:
        chosen = [row for row in rows if row.learner == name]
        if chosen:
            grouped[name] = chosen
    return grouped


def headline(rows: Sequence[BaseComparisonRow]) -> list[BaseComparisonRow]:
    """The rows the headline tables report: the smallest ensemble, then the largest context.

    A results file can hold several TabFM settings, because the context and ensemble sweeps
    add to it. The headline is the committed setting: the ensemble the package defaults to,
    which is the smallest one run, at the full pretrained context.

    Args:
        rows: Rows for one dataset.

    Returns:
        The rows at that one setting.
    """
    if not rows:
        return []
    ensemble = min(row.ensemble for row in rows)
    at_ensemble = [row for row in rows if row.ensemble == ensemble]
    context = max(row.context_rows for row in at_ensemble)
    return [row for row in at_ensemble if row.context_rows == context]


def comparison_table(
    rows: Sequence[BaseComparisonRow], metric: str = "qini", *, digits: int = 4
) -> str:
    """Result 1 for one metric: each base, and the paired difference.

    One row per meta-learner: the metric on LightGBM, on TabFM, and TabFM minus LightGBM with
    its paired interval, then on how many seeds TabFM was the better of the two. The
    difference is in bold when its interval excludes zero.

    Args:
        rows: Rows at one TabFM setting.
        metric: The metric.
        digits: Decimal places.

    Returns:
        A markdown table, or an empty string when no row carries the metric.
    """
    grouped = _group(rows)
    if not any(f"delta:{metric}" in row.metrics for group in grouped.values() for row in group):
        return ""
    better = "lower" if not HIGHER_IS_BETTER[metric] else "higher"
    lines = [
        f"| {_label(metric)} ({better} is better) | LightGBM | TabFM "
        "| TabFM minus LightGBM | Seeds TabFM is better |",
        "|---|---|---|---|---|",
    ]
    for learner, group in grouped.items():
        cells = [f"`{learner}`"]
        for base in BASES:
            pooled = _pooled([row.metrics[f"{base}:{metric}"] for row in group])
            cells.append("-" if pooled is None else pooled.format(digits))
        delta = _pooled([row.metrics[f"delta:{metric}"] for row in group])
        text = "-" if delta is None else delta.format(digits)
        if delta is not None and delta.excludes_zero:
            text = f"**{text}**"
        cells.append(text)
        values = [row.metrics[f"delta:{metric}"].value for row in group]
        wins = sum(1 for v in values if v != 0 and (v > 0) == HIGHER_IS_BETTER[metric])
        cells.append(f"{wins} of {len(group)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ranking_table(rows: Sequence[BaseComparisonRow], *, metric: str = "qini") -> str:
    """Result 1, read the other way: the order of the five learners on each base.

    Args:
        rows: Rows at one TabFM setting.
        metric: The metric to order by.

    Returns:
        A markdown table with one row per rank, or an empty string without rows.
    """
    grouped = _group(rows)
    if not grouped:
        return ""
    better = HIGHER_IS_BETTER[metric]
    order: dict[str, list[tuple[str, float]]] = {}
    for base in BASES:
        means = []
        for learner, group in grouped.items():
            pooled = _pooled([row.metrics[f"{base}:{metric}"] for row in group])
            if pooled is not None:
                means.append((learner, pooled.value))
        order[base] = sorted(means, key=lambda pair: pair[1], reverse=better)
    depth = max(len(ranked) for ranked in order.values())
    lines = [f"| Rank by {_label(metric)} | LightGBM | TabFM |", "|---|---|---|"]
    for rank in range(depth):
        cells = [str(rank + 1)]
        for base in BASES:
            ranked = order[base]
            cells.append(
                f"`{ranked[rank][0]}` ({ranked[rank][1]:.4f})" if rank < len(ranked) else "-"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def timing_table(rows: Sequence[BaseComparisonRow]) -> str:
    """Result 2: what each base spends, per fit, mean across seeds.

    Args:
        rows: Rows at one TabFM setting.

    Returns:
        A markdown table, or an empty string without rows.
    """
    grouped = _group(rows)
    if not grouped:
        return ""
    lines = [
        "| Meta-learner | LightGBM grid search (s) | LightGBM final fit and predict (s) "
        "| TabFM fit and predict (s) | TabFM's bill over LightGBM's |",
        "|---|---|---|---|---|",
    ]
    for learner, group in grouped.items():
        search = float(np.mean([row.lightgbm_select_seconds for row in group]))
        final = float(
            np.mean([row.lightgbm_fit_seconds + row.lightgbm_predict_seconds for row in group])
        )
        tabfm = float(np.mean([row.tabfm_seconds for row in group]))
        whole = search + final
        ratio = f"{tabfm / whole:,.0f}x" if whole > 0 else "-"
        lines.append(f"| `{learner}` | {search:,.1f} | {final:,.1f} | {tabfm:,.1f} | {ratio} |")
    return "\n".join(lines) + "\n"


def sweep_table(
    rows: Sequence[BaseComparisonRow],
    *,
    over: str,
    metric: str = "qini",
    digits: int = 4,
) -> str:
    """How TabFM's result moves with one setting, holding the other at the headline.

    Args:
        rows: Rows for one dataset, at several settings.
        over: ``"context_rows"`` or ``"ensemble"``.
        metric: The metric to show.
        digits: Decimal places.

    Returns:
        A markdown table, or an empty string when the setting was not varied.

    Raises:
        ValueError: If ``over`` is not one of the two settings.
    """
    if over not in ("context_rows", "ensemble"):
        msg = f"can only sweep context_rows or ensemble, not {over!r}"
        raise ValueError(msg)
    top = headline(rows)
    if not top:
        return ""
    held = "ensemble" if over == "context_rows" else "context_rows"
    held_value = getattr(top[0], held)
    varied = [row for row in rows if getattr(row, held) == held_value]
    values = sorted({getattr(row, over) for row in varied})
    if len(values) < 2:
        return ""
    noun = "context rows" if over == "context_rows" else "ensemble members"
    header = ["Meta-learner", "LightGBM"] + [f"TabFM, {v:,} {noun}" for v in values]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---" * len(header) + "|",
    ]
    for learner, group in _group(varied).items():
        lightgbm = _pooled([row.metrics[f"lightgbm:{metric}"] for row in group if row in top])
        cells = [f"`{learner}`", "-" if lightgbm is None else lightgbm.format(digits)]
        for value in values:
            pooled = _pooled(
                [row.metrics[f"tabfm:{metric}"] for row in group if getattr(row, over) == value]
            )
            cells.append("-" if pooled is None else pooled.format(digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def report(rows: Sequence[BaseComparisonRow]) -> str:
    """Every table for one dataset, as the README block ``bases-<dataset>`` carries it.

    Args:
        rows: Rows for one dataset.

    Returns:
        Markdown, or an empty string without rows.
    """
    top = headline(rows)
    if not top:
        return ""
    seeds = sorted({row.seed for row in top})
    first = top[0]
    capped = first.n_train > first.context_rows
    context = (
        f"each TabFM model sees at most {first.context_rows:,} context rows"
        if capped
        else f"below TabFM's {first.context_rows:,}-row context cap, so it sees them all"
    )
    parts = [
        f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''} "
        f"({', '.join(str(s) for s in seeds)}), {first.n_train:,} training rows ({context}), "
        f"{first.n_test:,} test rows, TabFM ensemble of {first.ensemble}. Each cell is the "
        f"mean across seeds with the averaged paired 95% bootstrap interval; a bold "
        f"difference excludes zero.",
    ]
    for metric in COMPARED:
        table = comparison_table(top, metric)
        if table:
            parts += ["", table.rstrip("\n")]
    parts += ["", ranking_table(top).rstrip("\n"), "", timing_table(top).rstrip("\n")]
    for over, title in (
        ("context_rows", "How TabFM's Qini moves with its context cap:"),
        ("ensemble", "How TabFM's Qini moves with its ensemble size:"),
    ):
        sweep = sweep_table(rows, over=over)
        if sweep:
            parts += ["", title, "", sweep.rstrip("\n")]
    return "\n".join(parts) + "\n"
