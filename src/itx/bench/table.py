"""Rendering benchmark rows into the table that goes in the README.

Two things are aggregated here and they are not the same, so they are kept apart and
labelled. Within a split, the bootstrap interval says how much the number would move if
the test sample had been drawn differently. Across the five split seeds, the spread says
how much it moves when the partition and the fit change. A table that silently mixed them
would be understating one or the other.

The README table reports the mean across seeds with the averaged bootstrap interval, which
is the quantity a reader wants: the metric, and how firm it is. The across-seed spread is
available next to it, and the full per-seed detail is written to JSON.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itx.bench.grid import OPERATING_BUDGET, Selection
from itx.bench.runner import BenchmarkRow, group_by_estimator, metric_names
from itx.estimators.lightgbm_base import DEFAULT_CONFIG
from itx.metrics.bootstrap import Estimate, bootstrap_over
from itx.policy.policy_value import share_gap_key

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Summary:
    """One estimator's result for one metric, across seeds.

    Attributes:
        dataset: Dataset name.
        estimator: Estimator name.
        metric: Metric name.
        pooled: Mean across seeds, with the averaged within-split bootstrap interval.
        across_seeds: Mean across seeds, with the spread across seeds.
        n_seeds: How many seeds contributed.
    """

    dataset: str
    estimator: str
    metric: str
    pooled: Estimate
    across_seeds: Estimate
    n_seeds: int


def summarise(rows: Sequence[BenchmarkRow]) -> list[Summary]:
    """Aggregate per-seed rows into one summary per estimator and metric.

    Args:
        rows: Rows for a single dataset.

    Returns:
        Summaries in estimator-then-metric order. Metrics missing from an estimator's rows
        are skipped rather than filled in. A metric that is present but undefined on every
        seed is kept and rendered as a dash, rather than raising.

    The dash case is here because it cost a benchmark run. Dragonnet's calibration is
    undefined on all five Lenta seeds, and :func:`itx.metrics.bootstrap.bootstrap_over`
    raises on an input with no finite values, which is right for its other callers and
    fatal here: the exception surfaced after Lenta's 35 fits had finished, at the table
    render, and took the whole run down with it including the Criteo dataset that had not
    started. One estimator having nothing to say about one metric is an ordinary thing
    that the table already knows how to print. It must not be able to destroy five hours
    of fits.
    """
    summaries: list[Summary] = []
    for estimator, estimator_rows in group_by_estimator(rows).items():
        for metric in metric_names(estimator_rows):
            estimates = [row.metrics[metric] for row in estimator_rows if metric in row.metrics]
            if not estimates:
                continue
            values = np.array([e.value for e in estimates], dtype=np.float64)
            across = (
                bootstrap_over(values, level=estimates[0].level)
                if np.isfinite(values).any()
                else Estimate(
                    value=math.nan,
                    low=math.nan,
                    high=math.nan,
                    level=estimates[0].level,
                    n_resamples=0,
                )
            )
            summaries.append(
                Summary(
                    dataset=estimator_rows[0].dataset,
                    estimator=estimator,
                    metric=metric,
                    pooled=Estimate(
                        value=float(np.mean(values)),
                        low=float(np.mean([e.low for e in estimates])),
                        high=float(np.mean([e.high for e in estimates])),
                        level=estimates[0].level,
                        n_resamples=sum(e.n_resamples for e in estimates),
                    ),
                    across_seeds=across,
                    n_seeds=len(estimates),
                )
            )
    return summaries


def to_markdown(
    rows: Sequence[BenchmarkRow],
    *,
    metrics: Sequence[str] | None = None,
    digits: int = 4,
) -> str:
    """Render a markdown table: one row per estimator, one column per metric.

    Args:
        rows: Rows for a single dataset.
        metrics: Metrics to show, in order. Every metric present by default.
        digits: Decimal places.

    Returns:
        A markdown table, with the estimator column first.
    """
    summaries = summarise(rows)
    if not summaries:
        return "| _no rows_ |\n|---|\n"

    columns = list(metrics) if metrics is not None else metric_names(rows)
    lookup = {(s.estimator, s.metric): s for s in summaries}
    estimators = list(dict.fromkeys(s.estimator for s in summaries))

    header = "| Estimator | " + " | ".join(_column_label(c) for c in columns) + " |"
    rule = "|---|" + "---|" * len(columns)
    lines = [header, rule]
    for estimator in estimators:
        cells = []
        for column in columns:
            summary = lookup.get((estimator, column))
            cells.append("-" if summary is None else summary.pooled.format(digits))
        lines.append(f"| `{estimator}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


#: Prefix marking a metric as a realised policy value rather than a ranking metric.
POLICY_PREFIX = "gain_"

#: Prefix of the treated-share diagnostic, which belongs beside the gains rather than beside
#: the ranking metrics: it says whether the IPW column next to it can be read at all.
SHARE_PREFIX = "share_gap@"

#: Budget the share gap is shown at. It is computed at every budget and stored at every
#: budget in the results file; only one is rendered, because the policy table is already
#: seven columns wide and three more of a diagnostic that moves slowly across budgets would
#: cost more readability than it buys.
SHARE_GAP_BUDGET = OPERATING_BUDGET


def results_table(rows: Sequence[BenchmarkRow], *, digits: int = 4) -> str:
    """The ranking table: every metric except the policy gains.

    Args:
        rows: Rows for a single dataset.
        digits: Decimal places.

    Returns:
        A markdown table.
    """
    return to_markdown(
        rows,
        metrics=[m for m in metric_names(rows) if not _is_policy(m)],
        digits=digits,
    )


def policy_table(rows: Sequence[BenchmarkRow], *, digits: int = 4) -> str:
    """The policy-value table: what each ranking buys, and nothing else.

    Its own block rather than six more columns on the results table, for the same reason
    the results table is generated at all: a table nobody can read across is not reporting
    anything. The two tables carry the same estimators in the same order, so a reader
    compares them line for line.

    Args:
        rows: Rows for a single dataset.
        digits: Decimal places.

    Returns:
        A markdown table, empty if these rows carry no policy metrics.
    """
    shown = share_gap_key(SHARE_GAP_BUDGET)
    columns = [m for m in metric_names(rows) if m.startswith(POLICY_PREFIX) or m == shown]
    if not any(m.startswith(POLICY_PREFIX) for m in columns):
        return ""
    return to_markdown(rows, metrics=columns, digits=digits)


def selected_configurations(rows: Sequence[BenchmarkRow]) -> str:
    """A markdown list of which configuration each estimator was given, and how often.

    The grid is committed, so the table is only reproducible if the choices it produced are
    visible too. Estimators with nothing to tune are left out.

    Args:
        rows: Rows for a single dataset.

    Returns:
        A markdown list, or an empty string when nothing was tuned.
    """
    chosen: dict[str, Counter[str]] = {}
    for row in rows:
        if row.selection is None or not row.selection.tuned:
            continue
        label = (
            f"min_child_samples={row.selection.config.min_child_samples}, "
            f"num_leaves={row.selection.config.num_leaves}"
        )
        chosen.setdefault(row.estimator, Counter())[label] += 1

    if not chosen:
        return ""
    lines = []
    for estimator, counts in chosen.items():
        picks = ", ".join(
            f"{label} ({count} of {sum(counts.values())} seeds)"
            for label, count in counts.most_common()
        )
        lines.append(f"- `{estimator}`: {picks}")
    return "\n".join(lines) + "\n"


def write_json(rows: Sequence[BenchmarkRow], path: Path) -> None:
    """Write every per-seed number to disk, so a table can be rebuilt without refitting.

    Args:
        rows: Rows to write.
        path: Destination file. Parent directories are created.
    """
    payload = [
        {
            "dataset": row.dataset,
            "estimator": row.estimator,
            "seed": row.seed,
            "n_test": row.n_test,
            "fit_seconds": round(row.fit_seconds, 3),
            "selection": None
            if row.selection is None or not row.selection.tuned
            else {
                "min_child_samples": row.selection.config.min_child_samples,
                "num_leaves": row.selection.config.num_leaves,
                "validation_qini": row.selection.score,
                # Two selection rules exist and they do not measure the same thing, so the
                # file has to say which one produced the number above it. The key keeps its
                # old name so that results files committed before the second rule existed
                # still read, and those files are all "qini" by construction.
                "rule": row.selection.rule,
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


def read_json(path: Path) -> list[BenchmarkRow]:
    """Rebuild benchmark rows from a results file, so a table can be redrawn without refitting.

    This is what ``write_json`` is for. Regenerating the README from a run that already
    happened should cost a second, not twenty minutes, and a reader who wants to check the
    table against the per-seed numbers should not have to own a machine that can refit five
    estimators.

    The scores are not stored, so the rows come back with an empty score array. Everything
    that needs them, which is the figures, has to refit; everything that needs only the
    metrics, which is every table, does not.

    Args:
        path: A file written by :func:`write_json`.

    Returns:
        The rows, in the order they were written.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[BenchmarkRow] = []
    for record in payload:
        selection = record.get("selection")
        rows.append(
            BenchmarkRow(
                dataset=record["dataset"],
                estimator=record["estimator"],
                seed=record["seed"],
                n_test=record["n_test"],
                metrics={
                    name: Estimate(
                        value=metric["value"],
                        low=metric["low"],
                        high=metric["high"],
                        level=metric["level"],
                        n_resamples=metric["n_resamples"],
                    )
                    for name, metric in record["metrics"].items()
                },
                fit_seconds=record["fit_seconds"],
                scores=np.empty(0, dtype=np.float64),
                selection=None
                if selection is None
                else Selection(
                    config=DEFAULT_CONFIG.with_(
                        min_child_samples=selection["min_child_samples"],
                        num_leaves=selection["num_leaves"],
                    ),
                    score=selection["validation_qini"],
                    scores=(),
                    rule=selection.get("rule", "qini"),
                ),
            )
        )
    return rows


def compare_results(
    baseline: Sequence[BenchmarkRow],
    current: Sequence[BenchmarkRow],
    *,
    tolerance: float = 0.0,
) -> list[str]:
    """Describe every way two runs disagree, ignoring how long the fits took.

    The obvious way to assert that a committed results file still matches a fresh run is
    ``git diff --exit-code`` on the file. That does not work here, and shipping it in CI was
    a mistake: :func:`write_json` records ``fit_seconds``, which is wall-clock and never
    reproduces, so the check would have failed on every scheduled run for a reason that has
    nothing to do with the numbers (PLAN.md change 28).

    Comparing the rows instead fixes that and gives a better failure message than a diff.
    A moved metric is reported as the metric that moved and by how much, which is the
    question a reader of a failing build actually has.

    Args:
        baseline: The committed rows.
        current: The rows a fresh run produced.
        tolerance: Absolute difference tolerated per number. Zero by default: the
            benchmark is seeded throughout and is meant to reproduce exactly, so a
            tolerance is something to reach for once a platform has been shown to need it,
            with the reason recorded.

    Returns:
        One line per disagreement, empty when the two runs agree.
    """
    differences: list[str] = []
    left = {(row.estimator, row.seed): row for row in baseline}
    right = {(row.estimator, row.seed): row for row in current}

    for estimator, seed in sorted(left.keys() - right.keys()):
        differences.append(f"{estimator} seed {seed}: committed, missing from the run")
    for estimator, seed in sorted(right.keys() - left.keys()):
        differences.append(f"{estimator} seed {seed}: produced by the run, not committed")

    for key in sorted(left.keys() & right.keys()):
        differences.extend(_compare_row(left[key], right[key], tolerance=tolerance))
    return differences


def _compare_row(
    baseline: BenchmarkRow, current: BenchmarkRow, *, tolerance: float
) -> list[str]:
    """Every disagreement between two rows for the same estimator and seed."""
    label = f"{baseline.estimator} seed {baseline.seed}"
    differences: list[str] = []

    if baseline.n_test != current.n_test:
        differences.append(
            f"{label}: test split has {current.n_test} rows, committed {baseline.n_test}"
        )

    left_selection = _selection_label(baseline.selection)
    right_selection = _selection_label(current.selection)
    if left_selection != right_selection:
        differences.append(f"{label}: selected {right_selection}, committed {left_selection}")

    for metric in sorted(set(baseline.metrics) | set(current.metrics)):
        if metric not in baseline.metrics:
            differences.append(f"{label}: {metric} is new")
            continue
        if metric not in current.metrics:
            differences.append(f"{label}: {metric} is gone")
            continue
        differences.extend(
            _compare_estimate(
                label, metric, baseline.metrics[metric], current.metrics[metric], tolerance
            )
        )
    return differences


def _compare_estimate(
    label: str, metric: str, baseline: Estimate, current: Estimate, tolerance: float
) -> list[str]:
    """Disagreements between two estimates of the same metric."""
    differences = []
    for part, was, now in (
        ("", baseline.value, current.value),
        (" low", baseline.low, current.low),
        (" high", baseline.high, current.high),
    ):
        if _moved(was, now, tolerance):
            differences.append(f"{label}: {metric}{part} is {now:.6g}, committed {was:.6g}")
    return differences


def _moved(was: float, now: float, tolerance: float) -> bool:
    """True when two numbers disagree by more than the tolerance, NaN counting as equal."""
    if np.isnan(was) and np.isnan(now):
        return False
    if np.isnan(was) or np.isnan(now):
        return True
    return abs(was - now) > tolerance


def _selection_label(selection: Selection | None) -> str:
    """A short, comparable description of a chosen configuration."""
    if selection is None:
        return "no tuning"
    return (
        f"min_child_samples={selection.config.min_child_samples}, "
        f"num_leaves={selection.config.num_leaves}"
    )


MARKER_START = "<!-- itx:table:{name} -->"
MARKER_END = "<!-- itx:end:{name} -->"


def update_markdown_file(path: Path, name: str, table: str) -> bool:
    """Replace a named table block in a markdown file with freshly generated content.

    The README's results table is generated, never typed: a hand-maintained table drifts
    from the numbers behind it, and this project's entire claim is that the two agree. The
    block is delimited by HTML comments, which render as nothing, so the file stays
    readable on GitHub.

    Args:
        path: The markdown file to edit.
        name: Block name, matching the markers ``<!-- itx:table:NAME -->`` and
            ``<!-- itx:end:NAME -->``.
        table: The markdown table to place between them.

    Returns:
        True if the file was rewritten, False if it has no such block.
    """
    if not path.is_file():
        return False
    start, end = MARKER_START.format(name=name), MARKER_END.format(name=name)
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return False
    before, _, rest = text.partition(start)
    _, _, after = rest.partition(end)
    body = "\n".join([start, table.strip(), end])
    path.write_text(f"{before}{body}{after}", encoding="utf-8")
    return True


def _column_label(metric: str) -> str:
    """Human-readable column heading for a metric name."""
    labels = {
        "qini": "Qini (95% CI)",
        "auuc": "Normalised AUUC",
        "pehe": "PEHE",
        "ate_error": "ATE error",
        "calibration_slope": "Calibration slope",
        "calibration_error": "Calibration error",
    }
    if metric.startswith(POLICY_PREFIX):
        estimator, _, budget = metric[len(POLICY_PREFIX) :].partition("@")
        return f"{estimator.upper()} gain at {budget}"
    if metric.startswith(SHARE_PREFIX):
        return f"Treated share gap at {metric[len(SHARE_PREFIX) :]}"
    return labels.get(metric, metric)


def _is_policy(metric: str) -> bool:
    """True for the metrics that belong in the policy table rather than the ranking one."""
    return metric.startswith((POLICY_PREFIX, SHARE_PREFIX))
