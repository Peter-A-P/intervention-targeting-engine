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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from itx.bench.runner import group_by_estimator, metric_names
from itx.metrics.bootstrap import Estimate, bootstrap_over

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from itx.bench.runner import BenchmarkRow


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
        are skipped rather than filled in.
    """
    summaries: list[Summary] = []
    for estimator, estimator_rows in group_by_estimator(rows).items():
        for metric in metric_names(estimator_rows):
            estimates = [row.metrics[metric] for row in estimator_rows if metric in row.metrics]
            if not estimates:
                continue
            values = np.array([e.value for e in estimates], dtype=np.float64)
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
                    across_seeds=bootstrap_over(values, level=estimates[0].level),
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
    }
    return labels.get(metric, metric)
