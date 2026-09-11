"""The Qini curve figure.

One picture, three lines, and the point of the whole project is the gap between two of
them: the fitted estimator, the outcome ranking that most targeting actually uses, and the
straight line you get for free. A curve above the diagonal is not a result on its own,
because the outcome ranking often manages that too.

Matplotlib runs on the Agg backend: figures are written to files, never shown, so the same
call works in CI and on a laptop.

Every figure is drawn inside an explicit style context, which is not fussiness. CausalML
imports seaborn, and importing seaborn rewrites matplotlib's global settings, so the same
plotting code produced a white figure before the R-learner existed and a grey one after.
A figure whose appearance depends on which estimators happened to be imported first is not
reproducible, and the fix is to stop inheriting whatever the last import decided.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from itx.bench.seeds import TIE_SEED
from itx.metrics.calibration import calibration_table
from itx.metrics.curves import qini_curve

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from itx.bench.runner import BenchmarkRow
    from itx.types import UpliftDataset

FIGURE_SIZE = (7.0, 4.5)
DPI = 160

#: Matplotlib's own defaults, applied as a context so that nothing an imported library did
#: to the global settings can reach the output.
FIGURE_STYLE = "default"


def plot_qini_curves(
    rows: Sequence[BenchmarkRow],
    test: UpliftDataset,
    path: Path,
    *,
    title: str | None = None,
    tie_seed: int = TIE_SEED,
) -> Path:
    """Draw the Qini curves for several estimators on one test split.

    Args:
        rows: Benchmark rows from the same split, each carrying its test-set scores.
        test: The test split those scores were computed on.
        path: Destination PNG. Parent directories are created.
        title: Figure title; built from the dataset name if omitted.
        tie_seed: Seed for tie-breaking, the same one the metrics used.

    Returns:
        The path written.

    Raises:
        ValueError: If no rows were given.
    """
    if not rows:
        msg = "nothing to plot: no benchmark rows"
        raise ValueError(msg)

    with plt.style.context(FIGURE_STYLE):
        return _draw_qini(rows, test, path, title=title, tie_seed=tie_seed)


def _draw_qini(
    rows: Sequence[BenchmarkRow],
    test: UpliftDataset,
    path: Path,
    *,
    title: str | None,
    tie_seed: int,
) -> Path:
    """The body of :func:`plot_qini_curves`, called inside the style context."""
    figure, axes = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    endpoint = 0.0
    for row in rows:
        curve = qini_curve(test.outcome, test.treatment, row.scores, seed=tie_seed)
        axes.plot(curve.fraction, curve.gain, label=_label(row), linewidth=1.6)
        endpoint = curve.endpoint

    axes.plot(
        [0.0, 1.0],
        [0.0, endpoint],
        linestyle="--",
        linewidth=1.2,
        color="0.45",
        label="random targeting",
    )
    axes.axhline(0.0, color="0.8", linewidth=0.8, zorder=0)

    axes.set_xlabel("Share of the population targeted")
    axes.set_ylabel("Cumulative incremental outcome")
    axes.set_title(title or f"Qini curves: {test.name} (test split, seed {rows[0].seed})")
    axes.set_xlim(0.0, 1.0)
    axes.legend(frameon=False, fontsize=9)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    return path


def _label(row: BenchmarkRow) -> str:
    """Legend entry: the estimator and its Qini coefficient with the interval."""
    qini = row.get("qini")
    if qini is None:
        return row.estimator
    return f"{row.estimator}  (Qini {qini.value:.4f})"


def plot_calibration(
    rows: Sequence[BenchmarkRow],
    test: UpliftDataset,
    path: Path,
    *,
    n_bins: int = 10,
    title: str | None = None,
    tie_seed: int = TIE_SEED,
) -> Path:
    """Draw predicted against realised uplift, by decile of predicted uplift.

    The diagonal is what a calibrated estimator looks like. Points below it in the
    right-hand deciles mean the model is promising more than it delivers where it is most
    confident, which is the failure that turns a good ranking into a bad forecast.

    Args:
        rows: Benchmark rows from the same split, each carrying its test-set scores.
        test: The test split those scores were computed on.
        path: Destination PNG. Parent directories are created.
        n_bins: Number of equal-sized bins.
        title: Figure title; built from the dataset name if omitted.
        tie_seed: Seed for tie-breaking, the same one the metrics used.

    Returns:
        The path written.

    Raises:
        ValueError: If no rows were given.
    """
    if not rows:
        msg = "nothing to plot: no benchmark rows"
        raise ValueError(msg)

    with plt.style.context(FIGURE_STYLE):
        return _draw_calibration(
            rows, test, path, n_bins=n_bins, title=title, tie_seed=tie_seed
        )


def _draw_calibration(
    rows: Sequence[BenchmarkRow],
    test: UpliftDataset,
    path: Path,
    *,
    n_bins: int,
    title: str | None,
    tie_seed: int,
) -> Path:
    """The body of :func:`plot_calibration`, called inside the style context."""
    figure, axes = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    span: list[float] = []
    for row in rows:
        table = calibration_table(
            test.outcome, test.treatment, row.scores, n_bins=n_bins, seed=tie_seed
        )
        usable = table.usable
        axes.errorbar(
            table.predicted[usable],
            table.realised[usable],
            yerr=table.standard_error[usable],
            marker="o",
            markersize=4,
            linewidth=1.2,
            capsize=2,
            label=row.estimator,
        )
        span.extend(table.predicted[usable].tolist())
        span.extend(table.realised[usable].tolist())

    if span:
        low, high = min(span), max(span)
        pad = 0.05 * (high - low) if high > low else 1.0
        axes.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            linestyle="--",
            linewidth=1.0,
            color="0.45",
            label="perfectly calibrated",
        )

    axes.set_xlabel("Mean predicted uplift in the decile")
    axes.set_ylabel("Uplift actually realised in the decile")
    axes.set_title(title or f"Calibration: {test.name} (test split, seed {rows[0].seed})")
    axes.legend(frameon=False, fontsize=9)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    return path
