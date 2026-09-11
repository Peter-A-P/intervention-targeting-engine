"""The benchmark: fit on train, score on test, report everything with an interval.

One row of the results table is one estimator on one dataset at one split seed. Nothing is
computed on the training or validation rows, no metric leaves this module without a
bootstrap interval, and the two baselines are run through the identical path so the
comparison is like for like.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from itx.bench.grid import (
    EstimatorFactory,
    Selection,
    default_selection,
    select_config,
)
from itx.bench.seeds import (
    RANDOM_BASELINE_SEED,
    SEEDS,
    TIE_SEED,
    bootstrap_seed_for,
)
from itx.data.acic import load_acic
from itx.data.hillstrom import load_hillstrom
from itx.data.ihdp import load_ihdp
from itx.data.splits import stratified_split
from itx.data.synthetic import (
    binary_outcome,
    complex_effect,
    confounded,
    heterogeneous_effect,
)
from itx.estimators.base import BaseUpliftEstimator
from itx.estimators.baselines import OutcomeRanking, RandomRanking
from itx.estimators.dr_learner import DRLearner
from itx.estimators.r_learner import RLearner
from itx.estimators.s_learner import SLearner
from itx.estimators.t_learner import TLearner
from itx.estimators.x_learner import XLearner
from itx.metrics.baselines import random_ranking_reference
from itx.metrics.bootstrap import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    Estimate,
    bootstrap_vector,
)
from itx.metrics.calibration import calibration_error, calibration_slope
from itx.metrics.ground_truth import ate_error, pehe
from itx.metrics.qini import ranking_metrics

if TYPE_CHECKING:
    from itx.types import FloatArray, IntArray, Split, UpliftDataset

#: Budgets the table reports uplift at, as shares of the population (PLAN.md section 1).
BUDGETS: tuple[float, ...] = (0.1, 0.2, 0.3)

#: Loaders the CLI can name. Datasets that ship replicates load replicate 0 here; the
#: averaging over replicates arrives with the ground-truth metrics in week 3.
DATASETS: dict[str, Callable[[], UpliftDataset]] = {
    "hillstrom": load_hillstrom,
    "ihdp": lambda: load_ihdp(0),
    "acic": lambda: load_acic(0),
    "synthetic-binary": binary_outcome,
    "synthetic-heterogeneous": heterogeneous_effect,
    "synthetic-complex": complex_effect,
    "synthetic-confounded": confounded,
}

#: Estimators the CLI can name. Each is built from a base-learner configuration and a
#: seed, so the same factory serves both the tuning loop and the final fit.
ESTIMATORS: dict[str, EstimatorFactory] = {
    "s-learner": lambda config, seed: SLearner(config, seed=seed),
    "t-learner": lambda config, seed: TLearner(config, seed=seed),
    "x-learner": lambda config, seed: XLearner(config, seed=seed),
    "dr-learner": lambda config, seed: DRLearner(config, seed=seed),
    "r-learner": lambda config, seed: RLearner(config, seed=seed),
    "outcome-ranking": lambda config, seed: OutcomeRanking(config, seed=seed),
    "random": lambda _config, seed: RandomRanking(seed=seed),
}

#: Estimators with nothing to tune. Running them through the grid would fit six identical
#: models and pick between six identical scores.
UNTUNED: frozenset[str] = frozenset({"random"})

#: What runs when no estimator is named. A single random ranking is left out on purpose:
#: the baseline the protocol calls for is the average of 200 of them, which is added
#: separately, and putting one noisy draw next to it in the same table invites the reader
#: to treat the gap between the two as a result. ``random`` can still be asked for by name.
DEFAULT_ESTIMATORS: tuple[str, ...] = (
    "s-learner",
    "t-learner",
    "x-learner",
    "dr-learner",
    "r-learner",
    "outcome-ranking",
)


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    """One estimator, one dataset, one split.

    Attributes:
        dataset: Dataset name.
        estimator: Estimator name.
        seed: The split seed.
        n_test: Rows the metrics were computed on.
        metrics: Named metrics, each with its interval.
        fit_seconds: Wall-clock time to fit, for the cost side of the comparison.
        scores: Predicted uplift on the test rows, kept for plotting and not serialised.
        selection: The configuration chosen on the validation split, and what it scored.
    """

    dataset: str
    estimator: str
    seed: int
    n_test: int
    metrics: dict[str, Estimate]
    fit_seconds: float
    scores: FloatArray = field(repr=False)
    selection: Selection | None = None

    def get(self, metric: str) -> Estimate | None:
        """Look up one metric, or None if this row does not carry it.

        Args:
            metric: Metric name.

        Returns:
            The estimate, or None. Ground-truth metrics are absent on the randomised
            datasets, and that absence is the honest answer rather than a zero.
        """
        return self.metrics.get(metric)


def evaluate(
    estimator: BaseUpliftEstimator,
    split: Split,
    *,
    budgets: Sequence[float] = BUDGETS,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    tie_seed: int = TIE_SEED,
    selection: Selection | None = None,
) -> BenchmarkRow:
    """Fit an estimator on the train split and measure it on the test split.

    Args:
        estimator: An unfitted estimator.
        split: The partition to use. Only ``split.train`` is fitted on and only
            ``split.test`` is measured on; the validation part is untouched here and
            belongs to hyperparameter selection.
        budgets: Shares of the population to report uplift at.
        n_resamples: Bootstrap resamples.
        level: Nominal coverage of the intervals.
        tie_seed: Seed for tie-breaking inside rankings.
        selection: The selection that produced this estimator's configuration, recorded on
            the row so the table says what was fitted rather than leaving it implied.

    Returns:
        The finished row.
    """
    started = time.perf_counter()
    estimator.fit(split.train)
    fit_seconds = time.perf_counter() - started

    test = split.test
    scores = estimator.predict_uplift(test.features)
    metrics = bootstrap_vector(
        _statistics(
            test,
            scores,
            budgets=budgets,
            tie_seed=tie_seed,
            with_ground_truth=estimator.estimates_effect,
            with_calibration=estimator.estimates_effect,
        ),
        test.n_units,
        n_resamples=n_resamples,
        level=level,
        seed=bootstrap_seed_for(split.seed),
    )
    return BenchmarkRow(
        dataset=test.name,
        estimator=estimator.name,
        seed=split.seed,
        n_test=test.n_units,
        metrics=metrics,
        fit_seconds=fit_seconds,
        scores=scores,
        selection=selection,
    )


def random_reference_row(
    split: Split,
    *,
    budgets: Sequence[float] = BUDGETS,
    n_rankings: int = 200,
    level: float = DEFAULT_LEVEL,
    tie_seed: int = TIE_SEED,
) -> BenchmarkRow:
    """The random-targeting baseline, averaged over many rankings rather than fitted once.

    A single random ranking is one draw from a wide distribution, and reporting it next to
    a fitted model invites a reader to compare a model against noise. This averages the
    metric over ``n_rankings`` draws and reports the spread across them.

    Args:
        split: The partition; only the test part is used.
        budgets: Shares of the population to report uplift at.
        n_rankings: Random rankings to draw.
        level: Nominal coverage.
        tie_seed: Seed for tie-breaking inside rankings.

    Returns:
        A row named ``random-200``, carrying the last drawn ranking as its scores.
    """
    test = split.test
    metrics: dict[str, Estimate] = {}
    for name in _metric_names_for(test, budgets):
        metrics[name] = random_ranking_reference(
            _one_metric_of_scores(test, name, budgets=budgets, tie_seed=tie_seed),
            test.n_units,
            n_rankings=n_rankings,
            level=level,
            seed=RANDOM_BASELINE_SEED + split.seed,
        )
    rng = np.random.default_rng(RANDOM_BASELINE_SEED + split.seed)
    return BenchmarkRow(
        dataset=test.name,
        estimator=f"random-{n_rankings}",
        seed=split.seed,
        n_test=test.n_units,
        metrics=metrics,
        fit_seconds=0.0,
        scores=rng.random(test.n_units),
    )


def run(
    dataset: str,
    *,
    estimators: Sequence[str] | None = None,
    seeds: Sequence[int] = SEEDS,
    budgets: Sequence[float] = BUDGETS,
    n_resamples: int = DEFAULT_RESAMPLES,
    include_random_reference: bool = True,
    tune: bool = True,
) -> list[BenchmarkRow]:
    """Run one dataset across estimators and seeds.

    Args:
        dataset: A key of :data:`DATASETS`.
        estimators: Keys of :data:`ESTIMATORS`; :data:`DEFAULT_ESTIMATORS` if omitted.
        seeds: Split seeds.
        budgets: Shares of the population to report uplift at.
        n_resamples: Bootstrap resamples per row.
        include_random_reference: Add the averaged random-targeting row per seed.
        tune: Select each estimator's configuration on the validation split from the
            committed grid. Turning it off uses the default configuration everywhere,
            which is faster and is what the fast tests do.

    Returns:
        Every row, in dataset-then-seed-then-estimator order.

    Raises:
        KeyError: If a dataset or estimator name is not registered.
    """
    if dataset not in DATASETS:
        msg = f"unknown dataset {dataset!r}; known: {', '.join(sorted(DATASETS))}"
        raise KeyError(msg)
    names = list(estimators) if estimators is not None else list(DEFAULT_ESTIMATORS)
    unknown = set(names) - set(ESTIMATORS)
    if unknown:
        msg = f"unknown estimators {sorted(unknown)}; known: {', '.join(sorted(ESTIMATORS))}"
        raise KeyError(msg)

    data = DATASETS[dataset]()
    rows: list[BenchmarkRow] = []
    for seed in seeds:
        split = stratified_split(data, seed)
        for name in names:
            factory = ESTIMATORS[name]
            selection = (
                select_config(factory, split, seed=seed)
                if tune and name not in UNTUNED
                else default_selection()
            )
            rows.append(
                evaluate(
                    factory(selection.config, seed),
                    split,
                    budgets=budgets,
                    n_resamples=n_resamples,
                    selection=selection,
                )
            )
        if include_random_reference:
            rows.append(random_reference_row(split, budgets=budgets))
    return rows


def refit_seed(
    dataset: str,
    seed: int,
    *,
    selections: Mapping[str, Selection],
    estimators: Sequence[str] | None = None,
) -> tuple[list[BenchmarkRow], Split]:
    """Refit one seed using configurations that were already chosen, and skip the metrics.

    The figures need per-unit scores, which the results file does not store, so redrawing
    them needs a fit. It does not need the twenty minutes the full run costs: one seed, no
    grid search because the winning configuration is read back from the run that happened,
    and no bootstrap because a figure does not use one.

    Args:
        dataset: A key of :data:`DATASETS`.
        seed: The split seed to refit.
        selections: The configuration each estimator was given, by estimator name.
        estimators: Which estimators to refit; the keys of ``selections`` by default.

    Returns:
        The rows, carrying scores but no metrics, and the split they were fitted on.
    """
    names = list(estimators) if estimators is not None else list(selections)
    split = stratified_split(DATASETS[dataset](), seed)
    rows: list[BenchmarkRow] = []
    for name in names:
        if name not in ESTIMATORS:
            continue
        selection = selections.get(name, default_selection())
        estimator = ESTIMATORS[name](selection.config, seed)
        started = time.perf_counter()
        estimator.fit(split.train)
        rows.append(
            BenchmarkRow(
                dataset=split.test.name,
                estimator=estimator.name,
                seed=seed,
                n_test=split.test.n_units,
                metrics={},
                fit_seconds=time.perf_counter() - started,
                scores=estimator.predict_uplift(split.test.features),
                selection=selection,
            )
        )
    return rows, split


def _statistics(
    test: UpliftDataset,
    scores: FloatArray,
    *,
    budgets: Sequence[float],
    tie_seed: int,
    with_ground_truth: bool = True,
    with_calibration: bool = True,
) -> Callable[[IntArray], dict[str, float]]:
    """Every metric for this row, as one function of the resampled row positions.

    One function rather than one per metric, because the ranking metrics share a sort and
    the ground-truth metrics have to be dropped as a group: when the dataset has no truth
    to compare against, and when the estimator produces a ranking score rather than an
    effect on the outcome's scale. Calibration is dropped on the same condition as the
    second of those, and for the same reason: asking whether a risk score is the right size
    to be a treatment effect is not a question about the score.
    """
    outcome, treatment = test.outcome, test.treatment
    truth = test.true_effect if with_ground_truth else None

    def statistics(index: IntArray) -> dict[str, float]:
        values = ranking_metrics(
            outcome[index],
            treatment[index],
            scores[index],
            budgets=budgets,
            seed=tie_seed,
        )
        if with_calibration:
            values["calibration_slope"] = calibration_slope(
                outcome[index], treatment[index], scores[index], seed=tie_seed
            )
            values["calibration_error"] = calibration_error(
                outcome[index], treatment[index], scores[index], seed=tie_seed
            )
        if truth is not None:
            values["pehe"] = pehe(scores[index], truth[index])
            values["ate_error"] = ate_error(scores[index], truth[index])
        return values

    return statistics


def _metric_names_for(test: UpliftDataset, budgets: Sequence[float]) -> list[str]:
    """Names of the ranking metrics reported for a dataset.

    Ground-truth metrics are deliberately absent: a random ranking has the same PEHE as
    any other constant-free score vector, so averaging one over random draws would put a
    meaningless number in the baseline row.
    """
    return ["qini", "auuc", *[_budget_key(budget) for budget in budgets]]


def _one_metric_of_scores(
    test: UpliftDataset,
    metric: str,
    *,
    budgets: Sequence[float],
    tie_seed: int,
) -> Callable[[FloatArray], float]:
    """One metric as a function of a score vector, for the random-targeting reference.

    The sample is fixed here and the ranking is what varies, the exact opposite of a
    bootstrap.
    """

    def statistic(candidate: FloatArray) -> float:
        return ranking_metrics(
            test.outcome, test.treatment, candidate, budgets=budgets, seed=tie_seed
        )[metric]

    return statistic


def _budget_key(budget: float) -> str:
    """Metric name for a budget, for example ``uplift@20%``."""
    return f"uplift@{budget:.0%}"


def metric_names(rows: Sequence[BenchmarkRow]) -> list[str]:
    """Every metric present across a set of rows, in first-seen order.

    Args:
        rows: Benchmark rows.

    Returns:
        Metric names.
    """
    seen: dict[str, None] = {}
    for row in rows:
        for name in row.metrics:
            seen.setdefault(name, None)
    return list(seen)


def group_by_estimator(rows: Sequence[BenchmarkRow]) -> Mapping[str, list[BenchmarkRow]]:
    """Group rows by estimator name, preserving order.

    Args:
        rows: Benchmark rows.

    Returns:
        Mapping from estimator name to its rows.
    """
    grouped: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault(row.estimator, []).append(row)
    return grouped
