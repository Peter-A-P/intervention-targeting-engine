"""The intervals have to cover.

An interval that is always reported and never checked is decoration. The coverage test
here is the one that matters: on data whose truth is known, a nominal 95% interval has to
contain the truth about 95% of the time.
"""

from __future__ import annotations

import numpy as np
import pytest

from itx.metrics.bootstrap import (
    Estimate,
    bootstrap_ci,
    bootstrap_many,
    bootstrap_over,
    bootstrap_vector,
)


def test_a_nominal_95_percent_interval_covers_about_95_percent_of_the_time():
    true_mean = 3.0
    rng = np.random.default_rng(0)
    covered = 0
    replications = 200
    for replication in range(replications):
        sample = rng.normal(loc=true_mean, scale=1.0, size=300)

        def mean_of(index, sample=sample):
            return float(sample[index].mean())

        estimate = bootstrap_ci(mean_of, sample.size, n_resamples=200, seed=replication)
        covered += int(estimate.low <= true_mean <= estimate.high)
    coverage = covered / replications
    # Wide acceptance band on purpose: 200 replications of a 200-resample bootstrap has
    # real Monte Carlo error, and a test that fails one run in twenty is worse than none.
    assert 0.88 <= coverage <= 0.99


def test_the_point_estimate_is_the_statistic_on_the_full_sample():
    sample = np.arange(100, dtype=np.float64)
    estimate = bootstrap_ci(
        lambda index: float(sample[index].mean()), sample.size, n_resamples=50, seed=0
    )
    assert estimate.value == pytest.approx(sample.mean())


def test_the_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    sample = rng.normal(size=500)
    estimate = bootstrap_ci(
        lambda index: float(sample[index].mean()), sample.size, n_resamples=300, seed=0
    )
    assert estimate.low <= estimate.value <= estimate.high


def test_a_wider_level_gives_a_wider_interval():
    rng = np.random.default_rng(2)
    sample = rng.normal(size=400)

    def statistic(index):
        return float(sample[index].mean())

    narrow = bootstrap_ci(statistic, sample.size, n_resamples=400, level=0.80, seed=0)
    wide = bootstrap_ci(statistic, sample.size, n_resamples=400, level=0.99, seed=0)
    assert wide.width > narrow.width


def test_a_bigger_sample_gives_a_narrower_interval():
    rng = np.random.default_rng(3)
    small = rng.normal(size=200)
    large = rng.normal(size=5_000)
    narrow = bootstrap_ci(
        lambda index: float(large[index].mean()), large.size, n_resamples=300, seed=0
    )
    broad = bootstrap_ci(
        lambda index: float(small[index].mean()), small.size, n_resamples=300, seed=0
    )
    assert narrow.width < broad.width


def test_the_same_seed_gives_the_same_interval():
    rng = np.random.default_rng(4)
    sample = rng.normal(size=300)

    def statistic(index):
        return float(sample[index].mean())

    first = bootstrap_ci(statistic, sample.size, n_resamples=200, seed=7)
    second = bootstrap_ci(statistic, sample.size, n_resamples=200, seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_undefined_resamples_are_dropped_rather_than_counted_as_zero():
    rng = np.random.default_rng(5)
    sample = rng.normal(size=100)

    def sometimes_undefined(index):
        return float("nan") if index.sum() % 3 == 0 else float(sample[index].mean())

    estimate = bootstrap_ci(sometimes_undefined, sample.size, n_resamples=90, seed=0)
    assert estimate.n_resamples < 90
    assert np.isfinite(estimate.low)


def test_an_all_undefined_statistic_is_an_error():
    with pytest.raises(ValueError, match="every bootstrap resample was undefined"):
        bootstrap_ci(lambda index: float("nan"), 50, n_resamples=10, seed=0)


def test_an_impossible_level_is_rejected():
    with pytest.raises(ValueError, match="level must be"):
        bootstrap_ci(lambda index: 0.0, 10, level=1.5)


def test_bootstrapping_several_statistics_together_matches_doing_them_separately():
    rng = np.random.default_rng(6)
    sample = rng.normal(size=400)

    def mean_of(index):
        return float(sample[index].mean())

    def spread_of(index):
        return float(sample[index].std())

    together = bootstrap_many(
        {"mean": mean_of, "spread": spread_of}, sample.size, n_resamples=200, seed=11
    )
    apart = bootstrap_ci(mean_of, sample.size, n_resamples=200, seed=11)
    assert together["mean"].value == pytest.approx(apart.value)
    assert together["mean"].low == pytest.approx(apart.low)
    assert together["spread"].value == pytest.approx(sample.std())


def test_the_vector_form_matches_the_named_form():
    rng = np.random.default_rng(7)
    sample = rng.normal(size=300)

    def both(index):
        return {"mean": float(sample[index].mean()), "spread": float(sample[index].std())}

    vector = bootstrap_vector(both, sample.size, n_resamples=150, seed=3)
    named = bootstrap_many(
        {
            "mean": lambda index: float(sample[index].mean()),
            "spread": lambda index: float(sample[index].std()),
        },
        sample.size,
        n_resamples=150,
        seed=3,
    )
    assert vector["mean"].low == pytest.approx(named["mean"].low)
    assert vector["spread"].high == pytest.approx(named["spread"].high)


def test_summarising_across_runs_uses_the_spread_across_runs():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    estimate = bootstrap_over(values)
    assert estimate.value == pytest.approx(3.0)
    assert estimate.low >= 1.0
    assert estimate.high <= 5.0
    assert estimate.n_resamples == 5


def test_summarising_across_runs_ignores_undefined_runs():
    estimate = bootstrap_over(np.array([1.0, np.nan, 3.0]))
    assert estimate.value == pytest.approx(2.0)
    assert estimate.n_resamples == 2


def test_summarising_nothing_is_an_error():
    with pytest.raises(ValueError, match="no finite values"):
        bootstrap_over(np.array([np.nan, np.nan]))


def test_an_estimate_renders_with_its_interval():
    estimate = Estimate(value=0.0116, low=0.0089, high=0.0143)
    assert str(estimate) == "0.0116 (0.0089, 0.0143)"
    assert estimate.format(digits=2) == "0.01 (0.01, 0.01)"


def test_an_estimate_knows_whether_it_excludes_zero():
    assert Estimate(0.5, 0.1, 0.9).excludes_zero
    assert Estimate(-0.5, -0.9, -0.1).excludes_zero
    assert not Estimate(0.1, -0.2, 0.4).excludes_zero
