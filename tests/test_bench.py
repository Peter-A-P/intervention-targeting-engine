"""The benchmark end to end, on synthetic data so it runs in CI in seconds."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from itx.bench.plots import plot_qini_curves
from itx.bench.runner import (
    DATASETS,
    DEFAULT_ESTIMATORS,
    ESTIMATORS,
    evaluate,
    random_reference_row,
    run,
)
from itx.bench.seeds import SEEDS, TIE_SEED, bootstrap_seed_for
from itx.bench.table import summarise, to_markdown, update_markdown_file, write_json
from itx.cli import app
from itx.data.splits import stratified_split
from itx.estimators.s_learner import SLearner

RESAMPLES = 40
runner = CliRunner()


@pytest.fixture(scope="module")
def rows():
    """Two seeds of the full benchmark on synthetic data, fitted once for the module."""
    return run("synthetic-binary", seeds=(11, 23), n_resamples=RESAMPLES)


class TestSeeds:
    def test_five_split_seeds_are_committed(self):
        assert len(SEEDS) == 5
        assert len(set(SEEDS)) == 5

    def test_bootstrap_seeds_differ_per_split(self):
        assert len({bootstrap_seed_for(seed) for seed in SEEDS}) == len(SEEDS)


class TestEvaluate:
    def test_a_row_carries_every_metric_with_an_interval(self, binary_data):
        split = stratified_split(binary_data, 11)
        row = evaluate(SLearner(seed=11), split, n_resamples=RESAMPLES)
        assert set(row.metrics) >= {"qini", "auuc", "uplift@10%", "uplift@20%", "uplift@30%"}
        for name, estimate in row.metrics.items():
            assert estimate.low <= estimate.value <= estimate.high, name
            assert estimate.level == 0.95

    def test_ground_truth_metrics_appear_only_where_there_is_truth(self, binary_data):
        with_truth = evaluate(
            SLearner(seed=11), stratified_split(binary_data, 11), n_resamples=RESAMPLES
        )
        assert "pehe" in with_truth.metrics

        no_truth = binary_data.take(np.arange(binary_data.n_units))
        object.__setattr__(no_truth, "true_effect", None)
        without = evaluate(
            SLearner(seed=11), stratified_split(no_truth, 11), n_resamples=RESAMPLES
        )
        assert "pehe" not in without.metrics
        assert without.get("pehe") is None

    def test_the_row_records_what_it_was_measured_on(self, binary_data):
        split = stratified_split(binary_data, 37)
        row = evaluate(SLearner(seed=37), split, n_resamples=RESAMPLES)
        assert row.seed == 37
        assert row.n_test == split.test.n_units
        assert row.scores.shape == (split.test.n_units,)
        assert row.fit_seconds > 0.0

    def test_evaluation_never_touches_the_validation_rows(self, binary_data):
        # A crude but decisive check: corrupting the validation part must not change any
        # reported number, because nothing in evaluate() is allowed to read it.
        split = stratified_split(binary_data, 11)
        clean = evaluate(SLearner(seed=11), split, n_resamples=RESAMPLES)

        poisoned = stratified_split(binary_data, 11)
        blank = np.zeros(poisoned.validation.n_units)
        object.__setattr__(poisoned.validation, "outcome", blank)
        after = evaluate(SLearner(seed=11), poisoned, n_resamples=RESAMPLES)
        assert clean.metrics["qini"].value == pytest.approx(after.metrics["qini"].value)


class TestRandomReference:
    def test_it_averages_over_many_rankings(self, binary_data):
        split = stratified_split(binary_data, 11)
        row = random_reference_row(split, n_rankings=40)
        assert row.estimator == "random-40"
        assert row.metrics["qini"].value == pytest.approx(0.0, abs=5e-3)
        assert row.fit_seconds == 0.0

    def test_it_does_not_report_ground_truth_metrics(self, binary_data):
        # PEHE of a random score vector is a number, but it is not a baseline for anything.
        row = random_reference_row(stratified_split(binary_data, 11), n_rankings=20)
        assert "pehe" not in row.metrics


class TestRun:
    def test_it_returns_a_row_per_estimator_per_seed_plus_the_reference(self, rows):
        expected = 2 * (len(DEFAULT_ESTIMATORS) + 1)
        assert len(rows) == expected

    def test_the_default_estimator_set_excludes_a_single_random_draw(self):
        assert "random" in ESTIMATORS
        assert "random" not in DEFAULT_ESTIMATORS

    def test_an_unknown_dataset_is_an_error(self):
        with pytest.raises(KeyError, match="unknown dataset"):
            run("not-a-dataset")

    def test_an_unknown_estimator_is_an_error(self):
        with pytest.raises(KeyError, match="unknown estimators"):
            run("synthetic-binary", estimators=["not-an-estimator"], seeds=(11,))

    def test_every_registered_dataset_loads(self):
        for key, loader in DATASETS.items():
            if key.startswith("synthetic"):
                assert loader().n_units > 0

    def test_an_estimator_that_knows_the_effect_beats_the_random_reference(self, rows):
        by_name = {row.estimator: row for row in rows if row.seed == 11}
        assert (
            by_name["s-learner"].metrics["qini"].value
            > by_name["random-200"].metrics["qini"].value
        )


class TestTable:
    def test_it_summarises_across_seeds(self, rows):
        summaries = summarise(rows)
        assert {s.n_seeds for s in summaries} == {2}
        for summary in summaries:
            assert summary.pooled.low <= summary.pooled.value <= summary.pooled.high

    def test_the_across_seed_spread_is_reported_separately(self, rows):
        summaries = summarise(rows)
        # Two different questions, kept apart: resampling the test set, and changing the
        # partition. They are not interchangeable and the table must not conflate them.
        assert any(
            s.across_seeds.n_resamples == 2 and s.pooled.n_resamples > 2 for s in summaries
        )

    def test_markdown_has_one_line_per_estimator(self, rows):
        table = to_markdown(rows)
        lines = table.strip().splitlines()
        estimators = {row.estimator for row in rows}
        assert len(lines) == 2 + len(estimators)
        assert "Qini (95% CI)" in lines[0]
        for name in estimators:
            assert f"`{name}`" in table

    def test_markdown_renders_intervals_not_bare_numbers(self, rows):
        # The repository rule: a bare Qini is a bug. A metric a row does not carry shows
        # as a dash, which is the honest answer; a number without an interval is not.
        for line in to_markdown(rows).strip().splitlines()[2:]:
            for cell in line.split("|")[2:-1]:
                assert cell.strip() == "-" or ("(" in cell and ")" in cell), cell

    def test_an_empty_set_of_rows_renders_without_crashing(self):
        assert "no rows" in to_markdown([])

    def test_json_round_trips_every_per_seed_number(self, rows, tmp_path):
        path = tmp_path / "results.json"
        write_json(rows, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload) == len(rows)
        assert payload[0]["metrics"]["qini"]["low"] <= payload[0]["metrics"]["qini"]["value"]
        assert payload[0]["seed"] == rows[0].seed


class TestPlots:
    def test_it_writes_a_figure(self, rows, tmp_path):
        split = stratified_split(DATASETS["synthetic-binary"](), 11)
        path = plot_qini_curves(
            [row for row in rows if row.seed == 11], split.test, tmp_path / "q.png"
        )
        assert path.exists()
        assert path.stat().st_size > 1_000

    def test_plotting_nothing_is_an_error(self, binary_data, tmp_path):
        with pytest.raises(ValueError, match="no benchmark rows"):
            plot_qini_curves([], binary_data, tmp_path / "q.png")

    def test_the_tie_seed_is_shared_with_the_metrics(self):
        # The figure and the table have to describe the same ranking, so the plot's default
        # tie seed is the committed one rather than a second copy of the same idea.
        default = inspect.signature(plot_qini_curves).parameters["tie_seed"].default
        assert default == TIE_SEED


class TestCli:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout

    def test_help_lists_the_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("data", "benchmark", "demo"):
            assert command in result.stdout

    def test_data_list_shows_every_source(self):
        result = runner.invoke(app, ["data", "list"])
        assert result.exit_code == 0
        assert "hillstrom" in result.stdout
        assert "criteo" in result.stdout

    def test_benchmark_runs_end_to_end(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "benchmark",
                "--dataset",
                "synthetic-binary",
                "--seeds",
                "11",
                "--resamples",
                "20",
                "--results-dir",
                str(tmp_path / "results"),
                "--figure-dir",
                str(tmp_path / "figures"),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "s-learner" in result.stdout
        assert (tmp_path / "results" / "synthetic-binary.json").exists()
        assert (tmp_path / "figures" / "qini-synthetic-binary.png").exists()

    def test_the_demo_says_it_is_not_built_yet(self):
        result = runner.invoke(app, ["demo"])
        assert result.exit_code == 1
        assert "week 7" in result.stdout


def test_ground_truth_metrics_are_skipped_for_a_ranking_baseline(binary_data):
    # The outcome ranking has scores, and they can be ranked, but they are not effects.
    split = stratified_split(binary_data, 11)
    from itx.estimators.baselines import OutcomeRanking

    row = evaluate(OutcomeRanking(seed=11), split, n_resamples=20)
    assert "qini" in row.metrics
    assert "pehe" not in row.metrics


class TestReadmeBlock:
    def test_a_named_block_is_replaced_in_place(self, tmp_path):
        path = tmp_path / "README.md"
        path.write_text(
            "before\n<!-- itx:table:demo -->\nold\n<!-- itx:end:demo -->\nafter\n",
            encoding="utf-8",
        )
        assert update_markdown_file(path, "demo", "| a |\n|---|\n| 1 |\n")
        text = path.read_text(encoding="utf-8")
        assert "old" not in text
        assert "| a |" in text
        assert text.startswith("before\n")
        assert text.endswith("after\n")

    def test_other_blocks_are_untouched(self, tmp_path):
        path = tmp_path / "README.md"
        path.write_text(
            "<!-- itx:table:one -->\nkeep\n<!-- itx:end:one -->\n"
            "<!-- itx:table:two -->\nreplace\n<!-- itx:end:two -->\n",
            encoding="utf-8",
        )
        update_markdown_file(path, "two", "new")
        text = path.read_text(encoding="utf-8")
        assert "keep" in text
        assert "replace" not in text

    def test_a_file_without_the_block_is_left_alone(self, tmp_path):
        path = tmp_path / "README.md"
        path.write_text("nothing here\n", encoding="utf-8")
        assert not update_markdown_file(path, "demo", "new")
        assert path.read_text(encoding="utf-8") == "nothing here\n"

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert not update_markdown_file(tmp_path / "absent.md", "demo", "new")

    def test_rewriting_the_same_table_twice_is_idempotent(self, tmp_path):
        path = tmp_path / "README.md"
        path.write_text("<!-- itx:table:demo -->\n<!-- itx:end:demo -->\n", encoding="utf-8")
        update_markdown_file(path, "demo", "| a |")
        once = path.read_text(encoding="utf-8")
        update_markdown_file(path, "demo", "| a |")
        assert path.read_text(encoding="utf-8") == once

    def test_the_readme_carries_a_block_for_every_benchmarked_dataset(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        for dataset in ("hillstrom", "ihdp"):
            assert f"<!-- itx:table:{dataset} -->" in readme
            assert f"<!-- itx:end:{dataset} -->" in readme
