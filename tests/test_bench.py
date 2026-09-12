"""The benchmark end to end, on synthetic data so it runs in CI in seconds."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from itx import cli
from itx.bench.grid import (
    GRID,
    MIN_CHILD_SAMPLES,
    NUM_LEAVES,
    Selection,
    applicable_grid,
    default_selection,
    select_config,
)
from itx.bench.plots import plot_qini_curves
from itx.bench.runner import (
    BENCHMARK_DATASETS,
    DATASETS,
    DEFAULT_ESTIMATORS,
    ESTIMATORS,
    UNTUNED,
    _duration,
    evaluate,
    random_reference_row,
    refit_seed,
    resamples_for,
    run,
)
from itx.bench.seeds import SEEDS, TIE_SEED, bootstrap_seed_for
from itx.bench.table import (
    compare_results,
    read_json,
    selected_configurations,
    summarise,
    to_markdown,
    update_markdown_file,
    write_json,
)
from itx.cli import app
from itx.data import synthetic
from itx.data.splits import stratified_split
from itx.estimators.lightgbm_base import DEFAULT_CONFIG
from itx.estimators.s_learner import SLearner

RESAMPLES = 40
runner = CliRunner()


@pytest.fixture(scope="module")
def rows():
    """Two seeds of the full benchmark on synthetic data, fitted once for the module.

    Tuning is off: selection multiplies the fits by the size of the grid, and what these
    tests check is the shape of the output, not which configuration wins. The grid itself
    is tested directly in TestGrid.
    """
    return run("synthetic-binary", seeds=(11, 23), n_resamples=RESAMPLES, tune=False)


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
                "--no-tune",
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


class TestGrid:
    def test_the_grid_is_every_combination_of_the_two_committed_knobs(self):
        assert len(GRID) == len(MIN_CHILD_SAMPLES) * len(NUM_LEAVES)
        assert {config.min_child_samples for config in GRID} == set(MIN_CHILD_SAMPLES)
        assert {config.num_leaves for config in GRID} == set(NUM_LEAVES)

    def test_the_grid_varies_nothing_else(self):
        # Identical across estimators is the point, and so is identical in every other
        # respect: if the learning rate moved with the leaf size, the table would be
        # reporting a two-dimensional search dressed up as one.
        assert {config.learning_rate for config in GRID} == {DEFAULT_CONFIG.learning_rate}
        assert {config.n_estimators for config in GRID} == {DEFAULT_CONFIG.n_estimators}

    def test_selection_picks_the_candidate_with_the_best_validation_score(self, binary_data):
        split = stratified_split(binary_data, 11)
        grid = (
            DEFAULT_CONFIG.with_(min_child_samples=5),
            DEFAULT_CONFIG.with_(min_child_samples=400),
        )
        selection = select_config(
            lambda config, seed: SLearner(config, seed=seed), split, seed=11, grid=grid
        )
        assert selection.tuned
        assert len(selection.scores) == 2
        assert selection.score == max(selection.scores)
        assert selection.config in grid

    def test_selection_never_reads_the_test_split(self, binary_data):
        # Corrupting the test rows must not change which configuration is chosen.
        split = stratified_split(binary_data, 11)
        grid = (DEFAULT_CONFIG.with_(min_child_samples=5), DEFAULT_CONFIG.with_(num_leaves=15))

        def factory(config, seed):
            return SLearner(config, seed=seed)

        clean = select_config(factory, split, seed=11, grid=grid)
        object.__setattr__(split.test, "outcome", np.zeros(split.test.n_units))
        after = select_config(factory, split, seed=11, grid=grid)
        assert clean.config == after.config
        assert clean.scores == pytest.approx(after.scores)

    def test_an_empty_grid_is_an_error(self, binary_data):
        with pytest.raises(ValueError, match="empty grid"):
            select_config(
                lambda config, seed: SLearner(config, seed=seed),
                stratified_split(binary_data, 11),
                seed=11,
                grid=(),
            )

    def test_an_untuned_selection_says_so(self):
        selection = default_selection()
        assert not selection.tuned
        assert "not tuned" in selection.describe()

    def test_a_tuned_selection_names_its_settings(self, binary_data):
        split = stratified_split(binary_data, 11)
        selection = select_config(
            lambda config, seed: SLearner(config, seed=seed),
            split,
            seed=11,
            grid=(DEFAULT_CONFIG,),
        )
        assert "min_child_samples=" in selection.describe()
        assert "num_leaves=" in selection.describe()

    def test_the_random_baseline_is_not_tuned(self):
        assert "random" in UNTUNED

    def test_a_tuned_run_records_what_it_chose(self, binary_data):
        rows = run(
            "synthetic-binary",
            estimators=["s-learner"],
            seeds=(11,),
            n_resamples=10,
            include_random_reference=False,
            tune=True,
        )
        assert rows[0].selection is not None
        assert rows[0].selection.tuned
        assert rows[0].selection.config in GRID

    def test_the_chosen_configurations_are_reported(self, binary_data):
        rows = run(
            "synthetic-binary",
            estimators=["s-learner"],
            seeds=(11,),
            n_resamples=10,
            include_random_reference=False,
            tune=True,
        )
        text = selected_configurations(rows)
        assert "s-learner" in text
        assert "min_child_samples=" in text

    def test_nothing_is_reported_when_nothing_was_tuned(self, rows):
        assert selected_configurations(rows) == ""


class TestApplicableGrid:
    """A candidate a dataset cannot fit is not a hyperparameter choice."""

    def test_a_large_training_set_can_afford_every_candidate(self):
        assert len(applicable_grid(25_000)) == len(GRID)

    def test_a_small_training_set_drops_the_heaviest_candidates(self):
        # IHDP's training split. A leaf size of 200 leaves room for two leaves, so the model
        # can barely split and its uplift collapses toward zero.
        affordable = applicable_grid(448)
        assert {config.min_child_samples for config in affordable} == {5, 20, 60}

    def test_it_never_returns_nothing(self):
        affordable = applicable_grid(4)
        assert affordable
        assert {config.min_child_samples for config in affordable} == {min(MIN_CHILD_SAMPLES)}

    def test_selection_only_ever_picks_something_affordable(self):
        data = synthetic.heterogeneous_effect(400, seed=0)
        split = stratified_split(data, 11)
        selection = select_config(
            lambda config, seed: SLearner(config, seed=seed), split, seed=11
        )
        assert selection.config in applicable_grid(split.train.n_units)


class TestReadJson:
    """A finished run should be redrawable without refitting anything."""

    def test_rows_survive_a_round_trip(self, rows, tmp_path):
        path = tmp_path / "results.json"
        write_json(rows, path)
        restored = read_json(path)
        assert [r.estimator for r in restored] == [r.estimator for r in rows]
        assert [r.seed for r in restored] == [r.seed for r in rows]
        for original, copy in zip(rows, restored, strict=True):
            assert set(copy.metrics) == set(original.metrics)
            for name, estimate in original.metrics.items():
                assert copy.metrics[name].value == pytest.approx(estimate.value)
                assert copy.metrics[name].low == pytest.approx(estimate.low)

    def test_the_table_is_identical_to_the_one_the_run_produced(self, rows, tmp_path):
        path = tmp_path / "results.json"
        write_json(rows, path)
        assert to_markdown(read_json(path)) == to_markdown(rows)

    def test_scores_are_not_stored_and_come_back_empty(self, rows, tmp_path):
        # Only the figures need them, and figures are cheap to redraw next to a refit.
        path = tmp_path / "results.json"
        write_json(rows, path)
        assert all(row.scores.size == 0 for row in read_json(path))

    def test_the_selected_configuration_survives(self, tmp_path):
        produced = run(
            "synthetic-binary",
            estimators=["s-learner"],
            seeds=(11,),
            n_resamples=10,
            include_random_reference=False,
            tune=True,
        )
        path = tmp_path / "results.json"
        write_json(produced, path)
        restored = read_json(path)[0]
        chosen = produced[0].selection
        assert chosen is not None
        assert restored.selection is not None
        assert restored.selection.config == chosen.config

    def test_the_report_command_redraws_from_disk(self, rows, tmp_path):
        write_json(rows, tmp_path / "synthetic-binary.json")
        readme = tmp_path / "README.md"
        readme.write_text(
            "<!-- itx:table:synthetic-binary -->\nold\n<!-- itx:end:synthetic-binary -->\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "report",
                "--dataset",
                "synthetic-binary",
                "--results-dir",
                str(tmp_path),
                "--readme",
                str(readme),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "s-learner" in readme.read_text(encoding="utf-8")
        assert "old" not in readme.read_text(encoding="utf-8")

    def test_reporting_a_run_that_never_happened_says_so(self, tmp_path):
        result = runner.invoke(
            app, ["report", "--dataset", "hillstrom", "--results-dir", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "run 'itx benchmark" in result.stdout


class TestRefitSeed:
    """Figures need scores, which are not stored. Refitting one seed is the cheap way back."""

    def test_it_returns_scores_but_no_metrics(self):
        selections = {"s-learner": default_selection()}
        rows, split = refit_seed("synthetic-binary", 11, selections=selections)
        assert len(rows) == 1
        assert rows[0].scores.size == split.test.n_units
        assert rows[0].metrics == {}

    def test_it_uses_the_configuration_it_is_given(self):
        chosen = Selection(
            config=DEFAULT_CONFIG.with_(min_child_samples=200, num_leaves=15),
            score=0.0,
            scores=(),
        )
        rows, _ = refit_seed("synthetic-binary", 11, selections={"s-learner": chosen})
        assert rows[0].selection is not None
        assert rows[0].selection.config.min_child_samples == 200

    def test_it_reproduces_the_scores_a_full_run_produced(self):
        # The figures have to describe the same fit the table does, or they are decoration.
        produced = run(
            "synthetic-binary",
            estimators=["s-learner"],
            seeds=(11,),
            n_resamples=10,
            include_random_reference=False,
            tune=False,
        )
        rows, _ = refit_seed(
            "synthetic-binary", 11, selections={"s-learner": default_selection()}
        )
        assert rows[0].scores == pytest.approx(produced[0].scores)

    def test_unknown_estimator_names_are_skipped_rather_than_raising(self):
        rows, _ = refit_seed(
            "synthetic-binary",
            11,
            selections={
                "s-learner": default_selection(),
                "not-an-estimator": default_selection(),
            },
        )
        assert [row.estimator for row in rows] == ["s-learner"]

    def test_the_figures_command_redraws_from_a_finished_run(self, rows, tmp_path):
        write_json(rows, tmp_path / "synthetic-binary.json")
        result = runner.invoke(
            app,
            [
                "figures",
                "--dataset",
                "synthetic-binary",
                "--results-dir",
                str(tmp_path),
                "--figure-dir",
                str(tmp_path / "figures"),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "figures" / "qini-synthetic-binary.png").exists()
        assert (tmp_path / "figures" / "calibration-synthetic-binary.png").exists()

    def test_asking_for_figures_from_a_run_that_never_happened_says_so(self, tmp_path):
        result = runner.invoke(
            app, ["figures", "--dataset", "hillstrom", "--results-dir", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "run 'itx benchmark" in result.stdout


class TestFigureStyle:
    """An imported library must not be able to change what the figures look like."""

    def test_the_figures_pin_their_style(self):
        from itx.bench import plots

        assert plots.FIGURE_STYLE == "default"

    def test_a_figure_is_the_same_whatever_seaborn_did_to_the_globals(self, rows, tmp_path):
        import matplotlib.pyplot as plt

        from itx.bench.plots import plot_qini_curves

        split = stratified_split(DATASETS["synthetic-binary"](), 11)
        seed_rows = [row for row in rows if row.seed == 11]

        first = plot_qini_curves(seed_rows, split.test, tmp_path / "a.png").read_bytes()
        # CausalML imports seaborn, which rewrites the global settings on import.
        plt.style.use("ggplot")
        try:
            second = plot_qini_curves(seed_rows, split.test, tmp_path / "b.png").read_bytes()
        finally:
            plt.style.use("default")
        assert first == second


class TestCompareResults:
    """The CI reproducibility check. It has to ignore timings and notice everything else."""

    def test_a_run_reproduces_itself(self, rows):
        assert compare_results(rows, rows) == []

    def test_a_different_fit_time_is_not_a_difference(self, rows):
        # The whole reason this exists rather than 'git diff': fit_seconds is wall-clock
        # and never reproduces, so a file diff would fail on every scheduled run.
        slower = [replace(row, fit_seconds=row.fit_seconds + 99.0) for row in rows]
        assert compare_results(rows, slower) == []

    def test_a_moved_metric_is_reported_with_both_numbers(self, rows):
        metric = next(iter(rows[0].metrics))
        moved = dict(rows[0].metrics)
        moved[metric] = replace(moved[metric], value=moved[metric].value + 0.5)
        current = [replace(rows[0], metrics=moved), *rows[1:]]

        differences = compare_results(rows, current)
        assert len(differences) == 1
        assert metric in differences[0]
        assert rows[0].estimator in differences[0]
        assert f"seed {rows[0].seed}" in differences[0]

    def test_a_moved_interval_is_reported_even_when_the_value_holds(self, rows):
        metric = next(iter(rows[0].metrics))
        moved = dict(rows[0].metrics)
        moved[metric] = replace(moved[metric], low=moved[metric].low - 0.5)
        current = [replace(rows[0], metrics=moved), *rows[1:]]

        differences = compare_results(rows, current)
        assert [f"{metric} low" in line for line in differences].count(True) == 1

    def test_a_tolerance_lets_a_small_move_through(self, rows):
        metric = next(iter(rows[0].metrics))
        moved = dict(rows[0].metrics)
        moved[metric] = replace(moved[metric], value=moved[metric].value + 1e-9)
        current = [replace(rows[0], metrics=moved), *rows[1:]]

        assert compare_results(rows, current) != []
        assert compare_results(rows, current, tolerance=1e-6) == []

    def test_a_missing_row_is_reported(self, rows):
        differences = compare_results(rows, rows[1:])
        assert len(differences) == 1
        assert "missing from the run" in differences[0]

    def test_an_extra_row_is_reported(self, rows):
        differences = compare_results(rows[1:], rows)
        assert len(differences) == 1
        assert "not committed" in differences[0]

    def test_a_changed_selection_is_reported(self, rows):
        chosen = Selection(config=DEFAULT_CONFIG.with_(num_leaves=99), score=0.5, scores=())
        current = [replace(rows[0], selection=chosen), *rows[1:]]
        differences = compare_results(rows, current)
        assert any("num_leaves=99" in line for line in differences)

    def test_a_changed_test_split_size_is_reported(self, rows):
        current = [replace(rows[0], n_test=rows[0].n_test + 1), *rows[1:]]
        assert any("test split has" in line for line in compare_results(rows, current))

    def test_two_nans_agree_rather_than_differing(self, rows):
        # A calibration slope can legitimately be NaN, and NaN != NaN would otherwise make
        # every such row a permanent CI failure.
        metric = next(iter(rows[0].metrics))
        blank = dict(rows[0].metrics)
        blank[metric] = replace(blank[metric], value=float("nan"))
        current = [replace(rows[0], metrics=blank), *rows[1:]]
        assert compare_results(current, current) == []
        assert compare_results(rows, current) != []


class TestCompareCommand:
    def test_it_reports_success_on_a_file_against_itself(self, tmp_path, rows):
        path = tmp_path / "a.json"
        write_json(rows, path)
        result = runner.invoke(app, ["compare", str(path), str(path)])
        assert result.exit_code == 0
        assert "reproduces" in result.stdout

    def test_it_exits_nonzero_and_names_what_moved(self, tmp_path, rows):
        baseline = tmp_path / "a.json"
        current = tmp_path / "b.json"
        write_json(rows, baseline)

        metric = next(iter(rows[0].metrics))
        moved = dict(rows[0].metrics)
        moved[metric] = replace(moved[metric], value=moved[metric].value + 0.5)
        write_json([replace(rows[0], metrics=moved), *rows[1:]], current)

        result = runner.invoke(app, ["compare", str(baseline), str(current)])
        assert result.exit_code == 1
        assert metric in result.stdout

    def test_a_missing_file_is_its_own_exit_code(self, tmp_path, rows):
        path = tmp_path / "a.json"
        write_json(rows, path)
        result = runner.invoke(app, ["compare", str(path), str(tmp_path / "nope.json")])
        assert result.exit_code == 2


class TestBenchmarkSweep:
    """``itx benchmark --all``, the command PLAN.md section 4 says the table comes out of."""

    def test_every_swept_dataset_is_a_real_one(self):
        for name in BENCHMARK_DATASETS:
            assert name in DATASETS

    def test_the_sweep_is_the_five_real_datasets(self):
        assert set(BENCHMARK_DATASETS) == {"hillstrom", "ihdp", "acic", "lenta", "criteo"}

    def test_synthetic_data_is_not_swept(self):
        # A results table on data this package invented would be a table about this
        # package. The generators exist to make tests fail meaningfully, not to be reported.
        assert not any(name.startswith("synthetic") for name in BENCHMARK_DATASETS)

    def test_the_full_criteo_fit_is_not_swept(self):
        # It is the headline single fit, run deliberately, not folded into a sweep.
        assert "criteo-full" not in BENCHMARK_DATASETS

    def test_the_cheapest_datasets_come_first(self):
        # So a sweep that is going to fail on a missing download fails in the first minute
        # rather than the third hour.
        assert BENCHMARK_DATASETS.index("ihdp") < BENCHMARK_DATASETS.index("hillstrom")
        assert BENCHMARK_DATASETS.index("hillstrom") < BENCHMARK_DATASETS.index("criteo")

    def test_the_protocol_resample_counts(self):
        # PLAN.md section 4: 1,000 on the small sets, 200 on the Criteo subsample.
        assert resamples_for("criteo") == 200
        assert resamples_for("criteo-full") == 200
        assert resamples_for("hillstrom") == 1_000
        assert resamples_for("ihdp") == 1_000

    def test_all_runs_every_dataset_once_and_ignores_dataset(self, monkeypatch):
        seen = []
        monkeypatch.setattr(cli, "_benchmark_one", lambda name, **_: seen.append(name))
        result = runner.invoke(app, ["benchmark", "--all", "--dataset", "hillstrom"])
        assert result.exit_code == 0
        assert seen == list(BENCHMARK_DATASETS)

    def test_without_all_only_the_named_dataset_runs(self, monkeypatch):
        seen = []
        monkeypatch.setattr(cli, "_benchmark_one", lambda name, **_: seen.append(name))
        result = runner.invoke(app, ["benchmark", "--dataset", "ihdp"])
        assert result.exit_code == 0
        assert seen == ["ihdp"]

    def test_an_explicit_resample_count_overrides_the_protocol(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            cli, "_benchmark_one", lambda name, **kw: seen.update({name: kw["resamples"]})
        )
        runner.invoke(app, ["benchmark", "--dataset", "criteo", "--resamples", "7"])
        assert seen == {"criteo": 7}

    def test_omitting_it_leaves_the_protocol_to_decide(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            cli, "_benchmark_one", lambda name, **kw: seen.update({name: kw["resamples"]})
        )
        runner.invoke(app, ["benchmark", "--dataset", "criteo"])
        assert seen == {"criteo": None}


class TestProgress:
    """A run that says nothing for hours cannot be told apart from a hung one."""

    def test_silent_by_default(self, binary_data, monkeypatch):
        # The tests want quiet; only the CLI passes a printer.
        monkeypatch.setattr("itx.bench.runner.DATASETS", {"probe": lambda: binary_data})
        captured = []
        monkeypatch.setattr("builtins.print", lambda *a, **k: captured.append(a))
        run("probe", estimators=["s-learner"], seeds=(11,), n_resamples=10, tune=False)
        assert captured == []

    def test_it_reports_the_shape_then_one_line_per_fit_then_the_total(
        self, binary_data, monkeypatch
    ):
        monkeypatch.setattr("itx.bench.runner.DATASETS", {"probe": lambda: binary_data})
        lines: list[str] = []
        run(
            "probe",
            estimators=["s-learner", "outcome-ranking"],
            seeds=(11, 23),
            n_resamples=10,
            tune=False,
            progress=lines.append,
        )
        assert lines[0].startswith("probe: ")
        assert "2 estimators over 2 seeds, 4 fits" in lines[0]
        assert len(lines) == 6
        assert lines[1].startswith("  [1/4] seed 11 s-learner:")
        assert lines[4].startswith("  [4/4] seed 23 outcome-ranking:")
        assert lines[-1].startswith("probe: finished in ")

    def test_one_seed_is_not_called_seeds(self, binary_data, monkeypatch):
        monkeypatch.setattr("itx.bench.runner.DATASETS", {"probe": lambda: binary_data})
        lines: list[str] = []
        run(
            "probe",
            estimators=["s-learner"],
            seeds=(11,),
            n_resamples=10,
            tune=False,
            progress=lines.append,
        )
        assert "over 1 seed," in lines[0]


class TestDuration:
    """Elapsed times are for a person reading a log, so they are not all in seconds."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0.0s"),
            (4.25, "4.2s"),
            (59.9, "59.9s"),
            (60.0, "1m00s"),
            (135.0, "2m15s"),
            (3599.0, "59m59s"),
            (3600.0, "1h00m"),
            (9061.0, "2h31m"),
        ],
    )
    def test_it_formats(self, seconds, expected):
        assert _duration(seconds) == expected


class TestCheckpointing:
    """A killed run has to cost one fit, not all of them (PLAN.md change 34)."""

    def test_on_row_is_called_once_per_finished_row(self, binary_data, monkeypatch):
        monkeypatch.setattr("itx.bench.runner.DATASETS", {"probe": lambda: binary_data})
        seen: list[str] = []
        rows = run(
            "probe",
            estimators=["s-learner", "outcome-ranking"],
            seeds=(11, 23),
            n_resamples=10,
            tune=False,
            on_row=lambda row: seen.append(f"{row.estimator}:{row.seed}"),
        )
        assert len(seen) == len(rows)
        assert "s-learner:11" in seen
        # The averaged random baseline is banked too: it is cheap, but leaving it out
        # would mean a resumed run silently redrew 200 rankings with a different draw.
        assert "random-200:11" in seen

    def test_completed_rows_are_reused_and_not_refitted(self, binary_data, monkeypatch):
        monkeypatch.setattr("itx.bench.runner.DATASETS", {"probe": lambda: binary_data})
        first = run(
            "probe", estimators=["s-learner"], seeds=(11, 23), n_resamples=10, tune=False
        )
        # Hand back everything from seed 11 and nothing from seed 23.
        banked = [row for row in first if row.seed == 11]

        refitted: list[str] = []
        second = run(
            "probe",
            estimators=["s-learner"],
            seeds=(11, 23),
            n_resamples=10,
            tune=False,
            completed=banked,
            on_row=lambda row: refitted.append(f"{row.estimator}:{row.seed}"),
        )
        assert all(":11" not in entry for entry in refitted)
        assert "s-learner:23" in refitted
        reused = next(r for r in second if r.seed == 11 and r.estimator == "s-learner")
        original = next(r for r in first if r.seed == 11 and r.estimator == "s-learner")
        assert reused is original

    def test_a_resumed_run_reproduces_an_uninterrupted_one(self, binary_data, monkeypatch):
        # The property that matters: resuming must not change a single number.
        monkeypatch.setattr("itx.bench.runner.DATASETS", {"probe": lambda: binary_data})
        whole = run(
            "probe", estimators=["s-learner"], seeds=(11, 23), n_resamples=10, tune=False
        )
        resumed = run(
            "probe",
            estimators=["s-learner"],
            seeds=(11, 23),
            n_resamples=10,
            tune=False,
            completed=[row for row in whole if row.seed == 11],
        )
        assert compare_results(whole, resumed, tolerance=0.0) == []

    def test_progress_says_which_rows_came_from_the_checkpoint(self, binary_data, monkeypatch):
        monkeypatch.setattr("itx.bench.runner.DATASETS", {"probe": lambda: binary_data})
        first = run(
            "probe", estimators=["s-learner"], seeds=(11, 23), n_resamples=10, tune=False
        )
        lines: list[str] = []
        run(
            "probe",
            estimators=["s-learner"],
            seeds=(11, 23),
            n_resamples=10,
            tune=False,
            completed=[row for row in first if row.seed == 11],
            progress=lines.append,
        )
        assert any("from checkpoint" in line for line in lines)
        assert lines[-1].endswith("1 from checkpoint")


class TestCheckpointFile:
    """The CLI end of it: the file, the opt-in, and the cleanup."""

    def test_the_path_is_a_sibling_nobody_mistakes_for_a_result(self, tmp_path):
        path = cli.checkpoint_path(tmp_path, "lenta")
        assert path.name == "lenta.checkpoint.json"
        assert path.name != "lenta.json"

    def test_a_finished_run_leaves_no_checkpoint_behind(
        self, tmp_path, binary_data, monkeypatch
    ):
        monkeypatch.setattr("itx.bench.runner.DATASETS", {"probe": lambda: binary_data})
        result = runner.invoke(
            app,
            [
                "benchmark",
                "--dataset",
                "probe",
                "--no-tune",
                "--no-plot",
                "--seeds",
                "11",
                "--resamples",
                "10",
                "--results-dir",
                str(tmp_path),
                "--readme",
                str(tmp_path / "r.md"),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "probe.json").is_file()
        assert not cli.checkpoint_path(tmp_path, "probe").exists()

    def test_a_stale_checkpoint_is_ignored_and_announced_without_resume(self, tmp_path, rows):
        checkpoint = cli.checkpoint_path(tmp_path, "probe")
        write_json(rows, checkpoint)
        lines: list[str] = []
        reused = cli._resume_from(checkpoint, "probe", resume=False, say=lines.append)
        assert reused == ()
        assert "--resume" in lines[0]

    def test_with_resume_the_rows_come_back(self, tmp_path, rows):
        checkpoint = cli.checkpoint_path(tmp_path, "probe")
        write_json(rows, checkpoint)
        lines: list[str] = []
        reused = cli._resume_from(checkpoint, "probe", resume=True, say=lines.append)
        assert len(reused) == len(rows)
        assert "resuming" in lines[0]

    def test_no_checkpoint_is_silent(self, tmp_path):
        lines: list[str] = []
        reused = cli._resume_from(
            cli.checkpoint_path(tmp_path, "probe"), "probe", resume=True, say=lines.append
        )
        assert reused == ()
        assert lines == []
