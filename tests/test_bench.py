"""The benchmark end to end, on synthetic data so it runs in CI in seconds."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

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
    DATASETS,
    DEFAULT_ESTIMATORS,
    ESTIMATORS,
    UNTUNED,
    evaluate,
    random_reference_row,
    refit_seed,
    run,
)
from itx.bench.seeds import SEEDS, TIE_SEED, bootstrap_seed_for
from itx.bench.table import (
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
