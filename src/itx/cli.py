"""The ``itx`` command line: pull data, run the benchmark, build the demo.

Everything in the results table has to come out of one command a stranger can run
(PLAN.md section 4), so the CLI is the interface the project is judged on, not a
convenience wrapper around a notebook.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer

from itx import __version__
from itx.demo.build import DEMO_RESAMPLES

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from itx.bench.grid import Selection
    from itx.bench.runner import BenchmarkRow
    from itx.sensitivity.rosenbaum import RosenbaumBound

from itx.metrics.power import DEFAULT_ALPHA, DEFAULT_POWER

app = typer.Typer(
    name="itx",
    help="Intervention Targeting Engine: uplift modelling with honest evaluation.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Download and verify the raw datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")

DEFAULT_RESULTS_DIR = Path("results")
#: Where ``itx demo build`` writes; the static page beside it reads ``data/*.json``.
DEFAULT_DEMO_DIR = Path("demo") / "data"
DEFAULT_FIGURE_DIR = Path("docs/figures")
DEFAULT_README = Path("README.md")


def _printer() -> Callable[[str], None]:
    """A progress line printer that survives being redirected to a file.

    Python block-buffers stdout when it is not a terminal, so a benchmark redirected to a
    log writes nothing at all until it exits. That is exactly the run where the progress
    matters, so every line is flushed as it is written.
    """

    def say(line: str) -> None:
        typer.echo(line)
        sys.stdout.flush()

    return say


def checkpoint_path(results_dir: Path, dataset: str) -> Path:
    """Where a run banks its finished rows.

    Args:
        results_dir: Where the final results file goes.
        dataset: The dataset key.

    Returns:
        A sibling of the results file, distinguishable from it by name so that nobody
        mistakes a half-finished run for a result. It is deleted when the run completes.
    """
    return results_dir / f"{dataset}.checkpoint.json"


def _resume_from(
    checkpoint: Path, dataset: str, *, resume: bool, say: Callable[[str], None]
) -> Sequence[BenchmarkRow]:
    """Rows to reuse from an interrupted run, and the warnings that go with them.

    Resuming is opt-in rather than automatic. A checkpoint written by different code is
    indistinguishable from one written by this code, and silently mixing the two would
    produce a results table whose rows came from two versions of the package. Requiring
    the flag puts that judgement with the person who knows whether anything changed.

    Args:
        checkpoint: The checkpoint file, which may not exist.
        dataset: The dataset key, for the message.
        resume: Whether the caller asked to resume.
        say: Progress printer.

    Returns:
        The rows to reuse, empty unless resuming from a checkpoint that exists.
    """
    from itx.bench.table import read_json

    if not checkpoint.is_file():
        return ()
    if not resume:
        say(
            f"note: {checkpoint} exists from an interrupted run and is being ignored. "
            f"Pass --resume to reuse it, but only if nothing has changed since it was "
            f"written; it will be overwritten as this run proceeds."
        )
        return ()

    rows = read_json(checkpoint)
    say(f"{dataset}: resuming, {len(rows)} rows from {checkpoint}")
    return rows


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool, typer.Option("--version", help="Print the version and exit.")
    ] = False,
) -> None:
    """Intervention Targeting Engine."""
    if version:
        typer.echo(__version__)
        raise typer.Exit


@data_app.command("pull")
def data_pull(
    keys: Annotated[
        list[str] | None,
        typer.Argument(help="Sources to fetch; all of them if omitted."),
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-download even when cached.")] = False,
) -> None:
    """Download raw files and verify them against the committed checksums."""
    from itx.data.download import disk_usage, fetch, human_bytes
    from itx.data.registry import SOURCES

    wanted = keys if keys else list(SOURCES)
    for key in wanted:
        fetch(key, force=force)
    typer.echo(f"cached data: {human_bytes(disk_usage())}")


@data_app.command("list")
def data_list() -> None:
    """Show every registered source, where it comes from and whether it is cached."""
    from itx.data.download import human_bytes
    from itx.data.registry import SOURCES, data_dir

    directory = data_dir()
    typer.echo(f"data directory: {directory}")
    for key, spec in SOURCES.items():
        path = directory / spec.filename
        state = f"cached, {human_bytes(path.stat().st_size)}" if path.exists() else "not cached"
        typer.echo(f"  {key:<14} {state:<20} {spec.licence}")


@data_app.command("verify")
def data_verify() -> None:
    """Re-hash every cached file and compare it with the committed checksum."""
    from itx.data.download import sha256_of
    from itx.data.registry import SOURCES, data_dir

    directory = data_dir()
    failures = 0
    for key, spec in SOURCES.items():
        path = directory / spec.filename
        if not path.exists():
            typer.echo(f"  {key:<14} not cached")
            continue
        actual = sha256_of(path)
        if actual == spec.expected_sha256:
            typer.echo(f"  {key:<14} ok")
        else:
            failures += 1
            typer.echo(f"  {key:<14} MISMATCH: {actual}")
    if failures:
        raise typer.Exit(code=1)


@app.command("benchmark")
def benchmark(
    dataset: Annotated[str, typer.Option(help="Dataset key to run.")] = "hillstrom",
    run_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Run every benchmark dataset in turn, cheapest first. Ignores --dataset.",
        ),
    ] = False,
    estimators: Annotated[
        str | None,
        typer.Option(help="Comma-separated estimator keys; the default set if omitted."),
    ] = None,
    seeds: Annotated[
        str | None,
        typer.Option(help="Comma-separated split seeds; the committed five if omitted."),
    ] = None,
    resamples: Annotated[
        int | None,
        typer.Option(
            help="Bootstrap resamples per row; the dataset's protocol value if omitted."
        ),
    ] = None,
    results_dir: Annotated[
        Path, typer.Option(help="Where the per-seed JSON goes.")
    ] = DEFAULT_RESULTS_DIR,
    figure_dir: Annotated[
        Path, typer.Option(help="Where the Qini figure goes.")
    ] = DEFAULT_FIGURE_DIR,
    plot: Annotated[bool, typer.Option(help="Write the Qini curve figure.")] = True,
    tune: Annotated[
        bool, typer.Option(help="Select hyperparameters on the validation split.")
    ] = True,
    readme: Annotated[
        Path, typer.Option(help="Markdown file whose results block is regenerated.")
    ] = DEFAULT_README,
    resume: Annotated[
        bool,
        typer.Option(help="Reuse finished rows from an interrupted run's checkpoint."),
    ] = False,
) -> None:
    """Fit, evaluate and report, with intervals and both baselines.

    One dataset by default, or every benchmark dataset with ``--all``, which is the command
    PLAN.md section 4 says the whole results table comes out of.
    """
    from itx.bench.runner import BENCHMARK_DATASETS
    from itx.bench.seeds import SEEDS

    estimator_keys = [e.strip() for e in estimators.split(",")] if estimators else None
    seed_values = [int(s) for s in seeds.split(",")] if seeds else list(SEEDS)
    wanted = list(BENCHMARK_DATASETS) if run_all else [dataset]

    for index, name in enumerate(wanted):
        if len(wanted) > 1:
            typer.echo("")
            typer.echo(f"=== {name} ({index + 1} of {len(wanted)}) ===")
        _benchmark_one(
            name,
            estimator_keys=estimator_keys,
            seed_values=seed_values,
            resamples=resamples,
            results_dir=results_dir,
            figure_dir=figure_dir,
            plot=plot,
            tune=tune,
            readme=readme,
            resume=resume,
        )


def _benchmark_one(
    dataset: str,
    *,
    estimator_keys: list[str] | None,
    seed_values: list[int],
    resamples: int | None,
    results_dir: Path,
    figure_dir: Path,
    plot: bool,
    tune: bool,
    readme: Path,
    resume: bool,
) -> None:
    """Run one dataset: fit, evaluate, then write the JSON, the table and the figures."""
    from itx.bench.plots import plot_calibration, plot_qini_curves
    from itx.bench.runner import DATASETS, resamples_for, run
    from itx.bench.table import (
        policy_table,
        results_table,
        selected_configurations,
        update_markdown_file,
        write_json,
    )
    from itx.data.splits import stratified_split

    say = _printer()
    checkpoint = checkpoint_path(results_dir, dataset)
    completed = _resume_from(checkpoint, dataset, resume=resume, say=say)
    banked: list[BenchmarkRow] = list(completed)

    def bank(row: BenchmarkRow) -> None:
        """Append a finished row to the checkpoint, so an interruption costs one fit."""
        banked.append(row)
        write_json(banked, checkpoint)

    rows = run(
        dataset,
        estimators=estimator_keys,
        seeds=seed_values,
        n_resamples=resamples if resamples is not None else resamples_for(dataset),
        tune=tune,
        progress=say,
        completed=completed,
        on_row=bank,
    )
    table = results_table(rows)
    chosen = selected_configurations(rows)
    if chosen:
        note = "Selected on the validation split from the committed grid:"
        table = "\n".join([table, note, "", chosen])
    policy = policy_table(rows)
    typer.echo("")
    typer.echo(table)
    if policy:
        typer.echo(policy)

    json_path = results_dir / f"{dataset}.json"
    write_json(rows, json_path)
    typer.echo(f"per-seed results: {json_path}")

    # The run finished, so the half-finished copy is no longer something anyone wants to
    # resume from, and leaving it would make the next run print a stale warning.
    checkpoint.unlink(missing_ok=True)

    if update_markdown_file(readme, dataset, table):
        typer.echo(f"results block updated: {readme}")
    if policy and update_markdown_file(readme, f"{dataset}-policy", policy):
        typer.echo(f"policy block updated: {readme}")

    if plot:
        first_seed = seed_values[0]
        # The random reference row carries one of its 200 draws as `scores`; drawn as a
        # solid curve under the 200-draw label it would be a picture of the wrong thing.
        # The figure's dashed line already is the random reference (PLAN.md change 56).
        seed_rows = [
            row
            for row in rows
            if row.seed == first_seed and not row.estimator.startswith("random-")
        ]
        # A row read back from a checkpoint carries its metrics but not its per-unit
        # scores, which are large and are not serialised. The tables need the metrics and
        # the figures need the scores, so a resumed run can write the first and not the
        # second. Refitting one seed is what `itx figures` is for.
        if any(row.scores.size == 0 for row in seed_rows):
            typer.echo(
                f"figures skipped: seed {first_seed} came from the checkpoint, which does "
                f"not carry per-unit scores. Run 'itx figures --dataset {dataset}'."
            )
            return
        split = stratified_split(DATASETS[dataset](), first_seed)
        qini_path = plot_qini_curves(seed_rows, split.test, figure_dir / f"qini-{dataset}.png")
        typer.echo(f"figure: {qini_path}")
        effect_rows = [row for row in seed_rows if "calibration_slope" in row.metrics]
        if effect_rows:
            calibration_path = plot_calibration(
                effect_rows, split.test, figure_dir / f"calibration-{dataset}.png"
            )
            typer.echo(f"figure: {calibration_path}")


@app.command("report")
def report(
    dataset: Annotated[str, typer.Option(help="Dataset key to redraw.")] = "hillstrom",
    results_dir: Annotated[
        Path, typer.Option(help="Where the per-seed JSON lives.")
    ] = DEFAULT_RESULTS_DIR,
    readme: Annotated[
        Path, typer.Option(help="Markdown file whose results block is regenerated.")
    ] = DEFAULT_README,
) -> None:
    """Redraw a results table from a finished run, without refitting anything."""
    from itx.bench.table import (
        policy_table,
        read_json,
        results_table,
        selected_configurations,
        update_markdown_file,
    )

    path = results_dir / f"{dataset}.json"
    if not path.is_file():
        typer.echo(f"no results at {path}; run 'itx benchmark --dataset {dataset}' first")
        raise typer.Exit(code=1)

    rows = read_json(path)
    table = results_table(rows)
    chosen = selected_configurations(rows)
    if chosen:
        note = "Selected on the validation split from the committed grid:"
        table = "\n".join([table, note, "", chosen])
    policy = policy_table(rows)
    typer.echo("")
    typer.echo(table)
    if policy:
        typer.echo(policy)
    if update_markdown_file(readme, dataset, table):
        typer.echo(f"results block updated: {readme}")
    if policy and update_markdown_file(readme, f"{dataset}-policy", policy):
        typer.echo(f"policy block updated: {readme}")


@app.command("compare")
def compare(
    baseline: Annotated[Path, typer.Argument(help="A results file from a previous run.")],
    current: Annotated[Path, typer.Argument(help="A results file from a fresh run.")],
    tolerance: Annotated[
        float, typer.Option(help="Absolute difference tolerated per number.")
    ] = 0.0,
) -> None:
    """Check that a fresh run reproduces a committed one, and say what moved if it did not.

    This is what CI asserts after re-running a benchmark. It is not a file diff: the
    results file records ``fit_seconds``, which is wall-clock and never reproduces, so a
    diff would fail on every run for a reason that has nothing to do with the numbers.
    """
    from itx.bench.table import compare_results, read_json

    for path in (baseline, current):
        if not path.is_file():
            typer.echo(f"no results at {path}")
            raise typer.Exit(code=2)

    differences = compare_results(read_json(baseline), read_json(current), tolerance=tolerance)
    if not differences:
        typer.echo(f"{current} reproduces {baseline}")
        return

    typer.echo(f"{len(differences)} difference(s) between {baseline} and {current}:")
    for line in differences:
        typer.echo(f"  {line}")
    raise typer.Exit(code=1)


@app.command("figures")
def figures(
    dataset: Annotated[str, typer.Option(help="Dataset key to redraw.")] = "hillstrom",
    results_dir: Annotated[
        Path, typer.Option(help="Where the per-seed JSON lives.")
    ] = DEFAULT_RESULTS_DIR,
    figure_dir: Annotated[
        Path, typer.Option(help="Where the figures go.")
    ] = DEFAULT_FIGURE_DIR,
) -> None:
    """Redraw a dataset's figures from a finished run, refitting only the first seed."""
    from itx.bench.plots import plot_calibration, plot_qini_curves
    from itx.bench.runner import refit_seed
    from itx.bench.table import read_json

    path = results_dir / f"{dataset}.json"
    if not path.is_file():
        typer.echo(f"no results at {path}; run 'itx benchmark --dataset {dataset}' first")
        raise typer.Exit(code=1)

    from itx.bench.grid import default_selection

    rows = read_json(path)
    seed = rows[0].seed
    # A run made with --no-tune records no selection, so fall back to the default rather
    # than dropping the estimator: the estimator list comes from the rows, not from which
    # of them happened to have been tuned.
    selections = {
        row.estimator: row.selection if row.selection is not None else default_selection()
        for row in rows
        if row.seed == seed
    }
    had_calibration = {row.estimator for row in rows if "calibration_slope" in row.metrics}

    refitted, split = refit_seed(dataset, seed, selections=selections)
    qini_path = plot_qini_curves(refitted, split.test, figure_dir / f"qini-{dataset}.png")
    typer.echo(f"figure: {qini_path}")

    effect_rows = [row for row in refitted if row.estimator in had_calibration]
    if effect_rows:
        calibration_path = plot_calibration(
            effect_rows, split.test, figure_dir / f"calibration-{dataset}.png"
        )
        typer.echo(f"figure: {calibration_path}")


@app.command("power")
def power(
    base_rate: Annotated[
        float | None,
        typer.Option(help="Outcome rate without the intervention, for a 0/1 outcome."),
    ] = None,
    outcome_sd: Annotated[
        float | None,
        typer.Option(help="Outcome standard deviation, for a continuous outcome."),
    ] = None,
    effect: Annotated[
        float | None, typer.Option(help="Average treatment effect expected, in outcome units.")
    ] = None,
    relative_effect: Annotated[
        float | None,
        typer.Option(help="Average effect as a share of the base rate; 0.1 is a 10% lift."),
    ] = None,
    budget: Annotated[
        float, typer.Option(help="Share of the population the intervention budget covers.")
    ] = 0.2,
    treated_share: Annotated[
        float, typer.Option(help="Share assigned to treatment; 0.5 is the cheapest split.")
    ] = 0.5,
    have: Annotated[
        int | None,
        typer.Option(help="Units you already have, to score instead of sizing from scratch."),
    ] = None,
    alpha: Annotated[float, typer.Option(help="Two-sided significance level.")] = DEFAULT_ALPHA,
    target_power: Annotated[
        float, typer.Option("--target-power", help="Power to quote everything at.")
    ] = DEFAULT_POWER,
) -> None:
    """Size the holdout a targeting decision needs, before collecting anything.

    Every other command here reads data that exists. This one is for the question that comes
    first: is the study worth running, and how big does it have to be. It answers in two
    parts, because detecting that an intervention works and being able to rank who should
    get it are different problems and the second is far more expensive.

    With ``--have`` it runs the other way, and reports how strong the heterogeneity would
    have to be for a study of that size to find it.
    """
    from itx.metrics.power import binary_outcome_sd, detectable_lift, requirement_table

    if (base_rate is None) == (outcome_sd is None):
        typer.echo("give exactly one of --base-rate (a 0/1 outcome) or --outcome-sd")
        raise typer.Exit(code=2)
    if (effect is None) == (relative_effect is None):
        typer.echo("give exactly one of --effect or --relative-effect")
        raise typer.Exit(code=2)

    try:
        spread = (
            binary_outcome_sd(base_rate) if base_rate is not None else float(outcome_sd or 0.0)
        )
        if relative_effect is not None:
            if base_rate is None:
                typer.echo(
                    "--relative-effect needs --base-rate; with --outcome-sd use --effect"
                )
                raise typer.Exit(code=2)
            absolute = relative_effect * base_rate
        else:
            absolute = float(effect or 0.0)

        if have is not None:
            needed = detectable_lift(
                n_units=have,
                outcome_sd=spread,
                average_effect=absolute,
                budget=budget,
                treated_share=treated_share,
                alpha=alpha,
                power=target_power,
            )
            typer.echo("")
            typer.echo(
                f"With {have:,} units, an average effect of {absolute:.4g} and a "
                f"{budget:.0%} budget:"
            )
            typer.echo("")
            typer.echo(
                f"  only a top group responding at least {needed:.2f}x the average could be "
                f"told from random targeting."
            )
            typer.echo(
                "\nAnd that is a floor: it prices measuring a ranking, not learning "
                "one from the same rows.\nIf you cannot argue the top group is that "
                "much better, this study cannot answer the targeting question, whatever "
                "a table of Qini numbers computed on it may look like."
            )
            return

        report = requirement_table(
            outcome_sd=spread,
            average_effect=absolute,
            budget=budget,
            treated_share=treated_share,
            alpha=alpha,
            power=target_power,
        )
    except ValueError as err:
        typer.echo(str(err))
        raise typer.Exit(code=2) from err

    typer.echo("")
    typer.echo(
        f"Units needed at a {budget:.0%} budget, {alpha:.0%} significance, "
        f"{target_power:.0%} power"
    )
    typer.echo("")
    typer.echo(report.to_markdown())
    typer.echo(report.summary())


@app.command("diagnose")
def diagnose(
    dataset: Annotated[str, typer.Option(help="Dataset key to diagnose.")] = "hillstrom",
    csv: Annotated[
        Path | None,
        typer.Option(help="Your own CSV instead of a built-in dataset."),
    ] = None,
    treatment: Annotated[
        str | None, typer.Option(help="With --csv: the 0/1 intervention column.")
    ] = None,
    outcome: Annotated[str | None, typer.Option(help="With --csv: the outcome column.")] = None,
    features: Annotated[
        str | None, typer.Option(help="With --csv: comma-separated feature columns.")
    ] = None,
    categorical: Annotated[
        str | None, typer.Option(help="With --csv: which features are categories.")
    ] = None,
    design: Annotated[
        str | None,
        typer.Option(
            help="With --csv: 'randomised' or 'observational'. No default, on purpose."
        ),
    ] = None,
    risk_is_low_outcome: Annotated[
        bool,
        typer.Option(
            "--risk-is-low-outcome",
            help="With --csv: set when a LOW outcome is the risky end, as on dollars retained.",
        ),
    ] = False,
    outcome_polarity: Annotated[
        str | None,
        typer.Option(
            "--outcome-polarity",
            help=(
                "With --csv: 'higher-is-better' (a response, dollars kept) or "
                "'lower-is-better' (churn, a readmission). No default, on purpose."
            ),
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="Split seed; the first committed one if omitted.")
    ] = None,
    bins: Annotated[int, typer.Option(help="Bands to cut the population into.")] = 10,
    resamples: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 500,
) -> None:
    """Say whether uplift modelling is worth the work here, for the price of one model.

    Cuts the population into bands of predicted risk and measures what the intervention did
    in each. Whether ranking by risk approximates ranking by uplift is a property of the
    data that varies enormously between problems, and this is the cheap way to find out
    before fitting anything (PLAN.md change 32).
    """
    from itx.bench.runner import DATASETS
    from itx.bench.seeds import SEEDS, TIE_SEED
    from itx.data.splits import stratified_split
    from itx.data.user_csv import ColumnSpec, UnusableDataError, load_csv
    from itx.metrics.risk_deciles import risk_deciles

    say = _printer()
    caveat = ""
    if csv is not None:
        missing = [
            flag
            for flag, value in (
                ("--treatment", treatment),
                ("--outcome", outcome),
                ("--features", features),
                ("--design", design),
                ("--outcome-polarity", outcome_polarity),
            )
            if value is None
        ]
        if missing:
            typer.echo(f"--csv needs {', '.join(missing)}")
            raise typer.Exit(code=2)
        if outcome_polarity not in ("higher-is-better", "lower-is-better"):
            typer.echo(
                f"--outcome-polarity must be 'higher-is-better' or 'lower-is-better', got "
                f"{outcome_polarity!r}. There is no default: the same file encoded as "
                f"'churned' and as 'retained' gets opposite verdicts out of this command, "
                f"and only you know which way yours runs."
            )
            raise typer.Exit(code=2)
        if design not in ("randomised", "observational"):
            typer.echo(
                f"--design must be 'randomised' or 'observational', got {design!r}. There is "
                f"no default: on an observational design every band's number mixes the effect "
                f"with whoever was likelier to be treated, and nothing here can detect that."
            )
            raise typer.Exit(code=2)
        try:
            data = load_csv(
                csv,
                ColumnSpec(
                    treatment=str(treatment),
                    outcome=str(outcome),
                    features=tuple(c.strip() for c in str(features).split(",") if c.strip()),
                    categorical=tuple(
                        c.strip() for c in (categorical or "").split(",") if c.strip()
                    ),
                    design="randomised" if design == "randomised" else "observational",
                    higher_outcome_is_better=outcome_polarity == "higher-is-better",
                    risk_is_low_outcome=risk_is_low_outcome,
                ),
                bins=bins,
            )
        except UnusableDataError as err:
            typer.echo("")
            typer.echo("this data cannot answer the question:")
            typer.echo("")
            typer.echo(f"  {err}")
            typer.echo("")
            raise typer.Exit(code=1) from err
        if design == "observational":
            from itx.data.user_csv import OBSERVATIONAL_WARNING

            caveat = OBSERVATIONAL_WARNING
    else:
        if dataset not in DATASETS:
            typer.echo(f"unknown dataset {dataset!r}; known: {', '.join(sorted(DATASETS))}")
            raise typer.Exit(code=1)
        data = DATASETS[dataset]()

    split = stratified_split(data, SEEDS[0] if seed is None else seed)
    say(f"{data.name}: one outcome model on {split.train.n_units:,} training rows")
    table = risk_deciles(
        split.train,
        split.test,
        bins=bins,
        seed=split.seed,
        tie_seed=TIE_SEED,
        n_resamples=resamples,
    )

    typer.echo("")
    if caveat:
        # Above the table rather than below it, and above rather than only in the docs,
        # because the table is what gets copied out and the caveat has to travel with it.
        typer.echo(f"READ THIS FIRST. {caveat}")
        typer.echo("")
    typer.echo(f"Risk bands on {table.dataset}, {table.n_test:,} test rows, band 1 riskiest")
    typer.echo("")
    typer.echo(table.to_markdown())
    typer.echo(table.summary())


@app.command("sensitivity")
def sensitivity(
    dataset: Annotated[str, typer.Option(help="Dataset key to test.")] = "hillstrom",
    estimator: Annotated[
        str, typer.Option(help="Which ranking defines the targeted group.")
    ] = "t-learner",
    seed: Annotated[
        int | None, typer.Option(help="Split seed; the first committed one if omitted.")
    ] = None,
    budget: Annotated[float, typer.Option(help="Share of the population targeted.")] = 0.2,
    control: Annotated[
        str | None,
        typer.Option(help="Covariate to use as the negative control; the hardest if omitted."),
    ] = None,
    resamples: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 500,
) -> None:
    """Price the unmeasured confounding this targeting result would survive.

    Three devices, answering different questions (PLAN.md section 4). The E-value prices a
    confounder's association with treatment and outcome; the Rosenbaum bound prices its
    effect on the odds of being treated; the negative control asks the pipeline about an
    outcome the treatment cannot have moved, and is the only one of the three that can fail.
    """
    from itx.bench.runner import DATASETS, ESTIMATORS, nuisances_for
    from itx.bench.seeds import SEEDS, TIE_SEED
    from itx.data.splits import stratified_split
    from itx.estimators.lightgbm_base import DEFAULT_CONFIG
    from itx.policy.rank_and_cut import rank_and_cut
    from itx.sensitivity import negative_control, targeting_e_value, targeting_rosenbaum

    if dataset not in DATASETS:
        typer.echo(f"unknown dataset {dataset!r}; known: {', '.join(sorted(DATASETS))}")
        raise typer.Exit(code=1)
    if estimator not in ESTIMATORS:
        typer.echo(f"unknown estimator {estimator!r}; known: {', '.join(sorted(ESTIMATORS))}")
        raise typer.Exit(code=1)

    say = _printer()
    chosen_seed = SEEDS[0] if seed is None else seed
    split = stratified_split(DATASETS[dataset](), chosen_seed)
    test = split.test

    say(f"{dataset}: fitting {estimator} on {split.train.n_units:,} training rows")
    fitted = ESTIMATORS[estimator](DEFAULT_CONFIG, chosen_seed)
    fitted.fit(split.train)
    scores = fitted.predict_uplift(test.features)

    say("fitting the propensity and outcome models the sensitivity work shares")
    nuisances = nuisances_for(split)
    targeted = rank_and_cut(scores, budget, seed=TIE_SEED)

    # The E-value prices the confounding that would explain away an *adjusted* estimate.
    # The ratio here is the crude treated-to-control contrast inside the targeted group,
    # which equals the adjusted one only when assignment was randomised. On a confounded
    # design it would be the E-value of the confounding the covariates already explain,
    # which is not the question (PLAN.md change 56).
    randomised = nuisances.propensity_fit.known
    e_value = None
    if randomised:
        say("E-value")
        e_value = targeting_e_value(
            test.outcome,
            test.treatment,
            scores,
            budget=budget,
            n_resamples=resamples,
            seed=chosen_seed,
            tie_seed=TIE_SEED,
        )
    else:
        say(
            "E-value not computed: assignment is not randomised, "
            "so the crude contrast is confounded"
        )
    say("Rosenbaum bound")
    bound = targeting_rosenbaum(
        nuisances.propensity,
        test.treatment,
        test.outcome,
        targeted,
        budget=budget,
        # On a randomised design the propensity is a constant and matching on it is
        # arbitrary pairing, so the prognostic score is the fallback key.
        prognostic=nuisances.mu0,
        seed=chosen_seed,
    )
    say("negative control")
    control_result = negative_control(
        split.train,
        test,
        column=control,
        seed=chosen_seed,
        n_resamples=resamples,
    )

    typer.echo("")
    typer.echo(
        f"Sensitivity on {dataset}, seed {chosen_seed}, targeting the top {budget:.0%} "
        f"by `{estimator}` on {test.n_units:,} test rows"
    )
    typer.echo("")
    typer.echo("| Device | Number | Reading |")
    typer.echo("|---|---|---|")
    if e_value is None:
        typer.echo(
            "| E-value | not computed | needs an adjusted estimate; on a design that is not "
            "randomised the crude contrast is confounded by the measured covariates |"
        )
    else:
        typer.echo(
            f"| E-value, point | {_maybe(e_value.point)} | "
            f"confounder association that moves the estimate to no effect |"
        )
        typer.echo(
            f"| E-value, interval | {_maybe(e_value.limit)} | "
            f"association that stops it excluding no effect |"
        )
    typer.echo(
        f"| Rosenbaum Gamma | {_gamma(bound)} | "
        f"hidden bias the result survives, on {bound.matched_on} pairs |"
    )
    typer.echo(
        f"| Negative control | {control_result.effect.format(4)} | "
        f"effect on `{control_result.column}`, which must be zero |"
    )
    typer.echo("")
    paragraphs = [bound.summary(), control_result.summary()]
    if e_value is not None:
        paragraphs.insert(0, e_value.summary())
    for paragraph in paragraphs:
        typer.echo(paragraph)
        typer.echo("")


def _maybe(value: float) -> str:
    """An E-value, or a dash where the risk ratio was undefined."""
    return "-" if not math.isfinite(value) else f"{value:.2f}"


def _gamma(bound: RosenbaumBound) -> str:
    """A Gamma, marked when the search was censored or the pairs were too thin."""
    if not math.isfinite(bound.gamma):
        return "-"
    return f"above {bound.gamma:.2f}" if bound.censored else f"{bound.gamma:.2f}"


@app.command("selection")
def selection(
    dataset: Annotated[
        str, typer.Option(help="Dataset key to run both rules on.")
    ] = "hillstrom",
    seeds: Annotated[
        str | None,
        typer.Option(help="Comma-separated split seeds; the committed five if omitted."),
    ] = None,
    estimators: Annotated[
        str | None,
        typer.Option(help="Comma-separated estimator keys; the default set if omitted."),
    ] = None,
) -> None:
    """Select hyperparameters both ways and report where the two rules disagree.

    PLAN.md change 9 promised this comparison once realised policy value existed. The
    default rule scores candidates on the validation Qini; the alternative scores them on
    what the ranking would buy at the operating budget. Whether that changes the chosen
    configuration is the thing worth reporting, and it is measured here rather than argued.
    """
    from itx.bench.grid import (
        OPERATING_BUDGET,
        policy_value_rule,
        qini_rule,
        select_config,
    )
    from itx.bench.runner import DATASETS, DEFAULT_ESTIMATORS, ESTIMATORS, UNTUNED
    from itx.bench.seeds import SEEDS
    from itx.data.splits import stratified_split

    if dataset not in DATASETS:
        typer.echo(f"unknown dataset {dataset!r}; known: {', '.join(sorted(DATASETS))}")
        raise typer.Exit(code=1)

    say = _printer()
    seed_values = [int(s) for s in seeds.split(",")] if seeds else list(SEEDS)
    names = (
        [e.strip() for e in estimators.split(",")] if estimators else list(DEFAULT_ESTIMATORS)
    )
    names = [name for name in names if name not in UNTUNED]

    data = DATASETS[dataset]()
    say(f"{dataset}: {len(names)} estimators over {len(seed_values)} seeds, both rules")

    lines = [
        f"| Estimator | Seed | On Qini | On policy value at {OPERATING_BUDGET:.0%} | Same? |",
        "|---|---|---|---|---|",
    ]
    agreements = 0
    total = 0
    for seed in seed_values:
        split = stratified_split(data, seed)
        by_policy = policy_value_rule(split, seed=seed)
        for name in names:
            factory = ESTIMATORS[name]
            on_qini = select_config(factory, split, seed=seed, rule=qini_rule())
            on_policy = select_config(factory, split, seed=seed, rule=by_policy)
            same = on_qini.config == on_policy.config
            agreements += same
            total += 1
            lines.append(
                f"| `{name}` | {seed} | {_config_label(on_qini)} | "
                f"{_config_label(on_policy)} | {'yes' if same else 'no'} |"
            )
            say(f"  seed {seed} {name}: {'same' if same else 'differs'}")

    typer.echo("")
    typer.echo("\n".join(lines))
    typer.echo("")
    typer.echo(
        f"The two rules chose the same configuration in {agreements} of {total} cases "
        f"({agreements / total:.0%})."
    )


def _config_label(chosen: Selection) -> str:
    """The winning settings of a selection, as ``min_child_samples/num_leaves``."""
    return f"{chosen.config.min_child_samples}/{chosen.config.num_leaves}"


@app.command("allocate")
def allocate(
    dataset: Annotated[
        str, typer.Option(help="Dataset key. Only the fraud case has per-unit costs.")
    ] = "ieee-fraud",
    estimator: Annotated[str, typer.Option(help="Which model ranks the queue.")] = "t-learner",
    hours: Annotated[float, typer.Option(help="Analyst hours available.")] = 1000.0,
    seed: Annotated[
        int | None, typer.Option(help="Split seed; the first committed one if omitted.")
    ] = None,
) -> None:
    """Spend a budget of analyst time four ways and say what each one bought.

    The fraud worked case is the only one here where treating a unit has a per-unit cost, so
    it is the only one where the cost-aware knapsack of PLAN.md section 7 does anything that
    rank-and-cut does not. Both are run against the same budget, beside the risk ranking a
    fraud team would use today, beside random, and beside the oracle ranking that only a
    simulated effect makes available.
    """
    from itx.bench.allocate import compare_queues, to_markdown
    from itx.bench.runner import DATASETS, ESTIMATORS, nuisances_for
    from itx.bench.seeds import SEEDS, TIE_SEED
    from itx.data.ieee_fraud import COST_COLUMN
    from itx.data.splits import stratified_split
    from itx.estimators.baselines import OutcomeRanking
    from itx.estimators.lightgbm_base import DEFAULT_CONFIG

    if dataset not in DATASETS:
        typer.echo(f"unknown dataset {dataset!r}; known: {', '.join(sorted(DATASETS))}")
        raise typer.Exit(code=1)

    say = _printer()
    chosen_seed = SEEDS[0] if seed is None else seed
    data = DATASETS[dataset]()
    if COST_COLUMN not in data.feature_names:
        typer.echo(
            f"{dataset} has no {COST_COLUMN!r} column, so every review costs the same and "
            f"the knapsack reduces to rank-and-cut. Try --dataset ieee-fraud."
        )
        raise typer.Exit(code=1)

    split = stratified_split(data, chosen_seed)
    test = split.test
    costs = test.features[COST_COLUMN].to_numpy()

    say(f"{dataset}: fitting {estimator} on {split.train.n_units:,} training rows")
    uplift = ESTIMATORS[estimator](DEFAULT_CONFIG, chosen_seed)
    uplift.fit(split.train)
    predicted = uplift.predict_uplift(test.features)

    say("fitting the risk ranking a fraud team would use today")
    # A risk model fitted on the untreated rows. The dataset declares that its risk is a low
    # outcome (dollars retained), and the baseline reads that, so the highest score here is
    # the transaction a fraud queue would review first.
    risk = OutcomeRanking(DEFAULT_CONFIG, seed=chosen_seed, fit_on="control")
    risk.fit(split.train)
    risk_scores = risk.predict_uplift(test.features)

    say("pricing each queue against the same budget")
    truth = test.require_true_effect()
    rng = np.random.default_rng(chosen_seed)
    queues = compare_queues(
        {
            "uplift-knapsack": predicted,
            "uplift-rank-and-cut": predicted,
            "risk": risk_scores,
            "random": rng.normal(size=test.n_units),
            # The ceiling, and the reason the case is worth simulating. It ranks by the
            # effect the simulation wrote, so no fitted model can beat it, and the distance
            # to it separates two very different failures: a problem with no signal to find,
            # and a problem whose signal these features and this estimator cannot reach.
            "oracle": truth,
        },
        costs,
        hours,
        outcome=test.outcome,
        treatment=test.treatment,
        nuisances=nuisances_for(split),
        truth=truth,
        knapsack_for=("uplift-knapsack", "oracle"),
        # Only the uplift queues stop at a predicted effect of zero. A risk score's zero is
        # not a prediction of harm and a random draw's zero is nothing, so those spend the
        # whole budget, as a real risk queue does.
        harm_aware_for=("uplift-rank-and-cut",),
        seed=TIE_SEED,
    )

    typer.echo("")
    typer.echo(
        f"{dataset}, seed {chosen_seed}, {test.n_units:,} held-out transactions worth "
        f"${test.features['TransactionAmt'].sum():,.0f}. The `{estimator}` ranking is fitted "
        f"at the default configuration, not the tuned one in the benchmark table. The DR "
        f"estimate's interval is a bootstrap over test rows with each queue held fixed."
    )
    typer.echo("")
    typer.echo(to_markdown(queues, hours))


demo_app = typer.Typer(help="Build the static budget-slider demo.", no_args_is_help=True)
app.add_typer(demo_app, name="demo")


@demo_app.command("build")
def demo_build(
    datasets: Annotated[
        str | None,
        typer.Option(help="Comma-separated dataset keys; every finished result if omitted."),
    ] = None,
    results_dir: Annotated[
        Path, typer.Option(help="Where the per-seed JSON lives.")
    ] = DEFAULT_RESULTS_DIR,
    out_dir: Annotated[
        Path, typer.Option(help="Where the demo's data files go.")
    ] = DEFAULT_DEMO_DIR,
    resamples: Annotated[
        int, typer.Option(help="Bootstrap resamples behind the interval band.")
    ] = DEMO_RESAMPLES,
) -> None:
    """Precompute the JSON the static page reads, from finished benchmark runs.

    The page has no backend (PLAN.md section 7), so every number the slider can show has to
    exist before anyone opens it. This refits the first seed of each dataset, because the
    results file stores metrics rather than per-unit scores and the demo needs the ranking
    itself, then writes one file per dataset next to the page.
    """
    from itx.bench.grid import default_selection
    from itx.bench.runner import BENCHMARK_DATASETS, refit_seed
    from itx.bench.table import read_json
    from itx.demo.build import build_payload, write_payloads

    say = _printer()
    # By default the benchmark datasets plus the fraud worked case, and only the ones with
    # results. results/ also holds the synthetic generators and possibly a checkpoint, which do
    # not belong on the page.
    #
    # The fraud case is here deliberately rather than by default-of-everything. It is
    # semi-synthetic, and the page labels it as such wherever it appears, but it is the only
    # dataset whose units are dollars, and "this budget saves $1.61 a transaction" is legible
    # to a reader who bounces off "+0.0055 outcome per head". Leaving the project's most
    # striking result out of the page that exists to show the project's results was the wrong
    # call, and this is the correction.
    on_the_page = (*BENCHMARK_DATASETS, "ieee-fraud")
    present = sorted(
        path.stem
        for path in results_dir.glob("*.json")
        if not path.stem.endswith(".checkpoint")
    )
    wanted = (
        [name.strip() for name in datasets.split(",")]
        if datasets
        else [name for name in on_the_page if name in present] or present
    )
    if not wanted:
        typer.echo(f"no results in {results_dir}; run 'itx benchmark --all' first")
        raise typer.Exit(code=1)

    payloads = []
    for name in wanted:
        path = results_dir / f"{name}.json"
        if not path.is_file():
            typer.echo(f"no results at {path}; skipping {name}")
            continue
        rows = read_json(path)
        seed = rows[0].seed
        selections = {
            row.estimator: row.selection if row.selection is not None else default_selection()
            for row in rows
            if row.seed == seed
        }
        say(f"{name}: refitting seed {seed} for its rankings")
        refitted, split = refit_seed(name, seed, selections=selections)
        say(f"{name}: pricing {len(refitted)} rankings at every budget")
        payloads.append(build_payload(name, refitted, split, n_resamples=resamples, seed=seed))

    if not payloads:
        typer.echo("nothing to build")
        raise typer.Exit(code=1)

    for written in write_payloads(payloads, out_dir):
        typer.echo(f"demo data: {written}")


if __name__ == "__main__":  # pragma: no cover
    app()
