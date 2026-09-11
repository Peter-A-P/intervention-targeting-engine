"""The ``itx`` command line: pull data, run the benchmark, build the demo.

Everything in the results table has to come out of one command a stranger can run
(PLAN.md section 4), so the CLI is the interface the project is judged on, not a
convenience wrapper around a notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from itx import __version__

if TYPE_CHECKING:
    from collections.abc import Callable

app = typer.Typer(
    name="itx",
    help="Intervention Targeting Engine: uplift modelling with honest evaluation.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Download and verify the raw datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")

DEFAULT_RESULTS_DIR = Path("results")
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
) -> None:
    """Run one dataset: fit, evaluate, then write the JSON, the table and the figures."""
    from itx.bench.plots import plot_calibration, plot_qini_curves
    from itx.bench.runner import DATASETS, resamples_for, run
    from itx.bench.table import (
        selected_configurations,
        to_markdown,
        update_markdown_file,
        write_json,
    )
    from itx.data.splits import stratified_split

    rows = run(
        dataset,
        estimators=estimator_keys,
        seeds=seed_values,
        n_resamples=resamples if resamples is not None else resamples_for(dataset),
        tune=tune,
        progress=_printer(),
    )
    table = to_markdown(rows)
    chosen = selected_configurations(rows)
    if chosen:
        note = "Selected on the validation split from the committed grid:"
        table = "\n".join([table, note, "", chosen])
    typer.echo("")
    typer.echo(table)

    json_path = results_dir / f"{dataset}.json"
    write_json(rows, json_path)
    typer.echo(f"per-seed results: {json_path}")

    if update_markdown_file(readme, dataset, table):
        typer.echo(f"results block updated: {readme}")

    if plot:
        first_seed = seed_values[0]
        split = stratified_split(DATASETS[dataset](), first_seed)
        seed_rows = [row for row in rows if row.seed == first_seed]
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
        read_json,
        selected_configurations,
        to_markdown,
        update_markdown_file,
    )

    path = results_dir / f"{dataset}.json"
    if not path.is_file():
        typer.echo(f"no results at {path}; run 'itx benchmark --dataset {dataset}' first")
        raise typer.Exit(code=1)

    rows = read_json(path)
    table = to_markdown(rows)
    chosen = selected_configurations(rows)
    if chosen:
        note = "Selected on the validation split from the committed grid:"
        table = "\n".join([table, note, "", chosen])
    typer.echo("")
    typer.echo(table)
    if update_markdown_file(readme, dataset, table):
        typer.echo(f"results block updated: {readme}")


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


@app.command("demo")
def demo() -> None:
    """Build the static budget-slider demo. Arrives in week 7 (PLAN.md section 6)."""
    typer.echo("not built yet: the demo is week 7. See PLAN.md section 7.")
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
