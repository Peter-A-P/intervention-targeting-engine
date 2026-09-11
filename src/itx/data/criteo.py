"""Criteo-UPLIFT v2.1: 13.9M randomised ad impressions, the only large set in the project.

Criteo randomised ad treatment with probability 0.85 and published the result under
CC BY-NC-SA 4.0. Twelve anonymised dense features, no names, no categoricals, and two
recorded outcomes: a site visit at 4.70% and a conversion at 0.29%. Card:
docs/data/criteo.md.

Four things about this file shape the loader, and each one is a decision rather than a
detail.

**``exposure`` is not a feature, and not the treatment either.** It records whether the
user was actually shown the ad. Zero of the 2,096,937 control rows have it set, because a
control user cannot be shown an ad that was never served, so it is a consequence of the
treatment and not a covariate. Putting it in the feature matrix would hand every estimator
the treatment indicator under a second name, and every uplift in the table would be an
artefact. It is dropped, and :data:`POST_TREATMENT` exists only so that the reason is
written down next to the code that drops it. Criteo published the column for the
one-sided-noncompliance question the dataset also supports, where the estimand is the
effect of *exposure* rather than of assignment. That is a different question from this
project's, and asking it needs an instrumental-variable argument this loader does not make.

**The arms are 85/15, not 50/50.** The control arm is 2.1M rows against the treated arm's
11.9M. Randomisation makes the assignment ignorable regardless, so the propensity is known
by design and recorded rather than estimated, but the imbalance is not free: every
control-arm quantity is estimated from a seventh of the data, and it is the control arm
that sets the precision of an uplift. It is also why the subsample below stratifies rather
than sampling rows at random.

**The outcome is rare.** A conversion happens 29 times in ten thousand rows. Under a plain
10% sample the smallest cell of interest, converting controls, would be a few hundred rows
and would move with the seed. Stratifying on treatment crossed with both outcomes holds all
eight cells at exactly a tenth of themselves, so the subsample has the same event structure
as the full file by construction rather than by luck.

**It is read lazily, subsampled once, and cached.** ``scan_csv`` reads the gzip in about
eleven seconds. The benchmark asks for the dataset once per seed per estimator per grid
candidate, which is hundreds of times, so the subsample is written to Parquet on first use
and read back in well under a second afterwards. The derived file lives beside the raw
download, under the same gitignored directory: it is reproducible from the committed seed
and the committed fraction, so committing it would only be committing data.

The subsample is 10% by PLAN.md section 3, and change 23 in that file records why: not
memory, which was the plan-time reason and was wrong, but wall-clock time. Five estimators
over an eight-candidate grid and five seeds on 8.4M training rows, with the DR and R
learners cross-fitting inside each fit, is hours. ``fraction=1.0`` is still available and is
what the headline single fit uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from itx.data.download import fetch, require_rows
from itx.data.registry import data_dir
from itx.types import IntArray, UpliftDataset

Outcome = Literal["visit", "conversion"]

#: The twelve anonymised dense features. Criteo publishes no meaning for any of them.
FEATURES: tuple[str, ...] = tuple(f"f{index}" for index in range(12))

#: Columns that exist in the file, are not features, and are not outcomes either.
#: ``exposure`` is post-treatment; see this module's docstring.
POST_TREATMENT: tuple[str, ...] = ("exposure",)

#: The design treatment probability, stated by Criteo and confirmed by the file: the
#: treated share is 0.85000013, which is 0.85 plus rounding on 13,979,592 rows.
KNOWN_PROPENSITY = 0.85

#: Rows in the published file. Checked on the full-file path so that a truncated or
#: re-released download is a failure rather than a quietly different result.
EXPECTED_ROWS = 13_979_592

#: The committed subsample. Both of these are protocol, not tuning knobs: changing either
#: changes every Criteo number in the results table.
SUBSAMPLE_FRACTION = 0.1
SUBSAMPLE_SEED = 20260928

#: Columns read on the first of the two lazy passes, to decide which rows to keep.
_STRATA_COLUMNS = ("treatment", "visit", "conversion")


def load_criteo(
    *,
    outcome: Outcome = "visit",
    fraction: float = SUBSAMPLE_FRACTION,
    seed: int = SUBSAMPLE_SEED,
    path: Path | None = None,
    cache: bool = True,
) -> UpliftDataset:
    """Load Criteo-UPLIFT v2.1 as a binary-treatment uplift dataset.

    Args:
        outcome: Which recorded outcome to model. ``visit`` is the headline one; at 0.29%
            ``conversion`` is rare enough that a Qini on it is noisy even at this size.
        fraction: Share of rows to keep, stratified on treatment crossed with both
            outcomes. 1.0 reads the whole file.
        seed: Seed for the subsample. Committed, and part of the protocol.
        path: Read from this file instead of the download cache. For tests.
        cache: Write and read the derived Parquet subsample. Off for tests, which want the
            sampling logic exercised rather than a cached answer.

    Returns:
        The dataset, with ``propensity`` set to the design value of 0.85.

    Raises:
        ValueError: If ``fraction`` is not in (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        msg = f"fraction must be in (0, 1], got {fraction}"
        raise ValueError(msg)

    source_path = path if path is not None else fetch("criteo", quiet=True)
    frame = _frame(
        source_path,
        fraction=fraction,
        seed=seed,
        cache=cache and path is None,
        verify=path is None,
    )
    suffix = outcome if fraction == 1.0 else f"{_pct(fraction)}pct-{outcome}"

    return UpliftDataset(
        name=f"criteo-{suffix}",
        features=frame.select(pl.col(name).cast(pl.Float64) for name in FEATURES),
        treatment=frame["treatment"].cast(pl.Int64).to_numpy(),
        outcome=frame[outcome].cast(pl.Float64).to_numpy(),
        propensity=np.full(frame.height, KNOWN_PROPENSITY),
    )


def _frame(
    source_path: Path, *, fraction: float, seed: int, cache: bool, verify: bool
) -> pl.DataFrame:
    """The rows to build the dataset from, read from the derived cache when there is one."""
    if fraction == 1.0:
        full = pl.scan_csv(source_path).collect()
        if verify:
            require_rows("criteo", full.height, EXPECTED_ROWS)
        return full

    cache_path = subsample_path(fraction, seed)
    if cache and cache_path.is_file():
        return pl.read_parquet(cache_path)

    frame = _subsample(source_path, fraction=fraction, seed=seed)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(cache_path)
    return frame


def _subsample(source_path: Path, *, fraction: float, seed: int) -> pl.DataFrame:
    """Draw the stratified subsample in two lazy passes over the compressed file.

    The first pass reads three integer columns and decides which rows survive; the second
    reads the whole width, but only for those rows. Peak memory is therefore the size of
    the subsample rather than the size of the file, which is the point of doing it this way
    rather than collecting once and slicing.

    Args:
        source_path: The compressed CSV.
        fraction: Share of each stratum to keep.
        seed: Seed for the sample.

    Returns:
        The kept rows, every column, in file order.
    """
    keys = pl.scan_csv(source_path).select(_STRATA_COLUMNS).collect()
    keep = _stratified_indices(
        treatment=keys["treatment"].to_numpy(),
        visit=keys["visit"].to_numpy(),
        conversion=keys["conversion"].to_numpy(),
        fraction=fraction,
        seed=seed,
    )
    wanted = pl.LazyFrame({"__row": keep.astype(np.uint32)})
    return (
        pl.scan_csv(source_path)
        .with_row_index("__row")
        .join(wanted, on="__row", how="semi")
        .drop("__row")
        .collect()
    )


def _stratified_indices(
    *,
    treatment: IntArray,
    visit: IntArray,
    conversion: IntArray,
    fraction: float,
    seed: int,
) -> IntArray:
    """Row positions of a stratified sample, sorted ascending.

    Strata are treatment crossed with both outcomes, eight cells. Each cell contributes the
    same share, so the sample reproduces the file's event structure exactly rather than in
    expectation. The rarest cell, a converting control, holds about four thousand rows in
    the full file, which a plain sample would leave visibly seed-dependent.

    Args:
        treatment: Binary assignment, one per row.
        visit: Binary visit outcome, one per row.
        conversion: Binary conversion outcome, one per row.
        fraction: Share of each stratum to keep.
        seed: Seed for the permutation inside each stratum.

    Returns:
        Sorted row positions, so that the second pass reads the file in order.
    """
    strata = treatment * 4 + visit * 2 + conversion
    rng = np.random.default_rng(seed)
    kept: list[IntArray] = []
    for value in np.unique(strata):
        members = np.flatnonzero(strata == value)
        take = round(members.size * fraction)
        if take == 0:
            continue
        kept.append(rng.permutation(members)[:take])
    if not kept:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(kept))


def subsample_path(fraction: float = SUBSAMPLE_FRACTION, seed: int = SUBSAMPLE_SEED) -> Path:
    """Where the derived Parquet subsample is cached.

    Args:
        fraction: Share of rows the file holds.
        seed: Seed the sample was drawn with.

    Returns:
        The path, which may not exist yet. Under the gitignored data directory: the file is
        reproducible from a committed seed and a committed fraction, so committing it would
        only be committing data.
    """
    return data_dir() / "derived" / f"criteo-{_pct(fraction)}pct-seed{seed}.parquet"


def _pct(fraction: float) -> str:
    """Format a fraction as a percentage, for a file name or a dataset name."""
    return f"{fraction * 100:g}"
