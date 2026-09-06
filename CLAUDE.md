# Working notes for Claude Code

This repository is the Intervention Targeting Engine: an uplift-modelling and
budget-constrained targeting toolkit with honest evaluation. The build plan is
[PLAN.md](PLAN.md).

## Read first

- [README.md](README.md): what this is and the current result table.
- [PLAN.md](PLAN.md): scope, data, evaluation protocol, package design, week-by-week
  schedule. Do not deviate from it silently; if something turns out wrong, change the plan
  in the same commit as the code and say why.

## Engineering standard

- Python 3.13, managed with `uv`. Typed throughout; `mypy --strict` and `ruff` clean in CI.
- Tests that fail meaningfully: every estimator recovers a known effect on synthetic data,
  every metric has an analytic check.
- `pyproject.toml` with pinned major versions and a comment saying why for each pin.
- Docs ship in the same commit as the change.
- Never commit raw data, credentials, or `.env`. Loaders download and verify checksums.

## Rules specific to this repository

- **Every metric carries a confidence interval.** A bare Qini is a bug.
- **Every ranking is shown next to the random and outcome-ranking baselines.** The
  outcome-ranking trap is the point, not an afterthought.
- **Base learner is LightGBM everywhere** so estimator differences are estimator
  differences.
- **Polars for data**, pandas only at library boundaries.
- **The fraud worked case is semi-synthetic and declared so** in its first sentence.
- **Plain punctuation** in everything written here: no em-dashes or other typographic
  dashes, straight quotes only.

## What goes in the README

The README opens with the one-liner, the results table, and the honest limitation, before
any installation instructions. The benchmark command regenerates the table; do not
hand-edit it. `docs/rejected.md` records one approach tried and rejected, with evidence.
