# Plan: Intervention Targeting Engine

**Written:** 2026-09-06. **Status:** weeks 1 and 2 built (2026-09-10). Changes to this plan
since it was written are listed in section 11, each with the reason and the week it
happened.
**Build window:** 8 weeks, 2026-09-07 to 2026-11-01, evenings and weekends, alongside the
setup of the drift runs in the release-gate repository during September.

---

## 1. What this produces

A typed, installable Python package that tells an organisation which customers, cases or
patients to spend a limited intervention budget on: only those whose outcome the
intervention actually changes. Retention offers, outreach, fraud review, clinical
follow-up: a large share of every such budget goes to people who would have behaved the
same way regardless, and this finds them.

The numbers a stranger can check, in one results table:

| Measure | Datasets | Why it is there |
|---|---|---|
| Qini coefficient and normalised AUUC, 95% bootstrap CI | All five | The standard uplift ranking metrics, never reported bare |
| Uplift at 10%, 20%, 30% of the population targeted | All five | What a budget holder actually asks |
| Realised policy value at a budget, vs random targeting and vs outcome-ranking | All five | The Qini-gaming trap made visible: outcome-ranking is the naive baseline that fools the curve |
| PEHE and ATE error | IHDP, ACIC | The only way to show an estimator is correct rather than self-consistent |
| Rosenbaum bound Gamma and E-value at which the targeting conclusion breaks | Hillstrom, Criteo, Lenta | Unmeasured confounding quantified, not assumed away |

Seven estimators, five datasets, five seeds each, one table. The definition of done requires
five estimators plus Dragonnet; the causal forest is the stretch.

## 2. Scope and boundaries

In scope:

- One common `UpliftEstimator` protocol: `fit(data)`, `predict_uplift(X)`,
  `policy(X, budget)`; every estimator implements it, every metric consumes it. `fit` takes
  an `UpliftDataset` rather than three loose arguments (change 1, section 11).
- Meta-learners S, T, X, DR and R on LightGBM base learners, plus a causal forest
  (EconML `CausalForestDML`). Own thin implementations of S, T and X so the mechanics are
  visible; EconML and CausalML wrapped for DR, R and the forest.
- **Dragonnet in PyTorch** as a neural CATE estimator behind the same protocol: the
  portfolio's example of a neural network on a classical prediction problem. Compared on
  the ground-truth sets like every other estimator; the write-up says where it earns its
  complexity and where the tree-based learners win.
- Evaluation: Qini, AUUC, uplift at k, policy value with an inverse-propensity and a
  doubly robust estimator on held-out data, bootstrap intervals, random and
  outcome-ranking baselines.
- Budget-constrained policy: rank-and-cut with a uniform cost, then a cost-aware variant
  where treatment cost varies by unit (knapsack by uplift per dollar).
- Sensitivity analysis: Rosenbaum bounds for the matched estimate and E-values for the
  targeted-group effect, plus a negative-control outcome test where the dataset allows.
- Calibration of predicted uplift: decile plot of predicted vs realised uplift.
- One worked case: fraud-review capacity allocation on public data, with a simulated
  review intervention, declared as semi-synthetic.
- A static budget-slider demo: precomputed rankings, JavaScript slider, the intervention
  list re-ranks live, hosted on Azure Static Web Apps at targeting.peterparker.ca.

Out of scope, on purpose:

- Instrumental variables, regression discontinuity, difference in differences. Different
  identification strategies, different project.
- Other deep-learning CATE models (TARNet, CEVAE). Dragonnet is in; the rest are noted
  in the "where each estimator breaks" write-up as comparisons from the literature.
- Continuous or multi-valued treatments.
- Online or bandit-style allocation.
- A server. The demo is static by design (CA$25 budget, Rule B satisfied by a URL).

## 3. Data

| Dataset | Size | Treatment | Outcome | Ground truth | Access and terms |
|---|---|---|---|---|---|
| Hillstrom MineThatData | ~64k | Email campaign (3 arms; use womens-email vs none) | Visit, conversion, spend | No | Public download; no formal licence, attributed to MineThatData. **Loader built, week 1** |
| Criteo-UPLIFT v2 | 13.9M rows, 12 features | Ad exposure | Visit, conversion | No | CC BY-NC-SA 4.0, downloaded through the official page. Downloaded and checksummed week 1; loader in week 4 |
| Lenta | ~687k | SMS campaign | Purchase | No | Ships with `scikit-uplift`; check licence in the package |
| IHDP | 747 units, 100 replicates | Home visits (semi-synthetic) | Cognitive score | Yes, simulated | The `ihdp_npci_1-100` benchmark archives the CEVAE and Dragonnet papers use, from fredjo.com. **Loader built, week 1** |
| ACIC 2016 | 4,802 units, 77 settings | Simulated | Simulated | Yes | Public via the `aciccomp2016` R package or mirrored CSVs |

Rules for data in this repository:

- Nothing raw is committed. A `data/` loader downloads each set, verifies a SHA-256
  checksum, and caches under `data/raw/` (gitignored). Checksums are committed.
- Every dataset gets a card in `docs/data/`: source, licence, treatment definition,
  known quirks, the exact split seeds.
- Criteo is the only one that needs care on a laptop: Polars lazy scan, categorical
  encoding, and a fixed 10% stratified subsample for the bootstrap and the demo, with the
  full 13.9M used once for the headline fit. Memory budget: 8 GB.
- The fraud worked case uses IEEE-CIS Fraud Detection features with a simulated "sent to
  manual review" treatment whose effect is generated from a known function of the
  features. The README says so in the first sentence of that section. Its purpose is to
  show the allocation logic on a fraud-shaped problem, not to claim a real effect.

## 4. Evaluation protocol

- Fixed stratified splits per dataset (train 60, validation 20, test 20), five seeds,
  seeds committed.
- All metrics computed on the test split only. Hyperparameters tuned on validation with a
  small fixed grid; the grid is committed and identical across estimators.
- Bootstrap: 1,000 resamples of the test split for every interval on the small sets; 200
  resamples on the Criteo subsample. Percentile intervals.
- Random-targeting baseline: the mean of 200 random rankings. Outcome-ranking baseline:
  rank by a plain outcome model's predicted probability.
- Policy value at budget b: expected outcome under "treat the top b% by predicted
  uplift" estimated with IPW and with a doubly robust estimator on the test split; both
  reported.
- Ground-truth sets: PEHE (root mean squared error of individual effects) and absolute ATE
  error, averaged over replicates with intervals across replicates.
- Sensitivity: Rosenbaum bounds on the matched pairs of the targeted group; E-value for
  the risk ratio of the targeted group's effect; report the Gamma and E-value at which the
  targeting decision would flip.
- Everything runs from `uv run itx benchmark --all` end to end and regenerates the results
  table in the README. A CI job runs the Hillstrom and IHDP benchmarks on every push;
  Criteo runs nightly or on demand.

## 5. Package design

```
src/itx/        (src layout, change 2 in section 11)
  data/         loaders, checksums, splits, dataset cards
  estimators/   protocol, s_learner, t_learner, x_learner, dr_learner, r_learner,
                causal_forest, baselines (random and outcome ranking), propensity
  policy/       rank_and_cut, cost_aware, policy_value (ipw, dr)
  metrics/      qini, auuc, uplift_at_k, calibration, bootstrap
  sensitivity/  rosenbaum, evalue, negative_control
  bench/        runner, results table renderer, seeds, grid (change 8)
  cli.py        typer CLI: itx data pull, itx benchmark, itx demo build
demo/           static site: precomputed rankings as JSON, slider, table
docs/
  data/         one card per dataset
  estimators.md where each estimator breaks
  rejected.md   Rule C
tests/          unit tests with synthetic data of known effect for every estimator and metric
```

Base learner is LightGBM everywhere so estimator differences are estimator differences,
not learner differences. Polars for data handling, pandas only at library boundaries.

Tests that matter: every estimator recovers a known constant effect and a known
heterogeneous effect on synthetic data within tolerance; Qini of a perfect ranking equals
the analytic maximum; policy value of random targeting equals the ATE within tolerance;
bootstrap intervals cover the truth at the nominal rate on synthetic data.

## 6. Week by week

| Week | Dates | Build | Done when |
|---|---|---|---|
| 1 | Sep 7 - 13 | Repo scaffold (`uv`, `ruff`, `mypy --strict`, `pytest`, CI); data loaders with checksums for Hillstrom and IHDP; `UpliftEstimator` protocol; S-learner end to end on Hillstrom | **Done 2026-09-10.** Ruff, `mypy --strict` and 161 tests green (152 of them needing no download); Qini curves plotted for both datasets; results table generated into the README; ahead of plan: PEHE and ATE error (week 3) landed early because the IHDP loader is untestable without them |
| 2 | Sep 14 - 20 | T and X learners; metrics module (Qini, AUUC, uplift at k, bootstrap); random and outcome-ranking baselines; synthetic-data tests | **Done 2026-09-10.** Three estimators with intervals on Hillstrom and IHDP, both baselines beaten on both; the metrics module and baselines had already landed in week 1, so the week also bought the validation grid from section 4 and a controlled demonstration of which meta-learner wins when. 212 tests |
| 3 | Sep 21 - 27 | DR and R learners via EconML/CausalML wrappers; IHDP and ACIC loaders; PEHE and ATE error; calibration plot | Ground-truth metrics for five estimators |
| 4 | Sep 28 - Oct 4 | Criteo and Lenta loaders; Polars pipeline and 10% subsample; full benchmark runner with seeds; results table renderer into README | `itx benchmark --all` runs end to end on a laptop overnight |
| 5 | Oct 5 - 11 | Policy module: rank-and-cut, cost-aware knapsack, IPW and DR policy value; the outcome-ranking trap demonstrated on every dataset | Policy value table with both baselines |
| 6 | Oct 12 - 18 | Sensitivity: Rosenbaum bounds, E-values, negative control; Dragonnet in PyTorch; `docs/estimators.md` "where each estimator breaks"; causal forest if on schedule | Sensitivity section with numbers; Dragonnet in the table; write-up drafted |
| 7 | Oct 19 - 25 | Fraud worked case (semi-synthetic, declared); static demo built from precomputed rankings; Azure Static Web Apps at targeting.peterparker.ca | Demo live, slider re-ranks |
| 8 | Oct 26 - Nov 1 | README to Rule A shape; `docs/rejected.md`; clean-environment rerun of the full benchmark; tag v0.1.0; flip the repository public | Definition of done all checked |

Slack: week 6's causal forest and week 7's cost-aware policy are the first things to
drop if behind. Neither is in the definition of done. Dragonnet stays: it is the
portfolio's neural-network placement and is cheap on these dataset sizes.

## 7. Demo

Static, because the budget is CA$25 and a static page is a URL a hiring manager can
click. Build step precomputes, per dataset, the ranked list with predicted uplift, cost
and cumulative policy value at every budget point; the page loads the JSON and a slider
moves the cutoff. Shows: who is in the treated set, the realised policy value at that
budget vs random, and the uplift-at-k curve with the cutoff marked. No backend, no
personal data (public datasets only, identifiers replaced by row numbers).

Hosting: Azure Static Web Apps free tier from the `demo/` build, custom domain
`targeting.peterparker.ca`. This is the portfolio's Azure hosting example; GitHub Pages
is the fallback if the free tier changes. DNS is a one-line change on the domain already
registered for Overload.

## 8. Risks

| Risk | Handling |
|---|---|
| Qini gamed by outcome ranking | Always show the outcome-ranking baseline and the realised policy value next to the curve. This is the point of section 4 |
| Criteo too big for the laptop | Lazy Polars, categorical encoding, 10% stratified subsample for everything except one headline fit. If the headline fit fails, report the subsample and say so |
| Sensitivity analysis becomes a textbook chapter | Two methods, one negative control, one paragraph each, a number each. Timebox to week 6 |
| Estimator list grows | Five plus one, fixed. Anything else goes in `docs/estimators.md` as literature, not code |
| Data terms | Criteo's research licence and the Hillstrom terms are read and recorded in the data card before download. If a set cannot be redistributed, the loader downloads from the source and nothing raw is committed |
| Fraud case misread as a real effect | Semi-synthetic, declared in the first sentence, effect function published |
| Drift-run setup in September eats week 1 to 3 | Both are planned for evenings; the drift work is about 25 hours total. If it slips, project 01 slips a week, not the drift runs |

## 9. Rule C candidates: what is expected not to work

Whichever produces the clearest evidence gets `docs/rejected.md`:

1. **Outcome-model ranking as a targeting policy.** Expected: good Qini on some sets,
   poor realised policy value. This is the classic trap and the most useful thing to show
   a reader.
2. **S-learner with a shared tree model.** Expected: regularisation swallows the
   treatment indicator and uplift collapses toward zero on Hillstrom.
3. **Class-transformation approach** (Jaskowski and Jaroszewicz). Expected: competitive
   on balanced designs, breaks under unequal treatment assignment.

## 10. Definition of done

Mirrors the portfolio's definition for this project:

- [ ] Five estimators benchmarked on five datasets, one results table in the README
- [ ] Dragonnet (PyTorch) in the same table, with a stated verdict on where it earned its complexity
- [ ] Qini and AUUC reported with bootstrap intervals, never as a bare number
- [ ] Policy value at budget reported against random and outcome-ranking baselines
- [ ] PEHE and ATE error on the ground-truth sets
- [ ] Sensitivity section with Rosenbaum bounds and E-values
- [ ] Live budget-slider demo at targeting.peterparker.ca
- [ ] README opens with the one-liner and the results table
- [ ] `docs/estimators.md` written: where each estimator breaks
- [ ] One rejected approach documented with evidence
- [ ] Clean-environment rerun reproduces the table
- [ ] Repository public, v0.1.0 tagged

---

## 11. Changes to this plan

The rule in `CLAUDE.md`: do not deviate silently, and change the plan in the same commit as
the code, with the reason. Each entry says what moved and why.

**1. The estimator protocol takes a dataset, not three arguments** (week 1). Section 2 said
`fit(X, treatment, outcome)`. It is `fit(data: UpliftDataset)`. Which feature columns hold
integer category codes has to travel with the matrix, and with three loose arguments every
estimator re-derives that from the dtypes and the LightGBM calls drift apart. The container
also validates lengths and the binary treatment at construction, so a mismatched array
fails at the loader rather than halfway through a benchmark.

**2. The package sits under `src/`** (week 1). Section 5 drew `itx/` at the repository
root. A src layout means the tests import the installed package rather than the working
directory, which is what the definition of done's "clean-environment rerun reproduces the
table" actually requires; with a flat layout a test can pass against files that were never
installed.

**3. The LightGBM boundary is numpy, not pandas** (week 1). Section 5 says Polars for data
and pandas only at library boundaries. The boundary turned out not to need pandas:
`DataFrame.to_pandas()` requires pyarrow, and passing a numeric numpy matrix with an
explicit list of categorical column positions is both lighter and safer, because category
codes cannot shift when a level is missing from a prediction batch. Pandas is no longer a
direct dependency. The rule is unchanged; nothing crosses into pandas at all.

**4. IHDP comes from the `ihdp_npci_1-100` archives** (week 1). Section 3 said "public via
the CEVAE and Dragonnet repositories". Those repositories carry per-replicate CSVs and the
set is incomplete: replicate 100 is missing from both. The two `.npz` archives at
fredjo.com hold all 100 replicates of all 747 units and are the files those papers actually
benchmark on, so the numbers here are comparable with the literature. Both are checksummed.

**5. `min_child_samples` moves into the tuning grid** (week 1). Not a change of plan so
much as the first thing the plan's own protocol caught. A single fixed value cannot serve a
747-unit dataset and a 64,000-unit one: at 100 the S-learner on IHDP never splits on the
treatment at all and returns exactly zero uplift with a PEHE identical to predicting
nothing, silently. The default is now LightGBM's own 20, the estimator base class warns on
a degenerate fit, and the parameter is in the week 2 validation grid. Evidence in
`docs/estimators.md`.

**6. Ground-truth metrics arrived in week 1 rather than week 3.** PEHE and absolute ATE
error are twenty lines and the IHDP loader cannot be checked without them. The rest of week
3 is unchanged.

**7. The reported random baseline is the 200-ranking average only.** Section 4 already
specified this; noting it because a single `random` estimator also exists and is
deliberately kept out of the default benchmark set. One random draw sitting next to a
fitted model in the same table invites the reader to read the gap between them as a result.

**8. The validation grid lives in `itx/bench/grid.py`** (week 2). Section 5's package
diagram has no module for it. Selection is part of the evaluation protocol rather than part
of an estimator: the grid has to be identical across estimators, and the thing that
guarantees that is one module the runner calls, not a convention each estimator follows.

**9. Selection is on the validation-split Qini** (week 2). Section 4 said hyperparameters
are tuned on validation with a small fixed grid but did not say on what. It is the Qini
coefficient, which needs the defence written into `itx/bench/grid.py`: selecting on a
metric this project calls a poor referee is defensible only because inside selection every
candidate is the same model class, so nothing can win by being a different kind of model
that games the curve. The comparison that the criticism applies to, between model classes,
is made on the test split against both baselines and never on this number. When realised
policy value exists in week 5, a second selection rule is added and the two are compared.

**10. A fourth synthetic generator, `complex_effect`** (week 2). Section 5 asks for
synthetic data with a known effect so that estimators can be shown to recover it. That is
what `heterogeneous_effect` does, and it turned out to be unfit for the *other* job the
synthetic sets were being asked to do. Its effect surface is simpler than its baseline,
which is exactly the arrangement an S-learner is built for, so comparing meta-learners on it
measures the generator. `complex_effect` reverses the arrangement, and the ordering of the
three estimators reverses with it. Both generators are in the benchmark and the comparison
is in `docs/estimators.md`. Recovery tests are not comparison benchmarks, and week 2 found
that out by making a prediction from the IHDP result that turned out to be wrong.

**11. `min_child_samples` is settled** (week 2, closing change 5). It is in the grid, with
candidates 5, 20 and 60 alongside `num_leaves` at 15 and 31: six configurations, identical
for every estimator, committed in `itx/bench/grid.py`.
