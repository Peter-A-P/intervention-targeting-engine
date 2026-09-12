# Plan: Intervention Targeting Engine

**Written:** 2026-09-06. **Status:** weeks 1 to 3 built (2026-09-11). Changes to this plan
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
| Criteo-UPLIFT v2.1 | 13.9M rows, 12 features | Ad campaign assignment | Visit, conversion | No | CC BY-NC-SA 4.0, downloaded through the official page. **Loader built, week 4**, with a committed 10% stratified subsample (change 23) |
| Lenta | 687,029 | SMS campaign | Store visit (`response_att`) | No | **No licence stated anywhere** (change 25). **Loader built, week 4**; downloaded from the bucket `scikit-uplift` reads |
| IHDP | 747 units, 100 replicates | Home visits (semi-synthetic) | Cognitive score | Yes, simulated | The `ihdp_npci_1-100` benchmark archives the CEVAE and Dragonnet papers use, from fredjo.com. **Loader built, week 1** |
| ACIC 2016 | 4,802 units, 10 settings | Simulated | Simulated | Yes | CDLA-Sharing-1.0, from the causallib CSV mirror. **Loader built, week 3** (change 13) |

Rules for data in this repository:

- Nothing raw is committed. A `data/` loader downloads each set, verifies a SHA-256
  checksum, and caches under `data/raw/` (gitignored). Checksums are committed.
- Every dataset gets a card in `docs/data/`: source, licence, treatment definition,
  known quirks, the exact split seeds.
- Criteo is the only one that needs care on a laptop: Polars lazy scan, categorical
  encoding, and a fixed 10% stratified subsample for the benchmark, the bootstrap and the
  demo, with the full 13.9M used once for the headline fit. The constraint is wall-clock
  time, not memory; the 8 GB budget written here at plan time was wrong (change 23).
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
                causal_forest, baselines (random and outcome ranking), propensity,
                sklearn_bridge (change 17)
  policy/       rank_and_cut, cost_aware, policy_value (ipw, dr)
  metrics/      qini, auuc, uplift_at_k, calibration, bootstrap, ground_truth
  sensitivity/  rosenbaum, evalue, negative_control
  bench/        runner, results table renderer, seeds, grid (change 8)
  cli.py        typer CLI: itx data pull, itx benchmark, itx report, itx figures,
                itx demo build (change 20)
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
| 3 | Sep 21 - 27 | DR and R learners via EconML/CausalML wrappers; IHDP and ACIC loaders; PEHE and ATE error; calibration plot | **Done 2026-09-11.** Five estimators with ground-truth metrics on IHDP and ACIC; calibration slope, calibration error and the decile plot; the CausalML build risk did not materialise, and the week instead turned up a real defect in the propensity model (change 14). 302 tests |
| 4 | Sep 28 - Oct 4 | Criteo and Lenta loaders; Polars pipeline and 10% subsample; full benchmark runner with seeds; results table renderer into README | **Done 2026-09-12.** All five tables measured. Both loaders, both cards, `itx benchmark --all`, and a covariate-balance leak detector that caught two contaminated Lenta columns. The week's real cost was discovering the tuning protocol did not scale (change 30); its results were Criteo showing the outcome-ranking trap is conditional (change 32) and Lenta showing nothing at all (change 33). Not overnight, because this machine reboots itself nightly (change 34): the Criteo and Lenta runs are 50 minutes and 4h41m and both must start in the morning. 398 tests |
| 5 | Oct 5 - 11 | Policy module: rank-and-cut, cost-aware knapsack, IPW and DR policy value; the outcome-ranking trap demonstrated on every dataset; `itx diagnose`, the risk-decile table of change 32 | Policy value table with both baselines |
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

**12. EconML and CausalML installed without a build** (week 3). Recorded because it was
listed as the week's main risk and it did not happen: both ship Python 3.13 wheels for
Windows, and `uv sync --extra learners` needed no compiler. The pins moved to 0.17 in week
1 for numpy 2 compatibility, which turned out to be the same release that fixed this.

**13. ACIC 2016 is ten settings, not seventy-seven** (week 3). Section 3 said 77 settings
via the `aciccomp2016` R package or mirrored CSVs. The loader uses the causallib mirror,
which is plain CSV, maintained, carries the organisers' licence and citation, and needs no R
toolchain. It holds ten simulation settings rather than the competition's full set. Ten is
enough to average across problem types and to give the benchmark a large, strongly
heterogeneous, badly confounded ground-truth dataset next to IHDP's small nearly homogeneous
one, which is the job it was in the plan to do.

**14. The propensity model predicts training rows out of fold** (week 3). Not in the plan at
all, and the most important thing the week produced. A propensity model asked to predict the
rows it was fitted on has partly memorised them, so `t - e(x)` stops being a residual with
conditional mean zero, and the R-learner is built on exactly that residual. On ACIC its PEHE
is 23.95 with an in-sample propensity and 1.20 with an out-of-fold one, against a true effect
whose standard deviation is 3.85, on every seed.

What makes it worth a plan entry rather than a commit message is how it was found. Every
diagnostic that a careful person would check said the two fits were nearly identical: the
share of units against the clipping bound moved from 48.2% to 47.8%, the median treatment
residual from 0.014 to 0.025. The failure was visible only against a known truth. Section 1
says the ground-truth datasets are there because they are the only way to show an estimator
is correct rather than self-consistent; this is that claim collecting on itself.

**15. A fifth synthetic generator, `confounded`** (week 3). The other four randomise
assignment, which makes them incapable of testing anything built for confounding: a broken
propensity model looks exactly like a working one when there is nothing to find. The new
generator drives treatment from the same covariates that drive the outcome, hard enough that
the naive difference in arm means has the wrong sign, and records the true assignment
probability so an estimated propensity can be scored against the right answer. It is what
the cross-fitting finding above is regression-tested on without a download.

**16. The grid gains a `min_child_samples` of 200** (week 3). Eight candidates now rather
than six. The DR-learner's final stage regresses on a pseudo-outcome that carries the
inverse-propensity correction's variance as well as the outcome's, and on 2,400 training
rows its PEHE was 1.28 at a leaf size of 20, worse than predicting a constant, against 0.40
at 200. The old ceiling of 60 could not reach it. Widened for every estimator rather than
for the one that needed it, so the grid stays identical across the table.

**17. `itx/estimators/sklearn_bridge.py`** (week 3). Section 5's rule is that the base
learner is LightGBM everywhere. EconML and CausalML build their own models by cloning an
estimator they are handed, and a bare `LGBMRegressor` would lose the categorical column
declaration, because `categorical_feature` is a `fit` argument neither library knows to
pass. Hillstrom's `zip_code` and `channel` would then be ordered quantities for the DR and R
learners and categories for the others, and the estimator column would be carrying an
encoding difference. Two thin scikit-learn estimators close that.

**18. Calibration bins are capped by the smaller arm** (week 3). Section 2 asks for a decile
plot. Ten deciles of IHDP's 150-row test split leaves under three treated units in each, and
most deciles then hold no treated unit at all and can report nothing, which produced a table
of NaN calibration slopes sitting beside finite-looking intervals. The count is capped so a
bin holds at least ten units of each arm: ten bins on Hillstrom and ACIC, two on IHDP. Two
bins is a weak calibration estimate and is reported as such rather than dressed up as ten.

**19. The grid is filtered by training size before selection** (week 3). Adding the leaf size
of 200 in change 16 immediately cost the S-learner on IHDP: one seed in five selected it, 200
leaves room for two leaves in 448 training rows, and the near-degenerate fit moved the
five-seed PEHE from 0.57 to 1.28. A configuration that cannot fit a model is not a
hyperparameter choice. Candidates whose leaf size exceeds a quarter of the training rows are
dropped before selection runs. The grid stays identical across estimators; what rules a
candidate out is the dataset.

**20. Two new commands, `itx report` and `itx figures`** (week 3). Not in section 5, and both
exist because the Hillstrom run reached twenty minutes with five estimators and an
eight-candidate grid. `report` redraws a results table from the committed per-seed JSON
without fitting anything, which is what that file was written for. `figures` redraws the
figures by refitting only the first seed at the configuration the run recorded, about a
minute rather than twenty. Neither invents a number: `report` reads them and `figures` needs
per-unit scores, which are not stored because they are large and only the pictures use them.

**21. Figures pin their matplotlib style** (week 3). CausalML imports seaborn, and importing
seaborn rewrites matplotlib's global settings, so the same plotting code produced a white
figure before the R-learner existed and a grey one after. A figure whose appearance depends
on which estimators were imported is not reproducible. Every figure is now drawn inside an
explicit style context and saved on an explicit white background.

**22. CI is three jobs, and Hillstrom is not on every push** (week 3). Section 4 says a CI job
runs the Hillstrom and IHDP benchmarks on every push. Hillstrom now takes around twenty
minutes, most of it in the DR and R learners' cross-fitting, so it moved to a weekly schedule
and manual dispatch, where it also asserts that the committed table still matches a fresh run.
IHDP and ACIC stay on every push, along with the tests that need real data.

**23. The Criteo memory budget was wrong, and the reason for subsampling changed** (week 4).
Section 3 said "Memory budget: 8 GB", written before any code existed and repeated in a
week 3 status note without being checked. The machine this is built on has 63.7 GB and 12
cores, and Criteo is smaller than the line implies: 13,979,592 rows by 16 columns is about
1.7 GB held as float64, and the 297 MB download is compressed text. Memory was never going
to be the binding constraint. Time is: five estimators, an eight-candidate grid and five
seeds over 8.4M training rows, with the DR and R learners cross-fitting on top, is hours
where Hillstrom's 25,615 rows is twenty minutes. The 10% subsample survives with its
justification restated, which is the honest outcome: the decision was right and the stated
reason for it was not.

**24. `n_jobs` raised from 4 to 8** (week 4). The 4 was a guess made on the first day and
never revisited, on a machine with 12 cores. Fit timings recorded before this change are not
comparable with those after it, which is the whole cost, and `deterministic` plus
`force_row_wise` are already on precisely so that the numbers do not move with the thread
count.

**25. Lenta has no licence, and it ships anyway with that said out loud** (week 4). Section 3
said "check licence in the package". The check was done: there is no licence statement on the
publisher's bucket, in `sklift/datasets/datasets.py`, or on the `fetch_lenta` documentation
page. The loader ships because this repository downloads at run time and redistributes
nothing, which is what scikit-uplift does and what every other loader here does, but the
dataset card leads with the gap rather than burying it, and no reader is told their own use
is licensed. It is the only one of the five in that position. If this turns out to block the
repository going public in week 8, dropping Lenta costs one dataset and no finding.

**26. A covariate-balance diagnostic, `itx.metrics.balance`** (week 4). Not in section 5. It
exists because Lenta ships 194 undocumented columns and two of them, `response_sms` and
`response_viber`, are responses to the campaign rather than covariates. Nothing in the
documentation says so and the names do not settle it. The standardised mean difference
between arms does: 0.198 and 0.068 against a median of 0.011 over the other 192 columns, in a
trial where the worst ordinary column is 0.025. Both are dropped from the feature matrix. The
check is a module rather than a script that was run once, so it runs against every dataset,
and it is the same defect Criteo's `exposure` has in a more obvious form. A leak detector
that costs two means and two standard deviations is worth having permanently.

**27. Criteo and Lenta are separate dataset keys, and `criteo-full` is one of them** (week 4).
Section 5 implies one loader per dataset. Criteo has two entries, `criteo` for the committed
10% subsample and `criteo-full` for all 13.9M rows, rather than one loader with a fraction
flag, because a results table has to record which of the two produced it and a dataset name
is where that belongs. The subsample is cached as Parquet under the gitignored data
directory: it is reproducible from a committed seed and a committed fraction, so committing
it would only be committing data.

**28. The CI reproducibility check is `itx compare`, not `git diff`** (week 4). Change 22
added a step asserting that the committed Hillstrom results still match a fresh run, and
wrote it as `git diff --exit-code -- README.md results/hillstrom.json`. That was wrong and
would have failed on every scheduled run: the results file records `fit_seconds`, which is
wall-clock and cannot reproduce. It had not fired yet because the job is weekly and week 3
ended before the first Monday. The README diff stays, because the rendered table carries no
timings. The JSON comparison moved to a new command that ignores timings and reports which
metric moved and by how much, which is a better failure message than a diff anyway. The job
also became a matrix over Hillstrom, Lenta and the Criteo subsample.

**29. The DR-learner had to be told to accept missing values** (week 4). Lenta has missing
values in 150 of its 191 feature columns, and the loader leaves them missing on purpose:
LightGBM routes NaN down its own branch at every split, and an imputation rule chosen in a
loader would put one modelling decision into every estimator's input. Five of the six
estimators handle that without being asked. EconML's DR-learner does not, because EconML
validates its inputs with scikit-learn's finiteness check before any model sees them, so
LightGBM's NaN handling never gets a chance to run. It is a hard failure, not a degradation,
and it would have left a blank row in the Lenta table. ``allow_missing=True`` turns the check
off; EconML then warns, per fold and per prediction, that missingness can break causal
identification. The warning is right in general and does not apply to a randomised design,
where assignment is independent of the covariates and of their missingness pattern by
construction, so it is filtered at the call site with that argument written next to it. The
episode is itself a section 2 finding and belongs in `docs/estimators.md`: a wrapped
estimator inherits its library's input contract, not only its statistics.

**30. Hyperparameter selection is capped at 50,000 training rows** (week 4). Section 4 says
hyperparameters are tuned on validation with a small fixed grid. It does not say what the
candidates are fitted on, and the implementation used the whole training split, which turned
out not to scale. Selection is nine fits per estimator per seed, eight candidates and the
winner, so it is about 89% of a benchmark. On Hillstrom's 25,615 training rows that is twenty
minutes. On Lenta's 412,217 it is not: the run reached three fits of thirty in 1h25m and
extrapolated to between nineteen and thirty-two hours, and a parallel Criteo run passed four
hours without finishing. Both were stopped. A protocol that costs a day per dataset cannot be
rerun, and one that cannot be rerun is not a protocol.

Candidates are now fitted on at most 50,000 rows, stratified on arm crossed with outcome. The
winner is still fitted on every training row, so what gets reported is unchanged in kind: the
cap applies to choosing a configuration, not to the model whose numbers appear in the table.
The argument for capping selection specifically is that it is a coarse decision, ranking eight
settings of leaf size and tree width, and that ranking stabilises long before accuracy does.

The cap is deliberately set above every dataset whose full run is affordable: Hillstrom trains
on 25,615 rows, ACIC on 2,881, IHDP on 448. So their committed tables are untouched, they act
as a control, and only Lenta and Criteo are capped at all.

**The justification above is wrong, and measuring it is how that came out.** Forcing
comparable reductions on Hillstrom and ACIC changes nearly every selection: 1 of 25 identical
on ACIC at 5.8x, 5 of 25 on Hillstrom at 5.1x, 1 of 25 at 8.5x. The ranking of eight
configurations does not stabilise before accuracy; it is not stable at all.

What rescues the cap is the follow-up, which asks the question the first probe should have.
Taking the twenty Hillstrom disagreements, fitting both configurations on the full training
split and scoring both on test: mean change in Qini -0.00015, the capped choice better in 9
cases of 20 and worse in 11, worst case -0.00078, against a Hillstrom Qini of 0.0028 to 0.0041
with intervals about 0.004 wide. The cap changes which configuration is chosen and does not
change what the chosen configuration is worth, because the candidates are near-ties on test.
The instability is a property of this grid on this data, not a cost of capping.

So the cap ships on measured grounds rather than the reasoned ones it was proposed with, and
`docs/estimators.md` carries both the failed prediction and the numbers. It also leaves a
sharper question for week 8: if the eight candidates are near-ties, the per-seed selection
block under each README table is mostly noise, and the 89% of compute spent on tuning is
buying very little. That belongs in `docs/rejected.md`, not in a quiet change here.

**31. Lenta and Criteo cannot be reproduced in GitHub CI, and the workflow says so** (week 4).
Change 22 moved the long benchmarks to a weekly job, and earlier in week 4 that job became a
matrix over Hillstrom, Lenta and Criteo with a 180-minute timeout. That cannot work.
GitHub-hosted runners have four cores against this laptop's twelve and a hard six-hour job
limit, and Criteo alone ran for four hours here. Even after change 30 the margin is not there.
So the weekly reproduction check is Hillstrom only, which is the one that fits, and Lenta and
Criteo get a cheaper CI job that loads them, checks the row counts, the arm balance and that
no post-treatment column reached the features, without refitting anything. That job answers
"is the pipeline still correct", which is what CI can afford to ask; "are the numbers still the
numbers" is answered locally against the committed results with `itx compare`.

**32. Criteo says the outcome-ranking trap is conditional, and the README now says so**
(week 4). Section 1 frames the project around the intervention list not being the risk list.
On Criteo it is: outcome ranking scores a Qini of 0.0031 (0.0024, 0.0038) against the best
estimator's 0.0032 (0.0025, 0.0038), and buys 0.0576 at a 10% budget against 0.0586, on
279,592 held-out rows where the intervals are tight enough for a real gap to show. Uplift
modelling buys nothing there. On ACIC the same baseline is worse than random by a factor of
ten. Both are now reported side by side rather than the second being the headline and the
first an inconvenience.

The mechanism is measurable and makes the pair interpretable rather than contradictory.
Absolute uplift is baseline risk times the relative effect, so risk ranking approximates
uplift ranking whenever the spread in risk dominates the spread in the multiplier. Criteo's
risk spans about 1,500x across deciles against a multiplier moving about 2x; Hillstrom's
spans about 6x against a multiplier moving about 2.8x, and its outcome ranking lands about
two thirds of the way; ACIC's effect runs against risk entirely. The claim the project can
defend is therefore narrower than the one it started with and more useful: the two lists
differ by an amount nobody can guess in advance and anybody can measure cheaply, and the cost
of assuming they agree runs from zero to ten times worse than doing nothing.

That decile table should become a command, since it decides whether the rest of the
repository is worth running on a given problem and costs one outcome model. It belongs with
the policy work in week 5 rather than bolted onto the week 4 loaders, and it is listed in
section 6 there.

**33. Criteo and Lenta need their own test marker, not a degree of slowness** (week 4). CI
was red from the commit that added the two loaders. Two faults, one in the tests and one in
the workflow.

The test fault: `test_the_subsample_is_a_tenth_of_the_published_row_count` asserted
`n_units == round(13,979,592 * 0.1)`, which is 1,397,959. The sampler rounds inside each
stratum and sums, and all six occupied strata round down, so it returns 1,397,958. Twenty
lines above, `test_it_loads_the_committed_subsample` asserts that same 1,397,958, so the two
tests contradicted each other and no sampler could satisfy both. The loader is right:
1,397,958 is the count `results/criteo.json` and change 32 are built on. A sum of rounded
shares is not a rounded total, and the test now asserts the property its name claims, within
the half a row per stratum that stratifying costs.

The workflow fault, which is why the benchmark job reached that assertion at all: its test
step was an unscoped `pytest --run-slow`, which collects the Criteo and Lenta classes, and
`fetch` downloads on a cache miss rather than failing. So the job that change 31 says must
not touch the large datasets was pulling 435 MB and rescanning fourteen million rows on
every push, past its own `itx data pull` list. Marking those two classes `slow` was never
enough, because `slow` is what the benchmark job asks for by definition. They are now
`large`, behind `--run-large`, which only the scheduled Lenta and Criteo job passes. The
same audit found `TestAcic` loading three replicates past the pull list: 1 and 3 by name, and
2 through `load_acic_replicates(3)`, which yields 0 to 2. Replicate `r` is file `zymu_{r+1}`,
so `acic-zymu-2`, `-3` and `-4` are now in the list and every download in that job is still
checksum-verified up front.

One thing this does not fix, recorded here because it costs more than the bug did: both
jobs spend about forty minutes on a cache-cold `uv sync`, because causalml 0.17.0 publishes
no cp313 wheel and gets built from Cython sources. The comment in `pyproject.toml` files
this under Windows; it is Python 3.13, and it lands on Linux CI too. The "about a minute"
checks job took 44:14. Left open.

**33. Lenta is a null, and it is reported as one** (week 4, run 2026-09-12). Every Qini
interval on Lenta contains zero: five estimators, the outcome ranking and random targeting all
overlap. The one thread is the S-learner's realised uplift excluding zero at all three budgets,
0.0184, 0.0132 and 0.0118 against random targeting's flat 0.0074, consistently across five
seeds, while its Qini does not. That gap between a budgeted number and a whole-curve metric is
an argument for the week 5 policy work rather than a result on its own.

The cause is power, not method: a 0.75-point effect on a 10.3% base rate, 137,406 test rows,
about a quarter of them controls. Section 3 chose Lenta for size and messiness and got both;
what it did not check in advance was whether the effect was large enough to have detectable
heterogeneity, and a note to check that before adopting a dataset would have been worth having.
The result stays in the README at full size. A benchmark on which every dataset produces a
clean answer has selected its datasets, and Rule C in the plan repository asks for what did not
work.

The run took 4h41m against the nineteen to thirty-two hours it was heading for before change
30, of which 75 minutes was final fits and the rest the capped grid search.

**34. This machine force-reboots nightly at 23:30** (week 4). Not a plan change, a fact the
plan has to live with. `shutdown.exe` is invoked by `NT AUTHORITY\SYSTEM` at 23:30:01 every
night and the machine is down by 23:50; confirmed on 9, 10 and 11 September. The first Lenta
attempt was killed by it about four hours in and lost everything, because the benchmark writes
its results file only after the last fit. "Runs overnight" is therefore not available as a
strategy here, and the week 4 acceptance criterion in section 6 quietly assumed it was. Two
consequences: long runs start in the morning, and the runner needs to checkpoint so that a
killed run costs one fit rather than all of them. The second is change 35.

**35. The benchmark checkpoints, and resuming is opt-in** (week 4). Not in section 5. Every
finished row is written to `results/<dataset>.checkpoint.json` as it completes, and
`itx benchmark --resume` reuses what is there instead of refitting it. The file is deleted
when the run finishes, so a checkpoint on disk always means an interrupted run.

Resuming is a flag rather than automatic behaviour on purpose. A checkpoint written by an
older version of this package is indistinguishable from one written by the current version,
and silently mixing the two would produce a results table whose rows came from two different
builds. The flag puts that judgement with the person who knows whether anything changed.

Verified by doing it: a six-fit run killed after four fits, then resumed, finishes in 4.0s
instead of 9.4s and `itx compare` reports the result identical to an uninterrupted run at zero
tolerance. One limitation is real and is announced rather than worked around: a row read back
from a checkpoint carries its metrics but not its per-unit scores, which are large and are not
serialised, so a resumed run writes its table and skips its figures and says to run
`itx figures`, which refits a single seed.
**36. The reported policy number is the gain against treating nobody, not the level** (week
5). Section 4 asks for "expected outcome under the policy", estimated by IPW and by a doubly
robust estimator. Both estimators are here and both report that level on request, but what
the results table carries is the difference between the policy's value and the value of
treating nobody.

The reason is variance, not taste. The level is dominated by the baseline outcome rate,
which every policy on a dataset shares: on Hillstrom it is around 0.15 while the differences
between policies are around 0.01, so bootstrap intervals on the levels overlap almost
completely even where the differences between them are firm. In the difference, every unit
the policy leaves alone cancels exactly and what is left is an estimate over the targeted set
only. It is the same information with readable intervals, it is on the scale of outcome per
head of population, and at a budget of 100% it is the average treatment effect, which is what
the tests assert.

It also sits next to `uplift@k` without duplicating it, and the difference between the two is
worth knowing. `uplift@k` compares the arms inside the targeted prefix and divides by the arm
counts in that prefix. The IPW gain divides by the arm probabilities in the population. Under
a randomised design they agree only when the prefix happens to carry the population's treated
share; the first is what a practitioner can compute without a propensity, the second is what
is unbiased for the quantity that gets deployed.

**37. The policy-value nuisance models are fitted once per split, not once per row** (week
5). Not in section 5's package diagram, which is silent on where they live. The propensity
and the two outcome models are properties of the split rather than of the estimator being
scored, so `itx.bench.runner.nuisances_for` fits them once per seed and hands the same arrays
to every row including both baselines. Fitting them inside each row would score each estimator
against a different referee, and an estimator whose nuisance models happened to be optimistic
would be rewarded for it. They are fitted on train and applied to test, so no cross-fitting
arises: the rows being scored were never seen by the models scoring them.

**38. The propensity diagnostics could not report clipping, and on IHDP that mattered** (week
5). A defect, found by reading a diagnostic line that said something impossible.
`PropensityModel.predict` clips on the way out, so the policy-value code built its
`PropensityFit` from an already-clipped array and `n_clipped` counted the units whose clipped
value differed from their clipped value: always zero. The overlap problem was invisible in
the one place it had to be visible. `predict_unclipped` now exists and the fit is built from
it.

What it was hiding is the explanation of an entire table. IHDP's treatment assignment is
observational and its propensity is estimated, and on the test split the raw values run down
to 0.0007 with 17.3% of rows against the 0.01 bound. Those rows carry inverse weights of 100
each, which is why IHDP's IPW policy gains have intervals spanning a factor of twenty while
its DR gains do not. The number that explains the table was one line away from being printed
and was being printed as a zero.

**39. The random-targeting reference draws one set of rankings for every metric** (week 5).
`random_ranking_references` replaces a loop that called the single-metric version once per
metric. Numerically identical, because each of those calls rebuilt its generator from the
same seed and therefore drew the same rankings; the results file is unchanged by it. What
changes is that a thirteen-metric table stops drawing and sorting 2,600 rankings to look at
200.

**40. `itx diagnose` lives in `itx/metrics/risk_deciles.py`** (week 5). Section 5's package
diagram has no home for it. It sits with the metrics rather than in the policy package
because it measures a property of a dataset rather than allocating a budget, and
`metrics/balance.py`, the covariate-balance leak detector, is already there on the same
argument. It is what change 32 asked for: one outcome model, fitted on the control rows
because that is what a churn or fraud score actually is, ten bands of predicted risk, and
what the intervention did in each.

Three summary numbers come out of it: the spread in baseline risk across bands, the spread
in the relative effect, and the rank correlation between a band's risk and its uplift. The
verdict reads the third. Every one of them carries a bootstrap interval drawn from the same
resamples as the bands, and the two confident verdicts are gated on that interval rather than
on the point estimate. That gate is not decoration: on a synthetic population whose effect
runs against risk at a twentieth of full strength, the point estimate happily reports the
right direction and the interval covers zero, and a diagnostic that announced a direction
from ten noisy bands would be committing the exact error the rest of this repository exists
to demonstrate. There is a test for it.
