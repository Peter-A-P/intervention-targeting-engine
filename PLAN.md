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
  demo/         build: what the static page precomputes (change 52)
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
| 5 | Oct 5 - 11 | Policy module: rank-and-cut, cost-aware knapsack, IPW and DR policy value; the outcome-ranking trap demonstrated on every dataset; `itx diagnose`, the risk-decile table of change 32 | **Done 2026-09-13.** All five policy tables measured with both baselines, from one `itx benchmark --all` taking 10h19m. The trap is demonstrated on every dataset and turns out to change sign across them: the same baseline buys -0.20 on ACIC and +0.0055 on Criteo at a 10% budget. Also closed change 9's second selection rule, and the cost-aware knapsack, which is built and tested but not applied until the fraud case in week 7. Three findings the week did not set out to get: the policy value reverses the sign of the ACIC recommendation where `uplift@k` cannot see it (change 36 and the ground-truth check), it turns Lenta's null into a signal (change 42), and a known propensity is not sufficient to trust IPW (change 43). 484 tests |
| 6 | Oct 12 - 18 | Sensitivity: Rosenbaum bounds, E-values, negative control; Dragonnet in PyTorch; `docs/estimators.md` "where each estimator breaks"; causal forest if on schedule | **Done 2026-09-13 except one refit.** All three sensitivity devices built with `itx sensitivity`, reported on three datasets, and the section leads on the limitation that none of the three can fail there (changes 46, 47). Dragonnet built, registered and in four of five tables; Lenta's row is being refitted after change 51. The treated-share column of change 43 landed and reproduced week 5's hand decomposition to four decimals on Criteo. The second selection rule's cost is measured and is nothing (change 49). Two defects found and fixed: a run-ending exception on an undefined metric (change 50) and a dead Dragonnet fit reporting random targeting as a result (change 51). Causal forest not started. 637 tests |
| 7 | Oct 19 - 25 | Fraud worked case (semi-synthetic, declared); static demo built from precomputed rankings; Azure Static Web Apps at targeting.peterparker.ca | **Done 2026-09-14 except hosting.** Demo built and tested (change 52). Fraud case built on IEEE-CIS, which was already on the machine for project 09 (53), benchmarked with seven estimators over five seeds, and allocated: the risk queue a fraud team already runs beats every fitted uplift model at every budget tried, $169,398 against the S-learner's $156,549 at 1,000 analyst hours with the oracle at $305,052, because separating fraud from legitimate is worth a step of $74 and ordering within fraud a slope the estimators cannot resolve. The sign defect that made the benchmark row and the decile diagnostic report the opposite was found by running the diagnostic and fixed with a declaration on the dataset (54). Hosted the same afternoon at https://targeting.peterparker.ca: a second free-tier Static Web App beside the portfolio site's, published from this machine with the deployment token so the repository carries no workflow and no credential, one DNS-only CNAME at Cloudflare, certificate managed by Azure; `docs/deploy.md` records it. 698 tests |
| 8 | Oct 26 - Nov 1 | README to Rule A shape; `docs/rejected.md`; clean-environment rerun of the full benchmark; tag v0.1.0; flip the repository public | **Started 2026-09-14.** README already in Rule A shape. `docs/rejected.md` written on the tuning grid with a fourth measurement made for it (change 55). Clean-environment rerun done in a fresh clone and virtual environment: all six datasets reproduce under `itx compare` at zero tolerance, the first five at commit f89b91e and the fraud case at faadc2b after change 58 fixed its loader. Tag and flip to public wait on Peter |

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

- [x] Five estimators benchmarked on five datasets, one results table in the README (weeks
      2 to 5; five tables, one per dataset, all from `itx benchmark`)
- [x] Dragonnet (PyTorch) in the same table, with a stated verdict on where it earned its
      complexity (week 6; in all five tables since the Lenta refit of 2026-09-14, verdict in
      `docs/estimators.md`: three of five on Qini, last on Lenta and the fraud case)
- [x] Qini and AUUC reported with bootstrap intervals, never as a bare number (week 2;
      `summarise` has no path that renders a point without its interval)
- [x] Policy value at budget reported against random and outcome-ranking baselines (weeks 2
      and 5; IPW and DR at three budgets, both baselines in every table)
- [x] PEHE and ATE error on the ground-truth sets (week 2; IHDP, ACIC and the fraud case)
- [x] Sensitivity section with Rosenbaum bounds and E-values (week 6; three datasets of
      five, and none of the three is capable of failing it, which the section says)
- [x] Live budget-slider demo at targeting.peterparker.ca (week 7; built, tested, committed
      under `demo/` and live since 2026-09-14)
- [x] README opens with the one-liner and the results table (one-liner, status, headline
      finding, results, limitations, then installation)
- [x] `docs/estimators.md` written: where each estimator breaks (weeks 2 to 7)
- [x] One rejected approach documented with evidence (week 8, change 55: the tuning grid,
      four measurements)
- [x] Clean-environment rerun reproduces the table (week 8: fresh clone, fresh venv from
      the lockfile, raw data verified against the committed digests, every dataset
      re-benchmarked and compared with `itx compare` at zero tolerance. IHDP, ACIC,
      Hillstrom, Criteo and Lenta at commit f89b91e; the fraud case at faadc2b, because the
      rerun is what found the encoding defect in change 58 and the fraud table had to be
      remeasured before it could reproduce. On the fraud case all 40 rows match on every
      metric and every tuning selection, `fit_seconds` is the only field that differs
      anywhere, and README.md and both figures regenerated byte for byte)
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

**41. The second selection rule exists and is not the default** (week 5, closing change 9).
Change 9 promised that once realised policy value existed, `itx/bench/grid.py` would gain a
second selection rule and the two would be compared. `policy_value_rule` scores each
candidate on the doubly robust gain its ranking would buy at the operating budget, 20%, on
the validation split; `qini_rule` is the existing behaviour, made explicit. Every selection
now records which rule produced its number, and the results files carry it, because two
rules that do not measure the same thing cannot share an unlabelled column.

The default did not move, and that is a result rather than an omission. `itx selection`
fits both rules on a dataset and reports where they disagree; `docs/estimators.md` carries
the table. The argument for selecting on the decision rather than the curve is a real one
about bias, and it is made against a variance cost: the policy value at a single budget reads
one cutoff of the validation ranking, where the Qini integrates the whole of it, and on
validation splits this size the noise is the dominant term.

The nuisance models for the rule are fitted on the training split and applied to the
validation split, once per split rather than once per candidate, for the same reason as in
change 37: a candidate must not supply the models that score it. There is a test that
corrupting the test split changes nothing about what the rule chooses.

**42. Lenta's null was partly a property of the metric, and change 33's guess was right**
(week 5). Change 33 reported every Qini interval on Lenta containing zero, noted that the
S-learner's realised uplift excluded zero at all three budgets while its Qini did not, and
said the gap was "an argument for the week 5 policy work rather than a result on its own".
It is now a result.

With realised policy value in the table, at a budget covering 20% of the population all five
uplift models' doubly robust gains exclude zero: 0.0024, 0.0021, 0.0020, 0.0025 and 0.0024,
against random targeting's 0.0012 and an outcome ranking that still contains zero. Two of the
five clear zero by less than 1e-5 and the README says so rather than counting them as five
clean wins.

The mechanism is the one guessed at: a coefficient integrating the whole curve averages a
front-of-ranking advantage away against the flat tail where every method is identical, and a
budgeted number reads only the front. On an underpowered dataset that is the difference
between seeing an effect and not.

It does not make Lenta a win and the README does not claim one. The gains are about twice
random targeting's with heavily overlapping intervals, so "these rankings buy something" is
established and "these rankings beat random" is not. What the dataset now demonstrates is
about the instrument rather than about retail marketing, which is a better reason to keep it
than the one it was kept for.

Also visible here: every IPW interval on Lenta contains zero at every budget, including for
the methods the DR column separates. Lenta was randomised but does not publish its assignment
probability, so section 3's decision to leave `propensity` as None and estimate it has a
measurable cost, and the doubly robust column is what pays it back.

**43. A known propensity is not enough to trust IPW, and Criteo is the counterexample**
(week 5). The rule this week was heading towards was "read the clipped share, and where it is
large read the doubly robust column". ACIC at 46% clipped and IHDP at 17.3% support it. Criteo
refutes it, and the refutation is worth more than the rule.

Criteo's propensity is the design constant 0.85, not one unit is clipped, and its IPW gain is
still half again its DR gain, on every seed, ratio 1.35 to 1.77. The DR column agrees with the
plain arm difference inside the prefix; IPW does not.

Decomposed on seed 11 it is exact rather than approximate. The top 10% of the S-learner's
ranking has a realised treated share of 0.8667 against the design value of 0.85, roughly eight
standard errors out. Horvitz-Thompson divides the control arm by 1 - 0.85, so each control unit
carries a weight of 6.67, and that 1.7-point shortfall predicts a gap of +0.003866 against an
observed +0.003866.

The reason a covariate ranking can move the treated share at all is that Criteo's arms are not
quite balanced: all twelve covariates lean the same way and the largest standardised mean
difference is 0.047, which on 1.4M rows is about twenty standard errors and still far below the
conventional 0.1 threshold `itx/metrics/balance.py` tests against. The balance detector passes
Criteo and is right to; an imbalance too small to fail a balance test is large enough to break
an estimator that divides by 0.15.

The corrected statement is about the shape of the design rather than about clipping.
Horvitz-Thompson weighting is fragile whenever one arm is small, because a selected subset need
not carry the population's treated share and the small arm's weight multiplies the difference.
Both failure modes end in the same advice, which is to read the doubly robust column.

Two consequences. The cheap diagnostic is to compare the realised treated share inside the
targeted prefix against the design propensity; it costs one mean, it is not yet a column in the
tables, and it belongs with the sensitivity work in week 6. And the claim in
`itx/policy/policy_value.py` that the gap between `uplift@k` and the IPW gain is "the sampling
noise in that share" was wrong: a ranking selects on covariates, so the concentration is
systematic rather than random. Both places are corrected.

**44. One Criteo fit took 1h27m against 4m16s for the same estimator on other seeds** (week 5).
Recorded because it looked exactly like a hang and very nearly was reported as one. The
R-learner on Criteo seed 37 took 1h27m; the identical estimator on seeds 11 and 23 took 4m16s,
and the fit immediately after it took 1m23s, so the run neither degraded nor recovered, it had
one pathological fit. What distinguished it from a hang, checked at the time, was that the
parent process was blocked at 0.1s of CPU per minute while about twenty causalml worker
processes underneath it were each burning 3 to 20 seconds per 25, with 32 GB of memory free.

Not diagnosed further. It cost the run about 80 minutes, `itx benchmark --all` still finished
in 10h19m end to end, and the cause matters only if it recurs. The note exists so that the next
person to see a stalled Criteo log checks the worker processes before killing the run.

**45. Dragonnet breaks the one-base-learner rule, and is untuned, and both are stated rather
than smoothed over** (week 6). CLAUDE.md says the base learner is LightGBM everywhere so that
differences in the table are differences between estimators. Dragonnet cannot honour that,
because a neural architecture is the thing being tested. Its column therefore answers a
different question from the other five, and the gap between it and the T-learner confounds
"a network with a propensity head" with "two gradient-boosted trees". It is in the table
because section 2's definition of done requires it, not because it is a clean experiment.

It is also in `UNTUNED`. The committed grid is over `min_child_samples` and `num_leaves`,
which are LightGBM's knobs and mean nothing to a network, so there was a choice between
running Dragonnet at the paper's published defaults and giving it a grid of its own. A
bespoke search for the one estimator that could not use the shared grid is a more visible
thumb on the scale than no search at all, given that the grid is identical across estimators
precisely so the estimator column does not become a compute column. So it runs at the
defaults and the table says "not tuned" next to it, the same way it does for the random
ranking.

Three things had to be added around the paper, all consequences of real data. Categorical
columns are one-hot encoded rather than passed through, because the loaders hand over integer
codes and a dense layer reads a code as a magnitude. Features are standardised on the
training rows. And there is a row cap of 200,000, because this is a CPU-only project by
budget and Criteo's 1.4M rows at a hundred epochs is not a benchmark anyone reruns. The cap
does not bind on IHDP, ACIC or Hillstrom, so those three are a control for what it does, and
what it costs is not yet measured. Change 30 is the template and the same measurement is owed.

**46. The E-value's continuous path produced 102 from five units, and the arm floor moved
from five to ten** (week 6). The first run of `itx sensitivity` on IHDP reported a targeted
group risk ratio of 51.6 and an E-value of 102.6, which reads as an overwhelming result and
is nothing of the kind.

Decomposed rather than assumed. IHDP's outcome is continuous, so there is no rate to divide
and the ratio comes from VanderWeele and Ding's conversion, `RR ~ exp(0.91 * d)`. At a 20%
budget the targeted group held 5 treated units against 25 controls, with a within-group
pooled standard deviation of 1.00 against the whole test split's 2.13, because a targeted
group is selected to be homogeneous. That gives `d` of 4.33, and the conversion is
exponential in `d`.

Two fixes, and the second is the more important. The arm floor rose from 5 to 10, which is
the same floor the risk-decile table already uses for a band and for the same reason: a rate
from five units is not a rate. IHDP at a 20% budget now reports no E-value at all, which is
the honest answer. And `d` is now carried on the result and named in the summary, with the
caveat that the conversion was built for smaller effect sizes, because an E-value from a
continuous outcome looks exactly like one from a real rate and is a weaker object. ACIC gives
`d` of 1.82 and a converted ratio of 5.24, already at the edge of what the approximation was
meant for; the three datasets section 2 names for this measure, Hillstrom, Criteo and Lenta,
are all binary and do not go through the conversion at all.

**47. "Matched pairs" on a randomised dataset were not matched on anything, and fixing it
made the number slightly worse** (week 6). Rosenbaum's bound is computed on matched pairs,
and the first version matched on the propensity score because that is what the method is
written around. On Hillstrom the design propensity is exactly 0.5 for all 42,693 rows. Every
unit is equidistant from every other, the caliper is infinite, and propensity matching
degenerates into pairing units in whatever order the greedy matcher happened to serve them.
The output still said "833 matched pairs".

The pairing is not invalid, because under randomisation any pairing gives a valid test, but
the word was claiming something the code was not doing. So where the propensity has no spread
the key now falls back to the prognostic score, the predicted outcome under control, and the
key that was used is named in the output.

The justification offered for that was power: pairs alike in baseline risk should carry more
signal. Measured over twelve matching seeds on Hillstrom at a 20% budget, that is wrong.
Arbitrary pairing gives Gamma 1.347 with a standard deviation of 0.043 on 833 pairs;
prognostic-score matching gives 1.300 with a standard deviation of 0.038 on 788, the caliper
having cost about forty-five pairs. No more stable, and slightly smaller. The fallback is
kept for the naming and because its error is in the conservative direction, not because it
works better, and the module says so.

The same table carries a limit on how the Gamma should be read anywhere in this project. It
moves by about 0.06 either way on nothing but the order the greedy matcher served units in,
so it belongs to one decimal place. Quoting 1.32 against 1.41 as though the gap meant
something would be reporting the matching seed.

**48. The treated-share gap is a column now, and it compares against the model rather than
against the design** (week 6, closing the loose end change 43 left). Change 43 ended with a
cheap diagnostic described and not built: compare the realised treated share inside the
targeted prefix against the propensity, because Criteo's IPW gain ran half again its doubly
robust one with a design propensity of 0.85, nothing clipped, and every overlap check
passing. It is now `share_gap@k`, computed inside `policy_metrics` from one more cumulative
sum over the same sorted order the gains use, so it costs nothing and cannot describe a
different set of people than the gains beside it.

One thing about it changed on the way in. Change 43 proposed comparing against the design
propensity, which only exists on the randomised datasets. Comparing against the *mean
propensity of the units in the prefix* is the same number where a design constant exists and
is defined everywhere else, and on the observational sets it doubles as a check that the
propensity model is calibrated on the units the policy actually picks, which is the only
place its calibration matters. So that is what it does.

It is computed at every budget and stored at every budget, and only the one at the operating
budget is rendered. The policy table is seven columns wide already and three more columns of
a diagnostic that moves slowly across budgets would cost more readability than it buys.

The column is empty until the next full benchmark run, because the results files were written
before it existed. Rather than backfill them, it arrives with Dragonnet on the same run.

**49. The second selection rule's cost is measured, and it is nothing** (week 6, closing the
loose end change 41 left). Change 41 added the policy-value selection rule and measured only
that it disagrees with the Qini rule on 26 of 30 ACIC cases. Disagreement is not a cost, and
the measurement that settles it is the one week 4 used on the tuning-rows cap: fit both
winning configurations on the full training split, score both on the held-out test split, and
report the difference.

All 26 disagreements, 2h40m. Changing from the Qini rule to the policy rule moves the true
gain at a 20% budget by a mean of -0.0011, bootstrap interval (-0.0137, +0.0099), with the
policy rule ahead in 15 of 26. ACIC's DR gain at that budget sits around 1.0 with a 95%
interval about 0.62 wide. The Qini moves by -0.0016 against intervals about 0.115 wide. Every
interval covers zero and every win rate is a coin flip. The rule changes 87% of the selections
and buys nothing measurable.

Two things are worth keeping out of it. First, ACIC is simulated, so the deciding row is the
true gain rather than an estimate, which matters because the rule under test selects on the DR
estimate and scoring it only on that would be marking its own homework. The DR row does favour
the policy rule more than the truth row does, +0.0274 against -0.0011, which is the direction
that suspicion predicts and is not evidence of it, since the DR interval is (-0.0241, +0.0856).

Second, `docs/estimators.md` previously reported a partial pass over seven cases from one seed
suggesting the differences might be larger here and in the policy rule's favour, and refused to
publish a number off seven cases. Finishing it refuted the suggestion.

The default stays on the Qini rule, but the reason in `itx/bench/grid.py` is downgraded from a
decision to an argument. That reason was about variance, that a single-budget policy value
reads one cutoff where the Qini integrates the whole curve. It may be right; it is not what
decides this. What decides it is that neither rule is measurably better, so section 4's
protocol keeps what is already there.

That is now three measurements from three directions saying the same uncomfortable thing.
Capping the tuning rows changes nearly every selection and costs nothing. Scoring on the
decision rather than the curve changes 87% of the selections and costs nothing. The selections
land somewhere different on nearly every partition of the same dataset. Roughly 89% of this
benchmark's compute goes on a choice among near-ties. Whether the grid earns its place is a
week 8 question for `docs/rejected.md`.

**50. One undefined metric took down a five-hour benchmark run, and the fragility was older
than the estimator that triggered it** (week 6). The week 6 rerun completed IHDP, ACIC,
Hillstrom and all thirty-five Lenta fits, then raised `ValueError: no finite values to
summarise` while rendering Lenta's table and exited, with Criteo not yet started.

The trigger was Dragonnet, whose calibration slope and calibration error are undefined on all
five Lenta seeds. The defect was `itx/bench/table.py`. `summarise` called `bootstrap_over` on
every estimator and metric, and that function raises when handed nothing finite, which is
correct for its other callers and wrong here: the table already renders an undefined estimate
as a dash, and one estimator having nothing to say about one column is an ordinary thing. It
should never have been able to end a five-dataset sweep at the final step.

Nothing was lost, because the fits were banked: Lenta resumed from its checkpoint in 8.2
seconds against the 5h11m it took to compute. That is the checkpointing of change 30 doing
exactly what it was built for, and it is the only reason this cost minutes rather than a day.

This entry originally said Dragonnet's Lenta fit was not degenerate, on the evidence that its
Qini, AUUC and uplift sat alongside the T-learner's. That was wrong, and the reasoning was
wrong in an instructive way: on Lenta the T-learner is itself near-random, so "alongside the
T-learner" carried no information at all. The fit was dead. Change 51 has it.

And the lesson generalises past this one function. A results table is the last step of a long
run, so anything that can raise there is expensive in proportion to everything that came
before it. `tests/test_bench.py::TestAnUndefinedMetric` reconstructs the failure from the
shape of the real data.


**51. Dragonnet produced an all-NaN ranking on Lenta, and it reported plausible numbers
instead of failing** (week 6). Chasing change 50's undefined calibration to its cause found
something worse than undefined calibration. Dragonnet's predicted uplift on Lenta is NaN for
every unit, on every seed.

The cause is a missing-value gap that only Lenta exposes. LightGBM accepts NaN natively, which
is why five meta-learners handle Lenta without anyone thinking about it, and Lenta is the only
one of the five datasets that has missing values at all: 19.5% of its cells, across 150 of its
191 columns, against exactly zero for IHDP, ACIC, Hillstrom and Criteo. A dense layer does not
accept NaN. Standardising a column that holds one gives a NaN mean, the design matrix goes NaN
on the first forward pass, and the weights never come back.

The way it surfaced is the part worth keeping. It did not look like a failure. The predictions
were all NaN, `rank_order` sorted them to one end, the ranking became the order the rows
happened to arrive in, and the results table reported a Qini of +0.0001 and a DR gain at 20% of
+0.0013 against random targeting's -0.0000 and +0.0012. A dead model produced numbers that
looked like a real estimator having a quiet day on a hard dataset. It was committed and pushed
before anyone knew.

Two things let it through, and both are fixed. The encoder now imputes with the training median
and adds a missingness indicator per affected column, because absence in Lenta is a fact about
the customer rather than a hole in the record: someone with no `cheque_count_3m_g20` never
bought from that group, and imputing a median asserts an average purchase history for people
who have none. And `_warn_if_degenerate` now tests for non-finite predictions by name. It was
written for the constant-zero case and `np.allclose(nan, 0.0)` is False, so it watched this
happen in silence.

The rule this leaves behind is worth more than the fix. The check that catches a dead model
must not be a check on the model's output looking wrong, because a dead model's output can
look perfectly ordinary once a ranking metric has finished with it. It has to be a check on the
output being a number.

**Refitted 2026-09-14**, a full 4h59m Lenta rerun rather than a surgical patch, because this
file's own note on change 30 warns against a results file that is half one version and half
another. Dragonnet's Lenta row now has a calibration slope of 0.030 instead of NaN and a Qini
of -0.0002 instead of +0.0001. The figures came back with it, since a full run carries per-unit
scores where a checkpoint resume does not.

The conclusion the broken row pointed at survives, which is luck and not vindication: Dragonnet
is still the weakest of the six modelled rankings on Lenta. One claim made from the broken row
does not survive and is corrected here. The commit that published the remeasured tables said
Dragonnet was "the only estimator there whose DR gain at 20% fails to exclude zero". On the
refitted numbers the outcome ranking covers zero too, at +0.0023 (-0.0001, +0.0049) against
Dragonnet's +0.0017 (-0.0001, +0.0035). The accurate statement is that Dragonnet is the weakest
of the six and that its interval covers zero where the five meta-learners' just exclude it.

**52. The demo's build step is a package, not a script in `demo/`** (week 7). Section 5 put
`demo/` at the repository root as the static site and gave the CLI an `itx demo build`
command, without saying where the code behind that command lives. It is `src/itx/demo/`,
beside the other packages, for the same reason change 40 put `itx diagnose` in
`itx/metrics/`: a command in `cli.py` should be a thin wrapper over something importable and
testable, and `tests/test_demo.py` checks the payload against `policy_metrics` rather than
against a stored copy of itself, which needs an import. The root `demo/` holds what gets
served: `index.html`, `app.js`, `style.css`, and the `data/` the build writes.

Three decisions inside it are worth recording because each is a limit rather than a feature.

**The page shows one split, and says so in its second paragraph.** The results table averages
five random splits; a slider cannot, because it has to move through a single ranking of actual
units rather than an average of five different rankings of five different test sets. So the
demo refits the first seed and its numbers sit near the table's without equalling them: on
ACIC at a 10% budget this split gives the risk-ranking baseline -0.2434 where the five-split
average is -0.1971. Both are correct and they are not the same quantity, which is exactly the
kind of gap a reader would otherwise find on their own and reasonably read as an error.

**The random baseline is computed exactly rather than sampled.** The benchmark's baseline is
the mean of 200 random rankings, which is right there because it carries an interval. The demo
draws a curve at fifty budgets, so it uses the closed form instead: a random subset carries the
population's average per-unit contribution, so the gain at share `b` is `b` times the gain from
treating everybody. Same quantity, no wobble for a reader to mistake for structure. A test
asserts the line is straight and that every ranking meets it at a 100% budget, since treating
everybody is the same policy however the list was sorted.

**Only the top 200 of each ranking ships.** Criteo's test split is 279,592 rows and the whole
ranking would be a multi-megabyte page load to render a list nobody scrolls; the page says how
many more it is not showing. Nothing identifying travels at all: row numbers and predicted
uplift, never feature values, and a test asserts no feature name appears in the file.

Not done in week 7 and not startable here: the fraud worked case. Section 3 specifies IEEE-CIS
Fraud Detection features, which is a Kaggle competition dataset behind an account, accepted
competition rules and an API token. There are none on this machine, and every other loader in
this project fetches from a direct URL and verifies a committed checksum. It needs either those
credentials or a substitute dataset, and substituting one is a change to section 3 rather than
something to do quietly. The cost-aware knapsack it was going to exercise stays built, tested
and unused, which is now the second week that has been true.
**53. The fraud worked case is built, on data that turned out to be next door** (week 7).
Change 52 recorded the case as blocked: IEEE-CIS is a Kaggle competition dataset behind an
account and accepted rules, with no credentials on this machine. Peter had already downloaded
it for project 09, with a manifest carrying per-file sha256 digests. The two training files
are hardlinked into this project's `data/raw/`, same volume so no copy, and verified against
the manifest's digests before the digest for `train_transaction.csv` was committed to
`checksums.sha256`. Only the training split is used; the test split has no labels.

Two mechanisms were added so that this file can sit beside the others without pretending to
be like them. `Source.manual` marks a file a loader cannot fetch, and `fetch` raises
`ManualDownloadRequiredError` with the Kaggle page and the target path when it is absent,
rather than an HTTP error that looks like a broken download. The card at
`docs/data/ieee-fraud.md` gives the three steps. Nothing downloads it and nothing commits it,
because the licence forbids redistribution.

The design is in `src/itx/data/ieee_fraud.py`, all of it in `simulate()` with named
constants, so the effect function is published rather than described. The features are real:
44 of the file's 394 columns, the `V1`-`V339` block dropped as uninterpretable. The treatment
is a fair coin, so the propensity is known and the case is about allocation rather than
identification. The outcome is dollars retained: a fraudulent transaction left alone is charged
back, a legitimate one earns a 3% margin, review catches fraud with a probability that *falls*
from 0.85 to 0.30 as the transaction's fraud signal rises, and wrongly declines legitimate
sales with a probability that rises from 0.01 to 0.15. Review costs 8 to 30 analyst minutes
depending on how much of the record is missing, which is the per-unit cost the cost-aware
knapsack of week 5 finally has something to spend.

The single load-bearing assumption is that review is hardest on what looks riskiest. It puts
the highest-risk transactions in the lost-causes quadrant and is the reason ranking a queue by
risk is not ranking it by what review buys. It is stated in the module docstring, the card, and
the README, and it is one constant: set `CATCH_FALL` to zero and the case becomes one where
risk ranking is optimal, which is a perfectly reasonable thing to believe about some review
operations.

Two consequences of the design worth knowing before reading any number from it. First, 96.5%
of the population are sleeping dogs, because review helps only the 3.50% that are fraudulent
and harms every legitimate transaction a little. Reviewing everything is worth about $2.32 a
transaction, and reviewing the right 2% by the true effect is worth more than reviewing all of
it, since the harm to the other 98% cancels most of the gain. Second, the per-unit truth is the
expected effect, not the realised coin flip, because an effect defined by one draw would be
unlearnable and PEHE against it would be measuring the coin. Both are asserted in
`tests/test_ieee_fraud.py`.

`itx allocate` runs the allocation comparison: four queues against the same budget of analyst
hours, scored on the doubly robust estimate this package would report on real data and on the
true value the simulation wrote, side by side. The dataset is registered in `DATASETS` so
`itx benchmark --dataset ieee-fraud` works, and deliberately not in `BENCHMARK_DATASETS`: it is
a worked case rather than a benchmark row, and putting an invented effect in the same sweep as
five measured datasets would invite a reader to compare a simulation against the world.

**54. The risk queue on the fraud case pointed the wrong way, and the diagnostic followed
it** (week 7). The first benchmark of ieee-fraud, 35 fits in 4h05m, put the outcome-ranking
baseline at a Qini of -0.4871 (-0.6142, -0.3615), far below random. `itx diagnose` on the same
split said "the effect runs against risk" with a correlation of -0.467 (-0.892, -0.333). And
`itx allocate`, on the same split, showed the risk queue worth $169,398 at 1,000 analyst hours
against random's $8,366 and every fitted uplift model's less. Three tools, one dataset, two
answers.

The cause is a convention that had never been tested against a value outcome. `OutcomeRanking`
ranks by predicted outcome, highest first, which is the risk queue on every dataset before this
one: the outcome is a response, a visit, a purchase or a score, and the risky unit is the one
most likely to have it. On dollars retained the risky unit is the one with the *lowest*
predicted outcome, and the baseline was queueing the most profitable legitimate sales under the
name of the thing every fraud team runs. The diagnostic took its "risk" from the same score and
its correlation from the raw control mean, so band 1 was the wrong end and the sign of the
correlation was the sign of the wrong thing. `itx allocate` was right only because it negated
the score by hand with a comment explaining why: one fact, encoded in one place and absent from
two, which is the shape of defect that survives review.

The fix is a declaration on the dataset. `UpliftDataset.risk_is_low_outcome`, default False,
set True by the fraud loader, is read by `OutcomeRanking` at fit time (it negates its score) and
by `risk_deciles` (the correlation and risk spread are taken against the signed control mean,
and band order follows the baseline). The hand negation in `itx allocate` is gone. Two ratio
columns in the decile table, the multiplier and the spreads, presume a positive baseline risk,
so they are now defined only where a band's control *risk* is positive: a band that loses money
has a multiplier (review that cuts the loss to 47% is 0.47x), a band that earns money has a
dash, and a spread across bands of both signs is undefined. Tests: the baseline reverses under
the flag; a dollars-retained population where the intervention removes loss in proportion to
risk gets "broadly agree" with the flag and "runs against risk" without it, which pins the
defect as a test; the flag survives `take`; the fraud loader sets it; and the ratio rules on
hand-built bands.

The five outcome-ranking rows were refitted from a checkpoint holding the other thirty, which
no code path in this change touches. The allocation numbers reproduce to the dollar, which is
the regression check that the negation moved and did nothing else. The corrected row and
diagnostic are in the README and `docs/estimators.md`.

Two other things in the same change. `itx allocate` gained an `oracle` queue, ranking by the
effect the simulation wrote, because the write-up needed the ceiling and the alternative was a
throwaway script, which is what week 5 was criticised for. And the case's own docstring, card
and README paragraph said the falling catch rate "means ranking the queue by fraud risk is not
ranking it by what review is worth"; measured, it is close to exactly that, and the three
places now say what was measured and why (`docs/estimators.md`, "where the risk queue wins").
The finding is the useful part of the week: "the most at-risk cases are the least movable" is
true on this case and is not sufficient, because a step of $74 between fraud and legitimate
dominates a slope of a factor of three inside fraud, and the README's two sentences that
presented a fraud queue as the natural home of the ACIC picture are qualified accordingly.

**55. `docs/rejected.md`: the hyperparameter grid, measured four times, does not earn its
place** (week 8). Section 9 listed three Rule C candidates and said whichever produced the
clearest evidence would get the document. None of the three did. The first, outcome ranking
as a policy, became the headline finding rather than a rejected approach. The second, the
S-learner collapsing on Hillstrom, was a prediction that failed: the S-learner has the best
realised value at 20% there. The third, the class-transformation method, was never built
(section 2 leaves it as literature). What did produce the clearest evidence was the tuning
grid itself, which changes 30 and 49 both deferred to this week.

The fourth measurement is the plain one the first three did not make: `--no-tune` on IHDP,
ACIC and Hillstrom against the committed tables. Fifty-four of fifty-four untuned five-seed
means fall inside their tuned intervals, the two rows the grid never touched reproduce to
every decimal as the control, and the estimator order changes only on the two datasets where
every interval overlaps every other. The largest move is the DR-learner's ACIC Qini, 0.2401
to 0.1862 inside (0.1765, 0.3085): the one estimator the grid was extended for in change 16
is the one that visibly used it, and it moves from best to worst without leaving anyone's
interval. The cost of the step is read off the fraud benchmark log against `fit_seconds`:
84% to 94% of an estimator's wall time on the large datasets.

The decision, stated in the document: the grid is rejected as a step that earns its cost;
the committed tables keep the protocol of section 4 because it was fixed before any number
was seen and a protocol changed at the end of a build to match its results is worse to ship
than a step that cost compute and changed nothing; the README tells anyone reusing this to
pass `--no-tune`; the per-seed selection blocks are provenance, not findings; a future version
that re-measures everything drops the step. Section 4's protocol text is unchanged for that
reason. The definition of done's Rule C line is ticked.

**56. The review before publication: what seven independent readings of the code and the
prose found, and what changed** (week 8). With every table measured and the clean rerun
under way, the whole repository was read again, in seven parts by seven reviewers who had not
written it: data and splits, ranking metrics and uncertainty, policy value and allocation,
estimators and tuning, sensitivity, prose against numbers, and the demo and CLI. Each finding
was verified against the code or the results files before anything moved. No benchmark
number changed. What follows is what did.

*Statistical claims that were wrong.* (a) The E-value was formed from the crude contrast
inside the targeted group, which equals the adjusted estimate only under randomisation; on
ACIC and IHDP it priced measured confounding, and the README read ACIC's 9.95 as the devices
"reporting the truth". The E-value is now not computed where the propensity is not a design
constant, and the section says what a Gamma is (a function of effect size and pair count,
detecting nothing) and what a negative control is on a confounded design (a
leave-one-covariate-out balance check). Three places said the negative-control test "plants
a confounder outside the covariate set"; it plants a measured one and removes it. (b) The
fraud allocation table's doubly robust estimate was a bare point, and the README concluded
from a $2,500 gap at 250 hours that the estimator "would have picked the wrong queue". The
column now carries a bootstrap interval over test rows with the queue fixed; the intervals
are about $65,000 wide, every truth is inside its interval, and the paragraph now says the
estimate cannot rank the queues. (c) The interval beside every README number is the mean of
the five per-seed bootstrap intervals, not an interval for the five-seed mean, and several
sentences leaned on it: "all five uplift models exclude zero" on Lenta at 20% is five of
five seeds for the DR-learner, four for R, three for S, two for T and X; "the whole interval
below zero" on ACIC is four of five seeds; the Lenta S-learner thread "consistent across all
five seeds" is three of five and below random on one; the S-learner's ACIC slope "the only
one whose interval excludes 1" is two of five seeds with the DR- and R-learners also excluding
it on two, on opposite sides. The README now says what the interval is and each of those
sentences says what the seeds say. (d) "Ten times worse than random" on ACIC was computed
from `uplift@10%`, the metric the same page says is confounded there; replaced by the doubly
robust sign change. (e) Hillstrom's "two thirds of the way" contradicted the same page; it is
none at 10% and a third at 30%. (f) Lenta's card and the README both had the Lenta-Criteo
effect-size comparison backwards; Criteo's is a third larger.

*Descriptions that did not match the code.* The fraud table's `outcome-ranking` row was
described as fitted on unreviewed rows; it is fitted on all rows, and only `itx allocate`
and `itx diagnose` fit on controls. Hillstrom's card justified the default arm as "the larger
of the two effects"; mens is larger on all three outcomes, and its spend cell was the mens
value. The fraud loader's categorical encoder promised sorted, row-order-independent codes;
polars assigns them by first appearance, which is harmless for LightGBM and Dragonnet and
now said. The fraud signal's tie-breaking is by file position, which is time order and in no
feature: about 4% of the signal's variance, $1.75 of a $74 mean effect among fraud, is
unlearnable, stated in the card as a floor under every PEHE rather than changed. The
review-minutes range was 9.2 to 19.0, not 18.4. "Lost-causes quadrant" overstated a catch
rate that bottoms at 0.30. IHDP's indicator x14 is coded 1 and 2. The IHDP seed-11 table in
`docs/estimators.md` was a week 2 fit and the README inherited "buys 0.05" for a row that
buys 1.00. The Criteo decomposition's "+0.003866 predicts +0.003866" was the week 5 fit; the
committed fit's gap is +0.0040 and the closure is asserted by test rather than quoted.
Normalised AUUC's docstring said "at most 1"; the outcome-ordered reference is not a ceiling
and random targeting lands anywhere from 0.04 to 0.83 under it. The Qini docstring's "one
extra event per hundred people" omitted that the coefficient scales with the treated share.
Calibration error is a mean absolute gap with a noise floor that its interval inherits, so it
is an upper bound to compare, not a quantity to read against zero; three committed rows have
the point outside their own interval for that reason and the README averaging hid them.

*Defects in what tools produced.* The demo tie-broke rankings with the split seed where the
benchmark uses its own tie seed, so on Criteo and Lenta, where LightGBM scores tie, the
page's numbers differed from the table's in the fourth decimal and its unit lists were a
different treated set; it now uses the benchmark's seed, and the units-treated count uses
the same ceiling as the policy (97 rows at 10% of 961, not 96). `itx benchmark` drew the
random reference's single stored draw as a solid Qini curve under the 200-draw label; the
dashed line was already the reference and the solid curve is gone. `itx demo build` with no
arguments would have published the fraud case and two synthetic sets; it now defaults to
the benchmark datasets. The E-value and the Rosenbaum bound in one `itx sensitivity` run
could be priced on marginally different top-20% sets because they used different tie seeds;
one seed now. `itx diagnose` used a tie seed of 0; the benchmark's now. The "never buy
predicted harm" stop in the allocation applied to the risk score and the random draw, whose
zero means nothing about harm; it now applies to the uplift queues only, and no published
number moved because the budget bound first. A non-finite test score now stops the run with
`DegenerateFitError` rather than a warning, which is what change 51 should have done. A seed
with an undefined metric no longer blanks a table cell; the cell is the mean of the seeds
that had one and `n_seeds` says how many. Dragonnet warns when it passes a wide categorical
through as a number, as its docstring already claimed.

*Deviations recorded rather than fixed.* IHDP and ACIC are reported on their first replicate
(index 0) rather than averaged over replicates as section 4 said; one replicate is one
problem, and ACIC's ten are ten different problems of which two have a constant effect, so
the average was never the right object. Dragonnet's 200,000-row cap binds on three of six
datasets and it does not standardise the outcome, both stated in the README limitation and
`docs/estimators.md`; a fix changes three tables and is for a later version. The DR-learner
alone estimates the propensity on randomised data, and its binary-outcome models are L2 on
0/1 where the S/T/X first stages are classifiers; both stated. The Rosenbaum caliper is 0.2
SD of the propensity rather than of its logit; stated. Lenta needs a declared design
propensity before its Rosenbaum bound means anything; stated. The risk-decile diagnostic does
not handle an intervention that reduces a bad outcome; stated.

*The claim that no benchmark number moved, measured rather than asserted.* IHDP and ACIC
were refitted from scratch with every fix above in place and compared with the committed
results by `itx compare` at zero tolerance: both reproduce exactly. Those are the two
datasets the changes could plausibly have touched, since they are the ones with a true
effect, the confounded propensity and the calibration columns. The remaining four are
covered by the clean-environment rerun.

*What the reviewers confirmed correct*, for the record: the doubly robust formula and its
fast path; nuisances fitted once per split on training rows and shared by every estimator,
with test rows never reaching any nuisance, tuning or selection model; the design
propensities of Hillstrom (0.5, two of three equal arms), Criteo (0.85) and the fraud case
(0.5); the splits' stratification, disjointness and seeding; every loader's feature set
free of treatment, outcome and counterfactual columns; the Qini, AUUC, uplift-at-k and
calibration-slope arithmetic against hand values; PEHE and ATE error on test rows only; the
knapsack's budget, ties and reduction to rank-and-cut; every meta-learner's formula, the
X-learner's weight direction included; the fraud simulation's truth equal to the expected
effect; and every number in the fraud section traceable to a table cell.

**57. CI had been failing since Dragonnet arrived, for a reason none of the tables could
show** (week 8). The GitHub workflow installs the project with the `learners` extra and not
`neural`, so PyTorch was absent on the runner. Dragonnet is in `DEFAULT_ESTIMATORS`, so
`mypy --strict` could not resolve the module and every test that builds a default benchmark
row stopped at the import. Both jobs had failed on every push since week 6, including the
scheduled weekly run, and the two jobs that reproduce the committed numbers were skipped
because they depend on the ones that failed. Nothing about the measurements is implicated:
the benchmark steps never ran, and every number in the README was produced and reproduced
locally. The fix is `--extra neural` on all four install steps. The lesson recorded here is
that a red CI badge on a private repository is easy to stop reading, and the three weeks of
red were three weeks in which the workflow was proving nothing.

A second failure surfaced once the first was fixed, on the first run where the benchmark
job got far enough to reach it. The slow test that reads the real IEEE-CIS file, added in
change 53, fails on a runner rather than skipping: every other dataset arrives from a URL, so
`--run-slow` could assume the data would be there, and this one cannot arrive at all because
Kaggle serves it to an authenticated account and the licence forbids a mirror. The class now
skips unless the file is present and matches its committed digest, which is the honest
outcome on a machine that is not allowed to have it, and still runs here.

One attempted improvement was reverted in the same change. On Linux the PyPI torch wheel
declares the CUDA stack, several gigabytes this CPU-only project never uses, and the fix for
that is to resolve torch from the PyTorch CPU index. Doing so broke every local `uv run`
immediately: that host's certificate does not verify through the TLS interception on the
development machine, so uv could not reach the index it had just been told was the only
source for torch. It is reverted, the reason is in the workflow beside the install step, and
the download stays on PyPI where the cost is install time rather than correctness.

**58. The fraud loader encoded its categories differently in every process, and the
clean-environment rerun is what caught it** (week 8). `_encode` in
`src/itx/data/ieee_fraud.py` took the physical codes of a polars `Categorical`. Those are
assigned in order of first appearance, and the appearance order depends on how the read is
split across threads, so the same file on the same machine encoded to different integers in
different processes. Measured with a digest of the loaded columns: `ProductCD` has five
levels and came back as codes 1, 5, 7, 12, 14 in one process and 3, 5, 9, 16, 18 in the
next. The fix is `rank("dense")`, so a level's code is its position in the sorted list of
levels and nothing but the set of levels can move it. `acic.py` had used that idiom since
week 2; this loader was the only one that did not.

The instability is a permutation, not an offset. Reading the raw file three times gives
`ProductCD` the level orders W,S,H,R,C then W,R,C,S,H then W,C,R,S,H: polars reads the CSV in
parallel chunks, so first appearance means first appearance in whichever chunk registered
first, and neither the codes nor their order survives.

The docstring on the old `_encode` had noticed the instability and argued it was harmless, in
two parts, and the refit shows the first part nearly right and the second part wrong.

Wrong first: it said Dragonnet one-hot encodes these columns, as though that ended the matter.
The one-hot block is ordered by sorted code, so a permutation reorders the block; the
network's weight initialisation is drawn in a fixed column order, so the same seed puts
different weights on the same category. Seed 11 early-stops at 21 epochs in one process and
33 in another. All five Dragonnet rows moved on all eighteen metrics.

Nearly right second: a tree does split a declared categorical by identity, and LightGBM
sorts categories by their gradient statistics before searching for a partition, so
relabelling almost always fits the same model. Almost. On four of the five seeds every
LightGBM row came back identical under the new encoding, and the clone's rerun had already
reproduced all five under a second unstable encoding. On seed 71 the tie broke the other way:
the nuisance models changed, and with them the doubly robust gain of every row including the
random baseline, whose own ranking cannot depend on a feature at all; and the T, X and
DR-learner fits changed too, with the tuning grid selecting the identical configuration in
every case, so this is the same hyperparameters fitting a different tree. The moves are
small, mostly under half a percent, the largest being the DR-learner's Qini at 0.4263 to
0.4541, and all of them sit inside the intervals they had before. The lesson is that
invariance to relabelling is a near-certainty rather than a guarantee, and a claim of exact
reproducibility cannot rest on a near-certainty.

All five Dragonnet rows of the fraud table were therefore not reproducible by anyone,
including by this machine, and 270 metric values moved between the committed file and the
rerun. Every one of them stayed inside its own bootstrap interval and no statement in the
README depended on any of them, which is the only reason this is a reproducibility defect
rather than a wrong result. The refit under the fixed encoding moved a further 162 values
across the other seven rows, all of them on seed 71 and all inside their old intervals. It is still a wrong result in the sense that matters here: the
project's whole claim is a number a stranger can check, and a stranger could not have got
that number.

Two things about how it was found are worth keeping. It was invisible to the test suite,
which checked that the encoder produced codes and never that it produced the *same* codes,
so the new tests include row-order independence, the property the old encoding actually
failed. And it was invisible to every within-process check: two fits in one process are
byte-identical, so it could only surface in a comparison between two runs, which is exactly
what the clean-environment rerun is for and the reason section 10 asks for one. The first
attempt at that rerun reported a Criteo reproduction that had not happened, because the
script compared a file the interrupted benchmark had never rewritten against the untouched
copy it had been taken from. Both defects are the same defect: a check that cannot fail
tells you nothing, and both were caught only by asking what the passing check had actually
compared.

**59. The rerun, finished** (week 8). All six datasets reproduce from a fresh clone, a fresh
virtual environment built from the lockfile, and raw files verified against their committed
digests, compared with `itx compare` at zero tolerance. IHDP, ACIC, Hillstrom, Criteo and
Lenta reproduced at commit f89b91e. The fraud case reproduced at faadc2b, one commit later,
because the rerun itself found the defect in change 58 and the fraud table had to be
remeasured under the fixed encoding before there was anything stable to reproduce. The only
source difference between those two commits is `_encode` in the fraud loader; the other
change is a docstring and the rest is tests, so the first five datasets' result stands at the
released commit.

On the fraud case the match is total: all 40 rows agree on every one of the 18 metrics and on
the configuration the tuning grid selected, `fit_seconds` is the only field that differs
anywhere in the file, and `README.md` and both figures regenerated byte for byte.

Two things this exercise is worth recording for. It was supposed to be a formality that
confirmed tables already believed correct, and instead it was the only check in the project
capable of finding change 58, because that defect was invisible within a single process and
invisible to a test suite that asked whether the encoder produced codes rather than whether
it produced the same codes twice. And the first attempt at it reported a Criteo reproduction
that had not happened, because the script compared a file the interrupted benchmark had never
rewritten against the untouched copy it had been taken from. A reproduction check is worth
exactly as much as the answer to "what did the passing check actually compare", and that
question had to be asked twice here to get two different defects out.

**60. Three additions that answer the question this repository could not: what do I do if I
have no data** (week 8, after the definition of done was met). Everything here had assumed a
reader who already ran a randomised intervention and kept the records. That is a narrow
audience, and the honest reading of the limitation is not "come back when you have data" but
"here is the smallest study that could answer your question".

`itx power` sizes it, in both directions: units needed from an assumed effect, and, with
`--have`, the heterogeneity a study of that size could have found. It reports two thresholds
because they differ by more than people expect. Detecting that an intervention does anything
needs one sample; showing that targeting beats random needs `1 / (budget x (lift - 1)^2)`
times more, which at a 20% budget and a top group twice as good is five times, and at a 10%
budget and a top group half again as good is forty times. A pilot sized for the first
produces a targeting table that is noise and does not look like noise.

The formula is checked the way everything else here is checked. Given four summary numbers
per dataset, the test size, the outcome spread, the average effect and the treated share, and
no model, no features and no ranking, it predicts which of the six benchmarks could be ranked
on. It gets all six right, including both failures: Lenta, where 687,029 customers needed a
top group responding 2.61x the average and got 2.03x, and IHDP. That retrodiction is in
`tests/test_power.py` and is the only reason the module ships.

`itx diagnose --csv` accepts a stranger's file. Most of that loader is refusal, because the
failure worth preventing is not a crash but a clean table computed on data that could never
have supported one. Two declarations have no default and no guess: `--design`, because on an
observational file each band's number mixes the effect with whoever was likelier to be
treated and nothing here can detect that, and the caveat prints above the table rather than
only in the docs because the table is what gets copied out; and `--outcome-polarity`, for the
reason below.

**And building it found a defect in the diagnostic, on the project's own headline example.**
Every dataset here has a good outcome at the high end: a response, a conversion, dollars
retained. A churn dataset does not, and a retention campaign is the first example in this
repository's opening paragraph. The verdict reads the correlation between a band's risk and
its uplift, and treated a positive uplift as a benefit unconditionally, so on churn, where a
working intervention makes the number smaller, it read success as harm. Measured on one
synthetic file encoded both ways: as `churned` the correlation was -0.903 and the verdict was
"the effect runs against risk ... the difference between helping and doing harm"; as
`retained`, the same population gave +0.867 and "worth measuring". `UpliftDataset` now carries
`higher_outcome_is_better`, separate from `risk_is_low_outcome` because a churn dataset has
risk at the high end and benefit at the low end and one flag cannot carry both. Every existing
dataset is True, so no committed number moves, and that is exactly why nothing caught it.

This is the third sign defect in the project, after change 51 and change 54, and the third to
be found by running the same data through two paths and noticing they disagreed rather than
by any test. The pattern is worth naming: a sign convention is invisible to a test suite whose
fixtures all share the same convention.
