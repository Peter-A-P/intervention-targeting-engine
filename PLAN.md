# Plan: Intervention Targeting Engine

**Written:** 2026-09-06. **Status:** plan only, nothing built.
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

Six estimators, five datasets, five seeds each, one table. The definition of done requires
five estimators; the sixth (causal forest) is the stretch.

## 2. Scope and boundaries

In scope:

- One common `UpliftEstimator` protocol: `fit(X, treatment, outcome)`, `predict_uplift(X)`,
  `policy(X, budget)`; every estimator implements it, every metric consumes it.
- Meta-learners S, T, X, DR and R on LightGBM base learners, plus a causal forest
  (EconML `CausalForestDML`). Own thin implementations of S, T and X so the mechanics are
  visible; EconML and CausalML wrapped for DR, R and the forest.
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
  list re-ranks live, hosted on GitHub Pages at targeting.peterparker.ca.

Out of scope, on purpose:

- Instrumental variables, regression discontinuity, difference in differences. Different
  identification strategies, different project.
- Deep-learning CATE models (TARNet, CEVAE, Dragonnet). Noted in the "where each
  estimator breaks" write-up as a comparison from the literature, not implemented.
- Continuous or multi-valued treatments.
- Online or bandit-style allocation.
- A server. The demo is static by design (CA$25 budget, Rule B satisfied by a URL).

## 3. Data

| Dataset | Size | Treatment | Outcome | Ground truth | Access and terms |
|---|---|---|---|---|---|
| Hillstrom MineThatData | ~64k | Email campaign (3 arms; use womens-email vs none) | Visit, conversion, spend | No | Public download; check the site's terms |
| Criteo-UPLIFT v2 | 13.9M rows, 12 features | Ad exposure | Visit, conversion | No | Criteo research licence; download through the official page |
| Lenta | ~687k | SMS campaign | Purchase | No | Ships with `scikit-uplift`; check licence in the package |
| IHDP | 747 units, 100 replicates | Home visits (semi-synthetic) | Cognitive score | Yes, simulated | Public via the CEVAE and Dragonnet repositories |
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
itx/
  data/         loaders, checksums, splits, dataset cards
  estimators/   protocol, s_learner, t_learner, x_learner, dr_learner, r_learner, causal_forest
  policy/       rank_and_cut, cost_aware, policy_value (ipw, dr)
  metrics/      qini, auuc, uplift_at_k, calibration, bootstrap
  sensitivity/  rosenbaum, evalue, negative_control
  bench/        runner, results table renderer, seeds
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
| 1 | Sep 7 - 13 | Repo scaffold (`uv`, `ruff`, `mypy --strict`, `pytest`, CI); data loaders with checksums for Hillstrom and IHDP; `UpliftEstimator` protocol; S-learner end to end on Hillstrom | CI green; first Qini curve plotted |
| 2 | Sep 14 - 20 | T and X learners; metrics module (Qini, AUUC, uplift at k, bootstrap); random and outcome-ranking baselines; synthetic-data tests | Three estimators, one table, intervals on Hillstrom |
| 3 | Sep 21 - 27 | DR and R learners via EconML/CausalML wrappers; IHDP and ACIC loaders; PEHE and ATE error; calibration plot | Ground-truth metrics for five estimators |
| 4 | Sep 28 - Oct 4 | Criteo and Lenta loaders; Polars pipeline and 10% subsample; full benchmark runner with seeds; results table renderer into README | `itx benchmark --all` runs end to end on a laptop overnight |
| 5 | Oct 5 - 11 | Policy module: rank-and-cut, cost-aware knapsack, IPW and DR policy value; the outcome-ranking trap demonstrated on every dataset | Policy value table with both baselines |
| 6 | Oct 12 - 18 | Sensitivity: Rosenbaum bounds, E-values, negative control; `docs/estimators.md` "where each estimator breaks"; causal forest if on schedule | Sensitivity section with numbers; write-up drafted |
| 7 | Oct 19 - 25 | Fraud worked case (semi-synthetic, declared); static demo built from precomputed rankings; GitHub Pages at targeting.peterparker.ca | Demo live, slider re-ranks |
| 8 | Oct 26 - Nov 1 | README to Rule A shape; `docs/rejected.md`; clean-environment rerun of the full benchmark; tag v0.1.0; flip the repository public | Definition of done all checked |

Slack: week 6's causal forest and week 7's cost-aware policy are the first things to
drop if behind. Neither is in the definition of done.

## 7. Demo

Static, because the budget is CA$25 and a static page is a URL a hiring manager can
click. Build step precomputes, per dataset, the ranked list with predicted uplift, cost
and cumulative policy value at every budget point; the page loads the JSON and a slider
moves the cutoff. Shows: who is in the treated set, the realised policy value at that
budget vs random, and the uplift-at-k curve with the cutoff marked. No backend, no
personal data (public datasets only, identifiers replaced by row numbers).

Hosting: GitHub Pages from the `demo/` build, CNAME `targeting.peterparker.ca`. DNS is a
one-line change on the domain already registered for Overload.

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
