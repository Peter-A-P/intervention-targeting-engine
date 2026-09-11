# Intervention Targeting Engine

Tells an organisation which customers, cases or patients to spend a limited intervention
budget on: only the ones whose outcome the intervention actually changes. Retention
offers, outreach, fraud review, clinical follow-up: a large share of every such budget
goes to people who would have behaved the same way regardless, and this finds them, and
shows the intervention list changing as the budget moves.

**Status: week 2 of 8.** Three estimators of the seven are benchmarked, on two datasets of
the five. The numbers below are real and reproducible; the table is not finished. Build
plan: [PLAN.md](PLAN.md).

## Results

Every table on this page is written by `uv run itx benchmark --dataset <name>` and is
never edited by hand. Each number is the mean across five committed split seeds, with the
95% bootstrap interval alongside it. Hyperparameters are chosen per seed on a validation
split from a grid that is identical for every estimator, and the test split is never
touched until the numbers below are computed. There are no bare point estimates here by
design: a Qini without an interval is treated in this repository as a defect.

### Hillstrom, 42,693 customers, randomised email campaign

<!-- itx:table:hillstrom -->
| Estimator | Qini (95% CI) | Normalised AUUC | uplift@10% | uplift@20% | uplift@30% |
|---|---|---|---|---|---|
| `s-learner` | 0.0042 (0.0020, 0.0063) | 0.2620 (0.1996, 0.3223) | 0.0883 (0.0387, 0.1355) | 0.0802 (0.0468, 0.1137) | 0.0803 (0.0545, 0.1077) |
| `t-learner` | 0.0031 (0.0009, 0.0053) | 0.2441 (0.1802, 0.3044) | 0.0876 (0.0380, 0.1386) | 0.0772 (0.0433, 0.1121) | 0.0715 (0.0435, 0.0980) |
| `x-learner` | 0.0032 (0.0011, 0.0054) | 0.2463 (0.1836, 0.3078) | 0.0823 (0.0346, 0.1325) | 0.0786 (0.0445, 0.1127) | 0.0748 (0.0477, 0.1026) |
| `outcome-ranking` | 0.0012 (-0.0008, 0.0033) | 0.2129 (0.1427, 0.2794) | 0.0532 (-0.0011, 0.1076) | 0.0598 (0.0222, 0.0978) | 0.0584 (0.0292, 0.0883) |
| `random-200` | -0.0000 (-0.0019, 0.0020) | 0.1931 (0.1605, 0.2267) | 0.0450 (0.0047, 0.0846) | 0.0451 (0.0181, 0.0728) | 0.0451 (0.0243, 0.0656) |

Selected on the validation split from the committed grid:

- `s-learner`: min_child_samples=5, num_leaves=15 (3 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds)
- `t-learner`: min_child_samples=20, num_leaves=15 (3 of 5 seeds), min_child_samples=60, num_leaves=15 (2 of 5 seeds)
- `x-learner`: min_child_samples=60, num_leaves=15 (2 of 5 seeds), min_child_samples=20, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds)
- `outcome-ranking`: min_child_samples=20, num_leaves=15 (3 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds)
<!-- itx:end:hillstrom -->

The outcome ranking is the ordinary way this job is done: model who is likely to respond,
spend the budget from the top of that list. Its Qini interval contains zero, so on this
dataset it has not been shown to beat picking at random. All three real estimators' do not.

At a budget covering 20% of the population, the S-learner's targeted group shows an
8.0-point difference in visit rate between arms, against random targeting's 4.5. The
outcome ranking manages 6.0, about a third of the way from doing nothing clever to doing
this properly. The three estimators' intervals overlap heavily, so the honest reading is
that they are not distinguishable from each other here, only from the two baselines.

### IHDP, 747 units, simulated outcomes with known individual effects

<!-- itx:table:ihdp -->
| Estimator | Qini (95% CI) | Normalised AUUC | uplift@10% | uplift@20% | uplift@30% | PEHE | ATE error |
|---|---|---|---|---|---|---|---|
| `s-learner` | 0.0176 (-0.0699, 0.1079) | 0.9064 (0.8315, 0.9670) | 4.0150 (2.8356, 4.9348) | 4.2496 (2.9210, 5.7731) | 4.3617 (3.2989, 5.3901) | 0.5661 (0.4344, 0.7022) | 0.1385 (0.0697, 0.2215) |
| `t-learner` | 0.0305 (-0.0578, 0.1222) | 0.9022 (0.8334, 0.9585) | 4.1149 (3.1768, 5.2630) | 4.2359 (3.4102, 5.4069) | 4.4967 (3.5423, 5.3866) | 0.8569 (0.7500, 0.9646) | 0.1248 (0.0508, 0.2608) |
| `x-learner` | 0.0468 (-0.0462, 0.1433) | 0.8962 (0.8303, 0.9548) | 3.9407 (2.7052, 5.5056) | 4.2619 (3.3427, 5.1678) | 4.2332 (3.4652, 5.0388) | 0.7870 (0.6081, 0.9695) | 0.1032 (0.0219, 0.2284) |
| `outcome-ranking` | 0.0088 (-0.0479, 0.0692) | 0.7345 (0.6413, 0.8244) | 1.3420 (-0.3277, 3.1934) | 2.2051 (0.9486, 3.5478) | 2.7399 (1.7099, 3.7437) | - | - |
| `random-200` | 0.0001 (-0.0751, 0.0712) | 0.8283 (0.7705, 0.8888) | 3.9298 (2.2726, 5.5296) | 3.9178 (2.8451, 5.0637) | 3.9275 (3.2068, 4.6875) | - | - |

Selected on the validation split from the committed grid:

- `s-learner`: min_child_samples=5, num_leaves=15 (2 of 5 seeds), min_child_samples=60, num_leaves=15 (2 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
- `t-learner`: min_child_samples=5, num_leaves=15 (2 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds), min_child_samples=20, num_leaves=15 (1 of 5 seeds)
- `x-learner`: min_child_samples=60, num_leaves=15 (3 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds)
- `outcome-ranking`: min_child_samples=5, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
<!-- itx:end:ihdp -->

The true average effect here is about 4.0, and PEHE is the per-unit error against an effect
somebody wrote down, so lower is better and only this dataset can report it. Which
estimator wins it is not a fact about the estimators: it depends on whether the treatment
effect is simpler or harder to describe than the baseline outcome, and
[docs/estimators.md](docs/estimators.md) shows the ordering reversing completely between
two synthetic problems that differ in nothing else.

Read the two right-hand columns, not the Qini. On the first seed the outcome ranking has
the **best Qini coefficient of the three** and buys 0.05 at a 10% budget, against random
targeting's 3.69. The curve ranks the worst policy first. Two reasons, both in
[docs/estimators.md](docs/estimators.md): the Qini is only identified when treatment was
assigned at random, and IHDP is confounded on purpose; and even on randomised data the
curve rewards sorting responders to the front whether or not the treatment caused them to
respond. That is the trap this project is built around, and it is why the headline number
is what a budget actually buys.

![Qini curves on IHDP](docs/figures/qini-ihdp.png)

## What this does not do

- It does not identify effects without an experiment or a credible ignorability
  assumption. The sensitivity analysis quantifies how wrong that assumption can be before
  the targeting decision flips; it does not remove the assumption.
- It does not handle continuous or multi-valued treatments, or online allocation.
- The fraud worked case uses a simulated review intervention on public data and says so.
- Not yet built, in schedule order: DR and R learners; the Criteo, Lenta and ACIC loaders;
  realised policy value under a budget; Rosenbaum bounds and E-values; Dragonnet; the
  budget-slider demo. Nothing above is a placeholder for them: the numbers reported are the
  numbers measured.

## Install and run

Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # create the environment
uv run itx data pull          # download the raw datasets and verify their checksums
uv run itx benchmark --dataset hillstrom
uv run itx benchmark --dataset ihdp
```

`itx benchmark` selects hyperparameters on the validation split, fits, evaluates on the
held-out test split, writes every per-seed number and the chosen configuration to
`results/<dataset>.json`, draws the Qini figure into `docs/figures/`, and rewrites the
table above. Add `--no-tune` to skip selection and use the default configuration, which is
several times faster. Raw data is never committed; the SHA-256 of every download is, in
[`src/itx/data/checksums.sha256`](src/itx/data/checksums.sha256), so a download can be
verified without running any of this code:

```bash
cd data/raw && sha256sum -c ../../src/itx/data/checksums.sha256
```

```bash
uv run pytest               # fast tests, synthetic data only
uv run pytest --run-slow    # adds the tests that need a real download
uv run ruff check . && uv run mypy
```

## How it works

Every organisation that spends money on people has to pick which people. A retention
offer, an outreach call, a transaction pulled for manual fraud review, a clinical
follow-up: the budget covers a fraction of the population, so somebody chooses the
fraction.

The last two look different from the first two and are the same problem. An analyst hour
spent on a transaction is worth spending only if the review changes what happens to it,
and most reviews do not: the blatant fraud is stopped by the automated rules anyway, and
the clean transaction was never going to be a loss either way. A follow-up call after
discharge is worth making only if it is the call that keeps the patient out of hospital,
not when they would have been readmitted whatever anyone did, or were never going back in.
In all four, the budget buys a fixed number of actions, and most of those actions land on
people whose outcome was already settled. The normal way to choose who gets them is to
rank everyone by risk and treat the top of the list.

That is the wrong list. It is wrong in a way that is easy to miss, because it still
produces a report that looks fine.

Ranking by risk finds the people most likely to have the bad outcome. It does not find the
people whose outcome the intervention would change. Those are different groups. Sort a
population by what the intervention actually does to each person and there are four kinds
of people:

| | If nobody acts | If somebody acts | Worth the budget |
|---|---|---|---|
| **Persuadables** | bad outcome | good outcome | Yes. They are the entire return. |
| **Sure things** | good outcome | good outcome | No. They were fine already. |
| **Lost causes** | bad outcome | bad outcome | No. Nothing on offer helps. |
| **Sleeping dogs** | good outcome | bad outcome | No. Acting does damage. |

A risk model ranks people by the first column alone. It puts lost causes near the top,
because they really are the highest risk, and it has nothing at all to say about whether
the phone call helps. Persuadables sit in the middle of a risk ranking and get missed. And
a risk model cannot see sleeping dogs, so a campaign can spend its budget making outcomes
worse and still report a healthy response rate.

This repository is the machinery for ranking on the second question instead, "would acting
change this person's outcome", and for the part that usually gets skipped: showing that the
ranking is real rather than plausible. It ends in the results table above, regenerated by
one command, which reports for every method how much outcome a fixed budget actually buys,
with a confidence interval, against two baselines: random targeting, and the risk-ranking
approach described here. A slider on the demo page moves the budget and the intervention
list re-ranks, because "who do we treat" has a different answer at 10 percent of the
population than at 30.

The design, the datasets, the evaluation protocol and the schedule are in
[PLAN.md](PLAN.md).

### In more detail

**Why this is harder than a normal model.** For any one person, only one of the two
outcomes ever happens. You either sent the offer or you did not, so you see either the
treated outcome or the untreated one, never both. The quantity this project estimates, the
difference the intervention made to that individual, is therefore missing for every
individual in the data. It is not scarce or noisy. It is absent, always, by construction.
That is why the usual machine-learning reflex does not work here: you cannot hold back a
test set and check predictions against the truth, because the truth was never recorded.
Any approach that claims otherwise is measuring something else, usually its own internal
consistency.

**How you check it anyway.** Two kinds of data, doing two different jobs. Randomised data,
where the treatment was assigned by coin flip: three public marketing and messaging
experiments, the largest with 13.9 million rows. Individual truth is still missing, but
because assignment was random, group averages are trustworthy, which is enough to answer
the practical question of what outcome a given ranking would have produced at a given
budget. And simulated data, where the effect was generated from a known formula: two
standard benchmark sets from the causal-inference literature. Here the individual truth
exists because somebody wrote it, so per-person error can be measured directly. That is
the only way to show an estimator is correct rather than merely self-consistent, which is
why both kinds are in the benchmark and why neither is a substitute for the other.

**Several ways of estimating the same thing.** The field has no single agreed method, so
five meta-learners, a neural estimator and, if it makes the schedule, a causal forest all
sit behind one interface and appear in one table. The meta-learners are recipes that build
an effect estimate out of ordinary prediction models, differing in how they handle the
missing half of the data and in how they behave when the treated group is much smaller
than the untreated one. The neural estimator is Dragonnet, written in PyTorch, and it is
there deliberately, as a test of whether the extra machinery earns its keep on problems
this size, with a stated verdict either way. All of them sit on the same LightGBM base
learner, so differences in the table are differences between the methods rather than
differences between their engines. [docs/estimators.md](docs/estimators.md) says where each
one breaks, which is usually the more useful half of a comparison.

**The trap this is built around.** The standard score for this kind of model is the Qini
coefficient, drawn as a curve: sort everyone by predicted effect, walk down the list, and
plot the cumulative gain against what random targeting would have given you. It is a good
diagnostic and a poor referee, because a plain risk model, the wrong list from the top of
this section, often scores respectably on it. So every ranking in the results table is
shown next to that risk-ranking baseline and next to random targeting, and the headline
number is not the curve but the realised policy value: the outcome that would actually
have been obtained, estimated on held-out data by an inverse-propensity and a doubly
robust estimator, from following the policy at a stated budget. Where a ranking looks good
on the curve and buys nothing in practice, the table says so. Every metric carries a
bootstrap confidence interval; a score reported without one is treated here as a defect.

**From a ranking to a decision.** A ranked list is not yet an allocation. Two allocation
rules are implemented: take the top N when every intervention costs the same, and solve
for the best affordable combination when costs differ per person, which is the usual
situation once a fraud review costs an analyst an hour and an automated message costs a
fraction of a cent.

**How wrong the assumptions can be.** On data that was not randomised, targeting rests on
an assumption that everything relevant was measured. That assumption is never quite true
and cannot be tested from the data itself. Rather than assume it away, the sensitivity
analysis quantifies its fragility with Rosenbaum bounds and E-values: how strong a hidden
factor would have to be before the targeting decision changes. The number reported is the
point at which the decision flips, not the point at which statistical significance flips,
because the decision is what the budget holder is actually buying.

## Documentation

- [PLAN.md](PLAN.md): scope, evaluation protocol, package design, week-by-week schedule.
- [docs/estimators.md](docs/estimators.md): where each estimator breaks, with evidence.
- [docs/data/hillstrom.md](docs/data/hillstrom.md),
  [docs/data/ihdp.md](docs/data/ihdp.md): one card per dataset, with source, licence,
  treatment definition, quirks and split seeds.

## Part of a portfolio

One of fifteen projects built over twelve months to make production ML work inspectable.

## How this was built

Design, methodology, evaluation choices and judgement are Peter Parker's. AI coding
assistants (Claude Code) were used for implementation and drafting, the way a senior
engineer uses them in 2026. Every number in the results table is reproducible from this
repository with one command, and that reproducibility is the evidence that matters.
