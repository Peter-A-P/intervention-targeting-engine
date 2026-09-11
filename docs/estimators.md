# Where each estimator breaks

The useful half of a comparison. This file accumulates as estimators land; it is complete
at week 6 (PLAN.md section 6). Every claim here has a number behind it from
`results/*.json`, produced by `itx benchmark`.

Estimators in the table so far: S, T and X learners. Baselines: outcome ranking, random
targeting.

---

## Which learner wins depends on the shape of the problem, and it is checkable

The most useful thing to know about meta-learners is that there is no ranking of them.
Which one is best on a dataset depends on whether the treatment effect is simpler or more
complicated than the baseline outcome it sits on, and that is a property of the problem
rather than of the method. The claim is standard. What follows is this repository checking
it rather than citing it, because a claim taken on trust is not evidence.

Two synthetic generators, identical in every other respect, differing only in that
arrangement. Five committed seeds, the same grid, PEHE on a held-out split with its 95%
bootstrap interval. Lower is better.

| Generator | Shape of the problem | S-learner | T-learner | X-learner |
|---|---|---|---|---|
| `synthetic-heterogeneous` | Nonlinear baseline in four covariates, near-linear effect in two: **the effect is the simpler surface** | **0.181** (0.164, 0.198) | 0.302 (0.288, 0.317) | 0.195 (0.184, 0.207) |
| `synthetic-complex` | Near-linear baseline in one covariate, an interaction gated by a hinge for the effect: **the effect is the harder surface** | 0.456 (0.402, 0.522) | 0.398 (0.351, 0.456) | **0.370 (0.317, 0.435)** |

The ordering reverses. It is the same five covariates, the same sample size, the same base
learner, the same committed grid and the same seeds; the only thing that changed is which
of the two surfaces is the complicated one.

Two of those intervals overlap at the edges, so the marginal comparison is not on its own
decisive. The per-seed comparison is, and it is the right one here because the five seeds
are the same partitions for every estimator, so the contrast is paired rather than
marginal. On `synthetic-heterogeneous` the S-learner has the lower PEHE than the T-learner
on **five seeds out of five**. On `synthetic-complex` it has the higher PEHE on **five out
of five**, and the X-learner wins every seed outright. Nothing here rests on a single
partition.

The mechanism is not mysterious. An S-learner puts everything in one model, so it can spend
its capacity on the baseline and describe the effect with a handful of splits on the
treatment indicator. That is efficient when the effect really is describable that way and
ruinous when it is not, because every extra piece of effect structure has to be bought with
splits that compete against the baseline for the same budget. A T-learner never faces that
competition, since each arm gets its own model, but it pays for it with the variance of two
independent fits, and the X-learner exists to keep the first property while reducing the
second. `tests/test_estimators.py::TestWhichLearnerWinsWhen` asserts the reversal in both
directions, so if a future change to the base learner flattens the difference, the build
says so.

**This also indicts the week 1 benchmark, which is why the second generator exists.**
`synthetic-heterogeneous` was written to test that estimators recover a known effect, not
to compare them, and it happens to be shaped the way an S-learner likes. Benchmarking
meta-learners on it alone would have produced a clean-looking result that was really a
property of the generator. The second generator was added for exactly that reason, after
the first attempt to explain the IHDP result below made a prediction and the prediction
turned out to be wrong.

**The prediction that failed, recorded because it was wrong.** The S-learner has the best
PEHE on IHDP, which is not what the literature would lead you to expect. The first
explanation offered was that IHDP's effect is nearly homogeneous, so an estimator biased
toward a constant effect is flattered: 91% of IHDP's units sit within 1.0 of a mean effect
of 4.0, a standard deviation of 0.86 against that mean. That explanation predicts the
S-learner should lose on a strongly heterogeneous set. It does not. On
`synthetic-heterogeneous`, whose effect has a standard deviation nearly twice its mean, the
S-learner still wins, on five seeds out of five. Homogeneity was the wrong explanation, and
the right one is the simplicity of the effect surface relative to the baseline, which the
two generators above separate cleanly. IHDP behaves like the first regime, and the
S-learner has the lowest PEHE there on all five seeds.

---

## S-learner

One model on the features plus a treatment indicator, predicted twice with the indicator
on and off.

**Where it breaks: the treatment indicator is one column among many.** A regularised tree
ensemble splits on whatever reduces loss most, and a treatment effect is usually small
next to the variation the other covariates explain. When the model declines to split on
the indicator, both predictions come from the same leaf, the difference is exactly zero,
and the estimator returns "no effect for anyone" without erroring.

This is not hypothetical. With `min_child_samples=100`, a value chosen as merely cautious
for Hillstrom's 25,000 training rows, the S-learner on IHDP's 448 training rows:

| min_child_samples | Splits on the treatment | Mean predicted uplift | PEHE |
|---|---|---|---|
| 100 | 0.0% | 0.000 | 4.187 |
| 40 | 3.9% | 4.259 | 0.507 |
| 20 (the default) | 2.0% | 4.242 | 0.554 |
| 5 | 1.4% | 4.121 | 0.548 |

The true average effect is 4.016. A PEHE of 4.187 at the top of that table is exactly the
PEHE of predicting zero for everybody, because that is what the model did.

Two consequences, both now in the code. `BaseUpliftEstimator` raises a
`DegenerateFitWarning` when a fitted estimator predicts zero uplift for every unit, so the
failure is loud instead of silent. And `min_child_samples` belongs in the per-dataset
validation grid rather than in a constant: one leaf-size setting cannot serve a 747-unit
dataset and a 64,000-unit one.

**Even when it works, it barely looks.** On Hillstrom's first seed the fitted S-learner
uses the treatment indicator in 5.3% of its splits. It recovers the average effect well,
mean predicted uplift 0.0463 against a sample ATE of 0.0454, and ranks better than anything
else in the table, but the mechanism is a model that spends 95% of its attention on a
variable the question is not about. That is the argument for the T-learner, which cannot
ignore the treatment because it fits the arms separately.

**What it is good at.** Recovering the average effect, and being simple enough that when
it fails the reason is visible. On the synthetic sets it recovers a known constant effect
of 1.0 to within 0.1 and ranks a known heterogeneous effect at a rank correlation above
0.6.

---

## T-learner

One model per arm, differenced. It is the direct fix for the S-learner's failure, and the
fix is structural rather than statistical: the two arms are fitted on disjoint rows, so no
amount of regularisation can collapse them into the same model. At a leaf size larger than
half the training set, where any split is impossible and the S-learner returns exactly zero
for everyone, the T-learner still answers, because each of its models only has to describe
one arm rather than the difference between two.
`tests/test_estimators.py::TestTLearner::test_it_cannot_regularise_the_treatment_away`
pins that down.

**Where it breaks: it pays twice for noise, and once more when an arm is small.** The two
models are fitted independently, so each contributes its own error and the difference
carries both, with nothing shared to cancel. When one arm is much smaller than the other,
which is the normal case in every application this package is aimed at, the model for the
small arm is fitted on few rows and its noise enters the predicted uplift as heterogeneity
that is not there.

IHDP is that case. `TLearner.arm_sizes()` reports **83 treated rows against 365 control**
on the first seed's training split. The treated surface is being estimated from 83
observations and then subtracted from a surface estimated from four times as many, and the
T-learner's PEHE of 0.857 is the worst of the three estimators there, on every seed. Its average effect is
fine, ATE error 0.125, which is the signature of this failure rather than a contradiction
of it: the errors cancel in the mean and survive in the ranking.

On Hillstrom, where the arms are 50/50 and there are 25,000 training rows, the penalty
mostly disappears: Qini 0.0031 (0.0009, 0.0053) against the S-learner's 0.0042
(0.0020, 0.0063). Those intervals overlap heavily and the honest reading is that the two are
not distinguishable on this dataset, not that one beat the other.

---

## X-learner

Impute each unit's own effect using the other arm's model, then fit a model to the imputed
effects, then combine the two by the propensity. Built for unequal arms, which is the
situation that hurts the T-learner most.

**It delivers on IHDP, which is where it should.** Against the T-learner on the same
83-versus-365 split, PEHE drops from 0.857 to 0.787 and ATE error from 0.125 to 0.103, the
best average-effect estimate of the three. It also wins `synthetic-complex` outright, on
all five seeds, PEHE 0.370 against the T-learner's 0.398 and the S-learner's 0.456. The
weighting is doing real work: the imputed effect that leans on the weakly estimated arm is
the one that gets down-weighted.

**Where it breaks: it inherits every bias in the arm models, invisibly.** Stage 2 fits a
smooth, confident model to the imputed effects, and if `mu0` is systematically wrong in
some region then every treated unit there has an imputed effect wrong by the same amount.
The X-learner cannot detect this and nothing in its output looks unusual when it happens.
`XLearner.stage_two_targets()` exists for that reason: it hands back the numbers stage 2
actually believed, which is the first place to look when the X and T learners disagree.

**Where it breaks second: it needs a propensity, and on confounded data that is the weak
point.** On Hillstrom the propensity is a design constant of 0.5 and is used rather than
estimated, because fitting a model to recover a number that is already known can only add
variance. On IHDP it has to be estimated, and the fit reports what that costs:

```
propensity estimated, raw range 0.0009 to 0.9664, 64 of 448 clipped at 0.01
```

Fourteen percent of the training rows have an estimated probability of treatment outside
`[0.01, 0.99]`. That is not a numerical detail, it is the dataset saying those units had
essentially no chance of being in the arm they are being compared against, which is exactly
what Hill's construction did on purpose when it removed a non-random subset of the treated.
Clipping keeps the arithmetic finite; it does not repair the overlap. `PropensityFit`
counts the clipped units and flags the fit, so the number appears next to the estimate
rather than being absorbed into it.

---

## Outcome ranking, the baseline that is supposed to lose

Not an estimator. A plain model of the outcome, ignoring the treatment, with its predicted
outcome used as a targeting score. It is here because it is what most targeting in the
wild actually does, and because it is the thing a Qini curve is worst at exposing.

**On Hillstrom it is indistinguishable from random.** Qini 0.0012, interval
(-0.0008, 0.0033), against the averaged random baseline's -0.0000: the interval still
contains zero after tuning on the validation split, while all three real estimators'
intervals exclude it. Uplift inside the top 20% is 0.0598 against random targeting's 0.0451
and the S-learner's 0.0802. So a targeting programme run this way, on a dataset where a
real effect exists and three different estimators find it, would have bought roughly a third
of the available advantage over doing nothing clever, and reported a healthy response rate
for it.

**On IHDP it is worse than random, and the Qini says it is the best.** This is the finding
worth the whole file. On the first seed:

| Ranking | Qini | Normalised AUUC | Uplift at 10% | Uplift at 20% | PEHE |
|---|---|---|---|---|---|
| S-learner | -0.0004 | 0.914 | 4.75 | 4.82 | 0.554 |
| Outcome ranking | **+0.0269** | 0.719 | 0.05 | 1.80 | not applicable |
| Random targeting | -0.0012 | 0.825 | 3.69 | 3.72 | not applicable |

Seed 11, test split, 150 units. The outcome ranking has the highest Qini coefficient of the
three and the lowest realised uplift at every budget, far below random: its top 10% of the
population buys 0.05 against random targeting's 3.69, on a true average effect of about 4.
A reader who picked the model with the best curve would pick the one that buys nothing.
`docs/figures/qini-ihdp.png` is the picture of it.

Two things are going on and both matter.

1. **A Qini coefficient is only identified when assignment is random.** The Qini curve
   rescales the control arm's outcomes to the size of the treated group in the prefix,
   which assumes the two arms are exchangeable. IHDP's treated units were thinned on
   purpose, so they are not, and the curve is measuring confounding as much as targeting.
   The Qini column for IHDP is reported because suppressing it would hide this, but it is
   not comparable to the Qini column for Hillstrom, and it is not the metric to read
   there. PEHE is.
2. **The curve rewards ranking on the outcome even when assignment is random**, because
   sorting the treated responders towards the front lifts the curve whether or not the
   treatment caused their response. That is the trap the project is built around, and it
   is why realised policy value at a budget, not the curve, is the headline number.

**Why it gets no PEHE.** Its scores are predicted outcomes, on the outcome's scale, not
effects. Running them through PEHE produces a large number that reads like a bad error
score and is really a units mismatch. The benchmark leaves those cells empty
(`estimates_effect = False`) rather than printing something meaningless.

---

## Random targeting

The floor. Reported as the average of 200 random rankings rather than one draw, because a
single random ranking is one sample from a wide distribution: on Hillstrom, individual
draws land anywhere in a Qini range of roughly ±0.002, which is half the size of the
S-learner's entire advantage. A model whose interval overlaps this one has not been shown
to beat a coin.

Its uplift at any budget is the population average treatment effect, by construction, and
the test suite asserts that. That makes it the right reference for the uplift-at-k
columns, where zero is not the null: targeting at random still buys the ATE.

---

## Tuning, and what the grid selection is worth

Every estimator gets the same six-candidate grid over `min_child_samples` and `num_leaves`,
selected on the validation split by Qini, never on test. `itx/bench/grid.py` sets out why
selecting on a metric this repository calls a poor referee is defensible: inside selection
every candidate is the same model class, so nothing can win by being a different kind of
model that games the curve. Across estimators, where that protection disappears, the
comparison is made on the test split against both baselines.

**On Hillstrom it helps a little and chooses consistently.** The S-learner's Qini moved
from 0.0040 untuned to 0.0042 tuned, which is nothing against an interval of roughly
±0.002. The choices are reasonably stable: the modal configuration takes three of five
seeds for the S-learner, the T-learner and the outcome ranking, and two of five for the
X-learner.

**On IHDP it chooses almost at random, and that is the honest reading.** The validation
split is 149 rows. Across five seeds the S-learner's grid picks three different
configurations, the T-learner four, the outcome ranking four. A selection rule that lands
somewhere different on nearly every partition of the same dataset is not finding a better
configuration, it is fitting the noise in a 149-row validation Qini. It is reported this
way rather than quietly suppressed, because the alternative, presenting per-seed tuning as
though it had found something, is the kind of thing this repository exists to argue against.
The consequence for reading the IHDP table is that the differences between estimators there
should be read against both the bootstrap interval and this instability.

---

## Still to come

DR and R learners (week 3), Dragonnet (week 6), causal forest if the schedule allows.
Approaches considered and left as literature rather than code, per PLAN.md section 2:
TARNet, CEVAE, and the class-transformation method.
