# Where each estimator breaks

The useful half of a comparison. This file accumulates as estimators land; it is complete
at week 6 (PLAN.md section 6). Every claim here has a number behind it from
`results/*.json`, produced by `itx benchmark`.

Estimators in the table so far: S, T, X, DR and R learners. Baselines: outcome ranking,
random targeting.

The single most useful thing in this file is in the outcome-ranking section: risk ranking
matches every uplift model on Criteo, is worse than random on ACIC, and on Lenta nothing
separates from random at all. One decile table says in advance which of those a problem is.

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
| `synthetic-heterogeneous` | Nonlinear baseline in four covariates, near-linear effect in two: **the effect is the simpler surface** | **0.181** (0.164, 0.198) | 0.308 (0.293, 0.324) | 0.208 (0.196, 0.221) |
| `synthetic-complex` | Near-linear baseline in one covariate, an interaction gated by a hinge for the effect: **the effect is the harder surface** | 0.456 (0.402, 0.522) | 0.398 (0.351, 0.456) | **0.370** (0.317, 0.435) |

The ordering reverses. It is the same five covariates, the same sample size, the same base
learner, the same committed grid and the same seeds; the only thing that changed is which
of the two surfaces is the complicated one.

Two of those intervals overlap at the edges, so the marginal comparison is not on its own
decisive. The per-seed comparison is, and it is the right one here because the five seeds
are the same partitions for every estimator, so the contrast is paired rather than
marginal. On `synthetic-heterogeneous` the S-learner has the lower PEHE than the T-learner
on **five seeds out of five**. On `synthetic-complex` it has the higher PEHE on **five out
of five**, against both the T-learner and the X-learner, and the X-learner wins every seed
outright. Nothing here rests on a single partition.

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

## The week 3 finding: an in-sample propensity is not a slightly worse propensity

The R-learner arrived, was benchmarked on ACIC 2016, and reported a PEHE of 23.95 against a
true effect whose standard deviation is 3.85. It ranked well while doing it, the best Qini
of the five, so the ordering was fine and the magnitudes were nonsense. That is a bug-shaped
result and it was one.

The cause was the propensity model, which fitted and then predicted the same rows. A model
asked about rows it has already seen has partly memorised them, so its prediction for a unit
is pulled toward that unit's own realised treatment. The residual ``t - e(x)`` then no longer
has conditional mean zero. The R-learner divides by exactly that residual, so it was dividing
by something systematically too small. Predicting each training row from folds that exclude
it fixes the orthogonality, and the numbers move like this:

| ACIC 2016, R-learner | In-sample propensity | Out-of-fold propensity |
|---|---|---|
| PEHE, seed 11 | 23.95 | 1.20 |
| PEHE, seed 23 | 24.35 | 1.42 |
| PEHE, seed 37 | 23.91 | 1.39 |
| Predicted effect, standard deviation | 24.2 | 3.79 |

The true effect's standard deviation is 3.85, so the out-of-fold version has the spread
about right and the in-sample one is off by a factor of six.

**The part worth keeping is what the diagnostics said while this was happening.** The
obvious things to check about a propensity model both looked fine, and both were the same
either way:

| Diagnostic | In-sample | Out-of-fold |
|---|---|---|
| Units against the clipping bound | 48.2% | 47.8% |
| Median treatment residual | 0.014 | 0.025 |
| PEHE of the estimator that uses it | **23.95** | **1.20** |

Nothing in the propensity's own summary statistics says one of these is unusable. The damage
is in the correlation between a unit's residual and its own outcome, which no marginal
summary of the propensity can see. It was visible only by comparing against a known truth.
PLAN.md section 1 justifies carrying the simulated datasets on the grounds that they are the
only way to show an estimator is correct rather than self-consistent; this is that argument
collecting on itself, and it is the clearest evidence in the repository for why a benchmark
of randomised datasets alone would not have been enough.

Cross-fitting is now the default and
`tests/test_estimators.py::TestCrossFittedPropensity` holds it there. The X-learner and the
DR-learner were unaffected: the X-learner uses the propensity only as a convex weight between
two bounded models, and EconML cross-fits its own nuisances internally.

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

## DR-learner

A pseudo-outcome combining an outcome model with an inverse-propensity correction, regressed
on the covariates. Doubly robust: consistent if either the outcome model or the propensity
model is right, which is two chances instead of one.

**Where it breaks: the guarantee is asymptotic and the variance is not.** The pseudo-outcome
carries the inverse-propensity correction's variance on top of the outcome's, and the final
stage has to regress on that. With enough rows this is the best estimator in the table. With
few it is the worst, by a distance:

| Benchmark dataset | Training rows | DR-learner PEHE | Best other estimator |
|---|---|---|---|
| IHDP | 448 | **8.54** | 0.57 (S-learner) |
| ACIC 2016 | 2,881 | 1.79 | 0.83 (X-learner) |

And a probe outside the benchmark, holding everything fixed except the sample size and the
leaf size, so the two effects can be seen apart. These are single untuned fits on
`synthetic-heterogeneous` rather than tuned five-seed benchmark rows, and are labelled that
way because they are not comparable with the table above:

| Rows | Leaf size 20 | Leaf size 60 | Leaf size 200 |
|---|---|---|---|
| 4,000 (2,400 training) | 1.28 | 0.67 | 0.40 |
| 20,000 (12,000 training) | 0.29 | 0.30 | 0.28 |

At 2,400 training rows the DR-learner is worse than predicting a constant unless it is
heavily regularised. At 12,000 the choice stops mattering and it lands at 0.29, better than
any other estimator manages on the same generator. Sample size is doing most of the work
here and regularisation is standing in for it.

The IHDP number deserves a moment. A PEHE of 8.54 on a dataset whose true effect has a
standard deviation of 0.86 is not a weak estimate, it is an actively harmful one: predicting
a constant zero scores 4.11 there, averaged over the same five seeds, so the DR-learner is
twice as wrong as doing nothing at all. Its
calibration error of 6.25 says the same thing from the other direction. Nothing about the
double robustness property protects against this, because double robustness is a statement
about bias in the limit and this is variance at n=448.

It is also the estimator most sensitive to how much regularisation the grid can offer, which
is why the grid gained a leaf size of 200 in week 3: at 20 its PEHE on 2,400 rows was 1.28,
worse than a constant, and at 200 it was 0.40.

**Where it works.** On ACIC's 2,881 rows it has the best Qini of the five, 0.2401
(0.1765, 0.3085), and a calibration slope of 1.01, the closest to honest in the table. Given
enough data it is doing exactly what it promises.

**Where it breaks for a reason that is not statistical at all: it refuses missing values.**
This is the only estimator of the six that will not accept a feature matrix containing NaN.
It is not a degradation, it is a `ValueError` before any model runs, and on Lenta, which has
missing values in 150 of its 191 columns, it is the difference between a results row and a
blank one.

The cause is worth naming precisely, because it is not a flaw in the DR-learner as a method.
EconML validates its inputs with scikit-learn's finiteness check at the top of `fit`, and
again inside every `effect` call. LightGBM handles NaN natively, routing it down a learned
branch at each split, and the S, T and X learners here use that without being asked, as does
CausalML's R-learner. But the check runs before EconML ever hands the matrix to the model it
was configured with, so LightGBM's capability never gets a chance to matter.

`allow_missing=True` turns the check off, and EconML then warns, per fold and per prediction:

> Input contains NaN. Causal identification strategy can be erroneous in the presence of
> missing values.

That warning is correct and is not boilerplate. If whether a covariate is observed depends on
the treatment, or on something that also drives the outcome, then the missingness pattern is
itself a confounder and no amount of doubly robust machinery repairs it. On Lenta it does not
bite, and the reason is the same property that made the leak in that dataset findable: the
assignment is randomised, so it is independent of the covariates and of their missingness
pattern by construction. On an observational dataset with missing values this warning would
need a real answer rather than a filter, and this repository does not currently have one.

The general lesson is the one that made this worth a section: **a wrapped estimator inherits
its library's input contract, not only its statistics.** Everything in this file up to here
compares methods. This is a place where the comparison was nearly decided by a validation
call, which is a thing worth knowing before choosing a library rather than after.

---

## R-learner

Residual on residual. Partial the covariates out of both the outcome and the treatment, then
regress what is left of one on what is left of the other, weighted by the square of the
treatment residual.

**Where it breaks: it is the estimator that most depends on its nuisances being right**, and
the week 3 finding above is the demonstration. Everything else in the table degraded
gracefully when the propensity was fitted in sample; the R-learner produced numbers that were
wrong by a factor of twenty. That is the flip side of its elegance: the other learners use
the propensity as a weight, and this one uses it as a denominator.

A claim in the first draft of this module's docstring has to be withdrawn, because the
evidence contradicts it. It said that a unit whose treatment was nearly perfectly predictable
contributes almost nothing, since its weight is the square of a near-zero residual, and so
the R-learner does not need clipping. That is true of the estimating equation and false of
the implementation: the target is divided by that same near-zero residual before the weight
is applied, and a tree fitted to enormous targets with tiny weights does not behave the way
the algebra suggests. Poor overlap hurts the R-learner like everything else.

**Where it works.** On ACIC it has the best ATE error of the five, 0.1387 (0.0733, 0.2141),
against a naive comparison that is out by 68%. On IHDP's 448 rows it is mid-table, PEHE
1.34, which is a reasonable showing for a method whose residuals have to be estimated well
before anything downstream means anything.

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

**On ACIC 2016 it is worse than random, and this time the Qini agrees.** The clearest
version of the whole argument:

| ACIC 2016 | Qini | Uplift at 10% |
|---|---|---|
| `x-learner` | 0.2273 | 8.41 |
| `dr-learner` | 0.2401 | 7.34 |
| Random targeting | -0.0000 | 3.42 |
| Outcome ranking | **-0.0233** | **0.34** |

Spending the budget on the highest-risk tenth of this population buys 0.34, and spending it
on a random tenth buys 3.42. The ordinary approach is not merely leaving most of the
available return on the table, it is ten times worse than not thinking about it at all. That
is what happens when the people most likely to have the outcome are the people least
susceptible to the intervention, which is the normal shape of a fraud queue or a clinical
follow-up list and the reason this project exists.

**On Criteo it is as good as everything else, and that is the result that makes the rest of
this file mean something.** 14 million randomised rows, 279,592 of them held out:

| Criteo | Qini | Uplift at 10% |
|---|---|---|
| `s-learner` | 0.0032 (0.0025, 0.0038) | 0.0556 |
| `r-learner` | 0.0026 (0.0019, 0.0033) | 0.0586 |
| Outcome ranking | **0.0031 (0.0024, 0.0038)** | **0.0576** |
| Random targeting | 0.0000 (-0.0005, 0.0005) | 0.0104 |

The baseline this repository exists to catch out is statistically indistinguishable from the
best estimator in the table, on the largest and best-powered dataset here, where the intervals
are tight enough that a real gap would show. Uplift modelling buys nothing on Criteo.

**Why, and the rule that falls out of it.** Rank the test rows by predicted risk and read both
arms inside each decile:

| Decile by risk | Control | Treated | Absolute uplift | Ratio |
|---|---|---|---|---|
| 1 (highest) | 0.3166 | 0.3743 | **+0.0577** | 1.18 |
| 2 | 0.0571 | 0.0684 | +0.0113 | 1.20 |
| 3 | 0.0172 | 0.0192 | +0.0020 | 1.12 |
| 4 | 0.0095 | 0.0078 | -0.0016 | 0.83 |
| 5 | 0.0043 | 0.0050 | +0.0006 | 1.15 |
| 10 (lowest) | 0.0002 | 0.0007 | +0.0004 | 2.82 |

Baseline risk falls by about 1,500x from the top decile to the bottom. The multiplier moves by
about 2x, and not monotonically. Absolute uplift is risk times multiplier, so on this data it
is essentially risk, and the two rankings coincide.

Hillstrom is the same table with a much flatter risk column: 0.1695 down to 0.0275, a spread
of about 6x, against a ratio moving from 0.96 to 2.68. There the multiplier's variation is
comparable to the risk's, so risk ranking captures part of the ordering and not all of it,
which is exactly the two-thirds-of-the-way result above. Correlation between decile risk and
realised uplift is -0.60 on Criteo and -0.24 on Hillstrom.

So the three datasets are three points on one spectrum, and the honest generalisation is
narrower and more useful than "risk ranking is a trap":

| | Risk spread | Uplift follows risk | Outcome ranking lands at |
|---|---|---|---|
| Criteo | about 1,500x | -0.60 | parity with the best estimator |
| Hillstrom | about 6x | -0.24 | about two thirds of the way |
| ACIC 2016 | effect runs against risk | positive | worse than random targeting |

**Ranking by risk approximates ranking by uplift exactly when the spread in baseline risk
dwarfs the spread in relative effect.** That is checkable before any meta-learner is fitted,
from one decile table on a randomised sample, and it decides whether the rest of this
repository is worth running on a given problem. Saying so costs the project its simplest
headline and is the finding most likely to be of use to somebody.

A caveat that belongs next to it: the decile table is only available where assignment is
random or credibly ignorable. On IHDP, where it is neither, the same table would be measuring
confounding rather than effect heterogeneity, which is the first of the two failure modes
described above. The diagnostic inherits every assumption the estimators do.

**On Lenta nothing separates, including the estimators.** Every Qini interval in that table
contains zero: five estimators, the outcome ranking, and random targeting all overlap.

| Lenta | Qini | Uplift at 10% |
|---|---|---|
| `s-learner` | 0.0006 (-0.0003, 0.0016) | 0.0184 (0.0008, 0.0357) |
| `dr-learner` | 0.0004 (-0.0006, 0.0014) | 0.0163 (-0.0003, 0.0337) |
| Outcome ranking | 0.0005 (-0.0005, 0.0014) | 0.0075 (-0.0118, 0.0267) |
| Random targeting | -0.0000 (-0.0008, 0.0007) | 0.0073 (-0.0034, 0.0175) |

It is a power problem rather than a modelling one: a 0.75-point effect on a 10.3% base rate,
137,406 test rows, about 34,000 of them controls. The only thread is that the S-learner's
realised uplift excludes zero at all three budgets, 0.0184, 0.0132 and 0.0118 against random
targeting's flat 0.0074, consistently across five seeds, while its Qini does not. A ranking
metric that integrates the whole curve is less sensitive to a good short prefix than the
budgeted number is, which is the argument for making realised policy value the headline and
is why week 5 exists.

Reported because a benchmark where every dataset gives a clean answer is a benchmark that
picked its datasets. Lenta is an ordinary retail campaign of an ordinary size, and the honest
answer on it is that uplift modelling buys nothing anyone could defend.

**Why it gets no PEHE or calibration.** Its scores are predicted outcomes, on the outcome's
scale, not effects. Running them through PEHE produces a large number that reads like a bad
error score and is really a units mismatch, and asking whether a risk score is the right size
to be a treatment effect is not a question about the score. The benchmark leaves those cells
empty (`estimates_effect = False`) rather than printing something meaningless.

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

## Calibration: ranking well and forecasting well are different jobs

Qini, AUUC and uplift at k are all invariant to multiplying every prediction by a constant.
An estimator that ranks perfectly and predicts effects three times too large scores
identically to one that gets the magnitudes right, and will then forecast three times the
return on the programme. The calibration columns are the only ones in the table that notice.

Two numbers, because they fail differently. The slope is the regression of realised uplift on
predicted uplift across deciles, so 1.0 is honest, below 1 means the predictions are more
spread out than reality and above 1 means they are more compressed. The error is the mean
absolute gap in the outcome's own units, which unlike the slope does see a constant offset.

On ACIC 2016, where the test split is large enough for ten deciles to mean something:

| Estimator | Calibration slope | Calibration error | PEHE |
|---|---|---|---|
| `s-learner` | 1.53 (1.15, 1.88) | 1.31 | 1.72 |
| `t-learner` | 1.08 (0.84, 1.32) | 1.11 | 1.22 |
| `x-learner` | 1.11 (0.85, 1.36) | 1.03 | 0.83 |
| `dr-learner` | 1.01 (0.79, 1.24) | 1.17 | 1.79 |
| `r-learner` | 1.07 (0.82, 1.32) | 0.95 | 1.23 |

The S-learner is the one to look at. Its slope of 1.53 is the only one whose interval
excludes 1, and it means the predictions are too compressed: realised uplift varies half as
much again as the model says it does. That is the shrinkage this file has described twice
already, now visible as a number rather than as an argument.

**And on Hillstrom the same shrinkage makes it the best calibrated of the five.**

| Estimator | Calibration slope, Hillstrom | Calibration slope, ACIC |
|---|---|---|
| `s-learner` | **0.69** (0.33, 1.04) | **1.53** (1.15, 1.88) |
| `t-learner` | 0.30 (0.09, 0.52) | 1.08 (0.84, 1.32) |
| `x-learner` | 0.35 (0.10, 0.59) | 1.11 (0.85, 1.36) |
| `dr-learner` | 0.26 (0.06, 0.46) | 1.01 (0.79, 1.24) |
| `r-learner` | 0.24 (0.05, 0.42) | 1.07 (0.82, 1.32) |

Every estimator is under 1 on Hillstrom and around or above 1 on ACIC, and the S-learner is
furthest from the pack in both directions. Hillstrom's real effect is small and fairly
uniform, so a flexible model finds heterogeneity that is mostly noise and over-spreads its
predictions; ACIC's effect genuinely varies more than it averages, so shrinking toward a
constant costs you. The S-learner shrinks hardest. That is one mechanism producing opposite
verdicts on two datasets, which is the same lesson as the regime table at the top of this
file arriving through a different column, and it is the reason the honest answer to "which
estimator should I use" starts with a question about the data.

A practical consequence worth stating for anyone reading the table to plan a budget: on
Hillstrom, a forecast built on any of these models' predicted decile spreads would be too
optimistic about the difference between the best and worst deciles, by a factor of between
one and a half and four.

**The bin count is capped by the smaller arm.** Ten deciles is the convention and IHDP cannot
support it: a 150-row test split with 28 treated units gives under three treated per decile,
and most deciles then contain none at all and can report nothing. The count is capped so a
bin holds at least ten of each arm, which on IHDP means two bins and a calibration estimate
too weak to lean on. That is reported as two bins rather than as ten bins mostly full of
missing values.

---

## Tuning, and what the grid selection is worth

Every estimator gets the same eight-candidate grid over `min_child_samples` and `num_leaves`,
selected on the validation split by Qini, never on test. `itx/bench/grid.py` sets out why
selecting on a metric this repository calls a poor referee is defensible: inside selection
every candidate is the same model class, so nothing can win by being a different kind of
model that games the curve. Across estimators, where that protection disappears, the
comparison is made on the test split against both baselines.

**Candidates a dataset cannot fit are removed first.** Week 3 added a leaf size of 200 for
the DR-learner's benefit and immediately broke the S-learner on IHDP: one seed in five
selected it, 200 leaves room for two leaves in 448 training rows, and the resulting
near-degenerate fit dragged the five-seed PEHE from 0.57 to 1.28. A configuration that cannot
fit a model is not a hyperparameter choice, so the grid is filtered by training size before
selection runs, at four leaves' worth of rows. The grid stays identical across estimators,
which is what the fairness rule requires; what rules a candidate out is the dataset.

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

**And on Hillstrom it chooses less consistently than the paragraph above claims.** That
paragraph was written in week 2 from the modal configuration per estimator, which is a weak
way to look at stability, and week 4 measured it properly while pricing something else.

The something else was a protocol change. Selection is nine fits per estimator per seed,
eight candidates and the winner, so it is about 89% of a benchmark's cost, and on Lenta's
412,217 training rows that extrapolated to between nineteen and thirty-two hours for one
dataset. So candidates are now fitted on at most 50,000 rows and only the winner sees the
whole training split (PLAN.md change 30). The justification offered for that was that
selection is a coarse decision: ranking eight settings of leaf size and tree width should
stabilise long before accuracy does.

**That justification was wrong, and the measurement says so.** Forcing comparable reductions
on the two datasets where both sides can be run:

| Dataset | Training rows | Reduction | Selections identical |
|---|---|---|---|
| ACIC 2016 | 2,881 | 5.8x | 1 of 25 |
| ACIC 2016 | 2,881 | 8.2x | 1 of 25 |
| Hillstrom | 25,615 | 5.1x | 5 of 25 |
| Hillstrom | 25,615 | 8.5x | 1 of 25 |
| Hillstrom | 25,615 | 17.1x | 4 of 25 |

Twenty-five selections is five estimators over five seeds. Capping changes nearly all of
them. Nothing about the choice is stable.

**What the cap costs, though, is nothing measurable**, and that is the number that decides
whether it survives. Taking the twenty Hillstrom disagreements at 5.1x, fitting *both*
configurations on the full training split and scoring both on the held-out test split:

| | Change in test Qini |
|---|---|
| mean | -0.00015 |
| median | -0.00016 |
| worst case | -0.00078 |
| best case | +0.00049 |
| capped choice better | 9 of 20 |
| capped choice worse | 11 of 20 |

Against a Hillstrom test Qini that runs 0.0028 to 0.0041 with 95% intervals roughly 0.004
wide, a mean of -0.00015 and a nine-to-eleven split is a coin flip inside the noise.

So the cap is defensible, but not for the reason it was introduced with. The selection is
not stable and does not become stable with more data; it is choosing among configurations
that are near-ties on the test split, and any perturbation reshuffles a choice that was
never load-bearing. That is a fact about this grid on this data, not about capping.

Two things follow, and both are uncomfortable enough to be worth stating. The first is that
the "Selected on the validation split from the committed grid" block under each README table
is largely reporting noise: it is an honest record of what the code did, and it should not be
read as a finding about which settings suit which estimator. The second is that the tuning
step buys very little here at all. Week 2 already measured the S-learner's Hillstrom Qini
moving from 0.0040 untuned to 0.0042 tuned. A reader entitled to ask why the project spends
89% of its compute on selection would be asking a fair question, and the answer for now is
that the protocol in PLAN.md section 4 commits to it and changing what is measured mid-build
is worse than paying for it. Whether the grid earns its place is a question for week 8, and
`docs/rejected.md` is where it will be answered either way.

---

## Still to come

Dragonnet (week 6), causal forest if the schedule allows. Approaches considered and left as
literature rather than code, per PLAN.md section 2: TARNet, CEVAE, and the
class-transformation method.
