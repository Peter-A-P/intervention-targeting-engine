# Rejected: per-estimator hyperparameter selection as part of the protocol

PLAN.md Rule C asks for one approach tried and rejected, with evidence. This is it. The
approach is the validation-split grid search that every LightGBM estimator in this repository
went through on every seed of every dataset: eight candidates over `min_child_samples` and
`num_leaves`, selected by Qini on the validation split, the winner refitted on the full
training split. It was in the protocol from week 2 (PLAN.md change 9), it cost about 89% of
every benchmark's compute, and four separate measurements over six weeks say it chose among
near-ties and changed nothing a reader would act on.

The decision it led to is at the end. The evidence comes first, in the order it was gathered,
because the order matters: each measurement was taken for a different reason, and each one
ended up saying the same thing.

## What was expected of it

Section 4 of PLAN.md committed to tuning on the validation split with a small grid, committed
and identical across estimators, so that a difference between two estimators could not be a
difference in how carefully each had been tuned. That is a fairness argument and it is a good
one. The expectation that went with it was that selection would also *help*: that a
DR-learner and an S-learner want different leaf sizes, that a 150-row IHDP split wants a
different tree from a 400,000-row Lenta split, and that letting the data choose would move
the numbers by more than the noise.

The fairness argument survives. The expectation did not.

## Measurement 1, week 2: it moves Hillstrom by a tenth of an interval

The S-learner's Hillstrom Qini went from 0.0040 untuned to 0.0042 tuned, against a bootstrap
interval roughly plus or minus 0.002 wide. The choices looked reasonably stable when read as
the modal configuration per estimator: three of five seeds for the S-learner, T-learner and
outcome ranking. That reading was too generous, as measurement 2 found while pricing
something else, but the size of the effect was right the first time: a tenth of an interval.

## Measurement 2, week 4: capping the rows it sees changes nearly every choice and costs nothing

Selection is nine fits per estimator per seed, and on Lenta's 412,217 training rows that
extrapolated to between nineteen and thirty-two hours for one dataset, so candidates were
capped at 50,000 training rows and only the winner saw the whole split (PLAN.md change 30).
The justification offered was that selecting among eight leaf sizes is a coarse decision that
should stabilise long before accuracy does. Forcing comparable reductions where both sides
could be run:

| Dataset | Training rows | Reduction | Selections identical |
|---|---|---|---|
| ACIC 2016 | 2,881 | 5.8x | 1 of 25 |
| ACIC 2016 | 2,881 | 8.2x | 1 of 25 |
| Hillstrom | 25,615 | 5.1x | 5 of 25 |
| Hillstrom | 25,615 | 8.5x | 1 of 25 |
| Hillstrom | 25,615 | 17.1x | 4 of 25 |

The justification was wrong: the selection is not stable under the cap. What decided the
cap's fate was the other half of the measurement. Taking the twenty Hillstrom disagreements
at 5.1x, fitting both configurations on the full training split and scoring both on test:

| Change in test Qini, capped choice against uncapped | |
|---|---|
| mean | -0.00015 |
| median | -0.00016 |
| worst case | -0.00078 |
| best case | +0.00049 |
| capped choice better | 9 of 20 |
| capped choice worse | 11 of 20 |

Against a Hillstrom test Qini of 0.0028 to 0.0041 with intervals about 0.004 wide, a
nine-to-eleven split with a mean of -0.00015 is a coin flip inside the noise. The cap stayed,
on measured grounds rather than the reasoned ones it arrived with, and the measurement left a
sharper question behind: if any perturbation reshuffles the choice and the reshuffle costs
nothing, the choice was never load-bearing.

## Measurement 3, week 6: a different selection rule changes 87% of the choices and costs nothing

Week 5 added the rule the Qini rule was always measured against: score each candidate on
what its ranking would buy at the operating budget, doubly robust, on the validation split
(PLAN.md change 41). On ACIC, across six estimators and five seeds, the two rules agreed on
4 of 30 selections. All 26 disagreements were then fitted both ways on the full training
split and scored on test, 2h40m of compute (change 49):

| Change from the policy rule instead of the Qini rule | mean | median | worst | best | policy rule better |
|---|---|---|---|---|---|
| True gain at 20%, from ACIC's own effects | -0.0011 | +0.0041 | -0.0984 | +0.0484 | 15 of 26 |
| DR gain at 20% | +0.0274 | +0.0026 | -0.1752 | +0.5157 | 14 of 26 |
| Qini | -0.0016 | -0.0011 | -0.0282 | +0.0148 | 11 of 26 |

Bootstrapped over the 26 cases, every mean covers zero: true gain (-0.0137, +0.0099), DR gain
(-0.0241, +0.0856), Qini (-0.0051, +0.0017), against a DR gain that sits around 1.0 with an
interval 0.62 wide. ACIC is simulated, so the deciding row is the true gain rather than an
estimate of it, which matters because the rule under test selects on the DR estimate and
scoring it on that alone would be marking its own homework. A partial pass from seven cases
of one seed had pointed the other way and was not published; finishing it refuted the hunch.

Three measurements from three directions, then: cap the rows and nearly every choice moves;
change the rule and nearly every choice moves; the choices land somewhere different on nearly
every partition of the same dataset. And each time the move costs nothing measurable.

## Measurement 4, week 8: the whole step against no step at all

The three measurements above compared one way of selecting with another. None of them asked
the plainer question, which is what the tables look like with no selection: every estimator
at the default configuration, the same one for all of them, which keeps the fairness argument
intact and spends one fit per estimator per seed instead of nine. That is the measurement a
reader who has read this far is entitled to, and it was the one deferred to week 8 twice
(PLAN.md changes 30 and 49) on the grounds that changing what is measured mid-build is worse
than paying for it.

`uv run itx benchmark --dataset <name> --no-tune` on IHDP, ACIC 2016 and Hillstrom,
2026-09-14, results written outside the repository and compared with the committed tuned
tables. Three headline metrics per estimator, Qini, uplift in the top 20% and doubly robust
gain at 20%, each as the five-seed mean; the question for each is whether the untuned mean
lands inside the committed tuned interval.

| Dataset | Untuned means inside the tuned interval | Estimator order by Qini | by uplift@20% | by DR gain@20% |
|---|---|---|---|---|
| IHDP | 18 of 18 | changed | changed | changed |
| ACIC 2016 | 18 of 18 | changed | changed | changed |
| Hillstrom | 18 of 18 | same | same | same |

Fifty-four of fifty-four. The two rows the grid never touched, Dragonnet and random
targeting, reproduced to every decimal in both runs, which is the check that the only thing
that moved between the runs was the tuning.

The largest single move anywhere is the DR-learner's Qini on ACIC, 0.2401 tuned to 0.1862
untuned, inside an interval of (0.1765, 0.3085). That is the estimator the grid was extended
for: the leaf size of 200 was added in week 3 for the DR-learner's benefit (PLAN.md change
16), and it is the one estimator that visibly used it. Tuned, the DR-learner has the best Qini
on ACIC; untuned, the worst of the five. Both positions are inside every other estimator's
interval. On Hillstrom nothing moves by more than 0.0005 on Qini or 0.004 on uplift, against
intervals 0.004 and 0.07 wide, and the order of all six rankings is identical under both
protocols on all three metrics.

The orderings on IHDP and ACIC do change, and that is the point rather than a caveat. Those
are the two datasets where the estimators' intervals overlap almost entirely, so any
perturbation reorders them; the grid is one such perturbation, and the order it produced is
not more meaningful than the order without it. Hillstrom, where the S-learner leads on the
point estimate though inside the others' intervals, keeps its order under both.

What the step costs is easiest to read off the fraud case, where the benchmark log records
the wall time of each estimator's selection-plus-refit and the results file records the
refit alone: the S-learner's 3m46s on seed 11 contained a 14.6-second refit, so selection
was 94% of it, and the DR-learner's 14m33s contained 143 seconds, 84%. On the small datasets
the wall clock is dominated by the bootstrap rather than the fits, so the saving there is in
compute rather than in minutes waited; on Lenta, Criteo and the fraud case, where fits are the
cost, it is most of the run.

## The decision

**The grid does not earn its place, and this repository says so rather than removing it.**

Four measurements, two datasets with a written truth among them, one clean control, and not
one number moved outside its interval. The per-seed selection blocks under each README table
are an honest record of what the code did and a finding about nothing; `docs/estimators.md`
has said so since week 4 and this document is where it becomes the conclusion.

What follows from it, and what does not:

- **The committed tables keep the protocol they were measured under.** PLAN.md section 4 fixed
  the protocol before any number was seen, and changing it after seeing the numbers, even to
  remove a step shown not to matter, is exactly the move this repository exists to argue
  against. Every table in the README was produced with `--tune`, the default, and the
  clean-environment rerun reproduces them with it. A reader can regenerate any table either
  way and get the same conclusions.
- **Anyone reusing this on their own data should pass `--no-tune`.** It keeps the fairness
  argument, one configuration for every estimator, at one ninth of the fitting cost, and on
  every dataset here it would have given the same answer. `README.md` says so under "Install
  and run".
- **The selection blocks under the tables are to be read as provenance, not as findings.**
  Which configuration won on which seed says which way the noise in a validation Qini broke.
- **A future version that re-measures everything drops the step from the protocol.** It is not
  dropped in this one, because a protocol changed at the end of a build to match its results
  is a worse thing to ship than a step that cost compute and changed nothing.

## What this is not

It is not a claim that hyperparameter selection is worthless for uplift models. It is a claim
about this grid, on these six datasets, with LightGBM as the one base learner, under a
protocol that already fixes the learner, the features and the splits. Eight settings of leaf
size and tree width turned out to be near-ties on every dataset that could be measured, and
the interval on the metric was wider than anything the choice among them moved. A wider grid,
a different learner, or a dataset where the estimators are not already close might come out
differently, and the machinery to find out is still here: `uv run itx benchmark --tune`.

It is also not the failed prediction that section 9 of PLAN.md listed as its second Rule C
candidate. That prediction was that a shared tree model would swallow the treatment indicator
and the S-learner's uplift would collapse toward zero on Hillstrom. It did not: on Hillstrom
the S-learner's top 20% is worth 0.0828 against random targeting's 0.0451, its realised value
at that budget, 0.0168, is the best of any estimator here, and its Qini of 0.0041 is second
only to Dragonnet's 0.0042. LightGBM found the treatment indicator perfectly well in 42,693
rows. A prediction that fails is worth recording, and `docs/estimators.md` records the
mechanism under the S-learner, but it is not an approach that was tried and set aside. The
grid is.
