# ACIC 2016, semi-synthetic

The second ground-truth dataset. It is here because it disagrees with the first one in the
ways that matter: six times larger than IHDP, far more heterogeneous, and confounded hard
enough that the naive comparison is off by more than half the true effect.

| | |
|---|---|
| **Source** | <https://github.com/BiomedSciAI/causallib/tree/master/causallib/datasets/data/acic_challenge_2016> |
| **Origin** | The causal inference competition at the 2016 Atlantic Causal Inference Conference. Covariates are real, from a study of twins linked to birth records; treatment and outcome are simulated by the organisers. Dorie, Hill, Shalit, Scott and Cervone, "Automated versus do-it-yourself methods for causal inference", *Statistical Science* 34(1), 2019. |
| **Licence** | Community Data License Agreement - Sharing, Version 1.0. Not redistributed here: the loader downloads it. |
| **Size** | 4,802 units, 58 covariates, 10 simulation settings. 3.4 MB in total. |
| **Loader** | `itx.data.acic.load_acic`, `itx.data.acic.load_acic_replicates` |
| **Ground truth** | Yes. `true_effect` is `mu1 - mu0`, the difference of the noiseless response surfaces. |

## Why both this and IHDP

Two simulated datasets look like redundancy until you compare them.

| | IHDP | ACIC 2016 |
|---|---|---|
| Units | 747 | 4,802 |
| Covariates | 25 | 58, three of them categorical |
| Treated share | 18.6% | 17.9% |
| True average effect | 4.02 | 2.13 (replicate 1; the ten range from 1.5 to 4.8) |
| Effect standard deviation | 0.86 | 3.98 |
| Effect spread relative to its mean | 0.21 | 1.87 |
| Naive difference in arm means | 4.02, close to the truth by luck | 3.58 against a truth of 2.13 |

IHDP is small and its effect is nearly constant: 91% of its units sit within 1.0 of the
mean effect. An estimator can score well there by getting the average right and the
heterogeneity roughly zero. ACIC will not let it: the effect varies more than it averages,
so the ranking has to be real. And IHDP's naive comparison happens to land near the truth,
which makes it a poor test of whether an estimator is doing anything about confounding;
ACIC's is out by 68%.

`docs/estimators.md` shows the two disagreeing about which meta-learner is best, which is
the point of having both.

## Layout

`x.csv` holds the covariates, shared by every replicate, and the loader reads and encodes it
once. `zymu_N.csv` holds one simulation setting: the treatment `z`, both noisy potential
outcomes `y0` and `y1`, and both noiseless response surfaces `mu0` and `mu1`.

Two decisions worth stating.

**The observed outcome is assembled, not read.** Both potential outcomes are in the file, so
the loader takes `y1` for the treated units and `y0` for the control ones. Handing an
estimator the counterfactual it is supposed to be predicting would be a spectacular way to
report perfect scores.

**The truth is `mu1 - mu0`, not `y1 - y0`.** The response surfaces are noiseless and the
potential outcomes are not. Differencing the noisy pair would add the outcome noise to the
target twice over and inflate every estimator's PEHE by a constant that has nothing to do
with the estimator.

Each replicate is a different simulation setting rather than another draw from one, so the
ten are ten different problems over the same people, and averaging PEHE across them is
averaging across problem types.

## Known quirks

- **The overlap is poor, and that is the interesting part.** A LightGBM propensity model
  puts about 48% of the training rows outside `[0.01, 0.99]`. Some of that is real: the
  organisers simulated assignment from the covariates and did not have to keep it gentle.
  Some of it is the propensity model being too confident, which is why
  `itx.estimators.propensity` predicts training rows out of fold. The gap between the two
  is where the R-learner went from a PEHE of 24 to a PEHE of 1.2; see `docs/estimators.md`.
- **Three covariates are letters.** `x_2`, `x_21` and `x_24` are published as A/B/C. They are
  encoded to integer codes in sorted order and declared to LightGBM as categorical, so the
  encoding does not depend on row order and no model reads them as quantities.
- **This is 10 settings, not the competition's full set.** The competition distributed 77
  settings through an R package; the README of this mirror mentions 20 conditions and the
  folder carries 10. That is enough to average over problem types and it is not the full
  competition. PLAN.md section 3 said 77 settings; the change and the reason are recorded in
  its section 11.
- **The covariate names carry no meaning.** Columns are `x_1` to `x_58` with no dictionary,
  so nothing here can be interpreted substantively, only predictively.

## Splits

60/20/20, stratified on the treatment arm, five committed seeds (11, 23, 37, 53, 71).
Metrics are computed on the test part only, which is 961 rows.
