# IHDP (Infant Health and Development Program), semi-synthetic

Real covariates, simulated outcomes, and therefore a known individual treatment effect for
every unit. That is the whole reason it is in the benchmark: it is one of only two
datasets here where an estimator can be shown to be **correct** rather than merely
self-consistent.

| | |
|---|---|
| **Source** | <https://www.fredjo.com/files/ihdp_npci_1-100.train.npz> and `ihdp_npci_1-100.test.npz` |
| **Origin** | Covariates from the Infant Health and Development Program, a 1985 randomised study of home visits to low-birth-weight infants. The simulation is Hill (2011), setting B; the file is the `ihdp_npci_1-100` benchmark distributed by the CFRNet and CEVAE authors and used by Dragonnet. |
| **Licence** | Public benchmark file, redistributed by the CFR/CEVAE authors. Not redistributed here: the loader downloads it. |
| **Size** | 747 units, 25 covariates, 100 replicates. 16.1 MB and 1.8 MB. |
| **SHA-256** | train `750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240`, test `a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c` |
| **Loader** | `itx.data.ihdp.load_ihdp`, `itx.data.ihdp.load_ihdp_replicates` |
| **Ground truth** | Yes. `true_effect` is `mu1 - mu0`, the difference of the two simulated response surfaces. |

## What is real and what is not

The covariates are real: 25 measurements on the child and the mother, six continuous and
nineteen indicators (one of them, x14, coded 1 and 2 rather than 0 and 1). The outcome is simulated from a published response surface, so both
potential outcomes exist for every unit and the individual effect is known exactly.

The confounding is deliberate and it is the interesting part. The original programme
randomised its home visits, which would make the targeting problem easy. Hill's
construction removes a non-random subset of the treated units, specifically children of
non-white mothers, so that treatment assignment becomes correlated with the covariates.
An estimator that ignores that correlation gets the wrong answer, and the difference in
arm means is badly biased: on replicate 0 it is about 6.43 against 2.41, an apparent
effect of 4.0 that happens to be near the truth by coincidence rather than by design, and
on other replicates it is not.

`propensity` is therefore left as None. It is not known, and the loader will not invent it.

## Layout

The file ships as a pre-split pair of archives, 672 training and 75 test units. The loader
rejoins them into all 747 and then applies this project's own committed split, because the
evaluation protocol has to be the same across every dataset in the table (PLAN.md section
4). Anyone comparing these PEHE numbers against a paper should know that: the papers
report on the shipped 672/75 split, which is a different test set from this one.

Each of the 100 replicates is a fresh draw of the simulated outcome on the same
covariates. Replicate 0 is what the benchmark runs today; averaging over all 100 with an
interval across replicates arrives with the ground-truth metrics in week 3.

## Reference values

| Quantity | Value |
|---|---|
| Units | 747 (672 + 75) |
| Treated share | 18.6% |
| True average effect, replicate 0 | 4.016 |
| Naive difference in arm means, replicate 0 | 4.021 |

The published average effect for this benchmark is about 4, and the loader test asserts it
to within 0.2.

## Known quirks

- **It is small.** 747 units, 139 of them treated, and a training split of 448. Settings
  that are merely cautious on a 64,000-row dataset are fatal here: a `min_child_samples`
  of 100 stops LightGBM splitting at all, and the S-learner then returns exactly zero
  uplift for every unit with a PEHE identical to predicting nothing. That failure is
  silent, which is why `BaseUpliftEstimator` now warns on it. See `docs/estimators.md`.
- **It is one simulation, not the world.** The response surface is a specific published
  choice. Good PEHE on IHDP means an estimator handles that surface, not that it handles
  yours. It is a necessary check, not a sufficient one.
- **The literature reports several different IHDP variants** (setting A, setting B, 100 or
  1000 replicates, in-sample or out-of-sample PEHE). Numbers are only comparable within a
  variant. This is `ihdp_npci_1-100`, and PEHE here is computed on a held-out split.
- **The covariate names are not in the file.** Columns are `x1` to `x25` in file order.
  The first six are continuous, the rest indicators; x14 is coded 1 and 2.

## Splits

60/20/20, stratified on the treatment arm (the outcome is continuous, so there is no
outcome stratum), five seeds (11, 23, 37, 53, 71). At 747 units the test part is 150 rows,
so intervals are wide, and they are supposed to be.
