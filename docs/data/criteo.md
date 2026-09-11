# Criteo-UPLIFT v2.1

The large one: 13.98 million randomised ad impressions, published by Criteo for uplift
research. It is here to answer a question the other four datasets cannot, which is whether
any of this survives contact with a realistic volume of data and a realistic event rate.

| | |
|---|---|
| **Source** | <http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz> |
| **Published** | 2018, revised 2019 (v2.1), by Criteo AI Lab. Diemert, Betlei, Renaudin and Amini, *A Large Scale Benchmark for Uplift Modeling*, AdKDD 2018 |
| **Licence** | CC BY-NC-SA 4.0. Non-commercial, attribution, share-alike. Not redistributed here: the loader downloads it from Criteo. |
| **Size** | 13,979,592 rows, 16 columns, 297 MB compressed and about 1.7 GB in memory |
| **SHA-256** | `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc` |
| **Loader** | `itx.data.criteo.load_criteo` |
| **Ground truth** | None. The individual effect is not recorded and cannot be. |

## The experiment

Users were randomised into treatment with probability 0.85 and the remaining 15% held back
from the advertising campaign. Visits and conversions were recorded over a two-week window.

The treated share in the file is 0.85000013, which is 0.85 plus rounding at this row count,
so the loader records the propensity as known by design rather than estimating it.

That decision is checked rather than assumed. Fitting the project's own out-of-fold
propensity model to the 10% subsample, with the design value hidden, gives a distribution
tightly concentrated on 0.85: mean 0.8500, middle 90% inside [0.842, 0.865], and 1.2% of
rows outside [0.80, 0.90]. The raw range is 0.42 to 0.99 and four rows in 1.4 million touch
the clip. There is no overlap problem and no meaningful departure from the design.

## Fields

| Column | Type | Notes |
|---|---|---|
| `f0` ... `f11` | numeric | Twelve dense features. Criteo publishes no meaning for any of them, so nothing here can be a story about which covariate mattered |
| `treatment` | 0/1 | Assigned to the campaign |
| `visit` | 0/1 | Visited the advertiser's site |
| `conversion` | 0/1 | Converted |
| `exposure` | 0/1 | **Not used.** See below |

No categorical columns: all twelve features are continuous, and
`UpliftDataset.categorical` is empty.

## `exposure` is dropped, and why that matters

`exposure` records whether the user was actually shown an ad, as opposed to being assigned
to the campaign. Zero of the 2,096,937 control rows have it set, because a control user
cannot be shown an ad that was never served.

It is therefore a consequence of the treatment, not a covariate. Putting it in the feature
matrix would hand every estimator a copy of the treatment indicator under another name, and
every uplift in the results table would be an artefact of that. It is dropped.

Criteo published the column deliberately: the dataset also supports a one-sided
noncompliance question, where the estimand is the effect of *being exposed* rather than the
effect of *being assigned*. That is a genuinely different quantity and recovering it needs
an instrumental-variable argument. This project asks the assignment question, which is the
one the randomisation answers directly, and does not make that argument.

## Outcomes

| Outcome | Mean, treated | Mean, control | Difference |
|---|---|---|---|
| `visit` (default) | 0.048543 | 0.038201 | 0.010342 |
| `conversion` | 0.003089 | 0.001938 | 0.001152 |

Measured over all 13,979,592 rows. `visit` is the headline outcome. `conversion` happens
29 times in ten thousand rows, and even at this size its Qini interval is wide enough to be
worth reporting separately rather than beside the visit numbers.

## The 10% subsample

The benchmark runs on a fixed 10% stratified subsample: 1,397,958 rows, drawn once with
seed 20260928 and cached as Parquet under the gitignored data directory. Both the fraction
and the seed are protocol, not tuning knobs, and changing either changes every Criteo number
in the results table.

**Why subsample at all.** Not memory. PLAN.md originally said "Memory budget: 8 GB", written
before any code existed, and it was wrong: 13.98M rows by 16 columns is about 1.7 GB held as
float64, and the 297 MB figure is compressed text. The binding constraint is wall-clock time.
Five estimators over an eight-candidate grid and five seeds on 8.4M training rows, with the
DR and R learners cross-fitting inside each fit, is hours; Hillstrom's 25,615 training rows
is already twenty minutes. PLAN.md change 23 records the correction.

**Why stratified rather than random.** The arms are 85/15 and a conversion happens 29 times
in ten thousand rows. Under a plain sample the smallest cell of interest, converting
controls, would be a few hundred rows and would move visibly with the seed. Stratifying on
treatment crossed with both outcomes holds all eight cells at exactly a tenth of themselves.
It works: the subsample's treated share is 0.8500005 against the file's 0.8500001, and its
visit rate is 0.046991 against the file's 0.046992.

**How it is read.** Two lazy passes over the compressed file. The first reads three integer
columns and decides which rows survive; the second reads the full width for those rows only,
so peak memory is the size of the subsample rather than the size of the file. It takes about
three minutes once, and under two seconds on every later call from the Parquet cache.

`criteo-full` is the same loader at `fraction=1.0` and is the headline single fit. It is a
separate dataset key rather than a flag so that a results table always says which of the two
produced it.

## Known quirks

- **The features are not perfectly balanced**, despite the randomisation. The worst
  standardised mean difference between arms is 0.047 on `f3`, against a conventional
  threshold of 0.1. At 1.4M rows that is about twenty standard errors, so it is real rather
  than noise, but it is small, and the propensity check above shows it does not move the
  assignment probability far from 0.85. Anonymous features mean there is no way to say more
  about where it comes from.
- **The arms are 85/15.** Randomisation makes assignment ignorable regardless, but the
  control arm is a seventh of the data and it is the control arm that sets the precision of
  an uplift. This is the opposite of the usual worry and it is worth keeping in mind when
  reading an interval.
- **Duplicate feature rows exist.** With twelve features and 14 million rows, identical
  feature vectors with different outcomes are common. This is expected for anonymised,
  quantised features and is not an error, but it does mean the feature space is coarser
  than the row count suggests.
- **Nothing is named.** No feature-importance story here can be interpreted, and none is
  told.

## Splits

The committed protocol (PLAN.md section 4): 60/20/20, stratified on treatment arm crossed
with the binary outcome, five seeds (11, 23, 37, 53, 71), applied to the subsample. Metrics
are computed on the test part only. Bootstrap resamples are 200 rather than 1,000 on this
dataset, per section 4.
