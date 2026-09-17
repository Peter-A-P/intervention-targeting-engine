# Intervention Targeting Engine

Tells an organisation which customers, cases or patients to spend a limited intervention
budget on: only the ones whose outcome the intervention actually changes. Retention
offers, outreach, fraud review, clinical follow-up: a large share of every such budget
goes to people who would have behaved the same way regardless, and this finds them, and
shows the intervention list changing as the budget moves.

**Status: week 8 of 8.** Seven estimators benchmarked on all five datasets, each reported with
what its ranking actually buys at a budget rather than only how well it ranks, plus a
sensitivity section, a fraud worked case that declares itself semi-synthetic in its first
sentence, and a budget-slider demo live at [targeting.peterparker.ca](https://targeting.peterparker.ca). The
numbers below are real and reproducible. Still to come: the clean-environment rerun and the
flip to public. Build plan: [PLAN.md](PLAN.md).

**The same baseline, the same code, opposite conclusions.** Ranking people by risk is how this
job is usually done. At a budget covering a tenth of the population it buys **-0.20 on ACIC**
and **+0.0055 on Criteo**, against random targeting's +0.26 and +0.0008. On ACIC it is worse
than spending nothing at all, interval entirely below zero. On Criteo it matches every uplift
model in this repository and beats random targeting sevenfold. On Hillstrom it is worth nothing
over a coin flip at a tight budget and recovers about a third of the gap by a wide one. On
Lenta nothing separates from random, including the uplift models.

So risk ranking is not a trap and it is not safe: it is one or the other depending on the data,
the difference is worth a change of sign, and nobody can guess which case they are in. Anybody
can measure it. `uv run itx diagnose --dataset <name>` decides it for the price of one outcome
model, before any of the rest of this is worth running.

## Results

Every table on this page is written by `uv run itx benchmark --dataset <name>` and is
never edited by hand. Each number is the mean across five committed split seeds. The interval
beside it is the mean of the five per-seed 95% bootstrap intervals: it shows the sampling
uncertainty of a typical split and carries no across-seed variance, so "the interval excludes
zero" on this page means the typical split's interval does, and where the five seeds disagree
about that the text says so. The per-seed numbers are in `results/<dataset>.json`. The
`random-200` row is different again: its interval is the spread across 200 random rankings of
the fixed test split, not a bootstrap over rows, because what varies for random targeting is
the ranking. Hyperparameters are chosen per seed on a validation split from a grid that is
identical for every estimator, and the test split is never touched until the numbers below
are computed. There are no bare point estimates here by design: a Qini without an interval
is treated in this repository as a defect.

Two columns need a word on how to read them. **Normalised AUUC** divides the area under a
ranking's uplift curve by the area under the curve of ranking on the true outcome; that
reference is not a ceiling, because a ranking can beat it, and random targeting lands anywhere
from 0.04 on Lenta to 0.83 on IHDP under it, so read the column against the random row rather
than against 1. **Calibration error** is a mean absolute gap across bins, which cannot reach
zero on finite data and whose bootstrap interval inherits the same floor, so it is an upper
bound on miscalibration to compare across estimators, not a quantity to read against zero.

Each dataset carries two tables. The first is the ranking metrics: how good the ordering is.
The second is **what each ranking buys**, which is the question a budget holder is actually
asking. Its numbers are the extra outcome per head of the whole population from treating the
top b% rather than treating nobody, estimated two ways: by inverse-probability weighting,
which relies only on knowing how treatment was assigned, and by a doubly robust estimator,
which adds outcome models and survives either one of the two being wrong. Both are reported
because a disagreement between them is worth seeing. On a binary outcome a gain of 0.017
means seventeen extra events per thousand people in the population. At a budget of 100% the
doubly robust number is the average treatment effect; the IPW number is its Horvitz-Thompson
form, which equals the difference in arm means only when the realised treated share matches
the design propensity, and the Criteo section below is about the gap when it does not.

### Hillstrom, 42,693 customers, randomised email campaign

<!-- itx:table:hillstrom -->
| Estimator | Qini (95% CI) | Normalised AUUC | uplift@10% | uplift@20% | uplift@30% | Calibration slope | Calibration error |
|---|---|---|---|---|---|---|---|
| `s-learner` | 0.0041 (0.0019, 0.0062) | 0.2609 (0.1984, 0.3210) | 0.0893 (0.0404, 0.1358) | 0.0828 (0.0494, 0.1160) | 0.0793 (0.0530, 0.1066) | 0.6926 (0.3295, 1.0444) | 0.0180 (0.0157, 0.0370) |
| `t-learner` | 0.0031 (0.0009, 0.0053) | 0.2442 (0.1803, 0.3047) | 0.0970 (0.0465, 0.1479) | 0.0779 (0.0441, 0.1120) | 0.0678 (0.0405, 0.0948) | 0.2987 (0.0901, 0.5153) | 0.0437 (0.0347, 0.0594) |
| `x-learner` | 0.0031 (0.0010, 0.0053) | 0.2452 (0.1827, 0.3065) | 0.0831 (0.0351, 0.1321) | 0.0788 (0.0448, 0.1124) | 0.0743 (0.0478, 0.1026) | 0.3477 (0.1011, 0.5944) | 0.0365 (0.0287, 0.0529) |
| `dr-learner` | 0.0028 (0.0007, 0.0051) | 0.2402 (0.1757, 0.3031) | 0.0841 (0.0339, 0.1344) | 0.0738 (0.0392, 0.1079) | 0.0736 (0.0463, 0.1014) | 0.2563 (0.0551, 0.4590) | 0.0463 (0.0377, 0.0625) |
| `r-learner` | 0.0029 (0.0007, 0.0051) | 0.2413 (0.1795, 0.3017) | 0.0920 (0.0417, 0.1434) | 0.0759 (0.0416, 0.1096) | 0.0745 (0.0473, 0.1012) | 0.2381 (0.0501, 0.4221) | 0.0508 (0.0419, 0.0667) |
| `dragonnet` | 0.0042 (0.0020, 0.0064) | 0.2629 (0.2020, 0.3225) | 0.0687 (0.0226, 0.1168) | 0.0765 (0.0435, 0.1089) | 0.0713 (0.0448, 0.0984) | 1.0648 (0.5185, 1.5984) | 0.0207 (0.0162, 0.0378) |
| `outcome-ranking` | 0.0012 (-0.0008, 0.0033) | 0.2129 (0.1431, 0.2796) | 0.0462 (-0.0070, 0.1011) | 0.0524 (0.0159, 0.0917) | 0.0578 (0.0288, 0.0882) | - | - |
| `random-200` | -0.0000 (-0.0019, 0.0020) | 0.1931 (0.1605, 0.2267) | 0.0450 (0.0047, 0.0846) | 0.0451 (0.0181, 0.0728) | 0.0451 (0.0243, 0.0656) | - | - |

Selected on the validation split from the committed grid:

- `s-learner`: min_child_samples=5, num_leaves=15 (3 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds)
- `t-learner`: min_child_samples=200, num_leaves=15 (2 of 5 seeds), min_child_samples=20, num_leaves=15 (2 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds)
- `x-learner`: min_child_samples=20, num_leaves=15 (2 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds)
- `dr-learner`: min_child_samples=60, num_leaves=15 (3 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds)
- `r-learner`: min_child_samples=60, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds)
- `outcome-ranking`: min_child_samples=200, num_leaves=31 (2 of 5 seeds), min_child_samples=20, num_leaves=15 (2 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
<!-- itx:end:hillstrom -->

**What each ranking buys.**

<!-- itx:table:hillstrom-policy -->
| Estimator | IPW gain at 10% | DR gain at 10% | IPW gain at 20% | DR gain at 20% | Treated share gap at 20% | IPW gain at 30% | DR gain at 30% |
|---|---|---|---|---|---|---|---|
| `s-learner` | 0.0096 (0.0043, 0.0147) | 0.0088 (0.0039, 0.0135) | 0.0178 (0.0106, 0.0252) | 0.0168 (0.0100, 0.0234) | 0.0108 (-0.0120, 0.0354) | 0.0265 (0.0177, 0.0356) | 0.0242 (0.0161, 0.0324) |
| `t-learner` | 0.0102 (0.0045, 0.0157) | 0.0096 (0.0045, 0.0147) | 0.0174 (0.0100, 0.0249) | 0.0155 (0.0087, 0.0224) | 0.0143 (-0.0094, 0.0379) | 0.0221 (0.0132, 0.0311) | 0.0201 (0.0119, 0.0282) |
| `x-learner` | 0.0084 (0.0030, 0.0137) | 0.0080 (0.0032, 0.0128) | 0.0174 (0.0100, 0.0249) | 0.0158 (0.0090, 0.0227) | 0.0129 (-0.0110, 0.0365) | 0.0244 (0.0157, 0.0338) | 0.0224 (0.0146, 0.0309) |
| `dr-learner` | 0.0089 (0.0034, 0.0145) | 0.0085 (0.0036, 0.0136) | 0.0158 (0.0082, 0.0234) | 0.0148 (0.0079, 0.0217) | 0.0080 (-0.0165, 0.0323) | 0.0239 (0.0149, 0.0330) | 0.0220 (0.0138, 0.0305) |
| `r-learner` | 0.0104 (0.0049, 0.0160) | 0.0092 (0.0041, 0.0144) | 0.0165 (0.0090, 0.0240) | 0.0151 (0.0084, 0.0221) | 0.0104 (-0.0134, 0.0355) | 0.0237 (0.0148, 0.0325) | 0.0221 (0.0138, 0.0304) |
| `dragonnet` | 0.0071 (0.0020, 0.0125) | 0.0070 (0.0024, 0.0117) | 0.0161 (0.0090, 0.0231) | 0.0154 (0.0088, 0.0220) | 0.0068 (-0.0160, 0.0316) | 0.0225 (0.0140, 0.0314) | 0.0215 (0.0137, 0.0295) |
| `outcome-ranking` | 0.0054 (-0.0006, 0.0116) | 0.0043 (-0.0011, 0.0099) | 0.0111 (0.0029, 0.0197) | 0.0103 (0.0028, 0.0183) | 0.0042 (-0.0187, 0.0286) | 0.0189 (0.0091, 0.0291) | 0.0171 (0.0083, 0.0263) |
| `random-200` | 0.0045 (0.0001, 0.0088) | 0.0044 (0.0004, 0.0085) | 0.0091 (0.0030, 0.0150) | 0.0088 (0.0034, 0.0144) | 0.0011 (-0.0177, 0.0219) | 0.0137 (0.0069, 0.0204) | 0.0132 (0.0068, 0.0193) |
<!-- itx:end:hillstrom-policy -->

At a budget covering a tenth of the customers, the outcome ranking buys 0.0043 extra visits
per customer in the population and random targeting buys 0.0044. On the tightest budget, the
ordinary way of doing this job is worth exactly nothing over a coin flip, while the five
LightGBM estimators buy about twice that, Dragonnet about 1.6 times, and all six exclude zero. The gap narrows as the budget widens:
measured as the share of the distance from random targeting to the best estimator, the
outcome ranking covers none of it at 10%, about a fifth at 20% and about a third at 30%. The
trap is worst exactly where budgets are tightest, which is where budgets usually are.

The two estimators agree to within 0.001 everywhere in this table. They should: Hillstrom is
randomised, its treatment probability is a design constant, and no unit is anywhere near the
clipping bound. Where they disagree, as on ACIC below, that is information about the data
rather than about the estimators.

The outcome ranking is the ordinary way this job is done: model who is likely to respond,
spend the budget from the top of that list. Its Qini interval contains zero, so on this
dataset it has not been shown to beat picking at random. All six modelled rankings' intervals
exclude zero.

At a budget covering 20% of the population, the S-learner's targeted group shows an
8.3-point difference in visit rate between arms, against random targeting's 4.5. The
outcome ranking manages 5.2, less than a quarter of the way from doing nothing clever to
doing this properly. The five estimators' intervals overlap heavily, so the honest reading
is that they are distinguishable from the two baselines and not from each other.

The calibration columns tell a different story from the ACIC ones below, and the difference
is the point. Every LightGBM estimator here has a slope well under 1, meaning its predictions
are more spread out than the uplift that actually materialises: on a dataset whose real effect
is small and fairly uniform, a flexible model finds heterogeneity that is mostly noise. The
S-learner is the least wrong of the five at 0.69, for the same reason it is the most wrong on
ACIC at 1.53. It shrinks predicted effects toward a constant, which is a liability where the
effect genuinely varies and a virtue where it does not. Dragonnet's slope is 1.06 (0.52, 1.60),
the closest to 1 in the table and the widest interval, so it is the one estimator here whose
predicted heterogeneity is not shown to be noise, and not shown to be real either.

![Qini curves on Hillstrom](docs/figures/qini-hillstrom.png)

### IHDP, 747 units, simulated outcomes with known individual effects

<!-- itx:table:ihdp -->
| Estimator | Qini (95% CI) | Normalised AUUC | uplift@10% | uplift@20% | uplift@30% | Calibration slope | Calibration error | PEHE | ATE error |
|---|---|---|---|---|---|---|---|---|---|
| `s-learner` | 0.0176 (-0.0699, 0.1079) | 0.9064 (0.8315, 0.9670) | 4.0150 (2.8356, 4.9348) | 4.2496 (2.9210, 5.7731) | 4.3617 (3.2989, 5.3901) | 0.8537 (-0.1313, 1.8405) | 0.3087 (0.1209, 0.8890) | 0.5661 (0.4344, 0.7022) | 0.1385 (0.0697, 0.2215) |
| `t-learner` | 0.0305 (-0.0578, 0.1222) | 0.9022 (0.8334, 0.9585) | 4.1149 (3.1768, 5.2630) | 4.2359 (3.4102, 5.4069) | 4.4967 (3.5423, 5.3866) | 0.4966 (-0.1301, 1.0832) | 0.6427 (0.2497, 1.1272) | 0.8569 (0.7500, 0.9646) | 0.1248 (0.0508, 0.2608) |
| `x-learner` | 0.0468 (-0.0462, 0.1433) | 0.8962 (0.8303, 0.9548) | 3.9407 (2.7052, 5.5056) | 4.2619 (3.3427, 5.1678) | 4.2332 (3.4652, 5.0388) | 1.9537 (-0.2795, 4.0129) | 0.3797 (0.1325, 1.0181) | 0.7870 (0.6081, 0.9695) | 0.1032 (0.0219, 0.2284) |
| `dr-learner` | -0.0004 (-0.0790, 0.0787) | 0.8104 (0.7258, 0.8873) | 3.3536 (1.9119, 5.2414) | 3.8849 (2.5805, 4.9767) | 3.7311 (2.7941, 4.6571) | -0.0022 (-0.1098, 0.0923) | 6.2535 (5.1738, 7.3768) | 8.5415 (7.1699, 9.9460) | 1.6629 (0.4919, 3.0153) |
| `r-learner` | 0.0021 (-0.0804, 0.0807) | 0.8878 (0.8067, 0.9464) | 4.1897 (3.1444, 5.2740) | 4.0266 (2.7901, 5.0829) | 4.0827 (3.0604, 5.0693) | 0.3630 (-0.2123, 0.8253) | 0.8393 (0.3955, 1.4655) | 1.3381 (1.1803, 1.5026) | 0.3380 (0.2001, 0.5365) |
| `dragonnet` | 0.0613 (-0.0106, 0.1380) | 0.8150 (0.7399, 0.8840) | 3.8501 (2.0733, 5.4645) | 3.8684 (2.6175, 5.1264) | 3.8825 (2.9698, 4.7932) | -0.2067 (-1.0650, 0.7451) | 0.8308 (0.4393, 1.4137) | 1.4047 (1.2060, 1.6127) | 0.5807 (0.3904, 0.7820) |
| `outcome-ranking` | 0.0088 (-0.0479, 0.0692) | 0.7345 (0.6413, 0.8244) | 1.3420 (-0.3277, 3.1934) | 2.2051 (0.9486, 3.5478) | 2.7399 (1.7099, 3.7437) | - | - | - | - |
| `random-200` | 0.0001 (-0.0751, 0.0712) | 0.8283 (0.7705, 0.8888) | 3.9298 (2.2726, 5.5296) | 3.9178 (2.8451, 5.0637) | 3.9275 (3.2068, 4.6875) | - | - | - | - |

Selected on the validation split from the committed grid:

- `s-learner`: min_child_samples=5, num_leaves=15 (2 of 5 seeds), min_child_samples=60, num_leaves=15 (2 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
- `t-learner`: min_child_samples=5, num_leaves=15 (2 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds), min_child_samples=20, num_leaves=15 (1 of 5 seeds)
- `x-learner`: min_child_samples=60, num_leaves=15 (3 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds)
- `dr-learner`: min_child_samples=60, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
- `r-learner`: min_child_samples=60, num_leaves=15 (3 of 5 seeds), min_child_samples=20, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds)
- `outcome-ranking`: min_child_samples=5, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
<!-- itx:end:ihdp -->

**What each ranking buys.**

<!-- itx:table:ihdp-policy -->
| Estimator | IPW gain at 10% | DR gain at 10% | IPW gain at 20% | DR gain at 20% | Treated share gap at 20% | IPW gain at 30% | DR gain at 30% |
|---|---|---|---|---|---|---|---|
| `s-learner` | 1.7187 (-0.1976, 4.7082) | 0.2030 (-0.3699, 0.7049) | 3.5632 (-0.2197, 9.5374) | 1.0295 (0.0006, 2.2763) | 0.0158 (-0.0980, 0.1588) | 4.7288 (0.1931, 11.8471) | 1.5607 (0.4896, 2.9262) |
| `t-learner` | 1.4916 (-0.1783, 4.8095) | 0.2627 (-0.2684, 0.6710) | 2.6797 (0.0485, 9.4154) | 0.5708 (-0.1205, 2.1799) | 0.0618 (-0.0662, 0.2174) | 7.2076 (0.7112, 15.1902) | 1.5599 (0.3319, 3.0994) |
| `x-learner` | 0.8674 (-0.2321, 3.0907) | 0.2087 (-0.3856, 0.5935) | 2.1143 (0.1205, 5.0318) | 0.6860 (-0.0159, 1.1697) | 0.0380 (-0.0942, 0.2060) | 4.4602 (0.5047, 10.5713) | 1.1157 (0.2666, 1.8676) |
| `dr-learner` | 1.3281 (-0.3865, 4.8262) | 0.2673 (-0.0358, 0.6035) | 3.1214 (-0.0152, 7.8435) | 0.7127 (0.0599, 1.1518) | 0.0523 (-0.1153, 0.1999) | 4.6005 (0.6226, 10.9085) | 0.9238 (0.2014, 1.7903) |
| `r-learner` | 1.7822 (-0.2609, 4.7336) | 0.4317 (-0.0340, 0.7649) | 2.5409 (-0.2897, 8.0141) | 0.8280 (0.2038, 1.2858) | 0.0184 (-0.0824, 0.1600) | 4.3654 (-0.0626, 11.2947) | 1.1517 (0.4078, 1.8167) |
| `dragonnet` | 1.9750 (-0.1552, 5.1707) | 0.3397 (-0.1518, 0.7445) | 3.3134 (0.0781, 9.5648) | 0.7649 (-0.0525, 1.4695) | 0.0554 (-0.0992, 0.2352) | 6.3596 (1.0713, 15.9175) | 1.1648 (0.2267, 2.5104) |
| `outcome-ranking` | 1.6706 (-0.8145, 6.1431) | 0.3756 (-0.0438, 1.0499) | 3.4835 (-0.6902, 9.4589) | 0.8084 (0.2226, 1.6685) | -0.0109 (-0.1939, 0.1475) | 4.5607 (-0.4979, 12.7002) | 1.2868 (0.5625, 2.3125) |
| `random-200` | 2.2266 (-0.2722, 7.9707) | 0.5154 (0.0215, 1.4131) | 4.4459 (-0.0230, 11.1432) | 1.0135 (0.2956, 2.0037) | 0.0434 (-0.0807, 0.1625) | 6.7272 (0.5701, 14.1585) | 1.5292 (0.6507, 2.6437) |
<!-- itx:end:ihdp-policy -->

This table is mostly a warning about itself, and it is left at full size for that reason. It
is the first of IHDP's hundred simulated replicates, not an average over them; PLAN.md
section 4 asked for the average and change 56 records why one replicate is reported.

Nothing in it separates. Every estimator's doubly robust gain at a 10% budget has an interval
containing zero, and random targeting's is the largest point estimate on the page. The test
split is 150 units; a policy value read off a tenth of it is being estimated from fifteen
people, and no amount of bootstrapping fixes that. The `uplift@10%` intervals in the ranking
table are thinner than they look for the same reason: a fifteen-row prefix with about three
treated units often loses an arm in a resample, and the bootstrap drops those resamples, so
the interval is conditional on the prefix holding both arms and was formed from as few as 640
of 1,000 draws.

The IPW column is worse than uninformative and shows why the propensity diagnostic exists.
IHDP's treatment was assigned observationally and its probabilities have to be estimated: on
the test split they run down to 0.0007, and 17.3% of rows sit against the 0.01 clipping bound
carrying inverse weights of 100 each. An estimate that divides by those numbers is resting on
a handful of rows, which is what an interval of (-0.20, 4.71) around a point estimate of 1.72
looks like. The doubly robust column is the one to read here, and it is reported next to the
other because the disagreement between them is the finding.

The true average effect here is about 4.0, and PEHE is the per-unit error against an effect
somebody wrote down, so lower is better and only the simulated datasets can report it. Which
estimator wins it is not a fact about the estimators: it depends on whether the treatment
effect is simpler or harder to describe than the baseline outcome, and
[docs/estimators.md](docs/estimators.md) shows the ordering reversing completely between two
synthetic problems that differ in nothing else.

IHDP is also small, at 448 training rows, and that is why the DR-learner is last here by a
distance. Its doubly robust guarantee is asymptotic; its pseudo-outcome's variance is not,
and at this size the variance wins. Predicting a constant zero would score 4.11.

Read the two right-hand columns, not the Qini. On the first seed the outcome ranking has
the **best Qini coefficient of the three** and buys 1.00 at a 10% budget, against random
targeting's 3.69. The curve ranks the worst policy first. Two reasons, both in
[docs/estimators.md](docs/estimators.md): the Qini is only identified when treatment was
assigned at random, and IHDP is confounded on purpose; and even on randomised data the
curve rewards sorting responders to the front whether or not the treatment caused them to
respond. That is the trap this project is built around, and it is why the headline number
is what a budget actually buys.

![Qini curves on IHDP](docs/figures/qini-ihdp.png)

### ACIC 2016, 4,802 units, simulated outcomes with known individual effects

The competition dataset from the 2016 Atlantic Causal Inference Conference: real covariates,
a simulated treatment assignment that depends on them, and a simulated outcome. Six times
IHDP's size, an effect that varies nearly twice as much as it averages, and confounding
strong enough that the naive difference in arm means is 3.58 against a true effect of 2.13.

<!-- itx:table:acic -->
| Estimator | Qini (95% CI) | Normalised AUUC | uplift@10% | uplift@20% | uplift@30% | Calibration slope | Calibration error | PEHE | ATE error |
|---|---|---|---|---|---|---|---|---|---|
| `s-learner` | 0.2170 (0.1582, 0.2785) | 0.7506 (0.6915, 0.8137) | 7.4746 (5.8664, 9.1483) | 6.9870 (5.6439, 8.3597) | 6.4845 (5.3567, 7.6608) | 1.5256 (1.1520, 1.8837) | 1.3076 (1.1003, 2.3021) | 1.7178 (1.5633, 1.8647) | 0.3182 (0.2163, 0.4252) |
| `t-learner` | 0.2107 (0.1520, 0.2712) | 0.7611 (0.6993, 0.8256) | 7.8915 (5.9614, 9.8037) | 7.0914 (5.6406, 8.5452) | 6.7091 (5.4234, 7.9346) | 1.0810 (0.8360, 1.3221) | 1.1050 (0.8915, 2.0397) | 1.2155 (1.1465, 1.2834) | 0.3585 (0.2839, 0.4292) |
| `x-learner` | 0.2273 (0.1681, 0.2895) | 0.7479 (0.6896, 0.8106) | 8.4128 (6.6035, 10.0808) | 7.3268 (5.8891, 8.6295) | 6.4870 (5.4129, 7.6685) | 1.1057 (0.8498, 1.3630) | 1.0296 (0.7755, 1.8750) | 0.8252 (0.7652, 0.8817) | 0.2023 (0.1513, 0.2534) |
| `dr-learner` | 0.2401 (0.1765, 0.3085) | 0.8059 (0.7402, 0.8749) | 7.3375 (5.6267, 9.1518) | 6.6582 (5.3928, 8.0011) | 6.5613 (5.4333, 7.6433) | 1.0110 (0.7946, 1.2441) | 1.1662 (0.9558, 2.1793) | 1.7920 (1.6956, 1.8894) | 0.4873 (0.3842, 0.5909) |
| `r-learner` | 0.2277 (0.1682, 0.2915) | 0.7413 (0.6809, 0.8052) | 7.8016 (6.0948, 9.6156) | 6.9500 (5.5382, 8.3248) | 6.4901 (5.3063, 7.6164) | 1.0724 (0.8202, 1.3236) | 0.9483 (0.7877, 1.8826) | 1.2290 (1.1519, 1.3097) | 0.1387 (0.0733, 0.2141) |
| `dragonnet` | 0.2249 (0.1659, 0.2884) | 0.7132 (0.6510, 0.7762) | 8.1097 (6.2400, 9.9492) | 6.7015 (5.2875, 8.1087) | 6.2858 (5.1038, 7.4153) | 1.0003 (0.7426, 1.2684) | 0.9397 (0.7773, 1.9097) | 1.2348 (1.1610, 1.3106) | 0.1258 (0.0644, 0.2039) |
| `outcome-ranking` | -0.0233 (-0.0674, 0.0216) | 0.4065 (0.3212, 0.4886) | 0.3415 (-1.5933, 2.4039) | 1.4755 (0.1591, 2.7197) | 1.7931 (0.6724, 2.9486) | - | - | - | - |
| `random-200` | -0.0000 (-0.0467, 0.0507) | 0.4881 (0.4251, 0.5563) | 3.4172 (0.9091, 6.0770) | 3.4402 (1.7333, 5.0483) | 3.4256 (2.2269, 4.6987) | - | - | - | - |

Selected on the validation split from the committed grid:

- `s-learner`: min_child_samples=60, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=20, num_leaves=15 (1 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds)
- `t-learner`: min_child_samples=20, num_leaves=31 (2 of 5 seeds), min_child_samples=20, num_leaves=15 (2 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds)
- `x-learner`: min_child_samples=20, num_leaves=15 (3 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
- `dr-learner`: min_child_samples=200, num_leaves=15 (4 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds)
- `r-learner`: min_child_samples=60, num_leaves=15 (2 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds)
- `outcome-ranking`: min_child_samples=5, num_leaves=31 (2 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=60, num_leaves=31 (1 of 5 seeds)
<!-- itx:end:acic -->

**What each ranking buys.**

<!-- itx:table:acic-policy -->
| Estimator | IPW gain at 10% | DR gain at 10% | IPW gain at 20% | DR gain at 20% | Treated share gap at 20% | IPW gain at 30% | DR gain at 30% |
|---|---|---|---|---|---|---|---|
| `s-learner` | 2.1482 (0.7636, 3.9902) | 0.7037 (0.5091, 0.9152) | 3.9917 (1.7828, 6.7722) | 1.3192 (1.0052, 1.6811) | 0.0470 (-0.0135, 0.1114) | 4.9032 (2.4120, 7.7670) | 1.7791 (1.4418, 2.1563) |
| `t-learner` | 2.0515 (0.7057, 3.7325) | 0.6739 (0.4368, 0.9041) | 3.2441 (1.4482, 5.6533) | 1.2102 (0.9219, 1.5393) | 0.0496 (-0.0041, 0.1132) | 4.5387 (2.1586, 7.3250) | 1.7099 (1.3660, 2.1032) |
| `x-learner` | 3.0468 (1.0616, 5.5227) | 0.8445 (0.5519, 1.1757) | 4.3136 (1.8900, 7.1794) | 1.4155 (1.0740, 1.8027) | 0.0404 (-0.0173, 0.1048) | 5.0233 (2.5381, 7.9043) | 1.8360 (1.4866, 2.2478) |
| `dr-learner` | 2.5187 (1.1620, 4.4216) | 0.7112 (0.5483, 0.9299) | 3.8450 (2.0130, 6.3262) | 1.2213 (0.9385, 1.5599) | 0.0542 (-0.0114, 0.1215) | 5.0669 (2.7208, 7.8254) | 1.6481 (1.2994, 2.0112) |
| `r-learner` | 2.1321 (0.7008, 3.8682) | 0.7482 (0.5295, 0.9970) | 4.2858 (1.9351, 7.0874) | 1.3051 (0.9738, 1.6945) | 0.0547 (-0.0084, 0.1175) | 5.3101 (2.7519, 8.2539) | 1.7844 (1.4092, 2.1854) |
| `dragonnet` | 3.2105 (1.2124, 5.6584) | 0.8130 (0.5525, 1.1458) | 4.2478 (2.0609, 7.0156) | 1.3117 (1.0001, 1.6851) | 0.0633 (0.0024, 0.1290) | 5.1241 (2.6802, 8.1155) | 1.7368 (1.3682, 2.1390) |
| `outcome-ranking` | -0.2396 (-1.0804, 0.8699) | -0.1971 (-0.3565, -0.0320) | 0.8724 (-0.7233, 3.0931) | 0.0567 (-0.1965, 0.3177) | 0.0018 (-0.0606, 0.0611) | 2.2734 (0.2457, 4.8577) | 0.4205 (0.1226, 0.7651) |
| `random-200` | 0.6073 (-0.0762, 1.7252) | 0.2564 (0.1040, 0.4336) | 1.2318 (0.1874, 2.6057) | 0.5075 (0.3018, 0.7321) | 0.0319 (-0.0140, 0.0805) | 1.8139 (0.5568, 3.4005) | 0.7587 (0.5256, 1.0176) |
<!-- itx:end:acic-policy -->

**This is the table the project was built to produce.** At a budget covering a tenth of the
population, the outcome ranking buys **-0.1971 (-0.357, -0.032)**. Negative, with the mean
interval below zero and four of the five seeds' own intervals entirely below it. Spending the budget on the highest-risk tenth of this population is
worse than spending nothing at all: not worse than uplift modelling, not worse than picking
names out of a hat, worse than leaving the money in the account. The five uplift models buy
0.67 to 0.84 at the same budget, and random targeting buys 0.26.

The ranking table above could not establish that. Its `uplift@10%` for the outcome ranking is
0.3415 with an interval of (-1.59, 2.40), which contains zero and overlaps random targeting's.
The reason is not precision, it is bias: ACIC's assignment is observational, so comparing the
arms *inside* a risk-selected group compares people who were not exchangeable to begin with.
Against the individual effects ACIC was simulated from, the true mean effect of that targeted
decile is **-2.46**, negative on all five seeds, while the naive arm difference averages
**+0.34**. Adjusting for the confounding does not sharpen the answer, it reverses it.
[docs/estimators.md](docs/estimators.md) has the per-seed numbers and the ground-truth check,
and `tests/test_policy_value.py` asserts it rather than asserting a paragraph.

The two estimators disagree by a factor of three here and the truth says the doubly robust one
is right, to a mean absolute error of 0.11 against IPW's 2.30. That was predictable without
any ground truth: 46% of ACIC's test rows sit against the clipping bound, and `itx diagnose`
and the propensity line both say so before anything is fitted.

This is where the trap is at its worst, and it is worth reading the two baseline rows
against the estimators:

| ACIC 2016 | Qini | Uplift at a 10% budget |
|---|---|---|
| `x-learner` | 0.2273 | 8.41 |
| Random targeting | -0.0000 | 3.42 |
| Outcome ranking | **-0.0233** | **0.34** |

By the within-prefix arm difference, spending the budget on the highest-risk tenth buys 0.34
and a random tenth buys 3.42. That is the metric the previous paragraph said is confounded
here, and its interval, (-1.59, 2.40), overlaps random's, so "ten times worse" is not a claim
this table can make. The doubly robust column, which is not confounded, puts the same queue at
-0.20 against random's +0.26: a change of sign, which is the claim this table can make, and
its Qini is negative rather than merely unimpressive. That happens when the people most likely to have the outcome are the people
least susceptible to the intervention, which is the shape a fraud queue or a clinical
follow-up list is usually assumed to have. The fraud worked case below was built on exactly
that assumption and shows it is not sufficient on its own: what decides the outcome is how
large the spread in susceptibility is against the spread in risk.

![Qini curves on ACIC 2016](docs/figures/qini-acic.png)

Five estimators sit well above the diagonal. The outcome ranking, in brown, is below it for
the first sixty percent of the population: for any budget in that range, the ordinary
approach delivers less than picking names out of a hat.

![Calibration on ACIC 2016](docs/figures/calibration-acic.png)

The calibration columns are the only ones in the table that notice magnitude. Qini, AUUC and
uplift at k are all unchanged if every prediction is multiplied by a constant, so a model
that ranks perfectly and predicts effects half the size they should be scores identically to
one that gets them right, and then forecasts half the return. The S-learner's calibration
slope of 1.53 is the furthest from 1 and its mean interval is the only one that excludes 1,
though per seed it does so on two of five, and the DR- and R-learners also exclude 1 on two
seeds each, on opposite sides, which averaging the endpoints hides. Read it as: the S-learner's
predictions are too compressed, realised uplift varies half as much again as it says, and the
others are not shown to be calibrated either.

### Lenta, 687,029 grocery customers, randomised SMS campaign

194 columns, no data dictionary, missing values in 150 of them, and a 0.75-point lift on a
10.3% base rate. It is the messiest dataset here and the closest in shape to a real customer
table.

<!-- itx:table:lenta -->
| Estimator | Qini (95% CI) | Normalised AUUC | uplift@10% | uplift@20% | uplift@30% | Calibration slope | Calibration error |
|---|---|---|---|---|---|---|---|
| `s-learner` | 0.0006 (-0.0003, 0.0016) | 0.0495 (0.0230, 0.0760) | 0.0184 (0.0008, 0.0357) | 0.0132 (0.0017, 0.0248) | 0.0118 (0.0031, 0.0207) | 0.7697 (-0.2513, 1.8054) | 0.0052 (0.0043, 0.0098) |
| `t-learner` | 0.0002 (-0.0008, 0.0012) | 0.0431 (0.0179, 0.0682) | 0.0159 (-0.0013, 0.0330) | 0.0114 (0.0007, 0.0227) | 0.0096 (0.0016, 0.0178) | 0.0568 (-0.1061, 0.2219) | 0.0193 (0.0166, 0.0234) |
| `x-learner` | 0.0002 (-0.0009, 0.0012) | 0.0430 (0.0185, 0.0676) | 0.0113 (-0.0053, 0.0284) | 0.0118 (0.0012, 0.0227) | 0.0101 (0.0019, 0.0182) | 0.0789 (-0.2035, 0.3460) | 0.0120 (0.0099, 0.0163) |
| `dr-learner` | 0.0004 (-0.0006, 0.0014) | 0.0460 (0.0207, 0.0704) | 0.0163 (-0.0003, 0.0337) | 0.0140 (0.0033, 0.0251) | 0.0113 (0.0032, 0.0196) | 0.0964 (-0.1011, 0.2946) | 0.0149 (0.0126, 0.0190) |
| `r-learner` | 0.0003 (-0.0007, 0.0013) | 0.0455 (0.0212, 0.0701) | 0.0093 (-0.0083, 0.0263) | 0.0131 (0.0022, 0.0241) | 0.0117 (0.0035, 0.0200) | 0.0634 (-0.1823, 0.3037) | 0.0136 (0.0113, 0.0177) |
| `dragonnet` | -0.0002 (-0.0011, 0.0008) | 0.0381 (0.0159, 0.0607) | 0.0100 (-0.0054, 0.0244) | 0.0084 (-0.0010, 0.0178) | 0.0079 (0.0009, 0.0150) | 0.0299 (-0.4319, 0.4804) | 0.0117 (0.0094, 0.0159) |
| `outcome-ranking` | 0.0005 (-0.0005, 0.0014) | 0.0469 (0.0175, 0.0763) | 0.0075 (-0.0118, 0.0267) | 0.0096 (-0.0030, 0.0224) | 0.0095 (-0.0003, 0.0191) | - | - |
| `random-200` | -0.0000 (-0.0008, 0.0007) | 0.0402 (0.0286, 0.0511) | 0.0073 (-0.0034, 0.0175) | 0.0074 (-0.0004, 0.0146) | 0.0074 (0.0019, 0.0125) | - | - |

Selected on the validation split from the committed grid:

- `s-learner`: min_child_samples=60, num_leaves=31 (2 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
- `t-learner`: min_child_samples=60, num_leaves=31 (2 of 5 seeds), min_child_samples=20, num_leaves=15 (2 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
- `x-learner`: min_child_samples=200, num_leaves=15 (2 of 5 seeds), min_child_samples=60, num_leaves=31 (2 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds)
- `dr-learner`: min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds), min_child_samples=200, num_leaves=31 (1 of 5 seeds), min_child_samples=60, num_leaves=31 (1 of 5 seeds)
- `r-learner`: min_child_samples=5, num_leaves=15 (3 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=200, num_leaves=31 (1 of 5 seeds)
- `outcome-ranking`: min_child_samples=60, num_leaves=31 (2 of 5 seeds), min_child_samples=200, num_leaves=31 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds)
<!-- itx:end:lenta -->

**What each ranking buys.**

<!-- itx:table:lenta-policy -->
| Estimator | IPW gain at 10% | DR gain at 10% | IPW gain at 20% | DR gain at 20% | Treated share gap at 20% | IPW gain at 30% | DR gain at 30% |
|---|---|---|---|---|---|---|---|
| `s-learner` | 0.0017 (-0.0003, 0.0038) | 0.0018 (0.0001, 0.0035) | 0.0022 (-0.0004, 0.0049) | 0.0024 (0.0002, 0.0046) | 0.0010 (-0.0040, 0.0062) | 0.0027 (-0.0002, 0.0057) | 0.0030 (0.0006, 0.0056) |
| `t-learner` | 0.0013 (-0.0007, 0.0033) | 0.0017 (0.0000, 0.0033) | 0.0017 (-0.0007, 0.0043) | 0.0021 (0.0001, 0.0042) | 0.0001 (-0.0050, 0.0052) | 0.0022 (-0.0006, 0.0049) | 0.0025 (0.0003, 0.0048) |
| `x-learner` | 0.0006 (-0.0013, 0.0026) | 0.0010 (-0.0006, 0.0027) | 0.0016 (-0.0008, 0.0041) | 0.0020 (0.0000, 0.0041) | -0.0005 (-0.0056, 0.0046) | 0.0022 (-0.0006, 0.0049) | 0.0024 (0.0002, 0.0047) |
| `dr-learner` | 0.0009 (-0.0011, 0.0029) | 0.0016 (0.0000, 0.0032) | 0.0019 (-0.0006, 0.0043) | 0.0025 (0.0005, 0.0046) | -0.0013 (-0.0064, 0.0037) | 0.0023 (-0.0003, 0.0051) | 0.0030 (0.0008, 0.0054) |
| `r-learner` | 0.0007 (-0.0014, 0.0027) | 0.0010 (-0.0006, 0.0026) | 0.0019 (-0.0006, 0.0044) | 0.0024 (0.0003, 0.0045) | -0.0003 (-0.0053, 0.0049) | 0.0027 (-0.0001, 0.0053) | 0.0031 (0.0008, 0.0054) |
| `dragonnet` | 0.0008 (-0.0010, 0.0025) | 0.0011 (-0.0004, 0.0025) | 0.0013 (-0.0008, 0.0034) | 0.0017 (-0.0001, 0.0035) | -0.0001 (-0.0051, 0.0051) | 0.0016 (-0.0007, 0.0040) | 0.0021 (0.0000, 0.0041) |
| `outcome-ranking` | 0.0002 (-0.0023, 0.0028) | 0.0012 (-0.0007, 0.0031) | 0.0011 (-0.0019, 0.0043) | 0.0023 (-0.0001, 0.0049) | -0.0004 (-0.0054, 0.0048) | 0.0021 (-0.0013, 0.0055) | 0.0032 (0.0004, 0.0060) |
| `random-200` | 0.0004 (-0.0008, 0.0016) | 0.0006 (-0.0004, 0.0016) | 0.0009 (-0.0008, 0.0025) | 0.0012 (-0.0002, 0.0025) | -0.0003 (-0.0048, 0.0040) | 0.0013 (-0.0005, 0.0030) | 0.0018 (0.0002, 0.0033) |
<!-- itx:end:lenta-policy -->
**Changing the metric turned this dataset from a null into a faint signal, which is the
result week 4 predicted and could not produce.** Not one Qini interval in the table above
excludes zero. In this table, at a budget covering a fifth of the customers, the mean
intervals of all five LightGBM models do: 0.0024, 0.0021, 0.0020, 0.0025 and 0.0024, against
random targeting's 0.0012 and an outcome ranking whose interval still contains zero. Per seed
it is thinner than the means suggest. The DR-learner clears zero on all five seeds, the
R-learner on four, the S-learner on three, and the T- and X-learners on two; the X-learner's
mean lower bound is +0.0000017. So the direction is consistent, the DR-learner's result is
established, and the rest are suggestive.

Dragonnet arrived after that paragraph was written and does not join it. At 0.0017
(-0.0001, +0.0035) its interval covers zero, as the outcome ranking's does, which makes Lenta
the one dataset here where the network is the weakest of the six modelled rankings. Its first
Lenta column was worse than weak: it was an all-NaN fit reporting random targeting's numbers
under an estimator's name, because Lenta is the only dataset here with missing values and a
dense layer does not take them where LightGBM does. That is fixed, the row above is the refit,
and [docs/estimators.md](docs/estimators.md) carries the whole account because how it hid is
more useful than what it was.

The mechanism is the one change 33 guessed at. A Qini coefficient integrates the whole curve,
so a real advantage confined to the front of the ranking is averaged away against the long
flat tail where every method is identical. A budgeted number reads only the front. On a
dataset this underpowered that is the difference between seeing the effect and not.

What it does not do is make Lenta a win. The uplift models buy about twice what random
targeting buys and their intervals overlap random's heavily, so "these rankings buy something"
is established and "these rankings beat picking names out of a hat" is not. The honest
summary of Lenta is still the week 4 one: a 0.75-point effect on a 10.3% base rate is too
small to target at this sample size. The finding is about the instrument, not the dataset.

The IPW column contains zero everywhere, at every budget, for every method including the ones
the DR column separates. Lenta was randomised but does not publish its assignment probability,
so the propensity is estimated here rather than known, and the extra variance that costs is
exactly what the doubly robust column is buying back.

**Nothing here beats random targeting, and that is the result.** Every Qini interval in the
table above contains zero, for all five estimators and for both baselines. On the Qini they
are indistinguishable from each other and from picking names out of a hat.

| Lenta | Qini | uplift at a 10% budget |
|---|---|---|
| `s-learner` | 0.0006 (-0.0003, 0.0016) | 0.0184 (0.0008, 0.0357) |
| `dr-learner` | 0.0004 (-0.0006, 0.0014) | 0.0163 (-0.0003, 0.0337) |
| Outcome ranking | 0.0005 (-0.0005, 0.0014) | 0.0075 (-0.0118, 0.0267) |
| Random targeting | -0.0000 (-0.0008, 0.0007) | 0.0073 (-0.0034, 0.0175) |

There is exactly one thread worth pulling. The S-learner is the only estimator whose mean
interval on realised uplift excludes zero at every budget: 0.0184 at 10%, 0.0132 at 20%,
0.0118 at 30%, against random targeting's 0.0073 to 0.0075. That is about two and a half times
random at a tight budget. Per seed it is three of five at 10% and 20% and four at 30%, and on
the first seed its top decile is worth 0.0043 against random's 0.0069, below random. Its
interval overlaps random's heavily, and the Qini, which integrates the whole curve rather than
one cut of it, declines to confirm anything. So it is a hint and it is reported as a hint.

**Why this dataset cannot answer the question.** The average effect is 0.75 percentage points
on a 10.3% base rate. The test split is 137,406 rows and only about a quarter of them are
controls, so roughly 34,000 control rows carry the comparison. Detecting *heterogeneity*
inside an effect that small, from that many controls, is a lot to ask. Criteo's effect is
about a third larger in absolute terms, 1.03 points against 0.75, and it has twice the test
rows with far more events, which is why its intervals are tight enough to separate things and
Lenta's are not.

This is worth more to the reader than a fifth win would have been. A benchmark where every
dataset produces a clean answer is a benchmark that has quietly selected its datasets. Lenta
is a real retail campaign of a perfectly ordinary size, and the honest finding is that uplift
modelling on it buys nothing you could defend to a sceptical colleague. The four datasets now
give four different answers, which is the actual state of this field:

| | What the data supports |
|---|---|
| ACIC 2016 | Uplift models work; risk ranking is ten times worse than random |
| Criteo | Uplift models work; risk ranking works just as well |
| Hillstrom | Uplift models beat both baselines; risk ranking is ambiguous |
| Lenta | Nothing is distinguishable from random |

Two of its columns never reach the model, and that is the other half of the story here.
`response_sms` and `response_viber` sit in the feature block with names that could plausibly
mean "responded to an earlier campaign". Nothing in the documentation says either way. In a
randomised trial the answer is checkable: a pre-treatment covariate has the same mean in both
arms, so anything that does not is not pre-treatment.

| Column | Standardised difference between arms |
|---|---|
| `response_sms` | **0.198** |
| `response_viber` | **0.068** |
| worst of the other 192 | 0.025 |
| median of the other 192 | 0.011 |

Eighteen times the median, in the only two columns whose names suggest they were recorded
after the campaign went out. They are responses to the campaign's own delivery channels, so
they are consequences of the treatment, and an estimator handed `response_sms` would have
been told part of the answer. Both are dropped. The check is a permanent part of the package
rather than a script that was run once: [`itx.metrics.balance`](src/itx/metrics/balance.py),
and it runs against every dataset.

### Criteo-UPLIFT, 13.98M randomised ad impressions

The volume test. Twelve anonymous features, a 4.7% visit rate, arms split 85/15, and enough
rows that nothing here is small-sample noise.

<!-- itx:table:criteo -->
| Estimator | Qini (95% CI) | Normalised AUUC | uplift@10% | uplift@20% | uplift@30% | Calibration slope | Calibration error |
|---|---|---|---|---|---|---|---|
| `s-learner` | 0.0032 (0.0025, 0.0038) | 0.1951 (0.1568, 0.2306) | 0.0556 (0.0387, 0.0700) | 0.0392 (0.0298, 0.0478) | 0.0296 (0.0233, 0.0358) | 0.9900 (0.7151, 1.2500) | 0.0012 (0.0010, 0.0035) |
| `t-learner` | 0.0022 (0.0016, 0.0029) | 0.1716 (0.1385, 0.2027) | 0.0565 (0.0430, 0.0695) | 0.0346 (0.0265, 0.0424) | 0.0256 (0.0200, 0.0310) | 0.5491 (0.3862, 0.7014) | 0.0069 (0.0056, 0.0090) |
| `x-learner` | 0.0025 (0.0018, 0.0032) | 0.1792 (0.1442, 0.2102) | 0.0585 (0.0443, 0.0721) | 0.0368 (0.0284, 0.0445) | 0.0266 (0.0210, 0.0322) | 0.6445 (0.4643, 0.8125) | 0.0053 (0.0040, 0.0075) |
| `dr-learner` | 0.0024 (0.0017, 0.0031) | 0.1768 (0.1396, 0.2099) | 0.0558 (0.0415, 0.0696) | 0.0356 (0.0270, 0.0441) | 0.0263 (0.0200, 0.0323) | 0.6958 (0.4737, 0.8917) | 0.0039 (0.0029, 0.0062) |
| `r-learner` | 0.0026 (0.0019, 0.0033) | 0.1813 (0.1455, 0.2138) | 0.0586 (0.0443, 0.0727) | 0.0376 (0.0292, 0.0458) | 0.0271 (0.0212, 0.0329) | 0.7735 (0.5600, 0.9586) | 0.0035 (0.0028, 0.0060) |
| `dragonnet` | 0.0032 (0.0025, 0.0039) | 0.1959 (0.1601, 0.2313) | 0.0603 (0.0449, 0.0755) | 0.0396 (0.0304, 0.0482) | 0.0294 (0.0234, 0.0357) | 0.7665 (0.5703, 0.9542) | 0.0033 (0.0023, 0.0055) |
| `outcome-ranking` | 0.0031 (0.0024, 0.0038) | 0.1946 (0.1563, 0.2309) | 0.0576 (0.0426, 0.0730) | 0.0397 (0.0303, 0.0486) | 0.0295 (0.0232, 0.0360) | - | - |
| `random-200` | 0.0000 (-0.0005, 0.0005) | 0.1141 (0.1022, 0.1262) | 0.0104 (0.0044, 0.0160) | 0.0104 (0.0064, 0.0145) | 0.0104 (0.0075, 0.0132) | - | - |

Selected on the validation split from the committed grid:

- `s-learner`: min_child_samples=200, num_leaves=15 (3 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
- `t-learner`: min_child_samples=60, num_leaves=31 (1 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds), min_child_samples=20, num_leaves=15 (1 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds)
- `x-learner`: min_child_samples=200, num_leaves=31 (4 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds)
- `dr-learner`: min_child_samples=5, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds), min_child_samples=20, num_leaves=15 (1 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds)
- `r-learner`: min_child_samples=200, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=20, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds)
- `outcome-ranking`: min_child_samples=20, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=15 (2 of 5 seeds), min_child_samples=200, num_leaves=31 (1 of 5 seeds)
<!-- itx:end:criteo -->

**What each ranking buys.**

<!-- itx:table:criteo-policy -->
| Estimator | IPW gain at 10% | DR gain at 10% | IPW gain at 20% | DR gain at 20% | Treated share gap at 20% | IPW gain at 30% | DR gain at 30% |
|---|---|---|---|---|---|---|---|
| `s-learner` | 0.0087 (0.0071, 0.0103) | 0.0057 (0.0044, 0.0069) | 0.0097 (0.0079, 0.0117) | 0.0069 (0.0054, 0.0085) | 0.0066 (0.0040, 0.0095) | 0.0100 (0.0081, 0.0119) | 0.0072 (0.0056, 0.0087) |
| `t-learner` | 0.0078 (0.0064, 0.0092) | 0.0053 (0.0041, 0.0064) | 0.0082 (0.0066, 0.0099) | 0.0057 (0.0043, 0.0070) | 0.0061 (0.0033, 0.0090) | 0.0085 (0.0068, 0.0102) | 0.0059 (0.0044, 0.0072) |
| `x-learner` | 0.0080 (0.0065, 0.0095) | 0.0056 (0.0043, 0.0067) | 0.0087 (0.0070, 0.0103) | 0.0061 (0.0047, 0.0074) | 0.0061 (0.0034, 0.0090) | 0.0088 (0.0071, 0.0105) | 0.0063 (0.0048, 0.0076) |
| `dr-learner` | 0.0078 (0.0064, 0.0094) | 0.0054 (0.0041, 0.0067) | 0.0087 (0.0068, 0.0105) | 0.0061 (0.0046, 0.0075) | 0.0062 (0.0035, 0.0091) | 0.0088 (0.0070, 0.0107) | 0.0064 (0.0048, 0.0078) |
| `r-learner` | 0.0080 (0.0066, 0.0096) | 0.0055 (0.0043, 0.0068) | 0.0088 (0.0071, 0.0105) | 0.0062 (0.0049, 0.0075) | 0.0054 (0.0027, 0.0083) | 0.0090 (0.0072, 0.0108) | 0.0065 (0.0051, 0.0079) |
| `dragonnet` | 0.0090 (0.0074, 0.0107) | 0.0059 (0.0047, 0.0072) | 0.0098 (0.0080, 0.0117) | 0.0069 (0.0054, 0.0084) | 0.0065 (0.0039, 0.0094) | 0.0100 (0.0082, 0.0119) | 0.0071 (0.0055, 0.0087) |
| `outcome-ranking` | 0.0082 (0.0065, 0.0101) | 0.0055 (0.0042, 0.0069) | 0.0098 (0.0080, 0.0118) | 0.0070 (0.0055, 0.0086) | 0.0064 (0.0036, 0.0091) | 0.0100 (0.0081, 0.0120) | 0.0072 (0.0055, 0.0088) |
| `random-200` | 0.0010 (0.0005, 0.0016) | 0.0008 (0.0003, 0.0013) | 0.0021 (0.0013, 0.0029) | 0.0015 (0.0009, 0.0022) | 0.0000 (-0.0026, 0.0027) | 0.0031 (0.0022, 0.0040) | 0.0023 (0.0015, 0.0030) |
<!-- itx:end:criteo-policy -->

**The mirror image of ACIC, measured the same way, and the reason this project reports a
condition rather than a rule.** The outcome ranking buys 0.0055 at a 10% budget. The five
uplift models buy 0.0053 to 0.0057. On the best-powered dataset here, with intervals tight
enough that a real gap would show, the baseline this repository exists to catch out is
indistinguishable from every estimator built to beat it, at every budget. Uplift modelling
buys nothing on Criteo.

What all seven rankings do buy is large: random targeting gets 0.0008, so any ranking at all
is worth about seven times a coin flip here. The choice that matters on this dataset is whether
to target, not what to target with.

Put the two tables side by side and the whole argument is in one line. At a 10% budget the
identical baseline, produced by the identical code, buys **-0.1971 on ACIC and +0.0055 on
Criteo**, against random targeting's +0.2564 and +0.0008. Ranking by risk is not a trap and is
not safe; it is one or the other depending on the data, the difference is worth a sign change,
and `uv run itx diagnose --dataset <name>` decides which case a problem is in for the price of
one outcome model.

**The IPW column is half again as high as the DR column here, and that is a defect rather
than noise.** It is worth setting out, because it refutes the tidy rule the rest of this page
was heading towards.

Criteo's propensity is a design constant of 0.85 and not one unit is clipped, so the overlap
problem that wrecks IPW on ACIC and IHDP is absent. The gap is systematic all the same: the
ratio is 1.35 to 1.77 across the five seeds, never below 1.3, and the doubly robust column
agrees with the plain arm difference in the prefix to three decimal places while IPW does not.

Decomposed on seed 11, it is exact. The top 10% of the S-learner's ranking has a realised
treated share of **0.8667**, not 0.85: about eight standard errors away from the design value.
Horvitz-Thompson divides the control arm by 1 - 0.85, so each control unit carries a weight of
6.67, and a prefix that is short of control units by 1.7 points is an estimate short of the
thing it subtracts. On the committed fit the observed gap is +0.0040. The exact closure of the
arithmetic, predicted gap equal to observed gap to the last digit, was verified in week 5 on
that week's fit and is asserted by `tests/test_policy_value.py`; the tables have been
remeasured since and the last digits moved while the mechanism did not.

The reason a covariate-based ranking can shift the treated share at all is that Criteo's arms
are not quite balanced. Every one of its twelve covariates leans the same way, the largest
standardised mean difference is 0.047, and on 1.4 million rows that is around twenty standard
errors. It is far below the conventional 0.1 threshold and this repository's own balance
detector passes it, correctly. It is still enough: a model ranking on those covariates
concentrates treated units at the top, and an estimator dividing by 0.15 turns 1.7 points of
concentration into 50% of the answer.

So the rule is not "check the clipped share and otherwise trust IPW". Nothing is clipped here.
The rule is that Horvitz-Thompson weighting is fragile whenever one arm is small, because that
arm's weight is large and any gap between the assumed and the realised share inside a
*selected* subset is multiplied by it. The cheap check is to compare the treated share inside
the targeted prefix with the propensity of those same units, which costs one mean and is the
`Treated share gap` column in the policy tables above. Read the doubly robust column on
Criteo.

**This is the dataset where the outcome-ranking trap does not happen, and it is the most
useful result in the project.**

| Criteo | Qini | uplift at a 10% budget |
|---|---|---|
| `s-learner` | 0.0032 (0.0025, 0.0038) | 0.0556 |
| `r-learner` | 0.0026 (0.0019, 0.0033) | 0.0586 |
| **Outcome ranking** | **0.0031 (0.0024, 0.0038)** | **0.0576** |
| Random targeting | 0.0000 (-0.0005, 0.0005) | 0.0104 |

On 279,592 held-out rows, ranking by predicted risk matches every uplift model in the table.
Its interval overlaps the best estimator's almost exactly. Compare that with ACIC above, where
the same baseline has a *negative* Qini and buys less than nothing, where picking names out
of a hat buys 0.26. Same baseline, same code, opposite verdict.

The reason is visible in one table. Ranking the test rows by predicted risk and reading the
two arms inside each decile:

| Decile by risk | Control rate | Treated rate | Absolute uplift | Ratio |
|---|---|---|---|---|
| 1 (highest risk) | 0.3166 | 0.3743 | **+0.0577** | 1.18 |
| 2 | 0.0571 | 0.0684 | +0.0113 | 1.20 |
| 3 | 0.0172 | 0.0192 | +0.0020 | 1.12 |
| ... | | | | |
| 10 (lowest risk) | 0.0002 | 0.0007 | +0.0004 | 2.82 |

Baseline risk falls by a factor of about 1,500 from the top decile to the bottom. The
multiplier the treatment applies moves by a factor of about 2, and not even reliably. So the
absolute uplift, which is risk times multiplier, is almost entirely decided by the risk.
Ranking by risk and ranking by uplift are nearly the same ordering, and the trap cannot spring.

That gives a rule worth more than the trap on its own: **risk ranking works when the spread in
baseline risk dwarfs the spread in relative effect, and fails when it does not.** The three
datasets are three points on that spectrum, and the same decile table separates them:

| | Risk spread across deciles | Uplift follows risk | Outcome ranking |
|---|---|---|---|
| Criteo | about 1,500x | -0.60 | at parity with the best |
| Hillstrom | about 6x | -0.24 | none of the way at 10%, a third at 30% |
| ACIC 2016 | effect runs against risk | positive | worse than random |

So the honest claim this project can make is not "the intervention list is never the risk
list". It is that the two lists differ by an amount you cannot guess in advance and can
measure cheaply, and that the cost of assuming they agree runs from nothing to worse than
doing nothing. On a fraud queue or a clinical follow-up list, where the most at-risk
cases are often the least movable, ACIC is the relevant picture. On an advertising set where
the high-risk group is a hundred times more likely to act, Criteo is. Building an uplift model
is worth it in the first case and close to pointless in the second, and a decile table like
the one above says which one you are in before anybody fits a meta-learner. The fraud worked
case below is the reminder that "the most at-risk are the least movable" can be true and
still put a problem on the Criteo side of this table, when the effect's sign is decided by
the risk itself.

The table is the committed 10% stratified subsample, 1,397,958 rows. Stratifying on the arm
crossed with both outcomes holds every cell at exactly a tenth of itself, so the subsample's
treated share is 0.8500005 against the full file's 0.8500001 and its visit rate is 0.046991
against 0.046992. The reason for subsampling is wall-clock time rather than memory, and
PLAN.md change 23 records that the plan originally said otherwise and was wrong.

Criteo's `exposure` column is dropped for the same reason Lenta's two are, in a more obvious
form: it records whether an ad was actually shown, and it is zero for every one of the
2,096,937 control rows, because a control user cannot be shown an ad that was never served.
It is a consequence of the treatment. Criteo published it deliberately, for a
noncompliance question this project does not ask.

### How much unmeasured confounding would overturn any of this

Every number above rests on an assumption nothing here can test: that the covariates carry
all of the confounding. Three devices price it, and `uv run itx sensitivity --dataset <name>`
runs all three. At a 20% budget on the T-learner's ranking, seed 11:

| Dataset | E-value | Rosenbaum Gamma | Negative control |
|---|---|---|---|
| Hillstrom | 2.52, falling to 1.79 at the interval | 1.3 | -0.013 (-0.050, 0.026) |
| ACIC 2016 | not computed: crude contrast is confounded | 6.6 | -0.021 (-0.203, 0.169) |
| IHDP | not computed: crude contrast is confounded | too few pairs to form one | -0.527 (-2.379, 0.756) |

The E-value is how strongly something unmeasured would have to be associated with both the
treatment and the outcome to explain the result away. The Gamma is how far it would have to
shift the odds of being treated. The negative control runs the whole pipeline on a
pre-treatment covariate, where the true effect is zero by construction, and reports what came
back in standard deviations.

**What each of these can and cannot say here.** The E-value is defined for an *adjusted*
estimate, and the ratio this repository forms is the crude treated-to-control contrast inside
the targeted group. On a randomised design the two coincide, so Hillstrom's E-value is a real
one. On ACIC and IHDP the crude contrast is confounded by the measured covariates, so an
E-value of it would price the confounding the covariates already explain rather than anything
unmeasured, and it is not computed there; an earlier version of this table reported 9.95 for
ACIC and read it as reassurance, which PLAN.md change 56 records as a mistake. The Gamma is a
function of the matched effect's size and the number of pairs: it says how much hidden bias a
result of that size would survive and detects nothing. ACIC's 6.6 says its matched effect is
large relative to its noise, and since ACIC's confounding is entirely measured (assignment is
simulated from the recorded covariates), the number is unfalsifiable there rather than
confirmed. The negative control is the only device that can fail, and on a design where
assignment depends on the covariates it is a leave-one-covariate-out balance check: it comes
back non-zero whenever the held-out covariate drives assignment and is not predicted by the
rest, which can happen with no unmeasured confounding at all. That it passes on ACIC is a fact
about which covariate was held out. The evidence that it works is a test that plants a
measured confounder, removes it from the covariate set, and requires the estimate to move.

**Hillstrom's Gamma of 1.3 is the number to sit with.** The strongest cleanly-readable result
here would be overturned by a hidden factor shifting the odds of treatment by a third. It was
a randomised experiment, so nothing is hiding, and the reading is what an observational
version of the same study would have had to argue against. It is not much.

[docs/estimators.md](docs/estimators.md) carries the rest, including why IHDP's row has
nothing but a negative control in it and why the Gamma is quoted to one decimal place. Criteo
and Lenta need a full-size fit and are queued, and Lenta also needs a declared design
propensity before the Rosenbaum bound means anything on it, because matching on a propensity
estimated from a randomised design is pairing on noise.

## The fraud worked case

**This case is semi-synthetic and says so before it says anything else. The features are real
and the treatment effect is invented.** IEEE-CIS Fraud Detection supplies 590,540 real card
transactions with real features and a real `isFraud` label. Nobody recorded which of them a
human analyst reviewed, because that is not in the data, so the review and everything it does
are simulated here from a published function
([src/itx/data/ieee_fraud.py](src/itx/data/ieee_fraud.py), card at
[docs/data/ieee-fraud.md](docs/data/ieee-fraud.md)). Nothing below is a measured fact about
fraud review. It is the allocation logic shown on a problem shaped like the ones this project
is aimed at.

What the simulation says: a transaction is worth its amount. Left alone, a fraudulent one is
charged back and the business loses that amount, and a legitimate one completes and earns a 3%
margin. Sent to review, a fraudulent one is caught with some probability and the loss is
avoided, and a legitimate one is wrongly declined with some probability and the margin is lost.
So review helps on fraud and hurts on everything else, which makes 96.5% of this population
sleeping dogs.

**The one assumption doing the work is that review is hardest on what looks riskiest.** The
catch rate falls from 0.85 to 0.30 as a transaction's fraud signal rises, because the obvious
fraud is stopped by rules before it reaches a queue and what arrives is the practised kind. A
reader who thinks real review works the other way can change one constant and rerun. That
assumption puts the highest-risk transactions in the lost-causes quadrant, and it was chosen
to make the risk queue lose. It did not. The tables say by how much and the text after them
says why, because the why is the useful part.

The two tables below are the standard ones, on 118,108 held-out transactions per seed, five
seeds. `outcome-ranking` here is a risk queue: a model of dollars retained, fitted on every
row, reviewed and not, with the *lowest* predicted value reviewed first. The allocation
table further down uses the same kind of model fitted on the unreviewed rows only, which is
what a fraud team's risk score actually is; the two are close and are not the same model. The
dataset declares that its risk is a low outcome and the baseline reads the declaration
(PLAN.md change 54, which is also the record of what the table said before it did).

<!-- itx:table:ieee-fraud -->
| Estimator | Qini (95% CI) | Normalised AUUC | uplift@10% | uplift@20% | uplift@30% | Calibration slope | Calibration error | PEHE | ATE error |
|---|---|---|---|---|---|---|---|---|---|
| `s-learner` | 0.4652 (0.3447, 0.5906) | 0.0789 (0.0637, 0.0932) | 17.1557 (12.4618, 22.0506) | 9.9734 (7.5248, 12.4965) | 7.0175 (5.3525, 8.7249) | 1.5325 (1.1985, 1.8800) | 0.8989 (0.5578, 1.4612) | 23.3784 (21.0590, 25.8840) | 0.3049 (0.1803, 0.4392) |
| `t-learner` | 0.4226 (0.2977, 0.5485) | 0.0757 (0.0604, 0.0899) | 17.3070 (12.6470, 22.1204) | 9.5967 (7.1883, 12.0609) | 6.7156 (5.0896, 8.3614) | 0.3999 (0.2810, 0.5122) | 1.9291 (1.4487, 2.4670) | 24.1094 (22.0752, 26.2834) | 0.0992 (0.0078, 0.2383) |
| `x-learner` | 0.4578 (0.3358, 0.5863) | 0.0784 (0.0637, 0.0928) | 17.8279 (13.2454, 22.7470) | 10.0149 (7.6415, 12.5112) | 6.9697 (5.3687, 8.6182) | 0.6519 (0.4903, 0.7977) | 1.0678 (0.7091, 1.6226) | 22.9896 (21.0951, 25.0188) | 0.1186 (0.0258, 0.2501) |
| `dr-learner` | 0.4271 (0.3013, 0.5547) | 0.0761 (0.0611, 0.0906) | 16.2790 (11.6711, 21.0252) | 9.5924 (7.2178, 12.0556) | 6.7984 (5.1800, 8.4516) | 0.5853 (0.4317, 0.7256) | 1.4275 (1.0065, 2.0020) | 24.6645 (22.8585, 26.5853) | 0.0560 (0.0077, 0.2032) |
| `r-learner` | 0.4265 (0.2997, 0.5527) | 0.0760 (0.0611, 0.0899) | 16.4785 (12.0197, 21.1627) | 9.5768 (7.2282, 11.9806) | 6.7820 (5.1959, 8.3992) | 0.6771 (0.5107, 0.8296) | 1.0844 (0.7644, 1.6768) | 24.8547 (22.6779, 27.1907) | 0.1068 (0.0089, 0.2451) |
| `dragonnet` | 0.3139 (0.1956, 0.4345) | 0.0673 (0.0521, 0.0815) | 12.5730 (8.1840, 17.1673) | 7.6659 (5.3363, 10.0499) | 5.6982 (4.1044, 7.3191) | 0.6936 (0.3880, 0.9883) | 1.1694 (0.7927, 1.7314) | 25.1120 (22.7257, 27.7288) | 0.8310 (0.6992, 0.9715) |
| `outcome-ranking` | 0.4625 (0.3395, 0.5875) | 0.0788 (0.0638, 0.0929) | 17.9672 (13.3958, 22.6635) | 9.9946 (7.6514, 12.4335) | 6.9599 (5.3604, 8.5863) | - | - | - | - |
| `random-200` | 0.0023 (-0.0766, 0.0841) | 0.0439 (0.0379, 0.0500) | 2.3484 (0.9081, 3.9574) | 2.3517 (1.2881, 3.4474) | 2.3446 (1.5297, 3.1682) | - | - | - | - |

Selected on the validation split from the committed grid:

- `s-learner`: min_child_samples=200, num_leaves=31 (2 of 5 seeds), min_child_samples=60, num_leaves=31 (1 of 5 seeds), min_child_samples=20, num_leaves=15 (1 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds)
- `t-learner`: min_child_samples=200, num_leaves=15 (3 of 5 seeds), min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds)
- `x-learner`: min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=60, num_leaves=31 (1 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds)
- `dr-learner`: min_child_samples=200, num_leaves=15 (2 of 5 seeds), min_child_samples=60, num_leaves=15 (2 of 5 seeds), min_child_samples=20, num_leaves=31 (1 of 5 seeds)
- `r-learner`: min_child_samples=60, num_leaves=15 (2 of 5 seeds), min_child_samples=5, num_leaves=15 (2 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds)
- `outcome-ranking`: min_child_samples=60, num_leaves=15 (1 of 5 seeds), min_child_samples=200, num_leaves=15 (1 of 5 seeds), min_child_samples=5, num_leaves=15 (1 of 5 seeds), min_child_samples=200, num_leaves=31 (1 of 5 seeds), min_child_samples=5, num_leaves=31 (1 of 5 seeds)
<!-- itx:end:ieee-fraud -->

<!-- itx:table:ieee-fraud-policy -->
| Estimator | IPW gain at 10% | DR gain at 10% | IPW gain at 20% | DR gain at 20% | Treated share gap at 20% | IPW gain at 30% | DR gain at 30% |
|---|---|---|---|---|---|---|---|
| `s-learner` | 1.7492 (1.2613, 2.2409) | 1.6800 (1.2791, 2.0967) | 2.0260 (1.5339, 2.5376) | 1.9510 (1.5414, 2.3748) | -0.0042 (-0.0105, 0.0022) | 2.1281 (1.6283, 2.6416) | 2.0560 (1.6403, 2.4877) |
| `t-learner` | 1.7387 (1.2662, 2.2259) | 1.6645 (1.2798, 2.0586) | 1.9416 (1.4590, 2.4376) | 1.8577 (1.4705, 2.2565) | -0.0033 (-0.0096, 0.0030) | 2.0297 (1.5387, 2.5299) | 1.9492 (1.5534, 2.3559) |
| `x-learner` | 1.8057 (1.3411, 2.3093) | 1.7312 (1.3524, 2.1276) | 2.0331 (1.5602, 2.5310) | 1.9522 (1.5532, 2.3564) | -0.0041 (-0.0103, 0.0023) | 2.1154 (1.6339, 2.6159) | 2.0359 (1.6319, 2.4439) |
| `dr-learner` | 1.6694 (1.2003, 2.1556) | 1.5947 (1.2153, 1.9851) | 1.9485 (1.4750, 2.4526) | 1.8716 (1.4770, 2.2796) | -0.0042 (-0.0104, 0.0022) | 2.0624 (1.5730, 2.5591) | 1.9868 (1.5877, 2.3992) |
| `r-learner` | 1.6566 (1.1987, 2.1267) | 1.5899 (1.2135, 1.9752) | 1.9411 (1.4715, 2.4283) | 1.8681 (1.4708, 2.2691) | -0.0036 (-0.0099, 0.0024) | 2.0534 (1.5710, 2.5427) | 1.9773 (1.5752, 2.3827) |
| `dragonnet` | 1.2648 (0.8148, 1.7235) | 1.1963 (0.8342, 1.5853) | 1.5364 (1.0749, 2.0155) | 1.4641 (1.0792, 1.8671) | -0.0007 (-0.0068, 0.0058) | 1.7148 (1.2410, 2.2033) | 1.6451 (1.2483, 2.0687) |
| `outcome-ranking` | 1.8486 (1.3852, 2.3336) | 1.7771 (1.4038, 2.1661) | 2.0497 (1.5744, 2.5452) | 1.9765 (1.5942, 2.3790) | -0.0053 (-0.0117, 0.0009) | 2.1260 (1.6441, 2.6228) | 2.0533 (1.6628, 2.4576) |
| `random-200` | 0.2349 (0.0910, 0.3966) | 0.2281 (0.1065, 0.3579) | 0.4706 (0.2577, 0.6895) | 0.4578 (0.2840, 0.6443) | -0.0010 (-0.0067, 0.0045) | 0.7037 (0.4592, 0.9510) | 0.6814 (0.4808, 0.8797) |
<!-- itx:end:ieee-fraud-policy -->

**Reading the ranking table.** The S-learner and X-learner lead on the Qini, 0.4652 and
0.4578 with intervals that overlap every other meta-learner's, and agree on realised
value at a 20% budget to within 0.002 (1.9510 against 1.9522 dollars per transaction, doubly
robust). The
X-learner has the lowest PEHE, 22.99, on a population where the true effect is about -$0.37
for 96.5% of transactions and about +$74 for the rest, so every estimator's individual-effect
error is of the order of the effect it is trying to find. Dragonnet is last of the six on
every ranking column and its ATE error, 0.83, is far above the meta-learners' 0.06 to 0.30.
The `random-200` row's uplift is $2.35 a transaction at every budget, which is the value of
reviewing everybody, and the top decile of any meta-learner is worth seven times that.

**The risk queue's Qini is 0.4625 (0.3395, 0.5875), level with the S-learner's 0.4652.** Its
top decile is worth $17.97 a transaction and its realised value at a 20% budget is 1.9765,
and both are the highest numbers in their columns. It is the best-ranked baseline this
project has measured on any dataset, and the reason is in the design: review helps only
fraud, and a model of dollars retained is very nearly a model of which transactions are
fraud. That
is a different picture from ACIC, where the effect runs against risk, and it is what the
allocation below is about.

### What 1,000 analyst hours buy

`uv run itx allocate --estimator s-learner --hours 1000` prices five queues against the same
budget on the first seed. Two values per queue: what the simulation's written effects say the
queue was worth, and the doubly robust estimate this package would have reported on real
data, where the first number does not exist.

| Queue at 1,000 analyst hours | Reviewed | Hours used | True value | DR estimate (95% CI) | $/analyst hour |
|---|---|---|---|---|---|
| `uplift-knapsack` | 4,312 | 1,000 | $156,549 | $131,724 ($90,236, $172,246) | $157 |
| `uplift-rank-and-cut` | 4,248 | 1,000 | $156,357 | $127,033 ($84,739, $167,967) | $156 |
| `risk` | 4,281 | 1,000 | $169,398 | $148,739 ($108,747, $188,973) | $169 |
| `random` | 4,264 | 1,000 | $8,366 | $5,096 (-$3,395, $12,406) | $8 |
| `oracle` | 4,122 | 947 | $305,052 | $307,446 ($254,563, $356,773) | $322 |

Seed 11, 118,108 held-out transactions worth $15.9M. `oracle` ranks by the effect the
simulation wrote and is available only because the case is semi-synthetic. The S-learner here
is fitted at the default configuration rather than the tuned one in the table above. The DR
estimate's interval is a bootstrap over the test rows with each queue held fixed.

**The risk queue beats every fitted uplift model, and not narrowly.** $169,398 against the
S-learner's $156,549, and the S-learner is the best of them: the X-learner's knapsack buys
$154,457, the T-learner's $147,306, the DR-learner's $140,740. The gap holds at 250 hours
($112,543 against $99,186) and at 100 ($75,693 against $68,726). On this case a fraud team
that kept its risk model and ignored this repository would be $13,000 a thousand hours better
off than one that switched.

**The reason is the size of two spreads, and it is the same mechanism as the decile table
above.** Fraud is 3.5% of transactions and review is worth about +$74 on one of them and
-$0.37 on anything else. Finding the fraud is worth a step of $74; ordering correctly inside
the fraud, where the catch rate runs from 0.85 down to 0.30, is worth a slope across a factor
of three. A risk model learns the step from 177,000 untreated rows with a clean label. An
uplift model has to learn the step and the slope from the difference between two noisy arms, and it gets
the step slightly worse. The case was built so that the slope runs against risk, and it does;
it is just that the step dominates. "The most at-risk cases are the least movable" was true
here, and was not enough.

**The oracle says the value is there.** $305,052 with 53 hours unspent, because it reviewed
every one of the 4,122 fraudulent transactions in the split and then stopped, since every
remaining transaction has a negative effect. A perfect ranking is worth 1.8 times the risk
queue at every budget tried, $136,060 against $75,693 at 100 hours. So this is not
a case where uplift modelling has nothing to add. It is a case where these estimators, on
these features, could not reach what there was to add, and a reader deciding whether to build
one should know that those are different situations with the same table.

**What the package would have reported without the truth: that it cannot tell the queues
apart.** Every true value sits inside its doubly robust interval, and every fitted queue's
interval covers every other's: at 1,000 hours the risk queue's estimate is $148,739 ($108,747,
$188,973) against the knapsack's $131,724 ($90,236, $172,246), and at 250 hours the two
estimates are $82,409 and $84,968 with intervals $65,000 wide. Only the oracle and random
separate from the rest. An earlier version of this paragraph read the two points at 250 hours
as the estimator "calling the comparison for the uplift model and being wrong", which was a
conclusion from a $2,500 gap inside a $65,000 interval, in a repository whose first rule is
that a bare point is a defect; PLAN.md change 56 records it. The honest statement is the one
above: on 1,000 reviews out of 118,000 transactions, the doubly robust estimate is too wide to
rank these queues, and a fraud team deciding between them on real data would need the truth
this simulation happens to have, or a much larger review sample.

**The knapsack bought nothing over rank-and-cut.** $192 at 1,000 hours, $431 at 250, and
$1,560 *less* at 100. Review costs run from 9 to 18 minutes and are set by how much of the
record is missing, which has little to do with how much review is worth, so ranking by effect
per minute changes which marginal transactions get in and the marginal transactions are where
the estimate is least reliable. The machinery is correct arithmetic, tested to reduce to
rank-and-cut exactly at uniform cost, and this is the honest report of what it did on the
one dataset where it could do anything: on a cost that varies by a factor of two and is
unrelated to the effect, it is a wash.

**The cheap diagnostic says so in advance, now.** `uv run itx diagnose --dataset ieee-fraud`
fits one risk model and cuts the split into ten bands of predicted risk. Band 1, the
riskiest tenth, loses $32 a transaction unreviewed and $15 reviewed, a multiplier of 0.47x
and an uplift of $17.00 (12.51, 21.28). Bands 3 to 10 earn money unreviewed, and their
measured uplift is small and positive in most of them, because the few frauds the risk model
missed are each worth two hundred times the harm review does to a legitimate sale. The rank
correlation between a band's risk and its uplift is
0.467 (0.333, 0.892), and the verdict is the middle one of the three it can give: the
rankings agree in part, uplift modelling is worth measuring, and the benchmark says how
much. That is the right call. It is not the call the same command made on the first run,
when it announced with a confident interval that the effect ran against risk, because it
had the riskiest band at the wrong end of the predicted outcome. PLAN.md change 54 is the
record of that defect, the fix, and the test that keeps it fixed.

What this case does not say: anything about fraud review. Every dollar above is a
consequence of `simulate()`, and a reader who believes obvious fraud is *easier* for an
analyst to catch can set `CATCH_FALL` to zero, rerun the same three commands, and get a case
where the risk queue is optimal by construction rather than by a narrow margin.

## The budget slider

`uv run itx demo build` precomputes what a static page needs, and `demo/` is the page. Open
`demo/index.html` over any static file server. Move the slider and two things change together:
the list of units the budget treats, and the realised policy value that budget buys against
spending the same money at random.

It is static because the budget for this project is CA$25 (PLAN.md section 7), and that
constraint decided the design rather than being worked around. Every number the slider can
display exists in `demo/data/*.json` before the page opens; the only arithmetic in the browser
is reading an index out of an array and drawing a line. No backend, no dependencies, no CDN,
and nothing identifying: units travel as row numbers and predicted uplift, never as feature
values.

Three things about it are limits rather than features, and the page says all three itself:

- **It shows one split, where the tables above average five.** A slider has to move through a
  single ranking of actual units, not an average of five rankings of five different test sets.
  So its numbers sit near the tables' without equalling them. On ACIC at a 10% budget this
  split gives the risk-ranking baseline -0.2434 against the five-split average of -0.1971.
  Both are correct; they are not the same quantity.
- **Only the top 200 of each ranking ships.** Criteo's test split is 279,592 rows and the whole
  list would be a multi-megabyte download to render something nobody scrolls.
- **Costs are uniform**, so the cost-aware knapsack in `itx/policy/cost_aware.py` reduces to
  rank-and-cut here. Varying cost per unit is what the fraud worked case above is for, and
  `uv run itx allocate` is where the knapsack is compared against rank-and-cut on a budget of
  analyst hours rather than a headcount.

Live at [https://targeting.peterparker.ca](https://targeting.peterparker.ca), on Azure Static Web Apps' free tier,
published from this machine with the deployment token rather than from a workflow in the
repository; [docs/deploy.md](docs/deploy.md) records the route and the DNS record. Or open
`demo/index.html` over any static file server, which is the same page.

## What this does not do

- It does not identify effects without an experiment or a credible ignorability
  assumption. The sensitivity analysis quantifies how wrong that assumption can be before
  the targeting decision flips; it does not remove the assumption.
- It does not handle continuous or multi-valued treatments, or online allocation.
- The fraud worked case uses a simulated review intervention on public data and says so.
- Not yet done: the clean-environment rerun that closes week 8, which is running as this is
  written and has reproduced the first datasets exactly. Nothing above is a placeholder for
  it: the numbers reported are the numbers measured.
- **The fraud case's IEEE-CIS file is the one input this repository cannot fetch for you.**
  It sits behind a Kaggle account and accepted competition rules and may not be redistributed,
  so [docs/data/ieee-fraud.md](docs/data/ieee-fraud.md) gives three steps and the loader
  checks the committed digest once the file is in place.
- **The sensitivity table covers three datasets of five, and none of the three can fail it.**
  Criteo and Lenta need a full-size fit and are queued. Of the three that are there, two are
  randomised and the third is confounded only through covariates it records, so a clean sweep
  is the expected result rather than evidence the devices work. See
  [docs/estimators.md](docs/estimators.md).
- **Dragonnet is not tuned and is not a LightGBM model.** Every other estimator here shares
  one base learner so that differences between columns are differences between estimators.
  Dragonnet cannot, because the architecture is the thing being tested, so its column
  confounds "a neural network with a propensity head" with "two gradient-boosted trees". It
  also runs at the paper's published defaults, because the committed grid is over LightGBM's
  leaf size and tree width and a bespoke grid for the one estimator that could not use the
  shared one would be worse than none. Two more things its column confounds: it is fitted on
  at most 200,000 training rows, so on Lenta, Criteo and the fraud case it saw less than half
  the training split the LightGBM learners saw; and it does not standardise the outcome, so on
  a dollar-scale outcome like the fraud case's the outcome loss dwarfs the treatment loss the
  paper's defaults were tuned against, which is the likeliest reason its fraud row is last on
  every column. Both are stated in `docs/estimators.md`; neither is fixed in this version.
  PLAN.md changes 45 and 56.
- **Lenta has no licence.** Not from the publisher, not in the package that distributes it.
  This repository downloads it and redistributes nothing, but nobody reading this is being
  told their own use of that dataset is permitted. See
  [docs/data/lenta.md](docs/data/lenta.md).
- **The Criteo table is a 10% subsample**, 1,397,958 of 13,979,592 rows, drawn once with a
  committed seed and stratified so the arm and event rates are preserved exactly. The reason
  is wall-clock time, not memory: see PLAN.md change 23. The headline fit on the full file is
  not done yet.
- **Hyperparameters are selected on at most 50,000 training rows**, with the winner then
  fitted on all of them. That cap does not bind on Hillstrom, IHDP or ACIC. It changes which
  configuration Criteo picks and, measured on Hillstrom, changes what the pick is worth by
  -0.00015 in Qini against intervals about 0.004 wide. PLAN.md change 30 and
  [docs/estimators.md](docs/estimators.md) carry the numbers and the prediction they refuted.

## Install and run

Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra learners      # create the environment; the extra adds EconML and CausalML
uv run itx data pull          # download every dataset and verify its checksum (about 460 MB)
uv run itx benchmark --dataset hillstrom
uv run itx benchmark --dataset ihdp
uv run itx benchmark --dataset acic
uv run itx benchmark --dataset lenta
uv run itx benchmark --dataset criteo     # the committed 10% subsample
uv run itx benchmark --dataset ieee-fraud # the worked case; needs the one file you fetch yourself
```

`itx data pull hillstrom ihdp-train ihdp-test acic-x acic-zymu-1` fetches only the small
sets, about 20 MB, which is enough for the first three benchmarks. Criteo is 297 MB and
Lenta is 138 MB.

Or run the whole table in one command, which is what PLAN.md section 4 asks for and takes a
few hours on a laptop:

```bash
uv run itx benchmark --all
```

The DR and R learners come from EconML and CausalML, which is why they sit behind an extra:
without it the other three estimators and every metric still run, and asking for a wrapped
one says how to install it.

`itx benchmark` selects hyperparameters on the validation split, fits, evaluates on the
held-out test split, writes every per-seed number and the chosen configuration to
`results/<dataset>.json`, draws the figures into `docs/figures/`, and rewrites the table
above. **On your own data, pass `--no-tune`.** It uses one default configuration for every
estimator, which keeps the comparison fair at one ninth of the fitting cost, and on every
dataset here it would have given the same answer: fifty-four of fifty-four untuned means
land inside the tuned intervals. The tables above keep the tuned protocol because it was
fixed before the numbers were seen; [docs/rejected.md](docs/rejected.md) is the evidence and
the decision.

The Hillstrom run takes about twenty minutes, so two commands exist to avoid repeating it
when only the presentation has changed:

```bash
uv run itx report  --dataset hillstrom   # redraw the table from results/, no fitting at all
uv run itx figures --dataset hillstrom   # redraw the figures, refitting only the first seed
```

Long runs bank every finished fit as they go, so an interruption costs one fit rather than
the lot. A killed run leaves a checkpoint beside its results file, and `--resume` picks it up:

```bash
uv run itx benchmark --dataset lenta --resume
```

Three commands answer a question without running the benchmark at all:

```bash
uv run itx diagnose    --dataset acic   # is uplift modelling worth it here, for one model
uv run itx sensitivity --dataset acic   # E-value, Rosenbaum bound, negative control
uv run itx selection   --dataset acic   # where the two hyperparameter rules disagree
```

Resuming is opt-in rather than automatic, because a checkpoint written by an older version of
this package looks exactly like one written by the current version, and nobody wants a results
table whose rows came from two builds. A finished run deletes its checkpoint, so one on disk
always means an interrupted run.

A third command checks that a rerun still produces the committed numbers, which is what CI asserts
after each scheduled benchmark. It is not a file diff: the results file records how long
each fit took, and wall-clock time never reproduces, so a diff would fail every run for a
reason that has nothing to do with the numbers.

```bash
uv run itx compare committed.json results/hillstrom.json
``` Raw data is never committed; the SHA-256 of every download is, in
[`src/itx/data/checksums.sha256`](src/itx/data/checksums.sha256), so a download can be
verified without running any of this code:

```bash
cd data/raw && sha256sum -c ../../src/itx/data/checksums.sha256
```

```bash
uv run pytest               # fast tests, synthetic data only
uv run pytest --run-slow    # adds the tests that need a real download
uv run pytest --run-large   # adds Criteo and Lenta: 435 MB, and slower again
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
- [docs/rejected.md](docs/rejected.md): the one approach tried and rejected, with four
  measurements: per-estimator hyperparameter selection.
- [docs/data/hillstrom.md](docs/data/hillstrom.md),
  [docs/data/ihdp.md](docs/data/ihdp.md), [docs/data/acic.md](docs/data/acic.md),
  [docs/data/criteo.md](docs/data/criteo.md), [docs/data/lenta.md](docs/data/lenta.md),
  [docs/data/ieee-fraud.md](docs/data/ieee-fraud.md): one card per dataset, with source,
  licence, treatment definition, quirks and split seeds.
- [docs/deploy.md](docs/deploy.md): how the demo is hosted and how to redeploy it.

## Part of a portfolio

One of fifteen projects built over twelve months to make production ML work inspectable.

## How this was built

Design, methodology, evaluation choices and judgement are Peter Parker's. AI coding
assistants (Claude Code) were used for implementation and drafting, the way a senior
engineer uses them in 2026. Every number in the results table is reproducible from this
repository with one command, and that reproducibility is the evidence that matters.
