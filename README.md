# Intervention Targeting Engine

Tells an organisation which customers, cases or patients to spend a limited intervention
budget on: only the ones whose outcome the intervention actually changes. Retention
offers, outreach, fraud review, clinical follow-up: a large share of every such budget
goes to people who would have behaved the same way regardless, and this finds them, and
shows the intervention list changing as the budget moves.

**Status: week 4 of 8.** Five estimators of the seven, benchmarked on all five datasets. The
numbers below are real and reproducible; two estimators and the policy, sensitivity and demo
work are still to come. Build plan: [PLAN.md](PLAN.md).

Four datasets, four different answers, and that is the finding. On ACIC the ordinary approach
of ranking by risk is ten times worse than picking names out of a hat. On Criteo's 14 million
randomised rows it is as good as every uplift model here. On Hillstrom the uplift models win
and risk ranking is ambiguous. On Lenta nothing, including the uplift models, is
distinguishable from random at all. Whether any of this is worth building is a question about
your data, and the [risk-decile diagnostic](#criteo-uplift-1398m-randomised-ad-impressions)
below answers it in one table before anybody fits a model.

## Results

Every table on this page is written by `uv run itx benchmark --dataset <name>` and is
never edited by hand. Each number is the mean across five committed split seeds, with the
95% bootstrap interval alongside it. Hyperparameters are chosen per seed on a validation
split from a grid that is identical for every estimator, and the test split is never
touched until the numbers below are computed. There are no bare point estimates here by
design: a Qini without an interval is treated in this repository as a defect.

Each dataset carries two tables. The first is the ranking metrics: how good the ordering is.
The second is **what each ranking buys**, which is the question a budget holder is actually
asking. Its numbers are the extra outcome per head of the whole population from treating the
top b% rather than treating nobody, estimated two ways: by inverse-probability weighting,
which relies only on knowing how treatment was assigned, and by a doubly robust estimator,
which adds outcome models and survives either one of the two being wrong. Both are reported
because a disagreement between them is worth seeing. On a binary outcome a gain of 0.017
means seventeen extra events per thousand people in the population, and at a budget of 100%
the number is the average treatment effect.

### Hillstrom, 42,693 customers, randomised email campaign

<!-- itx:table:hillstrom -->
| Estimator | Qini (95% CI) | Normalised AUUC | uplift@10% | uplift@20% | uplift@30% | Calibration slope | Calibration error |
|---|---|---|---|---|---|---|---|
| `s-learner` | 0.0041 (0.0019, 0.0062) | 0.2609 (0.1984, 0.3210) | 0.0893 (0.0404, 0.1358) | 0.0828 (0.0494, 0.1160) | 0.0793 (0.0530, 0.1066) | 0.6926 (0.3295, 1.0444) | 0.0180 (0.0157, 0.0370) |
| `t-learner` | 0.0031 (0.0009, 0.0053) | 0.2442 (0.1803, 0.3047) | 0.0970 (0.0465, 0.1479) | 0.0779 (0.0441, 0.1120) | 0.0678 (0.0405, 0.0948) | 0.2987 (0.0901, 0.5153) | 0.0437 (0.0347, 0.0594) |
| `x-learner` | 0.0031 (0.0010, 0.0053) | 0.2452 (0.1827, 0.3065) | 0.0831 (0.0351, 0.1321) | 0.0788 (0.0448, 0.1124) | 0.0743 (0.0478, 0.1026) | 0.3477 (0.1011, 0.5944) | 0.0365 (0.0287, 0.0529) |
| `dr-learner` | 0.0028 (0.0007, 0.0051) | 0.2402 (0.1757, 0.3031) | 0.0841 (0.0339, 0.1344) | 0.0738 (0.0392, 0.1079) | 0.0736 (0.0463, 0.1014) | 0.2563 (0.0551, 0.4590) | 0.0463 (0.0377, 0.0625) |
| `r-learner` | 0.0029 (0.0007, 0.0051) | 0.2413 (0.1795, 0.3017) | 0.0920 (0.0417, 0.1434) | 0.0759 (0.0416, 0.1096) | 0.0745 (0.0473, 0.1012) | 0.2381 (0.0501, 0.4221) | 0.0508 (0.0419, 0.0667) |
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
<!-- itx:end:hillstrom-policy -->

The outcome ranking is the ordinary way this job is done: model who is likely to respond,
spend the budget from the top of that list. Its Qini interval contains zero, so on this
dataset it has not been shown to beat picking at random. All five estimators' intervals
exclude zero.

At a budget covering 20% of the population, the S-learner's targeted group shows an
8.3-point difference in visit rate between arms, against random targeting's 4.5. The
outcome ranking manages 5.2, less than a quarter of the way from doing nothing clever to
doing this properly. The five estimators' intervals overlap heavily, so the honest reading
is that they are distinguishable from the two baselines and not from each other.

The calibration columns tell a different story from the ACIC ones below, and the difference
is the point. Every estimator here has a slope well under 1, meaning its predictions are
more spread out than the uplift that actually materialises: on a dataset whose real effect
is small and fairly uniform, a flexible model finds heterogeneity that is mostly noise. The
S-learner is the least wrong of them at 0.69, for the same reason it is the most wrong on
ACIC at 1.53. It shrinks predicted effects toward a constant, which is a liability where the
effect genuinely varies and a virtue where it does not.

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
| Estimator | IPW gain at 10% | DR gain at 10% | IPW gain at 20% | DR gain at 20% | IPW gain at 30% | DR gain at 30% |
|---|---|---|---|---|---|---|
| `s-learner` | 1.7187 (-0.1976, 4.7082) | 0.2030 (-0.3699, 0.7049) | 3.5632 (-0.2197, 9.5374) | 1.0295 (0.0006, 2.2763) | 4.7288 (0.1931, 11.8471) | 1.5607 (0.4896, 2.9262) |
| `t-learner` | 1.4916 (-0.1783, 4.8095) | 0.2627 (-0.2684, 0.6710) | 2.6797 (0.0485, 9.4154) | 0.5708 (-0.1205, 2.1799) | 7.2076 (0.7112, 15.1902) | 1.5599 (0.3319, 3.0994) |
| `x-learner` | 0.8674 (-0.2321, 3.0907) | 0.2087 (-0.3856, 0.5935) | 2.1143 (0.1205, 5.0318) | 0.6860 (-0.0159, 1.1697) | 4.4602 (0.5047, 10.5713) | 1.1157 (0.2666, 1.8676) |
| `dr-learner` | 1.3281 (-0.3865, 4.8262) | 0.2673 (-0.0358, 0.6035) | 3.1214 (-0.0152, 7.8435) | 0.7127 (0.0599, 1.1518) | 4.6005 (0.6226, 10.9085) | 0.9238 (0.2014, 1.7903) |
| `r-learner` | 1.7822 (-0.2609, 4.7336) | 0.4317 (-0.0340, 0.7649) | 2.5409 (-0.2897, 8.0141) | 0.8280 (0.2038, 1.2858) | 4.3654 (-0.0626, 11.2947) | 1.1517 (0.4078, 1.8167) |
| `outcome-ranking` | 1.6706 (-0.8145, 6.1431) | 0.3756 (-0.0438, 1.0499) | 3.4835 (-0.6902, 9.4589) | 0.8084 (0.2226, 1.6685) | 4.5607 (-0.4979, 12.7002) | 1.2868 (0.5625, 2.3125) |
| `random-200` | 2.2266 (-0.2722, 7.9707) | 0.5154 (0.0215, 1.4131) | 4.4459 (-0.0230, 11.1432) | 1.0135 (0.2956, 2.0037) | 6.7272 (0.5701, 14.1585) | 1.5292 (0.6507, 2.6437) |
<!-- itx:end:ihdp-policy -->

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
the **best Qini coefficient of the three** and buys 0.05 at a 10% budget, against random
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
<!-- itx:end:acic-policy -->

This is where the trap is at its worst, and it is worth reading the two baseline rows
against the estimators:

| ACIC 2016 | Qini | Uplift at a 10% budget |
|---|---|---|
| `x-learner` | 0.2273 | 8.41 |
| Random targeting | -0.0000 | 3.42 |
| Outcome ranking | **-0.0233** | **0.34** |

Spending the budget on the highest-risk tenth of this population buys 0.34. Spending it on a
random tenth buys 3.42. Ranking by risk is not leaving return on the table here, it is ten
times worse than not thinking about it at all, and its Qini is negative rather than merely
unimpressive. That happens when the people most likely to have the outcome are the people
least susceptible to the intervention, which is the normal shape of a fraud queue or a
clinical follow-up list.

![Qini curves on ACIC 2016](docs/figures/qini-acic.png)

Five estimators sit well above the diagonal. The outcome ranking, in brown, is below it for
the first sixty percent of the population: for any budget in that range, the ordinary
approach delivers less than picking names out of a hat.

![Calibration on ACIC 2016](docs/figures/calibration-acic.png)

The calibration columns are the only ones in the table that notice magnitude. Qini, AUUC and
uplift at k are all unchanged if every prediction is multiplied by a constant, so a model
that ranks perfectly and predicts effects half the size they should be scores identically to
one that gets them right, and then forecasts half the return. The S-learner's calibration
slope of 1.53 is the only one whose interval excludes 1: its predictions are too compressed,
and realised uplift varies half as much again as it says.

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
<!-- itx:end:lenta-policy -->

**Nothing here beats random targeting, and that is the result.** Every Qini interval in the
table above contains zero, for all five estimators and for both baselines. On the Qini they
are indistinguishable from each other and from picking names out of a hat.

| Lenta | Qini | uplift at a 10% budget |
|---|---|---|
| `s-learner` | 0.0006 (-0.0003, 0.0016) | 0.0184 (0.0008, 0.0357) |
| `dr-learner` | 0.0004 (-0.0006, 0.0014) | 0.0163 (-0.0003, 0.0337) |
| Outcome ranking | 0.0005 (-0.0005, 0.0014) | 0.0075 (-0.0118, 0.0267) |
| Random targeting | -0.0000 (-0.0008, 0.0007) | 0.0073 (-0.0034, 0.0175) |

There is exactly one thread worth pulling. The S-learner is the only estimator whose realised
uplift excludes zero at every budget: 0.0184 at 10%, 0.0132 at 20%, 0.0118 at 30%, against
random targeting's flat 0.0074. That is about two and a half times random at a tight budget,
and it is consistent across all three budgets and all five seeds. But its interval overlaps
random's heavily, and the Qini, which integrates the whole curve rather than one cut of it,
declines to confirm anything. So it is a hint and it is reported as a hint.

**Why this dataset cannot answer the question.** The average effect is 0.75 percentage points
on a 10.3% base rate. The test split is 137,406 rows and only about a quarter of them are
controls, so roughly 34,000 control rows carry the comparison. Detecting *heterogeneity*
inside an effect that small, from that many controls, is a lot to ask. Criteo has twice the
effect in absolute terms and twice the test rows with far more events, which is why its
intervals are tight enough to separate things and Lenta's are not.

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
<!-- itx:end:criteo-policy -->

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
the same baseline has a *negative* Qini and buys a tenth of what picking names out of a hat
buys. Same baseline, same code, opposite verdict.

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
| Hillstrom | about 6x | -0.24 | roughly two thirds of the way |
| ACIC 2016 | effect runs against risk | positive | worse than random |

So the honest claim this project can make is not "the intervention list is never the risk
list". It is that the two lists differ by an amount you cannot guess in advance and can
measure cheaply, and that the cost of assuming they agree runs from nothing to ten times worse
than doing nothing. On a fraud queue or a clinical follow-up list, where the most at-risk
cases are often the least movable, ACIC is the relevant picture. On an advertising set where
the high-risk group is a hundred times more likely to act, Criteo is. Building an uplift model
is worth it in the first case and close to pointless in the second, and a decile table like
the one above says which one you are in before anybody fits a meta-learner.

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

## What this does not do

- It does not identify effects without an experiment or a credible ignorability
  assumption. The sensitivity analysis quantifies how wrong that assumption can be before
  the targeting decision flips; it does not remove the assumption.
- It does not handle continuous or multi-valued treatments, or online allocation.
- The fraud worked case uses a simulated review intervention on public data and says so.
- Not yet built, in schedule order: realised policy value under a budget; Rosenbaum
  bounds and E-values; Dragonnet; the budget-slider demo. Nothing above is a placeholder
  for them: the numbers reported are the numbers measured.
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
above. Add `--no-tune` to skip selection and use the default configuration, which is several
times faster.

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
- [docs/data/hillstrom.md](docs/data/hillstrom.md),
  [docs/data/ihdp.md](docs/data/ihdp.md), [docs/data/acic.md](docs/data/acic.md),
  [docs/data/criteo.md](docs/data/criteo.md), [docs/data/lenta.md](docs/data/lenta.md): one
  card per dataset, with source, licence, treatment definition, quirks and split seeds.

## Part of a portfolio

One of fifteen projects built over twelve months to make production ML work inspectable.

## How this was built

Design, methodology, evaluation choices and judgement are Peter Parker's. AI coding
assistants (Claude Code) were used for implementation and drafting, the way a senior
engineer uses them in 2026. Every number in the results table is reproducible from this
repository with one command, and that reproducibility is the evidence that matters.
