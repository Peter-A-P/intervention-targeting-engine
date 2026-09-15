# Hillstrom MineThatData email challenge

The classic teaching set for uplift modelling: a real randomised email campaign, large
enough for stable estimates and small enough to fit in a second.

| | |
|---|---|
| **Source** | <http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv> |
| **Published** | 2008, by Kevin Hillstrom (MineThatData) |
| **Licence** | No formal licence stated. Published openly for public use as a modelling challenge, with attribution to MineThatData. Not redistributed here: the loader downloads it. |
| **Size** | 64,000 customers, 12 columns, 3.8 MB |
| **SHA-256** | `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece` |
| **Loader** | `itx.data.hillstrom.load_hillstrom` |
| **Ground truth** | None. The individual effect is not recorded and cannot be. |

## The experiment

Customers who had bought in the previous twelve months were randomised into three equal
arms: a womens-merchandise email, a mens-merchandise email, and no email. Outcomes were
recorded over the following two weeks.

The loader keeps two of the three arms and drops the third entirely, so the treatment is
binary and the control genuinely means "no email". By default the treatment is the
womens-merchandise arm, which is the one most of the uplift literature reports. It is the
*smaller* of the two effects on all three outcomes (mens: visit 0.1828, conversion 0.0125,
spend 1.42 against womens' 0.1514, 0.0089, 1.08), which makes it the harder targeting
problem; an earlier version of this card said the opposite. Folding the unused arm into the
control would have been the other option and it would be wrong: those customers were
emailed.

Because assignment was random and equal across three arms, the probability of treatment
within any two of them is exactly 0.5. The loader records that as a known propensity
rather than estimating it, which is what makes this dataset the right place to check that
an estimator recovers an effect it should be able to see.

## Fields

| Column | Type | Notes |
|---|---|---|
| `recency` | numeric | Months since the last purchase |
| `history` | numeric | Dollars spent in the previous year |
| `history_segment` | numeric | An ordered binning of `history`, kept as its bin index 1 to 7 rather than as a category, because the order carries information |
| `mens`, `womens` | 0/1 | Bought in that department in the previous year |
| `newbie` | 0/1 | New customer in the previous twelve months |
| `zip_code` | category | Rural, Suburban, Urban. The source file spells it `Surburban`; the loader keeps the source spelling so the encoding is checkable against the raw file |
| `channel` | category | Phone, Web, Multichannel |

Categories are integer-encoded and named in `UpliftDataset.categorical`, so LightGBM is
told which columns are codes rather than being left to infer it.

## Outcomes

| Outcome | Mean, treated | Mean, control | Difference |
|---|---|---|---|
| `visit` (default) | 0.1514 | 0.1062 | 0.0452 |
| `conversion` | 0.0089 | 0.0057 | 0.0032 |
| `spend` | 1.08 | 0.65 | 0.43 |

Measured on the womens-email arm against no email, over all 42,693 rows. These match the
figures usually quoted for this dataset, and the loader test asserts the visit rates, so a
change in the source file fails the build rather than moving the results table.

`visit` is the headline outcome. `conversion` has about 380 events in total, which is too
few for a stable Qini at any budget: the interval swamps the point estimate. `spend` is a
continuous outcome dominated by a handful of large orders, useful for checking that the
regression path works and not much else.

## Known quirks

- **The two email arms are not comparable to each other** without care: the campaign
  targeted merchandise at people who had bought in that department, so the arms differ in
  who they were relevant to, not only in what they said.
- **The outcome window is two weeks.** An effect that only shows up later is invisible
  here, and nothing in the data says whether one exists.
- **`spend` is zero for the overwhelming majority of customers** and the conditional
  distribution above zero is heavy-tailed. Treat the spend results as a demonstration of
  the continuous-outcome path rather than as a finding.
- **`history_segment` is a deterministic function of `history`.** It is kept because it is
  in the source data and a tree can use the binning directly, but the two columns are not
  independent evidence.

## Splits

The committed protocol (PLAN.md section 4): 60/20/20, stratified on treatment arm crossed
with the binary outcome, five seeds (11, 23, 37, 53, 71). Metrics are computed on the test
part only.
