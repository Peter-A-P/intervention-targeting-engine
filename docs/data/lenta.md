# Lenta SMS campaign

687,029 grocery customers, an SMS campaign, and 194 columns with no data dictionary. It is
the widest and messiest dataset in the project, and both of those are the reason it is here:
Hillstrom has eight tidy features and Criteo has twelve anonymous ones, and neither looks
much like a real customer table.

| | |
|---|---|
| **Source** | <https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz> |
| **Published** | By Lenta, a Russian grocery retailer, distributed through the scikit-uplift project |
| **Licence** | **None stated.** See below. Not redistributed here: the loader downloads it from the same bucket scikit-uplift reads. |
| **Size** | 687,029 rows, 195 columns, 138 MB compressed |
| **SHA-256** | `b531544f6c072d22f232d91e20ffcaca265acf224e02687502a40e6d56135682` |
| **Loader** | `itx.data.lenta.load_lenta` |
| **Ground truth** | None. The individual effect is not recorded and cannot be. |

## The licence question, stated plainly

There is no licence. Not on the publisher's bucket, not in scikit-uplift's
`sklift/datasets/datasets.py`, and not on its documentation page for `fetch_lenta`. PLAN.md
section 3 said "check licence in the package"; the check was done and the answer is that
there is not one (change 25).

What this repository does is download the file from the publisher's own bucket at run time
and redistribute nothing, which is exactly what scikit-uplift does and what every other
loader here does. What it cannot do is tell a reader that their own use is licensed. Anyone
wanting to use this data for anything beyond reproducing this benchmark has nothing to rely
on and should treat that as a real constraint rather than an oversight to work around.

This is the only dataset in the project in that position. Hillstrom has no formal licence
either but was published openly as a public modelling challenge with an explicit invitation
to use it; Criteo, IHDP and ACIC all state terms.

## The experiment

Customers were sent an SMS campaign; `group` records `test` or `control`. Store visits were
recorded over the campaign window as `response_att`.

The arms are 75/25: 515,892 treated against 171,137 control. The design probability is not
documented anywhere, so unlike Hillstrom's 0.5 and Criteo's 0.85 the loader leaves the
propensity to be estimated rather than declaring it known. The evidence that it was
randomised at all is the covariate balance below, which is strong, and estimating a
propensity that is really constant costs a little precision and no correctness, which is the
right way round for an assumption this thin.

## Two columns are leaks, and balance is what proved it

`response_sms` and `response_viber` sit in the feature block with names that could plausibly
mean "responded to some earlier campaign". They do not.

In a randomised trial every pre-treatment covariate has the same mean in both arms up to
sampling noise. Across the 192 numeric columns the median absolute standardised mean
difference is 0.011 and the worst ordinary column is 0.025:

| Column | Standardised mean difference between arms |
|---|---|
| `response_sms` | **0.198** |
| `response_viber` | **0.068** |
| `k_var_count_per_cheque_1m_g34` | 0.025 |
| `crazy_purchases_cheque_count_12m` | 0.025 |
| ... | ... |
| median over 192 columns | 0.011 |

One column is eighteen times the median and three times the next worst, and the second worst
is the only other column whose name suggests a campaign response. They are responses to the
campaign's own delivery channels, so they are consequences of the treatment and cannot be
features; an estimator handed `response_sms` would be told part of the answer.

Both are dropped. The diagnostic lives in `itx.metrics.balance` rather than in a script that
was run once, so the same check runs against every dataset. After dropping them the worst
remaining imbalance is 0.025.

This is also the cleanest demonstration in the project of why a randomised dataset is worth
carrying: on an observational dataset this check is unavailable, and the leak would have
been found only by noticing an implausibly good result, or not at all.

## Fields

191 features after the drops. Names follow a scheme that is never documented: `k_var_*` for
coefficients of variation, `*_15d` / `*_1m` / `*_3m` / `*_6m` / `*_12m` for lookback windows,
`_g*` suffixes for product groups. Nothing states what group 34 or group 49 is.

| Column | Type | Notes |
|---|---|---|
| `gender` | category | Cyrillic in the source. Female 433,448, male 243,910, and 9,671 unknown |
| `main_format` | category | Already 0/1 in the file, so its coding is its own |
| `age`, `children`, `months_from_register`, ... | numeric | The interpretable handful |
| `k_var_*`, `sale_sum_*`, `cheque_count_*`, ... | numeric | The undocumented majority |
| `group` | 0/1 | **Treatment.** Not a feature |
| `response_att` | 0/1 | **Outcome.** Not a feature |
| `response_sms`, `response_viber` | 0/1 | **Dropped.** Post-treatment, see above |

`gender` folds the file's explicit "not determined" level (1,090 rows) together with its
8,581 nulls into one unknown code. They mean the same thing to a model, and 1,090 rows
cannot support a level of their own at a split.

## Outcomes

| Outcome | Mean, treated | Mean, control | Difference |
|---|---|---|---|
| `response_att` | 0.11013 | 0.10258 | 0.00755 |

A 0.75-point lift on a 10.3% base rate. Smaller in relative terms than Hillstrom's email
effect and much larger in absolute terms than Criteo's, which makes it a useful third point
on the scale.

## Known quirks

- **150 of the 191 feature columns have missing values**, and they are left missing.
  LightGBM routes NaN down its own branch at every split, so no imputation is needed and
  none is done: an imputation rule chosen here would put a modelling decision made in the
  loader into every estimator's input, and the estimator column would report partly on that
  decision. The worst column, `k_var_sku_price_15d_g49`, is absent for 72% of rows and is
  kept on the same reasoning.
- **The missingness is almost certainly not random.** A coefficient of variation over a
  15-day window cannot exist for a customer who shopped once in that window, so "missing"
  here usually encodes low activity. That is information, and it is one more reason not to
  impute it away, but it also means these columns are not missing-at-random in any sense a
  standard imputation method assumes.
- **No data dictionary.** No feature-importance story here can be interpreted beyond the
  handful of named columns, and none is told.
- **Nothing states the randomisation design.** The balance table is the only evidence, and
  it is good evidence, but it is evidence rather than documentation.
- **The campaign window is not documented either.** An effect that only appears later is
  invisible here and nothing in the data says whether one exists.

## Splits

The committed protocol (PLAN.md section 4): 60/20/20, stratified on treatment arm crossed
with the binary outcome, five seeds (11, 23, 37, 53, 71). Metrics are computed on the test
part only.
