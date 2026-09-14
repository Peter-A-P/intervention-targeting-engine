# IEEE-CIS Fraud Detection, as a semi-synthetic worked case

**The features are real. The treatment and its effect are invented.** Nothing computed from
this dataset is a measured fact about fraud review, and the only reason it is in this
repository is to show the allocation logic on a problem shaped like the ones the project is
aimed at (PLAN.md section 3).

## What is real

| | |
|---|---|
| Source | [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data), Vesta Corporation via Kaggle |
| File used | `train_transaction.csv`, renamed `ieee_train_transaction.csv` |
| Rows | 590,540 transactions |
| Columns in the file | 394 |
| Columns used | 44: `TransactionAmt`, `card1`-`card6`, `addr1`-`addr2`, `dist1`-`dist2`, the two email domains, `C1`-`C14`, `D1`-`D15` |
| Label | `isFraud`, 3.50% positive (20,663 transactions) |
| Value at stake | $79.7M of transactions, of which $3.08M is fraudulent |
| Licence | Kaggle competition rules. Non-commercial and academic use, not redistributable |

The `V1`-`V339` block is dropped. It is 339 anonymised engineered columns that treble the fit
time and that nobody reading this case can interpret; what is kept is the part a fraud analyst
would recognise. `TransactionID` is dropped because it is an identifier, and `isFraud` is never
offered to a model, since it decides the counterfactual and a model given it would be reading
the answer.

## Getting it

This is the only file in the project that is not downloaded. Kaggle serves competition data to
an authenticated account that has accepted the competition rules, which is not something a
loader can do, and the licence forbids redistributing it, so no mirror is offered either.

1. Sign in to Kaggle and accept the rules at the competition page above.
2. Download `train_transaction.csv` from the Data tab. The test split is not used: it has no
   labels.
3. Put it at `data/raw/ieee_train_transaction.csv`.

It is then checked against the committed digest in `src/itx/data/checksums.sha256` exactly
like every other file here, so a truncated or wrong copy will not be used. Running anything
without it raises `ManualDownloadRequiredError` with these instructions rather than an HTTP
error that looks like a broken download.

```
3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642  ieee_train_transaction.csv
```

## What is invented

All of it is in `simulate()` in `src/itx/data/ieee_fraud.py`, and the constants are module
level so the effect function is published rather than described.

**Assignment.** Half the transactions are reviewed, by a fair coin that depends on nothing. So
the propensity is known and exactly 0.5. Real review queues are not randomised, and that is
what makes real fraud data hard; this case is about allocation under a budget rather than about
identification, and the repository already has ACIC and IHDP for the other argument.

**Outcome**, in dollars of value retained per transaction:

| | Not reviewed | Reviewed |
|---|---|---|
| Fraudulent | `-amount` (charged back) | `0` if caught, `-amount` if not |
| Legitimate | `+0.03 * amount` (margin earned) | `0` if wrongly declined, `+0.03 * amount` if not |

**The two probabilities**, both functions of a fraud signal built from `TransactionAmt`, `C1`
and `D1` and scaled to `[0, 1]`:

- caught: `0.85 - 0.55 * signal`
- wrongly declined: `0.01 + 0.14 * signal`

**Review cost**, which is the point of the case: `8 + 22 * (share of numeric fields missing)`
analyst minutes, so a review takes between 9.2 and 18.4 minutes depending on how complete the
record is. This is the only dataset in the project where treating a unit has a per-unit cost,
and therefore the only one where the cost-aware knapsack does anything that rank-and-cut does
not.

## The assumption the case rests on

**Review is hardest on exactly what looks riskiest.** The catch rate falls from 0.85 to 0.30 as
the fraud signal rises, on the argument that obvious fraud is stopped by rules before it
reaches a human queue, so what arrives is the practised kind.

That single choice is what makes this a worked case for this repository rather than a
demonstration that expensive things are worth doing. It puts the highest-risk transactions in
the lost-causes quadrant. Change `CATCH_FALL` to zero and the case becomes one where risk
ranking is optimal, which is a perfectly reasonable thing to believe about some review
operations and is a one-line experiment.

## Consequences worth knowing before reading any number from it

- **96.5% of the population are sleeping dogs.** Review helps only the 3.50% that are
  fraudulent and harms every legitimate transaction a little, so the effect is negative for
  almost everybody. That is an unusually stark version of the four-quadrant picture and it is
  a property of the simulation, not of card fraud.
- **Reviewing everything is worth about $2.32 a transaction**, and reviewing the right 2% by
  the true effect is worth more than reviewing all of it, because the harm done to the other
  98% cancels most of the gain.
- **The per-unit truth is the expected effect, not a realised one.** The simulation draws one
  coin per transaction for catching and one for declining, but `true_effect` records
  `amount * catch_probability` rather than the draw. An effect defined by a single coin flip
  would be unlearnable and a PEHE scored against it would be measuring the coin.
- **The outcome has a long left tail.** The largest single transaction is $31,937, so one
  fraudulent transaction can move a mean. Totals in dollars are the honest unit here, which is
  why `itx allocate` reports them that way rather than per head.
