# Architecture

Three figures: the system as a whole, how a results table is produced, and the budget decision
the demo makes. Each names the module that implements what it shows; [PLAN.md](../PLAN.md)
says why, and [estimators.md](estimators.md) says where each estimator breaks.

## The system

![Six public datasets are loaded with checksums and split on committed seeds; five meta-learners, Dragonnet and two baselines rank people through one interface; the metrics score every ranking with bootstrap intervals, the policy layer turns a ranking into an allocation, the sensitivity analysis says how much hidden confounding would flip it, and the benchmark runner writes the results and the demo.](architecture/system.svg)

Six public datasets (`itx.data`, each with a data card in [data/](data/) and a checksum in
`checksums.sha256`) are split on five committed seeds into train, validation and test
(`itx.data.splits`, `itx.bench.seeds`). Eight estimators sit behind one interface
(`itx.estimators`): the S, T, X, DR and R meta-learners on a shared LightGBM base, Dragonnet in
PyTorch, and the random and outcome-ranking baselines; their hyperparameters are chosen per
seed on the validation split from a grid that is identical for every estimator
(`itx.bench.grid`). The metrics (`itx.metrics`) score every ranking on the test split with
bootstrap intervals: Qini, normalised AUUC, calibration and the risk-decile table. The policy
layer (`itx.policy`, `itx.metrics.ground_truth`) turns a ranking into an allocation at a budget,
top-N or cost-aware, and estimates its realised value by inverse-probability weighting and a
doubly robust estimator; the sensitivity analysis (`itx.sensitivity`) reports how strong a
hidden confounder would have to be before the decision flips. `itx benchmark`
(`itx.bench.runner`, `itx.bench.table`) runs it all, averages the seeds, writes
`results/<dataset>.json`, and from those files the README tables and figures are written and
the demo is built (`itx.demo`) for targeting.peterparker.ca on Azure Static Web Apps.

## How a results table is produced

![For each dataset and each of five committed seeds the data is split, hyperparameters are chosen on the validation split from one grid, the estimator is fitted on train and scored once on test with bootstrap intervals, the seeds are averaged, and the table is written by one command; a clean-environment rerun reproduces every table at zero tolerance.](architecture/measurement.svg)

Each dataset is either randomised, so group averages are trustworthy, or simulated with known
individual effects (IHDP and ACIC), so per-person error can be measured. For each of five
committed seeds the rows are split; the hyperparameters are chosen on the validation split; the
estimator is fitted on train and ranks the test split, which is touched once; the ranking is
scored by Qini, AUUC and policy value by IPW and DR beside the random and outcome-ranking
baselines; and every number gets a 95% bootstrap interval over rows
(`itx.metrics.bootstrap`). The runner averages the five seeds and reports the interval as the
mean of the per-seed intervals, and `itx benchmark --dataset <name>` writes the table; nothing
in the README is edited by hand. A clean-environment rerun from a fresh clone reproduces
every table at zero tolerance, and it was that rerun that caught the loader defect that had
made five published rows unreproducible.

## From a ranking to a budget decision

![Each person gets a predicted effect and a predicted risk; at a chosen budget the top of each ranking is treated, by count or by cost, and the realised policy value says what each list buys, against random targeting; the risk-decile table says in advance when ranking by risk is the trap and when it is not.](architecture/decision.svg)

Every person in the test split gets two scores: the predicted effect of acting from an uplift
estimator, and the predicted risk of the bad outcome from the outcome-ranking baseline. At a
budget, 10, 20 or 30 percent of the population on the demo's slider, each ranking's top is
treated, by count when every intervention costs the same or as the best affordable set when
costs differ (`itx.policy.cost_aware`, `itx.bench.allocate`). The realised policy value
(`itx.metrics.ground_truth`) is the extra outcome per head of the population from following
each list, by IPW and DR, against 200 random rankings. The risk-decile table
(`itx.metrics.risk_deciles`) is the one-model check that says in advance whether ranking by
risk is the trap on this data: on ACIC it is ten times worse than random, on Criteo it matched
every uplift model. The demo shows both lists and both numbers re-ranking as the budget moves.
