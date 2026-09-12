"""The committed hyperparameter grid, and selection on the validation split.

PLAN.md section 4: hyperparameters are tuned on validation with a small fixed grid, the
grid is committed, and it is identical across estimators. Identical matters more than it
sounds. If each estimator got its own grid, or its own number of candidates, the results
table would be reporting how much search each one was given rather than how good each one
is, and the estimator column would quietly become a compute column.

## What is tuned, and why these two knobs

``min_child_samples`` and ``num_leaves``, eight combinations. The learning rate and the
number of boosting rounds are held fixed so that every candidate sees the same model
capacity from the boosting side, and because tuning all four turns a fixed grid into a
search whose cost depends on the dataset.

``min_child_samples`` is in the grid because week 1 found it deciding whether an estimator
works at all: at 100 the S-learner on IHDP's 448 training rows never split on the treatment
and returned exactly zero uplift, while at 20 it recovered the effect. One constant cannot
serve a 747-unit dataset and a 64,000-unit one, so the grid picks per dataset and per seed.

The 200 candidate was added in week 3, when the DR-learner arrived and the grid's old
ceiling of 60 turned out to be too low for it. Its final stage regresses on a doubly robust
pseudo-outcome, which carries the inverse-propensity correction's variance as well as the
outcome's: on 2,400 training rows its PEHE was 1.28 at a leaf size of 20, worse than
predicting a constant, and 0.40 at 200. That is the method's known weakness rather than a
defect in it, and the grid has to be wide enough for the weakness to be managed, or the
results table reports how badly suited one shared constant was to one estimator. Widening it
for every estimator rather than for the one that needed it keeps the comparison level.

## What it is selected on, and the honest problem with that

The validation-split Qini coefficient.

This deserves a flag, because this project's headline finding is that the Qini is a poor
referee: a plain risk model can score well on it while buying nothing. Selecting on it
looks like using the metric the repository criticises.

The distinction is between selecting and refereeing, and it is a real one. Inside this
function the comparison is between six configurations of *the same estimator* on *the same
data*: nothing here can win by being a different kind of model that games the curve,
because every candidate is the same kind of model. The Qini is being used as a stable
ranking signal, and it is the most stable one available without ground truth, since it
integrates the whole ranking rather than a single cutoff. Across estimators, where the
model class itself varies, the Qini stops being safe, and that comparison is never made
here: it is made on the test split, against both baselines, and from week 5 on the realised
policy value rather than the curve.

The alternative, selecting on realised policy value at the operating budget, is better
aligned with the decision. It arrived in week 5 as :func:`policy_value_rule`, and it is
available rather than default. Which of the two a run used is recorded on the selection, and
`itx selection` fits both on a dataset and reports where they disagree.

The default did not move, and the reason is worth stating rather than assuming. The policy
value at a single budget reads one cutoff of the validation ranking, so it is a noisier
signal than a coefficient integrating the whole curve, and on the small validation splits
this benchmark uses that noise is the dominant term. The argument for selecting on the
decision rather than the curve is a good one about bias, made against a variance cost that
happens to be larger here. What settles it is the measurement in `docs/estimators.md`, not
the argument.

## Candidates a dataset cannot support are removed before selection

A leaf size of 200 on IHDP's 448 training rows leaves room for two leaves at most, so the
model can barely split at all and its predicted uplift collapses toward zero. That is not a
hyperparameter setting, it is a broken model, and offering it to a selection rule running on
a 149-row validation split means the rule will occasionally pick it.

That is not hypothetical either. Adding the 200 candidate for the DR-learner's benefit moved
the S-learner's IHDP PEHE from 0.57 to 1.28, entirely because one seed in five selected it
and produced a near-degenerate fit. So :func:`applicable_grid` drops candidates whose leaf
size exceeds a quarter of the training rows, which is the point below which four leaves stop
being possible. The grid stays identical across estimators, which is what the fairness rule
actually requires; what varies is the dataset, and a dataset is allowed to rule out a
configuration it cannot fit.

Nothing selected here is ever reported. The validation split exists so that the test split
stays untouched, and a number computed during selection is not a result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from itx.bench.seeds import TIE_SEED
from itx.data.splits import stratified_subsample
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig
from itx.metrics.qini import qini_coefficient
from itx.policy.policy_value import fit_nuisances, gain_key, policy_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itx.estimators.base import BaseUpliftEstimator
    from itx.types import FloatArray, Split, UpliftDataset

#: Candidate leaf sizes. 5 is small enough for IHDP's 448 training rows to split at all;
#: 200 is heavy enough to hold the DR-learner's final stage back from chasing the variance
#: of its own pseudo-outcome on a small dataset.
MIN_CHILD_SAMPLES: tuple[int, ...] = (5, 20, 60, 200)

#: Candidate tree widths.
NUM_LEAVES: tuple[int, ...] = (15, 31)

#: The committed grid: every combination of the two, in a fixed order, so the same
#: candidate wins a tie on every machine.
GRID: tuple[BaseLearnerConfig, ...] = tuple(
    DEFAULT_CONFIG.with_(min_child_samples=minimum, num_leaves=leaves)
    for minimum in MIN_CHILD_SAMPLES
    for leaves in NUM_LEAVES
)

#: An estimator is built from a config and a seed, so one factory serves both the tuning
#: loop and the final fit.
type EstimatorFactory = Callable[[BaseLearnerConfig, int], BaseUpliftEstimator]


#: A candidate needs room for at least this many leaves to be worth trying, so its leaf
#: size may not exceed the training rows divided by this.
MIN_LEAVES_AFFORDABLE = 4

#: Most training rows the grid search fits a candidate on. The winner is then fitted on the
#: whole training split, so this caps selection, not the model that gets reported.
#:
#: It is here because of an arithmetic error rather than a principle. Selection runs nine
#: fits per estimator per seed, eight candidates and the winner, so it is about 89% of a
#: benchmark's cost, and on the week 4 datasets that stopped being affordable: Lenta's
#: 412,217 training rows extrapolated to somewhere between nineteen and thirty-two hours
#: for one dataset, against about twenty minutes for Hillstrom. A protocol nobody can rerun
#: is not a protocol, and one that costs a day per estimator change stops anyone from
#: changing an estimator.
#:
#: The reason it is safe is not the one it was proposed with. The proposal was that selection
#: is a coarse decision whose ranking stabilises early; measuring it refuted that, since a
#: comparable reduction changes nearly every selection. What makes it safe is that the change
#: costs nothing: fitting both the capped and the uncapped choice on the full training split
#: and scoring both on test moves the Qini by -0.00015 on average, and the capped choice wins
#: 9 times in 20. The candidates are near-ties, so the selection was never load-bearing.
#: `docs/estimators.md` carries the tables and the withdrawn prediction.
#:
#: 50,000 is chosen so that it does not bind on any dataset whose full run is affordable:
#: Hillstrom trains on 25,615 rows, ACIC on 2,881, IHDP on 448. That is deliberate, because
#: it makes those three a control, and only Lenta and Criteo are capped at all
#: (PLAN.md change 30).
TUNING_ROWS_CAP = 50_000


def applicable_grid(
    n_train: int, grid: Sequence[BaseLearnerConfig] = GRID
) -> tuple[BaseLearnerConfig, ...]:
    """The candidates a training set of this size can actually fit.

    Args:
        n_train: Rows the candidates would be fitted on.
        grid: Candidate configurations.

    Returns:
        The candidates whose leaf size leaves room for at least
        :data:`MIN_LEAVES_AFFORDABLE` leaves. Never empty: if every candidate is too large
        the smallest one is kept, because selecting from nothing is worse than selecting
        from a bad option, and the degenerate-fit warning will catch the result.
    """
    ceiling = max(1, n_train // MIN_LEAVES_AFFORDABLE)
    affordable = tuple(config for config in grid if config.min_child_samples <= ceiling)
    if affordable:
        return affordable
    smallest = min(grid, key=lambda config: config.min_child_samples)
    return tuple(
        config for config in grid if config.min_child_samples == smallest.min_child_samples
    )


#: Budget the policy-value selection rule scores at. The middle of the three the results
#: table reports (PLAN.md section 1): a rule that selects on the decision has to select on
#: one decision, and a budget covering a fifth of the population is the one a reader of that
#: table is most likely to be imagining.
OPERATING_BUDGET = 0.2


@dataclass(frozen=True, slots=True)
class SelectionRule:
    """What a candidate is scored on, so that the two rules can be run against each other.

    Attributes:
        name: Short identifier, recorded on the selection so a table says what chose it.
        score: Takes the validation split and a candidate's predictions on it, and returns
            a number to be maximised.
    """

    name: str
    score: Callable[[UpliftDataset, FloatArray], float]


def qini_rule(*, tie_seed: int = TIE_SEED) -> SelectionRule:
    """Select on the validation-split Qini coefficient. The default, defended above.

    Args:
        tie_seed: Seed for tie-breaking inside the scoring ranking.

    Returns:
        The rule.
    """

    def score(validation: UpliftDataset, predictions: FloatArray) -> float:
        return qini_coefficient(
            validation.outcome, validation.treatment, predictions, seed=tie_seed
        )

    return SelectionRule(name="qini", score=score)


def policy_value_rule(
    split: Split,
    *,
    budget: float = OPERATING_BUDGET,
    seed: int = 0,
    tie_seed: int = TIE_SEED,
) -> SelectionRule:
    """Select on what the candidate's ranking would actually buy at the operating budget.

    The rule change 9 promised. It scores the decision rather than the curve, using the
    doubly robust gain because the validation splits here are small and it is the
    lower-variance of the two estimators.

    The nuisance models are fitted once, on the training split, and applied to the
    validation split. Every candidate is then scored against the same referee, which is the
    same argument as in :mod:`itx.policy.policy_value`: a candidate must not be allowed to
    supply the models that score it.

    Args:
        split: The partition. Nuisances are fitted on ``split.train`` and applied to
            ``split.validation``; ``split.test`` is not touched.
        budget: Share of the population the score is read at.
        seed: Seed for the nuisance models.
        tie_seed: Seed for tie-breaking inside the scoring ranking.

    Returns:
        The rule, carrying its fitted nuisances.
    """
    nuisances = fit_nuisances(split.train, split.validation, seed=seed)
    key = gain_key("dr", budget)

    def score(validation: UpliftDataset, predictions: FloatArray) -> float:
        return policy_metrics(
            validation.outcome,
            validation.treatment,
            predictions,
            nuisances,
            budgets=(budget,),
            seed=tie_seed,
        )[key]

    return SelectionRule(name=f"policy-value@{budget:.0%}", score=score)


@dataclass(frozen=True, slots=True)
class Selection:
    """Which configuration was chosen for one estimator on one split, and what it scored.

    Attributes:
        config: The winning configuration.
        score: Its validation score.
        scores: Every candidate's validation score, in grid order, for inspection.
        tuned: False when selection was skipped and the default was used.
        rule: Which rule produced the score. Two rules exist and they do not measure the
            same thing, so a bare number would not say what it was.
    """

    config: BaseLearnerConfig
    score: float
    scores: tuple[float, ...]
    tuned: bool = True
    rule: str = "qini"

    def describe(self) -> str:
        """One line naming the winning settings."""
        if not self.tuned:
            return "not tuned, default configuration"
        return (
            f"min_child_samples={self.config.min_child_samples}, "
            f"num_leaves={self.config.num_leaves} "
            f"(validation {self.rule} {self.score:+.5f})"
        )


def select_config(
    factory: EstimatorFactory,
    split: Split,
    *,
    seed: int,
    grid: Sequence[BaseLearnerConfig] = GRID,
    tie_seed: int = TIE_SEED,
    rows_cap: int = TUNING_ROWS_CAP,
    rule: SelectionRule | None = None,
) -> Selection:
    """Choose a configuration by fitting each candidate on train and scoring on validation.

    Args:
        factory: Builds an unfitted estimator from a configuration and a seed.
        split: The partition. Fitting uses ``split.train`` and scoring uses
            ``split.validation``; ``split.test`` is not touched.
        seed: Seed for the candidate fits.
        grid: Candidate configurations.
        tie_seed: Seed for tie-breaking inside the scoring ranking.
        rows_cap: Most training rows any candidate is fitted on. See
            :data:`TUNING_ROWS_CAP`.
        rule: What to score candidates on; :func:`qini_rule` if omitted. The alternative is
            :func:`policy_value_rule`, and ``itx selection`` reports where they disagree.

    Returns:
        The winning configuration and every candidate's score.

    Raises:
        ValueError: If the grid is empty.
    """
    if not grid:
        msg = "cannot select from an empty grid"
        raise ValueError(msg)
    scoring = rule if rule is not None else qini_rule(tie_seed=tie_seed)
    # Computed from the full training split, not the capped one: what a candidate has to
    # be able to fit is the data the winner will finally be fitted on.
    grid = applicable_grid(split.train.n_units, grid)
    train = stratified_subsample(split.train, rows_cap, seed=seed)

    validation = split.validation
    scores: list[float] = []
    for config in grid:
        estimator = factory(config, seed)
        estimator.fit(train)
        predictions = estimator.predict_uplift(validation.features)
        scores.append(scoring.score(validation, predictions))

    best = max(range(len(grid)), key=lambda index: scores[index])
    return Selection(
        config=grid[best],
        score=scores[best],
        scores=tuple(scores),
        rule=scoring.name,
    )


def default_selection() -> Selection:
    """The stand-in for an estimator that has nothing to tune, such as a random ranking."""
    return Selection(config=DEFAULT_CONFIG, score=float("nan"), scores=(), tuned=False)
