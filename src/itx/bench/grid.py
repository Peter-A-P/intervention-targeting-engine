"""The committed hyperparameter grid, and selection on the validation split.

PLAN.md section 4: hyperparameters are tuned on validation with a small fixed grid, the
grid is committed, and it is identical across estimators. Identical matters more than it
sounds. If each estimator got its own grid, or its own number of candidates, the results
table would be reporting how much search each one was given rather than how good each one
is, and the estimator column would quietly become a compute column.

## What is tuned, and why these two knobs

``min_child_samples`` and ``num_leaves``, six combinations. The learning rate and the
number of boosting rounds are held fixed so that every candidate sees the same model
capacity from the boosting side, and because tuning all four turns a fixed grid into a
search whose cost depends on the dataset.

``min_child_samples`` is in the grid because week 1 found it deciding whether an estimator
works at all: at 100 the S-learner on IHDP's 448 training rows never split on the treatment
and returned exactly zero uplift, while at 20 it recovered the effect. One constant cannot
serve a 747-unit dataset and a 64,000-unit one, so the grid picks per dataset and per seed.

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
aligned with the decision and is not available until week 5. When it is, this module gets a
second selection rule and the two are compared, because whether it changes the chosen
configurations is itself worth reporting.

Nothing selected here is ever reported. The validation split exists so that the test split
stays untouched, and a number computed during selection is not a result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from itx.bench.seeds import TIE_SEED
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, BaseLearnerConfig
from itx.metrics.qini import qini_coefficient

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itx.estimators.base import BaseUpliftEstimator
    from itx.types import Split

#: Candidate leaf sizes. 5 is small enough for IHDP's 448 training rows to split at all,
#: 60 is large enough to hold a 25,000-row fit back from memorising noise.
MIN_CHILD_SAMPLES: tuple[int, ...] = (5, 20, 60)

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


@dataclass(frozen=True, slots=True)
class Selection:
    """Which configuration was chosen for one estimator on one split, and what it scored.

    Attributes:
        config: The winning configuration.
        score: Its validation score.
        scores: Every candidate's validation score, in grid order, for inspection.
        tuned: False when selection was skipped and the default was used.
    """

    config: BaseLearnerConfig
    score: float
    scores: tuple[float, ...]
    tuned: bool = True

    def describe(self) -> str:
        """One line naming the winning settings."""
        if not self.tuned:
            return "not tuned, default configuration"
        return (
            f"min_child_samples={self.config.min_child_samples}, "
            f"num_leaves={self.config.num_leaves} "
            f"(validation qini {self.score:+.5f})"
        )


def select_config(
    factory: EstimatorFactory,
    split: Split,
    *,
    seed: int,
    grid: Sequence[BaseLearnerConfig] = GRID,
    tie_seed: int = TIE_SEED,
) -> Selection:
    """Choose a configuration by fitting each candidate on train and scoring on validation.

    Args:
        factory: Builds an unfitted estimator from a configuration and a seed.
        split: The partition. Fitting uses ``split.train`` and scoring uses
            ``split.validation``; ``split.test`` is not touched.
        seed: Seed for the candidate fits.
        grid: Candidate configurations.
        tie_seed: Seed for tie-breaking inside the scoring ranking.

    Returns:
        The winning configuration and every candidate's score.

    Raises:
        ValueError: If the grid is empty.
    """
    if not grid:
        msg = "cannot select from an empty grid"
        raise ValueError(msg)

    validation = split.validation
    scores: list[float] = []
    for config in grid:
        estimator = factory(config, seed)
        estimator.fit(split.train)
        predictions = estimator.predict_uplift(validation.features)
        scores.append(
            qini_coefficient(
                validation.outcome, validation.treatment, predictions, seed=tie_seed
            )
        )

    best = max(range(len(grid)), key=lambda index: scores[index])
    return Selection(config=grid[best], score=scores[best], scores=tuple(scores))


def default_selection() -> Selection:
    """The stand-in for an estimator that has nothing to tune, such as a random ranking."""
    return Selection(config=DEFAULT_CONFIG, score=float("nan"), scores=(), tuned=False)
