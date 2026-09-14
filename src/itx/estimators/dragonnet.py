"""Dragonnet: one network that predicts both arms and the propensity from a shared body.

The five meta-learners in this package all take the same shape. Fit some off-the-shelf
regressors, combine their predictions arithmetically, and the causal thinking lives in the
combination rather than in the models. Dragonnet (Shi, Blei and Veitch, 2019) is the other
idea: put the causal structure inside the model and train the whole thing at once.

The architecture is three pieces on one shared representation of the covariates. Two
outcome heads, one per arm, and a propensity head that has to predict treatment from that
same representation. The propensity head is the point. Gradient descent on the outcome
alone will happily build a representation that keeps every scrap of covariate information,
most of which is irrelevant to the effect and all of which is available to overfit on.
Forcing the representation to also predict treatment pushes it toward the part of the
covariate space that assignment actually depended on, which is the part that confounding
lives in, and the paper's claim is that this improves the effect estimate rather than the
prediction. Predicted uplift is then the difference of the two heads.

Targeted regularisation is the second piece, on by default. A single scalar is fitted
alongside everything else and used to perturb the outcome predictions in the direction the
doubly robust correction points, which gives the fitted model the same one-step property an
AIPW estimator has: consistent if either the outcome heads or the propensity head is right.
It costs one parameter and is what makes the architecture more than a multi-task network.

## This is the one estimator not built on LightGBM, and that is a real cost

CLAUDE.md sets a rule for this repository: the base learner is LightGBM everywhere, so that
differences in the results table are differences between estimators rather than differences
between the models underneath them. Dragonnet cannot honour it, because a neural
architecture is the thing being tested and there is no way to express it in boosted trees.

So its column answers a different question from the other five, and a reader has to be told
which. A gap between Dragonnet and the T-learner is a gap between "a neural network with a
propensity head" and "two gradient-boosted trees", and the architecture and the function
class are confounded in it. It is in the table because PLAN.md's definition of done requires
it and because the comparison is worth having anyway, not because it is a clean experiment.
The honest reading is in `docs/estimators.md`.

## It is not tuned, and that is deliberate

The committed grid is over ``min_child_samples`` and ``num_leaves``, which are LightGBM's
knobs and mean nothing here, so Dragonnet is in :data:`itx.bench.runner.UNTUNED` and runs at
the paper's published defaults. Giving it its own grid would be worse than giving it none:
the grid is identical across estimators precisely so that the estimator column does not
become a compute column, and a bespoke search for the one estimator that could not use the
shared grid would put the thumb on the scale in the most visible possible way. The table
says "not tuned" next to it.

## What had to be added around the paper

Three things, all of them consequences of running on real data rather than on the
semi-synthetic sets the paper used.

Categorical columns are one-hot encoded rather than passed through. The loaders hand over
integer codes, which LightGBM reads as categories and a dense layer reads as a magnitude, so
feeding codes to a network would quietly assert that zip code 3 sits between 2 and 4.
Columns are encoded against the categories seen in training, and anything unseen at
prediction time lands in a spare column rather than raising.

Features are standardised, since a network trained by gradient descent on raw columns
spanning six orders of magnitude spends its first epochs undoing the scale. The shift and
scale come from the training rows only.

Missing values are imputed with the training median and flagged with an indicator column,
which is the change that cost a result before it was made. LightGBM takes NaN natively, so
every meta-learner here handles Lenta without anyone thinking about it, and Lenta is the
only one of the five datasets with missing values: 19.5% of its cells, across 150 of its 191
columns. A dense layer does not take NaN. Standardising a column that contains one gives a
NaN mean, the whole design matrix goes NaN on the first forward pass, and the weights never
come back. That happened, and the way it surfaced is the part worth remembering: the
predictions were all NaN, :func:`itx.metrics.curves.rank_order` sorted them to one end, the
ranking became the order the rows arrived in, and the results table reported a Qini and a
policy gain indistinguishable from random targeting instead of reporting nothing. A dead
model that produces plausible numbers is worse than one that raises.

The indicator column is there because the missingness is informative rather than incidental.
A Lenta customer with no ``cheque_count_3m_g20`` did not have that field lost, they never
bought from that group, and imputing a median over that would assert an average purchase
history for people who have none. So the fact of the absence is kept as its own feature and
the imputed value fills the hole underneath it.

And there is a row cap, :data:`DEFAULT_MAX_ROWS`, because this is a CPU-only project by
budget (PLAN.md section 1) and Criteo's 1.4 million rows at a hundred epochs is not a
benchmark anyone reruns. What the cap costs is not yet measured; PLAN.md change 30 is the
template for measuring it and the same measurement is owed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from itx.estimators.base import BaseUpliftEstimator

if TYPE_CHECKING:
    from itx.types import FloatArray, IntArray, UpliftDataset

INSTALL_HINT = (
    "Dragonnet needs PyTorch, which is an optional dependency: install it with "
    "'uv sync --extra neural'"
)

#: Most rows the network is trained on. A cost cap rather than a modelling choice: it does
#: not bind on IHDP, ACIC or Hillstrom, so those three are a control for what it does.
DEFAULT_MAX_ROWS = 200_000

#: Widest a categorical column may be before one-hot encoding it stops being sensible. Above
#: this the column is passed through as a number with a warning in the data card rather than
#: exploding the input layer; none of the five datasets reaches it.
MAX_CATEGORIES = 64


@dataclass(frozen=True, slots=True)
class DragonnetConfig:
    """Architecture and training settings, at the paper's defaults unless noted.

    Attributes:
        representation_units: Width of each of the three shared layers. 200 in the paper.
        head_units: Width of each of the two layers in an outcome head. 100 in the paper.
        alpha: Weight on the treatment-prediction loss. 1.0 in the paper.
        beta: Weight on the targeted regularisation term. 1.0 in the paper.
        targeted: Whether to fit the targeted regularisation scalar at all. Off makes this
            an ordinary multi-task network, which is the ablation the paper reports.
        epochs: Most passes over the training rows.
        batch_size: Rows per gradient step.
        learning_rate: Adam's step size.
        weight_decay: L2 penalty, applied by Adam.
        patience: Epochs without validation improvement before training stops.
        validation_share: Rows held out of training to decide when to stop. Taken from the
            training split, never from test.
        clip: Propensity predictions are held inside ``[clip, 1 - clip]`` before they are
            divided by, the same floor the rest of the package uses.
        max_rows: Row cap; see :data:`DEFAULT_MAX_ROWS`.
        threads: Torch threads. Eight, matching the LightGBM setting, so a benchmark's
            timings are comparable across estimators.
    """

    representation_units: int = 200
    head_units: int = 100
    alpha: float = 1.0
    beta: float = 1.0
    targeted: bool = True
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    validation_share: float = 0.2
    clip: float = 0.01
    max_rows: int = DEFAULT_MAX_ROWS
    threads: int = 8


DEFAULT_DRAGONNET = DragonnetConfig()


class Dragonnet(BaseUpliftEstimator):
    """Shared-representation network with two outcome heads and a propensity head."""

    name = "dragonnet"

    def __init__(self, config: DragonnetConfig = DEFAULT_DRAGONNET, *, seed: int = 0) -> None:
        """Build an unfitted Dragonnet.

        Args:
            config: Architecture and training settings.
            seed: Seed for the weight initialisation, the batch shuffling, the validation
                split and the row cap, so that two runs on one machine agree.
        """
        super().__init__()
        self.config = config
        self.seed = seed
        self._net: Any = None
        self._encoder: _Encoder | None = None
        self._epochs_run: int = 0
        self._best_validation: float = float("nan")

    @property
    def epochs_run(self) -> int:
        """Epochs actually taken before early stopping, for the cost column."""
        return self._epochs_run

    def _fit(self, data: UpliftDataset) -> None:
        """Encode, standardise, split off a validation fraction, and train to early stop."""
        torch = _torch()
        torch.set_num_threads(self.config.threads)
        torch.manual_seed(self.seed)

        rows = _capped_rows(data, self.config.max_rows, self.seed)
        self._encoder = _Encoder.fit(
            data.features[rows], self._categorical, max_categories=MAX_CATEGORIES
        )
        design = self._encoder.transform(data.features[rows])

        treatment = data.treatment[rows].astype(np.float64)
        outcome = data.outcome[rows]
        train_rows, validation_rows = _holdout(
            treatment, self.config.validation_share, self.seed
        )

        self._net = _Network(
            n_features=design.shape[1],
            config=self.config,
            binary_outcome=self._outcome_is_binary,
        )
        optimiser = torch.optim.Adam(
            self._net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        tensors = {
            "x": torch.tensor(design, dtype=torch.float32),
            "t": torch.tensor(treatment, dtype=torch.float32),
            "y": torch.tensor(outcome, dtype=torch.float32),
        }
        generator = torch.Generator().manual_seed(self.seed)
        best = float("inf")
        best_state: dict[str, Any] | None = None
        waited = 0

        for epoch in range(1, self.config.epochs + 1):
            self._net.train()
            order = torch.randperm(train_rows.size, generator=generator)
            for start in range(0, train_rows.size, self.config.batch_size):
                batch = train_rows[order[start : start + self.config.batch_size].numpy()]
                optimiser.zero_grad()
                loss = self._loss(tensors, batch)
                loss.backward()
                optimiser.step()

            self._net.eval()
            with torch.no_grad():
                score = float(self._loss(tensors, validation_rows).item())
            self._epochs_run = epoch
            if score < best - 1e-6:
                best = score
                best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                waited = 0
            else:
                waited += 1
                if waited >= self.config.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        self._best_validation = best

    def _loss(self, tensors: dict[str, Any], rows: IntArray) -> Any:  # noqa: ANN401
        """Outcome loss, treatment loss, and the targeted regularisation term.

        Args:
            tensors: The whole encoded training set, as tensors.
            rows: Row positions this loss is computed over.

        Returns:
            The scalar loss for those rows.
        """
        torch = _torch()
        index = torch.from_numpy(rows)
        x, t, y = tensors["x"][index], tensors["t"][index], tensors["y"][index]
        mu0, mu1, logit, epsilon = self._net(x)

        predicted = t * mu1 + (1.0 - t) * mu0
        outcome_loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(predicted, y, reduction="mean")
            if self._outcome_is_binary
            else torch.nn.functional.mse_loss(predicted, y)
        )
        treatment_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, t, reduction="mean"
        )
        total = outcome_loss + self.config.alpha * treatment_loss

        if self.config.targeted:
            propensity = torch.clamp(
                torch.sigmoid(logit), self.config.clip, 1.0 - self.config.clip
            )
            correction = t / propensity - (1.0 - t) / (1.0 - propensity)
            observed = torch.sigmoid(predicted) if self._outcome_is_binary else predicted
            perturbed = observed + epsilon * correction
            total = total + self.config.beta * torch.nn.functional.mse_loss(perturbed, y)
        return total

    def _predict_uplift(self, features: pl.DataFrame) -> FloatArray:
        """The difference between the two outcome heads."""
        torch = _torch()
        if self._net is None or self._encoder is None:  # pragma: no cover - fit() guards
            msg = "dragonnet: fit before predicting"
            raise RuntimeError(msg)

        design = self._encoder.transform(features)
        with torch.no_grad():
            mu0, mu1, _, _ = self._net(torch.tensor(design, dtype=torch.float32))
            if self._outcome_is_binary:
                mu0, mu1 = torch.sigmoid(mu0), torch.sigmoid(mu1)
        effect: FloatArray = (mu1 - mu0).numpy().astype(np.float64)
        return effect


def _torch() -> Any:  # noqa: ANN401
    """Import torch, with the install hint rather than a bare ModuleNotFoundError.

    Returns:
        The torch module.

    Raises:
        ImportError: If the ``neural`` extra is not installed.
    """
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(INSTALL_HINT) from error
    return torch


def _network_classes() -> tuple[Any, Any]:
    """Build the module classes, deferred so importing this file does not need torch.

    Returns:
        The network class and the torch ``nn`` module.
    """
    torch = _torch()
    return torch, torch.nn


class _Network:
    """The three-headed network, wrapped so torch is imported only when one is built."""

    def __new__(cls, *, n_features: int, config: DragonnetConfig, binary_outcome: bool) -> Any:  # noqa: ANN401
        """Build a torch module rather than an instance of this class.

        Args:
            n_features: Width of the encoded design matrix.
            config: Architecture settings.
            binary_outcome: Whether the heads predict a logit or a level. The heads are
                linear either way; what changes is the loss and the inverse link applied
                on the way out.

        Returns:
            A ``torch.nn.Module`` returning ``(mu0, mu1, treatment_logit, epsilon)``.
        """
        torch, nn = _network_classes()

        class Net(nn.Module):  # type: ignore[misc, name-defined]
            def __init__(self) -> None:
                super().__init__()
                width = config.representation_units
                self.body = nn.Sequential(
                    nn.Linear(n_features, width),
                    nn.ELU(),
                    nn.Linear(width, width),
                    nn.ELU(),
                    nn.Linear(width, width),
                    nn.ELU(),
                )
                self.head0 = _head(nn, width, config.head_units)
                self.head1 = _head(nn, width, config.head_units)
                # One linear layer, as in the paper: the propensity head is kept weak on
                # purpose so that it shapes the representation rather than solving the
                # treatment problem in its own parameters and leaving the body alone.
                self.propensity = nn.Linear(width, 1)
                self.epsilon = nn.Parameter(torch.zeros(1))

            def forward(self, x: Any) -> tuple[Any, Any, Any, Any]:  # noqa: ANN401
                z = self.body(x)
                return (
                    self.head0(z).squeeze(-1),
                    self.head1(z).squeeze(-1),
                    self.propensity(z).squeeze(-1),
                    self.epsilon,
                )

        del binary_outcome  # the heads are linear; the loss decides how to read them
        return Net()


def _head(nn: Any, width: int, units: int) -> Any:  # noqa: ANN401
    """One outcome head: two hidden layers and a linear output.

    Args:
        nn: The torch ``nn`` module.
        width: Width of the shared representation feeding it.
        units: Width of the head's hidden layers.

    Returns:
        The head, as a sequential module.
    """
    return nn.Sequential(
        nn.Linear(width, units),
        nn.ELU(),
        nn.Linear(units, units),
        nn.ELU(),
        nn.Linear(units, 1),
    )


@dataclass(frozen=True, slots=True)
class _Encoder:
    """One-hot for the declared categoricals, standardisation for everything else.

    Attributes:
        numeric: Names of the columns passed through as numbers, in order.
        centre: Mean of each numeric column on the training rows, computed after
            imputation so that a column with missing values still has a finite one.
        scale: Standard deviation of each, floored so a constant column does not divide
            by zero.
        fill: Median of each numeric column on the training rows, ignoring missing values,
            used to fill them. NaN only where a training column was entirely missing, and
            :meth:`transform` then falls back to zero, which is the column mean after
            standardisation and so the least informative value available.
        missing: Indices into ``numeric`` of the columns that had missing values in
            training. Each gets an indicator column, because absence is a fact about the
            customer rather than a hole in the record.
        categorical: Names of the one-hot columns, in order.
        levels: The category codes seen in training, per categorical column. Anything else
            lands in a spare final column rather than raising, because a rare category
            absent from one bootstrap resample is not an error.
    """

    numeric: tuple[str, ...]
    centre: FloatArray
    scale: FloatArray
    fill: FloatArray
    missing: tuple[int, ...]
    categorical: tuple[str, ...]
    levels: tuple[tuple[float, ...], ...]

    @classmethod
    def fit(
        cls, features: pl.DataFrame, categorical: tuple[str, ...], *, max_categories: int
    ) -> _Encoder:
        """Learn the encoding from the training rows.

        Args:
            features: Training feature matrix.
            categorical: Columns the loader declared as integer-coded categories.
            max_categories: Widest a column may be before it is treated as a number.

        Returns:
            The fitted encoder.
        """
        wide = {
            name
            for name in categorical
            if np.unique(features[name].to_numpy()).size > max_categories
        }
        one_hot = tuple(name for name in features.columns if name in set(categorical) - wide)
        numeric = tuple(name for name in features.columns if name not in set(one_hot))

        block = features.select(numeric).to_numpy().astype(np.float64) if numeric else None
        if block is None:
            fill = np.zeros(0)
            missing: tuple[int, ...] = ()
            centre = np.zeros(0)
            spread = np.zeros(0)
        else:
            absent = np.isnan(block)
            missing = tuple(int(i) for i in np.flatnonzero(absent.any(axis=0)))
            # A column that is entirely missing in training has no median; transform falls
            # back to zero for it, which is the column mean once standardised.
            fill = np.where(absent.all(axis=0), 0.0, _column_medians(block))
            filled = np.where(absent, fill, block)
            centre = filled.mean(axis=0)
            spread = filled.std(axis=0)
        return cls(
            numeric=numeric,
            centre=centre,
            scale=np.where(spread > 0.0, spread, 1.0),
            fill=fill,
            missing=missing,
            categorical=one_hot,
            levels=tuple(
                tuple(float(v) for v in np.unique(features[name].to_numpy()))
                for name in one_hot
            ),
        )

    def transform(self, features: pl.DataFrame) -> FloatArray:
        """Encode a feature matrix into the network's input.

        Args:
            features: A matrix with the columns this encoder was fitted on.

        Returns:
            A dense float array: standardised numerics first, then the one-hot blocks.
        """
        parts: list[FloatArray] = []
        if self.numeric:
            block = features.select(self.numeric).to_numpy().astype(np.float64)
            absent = np.isnan(block)
            filled = np.where(absent, self.fill, block)
            parts.append((filled - self.centre) / self.scale)
            if self.missing:
                # One indicator per column that was ever missing in training, so the
                # network can tell an imputed median from an observed one.
                parts.append(absent[:, list(self.missing)].astype(np.float64))
        for name, levels in zip(self.categorical, self.levels, strict=True):
            column = features[name].to_numpy().astype(np.float64)
            # One extra column for anything unseen, so a missing level is recorded rather
            # than silently folded into whichever category happens to be first.
            block = np.zeros((column.size, len(levels) + 1), dtype=np.float64)
            position = {level: index for index, level in enumerate(levels)}
            slots = np.array(
                [position.get(float(value), len(levels)) for value in column], dtype=np.int64
            )
            block[np.arange(column.size), slots] = 1.0
            parts.append(block)
        if not parts:  # pragma: no cover - a dataset with no columns cannot be loaded
            return np.zeros((features.height, 0), dtype=np.float64)
        stacked: FloatArray = np.hstack(parts)
        return stacked


def _column_medians(block: FloatArray) -> FloatArray:
    """Median of each column ignoring missing values, without NumPy's all-NaN warning.

    Args:
        block: Training rows by numeric column.

    Returns:
        One median per column; NaN for a column with no observed value at all, which the
        caller replaces.
    """
    with np.errstate(all="ignore"):
        medians: FloatArray = np.nanmedian(
            np.where(np.isnan(block).all(axis=0), 0.0, block), axis=0
        )
    return medians


def _capped_rows(data: UpliftDataset, max_rows: int, seed: int) -> IntArray:
    """Row positions to train on, subsampled inside each arm when the cap binds.

    Args:
        data: The training dataset.
        max_rows: The cap.
        seed: Seed for the subsample.

    Returns:
        Row positions, sorted, so the subsample does not also reorder the data.
    """
    if data.n_units <= max_rows:
        return np.arange(data.n_units)
    rng = np.random.default_rng(seed)
    share = max_rows / data.n_units
    keep: list[IntArray] = []
    for arm in (0, 1):
        rows = np.flatnonzero(data.treatment == arm)
        take = max(1, round(rows.size * share))
        keep.append(rng.choice(rows, size=take, replace=False))
    return np.sort(np.concatenate(keep))


def _holdout(treatment: FloatArray, share: float, seed: int) -> tuple[IntArray, IntArray]:
    """Split training rows into a fitting set and an early-stopping set, stratified by arm.

    Args:
        treatment: Treatment indicator per row.
        share: Fraction held out.
        seed: Seed for the split.

    Returns:
        Row positions to train on and row positions to stop on.
    """
    rng = np.random.default_rng(seed)
    held: list[IntArray] = []
    kept: list[IntArray] = []
    for arm in (0.0, 1.0):
        rows = np.flatnonzero(treatment == arm)
        shuffled = rng.permutation(rows)
        cut = max(1, round(rows.size * share)) if rows.size > 1 else 0
        held.append(shuffled[:cut])
        kept.append(shuffled[cut:])
    return np.sort(np.concatenate(kept)), np.sort(np.concatenate(held))
