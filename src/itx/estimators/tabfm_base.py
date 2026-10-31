"""TabFM, the second base learner, behind the same interface the LightGBM one uses.

The repository rule was one base learner everywhere, so that a difference between two rows
of the results table is a difference between estimators rather than between engines. That
rule survives here in the form that matters: within one run every meta-learner is built on
the same base. What this module adds is a second, whole column of the comparison, so the
question "does the engine change which estimator wins?" can be asked at all (PLAN.md
change 60, ``docs/estimators.md``).

TabFM is Google's zero-shot tabular foundation model. It is not trained on the data in
front of it: the training rows are handed to a pretrained transformer as context and every
prediction is one forward pass. So there is nothing to tune, and what LightGBM spends on the
validation grid TabFM spends on inference instead. That trade is the second result this
module exists to measure.

Three things here are decisions rather than plumbing, and each is a stated rule:

- **The context cap.** TabFM was pretrained on up to 16,384 context rows and its memory
  grows with the context, so a training set larger than :attr:`TabFMConfig.context_rows`
  is subsampled once per fit, with a seed derived from the fit's own seed. Binary targets
  are subsampled within each class so the context keeps the training base rate, which on
  a rare outcome is otherwise left to chance.
- **Sample weights.** The R-learner's final stage is a weighted regression, and TabFM has
  no weight argument. A weighted fit is therefore turned into a resampled one: context rows
  are drawn with replacement, with probability proportional to weight, which is the
  standard importance-resampling equivalence. It is an approximation and is labelled one.
- **Categorical columns** are handed over as pandas categoricals, which TabFM detects and
  encodes itself, so Hillstrom's ``zip_code`` is a category to TabFM exactly as it is to
  LightGBM.

The weights are under Google's non-commercial, non-production licence
(``tabfm-non-commercial-v1.0``; the package code is Apache-2.0). That is why this is an
optional extra and why nothing on the default install path imports it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from itx.types import FloatArray, IntArray

#: Hugging Face repository the weights come from.
REPOSITORY = "google/tabfm-1.0.0-pytorch"

#: The commit the weights are pinned to. A repository on the Hub can be rewritten under the
#: same name, so the revision is pinned and the files are checked against the digests below,
#: which is the same promise the dataset loaders make (PLAN.md section 3).
REVISION = "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"

#: SHA-256 of each checkpoint at :data:`REVISION`, as the Hub's LFS metadata reports it.
WEIGHT_DIGESTS: dict[str, str] = {
    "classification": "928cb350becdc77cdb7a9e8c36deda88917bfd14a3091894a2dc516db58a2085",
    "regression": "bd5a615b0322a8f04a895038de6df6fbd71430eca750e1d792f31048654674a9",
}

WEIGHTS_FILE = "model.safetensors"

#: Rows TabFM was pretrained with as context; the default cap.
PRETRAINED_CONTEXT_ROWS = 16_384

LICENCE = "tabfm-non-commercial-v1.0 (weights); Apache-2.0 (code)"

INSTALL_HINT = (
    "the TabFM base learner needs the optional 'foundation' extra: install it with "
    "'uv sync --extra foundation'. Its weights are licensed for non-commercial, "
    "non-production use only; see docs/estimators.md"
)


class WeightsMismatchError(RuntimeError):
    """A downloaded TabFM checkpoint does not match its committed digest."""


@dataclass(frozen=True, slots=True)
class TabFMConfig:
    """TabFM settings shared by every estimator built on it.

    Attributes:
        context_rows: Most training rows any one model is shown as context. See the module
            docstring for the rule that draws them.
        ensemble: Ensemble members per prediction, TabFM's own ``n_estimators``. Each
            member is a full forward pass over the context, so this is the cost knob.
            The package default is 32, which on a CPU costs 32 forward passes per
            prediction; see ``docs/estimators.md`` for the measured cost behind this default.
        threads: Torch CPU threads, for the same reason ``BaseLearnerConfig.n_jobs`` is
            explicit: so the machine stays usable and the timings are comparable.
        float32: Run in float32 rather than the bfloat16 the model was designed for. Off by
            default; kept for the quality check TabFM's own loader offers it for.
    """

    context_rows: int = PRETRAINED_CONTEXT_ROWS
    ensemble: int = 4
    threads: int = 8
    float32: bool = False

    def with_(self, **changes: Any) -> TabFMConfig:  # noqa: ANN401
        """Return a copy with some settings replaced.

        Args:
            **changes: Field names and values to override.

        Returns:
            A new config.
        """
        return replace(self, **changes)


DEFAULT_TABFM = TabFMConfig()


def context_rows(
    target: FloatArray,
    *,
    cap: int,
    seed: int,
    binary: bool,
    sample_weight: FloatArray | None = None,
) -> IntArray:
    """Which training rows a TabFM model is shown, by the rule in the module docstring.

    Args:
        target: The training target.
        cap: Most rows to keep.
        seed: Seed for the draw.
        binary: Stratify on the target's two classes.
        sample_weight: Per-row weights. When given, rows are drawn with replacement with
            probability proportional to weight, ``min(n, cap)`` of them, whether or not the
            cap binds, because that draw is what stands in for the weighting.

    Returns:
        Row positions, sorted so the context keeps the training order.

    Raises:
        ValueError: If the cap is not positive or the weights are unusable.
    """
    if cap < 1:
        msg = f"context cap must be positive, got {cap}"
        raise ValueError(msg)
    n = int(target.shape[0])
    rng = np.random.default_rng(seed)
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape[0] != n or not np.isfinite(weight).all() or (weight < 0).any():
            msg = "sample weights must be finite, non-negative and one per row"
            raise ValueError(msg)
        total = float(weight.sum())
        if total <= 0.0:
            msg = "sample weights sum to zero; there is nothing to fit"
            raise ValueError(msg)
        drawn: IntArray = rng.choice(n, size=min(n, cap), replace=True, p=weight / total)
        return np.sort(drawn)
    if n <= cap:
        return np.arange(n)
    if not binary:
        return np.sort(rng.choice(n, size=cap, replace=False))
    positives = np.flatnonzero(target == 1)
    negatives = np.flatnonzero(target != 1)
    keep_positive = min(positives.size, max(1, round(cap * positives.size / n)))
    if positives.size == 0:
        keep_positive = 0
    keep_negative = min(negatives.size, cap - keep_positive)
    chosen = np.concatenate(
        [
            rng.choice(positives, size=keep_positive, replace=False),
            rng.choice(negatives, size=keep_negative, replace=False),
        ]
    )
    return np.sort(chosen)


class TabFMLearner:
    """One TabFM model of a target: a probability for binary targets, a regression otherwise.

    The counterpart of a single LightGBM model, and what both
    :class:`itx.estimators.lightgbm_base.OutcomeLearner` and the scikit-learn bridge hand
    work to when they are given a :class:`TabFMConfig`.
    """

    def __init__(
        self,
        config: TabFMConfig = DEFAULT_TABFM,
        *,
        binary: bool,
        seed: int = 0,
        categorical: Sequence[int] = (),
    ) -> None:
        """Build an unfitted learner.

        Args:
            config: Shared TabFM settings.
            binary: True to model a probability, False to regress.
            seed: Seed for the context draw and for TabFM's own ensemble randomness.
            categorical: Column positions holding integer category codes.
        """
        self.config = config
        self.binary = binary
        self.seed = seed
        self.categorical = tuple(categorical)
        self._model: Any = None
        self._single_class: float | None = None
        self.n_rows = 0
        self.n_context = 0

    def fit(
        self,
        matrix: FloatArray,
        target: FloatArray,
        *,
        sample_weight: FloatArray | None = None,
    ) -> None:
        """Choose the context and hand it to TabFM. No parameters are learned.

        Args:
            matrix: Feature matrix.
            target: Target values.
            sample_weight: Optional per-row weights, honoured by resampling.
        """
        self.n_rows = int(matrix.shape[0])
        rows = context_rows(
            np.asarray(target, dtype=np.float64),
            cap=self.config.context_rows,
            seed=self.seed,
            binary=self.binary,
            sample_weight=sample_weight,
        )
        self.n_context = int(rows.size)
        context_target = np.asarray(target)[rows]
        if self.binary and np.unique(context_target).size < 2:
            self._single_class = float(context_target[0]) if context_target.size else 0.0
            return
        self._single_class = None
        self._model = _estimator(self.config, binary=self.binary, seed=self.seed)
        frame = _frame(matrix[rows], self.categorical)
        self._model.fit(
            frame, context_target.astype(np.int64) if self.binary else context_target
        )

    def predict(self, matrix: FloatArray) -> FloatArray:
        """Predict the target, as a probability for binary targets.

        Args:
            matrix: Feature matrix.

        Returns:
            One prediction per row.
        """
        if self._single_class is not None:
            return np.full(matrix.shape[0], self._single_class, dtype=np.float64)
        if self._model is None:
            msg = "tabfm: fit before predicting"
            raise RuntimeError(msg)
        frame = _frame(matrix, self.categorical)
        _set_threads(self.config.threads)
        if self.binary:
            probabilities = np.asarray(self._model.predict_proba(frame), dtype=np.float64)
            classes = list(np.asarray(self._model.classes_))
            column = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
            positive: FloatArray = probabilities[:, column]
            return positive
        predictions: FloatArray = np.asarray(self._model.predict(frame), dtype=np.float64)
        return predictions


def _estimator(config: TabFMConfig, *, binary: bool, seed: int) -> Any:  # noqa: ANN401
    """A fresh TabFM scikit-learn estimator around the shared pretrained model."""
    try:
        from tabfm import TabFMClassifier, TabFMRegressor
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(INSTALL_HINT) from error
    _set_threads(config.threads)
    model = pretrained("classification" if binary else "regression", float32=config.float32)
    # Ensemble randomness is seeded from the fit's seed and kept within 32 bits, which is
    # what TabFM's numpy and random generators accept.
    common = {"n_estimators": config.ensemble, "random_state": seed % (2**31)}
    if binary:
        return TabFMClassifier(model, **common)
    return TabFMRegressor(model, **common)


def _set_threads(threads: int) -> None:
    """Pin torch's CPU thread count. Global to the process, as Dragonnet's is."""
    import torch

    torch.set_num_threads(threads)


def _frame(matrix: FloatArray, categorical: Sequence[int]) -> Any:  # noqa: ANN401
    """The matrix as a DataFrame TabFM can read column types from.

    pandas only at the library boundary, as everywhere else in this package. Categorical
    codes go over as pandas categoricals, which TabFM's own type detection encodes as
    categories; every other column stays numeric, and missing values are left for TabFM's
    imputer.
    """
    import pandas as pd

    frame = pd.DataFrame(matrix, columns=[f"x{index}" for index in range(matrix.shape[1])])
    for index in categorical:
        name = f"x{index}"
        frame[name] = pd.Categorical(frame[name])
    return frame


@cache
def pretrained(model_type: str, *, float32: bool = False) -> Any:  # noqa: ANN401
    """The pretrained TabFM model for a task, downloaded once, verified, loaded once.

    Args:
        model_type: ``"classification"`` or ``"regression"``.
        float32: Keep the float32 checkpoint rather than casting to bfloat16.

    Returns:
        TabFM's eval-mode model, shared by every estimator in the process.
    """
    try:
        from tabfm import tabfm_v1_0_0_pytorch
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(INSTALL_HINT) from error
    import torch

    directory = weights_dir(model_type)
    return tabfm_v1_0_0_pytorch.load(
        model_type,
        checkpoint_path=str(directory),
        dtype=None if float32 else torch.bfloat16,
    )


@cache
def weights_dir(model_type: str) -> Path:
    """Download one checkpoint at the pinned revision and check it against its digest.

    The Hub's own cache holds the file, so a second process does not download it again;
    the digest is checked once per process all the same, because a cache that has been
    truncated or replaced is exactly what a digest exists to catch.

    Args:
        model_type: ``"classification"`` or ``"regression"``.

    Returns:
        The directory holding ``config.json`` and ``model.safetensors``.

    Raises:
        KeyError: If the model type is not one TabFM ships.
        WeightsMismatchError: If the file does not match the committed digest.
    """
    if model_type not in WEIGHT_DIGESTS:
        msg = f"unknown TabFM model type {model_type!r}; known: {sorted(WEIGHT_DIGESTS)}"
        raise KeyError(msg)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(INSTALL_HINT) from error

    root = Path(
        snapshot_download(REPOSITORY, revision=REVISION, allow_patterns=[f"{model_type}/*"])
    )
    directory = root / model_type
    actual = _sha256(directory / WEIGHTS_FILE)
    expected = WEIGHT_DIGESTS[model_type]
    if actual != expected:
        msg = (
            f"TabFM {model_type} weights at {directory} have digest {actual}, expected "
            f"{expected}. Delete the cached copy and download again; do not use this file."
        )
        raise WeightsMismatchError(msg)
    return directory


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks: the checkpoints are 6.5 GB each."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def is_available() -> bool:
    """True when the optional extra is installed, so tests can skip cleanly without it."""
    try:
        import tabfm  # noqa: F401
    except ImportError:
        return False
    return True
