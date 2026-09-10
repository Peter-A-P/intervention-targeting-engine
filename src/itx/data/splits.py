"""Fixed, stratified train/validation/test splits.

The protocol (PLAN.md section 4) is 60/20/20, stratified, five committed seeds, and every
reported number computed on the test part only. Stratification is on the treatment arm,
and on the outcome as well when the outcome is binary, so a small treated-and-responded
cell cannot land mostly in one part and move a Qini by more than the estimator does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from itx.types import IntArray, Split, UpliftDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_FRACTIONS = (0.6, 0.2, 0.2)


def stratified_split(
    dataset: UpliftDataset,
    seed: int,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> Split:
    """Partition a dataset into train, validation and test.

    Args:
        dataset: The dataset to split.
        seed: Seed for the permutation; the same seed always gives the same partition.
        fractions: Train, validation and test shares. Must sum to 1.

    Returns:
        The three parts, each carrying the parent's ground truth and propensity along.

    Raises:
        ValueError: If the fractions are not three positive numbers summing to 1.
    """
    if len(fractions) != 3 or any(f <= 0 for f in fractions):
        msg = f"fractions must be three positive shares, got {tuple(fractions)}"
        raise ValueError(msg)
    if abs(sum(fractions) - 1.0) > 1e-9:
        msg = f"fractions must sum to 1, got {sum(fractions)}"
        raise ValueError(msg)

    strata = _strata(dataset)
    rng = np.random.default_rng(seed)
    parts: list[list[IntArray]] = [[], [], []]

    for value in np.unique(strata):
        members = np.flatnonzero(strata == value)
        rng.shuffle(members)
        for part, chunk in zip(parts, _cut(members, fractions), strict=True):
            part.append(chunk)

    indices = [np.sort(np.concatenate(part)) for part in parts]
    return Split(
        train=dataset.take(indices[0], name=dataset.name),
        validation=dataset.take(indices[1], name=dataset.name),
        test=dataset.take(indices[2], name=dataset.name),
        seed=seed,
    )


def _strata(dataset: UpliftDataset) -> IntArray:
    """Stratum label per row: treatment arm, crossed with the outcome when it is binary."""
    if dataset.outcome_is_binary:
        return (dataset.treatment * 2 + dataset.outcome.astype(np.int64)).astype(np.int64)
    return dataset.treatment.astype(np.int64)


def _cut(members: IntArray, fractions: Sequence[float]) -> list[IntArray]:
    """Split one shuffled stratum into three contiguous blocks.

    Rounding is by cumulative fraction rather than per-part, so the parts of a stratum of
    size 7 come out 4/1/2 rather than losing a row to three separate roundings.
    """
    size = members.size
    first = round(size * fractions[0])
    second = round(size * (fractions[0] + fractions[1]))
    return [members[:first], members[first:second], members[second:]]
