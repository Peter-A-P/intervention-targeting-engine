"""The seeds, committed.

Five seeds per dataset (PLAN.md section 4). They are written down here rather than drawn
at run time for one reason: a result that only reproduces on the machine that produced it
is not a result. Anyone can rerun the benchmark and get this table back, and if a number
moves, the seeds are not what moved.

The three roles are kept separate so that changing one does not silently move the others:
the split seed decides who is in the test set, the tie seed decides how equal-scoring units
are ordered inside a ranking, and the bootstrap seed decides the resamples.
"""

from __future__ import annotations

#: Split seeds. Every dataset is split five times and every metric is reported across all
#: five, so a lucky partition cannot carry a row on its own.
SEEDS: tuple[int, ...] = (11, 23, 37, 53, 71)

#: Tie-breaking seed for rankings. Tree ensembles produce large blocks of identical
#: scores; this fixes the order inside them. Shared by the metrics and the policy so that
#: a curve at budget b and the treated set at budget b describe the same people.
TIE_SEED = 20260907

#: Base seed for bootstrap resampling. The seed actually used for a row is this plus the
#: split seed, so two splits do not draw identical resample patterns.
BOOTSTRAP_SEED = 4_242

#: Base seed for the random-targeting reference draws.
RANDOM_BASELINE_SEED = 909


def bootstrap_seed_for(split_seed: int) -> int:
    """Bootstrap seed for a given split.

    Args:
        split_seed: The seed the split was made with.

    Returns:
        A seed unique to that split.
    """
    return BOOTSTRAP_SEED + split_seed
