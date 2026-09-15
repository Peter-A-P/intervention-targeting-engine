"""How much hidden bias in who got treated it would take to overturn the targeting result.

The E-value in :mod:`itx.sensitivity.evalue` asks about a confounder's association with the
treatment and the outcome. Rosenbaum's bound asks a different question about the same worry,
and the two are worth reporting together because they fail in different places.

The question here is about assignment odds. Take two units that look identical on every
covariate we measured. In a randomised experiment their odds of being treated are equal. If
something unmeasured is at work, one of them may have been up to ``Gamma`` times more likely
to be treated than the other. Gamma of 1 is the randomised case. The sensitivity analysis
asks: how large does Gamma have to get before the evidence that the intervention helped the
targeted group stops being significant?

A Gamma of 1.05 means a hidden factor shifting the assignment odds by five percent could
account for everything, and no observational claim survives that. A Gamma of 3 means it
would take a hidden factor tripling the odds of treatment, which is larger than most of the
covariates we *did* measure, and at that point the finding deserves some weight. The number
is not a p-value and not a probability: it is the price of the objection.

## What it is computed on

Matched pairs inside the targeted group, because the decision under test is the targeting
decision (PLAN.md section 4). Pairs are formed by nearest-neighbour matching, one control per
treated unit, without replacement and inside a caliper, so that a pair really is two units
who looked alike. Units that cannot find a partner inside the caliper are dropped and
counted, and :attr:`MatchedPairs.matched_share` reports what fraction of the treated group
survived, because a Gamma computed from a third of the group is a statement about that third.

What they are matched on depends on the design, and getting this wrong is silent.
:func:`matching_key` picks. On an observational dataset the key is the estimated propensity
score, which is what Rosenbaum's method is written around: pairs alike in their probability
of treatment are the pairs a hidden bias would have had to act within. On a randomised
dataset the design propensity is a constant, every unit is equidistant from every other, and
propensity matching silently degenerates into arbitrary pairing. Hillstrom's is exactly 0.5
for all 42,693 rows, and the first version of this module paired its units on it and called
them matched.

The pairing is not wrong there, because under randomisation any pairing gives a valid test,
but the word "matched" claims something it is not doing. So where the propensity has no
spread the key falls back to the prognostic score, the predicted outcome under control,
which is Hansen's (2008) device. The key that was used is reported on the result, because a
reader comparing a Gamma across two datasets needs to know whether the same thing was done
to both.

The reason for the fallback is honesty rather than power, and the distinction is not a
quibble, because power is what it was changed for and the measurement said no. On Hillstrom
at a 20% budget, over twelve matching seeds:

    arbitrary pairing    Gamma 1.347 (sd 0.043, 1.29 to 1.41), 833 pairs, 217 discordant
    prognostic score     Gamma 1.300 (sd 0.038, 1.23 to 1.36), 788 pairs, 207 discordant

Matching on the prognostic score is no more stable across seeds, and it costs about
forty-five pairs to the caliper and a little Gamma. It is kept because pairing units on a
constant and calling the result a matched pair is a claim the code should not make, and
because the direction of the cost is conservative. It is not kept because it works better.

That table says something else worth reading. A Gamma moves by about 0.06 either way purely
on which arbitrary order the greedy matcher served units in, so the number belongs to one
decimal place. A write-up quoting 1.32 against 1.41 as though the difference meant something
would be reporting the matching seed.

## Binary outcomes come out right without a second implementation

The statistic is Wilcoxon's signed rank over the within-pair differences, with zeros dropped
and ties given average ranks. That is the textbook choice for a continuous outcome, and for
a binary one it quietly becomes the right thing too: every surviving difference is then plus
or minus one, every rank is the same average rank, and the whole expression reduces
algebraically to McNemar's test on the discordant pairs, which is what Rosenbaum's chapter
prescribes for binary matched pairs. ``tests/test_rosenbaum.py`` asserts that reduction
against a direct binomial calculation rather than trusting the algebra, since the four
datasets this is reported on all have binary outcomes and the reduction is the whole reason
one implementation is enough.

## The bound itself

Under a hidden bias of at most Gamma, the probability that the higher-ranked member of a
pair is the treated one lies between ``1 / (1 + Gamma)`` and ``Gamma / (1 + Gamma)``. Taking
the worst case gives an upper bound on the p-value:

    E = p * sum(ranks),  V = p * (1 - p) * sum(ranks^2),  with p = Gamma / (1 + Gamma)

and the deviate ``(W - E) / sqrt(V)`` read against the normal. At Gamma of 1 this is the
ordinary one-sided signed-rank test. :func:`breaking_point` then walks Gamma upward and
reports where the bound crosses the significance level, which is the one number the write-up
quotes.

Two honest limits. The normal approximation is used rather than the exact distribution,
which is standard and is accurate at the pair counts here but would not be at a dozen pairs,
so :data:`MIN_PAIRS` refuses below a floor.

And the test is one-sided in a fixed direction: it asks whether the treated member of a pair
tends to have the *higher* outcome. That is not "whichever way the data points", and the
difference matters. Choosing the direction after seeing the data would inflate the error
rate, which is the thing a sensitivity analysis exists to be careful about, so the direction
is fixed and a result pointing the other way correctly returns a Gamma of 1.0 rather than a
large number for a finding in the opposite direction. Every outcome in this project is one
where more is better, a visit or a conversion or a purchase, so the fixed direction is the
right one here. A caller working with an outcome where less is better, a default or a
readmission, negates the outcome before calling and negates nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from itx.types import BoolArray, FloatArray, IntArray

#: Fewest matched pairs the normal approximation is trusted at. Below this the bound is
#: reported as undefined rather than as a number the approximation cannot support.
MIN_PAIRS = 20

#: Caliper for the propensity match, in standard deviations of the propensity score among
#: eligible units. 0.2 SD is the conventional width, but the convention (Rosenbaum and Rubin
#: 1985) is on the logit of the propensity and this is applied to the propensity itself,
#: which is looser in the tails. It is a constant rather than a parameter so that a Gamma
#: cannot be improved by widening it.
DEFAULT_CALIPER_SDS = 0.2

#: Largest Gamma the search will report. A result that survives this is reported as "above
#: this" rather than as a precise number, because the upper tail of the search is where the
#: normal approximation and the matching assumptions matter most and a reader who needs to
#: distinguish 14 from 16 is over-reading the method.
MAX_GAMMA = 15.0

#: Step of the Gamma search. Fine enough that the reported value is not visibly grid-bound,
#: coarse enough that the search is instant.
GAMMA_STEP = 0.01

#: Significance level the breaking point is read at, matching the 95% intervals everywhere
#: else in this package.
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True, slots=True)
class MatchedPairs:
    """Treated units paired with lookalike controls, and what the outcome did in each pair.

    Attributes:
        treated: Row positions of the treated member of each pair.
        control: Row positions of the control member, aligned with ``treated``.
        difference: ``outcome[treated] - outcome[control]``, one per pair.
        distance: Absolute gap in the matching key within each pair, for match quality.
        n_eligible_treated: Treated units that were offered to the matcher, matched or not.
        matched_on: Which key the pairs were formed on; see :func:`matching_key`.
    """

    treated: IntArray
    control: IntArray
    difference: FloatArray
    distance: FloatArray
    n_eligible_treated: int
    matched_on: str = "propensity"

    @property
    def n_pairs(self) -> int:
        """How many pairs were formed."""
        return int(self.treated.size)

    @property
    def matched_share(self) -> float:
        """Share of the eligible treated units that found a partner inside the caliper."""
        if self.n_eligible_treated == 0:
            return math.nan
        return self.n_pairs / self.n_eligible_treated

    @property
    def n_discordant(self) -> int:
        """Pairs whose two members did not have the same outcome."""
        return int(np.count_nonzero(self.difference))


@dataclass(frozen=True, slots=True)
class RosenbaumBound:
    """The hidden bias the targeting result would survive, and what it was read from.

    Attributes:
        gamma: Largest hidden bias at which the one-sided bound is still significant. 1.0
            means the result does not survive any hidden bias at all, and a value at
            :data:`MAX_GAMMA` means the search stopped rather than that the truth is there.
        p_value: The bound's p-value at Gamma of 1, which is the ordinary signed-rank test.
        n_pairs: Matched pairs the statistic was computed on.
        n_discordant: Of those, how many had different outcomes and so carried information.
        matched_share: Share of eligible treated units that found a partner.
        budget: Share of the population the targeted group is.
        alpha: Significance level the breaking point was read at.
        censored: True when the search hit :data:`MAX_GAMMA` without the bound crossing.
        matched_on: Which key the pairs were formed on; see :func:`matching_key`.
    """

    gamma: float
    p_value: float
    n_pairs: int
    n_discordant: int
    matched_share: float
    budget: float
    alpha: float
    censored: bool = False
    matched_on: str = "propensity"

    @property
    def survives_any_bias(self) -> bool:
        """True when the result is still significant under some hidden bias."""
        return math.isfinite(self.gamma) and self.gamma > 1.0

    def summary(self) -> str:
        """One paragraph a reader can act on."""
        if not math.isfinite(self.gamma):
            if self.n_discordant == self.n_pairs:
                # Every pair informative and still too few: a continuous outcome, where
                # exact ties do not happen, or a group that was simply too small.
                return (
                    f"At a {self.budget:.0%} budget only {self.n_pairs:,} pairs could be "
                    f"matched, below the {MIN_PAIRS} the normal approximation needs, so "
                    f"there is no Rosenbaum bound to report."
                )
            return (
                f"At a {self.budget:.0%} budget {self.n_pairs:,} pairs matched but only "
                f"{self.n_discordant:,} of them were discordant, below the {MIN_PAIRS} the "
                f"normal approximation needs. Concordant pairs carry no information about "
                f"the effect, so there is no Rosenbaum bound to report."
            )
        quality = (
            f"{self.n_pairs:,} pairs matched on the {self.matched_on} "
            f"({self.matched_share:.0%} of the eligible treated units), "
            f"{self.n_discordant:,} of them discordant"
        )
        if not self.survives_any_bias:
            return (
                f"At a {self.budget:.0%} budget the one-sided signed-rank test on {quality} "
                f"gives p = {self.p_value:.4f}, which does not clear {self.alpha:.0%} even "
                f"with no hidden bias at all, so there is nothing for a sensitivity analysis "
                f"to protect."
            )
        if self.censored:
            return (
                f"At a {self.budget:.0%} budget, on {quality}, the result stays significant "
                f"at hidden bias above Gamma = {MAX_GAMMA:.0f}, where the search stops. "
                f"Read that as very insensitive rather than as a measured number."
            )
        return (
            f"At a {self.budget:.0%} budget, on {quality}, the result stays significant at "
            f"the {self.alpha:.0%} level up to a hidden bias of Gamma = {self.gamma:.2f}. "
            f"An unmeasured factor would have to make one of two covariate-identical units "
            f"{self.gamma:.2f} times more likely to be treated than the other before this "
            f"conclusion could be explained by assignment rather than by the intervention."
        )


def matching_key(
    propensity: FloatArray, prognostic: FloatArray | None = None
) -> tuple[FloatArray, str]:
    """Decide what pairs should be formed on, given the design.

    Args:
        propensity: Probability of treatment per unit. Used when it varies, which is the
            observational case Rosenbaum's method is written around.
        prognostic: Predicted outcome under control per unit, used when the propensity is a
            constant and matching on it would be arbitrary pairing under another name.

    Returns:
        The key to match on and a word naming it, for the result to carry.
    """
    if float(np.std(propensity)) > 0.0:
        return propensity, "propensity"
    if prognostic is not None and float(np.std(prognostic)) > 0.0:
        return prognostic, "prognostic score"
    return propensity, "nothing"


def match_pairs(
    score: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    *,
    eligible: BoolArray | None = None,
    caliper_sds: float = DEFAULT_CALIPER_SDS,
    seed: int = 0,
    matched_on: str = "propensity",
) -> MatchedPairs:
    """Pair each treated unit with its nearest unused control on the matching key.

    Greedy nearest neighbour without replacement, inside a caliper. Greedy rather than
    optimal because the optimal assignment on this many rows is a transportation problem and
    the difference between the two is small next to the thing being measured; the order
    treated units are served in is a seeded permutation rather than the data's own order, so
    that a dataset sorted by anything cannot hand the best controls to one end of it.

    Args:
        score: The key units are matched on, one per unit. See :func:`matching_key`.
        treatment: Binary treatment indicator per unit.
        outcome: Observed outcome per unit.
        eligible: Units the matcher may use; all of them if omitted. This is how the
            targeted group is passed in.
        caliper_sds: Widest acceptable gap in the key, in standard deviations of the key
            among eligible units.
        seed: Seed for the order treated units are served in.
        matched_on: A word naming the key, carried onto the result.

    Returns:
        The pairs, with the within-pair outcome differences.

    Raises:
        ValueError: If the arrays do not all have the same length.
    """
    sizes = {score.size, treatment.size, outcome.size}
    if len(sizes) != 1:
        msg = f"score, treatment and outcome must align, got sizes {sorted(sizes)}"
        raise ValueError(msg)

    inside = np.ones(score.size, dtype=bool) if eligible is None else eligible
    treated = np.flatnonzero(inside & (treatment == 1))
    control = np.flatnonzero(inside & (treatment == 0))
    if treated.size == 0 or control.size == 0:
        return _no_pairs(int(treated.size), matched_on)

    spread = float(np.std(score[inside]))
    caliper = caliper_sds * spread if spread > 0.0 else math.inf

    by_score = control[np.argsort(score[control], kind="stable")]
    control_scores = score[by_score]
    free = _FreeList(by_score.size)

    rng = np.random.default_rng(seed)
    served = rng.permutation(treated.size)
    matched_treated: list[int] = []
    matched_control: list[int] = []
    gaps: list[float] = []

    for position in served:
        unit = int(treated[position])
        slot = free.nearest(control_scores, float(score[unit]))
        if slot is None:
            continue
        gap = abs(float(control_scores[slot]) - float(score[unit]))
        if gap > caliper:
            continue
        free.take(slot)
        matched_treated.append(unit)
        matched_control.append(int(by_score[slot]))
        gaps.append(gap)

    if not matched_treated:
        return _no_pairs(int(treated.size), matched_on)

    # Pairs come out in the order they were served, which is the seeded permutation. Sorting
    # by the treated row position makes the result depend on the data rather than on the
    # permutation, so two runs with different seeds are comparable side by side.
    treated_index = np.array(matched_treated, dtype=np.int64)
    control_index = np.array(matched_control, dtype=np.int64)
    order = np.argsort(treated_index, kind="stable")
    treated_index = treated_index[order]
    control_index = control_index[order]
    return MatchedPairs(
        treated=treated_index,
        control=control_index,
        difference=outcome[treated_index] - outcome[control_index],
        distance=np.array(gaps, dtype=np.float64)[order],
        n_eligible_treated=int(treated.size),
        matched_on=matched_on,
    )


def signed_rank(difference: FloatArray) -> tuple[float, FloatArray]:
    """Wilcoxon's signed-rank statistic and the ranks it was built from.

    Args:
        difference: Within-pair outcome differences. Zeros are dropped, which is the
            standard handling and the reason a binary outcome reduces to McNemar's test:
            only the discordant pairs survive.

    Returns:
        The sum of ranks over pairs with a positive difference, and the ranks of the
        absolute differences with ties given their average rank.
    """
    nonzero = difference[difference != 0.0]
    if nonzero.size == 0:
        return 0.0, np.zeros(0, dtype=np.float64)
    ranks: FloatArray = stats.rankdata(np.abs(nonzero)).astype(np.float64)
    statistic = float(ranks[nonzero > 0.0].sum())
    return statistic, ranks


def bound_p_value(difference: FloatArray, gamma: float) -> float:
    """Worst-case one-sided p-value under a hidden bias of at most ``gamma``.

    Args:
        difference: Within-pair outcome differences, treated minus control. The test is
            one-sided for a positive difference; see the module docstring.
        gamma: Hidden bias, at least 1. Gamma of 1 gives the ordinary signed-rank test.

    Returns:
        The upper bound on the p-value, or NaN when there are too few informative pairs for
        the normal approximation.

    Raises:
        ValueError: If gamma is below 1, which is not a hidden bias but its reciprocal.
    """
    if gamma < 1.0:
        msg = f"gamma must be at least 1, got {gamma}"
        raise ValueError(msg)
    statistic, ranks = signed_rank(difference)
    if ranks.size < MIN_PAIRS:
        return math.nan

    probability = gamma / (1.0 + gamma)
    expected = probability * float(ranks.sum())
    variance = probability * (1.0 - probability) * float((ranks**2).sum())
    if variance <= 0.0:
        return math.nan
    deviate = (statistic - expected) / math.sqrt(variance)
    return float(stats.norm.sf(deviate))


def breaking_point(
    difference: FloatArray,
    *,
    alpha: float = DEFAULT_ALPHA,
    max_gamma: float = MAX_GAMMA,
    step: float = GAMMA_STEP,
) -> tuple[float, bool]:
    """Largest hidden bias at which the bound is still significant.

    The bound's p-value rises monotonically in gamma, so the crossing is found by bisection
    on a grid rather than by walking it, and the answer is rounded down to the grid so that
    the reported gamma is one the bound was actually evaluated at.

    Args:
        difference: Within-pair outcome differences.
        alpha: Significance level.
        max_gamma: Where the search stops.
        step: Grid resolution.

    Returns:
        The gamma and whether the search was censored at ``max_gamma``. A gamma of 1.0 means
        the result was not significant even with no hidden bias; NaN means there were too
        few informative pairs to say.
    """
    if not math.isfinite(bound_p_value(difference, 1.0)):
        return math.nan, False
    if bound_p_value(difference, 1.0) > alpha:
        return 1.0, False
    if bound_p_value(difference, max_gamma) <= alpha:
        return max_gamma, True

    steps = round((max_gamma - 1.0) / step)
    low, high = 0, steps  # low is known significant, high is known not
    while high - low > 1:
        middle = (low + high) // 2
        if bound_p_value(difference, 1.0 + middle * step) <= alpha:
            low = middle
        else:
            high = middle
    return 1.0 + low * step, False


def targeting_rosenbaum(
    propensity: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    targeted: BoolArray,
    *,
    budget: float,
    prognostic: FloatArray | None = None,
    alpha: float = DEFAULT_ALPHA,
    caliper_sds: float = DEFAULT_CALIPER_SDS,
    seed: int = 0,
) -> RosenbaumBound:
    """The hidden bias one ranking's targeting decision would survive.

    Args:
        propensity: Probability of treatment per unit.
        treatment: Binary treatment indicator per unit.
        outcome: Observed outcome per unit.
        targeted: True for the units the policy would treat.
        budget: Share of the population the targeted group is, for reporting.
        prognostic: Predicted outcome under control per unit. Used as the matching key on
            a randomised design, where the propensity is constant and matching on it would
            be arbitrary pairing; see :func:`matching_key`.
        alpha: Significance level.
        caliper_sds: Match caliper, in standard deviations of the matching key.
        seed: Seed for the matching order.

    Returns:
        The bound, with the pair counts and the matching key it was read from.
    """
    key, named = matching_key(propensity, prognostic)
    pairs = match_pairs(
        key,
        treatment,
        outcome,
        eligible=targeted,
        caliper_sds=caliper_sds,
        seed=seed,
        matched_on=named,
    )
    gamma, censored = breaking_point(pairs.difference, alpha=alpha)
    return RosenbaumBound(
        gamma=gamma,
        p_value=bound_p_value(pairs.difference, 1.0),
        n_pairs=pairs.n_pairs,
        n_discordant=pairs.n_discordant,
        matched_share=pairs.matched_share,
        budget=budget,
        alpha=alpha,
        censored=censored,
        matched_on=named,
    )


def _no_pairs(n_eligible_treated: int, matched_on: str) -> MatchedPairs:
    """The empty result, for a group with nothing to match."""
    empty_int: IntArray = np.zeros(0, dtype=np.int64)
    empty_float: FloatArray = np.zeros(0, dtype=np.float64)
    return MatchedPairs(
        treated=empty_int,
        control=empty_int.copy(),
        difference=empty_float,
        distance=empty_float.copy(),
        n_eligible_treated=n_eligible_treated,
        matched_on=matched_on,
    )


class _FreeList:
    """Which controls are still unmatched, with nearest-free lookup in near-constant time.

    Greedy matching without replacement needs "the closest control that nobody has taken
    yet", and rescanning a used mask makes the matcher quadratic, which on Criteo's scale is
    the difference between seconds and hours. Two union-find chains over the
    propensity-sorted controls keep, for every slot, the nearest free slot below it and the
    nearest free slot above it, so a taken control is skipped rather than searched past.
    Paths are compressed on the way out, and iteratively rather than recursively, because
    1.4 million controls is far past the recursion limit.
    """

    def __init__(self, size: int) -> None:
        """Start with every slot free.

        Args:
            size: Number of controls.
        """
        self._size = size
        self._below = np.arange(-1, size - 1, dtype=np.int64)  # below[i] starts at i - 1
        self._above = np.arange(1, size + 1, dtype=np.int64)  # above[i] starts at i + 1
        self._taken = np.zeros(size, dtype=bool)

    def nearest(self, scores: FloatArray, target: float) -> int | None:
        """The free slot whose score is closest to ``target``.

        Args:
            scores: Control scores in ascending order.
            target: The treated unit's score.

        Returns:
            The slot, or None when every control is taken.
        """
        position = int(np.searchsorted(scores, target))
        left = self._free_below(position - 1)
        right = self._free_above(position)
        if left < 0:
            return right if right < self._size else None
        if right >= self._size:
            return left
        if abs(float(scores[left]) - target) <= abs(float(scores[right]) - target):
            return left
        return right

    def take(self, slot: int) -> None:
        """Mark one slot used.

        Args:
            slot: Index into the sorted controls.
        """
        self._taken[slot] = True

    def _free_below(self, start: int) -> int:
        """Nearest free slot at or below ``start``, or -1 if there is none."""
        path: list[int] = []
        current = start
        while current >= 0 and self._taken[current]:
            path.append(current)
            current = int(self._below[current])
        for slot in path:
            self._below[slot] = current
        return current

    def _free_above(self, start: int) -> int:
        """Nearest free slot at or above ``start``, or ``size`` if there is none."""
        path: list[int] = []
        current = start
        while current < self._size and self._taken[current]:
            path.append(current)
            current = int(self._above[current])
        for slot in path:
            self._above[slot] = current
        return current
