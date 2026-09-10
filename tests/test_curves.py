"""Analytic checks on the curves.

Every assertion here has a value worked out by hand or derivable in closed form. A metric
that is only ever compared against its own previous output cannot catch the kind of bug
this project would be embarrassed by, so the reference is arithmetic, not a golden file.
"""

from __future__ import annotations

import numpy as np
import pytest

from itx.metrics.curves import (
    optimal_scores,
    prefix_sums,
    qini_curve,
    qini_gain,
    rank_order,
    uplift_curve,
)
from itx.metrics.qini import auuc, auuc_normalised, qini_coefficient, ranking_metrics

# A six-unit example small enough to compute entirely by hand. Ranked in index order:
#   k=1  treated,  y=1  -> no control yet, gain 1
#   k=2  treated,  y=1  -> gain 2
#   k=3  treated,  y=0  -> gain 2
#   k=4  control,  y=1  -> gain 2 - 1 * 3/1 = -1
#   k=5  control,  y=0  -> gain 2 - 1 * 3/2 = 0.5
#   k=6  control,  y=0  -> gain 2 - 1 * 3/3 = 1
HAND_OUTCOME = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
HAND_TREATMENT = np.array([1, 1, 1, 0, 0, 0])
HAND_SCORES = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
HAND_GAIN = [0.0, 1.0, 2.0, 2.0, -1.0, 0.5, 1.0]


def test_qini_curve_matches_the_hand_computed_gain():
    curve = qini_curve(HAND_OUTCOME, HAND_TREATMENT, HAND_SCORES)
    assert curve.gain == pytest.approx(HAND_GAIN)
    assert curve.fraction == pytest.approx(np.arange(7) / 6)


def test_qini_coefficient_matches_the_hand_computed_area():
    # Trapezoid area of HAND_GAIN with dx = 1/6 is 5/6; per unit that is 5/36. The random
    # line ends at 1/6, so its area is 1/12. The coefficient is the difference, 1/18.
    coefficient = qini_coefficient(HAND_OUTCOME, HAND_TREATMENT, HAND_SCORES)
    assert coefficient == pytest.approx(1 / 18)


def test_qini_endpoint_is_the_rescaled_difference_of_arm_sums():
    curve = qini_curve(HAND_OUTCOME, HAND_TREATMENT, HAND_SCORES)
    treated = HAND_TREATMENT == 1
    expected = HAND_OUTCOME[treated].sum() - HAND_OUTCOME[~treated].sum() * (
        treated.sum() / (~treated).sum()
    )
    assert curve.endpoint == pytest.approx(expected)


def test_qini_endpoint_does_not_depend_on_the_ranking(binary_data):
    # The end of the curve is the whole sample, so every ordering has to arrive there.
    rng = np.random.default_rng(0)
    endpoints = [
        qini_curve(
            binary_data.outcome, binary_data.treatment, rng.random(binary_data.n_units)
        ).endpoint
        for _ in range(5)
    ]
    assert endpoints == pytest.approx([endpoints[0]] * 5)


def test_reversing_a_ranking_flips_the_sign_of_the_qini_coefficient(binary_data):
    scores = binary_data.require_true_effect()
    forward = qini_coefficient(binary_data.outcome, binary_data.treatment, scores)
    backward = qini_coefficient(binary_data.outcome, binary_data.treatment, -scores)
    assert forward > 0
    assert backward == pytest.approx(-forward, rel=0.15)


def test_the_oracle_ordering_normalises_to_exactly_one(binary_data):
    oracle = optimal_scores(binary_data.outcome, binary_data.treatment)
    assert auuc_normalised(binary_data.outcome, binary_data.treatment, oracle) == pytest.approx(
        1.0
    )


def test_no_ranking_beats_the_oracle(binary_data):
    oracle = auuc(
        binary_data.outcome,
        binary_data.treatment,
        optimal_scores(binary_data.outcome, binary_data.treatment),
    )
    rng = np.random.default_rng(1)
    for _ in range(10):
        candidate = auuc(
            binary_data.outcome, binary_data.treatment, rng.random(binary_data.n_units)
        )
        assert candidate <= oracle + 1e-9
    truth = auuc(binary_data.outcome, binary_data.treatment, binary_data.require_true_effect())
    assert truth <= oracle + 1e-9


def test_a_random_ranking_scores_about_zero_qini(binary_data):
    rng = np.random.default_rng(2)
    values = [
        qini_coefficient(
            binary_data.outcome, binary_data.treatment, rng.random(binary_data.n_units)
        )
        for _ in range(40)
    ]
    # The mean over draws is zero by construction; a single draw is not, which is exactly
    # why the reported baseline averages over 200 of them.
    assert float(np.mean(values)) == pytest.approx(0.0, abs=2e-3)


def test_ranking_on_the_true_effect_beats_ranking_on_the_outcome(binary_data):
    by_effect = qini_coefficient(
        binary_data.outcome, binary_data.treatment, binary_data.require_true_effect()
    )
    # The generator makes the control probability depend on covariates 2 and 3 and the
    # effect on covariate 0, so a perfect outcome model ranks on the wrong thing entirely.
    by_outcome = qini_coefficient(
        binary_data.outcome,
        binary_data.treatment,
        binary_data.features["x2"].to_numpy(),
    )
    assert by_effect > by_outcome


def test_tie_breaking_is_reproducible_and_seed_dependent():
    tied = np.zeros(50)
    first = rank_order(tied, seed=1)
    assert np.array_equal(first, rank_order(tied, seed=1))
    assert not np.array_equal(first, rank_order(tied, seed=2))


def test_rank_order_sorts_highest_first():
    scores = np.array([0.1, 0.9, 0.5])
    assert list(rank_order(scores)) == [1, 2, 0]


def test_uplift_curve_is_zero_while_the_prefix_holds_one_arm_only():
    curve = uplift_curve(HAND_OUTCOME, HAND_TREATMENT, HAND_SCORES)
    assert curve.gain[1:4] == pytest.approx([0.0, 0.0, 0.0])


def test_prefix_sums_agree_with_a_direct_loop():
    order = rank_order(HAND_SCORES)
    sums = prefix_sums(HAND_OUTCOME, HAND_TREATMENT, order)
    for k in range(1, 7):
        prefix = order[:k]
        treated = HAND_TREATMENT[prefix] == 1
        assert sums.treated_sum[k - 1] == pytest.approx(HAND_OUTCOME[prefix][treated].sum())
        assert sums.n_treated[k - 1] == pytest.approx(treated.sum())


def test_gain_from_prefix_sums_matches_the_curve():
    order = rank_order(HAND_SCORES)
    gain = qini_gain(prefix_sums(HAND_OUTCOME, HAND_TREATMENT, order))
    assert list(gain) == pytest.approx(HAND_GAIN[1:])


def test_single_pass_metrics_match_the_separate_ones(binary_data):
    rng = np.random.default_rng(4)
    jitter = rng.normal(scale=0.02, size=binary_data.n_units)
    scores = binary_data.require_true_effect() + jitter
    together = ranking_metrics(
        binary_data.outcome,
        binary_data.treatment,
        scores,
        budgets=(0.1, 0.2),
        seed=9,
    )
    assert together["qini"] == pytest.approx(
        qini_coefficient(binary_data.outcome, binary_data.treatment, scores, seed=9)
    )
    assert together["auuc"] == pytest.approx(
        auuc_normalised(binary_data.outcome, binary_data.treatment, scores, seed=9)
    )


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        qini_curve(HAND_OUTCOME, HAND_TREATMENT, HAND_SCORES[:3])


def test_an_empty_sample_is_rejected():
    empty = np.array([])
    with pytest.raises(ValueError, match="empty"):
        qini_curve(empty, empty.astype(np.int64), empty)
