"""The TabFM base learner: the context rule, without the weights, and a real fit with them.

The context rule is pure numpy and is tested everywhere. Anything that loads the 6.5 GB
checkpoints is marked ``tabfm`` and skipped unless the optional extra is installed and
``--run-tabfm`` is passed, because CI installs neither the extra nor the weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from itx.estimators.tabfm_base import PRETRAINED_CONTEXT_ROWS, TabFMConfig, context_rows


class TestContextRows:
    def test_a_training_set_under_the_cap_is_used_whole(self):
        target = np.zeros(500)
        assert np.array_equal(
            context_rows(target, cap=1_000, seed=1, binary=False), np.arange(500)
        )

    def test_a_binary_target_keeps_its_base_rate_when_subsampled(self):
        rng = np.random.default_rng(0)
        target = (rng.random(50_000) < 0.02).astype(np.float64)
        rows = context_rows(target, cap=4_096, seed=3, binary=True)
        assert rows.size == 4_096
        assert np.unique(rows).size == rows.size
        assert target[rows].mean() == pytest.approx(target.mean(), abs=1 / 4_096)

    def test_the_draw_is_fixed_by_the_seed(self):
        target = np.arange(20_000, dtype=np.float64)
        first = context_rows(target, cap=1_000, seed=7, binary=False)
        assert np.array_equal(first, context_rows(target, cap=1_000, seed=7, binary=False))
        assert not np.array_equal(first, context_rows(target, cap=1_000, seed=8, binary=False))

    def test_weights_are_honoured_by_resampling_in_proportion(self):
        target = np.zeros(1_000)
        weight = np.where(np.arange(1_000) < 500, 3.0, 1.0)
        rows = context_rows(target, cap=100_000, seed=0, binary=False, sample_weight=weight)
        # Three times the weight, three times the draws: 0.75 of the context.
        assert rows.size == 1_000
        heavy = (rows < 500).mean()
        assert heavy == pytest.approx(0.75, abs=0.05)

    def test_a_zero_weight_row_is_never_shown(self):
        target = np.zeros(200)
        weight = np.ones(200)
        weight[:100] = 0.0
        rows = context_rows(target, cap=200, seed=0, binary=False, sample_weight=weight)
        assert rows.min() >= 100

    @pytest.mark.parametrize("weight", [np.full(10, -1.0), np.zeros(10), np.full(10, np.nan)])
    def test_unusable_weights_are_refused(self, weight):
        with pytest.raises(ValueError, match="weights"):
            context_rows(np.zeros(10), cap=10, seed=0, binary=False, sample_weight=weight)

    def test_the_default_cap_is_the_pretraining_context(self):
        assert TabFMConfig().context_rows == PRETRAINED_CONTEXT_ROWS == 16_384
