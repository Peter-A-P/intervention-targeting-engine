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


class TestPredictionBatches:
    def test_a_long_prediction_is_split_and_put_back_in_order(self):
        # A test split too large for the GPU is predicted in batches; the rows must come
        # back once each, in their own order, as if predicted in one pass.
        from itx.estimators.tabfm_base import TabFMLearner

        sizes: list[int] = []

        class Echo:
            def predict(self, frame):
                sizes.append(len(frame))
                return np.asarray(frame.iloc[:, 0], dtype=np.float64)

        learner = TabFMLearner(TabFMConfig(predict_rows=4), binary=False)
        learner._model = Echo()
        matrix = np.arange(10, dtype=np.float64).reshape(-1, 1)
        assert learner.predict(matrix).tolist() == list(range(10))
        assert sizes == [4, 4, 2]


class TestDevice:
    def test_the_cpu_is_the_default(self):
        assert TabFMConfig().device == "cpu"

    def test_the_device_reaches_tabfms_loader(self, monkeypatch, tmp_path):
        # TabFM's loader defaults to the CPU, so a GPU run that forgot to pass the device on
        # would be a CPU run with a GPU's bill. A stand-in package records what it was asked.
        import sys
        import types

        from itx.estimators import tabfm_base

        asked: dict[str, object] = {}
        loader = types.SimpleNamespace(load=lambda model_type, **kwargs: asked.update(kwargs))
        monkeypatch.setitem(
            sys.modules, "tabfm", types.SimpleNamespace(tabfm_v1_0_0_pytorch=loader)
        )
        monkeypatch.setattr(tabfm_base, "weights_dir", lambda model_type: tmp_path)
        tabfm_base.pretrained.cache_clear()
        try:
            tabfm_base.pretrained("regression", float32=True, device="cuda")
        finally:
            tabfm_base.pretrained.cache_clear()
        assert asked["device"] == "cuda"
