"""The base-learner comparison: pairing, bookkeeping and tables, without TabFM's weights.

The fast tests swap TabFM for a stand-in that is LightGBM underneath, with the same settings
and seeds. Two identical engines must produce identical rankings, so every paired difference
must be exactly zero with an interval of exactly zero: anything else means the comparison
is not comparing like with like. The real TabFM fit is in the ``tabfm`` tests at the end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor

from itx.bench import bases
from itx.bench.runner import ESTIMATORS, evaluate, nuisances_for
from itx.data import synthetic
from itx.data.splits import stratified_split
from itx.estimators import lightgbm_base, sklearn_bridge
from itx.estimators.lightgbm_base import DEFAULT_CONFIG
from itx.estimators.tabfm_base import TabFMConfig, is_available
from itx.metrics.bootstrap import Estimate

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from itx.types import FloatArray, Split


class LightGBMInDisguise:
    """Stands where TabFMLearner stands, and is the LightGBM default underneath."""

    def __init__(
        self,
        config: TabFMConfig,
        *,
        binary: bool,
        seed: int = 0,
        categorical: Sequence[int] = (),
    ) -> None:
        self.binary = binary
        self.seed = seed
        self.categorical = list(categorical)
        kwargs = DEFAULT_CONFIG.to_kwargs(seed)
        self.model: Any = LGBMClassifier(**kwargs) if binary else LGBMRegressor(**kwargs)
        self.single: float | None = None

    def fit(
        self, matrix: FloatArray, target: FloatArray, *, sample_weight: FloatArray | None = None
    ) -> None:
        if self.binary and np.unique(target).size < 2:
            self.single = float(target[0])
            return
        self.model.fit(
            matrix, target, sample_weight=sample_weight, categorical_feature=self.categorical
        )

    def predict(self, matrix: FloatArray) -> FloatArray:
        if self.single is not None:
            return np.full(matrix.shape[0], self.single)
        if self.binary:
            return np.asarray(self.model.predict_proba(matrix), dtype=np.float64)[:, 1]
        return np.asarray(self.model.predict(matrix), dtype=np.float64)


@pytest.fixture
def disguised(monkeypatch):
    monkeypatch.setattr(lightgbm_base, "TabFMLearner", LightGBMInDisguise)
    monkeypatch.setattr(sklearn_bridge, "TabFMLearner", LightGBMInDisguise)


@pytest.fixture(scope="module")
def split() -> Split:
    return stratified_split(synthetic.heterogeneous_effect(1_500, seed=4), 11)


@pytest.mark.usefixtures("disguised")
class TestPairing:
    @pytest.mark.parametrize("learner", bases.META_LEARNERS)
    def test_two_identical_engines_differ_by_exactly_nothing(self, learner, split):
        row = bases.compare("toy", learner, split, n_resamples=50, tune=False)
        for metric in ("qini", "auuc", "gain_dr@10%", "pehe"):
            delta = row.metrics[f"delta:{metric}"]
            assert (delta.value, delta.low, delta.high) == (0.0, 0.0, 0.0), metric

    def test_the_lightgbm_side_is_the_main_benchmark_number(self, split):
        # Same split, config, seed and bootstrap seed as the main table's row, so the same
        # number. If these drift apart the comparison is not describing the published table.
        row = bases.compare("toy", "t-learner", split, n_resamples=50, tune=False)
        main = evaluate(
            ESTIMATORS["t-learner"](DEFAULT_CONFIG, split.seed),
            split,
            n_resamples=50,
            nuisances=nuisances_for(split),
        )
        assert row.metrics["lightgbm:qini"].value == main.metrics["qini"].value
        assert row.metrics["lightgbm:pehe"].value == main.metrics["pehe"].value

    def test_a_tabfm_estimator_says_so_in_its_name(self):
        estimator = ESTIMATORS["dr-learner"](TabFMConfig(), 0)
        assert estimator.name == "dr-learner-tabfm"
        assert ESTIMATORS["dr-learner"](DEFAULT_CONFIG, 0).name == "dr-learner"

    def test_an_unknown_learner_is_refused(self, split):
        with pytest.raises(KeyError, match="meta-learner"):
            bases.compare("toy", "dragonnet", split, tune=False)


def _row(
    learner: str, seed: int, *, delta: float, context: int = 16_384, ensemble: int = 4
) -> bases.BaseComparisonRow:
    metrics: dict[str, Estimate] = {}
    for metric in ("qini", "auuc", "gain_dr@10%"):
        metrics[f"lightgbm:{metric}"] = Estimate(0.10, 0.05, 0.15)
        metrics[f"tabfm:{metric}"] = Estimate(0.10 + delta, 0.05 + delta, 0.15 + delta)
        metrics[f"delta:{metric}"] = Estimate(delta, delta - 0.01, delta + 0.01)
    return bases.BaseComparisonRow(
        dataset="toy",
        learner=learner,
        seed=seed,
        context_rows=context,
        ensemble=ensemble,
        n_train=448,
        n_test=150,
        metrics=metrics,
        lightgbm_config="min_child_samples=20, num_leaves=31",
        lightgbm_select_seconds=8.0,
        lightgbm_fit_seconds=1.0,
        lightgbm_predict_seconds=0.5,
        tabfm_fit_seconds=200.0,
        tabfm_predict_seconds=70.0,
    )


class TestBookkeeping:
    def test_rows_survive_a_round_trip_through_the_file(self, tmp_path: Path):
        rows = [_row("s-learner", 11, delta=0.02), _row("t-learner", 11, delta=-0.03)]
        path = tmp_path / "bases" / "toy.json"
        bases.write_json(rows, path)
        assert bases.read_json(path) == rows

    def test_the_hardware_survives_the_file_and_older_files_read_as_the_laptop(
        self, tmp_path: Path
    ):
        import dataclasses
        import json

        gpu = dataclasses.replace(_row("s-learner", 11, delta=0.0), hardware="H100, bfloat16")
        path = tmp_path / "toy.json"
        bases.write_json([gpu], path)
        assert bases.read_json(path)[0].hardware == "H100, bfloat16"
        assert "on H100, bfloat16." in bases.report([gpu])
        record = json.loads(path.read_text(encoding="utf-8"))
        del record[0]["hardware"]
        path.write_text(json.dumps(record), encoding="utf-8")
        older = bases.read_json(path)
        assert older[0].hardware == ""
        assert "on CPU, 8 threads, float32." in bases.report(older)

    def test_a_context_sweep_adds_to_the_file_rather_than_replacing_it(self):
        first = [_row("s-learner", 11, delta=0.01, context=16_384)]
        second = [_row("s-learner", 11, delta=0.02, context=1_024)]
        merged = bases.merge(first, second)
        assert [row.context_rows for row in merged] == [1_024, 16_384]
        rerun = bases.merge(merged, [_row("s-learner", 11, delta=0.05, context=1_024)])
        assert len(rerun) == 2
        assert rerun[0].metrics["delta:qini"].value == 0.05


class TestTables:
    def test_a_difference_that_excludes_zero_is_bold_and_seeds_are_counted(self):
        rows = [_row("s-learner", seed, delta=0.02) for seed in (11, 23, 37)]
        rows += [_row("t-learner", seed, delta=0.0) for seed in (11, 23, 37)]
        table = bases.comparison_table(rows, "qini")
        s_line = next(line for line in table.splitlines() if "`s-learner`" in line)
        t_line = next(line for line in table.splitlines() if "`t-learner`" in line)
        assert "**0.0200 (0.0100, 0.0300)**" in s_line
        assert s_line.endswith("3 of 3 |")
        assert "**" not in t_line
        assert t_line.endswith("0 of 3 |")

    def test_a_metric_no_row_carries_gives_no_table(self):
        assert bases.comparison_table([_row("s-learner", 11, delta=0.0)], "pehe") == ""

    def test_the_ranking_table_orders_each_base_separately(self):
        rows = [_row("s-learner", 11, delta=-0.05), _row("t-learner", 11, delta=0.05)]
        # Equal on LightGBM; on TabFM the T-learner is ahead.
        table = bases.ranking_table(rows)
        first = table.splitlines()[2]
        assert first.split("|")[3].strip().startswith("`t-learner`")

    def test_the_timing_table_reports_the_whole_bill(self):
        table = bases.timing_table([_row("s-learner", 11, delta=0.0)])
        assert "| `s-learner` | 8.0 | 1.5 | 270.0 | 28x |" in table

    def test_a_tabfm_bill_below_lightgbms_is_not_printed_as_zero(self):
        import dataclasses

        fast = dataclasses.replace(
            _row("s-learner", 11, delta=0.0), tabfm_fit_seconds=2.0, tabfm_predict_seconds=1.0
        )
        assert "| 3.0 | 0.32x |" in bases.timing_table([fast])

    def test_the_headline_is_the_smallest_ensemble_at_the_largest_context(self):
        rows = [
            _row("s-learner", 11, delta=0.0),
            _row("s-learner", 11, delta=0.0, context=1_024),
            _row("s-learner", 11, delta=0.0, ensemble=8),
        ]
        top = bases.headline(rows)
        assert [(r.context_rows, r.ensemble) for r in top] == [(16_384, 4)]

    def test_a_sweep_table_appears_only_when_the_setting_was_varied(self):
        one = [_row("s-learner", 11, delta=0.0)]
        assert bases.sweep_table(one, over="context_rows") == ""
        two = [*one, _row("s-learner", 11, delta=0.01, context=1_024)]
        assert "1,024 context rows" in bases.sweep_table(two, over="context_rows")
        assert bases.sweep_table(two, over="ensemble") == ""
        assert "context cap" in bases.report(two)

    def test_a_pehe_sweep_appears_only_where_pehe_was_measured(self):
        two = [_row("s-learner", 11, delta=0.0), _row("s-learner", 11, delta=0.0, context=512)]
        assert bases.sweep_table(two, over="context_rows", metric="pehe") == ""
        assert "PEHE moves" not in bases.report(two)
        for row in two:
            row.metrics["lightgbm:pehe"] = Estimate(1.0, 0.9, 1.1)
            row.metrics["tabfm:pehe"] = Estimate(2.0, 1.9, 2.1)
        sweep = bases.sweep_table(two, over="context_rows", metric="pehe")
        assert "512 context rows" in sweep
        assert "PEHE moves with its context cap" in bases.report(two)


@pytest.mark.tabfm
@pytest.mark.skipif(not is_available(), reason="needs the 'foundation' extra")
def test_tabfm_recovers_a_known_effect_through_the_t_learner():
    # The bar every estimator meets: a known effect on synthetic data. IHDP-sized, so a CPU
    # can afford it; the effect here is 1.0 for every unit.
    data = synthetic.constant_effect(600, effect=1.0, seed=5)
    split = stratified_split(data, 11)
    estimator = ESTIMATORS["t-learner"](TabFMConfig(ensemble=1), 11).fit(split.train)
    predicted = estimator.predict_uplift(split.test.features)
    assert abs(float(predicted.mean()) - 1.0) < 0.25
