"""Dragonnet, on effects it has to recover and on the encoding it has to get right.

The network is checked the same way every other estimator here is: against synthetic data
whose true effect is written down. A constant effect has to come back as that constant, and
a heterogeneous one has to come back in the right order, or the column in the results table
is decoration.

The encoder gets its own tests because its failures are silent. A categorical column fed
through as an integer code asserts that zip code 3 sits between 2 and 4, which a network
will happily fit and nobody will see in a metric. A level present at prediction time but
absent from training would, with a naive one-hot, either raise or be folded into the first
category; here it has to land in the spare column. And standardisation computed on anything
other than the training rows is a leak.

The fast tests use a small network and few epochs on purpose: they are checking that the
plumbing is right, not that the published architecture is. The two that use the real
configuration are marked slow.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch", reason="Dragonnet needs the 'neural' extra")

from itx.data import synthetic  # noqa: E402
from itx.estimators.base import DegenerateFitWarning, NotFittedError  # noqa: E402
from itx.estimators.dragonnet import (  # noqa: E402
    DEFAULT_DRAGONNET,
    Dragonnet,
    DragonnetConfig,
    _capped_rows,
    _Encoder,
    _holdout,
)
from itx.types import UpliftDataset  # noqa: E402

#: Small enough to train in seconds, large enough to fit a smooth effect. The plumbing is
#: what these tests are about; the published architecture is exercised by the slow ones.
QUICK = DragonnetConfig(
    representation_units=32, head_units=16, epochs=40, batch_size=256, patience=8
)


class TestTheEncoder:
    def test_numeric_columns_are_standardised_on_the_training_rows(self):
        frame = pl.DataFrame({"a": [1.0, 3.0, 5.0, 7.0], "b": [10.0, 10.0, 10.0, 10.0]})
        encoder = _Encoder.fit(frame, (), max_categories=64)
        encoded = encoder.transform(frame)
        assert encoded[:, 0].mean() == pytest.approx(0.0, abs=1e-12)
        assert encoded[:, 0].std() == pytest.approx(1.0)
        # A constant column has no spread to divide by and must not become NaN or inf.
        assert np.isfinite(encoded[:, 1]).all()

    def test_the_shift_and_scale_come_from_training_only(self):
        # Standardising a prediction batch by its own statistics is a leak, and it is the
        # kind that improves the numbers rather than breaking them.
        train = pl.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0]})
        encoder = _Encoder.fit(train, (), max_categories=64)
        shifted = pl.DataFrame({"a": [100.0, 101.0, 102.0, 103.0, 104.0]})
        encoded = encoder.transform(shifted)
        assert encoded.mean() > 30.0  # it was not re-centred on its own mean

    def test_a_categorical_column_becomes_one_hot(self):
        frame = pl.DataFrame({"zip": [0.0, 1.0, 2.0, 1.0], "x": [1.0, 2.0, 3.0, 4.0]})
        encoder = _Encoder.fit(frame, ("zip",), max_categories=64)
        encoded = encoder.transform(frame)
        # One numeric column, then three levels plus one spare.
        assert encoded.shape == (4, 1 + 4)
        assert encoded[:, 1:].sum(axis=1).tolist() == [1.0, 1.0, 1.0, 1.0]
        assert encoded[0, 1] == 1.0
        assert encoded[2, 3] == 1.0

    def test_a_level_absent_from_training_lands_in_the_spare_column(self):
        # A rare category missing from one bootstrap resample is not an error, and folding
        # it into whichever level happens to be first would be a silent mislabelling.
        train = pl.DataFrame({"zip": [0.0, 1.0], "x": [1.0, 2.0]})
        encoder = _Encoder.fit(train, ("zip",), max_categories=64)
        encoded = encoder.transform(pl.DataFrame({"zip": [9.0], "x": [3.0]}))
        assert encoded[0, -1] == 1.0
        assert encoded[0, 1:-1].sum() == 0.0

    def test_a_very_wide_categorical_is_left_as_a_number(self):
        frame = pl.DataFrame({"id": [float(i) for i in range(50)]})
        encoder = _Encoder.fit(frame, ("id",), max_categories=10)
        assert encoder.categorical == ()
        assert encoder.numeric == ("id",)


class TestMissingValues:
    """Lenta is 19.5% missing and it produced a committed table full of nothing.

    LightGBM takes NaN natively, so every meta-learner handles Lenta without anyone
    thinking about it, and Lenta is the only one of the five datasets with missing values.
    A dense layer does not. Standardising a column holding a NaN gives a NaN mean, the whole
    matrix goes NaN on the first forward pass, and the weights never return. What made it
    expensive was that it did not look like a failure: the predictions were all NaN,
    `rank_order` sorted them to one end, the ranking became the order the rows arrived in,
    and the results table reported a Qini indistinguishable from random targeting.
    """

    def with_missing(self, share: float, n: int = 4_000, p: int = 12, seed: int = 0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        treatment = (rng.random(n) < 0.5).astype(np.int64)
        outcome = x[:, 0] + treatment * (1.0 + x[:, 1]) + rng.normal(0, 0.2, n)
        if share:
            x[rng.random((n, p)) < share] = np.nan
        frame = pl.DataFrame({f"f{i}": x[:, i] for i in range(p)})
        return UpliftDataset(
            name="missing", features=frame, treatment=treatment, outcome=outcome
        )

    def test_a_column_with_missing_values_does_not_poison_the_matrix(self):
        encoder = _Encoder.fit(self.with_missing(0.2).features, (), max_categories=64)
        encoded = encoder.transform(self.with_missing(0.2).features)
        assert np.isfinite(encoded).all()

    def test_a_missingness_indicator_is_added_per_affected_column(self):
        data = self.with_missing(0.2, p=12)
        encoder = _Encoder.fit(data.features, (), max_categories=64)
        encoded = encoder.transform(data.features)
        # Twelve standardised columns plus one indicator for each column that had a gap.
        assert encoded.shape[1] == 12 + len(encoder.missing)
        assert len(encoder.missing) > 0

    def test_the_indicator_marks_the_rows_that_were_missing(self):
        # Absence in this data is a fact about the unit, not a hole in the record, so it
        # has to reach the network rather than be smoothed over by the median.
        frame = pl.DataFrame({"a": [1.0, 2.0, float("nan"), 4.0]})
        encoder = _Encoder.fit(frame, (), max_categories=64)
        encoded = encoder.transform(frame)
        assert encoded[:, -1].tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_the_hole_is_filled_with_the_training_median(self):
        frame = pl.DataFrame({"a": [1.0, 2.0, 3.0, float("nan")]})
        encoder = _Encoder.fit(frame, (), max_categories=64)
        assert encoder.fill[0] == pytest.approx(2.0)

    def test_a_column_missing_everywhere_does_not_raise(self):
        frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [float("nan")] * 3})
        encoder = _Encoder.fit(frame, (), max_categories=64)
        assert np.isfinite(encoder.transform(frame)).all()

    def test_no_indicators_when_nothing_is_missing(self):
        encoder = _Encoder.fit(self.with_missing(0.0).features, (), max_categories=64)
        assert encoder.missing == ()

    def test_it_fits_and_predicts_finite_uplift_on_missing_data(self):
        data = self.with_missing(0.195)
        predicted = Dragonnet(QUICK, seed=0).fit(data).predict_uplift(data.features)
        assert np.isfinite(predicted).all()
        assert np.unique(predicted).size > 100  # a real ranking, not a constant

    def test_a_non_finite_prediction_is_warned_about_loudly(self):
        # The guard that would have caught this at the source. np.allclose(nan, 0) is
        # False, so the old all-zero check stayed silent while the run produced numbers.
        class Broken(Dragonnet):
            def _predict_uplift(self, features):
                return np.full(features.height, np.nan)

        with pytest.warns(DegenerateFitWarning, match="not finite"):
            Broken(QUICK, seed=0).fit(self.with_missing(0.0, n=600))


class TestTheRowCapAndHoldout:
    def test_the_cap_does_not_bind_on_a_small_dataset(self):
        data = synthetic.constant_effect(500, effect=1.0, seed=0)
        assert _capped_rows(data, 200_000, 0).tolist() == list(range(500))

    def test_the_cap_keeps_both_arms_in_proportion(self):
        data = synthetic.constant_effect(4_000, effect=1.0, seed=0)
        rows = _capped_rows(data, 1_000, seed=0)
        assert 900 <= rows.size <= 1_100
        share_before = data.treatment.mean()
        share_after = data.treatment[rows].mean()
        assert share_after == pytest.approx(share_before, abs=0.03)

    def test_the_cap_returns_sorted_positions(self):
        data = synthetic.constant_effect(4_000, effect=1.0, seed=0)
        rows = _capped_rows(data, 1_000, seed=0)
        assert rows.tolist() == sorted(rows.tolist())

    def test_the_holdout_is_disjoint_and_covers_everything(self):
        treatment = np.array([1.0] * 300 + [0.0] * 700)
        kept, held = _holdout(treatment, 0.2, seed=0)
        assert set(kept.tolist()) & set(held.tolist()) == set()
        assert sorted(kept.tolist() + held.tolist()) == list(range(1_000))

    def test_the_holdout_is_stratified_by_arm(self):
        treatment = np.array([1.0] * 200 + [0.0] * 800)
        _, held = _holdout(treatment, 0.25, seed=0)
        assert int((treatment[held] == 1.0).sum()) == 50
        assert int((treatment[held] == 0.0).sum()) == 200


class TestWhatItLearns:
    def test_it_recovers_a_constant_effect(self):
        data = synthetic.constant_effect(6_000, effect=1.0, seed=1)
        fitted = Dragonnet(QUICK, seed=0).fit(data)
        predicted = fitted.predict_uplift(data.features)
        assert predicted.mean() == pytest.approx(1.0, abs=0.25)

    def test_it_ranks_a_heterogeneous_effect(self):
        data = synthetic.heterogeneous_effect(8_000, seed=2)
        fitted = Dragonnet(QUICK, seed=0).fit(data)
        predicted = fitted.predict_uplift(data.features)
        truth = data.require_true_effect()
        assert float(np.corrcoef(predicted, truth)[0, 1]) > 0.5

    def test_a_binary_outcome_stays_inside_its_range(self):
        # The heads are linear and the loss reads them as logits, so the inverse link has
        # to be applied on the way out. Without it the predicted uplift is a difference of
        # logits, which is a number on the wrong scale and cannot be compared with the
        # other estimators' in the same table.
        data = synthetic.binary_outcome(6_000, seed=3)
        predicted = Dragonnet(QUICK, seed=0).fit(data).predict_uplift(data.features)
        assert predicted.min() >= -1.0
        assert predicted.max() <= 1.0

    def test_the_same_seed_gives_the_same_answer(self):
        data = synthetic.constant_effect(2_000, effect=1.0, seed=4)
        first = Dragonnet(QUICK, seed=7).fit(data).predict_uplift(data.features)
        second = Dragonnet(QUICK, seed=7).fit(data).predict_uplift(data.features)
        assert first.tolist() == second.tolist()

    def test_early_stopping_reports_what_it_did(self):
        data = synthetic.constant_effect(2_000, effect=1.0, seed=5)
        fitted = Dragonnet(QUICK, seed=0).fit(data)
        assert 0 < fitted.epochs_run <= QUICK.epochs

    def test_it_runs_without_the_targeted_regularisation(self):
        # The paper's own ablation. It has to remain a working estimator, because the
        # difference between the two is the only way to say what the extra term bought.
        data = synthetic.constant_effect(4_000, effect=1.0, seed=6)
        plain = DragonnetConfig(
            representation_units=32,
            head_units=16,
            epochs=40,
            batch_size=256,
            patience=8,
            targeted=False,
        )
        predicted = Dragonnet(plain, seed=0).fit(data).predict_uplift(data.features)
        assert predicted.mean() == pytest.approx(1.0, abs=0.35)

    def test_it_handles_a_categorical_column_end_to_end(self):
        base = synthetic.constant_effect(3_000, effect=1.0, seed=8)
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 4, base.n_units).astype(np.float64)
        data = UpliftDataset(
            name="with-category",
            features=base.features.with_columns(pl.Series("region", codes)),
            treatment=base.treatment,
            outcome=base.outcome,
            categorical=("region",),
        )
        predicted = Dragonnet(QUICK, seed=0).fit(data).predict_uplift(data.features)
        assert np.isfinite(predicted).all()
        assert predicted.mean() == pytest.approx(1.0, abs=0.3)

    def test_predicting_before_fitting_is_an_error(self):
        data = synthetic.constant_effect(200, effect=1.0, seed=9)
        with pytest.raises(NotFittedError):
            Dragonnet(QUICK, seed=0).predict_uplift(data.features)

    def test_a_different_feature_matrix_is_refused(self):
        data = synthetic.constant_effect(1_000, effect=1.0, seed=10)
        fitted = Dragonnet(QUICK, seed=0).fit(data)
        with pytest.raises(ValueError, match="feature columns do not match"):
            fitted.predict_uplift(data.features.drop(data.feature_names[0]))


@pytest.mark.slow
class TestAtThePublishedSettings:
    def test_it_recovers_a_constant_effect_at_full_size(self):
        data = synthetic.constant_effect(8_000, effect=1.0, seed=11)
        predicted = Dragonnet(DEFAULT_DRAGONNET, seed=0).fit(data).predict_uplift(data.features)
        assert predicted.mean() == pytest.approx(1.0, abs=0.15)

    def test_it_beats_a_coin_flip_on_a_heterogeneous_effect(self):
        data = synthetic.heterogeneous_effect(8_000, seed=12)
        predicted = Dragonnet(DEFAULT_DRAGONNET, seed=0).fit(data).predict_uplift(data.features)
        assert float(np.corrcoef(predicted, data.require_true_effect())[0, 1]) > 0.7
