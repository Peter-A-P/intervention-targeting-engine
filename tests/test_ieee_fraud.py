"""The fraud worked case: the simulation has to say what its docstring says it says.

Everything invented lives in one function, and these tests pin it to the published effect
function so that a change to a constant cannot drift the case away from its own description.
The properties checked are the ones a reader of the README is asked to trust: that review
helps fraud and hurts everything else, that catching is hardest where the signal is highest,
that the label never reaches the feature matrix, and that the per-unit truth is the expected
effect rather than one coin flip.

The loader is exercised on a hand-built frame with the file's columns rather than on the
file, which is 650 MB behind a Kaggle login. Reading the real file is a slow test, and it is
the one that also confirms the digest, since a wrong file would fail there.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest

from itx.data.ieee_fraud import (
    CATCH_AT_ZERO_SIGNAL,
    CATCH_FALL,
    CATEGORICAL,
    COST_COLUMN,
    DECLINE_AT_ZERO_SIGNAL,
    DECLINE_RISE,
    FEATURES,
    KNOWN_PROPENSITY,
    LABEL,
    MARGIN,
    NUMERIC,
    REVIEW_MINUTES_BASE,
    REVIEW_MINUTES_PER_MISSING,
    _encode,
    catch_probability,
    decline_probability,
    load_ieee_fraud,
    simulate,
)


def tiny_transactions(n: int = 400, seed: int = 0) -> pl.DataFrame:
    """A frame with the file's used columns, a 10% fraud rate, and some missing values."""
    rng = np.random.default_rng(seed)
    columns: dict[str, object] = {LABEL: (rng.random(n) < 0.10).astype(int)}
    for name in NUMERIC:
        values = rng.exponential(50.0, n) if name == "TransactionAmt" else rng.normal(size=n)
        if name in {"dist1", "D5", "addr2"}:
            values = np.where(rng.random(n) < 0.3, np.nan, values)
        columns[name] = values
    for name in CATEGORICAL:
        # A plain list of str | None, so polars builds a String column with nulls. A numpy
        # object array here would make the CSV writer refuse the frame.
        draws = rng.integers(0, 4, n)
        columns[name] = [("a", "b", "c", None)[int(d)] for d in draws]
    return pl.DataFrame(columns)


class TestThePublishedEffectFunction:
    def test_catching_falls_with_the_signal_and_stays_a_probability(self):
        signal = np.linspace(0.0, 1.0, 11)
        caught = catch_probability(signal)
        assert caught[0] == pytest.approx(CATCH_AT_ZERO_SIGNAL)
        assert caught[-1] == pytest.approx(CATCH_AT_ZERO_SIGNAL - CATCH_FALL)
        assert np.all(np.diff(caught) < 0)
        assert np.all((caught > 0.0) & (caught < 1.0))

    def test_wrongly_declining_rises_with_the_signal_and_stays_a_probability(self):
        signal = np.linspace(0.0, 1.0, 11)
        declined = decline_probability(signal)
        assert declined[0] == pytest.approx(DECLINE_AT_ZERO_SIGNAL)
        assert declined[-1] == pytest.approx(DECLINE_AT_ZERO_SIGNAL + DECLINE_RISE)
        assert np.all(np.diff(declined) > 0)
        assert np.all((declined > 0.0) & (declined < 1.0))

    def test_the_riskiest_looking_fraud_is_the_hardest_to_catch(self):
        # The assumption the case rests on, stated in the module docstring. If somebody
        # flips the sign of CATCH_FALL the case becomes one where risk ranking is optimal,
        # and this test is what tells them the README no longer describes it.
        assert CATCH_FALL > 0


class TestWhatTheSimulationWrites:
    def make(self, n: int = 5_000, seed: int = 1):
        rng = np.random.default_rng(seed)
        fraud = (rng.random(n) < 0.2).astype(np.int64)
        amount = rng.exponential(100.0, n) + 1.0
        signal = rng.random(n)
        return fraud, amount, signal, simulate(fraud, amount, signal, seed=seed)

    def test_review_helps_fraud_and_hurts_everything_else(self):
        fraud, _, _, (_, _, effect) = self.make()
        assert np.all(effect[fraud == 1] > 0)
        assert np.all(effect[fraud == 0] < 0)

    def test_the_truth_is_the_expected_effect_not_a_coin_flip(self):
        # A per-unit effect defined by a single draw would be unlearnable, and PEHE against it
        # would be measuring the coin. The truth has to be the expectation.
        fraud, amount, signal, (_, _, effect) = self.make()
        expected = np.where(
            fraud == 1,
            amount * catch_probability(signal),
            -MARGIN * amount * decline_probability(signal),
        )
        assert effect == pytest.approx(expected)

    def test_the_realised_outcome_is_one_of_the_four_cells(self):
        fraud, amount, _, (reviewed, outcome, _) = self.make()
        is_fraud = fraud == 1
        legit_margin = MARGIN * amount
        # Fraud: charged back or caught. Legitimate: margin kept or wrongly declined.
        fraud_ok = np.isclose(outcome, -amount) | np.isclose(outcome, 0.0)
        legit_ok = np.isclose(outcome, legit_margin) | np.isclose(outcome, 0.0)
        assert np.all(np.where(is_fraud, fraud_ok, legit_ok))
        # And nothing changes for a unit that was not reviewed.
        untouched = reviewed == 0
        assert np.allclose(
            outcome[untouched], np.where(is_fraud, -amount, legit_margin)[untouched]
        )

    def test_the_realised_effect_averages_to_the_written_truth(self):
        # The coin flips have to be consistent with the expectation they were drawn from,
        # or the estimators are being asked to recover a different number from the one
        # PEHE scores them against.
        n = 200_000
        fraud = np.ones(n, dtype=np.int64)
        amount = np.full(n, 10.0)
        signal = np.full(n, 0.5)
        reviewed, outcome, effect = simulate(fraud, amount, signal, seed=3)
        realised = outcome[reviewed == 1].mean() - outcome[reviewed == 0].mean()
        assert realised == pytest.approx(effect.mean(), abs=0.05)

    def test_assignment_is_a_fair_coin(self):
        _, _, _, (reviewed, _, _) = self.make(n=50_000)
        assert reviewed.mean() == pytest.approx(KNOWN_PROPENSITY, abs=0.01)

    def test_it_is_deterministic_in_the_seed(self):
        fraud, amount, signal, first = self.make(seed=4)
        second = simulate(fraud, amount, signal, seed=4)
        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a, b)


class TestTheLoader:
    def test_it_declares_itself_semi_synthetic_first(self):
        # CLAUDE.md: the fraud worked case is semi-synthetic and declared so in its first
        # sentence. The module docstring is where a reader of the code meets it.
        import itx.data.ieee_fraud as module

        first_paragraph = (module.__doc__ or "").strip().split("\n\n")[0].lower()
        assert "semi-synthetic" in first_paragraph or "invented" in first_paragraph

    def test_the_label_never_reaches_the_features(self, tmp_path):
        path = tmp_path / "t.csv"
        tiny_transactions().write_csv(path)
        data = load_ieee_fraud(path=path)
        assert LABEL not in data.feature_names
        assert set(FEATURES) <= set(data.feature_names)

    def test_it_declares_that_risk_is_a_low_outcome(self, tmp_path):
        # Dollars retained: the transaction a fraud queue takes first is the one predicted
        # to lose the most, which is the lowest outcome. Without this flag the benchmark's
        # outcome-ranking row is a queue of the most profitable legitimate sales.
        path = tmp_path / "t.csv"
        tiny_transactions().write_csv(path)
        assert load_ieee_fraud(path=path).risk_is_low_outcome

    def test_it_carries_the_truth_and_the_known_propensity(self, tmp_path):
        path = tmp_path / "t.csv"
        tiny_transactions().write_csv(path)
        data = load_ieee_fraud(path=path)
        assert data.true_effect is not None
        assert data.propensity is not None
        assert np.all(data.propensity == KNOWN_PROPENSITY)

    def test_review_cost_rises_with_missing_fields_and_is_bounded(self, tmp_path):
        path = tmp_path / "t.csv"
        tiny_transactions().write_csv(path)
        data = load_ieee_fraud(path=path)
        minutes = data.features[COST_COLUMN].to_numpy()
        assert minutes.min() >= REVIEW_MINUTES_BASE
        assert minutes.max() <= REVIEW_MINUTES_BASE + REVIEW_MINUTES_PER_MISSING
        # Three of the numeric columns are 30% missing, so some rows cost more than the base.
        assert (minutes > REVIEW_MINUTES_BASE).any()

    def test_categoricals_are_integer_codes_with_a_reserved_missing_code(self, tmp_path):
        path = tmp_path / "t.csv"
        tiny_transactions().write_csv(path)
        data = load_ieee_fraud(path=path)
        for name in CATEGORICAL:
            codes = data.features[name].to_numpy()
            assert np.all(codes == np.round(codes))
            assert codes.min() >= 0.0
        assert data.categorical == CATEGORICAL

    def test_a_subsample_preserves_the_fraud_rate(self, tmp_path):
        frame = tiny_transactions(n=4_000)
        path = tmp_path / "t.csv"
        frame.write_csv(path)
        full_rate = frame[LABEL].mean()
        data = load_ieee_fraud(fraction=0.25, path=path)
        # Stratified on the label, so the rate survives to within one unit's rounding.
        recovered = (data.require_true_effect() > 0).mean()  # positive only for fraud
        assert recovered == pytest.approx(full_rate, abs=0.005)
        assert data.n_units == pytest.approx(1_000, abs=2)

    def test_a_bad_fraction_is_refused(self, tmp_path):
        path = tmp_path / "t.csv"
        tiny_transactions().write_csv(path)
        with pytest.raises(ValueError, match="fraction"):
            load_ieee_fraud(fraction=0.0, path=path)


class TestTheCategoryCodes:
    """The codes must be a function of the level alone.

    They were not. Until 2026-09-16 the encoder took the physical codes of a polars
    Categorical, which are assigned by first appearance, and the appearance order varies
    with how a read is split across threads. The same file therefore encoded differently in
    different processes, and the five Dragonnet rows of the fraud benchmark could not be
    reproduced by anyone, including by this machine. PLAN.md change 58.
    """

    def a_frame(self, values: Sequence[str | None]) -> pl.DataFrame:
        return pl.DataFrame({"c": values})

    def test_the_code_is_the_position_in_the_sorted_levels(self):
        frame = self.a_frame(["gmail", "aol", "zoho", "aol"])
        codes = frame.select(_encode("c")).to_series().to_list()
        # aol, gmail, zoho sorted, so 1, 2, 3; nothing here is missing.
        assert codes == [2.0, 1.0, 3.0, 1.0]

    def test_missing_takes_the_reserved_zero(self):
        frame = self.a_frame(["gmail", None, "aol"])
        assert frame.select(_encode("c")).to_series().to_list() == [2.0, 0.0, 1.0]

    def test_row_order_does_not_change_the_mapping(self):
        """The property the old encoding failed: shuffling the rows must not relabel."""
        levels = ["gmail", "aol", "zoho", "yahoo", None]
        first = self.a_frame(levels)
        second = self.a_frame(list(reversed(levels)))
        mapping = dict(
            zip(levels, first.select(_encode("c")).to_series().to_list(), strict=True)
        )
        reversed_mapping = dict(
            zip(
                list(reversed(levels)),
                second.select(_encode("c")).to_series().to_list(),
                strict=True,
            )
        )
        assert mapping == reversed_mapping

    def test_a_subsample_encodes_a_level_the_same_way_the_whole_column_does(self):
        whole = self.a_frame(["aol", "gmail", "zoho"])
        part = self.a_frame(["aol", "gmail"])
        whole_codes = dict(
            zip(["aol", "gmail", "zoho"], whole.select(_encode("c")).to_series(), strict=True)
        )
        part_codes = dict(
            zip(["aol", "gmail"], part.select(_encode("c")).to_series(), strict=True)
        )
        # Dropping a level above them cannot move the levels below it.
        assert part_codes["aol"] == whole_codes["aol"]
        assert part_codes["gmail"] == whole_codes["gmail"]

    def test_the_loader_gives_the_same_codes_every_time_it_is_called(self, tmp_path):
        path = tmp_path / "t.csv"
        tiny_transactions().write_csv(path)
        first = load_ieee_fraud(path=path)
        second = load_ieee_fraud(path=path)
        for name in CATEGORICAL:
            assert np.array_equal(
                first.features[name].to_numpy(), second.features[name].to_numpy()
            ), name


def the_real_file_is_here() -> bool:
    """Whether the manually-placed IEEE-CIS file is present and matches its digest.

    Every other dataset here is fetched from a URL, so ``--run-slow`` can assume it will
    arrive. This one cannot: Kaggle serves it to an authenticated account that has accepted
    the competition rules, and the licence forbids a mirror, so a machine that has not been
    given the file can never run the test below. Skipping is the honest outcome there, and
    failing is not: CI failed this way on the first green run after PLAN.md change 57, on a
    runner where the file is absent by design rather than by mistake.
    """
    from itx.data.download import sha256_of
    from itx.data.registry import data_dir, source

    spec = source("ieee-fraud-train")
    path = data_dir() / spec.filename
    return path.is_file() and sha256_of(path) == spec.expected_sha256


@pytest.mark.slow
@pytest.mark.skipif(
    not the_real_file_is_here(),
    reason="IEEE-CIS train_transaction.csv is not here; see docs/data/ieee-fraud.md",
)
class TestOnTheRealFile:
    def test_the_real_file_loads_and_matches_its_card(self):
        # Goes through fetch, so this is also the test that the committed digest is right.
        data = load_ieee_fraud(fraction=0.02)
        assert data.name == "ieee-fraud"
        assert len(data.feature_names) == 44
        fraud_rate = (data.require_true_effect() > 0).mean()
        assert fraud_rate == pytest.approx(0.035, abs=0.003)
