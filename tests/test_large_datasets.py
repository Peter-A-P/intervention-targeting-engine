"""Criteo and Lenta: the subsampling logic, the dropped leaks, and the real loaders.

Split from ``test_data.py`` because these two are the ones with a download measured in
hundreds of megabytes. The stratified subsample is tested on a tiny CSV built here, so the
part with the arithmetic in it runs in a fast pass; the loaders themselves are marked slow.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from itx.data.criteo import (
    EXPECTED_ROWS as CRITEO_ROWS,
)
from itx.data.criteo import (
    FEATURES as CRITEO_FEATURES,
)
from itx.data.criteo import (
    KNOWN_PROPENSITY,
    SUBSAMPLE_FRACTION,
    SUBSAMPLE_SEED,
    _stratified_indices,
    load_criteo,
    subsample_path,
)
from itx.data.criteo import (
    POST_TREATMENT as CRITEO_POST_TREATMENT,
)
from itx.data.download import require_rows
from itx.data.lenta import (
    CATEGORICAL as LENTA_CATEGORICAL,
)
from itx.data.lenta import (
    EXPECTED_ROWS as LENTA_ROWS,
)
from itx.data.lenta import (
    GENDER_FEMALE,
    GENDER_MALE,
    GENDER_UNKNOWN,
    GENDER_UNKNOWN_CODE,
    feature_names,
    load_lenta,
)
from itx.data.lenta import (
    POST_TREATMENT as LENTA_POST_TREATMENT,
)
from itx.metrics.balance import CONVENTIONAL_THRESHOLD, worst_imbalance

#: Rows in the built-here Criteo stand-in. Big enough that the balance check below is
#: about the planted leak rather than about sampling noise in twelve normal columns.
TINY_ROWS = 20_000


class TestTheRowGuard:
    def test_the_right_count_passes(self):
        require_rows("x", 10, 10)

    def test_a_wrong_count_names_both_numbers_and_says_what_to_do(self):
        with pytest.raises(ValueError, match=r"expected 10 rows, read 9.*itx data verify"):
            require_rows("x", 9, 10)


class TestTheStratifiedSubsample:
    """The arithmetic, on arrays small enough to count by hand."""

    def make(self, treated: int, control: int, visits: int):
        treatment = np.array([1] * treated + [0] * control, dtype=np.int64)
        visit = np.zeros(treated + control, dtype=np.int64)
        visit[:visits] = 1
        conversion = np.zeros(treated + control, dtype=np.int64)
        return treatment, visit, conversion

    def test_it_keeps_the_requested_share_of_every_cell(self):
        treatment, visit, conversion = self.make(treated=800, control=200, visits=100)
        keep = _stratified_indices(
            treatment=treatment, visit=visit, conversion=conversion, fraction=0.1, seed=1
        )
        assert keep.size == 100
        # 100 treated-and-visited, 700 treated-not, 200 control-not: a tenth of each.
        assert treatment[keep].sum() == 80
        assert visit[keep].sum() == 10

    def test_the_event_rate_is_preserved_exactly_not_in_expectation(self):
        treatment, visit, conversion = self.make(treated=8_000, control=2_000, visits=1_000)
        keep = _stratified_indices(
            treatment=treatment, visit=visit, conversion=conversion, fraction=0.1, seed=1
        )
        assert visit[keep].mean() == pytest.approx(visit.mean())
        assert treatment[keep].mean() == pytest.approx(treatment.mean())

    def test_the_same_seed_gives_the_same_rows(self):
        treatment, visit, conversion = self.make(treated=800, control=200, visits=100)
        args = {"treatment": treatment, "visit": visit, "conversion": conversion}
        first = _stratified_indices(**args, fraction=0.1, seed=5)
        second = _stratified_indices(**args, fraction=0.1, seed=5)
        assert np.array_equal(first, second)

    def test_a_different_seed_gives_different_rows(self):
        treatment, visit, conversion = self.make(treated=800, control=200, visits=100)
        args = {"treatment": treatment, "visit": visit, "conversion": conversion}
        first = _stratified_indices(**args, fraction=0.1, seed=5)
        second = _stratified_indices(**args, fraction=0.1, seed=6)
        assert not np.array_equal(first, second)

    def test_the_rows_come_back_sorted_so_the_second_pass_reads_in_order(self):
        treatment, visit, conversion = self.make(treated=800, control=200, visits=100)
        keep = _stratified_indices(
            treatment=treatment, visit=visit, conversion=conversion, fraction=0.5, seed=1
        )
        assert np.array_equal(keep, np.sort(keep))
        assert keep.size == len(set(keep.tolist()))

    def test_a_cell_too_small_to_contribute_a_row_is_dropped_not_rounded_up(self):
        # Four converting controls at a 10% sample rounds to zero, not to one.
        treatment = np.array([0] * 4 + [1] * 96, dtype=np.int64)
        visit = np.array([1] * 4 + [0] * 96, dtype=np.int64)
        conversion = np.array([1] * 4 + [0] * 96, dtype=np.int64)
        keep = _stratified_indices(
            treatment=treatment, visit=visit, conversion=conversion, fraction=0.1, seed=1
        )
        assert conversion[keep].sum() == 0

    def test_a_fraction_of_one_keeps_everything(self):
        treatment, visit, conversion = self.make(treated=80, control=20, visits=10)
        keep = _stratified_indices(
            treatment=treatment, visit=visit, conversion=conversion, fraction=1.0, seed=1
        )
        assert np.array_equal(keep, np.arange(100))


class TestCriteoOnATinyFile:
    """The loader end to end, on a file built here rather than downloaded."""

    @pytest.fixture
    def tiny(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(0)
        n = TINY_ROWS
        treatment = rng.binomial(1, 0.85, size=n)
        frame = pl.DataFrame(
            {
                **{name: rng.normal(size=n) for name in CRITEO_FEATURES},
                "treatment": treatment,
                "conversion": rng.binomial(1, 0.05, size=n),
                "visit": rng.binomial(1, 0.2, size=n),
                # The leak: only ever set for treated rows, as in the real file.
                "exposure": treatment * rng.binomial(1, 0.3, size=n),
            }
        )
        buffer = io.BytesIO()
        frame.write_csv(buffer)
        path = tmp_path / "criteo-tiny.csv.gz"
        path.write_bytes(gzip.compress(buffer.getvalue()))
        return path

    def test_it_keeps_only_the_twelve_features(self, tiny: Path):
        data = load_criteo(path=tiny)
        assert data.feature_names == list(CRITEO_FEATURES)

    def test_the_post_treatment_column_never_reaches_the_features(self, tiny: Path):
        data = load_criteo(path=tiny)
        for column in CRITEO_POST_TREATMENT:
            assert column not in data.feature_names

    def test_dropping_exposure_is_what_keeps_the_data_balanced(self, tiny: Path):
        # The check that would have caught it if the loader had not. Compared against the
        # other twelve columns rather than against the 0.1 convention: two thousand rows of
        # noise is enough to push an honest column past a fixed threshold, and the claim
        # here is about the gap between the leak and everything else, which is enormous.
        data = load_criteo(path=tiny, fraction=1.0)
        _, clean = worst_imbalance(data)

        raw = pl.read_csv(tiny)
        leaked = type(data)(
            name="leaked",
            features=raw.select([*CRITEO_FEATURES, "exposure"]).cast(pl.Float64),
            treatment=raw["treatment"].cast(pl.Int64).to_numpy(),
            outcome=raw["visit"].cast(pl.Float64).to_numpy(),
        )
        name, value = worst_imbalance(leaked)
        assert name == "exposure"
        assert value > 10 * clean

    def test_the_design_propensity_is_recorded_rather_than_estimated(self, tiny: Path):
        data = load_criteo(path=tiny)
        assert data.propensity is not None
        assert data.propensity == pytest.approx(np.full(data.n_units, KNOWN_PROPENSITY))

    def test_the_subsample_is_a_tenth_and_says_so_in_its_name(self, tiny: Path):
        data = load_criteo(path=tiny)
        assert data.n_units == pytest.approx(TINY_ROWS // 10, abs=5)
        assert data.name == "criteo-10pct-visit"

    def test_the_full_read_says_so_in_its_name_and_keeps_every_row(self, tiny: Path):
        data = load_criteo(path=tiny, fraction=1.0)
        assert data.n_units == TINY_ROWS
        assert data.name == "criteo-visit"

    def test_conversion_is_available_as_an_outcome(self, tiny: Path):
        data = load_criteo(path=tiny, outcome="conversion")
        assert data.name.endswith("-conversion")
        assert data.outcome_is_binary

    def test_the_same_seed_reads_the_same_rows(self, tiny: Path):
        first = load_criteo(path=tiny)
        second = load_criteo(path=tiny)
        assert first.features.equals(second.features)

    def test_a_different_seed_reads_different_rows(self, tiny: Path):
        first = load_criteo(path=tiny, seed=1)
        second = load_criteo(path=tiny, seed=2)
        assert not first.features.equals(second.features)

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_an_impossible_fraction_is_an_error(self, tiny: Path, fraction: float):
        with pytest.raises(ValueError, match=r"fraction must be in \(0, 1\]"):
            load_criteo(path=tiny, fraction=fraction)

    def test_a_test_path_never_writes_to_the_shared_cache(self, tiny: Path):
        before = subsample_path().exists()
        load_criteo(path=tiny, seed=999)
        assert subsample_path(SUBSAMPLE_FRACTION, 999).exists() is False
        assert subsample_path().exists() == before


class TestLentaColumnSelection:
    """What is and is not a feature, checkable without the download."""

    def test_the_treatment_and_outcome_are_not_features(self):
        columns = ["group", "response_att", "age", "gender"]
        assert feature_names(columns) == ["age", "gender"]

    def test_the_two_post_treatment_columns_are_not_features(self):
        columns = ["age", *LENTA_POST_TREATMENT, "gender"]
        assert feature_names(columns) == ["age", "gender"]

    def test_the_order_of_the_file_is_kept(self):
        columns = ["z", "a", "group", "m", "response_att"]
        assert feature_names(columns) == ["z", "a", "m"]


@pytest.mark.slow
class TestCriteo:
    def test_it_loads_the_committed_subsample(self):
        data = load_criteo()
        assert data.n_units == 1_397_958
        assert data.feature_names == list(CRITEO_FEATURES)
        assert data.categorical == ()
        assert data.outcome_is_binary
        assert data.true_effect is None

    def test_the_subsample_reproduces_the_full_file_rates(self):
        # The point of stratifying. These are the published rates over all 13,979,592 rows.
        data = load_criteo()
        treated = data.treatment == 1
        assert data.treatment.mean() == pytest.approx(0.85, abs=1e-5)
        assert data.outcome[treated].mean() == pytest.approx(0.048543, abs=5e-4)
        assert data.outcome[~treated].mean() == pytest.approx(0.038201, abs=5e-4)

    def test_the_subsample_is_a_tenth_of_the_published_row_count(self):
        assert load_criteo().n_units == round(CRITEO_ROWS * SUBSAMPLE_FRACTION)

    def test_the_features_are_balanced_enough_to_believe_the_design(self):
        # Not perfectly: the worst is about 0.047, real at this sample size but well under
        # the conventional 0.1, and the card says so.
        _, value = worst_imbalance(load_criteo())
        assert value < CONVENTIONAL_THRESHOLD

    def test_the_cache_is_where_the_module_says_it_is(self):
        load_criteo()
        assert subsample_path(SUBSAMPLE_FRACTION, SUBSAMPLE_SEED).is_file()


@pytest.mark.slow
class TestLenta:
    def test_it_loads_every_row(self):
        data = load_lenta()
        assert data.n_units == LENTA_ROWS
        assert data.categorical == LENTA_CATEGORICAL
        assert data.outcome_is_binary
        assert data.true_effect is None

    def test_the_propensity_is_not_claimed_to_be_known(self):
        # Nothing documents the design probability, so it is estimated, not asserted.
        assert load_lenta().propensity is None

    def test_the_published_response_rates(self):
        data = load_lenta()
        treated = data.treatment == 1
        assert data.treatment.mean() == pytest.approx(0.7509, abs=1e-3)
        assert data.outcome[treated].mean() == pytest.approx(0.11013, abs=1e-4)
        assert data.outcome[~treated].mean() == pytest.approx(0.10258, abs=1e-4)

    def test_the_two_leaks_are_gone(self):
        data = load_lenta()
        assert len(data.feature_names) == 191
        for column in LENTA_POST_TREATMENT:
            assert column not in data.feature_names

    def test_dropping_them_is_what_makes_the_arms_balanced(self):
        # The finding that put the balance module in the package. With the two columns in,
        # the worst imbalance is 0.198; with them out it is 0.025.
        _, value = worst_imbalance(load_lenta())
        assert value < CONVENTIONAL_THRESHOLD

    def test_gender_becomes_three_codes_with_the_unknowns_folded_together(self):
        data = load_lenta()
        codes = data.features["gender"].to_numpy()
        assert set(np.unique(codes)) == {0.0, 1.0, 2.0}
        # 8,581 nulls plus 1,090 rows labelled "not determined" in the source.
        assert int((codes == float(GENDER_UNKNOWN_CODE)).sum()) == 9_671

    def test_the_gender_labels_are_the_ones_in_the_file(self):
        raw = pl.scan_csv(
            pytest.importorskip("itx.data.download").fetch("lenta", quiet=True),
            infer_schema_length=None,
        )
        labels = set(raw.select("gender").unique().collect()["gender"].to_list())
        assert {GENDER_FEMALE, GENDER_MALE, GENDER_UNKNOWN} <= labels | {None}

    def test_missing_values_are_left_missing(self):
        # 150 columns with NaN, and no imputation anywhere. LightGBM handles them.
        data = load_lenta()
        with_nan = sum(
            1 for name in data.feature_names if np.isnan(data.features[name].to_numpy()).any()
        )
        assert with_nan == 150
