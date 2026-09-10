"""The dataset container, the splits, the registry, and the loaders.

The loader tests are marked slow because they need a real download. Everything that can be
checked without one is checked without one, including that every registered source has a
committed checksum, which is the property that keeps "verify before use" from silently
turning into "use".
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from itx.data import synthetic
from itx.data.download import ChecksumMismatchError, fetch, human_bytes, sha256_of
from itx.data.hillstrom import CATEGORICAL, FEATURES, load_hillstrom
from itx.data.ihdp import N_REPLICATES, N_UNITS, load_ihdp, load_ihdp_replicates
from itx.data.registry import SOURCES, checksums, data_dir, source
from itx.data.splits import stratified_split
from itx.types import Split, UpliftDataset


def simple_dataset(n_units: int = 100) -> UpliftDataset:
    rng = np.random.default_rng(0)
    return UpliftDataset(
        name="simple",
        features=pl.DataFrame({"a": rng.normal(size=n_units)}),
        treatment=rng.binomial(1, 0.5, size=n_units).astype(np.int64),
        outcome=rng.binomial(1, 0.3, size=n_units).astype(np.float64),
    )


class TestDataset:
    def test_a_mismatched_treatment_length_is_rejected(self):
        with pytest.raises(ValueError, match="treatment has shape"):
            UpliftDataset(
                name="bad",
                features=pl.DataFrame({"a": [1.0, 2.0, 3.0]}),
                treatment=np.array([0, 1]),
                outcome=np.zeros(3),
            )

    def test_a_non_binary_treatment_is_rejected(self):
        with pytest.raises(ValueError, match="binary"):
            UpliftDataset(
                name="bad",
                features=pl.DataFrame({"a": [1.0, 2.0]}),
                treatment=np.array([0, 2]),
                outcome=np.zeros(2),
            )

    def test_a_categorical_column_that_is_not_a_feature_is_rejected(self):
        with pytest.raises(ValueError, match="categorical columns not in features"):
            UpliftDataset(
                name="bad",
                features=pl.DataFrame({"a": [1.0, 2.0]}),
                treatment=np.array([0, 1]),
                outcome=np.zeros(2),
                categorical=("missing",),
            )

    def test_a_mismatched_ground_truth_length_is_rejected(self):
        with pytest.raises(ValueError, match="true_effect has shape"):
            UpliftDataset(
                name="bad",
                features=pl.DataFrame({"a": [1.0, 2.0]}),
                treatment=np.array([0, 1]),
                outcome=np.zeros(2),
                true_effect=np.zeros(3),
            )

    def test_taking_a_subset_carries_every_array_with_it(self):
        data = synthetic.heterogeneous_effect(200, seed=0)
        subset = data.take(np.array([3, 1, 4]))
        assert subset.n_units == 3
        assert subset.treatment.tolist() == data.treatment[[3, 1, 4]].tolist()
        assert subset.require_true_effect() == pytest.approx(
            data.require_true_effect()[[3, 1, 4]]
        )
        assert subset.propensity is not None

    def test_asking_for_ground_truth_that_is_not_there_is_an_error(self):
        with pytest.raises(ValueError, match="no ground truth"):
            simple_dataset().require_true_effect()

    def test_a_binary_outcome_is_recognised(self):
        assert simple_dataset().outcome_is_binary
        assert not synthetic.heterogeneous_effect(50, seed=0).outcome_is_binary

    def test_describe_reports_both_arms(self):
        text = simple_dataset().describe()
        assert "treated" in text
        assert "control" in text


class TestSplits:
    def test_the_three_parts_hold_every_row_exactly_once(self):
        data = simple_dataset(500)
        split = stratified_split(data, 11)
        total = split.train.n_units + split.validation.n_units + split.test.n_units
        assert total == data.n_units

    def test_the_parts_are_about_the_requested_size(self):
        data = simple_dataset(1_000)
        split = stratified_split(data, 11)
        assert split.train.n_units == pytest.approx(600, abs=5)
        assert split.validation.n_units == pytest.approx(200, abs=5)
        assert split.test.n_units == pytest.approx(200, abs=5)

    def test_the_parts_do_not_overlap(self):
        data = synthetic.heterogeneous_effect(600, seed=0)
        split = stratified_split(data, 11)
        # The true effect is unique per row here, so it doubles as a row identifier.
        parts = [
            set(part.require_true_effect().tolist())
            for part in (split.train, split.validation, split.test)
        ]
        assert parts[0].isdisjoint(parts[1])
        assert parts[0].isdisjoint(parts[2])
        assert parts[1].isdisjoint(parts[2])

    def test_the_same_seed_gives_the_same_partition(self):
        data = simple_dataset(300)
        first = stratified_split(data, 11)
        second = stratified_split(data, 11)
        assert first.test.outcome.tolist() == second.test.outcome.tolist()

    def test_a_different_seed_gives_a_different_partition(self):
        data = simple_dataset(300)
        first = stratified_split(data, 11)
        other = stratified_split(data, 23)
        assert first.test.outcome.tolist() != other.test.outcome.tolist()

    def test_the_treated_share_is_preserved_in_every_part(self):
        data = synthetic.binary_outcome(4_000, propensity=0.3, seed=0)
        split = stratified_split(data, 11)
        for part in (split.train, split.validation, split.test):
            assert part.treatment.mean() == pytest.approx(0.3, abs=0.02)

    def test_the_outcome_rate_is_preserved_in_every_part(self):
        data = synthetic.binary_outcome(4_000, seed=0)
        overall = data.outcome.mean()
        split = stratified_split(data, 11)
        for part in (split.train, split.validation, split.test):
            assert part.outcome.mean() == pytest.approx(overall, abs=0.02)

    def test_a_continuous_outcome_is_stratified_on_the_arm_alone(self):
        data = synthetic.heterogeneous_effect(2_000, propensity=0.25, seed=0)
        split = stratified_split(data, 11)
        for part in (split.train, split.validation, split.test):
            assert part.treatment.mean() == pytest.approx(0.25, abs=0.03)

    @pytest.mark.parametrize("fractions", [(0.5, 0.5), (0.5, 0.3, 0.3), (0.6, 0.2, -0.2)])
    def test_impossible_fractions_are_rejected(self, fractions):
        with pytest.raises(ValueError, match="fractions"):
            stratified_split(simple_dataset(), 11, fractions)

    def test_the_split_records_its_seed(self):
        split = stratified_split(simple_dataset(), 37)
        assert isinstance(split, Split)
        assert split.seed == 37
        assert "37" in split.describe()


class TestRegistry:
    def test_every_source_has_a_committed_checksum(self):
        for key, spec in SOURCES.items():
            assert len(spec.expected_sha256) == 64, key

    def test_checksums_parse_into_filename_and_digest(self):
        digests = checksums()
        assert "hillstrom.csv" in digests
        assert all(len(value) == 64 for value in digests.values())

    def test_an_unknown_source_is_an_error(self):
        with pytest.raises(KeyError, match="unknown data source"):
            source("not-a-dataset")

    def test_the_data_directory_can_be_overridden(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ITX_DATA_DIR", str(tmp_path))
        assert data_dir() == tmp_path

    def test_a_source_without_a_checksum_says_how_to_add_one(self):
        from itx.data.registry import Source

        unregistered = Source(
            key="ghost", filename="ghost.csv", url="https://example.invalid", licence="none"
        )
        with pytest.raises(KeyError, match="no committed checksum"):
            _ = unregistered.expected_sha256


class TestDownload:
    def test_hashing_a_file_matches_the_reference_digest(self, tmp_path):
        path = tmp_path / "content.txt"
        path.write_bytes(b"itx")
        # sha256 of the three bytes "itx", from the coreutils sha256sum, not from this code.
        assert sha256_of(path) == (
            "8f4b775c10828a9bf0e10e652b78245e8384497c08210d855a5f1170e91e34ca"
        )

    def test_hashing_reads_in_chunks_rather_than_loading_the_file(self, tmp_path):
        # The Criteo download is 297 MB; the digest has to be streamed. A file larger than
        # one chunk proves the loop runs more than once.
        path = tmp_path / "big.bin"
        path.write_bytes(b"x" * (3 * (1 << 20) + 17))
        assert len(sha256_of(path)) == 64

    def test_a_corrupted_cache_is_detected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ITX_DATA_DIR", str(tmp_path))
        spec = source("hillstrom")
        (tmp_path / spec.filename).write_bytes(b"not the real file")

        def refuse(*args, **kwargs):
            msg = "download blocked in tests"
            raise RuntimeError(msg)

        monkeypatch.setattr("itx.data.download._download", refuse)
        with pytest.raises(RuntimeError, match="download blocked"):
            fetch("hillstrom", quiet=True)

    def test_byte_counts_render_readably(self):
        assert human_bytes(512) == "512.0 B"
        assert human_bytes(2048) == "2.0 KB"
        assert human_bytes(5 * 1024**3) == "5.0 GB"


@pytest.mark.slow
class TestLoaders:
    def test_hillstrom_loads_the_two_arm_experiment(self):
        data = load_hillstrom()
        assert data.n_units == 42_693
        assert data.feature_names == list(FEATURES)
        assert data.categorical == CATEGORICAL
        assert data.outcome_is_binary
        # Randomised in three equal arms, so two of them split about evenly.
        assert data.treatment.mean() == pytest.approx(0.5, abs=0.01)
        assert data.propensity == pytest.approx(np.full(data.n_units, 0.5))

    def test_hillstrom_reproduces_the_published_visit_rates(self):
        data = load_hillstrom()
        treated = data.treatment == 1
        assert data.outcome[treated].mean() == pytest.approx(0.1514, abs=0.001)
        assert data.outcome[~treated].mean() == pytest.approx(0.1062, abs=0.001)

    def test_hillstrom_can_return_a_continuous_outcome(self):
        data = load_hillstrom(outcome="spend")
        assert not data.outcome_is_binary
        assert data.outcome.max() > 100

    def test_ihdp_loads_all_the_units_and_the_truth(self):
        data = load_ihdp(0)
        assert data.n_units == N_UNITS
        assert data.true_effect is not None
        # Assignment was not randomised: the propensity is unknown by design.
        assert data.propensity is None
        # The published average effect for this benchmark is about 4.
        assert data.true_effect.mean() == pytest.approx(4.0, abs=0.2)

    def test_ihdp_replicates_differ_from_each_other(self):
        first = load_ihdp(0)
        second = load_ihdp(1)
        assert not np.allclose(first.outcome, second.outcome)

    def test_ihdp_yields_the_requested_number_of_replicates(self):
        replicates = list(load_ihdp_replicates(3))
        assert [r.replicate for r in replicates] == [0, 1, 2]

    @pytest.mark.parametrize("replicate", [-1, N_REPLICATES])
    def test_an_out_of_range_replicate_is_an_error(self, replicate):
        with pytest.raises(ValueError, match="replicate must be"):
            load_ihdp(replicate)

    def test_the_cached_files_still_match_their_committed_checksums(self):
        for key, spec in SOURCES.items():
            path = data_dir() / spec.filename
            if path.exists():
                assert sha256_of(path) == spec.expected_sha256, key


def test_checksum_mismatch_is_its_own_error_type():
    assert issubclass(ChecksumMismatchError, RuntimeError)
