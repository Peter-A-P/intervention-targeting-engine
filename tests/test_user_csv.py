"""Tests for the loader that reads a stranger's CSV.

Most of these are refusals, because most of this loader is refusal. The failure it exists to
prevent is not an exception, it is a clean table computed on data that could never have
supported one.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from itx.data.user_csv import (
    MIN_PER_ARM_PER_BAND,
    ColumnSpec,
    UnusableDataError,
    load_csv,
)


def a_usable_frame(n: int = 4_000, seed: int = 0) -> pl.DataFrame:
    """A file that passes every check, as the baseline the refusals deviate from."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "age": rng.normal(40.0, 10.0, n),
            "plan": rng.choice(["basic", "plus", "premium"], n),
            "got_offer": rng.binomial(1, 0.5, n),
            "churned": rng.binomial(1, 0.2, n),
        }
    )


def a_spec(**overrides: object) -> ColumnSpec:
    fields: dict[str, object] = {
        "treatment": "got_offer",
        "outcome": "churned",
        "features": ("age", "plan"),
        "categorical": ("plan",),
        "design": "randomised",
        "higher_outcome_is_better": False,
    }
    fields.update(overrides)
    return ColumnSpec(**fields)  # type: ignore[arg-type]


def write(frame: pl.DataFrame, tmp_path, name: str = "d.csv"):
    path = tmp_path / name
    frame.write_csv(path)
    return path


class TestTheHappyPath:
    def test_it_loads(self, tmp_path):
        data = load_csv(write(a_usable_frame(), tmp_path), a_spec())
        assert data.n_units == 4_000
        assert data.features.columns == ["age", "plan"]
        assert data.categorical == ("plan",)
        assert set(np.unique(data.treatment)) == {0, 1}

    def test_the_declared_polarity_reaches_the_dataset(self, tmp_path):
        path = write(a_usable_frame(), tmp_path)
        assert load_csv(path, a_spec()).higher_outcome_is_better is False
        assert (
            load_csv(path, a_spec(higher_outcome_is_better=True)).higher_outcome_is_better
            is True
        )

    def test_the_name_falls_back_to_the_file_stem(self, tmp_path):
        data = load_csv(write(a_usable_frame(), tmp_path, "customers.csv"), a_spec())
        assert data.name == "customers"

    def test_a_category_encodes_by_sorted_level_and_is_stable(self, tmp_path):
        frame = a_usable_frame()
        path = write(frame, tmp_path)
        first = load_csv(path, a_spec())
        shuffled = write(frame.sample(fraction=1.0, shuffle=True, seed=1), tmp_path, "b.csv")
        second = load_csv(shuffled, a_spec())
        # Same levels, so the same set of codes, whatever order the rows arrive in.
        assert set(first.features["plan"].to_list()) == set(second.features["plan"].to_list())


class TestTheRefusals:
    def test_a_missing_file(self, tmp_path):
        with pytest.raises(UnusableDataError, match="no such file"):
            load_csv(tmp_path / "absent.csv", a_spec())

    def test_a_missing_column(self, tmp_path):
        path = write(a_usable_frame(), tmp_path)
        with pytest.raises(UnusableDataError, match="columns not in the file"):
            load_csv(path, a_spec(features=("age", "income")))

    def test_the_outcome_used_as_a_feature(self, tmp_path):
        """The leak that makes a model look excellent and mean nothing."""
        path = write(a_usable_frame(), tmp_path)
        with pytest.raises(UnusableDataError, match="both a feature and the treatment"):
            load_csv(path, a_spec(features=("age", "churned")))

    def test_the_treatment_used_as_a_feature(self, tmp_path):
        path = write(a_usable_frame(), tmp_path)
        with pytest.raises(UnusableDataError, match="both a feature and the treatment"):
            load_csv(path, a_spec(features=("age", "got_offer")))

    def test_no_features_at_all(self, tmp_path):
        path = write(a_usable_frame(), tmp_path)
        with pytest.raises(UnusableDataError, match="nothing to model on"):
            load_csv(path, a_spec(features=(), categorical=()))

    def test_a_categorical_that_is_not_a_feature(self, tmp_path):
        path = write(a_usable_frame(), tmp_path)
        with pytest.raises(UnusableDataError, match="declared categorical but not a feature"):
            load_csv(path, a_spec(categorical=("region",)))

    def test_a_treatment_with_more_than_two_arms(self, tmp_path):
        frame = a_usable_frame().with_columns(
            pl.Series("got_offer", np.tile([0, 1, 2, 1], 1_000))
        )
        with pytest.raises(UnusableDataError, match="must be 0 or 1"):
            load_csv(write(frame, tmp_path), a_spec())

    def test_everybody_treated(self, tmp_path):
        frame = a_usable_frame().with_columns(pl.lit(1).alias("got_offer"))
        with pytest.raises(UnusableDataError, match="no untreated rows"):
            load_csv(write(frame, tmp_path), a_spec())

    def test_nobody_treated(self, tmp_path):
        frame = a_usable_frame().with_columns(pl.lit(0).alias("got_offer"))
        with pytest.raises(UnusableDataError, match="no treated rows"):
            load_csv(write(frame, tmp_path), a_spec())

    def test_a_constant_outcome(self, tmp_path):
        frame = a_usable_frame().with_columns(pl.lit(0).alias("churned"))
        with pytest.raises(UnusableDataError, match="one value"):
            load_csv(write(frame, tmp_path), a_spec())

    def test_a_text_outcome(self, tmp_path):
        frame = a_usable_frame().with_columns(
            pl.when(pl.col("churned") == 1)
            .then(pl.lit("yes"))
            .otherwise(pl.lit("no"))
            .alias("churned")
        )
        with pytest.raises(UnusableDataError, match="not a number"):
            load_csv(write(frame, tmp_path), a_spec())

    def test_a_missing_outcome_is_not_imputed(self, tmp_path):
        frame = a_usable_frame().with_columns(
            pl.when(pl.arange(0, pl.len()) < 5)
            .then(None)
            .otherwise(pl.col("churned"))
            .alias("churned")
        )
        with pytest.raises(UnusableDataError, match="missing values"):
            load_csv(write(frame, tmp_path), a_spec())

    def test_too_few_rows_for_the_bands_asked_for(self, tmp_path):
        """The refusal that points at 'itx power' rather than shrugging."""
        frame = a_usable_frame(n=200)
        with pytest.raises(UnusableDataError, match="itx power"):
            load_csv(write(frame, tmp_path), a_spec(), bins=10)

    def test_the_same_small_file_is_fine_with_fewer_bands(self, tmp_path):
        frame = a_usable_frame(n=200)
        data = load_csv(write(frame, tmp_path), a_spec(), bins=2)
        assert data.n_units == 200

    def test_the_threshold_is_per_arm_per_band(self, tmp_path):
        """A file large enough overall can still be too thin once it is cut."""
        frame = a_usable_frame(n=MIN_PER_ARM_PER_BAND * 10)
        with pytest.raises(UnusableDataError, match="per arm per band"):
            load_csv(write(frame, tmp_path), a_spec(), bins=10)

    def test_a_propensity_outside_the_open_interval(self, tmp_path):
        frame = a_usable_frame().with_columns(pl.lit(1.0).alias("p"))
        with pytest.raises(UnusableDataError, match="strictly between 0 and 1"):
            load_csv(write(frame, tmp_path), a_spec(propensity="p"))

    def test_an_empty_file(self, tmp_path):
        frame = a_usable_frame().head(0)
        with pytest.raises(UnusableDataError, match="no rows"):
            load_csv(write(frame, tmp_path), a_spec())


class TestTheObservationalWarning:
    def test_it_warns_and_still_loads(self, tmp_path):
        path = write(a_usable_frame(), tmp_path)
        with pytest.warns(UserWarning, match="declared observational"):
            data = load_csv(path, a_spec(design="observational"))
        assert data.n_units == 4_000

    def test_a_randomised_declaration_is_silent(self, tmp_path):
        path = write(a_usable_frame(), tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            load_csv(path, a_spec(design="randomised"))
