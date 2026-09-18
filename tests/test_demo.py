"""The demo payload, against the same functions the results table is built from.

The demo's whole claim is that the page does no modelling: every number it can display was
computed here first. That claim is only worth anything if what gets written matches what the
benchmark would have said, so these tests check the payload against
``itx.policy.policy_value.policy_metrics`` rather than against a stored copy of itself.

The other thing asserted is what the file must not contain. It ships to a public URL, so a
feature value leaking into it would be a data problem rather than a bug, and the list of
units is capped so that Criteo's 279,592-row test split cannot turn the page into a
multi-megabyte download.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from itx.bench.runner import BenchmarkRow, nuisances_for
from itx.data import synthetic
from itx.data.splits import stratified_split
from itx.demo.build import (
    DEMO_BUDGETS,
    TOP_UNITS,
    build_payload,
    write_payloads,
)
from itx.metrics.curves import rank_order
from itx.policy.policy_value import gain_key, policy_metrics


@pytest.fixture(scope="module")
def split():
    """One split of a synthetic dataset with a real, heterogeneous effect."""
    return stratified_split(synthetic.heterogeneous_effect(3_000, seed=4), 11)


@pytest.fixture(scope="module")
def rows(split):
    """Two rankings over that split: a real one and a deliberately useless one."""
    rng = np.random.default_rng(0)
    truth = split.test.require_true_effect()
    return [
        BenchmarkRow(
            dataset="synthetic",
            estimator=name,
            seed=11,
            n_test=split.test.n_units,
            metrics={},
            fit_seconds=1.0,
            scores=scores,
        )
        for name, scores in (
            ("oracle", truth + rng.normal(0, 0.1, truth.size)),
            ("noise", rng.normal(size=truth.size)),
        )
    ]


@pytest.fixture(scope="module")
def payload(rows, split):
    return build_payload("synthetic", rows, split, n_resamples=40, seed=11)


class TestTheBudgetGrid:
    def test_the_reported_budgets_land_exactly_on_it(self):
        # A reader checking the demo against the results table should not have to
        # interpolate, so the three budgets the table reports have to be grid points.
        for budget in (0.1, 0.2, 0.3):
            assert budget in DEMO_BUDGETS

    def test_it_runs_from_a_slice_to_the_whole_population(self):
        assert DEMO_BUDGETS[0] == pytest.approx(0.02)
        assert DEMO_BUDGETS[-1] == pytest.approx(1.0)
        assert list(DEMO_BUDGETS) == sorted(DEMO_BUDGETS)


class TestTheCurveMatchesTheBenchmark:
    def test_the_point_estimates_are_the_policy_metrics(self, payload, rows, split):
        # The claim the page rests on. If these ever diverge, the demo is showing something
        # the repository never measured.
        nuisances = nuisances_for(split)
        for row in rows:
            expected = policy_metrics(
                split.test.outcome,
                split.test.treatment,
                row.scores,
                nuisances,
                budgets=DEMO_BUDGETS,
                seed=11,
            )
            curve = payload["estimators"][row.estimator]
            for index, budget in enumerate(DEMO_BUDGETS):
                assert curve["dr"]["value"][index] == pytest.approx(
                    expected[gain_key("dr", budget)], abs=1e-6
                )
                assert curve["ipw"]["value"][index] == pytest.approx(
                    expected[gain_key("ipw", budget)], abs=1e-6
                )

    def test_the_band_is_ordered_and_mostly_contains_the_estimate(self, payload):
        # A percentile band is not built around the point estimate and is not required to
        # contain it, so "low <= value <= high" everywhere would be the wrong assertion and
        # would flake. What must hold is that the bounds are ordered, and that the band is
        # doing its job rather than sitting somewhere else entirely: at a few budgets the
        # resample distribution can straddle differently, but not at most of them.
        for curve in payload["estimators"].values():
            for name in ("dr", "ipw"):
                low, value, high = (curve[name][k] for k in ("low", "value", "high"))
                assert all(a <= b for a, b in zip(low, high, strict=True))
                inside = sum(a <= v <= b for a, v, b in zip(low, value, high, strict=True))
                assert inside >= 0.9 * len(value)

    def test_a_real_ranking_beats_a_useless_one(self, payload):
        # Not a property of the packaging, but if this ever failed the demo would be
        # showing a ranking that does not rank and nobody would be able to tell by eye.
        index = DEMO_BUDGETS.index(0.2)
        assert (
            payload["estimators"]["oracle"]["dr"]["value"][index]
            > payload["estimators"]["noise"]["dr"]["value"][index]
        )


class TestTheRandomBaseline:
    def test_it_is_a_straight_line_through_the_origin(self, payload):
        # The expected gain from treating a random share b is b times the gain from
        # treating everybody, so the curve has to be exactly proportional to the budget.
        random = payload["random"]["dr"]
        whole = random[-1]
        for budget, value in zip(DEMO_BUDGETS, random, strict=True):
            assert value == pytest.approx(whole * budget, abs=1e-6)

    def test_at_a_full_budget_it_equals_every_ranking(self, payload):
        # Treating everybody is the same policy however you sorted them, so at 100% the
        # random line and every estimator's line have to meet. A demo where they did not
        # would be drawing a ranking effect that does not exist.
        whole = payload["random"]["dr"][-1]
        for curve in payload["estimators"].values():
            assert curve["dr"]["value"][-1] == pytest.approx(whole, abs=1e-6)


class TestTheUnitList:
    def test_it_is_the_order_the_policy_would_treat_them(self, payload, rows):
        scores = next(row.scores for row in rows if row.estimator == "oracle")
        expected = rank_order(scores, seed=11)[:TOP_UNITS]
        listed = [unit["row"] for unit in payload["estimators"]["oracle"]["top"]]
        assert listed == [int(position) for position in expected]

    def test_it_is_capped(self, payload):
        for curve in payload["estimators"].values():
            assert len(curve["top"]) <= TOP_UNITS

    def test_it_carries_row_numbers_and_never_feature_values(self, payload, split):
        # The file goes to a public URL. Row numbers cannot be re-identified; feature
        # values are the thing that could be, so they must not be in here at all.
        serialised = json.dumps(payload)
        for column in split.test.feature_names:
            assert f'"{column}"' not in serialised
        for unit in payload["estimators"]["oracle"]["top"]:
            assert set(unit) == {"row", "uplift", "cost"}
            assert 0 <= unit["row"] < split.test.n_units

    def test_costs_are_uniform_on_the_public_datasets(self, payload):
        # Which is why the cost-aware knapsack reduces to rank-and-cut here. The field is
        # carried so the schema does not change when the fraud case varies it.
        for curve in payload["estimators"].values():
            assert {unit["cost"] for unit in curve["top"]} == {1.0}


class TestRefusals:
    def test_rows_without_scores_are_refused(self, split):
        # The mistake this catches is passing `read_json` output instead of a refit: the
        # results file stores metrics, not per-unit scores, so the demo would silently
        # have nothing to rank.
        stored = [
            BenchmarkRow(
                dataset="synthetic",
                estimator="t-learner",
                seed=11,
                n_test=split.test.n_units,
                metrics={},
                fit_seconds=1.0,
                scores=np.zeros(0),
            )
        ]
        with pytest.raises(ValueError, match="no rows carry per-unit scores"):
            build_payload("synthetic", stored, split)


class TestWritingItOut:
    def test_it_writes_one_file_per_dataset_plus_an_index(self, payload, tmp_path):
        written = write_payloads([payload], tmp_path)
        assert [path.name for path in written] == ["synthetic.json", "index.json"]
        index = json.loads((tmp_path / "index.json").read_text())
        assert index["datasets"][0]["key"] == "synthetic"
        assert index["datasets"][0]["n_test"] == payload["n_test"]

    def test_what_it_writes_is_json_the_page_can_read(self, payload, tmp_path):
        write_payloads([payload], tmp_path)
        loaded = json.loads((tmp_path / "synthetic.json").read_text())
        assert loaded["budgets"] == list(DEMO_BUDGETS)
        assert set(loaded["estimators"]) == {"oracle", "noise"}

    def test_rebuilding_one_dataset_keeps_the_others_in_the_index(self, payload, tmp_path):
        # The bug this is here for, found by using the tool rather than by reading it.
        # Building in two batches wrote an index from the second batch alone: the first
        # batch's files stayed on disk, the page stopped offering them, and nothing errored.
        # The shipped demo came out listing three of the five datasets beside it.
        first = payload | {"dataset": "alpha"}
        second = payload | {"dataset": "beta"}
        write_payloads([first], tmp_path)
        write_payloads([second], tmp_path)

        index = json.loads((tmp_path / "index.json").read_text())
        assert [entry["key"] for entry in index["datasets"]] == ["alpha", "beta"]

    def test_the_index_is_ordered_so_two_builds_produce_the_same_file(self, payload, tmp_path):
        write_payloads([payload | {"dataset": "zulu"}], tmp_path)
        write_payloads([payload | {"dataset": "alpha"}], tmp_path)
        index = json.loads((tmp_path / "index.json").read_text())
        keys = [entry["key"] for entry in index["datasets"]]
        assert keys == sorted(keys)

    def test_the_index_describes_each_dataset_without_loading_it(self, payload, tmp_path):
        write_payloads([payload], tmp_path)
        entry = json.loads((tmp_path / "index.json").read_text())["datasets"][0]
        assert set(entry) == {"key", "n_test", "outcome_is_binary"}

    def test_it_creates_the_directory(self, payload, tmp_path):
        target = tmp_path / "does" / "not" / "exist"
        write_payloads([payload], target)
        assert (target / "index.json").is_file()


class TestTheShippedPage:
    """The page is committed, not generated, so these are the checks a build step would do."""

    def test_the_page_loads_the_files_the_builder_writes(self):
        from pathlib import Path

        app = Path("demo/app.js").read_text(encoding="utf-8")
        assert "data/index.json" in app
        assert "data/${key}.json" in app

    def test_the_page_references_only_local_assets(self):
        # No CDN, no analytics, no fonts from elsewhere: a static page whose argument is
        # that you can read it should not be fetching anything you cannot. The stylesheet is
        # checked too, because that is where a font or a background image would hide, and the
        # page passed this test for a while with the check reading the HTML alone.
        from pathlib import Path

        for name in ("index.html", "style.css", "app.js", "content.js", "power.js"):
            text = Path("demo", name).read_text(encoding="utf-8")
            assert "http://" not in text, name
            assert "https://" not in text, name

    def test_the_fonts_it_asks_for_are_in_the_folder(self):
        # The two faces are copied from peterparker.ca rather than linked to it. A missing
        # file here is not a crash, it is a silent fallback to a system serif, which is the
        # kind of defect nobody notices until somebody else does.
        import re
        from pathlib import Path

        css = Path("demo/style.css").read_text(encoding="utf-8")
        referenced = set(re.findall(r"url\(([^)]+)\)", css))
        assert referenced, "the stylesheet declares no font files"
        for target in referenced:
            assert Path("demo", target.strip("\"'")).is_file(), target

    def test_the_hosting_policy_allows_the_fonts_it_serves(self):
        # default-src is 'none', so anything not named is blocked. A font served from the
        # same origin still needs font-src.
        import json
        from pathlib import Path

        config = json.loads(Path("demo/staticwebapp.config.json").read_text(encoding="utf-8"))
        policy = config["globalHeaders"]["Content-Security-Policy"]
        assert "font-src 'self'" in policy
        for directive in policy.split(";"):
            assert "http" not in directive, directive
