"""The browser's copy of the sample-size arithmetic must agree with the real one.

`demo/power.js` reimplements `src/itx/metrics/power.py` so the page's slider can answer any
input a visitor types without a backend. Two copies of a formula drift, and the copy that
drifts is always the one nobody runs, so this feeds a grid of inputs through both and fails if
any answer differs.

It needs Node, which is on this machine because the demo deploys through `npx`, and it skips
where Node is absent rather than failing: a contributor without Node should still get a green
suite, and CI is where this has to pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from itx.metrics.power import (
    binary_outcome_sd,
    detectable_lift,
    units_to_detect_an_effect,
    units_to_rank,
)

POWER_JS = Path(__file__).resolve().parents[1] / "demo" / "power.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on PATH")

#: Spans the plausible range of each input and includes the awkward ends: a rare outcome, a
#: tiny effect, a tight budget and a lopsided holdout.
BASE_RATES = (0.01, 0.05, 0.1, 0.3, 0.5)
RELATIVE_EFFECTS = (0.02, 0.1, 0.25, 0.5)
BUDGETS = (0.05, 0.1, 0.2, 0.5)
TREATED_SHARES = (0.1, 0.5, 0.85)
LIFT_RATIOS = (1.25, 1.5, 2.0, 3.0)


def run_js(cases: list[dict[str, float]]) -> list[dict[str, float]]:
    """Ask Node for the browser's answer to every case.

    Args:
        cases: One dict per case, with the keys the JS functions take.

    Returns:
        One dict of answers per case, in the same order.
    """
    # Through a file rather than argv: a thousand cases of JSON is past the Windows command
    # line limit, which fails as "the filename or extension is too long".
    with tempfile.TemporaryDirectory() as folder:
        payload = Path(folder) / "cases.json"
        payload.write_text(json.dumps(cases), encoding="utf-8")
        return list(json.loads(_node(payload)))


def _node(payload: Path) -> str:
    """Run the browser's copy over a file of cases and return its stdout.

    Args:
        payload: JSON file holding the list of cases.

    Returns:
        Node's stdout, which is a JSON list of answers.
    """
    script = f"""
const fs = require("fs");
const p = require({json.dumps(str(POWER_JS))});
const cases = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
const out = cases.map((c) => {{
  const outcomeSd = p.binaryOutcomeSd(c.base_rate);
  const averageEffect = c.relative_effect * c.base_rate;
  return {{
    sd: outcomeSd,
    floor: p.unitsToDetectAnEffect({{outcomeSd, averageEffect, treatedShare: c.treated_share}}),
    rank: p.unitsToRank({{outcomeSd, averageEffect, liftRatio: c.lift_ratio,
                          budget: c.budget, treatedShare: c.treated_share}}),
    lift: p.detectableLift({{nUnits: c.n_units, outcomeSd, averageEffect,
                             budget: c.budget, treatedShare: c.treated_share}}),
  }};
}});
process.stdout.write(JSON.stringify(out));
"""
    result = subprocess.run(
        [str(NODE), "-e", script, str(payload)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return result.stdout


def every_case() -> list[dict[str, float]]:
    """The grid, as a flat list so one Node process answers all of it."""
    cases = []
    for base_rate in BASE_RATES:
        for relative_effect in RELATIVE_EFFECTS:
            for budget in BUDGETS:
                for treated_share in TREATED_SHARES:
                    for lift_ratio in LIFT_RATIOS:
                        cases.append(
                            {
                                "base_rate": base_rate,
                                "relative_effect": relative_effect,
                                "budget": budget,
                                "treated_share": treated_share,
                                "lift_ratio": lift_ratio,
                                "n_units": 50_000,
                            }
                        )
    return cases


class TestTheTwoImplementationsAgree:
    @pytest.fixture(scope="class")
    def answers(self) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        cases = every_case()
        return cases, run_js(cases)

    def test_the_grid_is_worth_running(self, answers):
        cases, _ = answers
        assert len(cases) == (
            len(BASE_RATES)
            * len(RELATIVE_EFFECTS)
            * len(BUDGETS)
            * len(TREATED_SHARES)
            * len(LIFT_RATIOS)
        )
        assert len(cases) >= 900

    def test_the_outcome_spread_matches(self, answers):
        cases, js = answers
        for case, got in zip(cases, js, strict=True):
            assert got["sd"] == pytest.approx(binary_outcome_sd(case["base_rate"]), rel=1e-12)

    def test_the_effect_detection_floor_matches_exactly(self, answers):
        """Both round up to a whole number of people, so this is equality, not approximation.

        It was not, at first. The browser copy started with a cheaper normal quantile good to
        about 1e-8, and this test caught the two sides disagreeing by 217 people out of 621
        million, because an error that small can still fall on the wrong side of a rounding
        boundary. The fix was a better quantile rather than a looser assertion.
        """
        cases, js = answers
        for case, got in zip(cases, js, strict=True):
            expected = units_to_detect_an_effect(
                outcome_sd=binary_outcome_sd(case["base_rate"]),
                average_effect=case["relative_effect"] * case["base_rate"],
                treated_share=case["treated_share"],
            )
            assert got["floor"] == expected, case

    def test_the_ranking_requirement_matches_exactly(self, answers):
        cases, js = answers
        for case, got in zip(cases, js, strict=True):
            expected = units_to_rank(
                outcome_sd=binary_outcome_sd(case["base_rate"]),
                average_effect=case["relative_effect"] * case["base_rate"],
                lift_ratio=case["lift_ratio"],
                budget=case["budget"],
                treated_share=case["treated_share"],
            )
            assert got["rank"] == expected, case

    def test_the_detectable_lift_matches(self, answers):
        cases, js = answers
        for case, got in zip(cases, js, strict=True):
            expected = detectable_lift(
                n_units=int(case["n_units"]),
                outcome_sd=binary_outcome_sd(case["base_rate"]),
                average_effect=case["relative_effect"] * case["base_rate"],
                budget=case["budget"],
                treated_share=case["treated_share"],
            )
            assert got["lift"] == pytest.approx(expected, rel=1e-12), case


class TestTheNormalQuantile:
    """The one piece of the JS that is an approximation rather than the same arithmetic."""

    def test_it_matches_scipy_across_the_range(self):
        from scipy import stats

        probabilities = [0.001, 0.01, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.99, 0.999]
        script = f"""
const p = require({json.dumps(str(POWER_JS))});
const ps = JSON.parse(process.argv[1]);
process.stdout.write(JSON.stringify(ps.map(p.normalQuantile)));
"""
        result = subprocess.run(
            [str(NODE), "-e", script, json.dumps(probabilities)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        got = json.loads(result.stdout)
        # AS241 is good to about 1e-16, so this is as tight as double precision allows.
        for probability, value in zip(probabilities, got, strict=True):
            assert value == pytest.approx(float(stats.norm.ppf(probability)), abs=1e-12)
