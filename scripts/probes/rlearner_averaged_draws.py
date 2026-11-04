"""Does averaging several weighted context draws steady the R-learner on TabFM, on ACIC?

The follow-up to ``rlearner_context_draw.py``, which found the R-learner's TabFM result is
decided by which rows one weighted draw puts in the context. This fits the R-learner on
TabFM (CPU, float32) at each of the five committed split seeds with
``TabFMConfig.weighted_draws`` set, and prints its PEHE beside the one-draw CPU run recorded
in docs/estimators.md and LightGBM's. Each finished seed is written to the checkpoint file,
so a run cut short by a reboot resumes where it stopped.

Run from the repository root with the 'learners' and 'foundation' extras installed:

    uv run python scripts/probes/rlearner_averaged_draws.py 8 averaged-draws.json

About 45 minutes a seed on an 8-thread laptop CPU at eight draws: the outcome model's
cross-fitting is unchanged, and each extra draw adds one final-stage fit and prediction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from itx.data.acic import load_acic
from itx.data.splits import stratified_split
from itx.estimators.r_learner import RLearner
from itx.estimators.tabfm_base import DEFAULT_TABFM
from itx.metrics.ground_truth import pehe

#: The committed split seeds, in the order docs/estimators.md lists them.
SEEDS = (11, 23, 37, 53, 71)

#: The one-draw R-learner on TabFM, CPU float32, as recorded in docs/estimators.md.
ONE_DRAW_CPU = {11: 3.48, 23: 16.42, 37: 36.97, 53: 0.89, 71: 43.91}


def main(draws: int, checkpoint: Path) -> None:
    """Fit each seed not yet in the checkpoint, then print every seed's PEHE."""
    done: dict[str, float] = (
        json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.is_file() else {}
    )
    data = load_acic()
    config = DEFAULT_TABFM.with_(weighted_draws=draws)
    for seed in SEEDS:
        if str(seed) in done:
            continue
        split = stratified_split(data, seed)
        model = RLearner(config, seed=seed).fit(split.train)
        done[str(seed)] = pehe(
            model.predict_uplift(split.test.features), split.test.require_true_effect()
        )
        checkpoint.write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(f"  seed {seed}: PEHE {done[str(seed)]:.4f} at {draws} draws", flush=True)
    print(f"seed, one draw (CPU), {draws} draws (CPU)")
    for seed in SEEDS:
        print(f"{seed}, {ONE_DRAW_CPU[seed]:.2f}, {done[str(seed)]:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]), Path(sys.argv[2]))
