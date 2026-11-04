"""What the DR-learner's two nuisance models look like on Criteo, TabFM against LightGBM.

The evidence behind the Criteo DR-learner paragraph in docs/estimators.md. EconML's
DR-learner divides by a fitted propensity (clipped at 0.01) and leans on one regression of
the outcome on the covariates with the treatment as one more column. Criteo's treatment is
randomised at a constant 0.85, so a good propensity model is flat near 0.85, and a useful
outcome model predicts a different outcome with the treatment on than off.

This fits both models, as EconML builds them, on the seed-11 training split on each base,
and scores 4,000 random training rows: the propensity's spread, the control weight it
implies, and the predicted effect mu(x, 1) - mu(x, 0). Run from the repository root with the
'learners' and 'foundation' extras installed and Criteo pulled:

    uv run python scripts/probes/criteo_dr_nuisances.py

About 45 minutes on an 8-thread laptop CPU, nearly all of it TabFM.
"""

from __future__ import annotations

import json

import numpy as np

from itx.bench.runner import DATASETS
from itx.data.splits import stratified_split
from itx.estimators.dr_learner import DRLearner
from itx.estimators.lightgbm_base import DEFAULT_CONFIG, LearnerConfig
from itx.estimators.sklearn_bridge import classifier_for, regressor_for
from itx.estimators.tabfm_base import DEFAULT_TABFM

QUANTILES = [0.0, 0.01, 0.5, 0.99, 1.0]


def main() -> None:
    """Fit both nuisance models on each base and print what they predict."""
    split = stratified_split(DATASETS["criteo"](), 11)
    helper = DRLearner(DEFAULT_TABFM, seed=11)
    matrix = helper._matrix(split.train.features)
    categorical = helper._categorical_indices(split.train.features)
    t = np.asarray(split.train.treatment, dtype=np.float64)
    y = np.asarray(split.train.outcome, dtype=np.float64)
    rows = np.random.default_rng(0).choice(matrix.shape[0], size=4_000, replace=False)
    print(f"train {matrix.shape}, treated share {t.mean():.4f}, outcome mean {y.mean():.5f}")

    bases: tuple[tuple[str, LearnerConfig], ...] = (
        ("lightgbm", DEFAULT_CONFIG),
        ("tabfm", DEFAULT_TABFM),
    )
    for name, config in bases:
        classifier = classifier_for(config, 12, categorical).fit(matrix, t.astype(int))
        e = classifier.predict_proba(matrix[rows])[:, 1]
        model = regressor_for(config, 11, categorical).fit(np.hstack([matrix, t[:, None]]), y)
        off = model.predict(np.hstack([matrix[rows], np.zeros((rows.size, 1))]))
        on = model.predict(np.hstack([matrix[rows], np.ones((rows.size, 1))]))
        report = {
            "propensity_quantiles": np.quantile(e, QUANTILES).round(4).tolist(),
            "control_weight_quantiles": np.quantile(1 / np.clip(1 - e, 0.01, None), QUANTILES)
            .round(1)
            .tolist(),
            "predicted_effect_quantiles": np.quantile(on - off, QUANTILES).round(5).tolist(),
            "predicted_effect_mean": float((on - off).mean()),
        }
        print(name, json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
