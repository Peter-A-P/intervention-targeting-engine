"""Intervention Targeting Engine: uplift modelling with budget-constrained targeting.

The public surface is the estimator protocol, the dataset container, the metrics and the
policy rules. Everything a benchmark reports comes from a held-out test split and carries
a bootstrap interval; a bare point estimate is treated here as a defect.
"""

from itx.types import BoolArray, FloatArray, IntArray, Split, UpliftDataset

__version__ = "0.1.0.dev0"

__all__ = [
    "BoolArray",
    "FloatArray",
    "IntArray",
    "Split",
    "UpliftDataset",
    "__version__",
]
