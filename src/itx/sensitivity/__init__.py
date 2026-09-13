"""Sensitivity to unmeasured confounding: what it would take to overturn the result.

Three devices, and they answer different questions. :mod:`itx.sensitivity.evalue` prices a
confounder's association with treatment and outcome, :mod:`itx.sensitivity.rosenbaum` prices
its effect on the odds of being treated, and :mod:`itx.sensitivity.negative_control` is the
only one that can fail, because it asks the pipeline a question whose answer is already
known (PLAN.md section 6).
"""

from itx.sensitivity.evalue import (
    EValue,
    e_value_of,
    e_value_of_limit,
    risk_ratio_of,
    targeting_e_value,
)
from itx.sensitivity.negative_control import (
    NegativeControl,
    hardest_control,
    negative_control,
)
from itx.sensitivity.rosenbaum import (
    MatchedPairs,
    RosenbaumBound,
    bound_p_value,
    breaking_point,
    match_pairs,
    matching_key,
    signed_rank,
    targeting_rosenbaum,
)

__all__ = [
    "EValue",
    "MatchedPairs",
    "NegativeControl",
    "RosenbaumBound",
    "bound_p_value",
    "breaking_point",
    "e_value_of",
    "e_value_of_limit",
    "hardest_control",
    "match_pairs",
    "matching_key",
    "negative_control",
    "risk_ratio_of",
    "signed_rank",
    "targeting_e_value",
    "targeting_rosenbaum",
]
