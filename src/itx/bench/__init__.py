"""The benchmark: the one command the results table has to come out of."""

from itx.bench.runner import (
    BUDGETS,
    DATASETS,
    DEFAULT_ESTIMATORS,
    ESTIMATORS,
    BenchmarkRow,
    evaluate,
    run,
)
from itx.bench.seeds import SEEDS, TIE_SEED
from itx.bench.table import (
    Summary,
    summarise,
    to_markdown,
    update_markdown_file,
    write_json,
)

__all__ = [
    "BUDGETS",
    "DATASETS",
    "DEFAULT_ESTIMATORS",
    "ESTIMATORS",
    "SEEDS",
    "TIE_SEED",
    "BenchmarkRow",
    "Summary",
    "evaluate",
    "run",
    "summarise",
    "to_markdown",
    "update_markdown_file",
    "write_json",
]
