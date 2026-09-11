"""The benchmark: the one command the results table has to come out of."""

from itx.bench.grid import GRID, Selection, applicable_grid, select_config
from itx.bench.runner import (
    BUDGETS,
    DATASETS,
    DEFAULT_ESTIMATORS,
    ESTIMATORS,
    BenchmarkRow,
    evaluate,
    refit_seed,
    run,
)
from itx.bench.seeds import SEEDS, TIE_SEED
from itx.bench.table import (
    Summary,
    selected_configurations,
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
    "GRID",
    "SEEDS",
    "TIE_SEED",
    "BenchmarkRow",
    "Selection",
    "Summary",
    "applicable_grid",
    "evaluate",
    "refit_seed",
    "run",
    "select_config",
    "selected_configurations",
    "summarise",
    "to_markdown",
    "update_markdown_file",
    "write_json",
]
