"""Configuration for cluster-wise baseline model selection."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


def find_project_root() -> Path:
    """Find the repository root from this nested script location."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "transactions_dst_over_weeks"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "results"
DEFAULT_GROUP_COLS: tuple[str, ...] = ("ARTIKEL_ID", "MARKT_ID")
DEFAULT_DEMAND_COL = "ABVERKAUFTE_MENGE"

# Horizon is fixed to one week: each sliding window predicts the next week.
HORIZON_WEEKS = 1

DEFAULT_MIN_DEMAND_WEEKS = 10
DEFAULT_MIN_TRAIN_SIZE = 24
DEFAULT_STEP = 1

DEFAULT_SELECTION_METRIC = "wape_median"
# Drop models whose total forecast across the holdout windows covers less than
# this fraction of total actuals — they grossly under-forecast at the cluster
# level and would win wape_median by predicting near zero.
DEFAULT_UNDERFORECAST_RATIO = 0.5


@dataclass(frozen=True)
class SelectionConfig:
    """Parameters that drive the end-to-end selection workflow."""

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    demand_col: str = DEFAULT_DEMAND_COL
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS

    min_demand_weeks: int = DEFAULT_MIN_DEMAND_WEEKS
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE
    step: int = DEFAULT_STEP
    max_series_per_class: int | None = None

    metric: str = DEFAULT_SELECTION_METRIC
    underforecast_ratio: float = DEFAULT_UNDERFORECAST_RATIO
    model_names: tuple[str, ...] | None = None
    random_state: int = 42
