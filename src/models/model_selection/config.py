"""Configuration for cluster-wise baseline model selection."""
from __future__ import annotations

import os
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


DEFAULT_OUTPUT_DIR = ROOT / "reports" / "results"
DEFAULT_GROUP_COLS: tuple[str, ...] = ("ARTIKEL_ID", "MARKT_ID")
DEFAULT_DEMAND_COL = "ABVERKAUFTE_MENGE"

# Choose the aggregation horizon used by model selection.
# Valid values: "daily", "weekly", "monthly".
HORIZON = "weekly"

# How many future periods each window predicts (1 day, 1 week, 1 month etc.)
FORECAST_PERIODS = 1

SUPPORTED_HORIZONS: tuple[str, ...] = ("daily", "weekly", "monthly")
HORIZON_ALIASES = {
    "day": "daily",
    "daily": "daily",
    "week": "weekly",
    "weekly": "weekly",
    "month": "monthly",
    "monthly": "monthly",
}
HORIZON_DATA_DIRS = {
    "daily": ROOT / "data" / "processed" / "transactions_dst_over_days",
    "weekly": ROOT / "data" / "processed" / "transactions_dst_over_weeks",
    "monthly": ROOT / "data" / "processed" / "transactions_dst_over_months",
}
HORIZON_PERIOD_LABELS = {
    "daily": "day",
    "weekly": "week",
    "monthly": "month",
}


def normalize_horizon(horizon: str) -> str:
    """Return the canonical horizon name."""
    normalized = HORIZON_ALIASES.get(str(horizon).strip().lower())
    if normalized is None:
        allowed = ", ".join(SUPPORTED_HORIZONS)
        raise ValueError(f"horizon must be one of: {allowed}")
    return normalized


def data_dir_for_horizon(horizon: str) -> Path:
    """Return the processed parquet directory for a horizon."""
    return HORIZON_DATA_DIRS[normalize_horizon(horizon)]


def period_label_for_horizon(horizon: str) -> str:
    """Return the singular period label for a horizon."""
    return HORIZON_PERIOD_LABELS[normalize_horizon(horizon)]


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


DEFAULT_HORIZON = normalize_horizon(HORIZON)
DEFAULT_FORECAST_PERIODS = _positive_int(FORECAST_PERIODS, "FORECAST_PERIODS")
DEFAULT_DATA_DIR = data_dir_for_horizon(DEFAULT_HORIZON)

# Number of periods with positive demand (used for ADI/CV2 clustering)
DEFAULT_MIN_DEMAND_PERIODS = 24

# Number of available active periods overall (not necessarily with + demand)
# Used for fitting the models
DEFAULT_MIN_TRAIN_SIZE = 52

# How far rolling origin moves between windows
# Here: no overlapping windows (e.g. predict 4 weeks, move forward 4 weeks -> no overlapping)
# If different: overlapping (e.g. predict 4 weeks, move 1 week forward, predict 4 weeks)
DEFAULT_STEP = DEFAULT_FORECAST_PERIODS

# Utilize parallel execution with 7 cpus
DEFAULT_N_JOBS = max(1, (os.cpu_count() or 1) - 1)

DEFAULT_SELECTION_METRIC = "wape_pooled"

DEFAULT_MAX_SERIES_PER_CLUSTER = 2000


@dataclass(frozen=True)
class SelectionConfig:
    """Parameters that drive the end-to-end selection workflow."""

    horizon: str = DEFAULT_HORIZON
    data_dir: Path | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    demand_col: str = DEFAULT_DEMAND_COL
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS

    min_demand_periods: int = DEFAULT_MIN_DEMAND_PERIODS
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE
    forecast_periods: int = DEFAULT_FORECAST_PERIODS
    step: int = DEFAULT_STEP
    n_jobs: int = DEFAULT_N_JOBS
    max_series_per_class: int | None = DEFAULT_MAX_SERIES_PER_CLUSTER

    metric: str = DEFAULT_SELECTION_METRIC
    model_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        horizon = normalize_horizon(self.horizon)
        forecast_periods = _positive_int(self.forecast_periods, "forecast_periods")
        step = forecast_periods if self.step is None else _positive_int(
            self.step, "step"
        )
        data_dir = (
            data_dir_for_horizon(horizon)
            if self.data_dir is None
            else Path(self.data_dir)
        )

        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "forecast_periods", forecast_periods)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "data_dir", data_dir)

    @property
    def period_label(self) -> str:
        """Singular label for one forecast period."""
        return period_label_for_horizon(self.horizon)
