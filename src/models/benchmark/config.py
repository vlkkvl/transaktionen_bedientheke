"""Configuration for the mature-series benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DESIGN_PATH = ROOT / "reports" / "config" / "forecast_design.json"
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "transactions"


def _positive_int(value: object, name: str) -> int:
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


@dataclass(frozen=True)
class BenchmarkDesign:
    """Fixed forecast design selected by the preparation notebooks."""

    min_active_days: int
    first_origin: pd.Timestamp
    origin_spacing_days: int
    forecast_horizon_days: int
    max_origins: int = 20
    data_dir: Path = DEFAULT_DATA_DIR

    def origins_through(self, last_observed_date: object) -> pd.DatetimeIndex:
        """Return configured origins with a complete observed forecast horizon."""
        last_observed = pd.Timestamp(last_observed_date).normalize()
        last_complete_origin = last_observed - pd.Timedelta(
            days=self.forecast_horizon_days - 1
        )
        if last_complete_origin < self.first_origin:
            return pd.DatetimeIndex([], name="origin")
        origins = pd.date_range(
            self.first_origin,
            last_complete_origin,
            freq=f"{self.origin_spacing_days}D",
            name="origin",
        )
        return origins[-self.max_origins :]


def load_benchmark_design(path: Path = DEFAULT_DESIGN_PATH) -> BenchmarkDesign:
    """Load and validate the shared forecast-design JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Forecast design not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)

    required = {
        "DATA_DIR",
        "MIN_ACTIVE_DAYS",
        "FIRST_ORIGIN",
        "ORIGIN_SPACING_DAYS",
        "FORECAST_HORIZON_DAYS",
        "MAX_ORIGINS",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise KeyError(f"Forecast design is missing: {', '.join(missing)}")

    first_origin = pd.Timestamp(values["FIRST_ORIGIN"]).normalize()
    if pd.isna(first_origin):
        raise ValueError("FIRST_ORIGIN must be a valid date")
    if first_origin.weekday() != 0:
        raise ValueError("FIRST_ORIGIN must be a Monday")
    configured_data_dir = Path(values["DATA_DIR"])
    if not configured_data_dir.is_absolute():
        configured_data_dir = ROOT / configured_data_dir
    return BenchmarkDesign(
        min_active_days=_positive_int(values["MIN_ACTIVE_DAYS"], "MIN_ACTIVE_DAYS"),
        first_origin=first_origin,
        origin_spacing_days=_positive_int(
            values["ORIGIN_SPACING_DAYS"], "ORIGIN_SPACING_DAYS"
        ),
        forecast_horizon_days=_positive_int(
            values["FORECAST_HORIZON_DAYS"], "FORECAST_HORIZON_DAYS"
        ),
        max_origins=_positive_int(values["MAX_ORIGINS"], "MAX_ORIGINS"),
        data_dir=configured_data_dir,
    )
