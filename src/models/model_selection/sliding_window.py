"""Rolling-origin one-week-ahead forecasts for every series/model pair.

For each weekly series of length ``N`` we slide a window forward starting at
``min_train_size``: every step fits a model on ``y[:t]`` and predicts ``y[t]``
(horizon = 1 week). The output is a long-form dataframe with one row per
(series, model, window) — downstream code reduces those rows to series-level
and cluster-level metrics.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.models.baseline.registry import build_models
from src.models.model_selection.config import (
    DEFAULT_GROUP_COLS,
    DEFAULT_MIN_TRAIN_SIZE,
    DEFAULT_STEP,
    HORIZON_WEEKS,
)

WINDOW_COLUMNS = (
    "demand_class",
    "model",
    "window_index",
    "train_size",
    "period_start",
    "actual",
    "forecast",
)


def _window_origins(n_obs: int, min_train_size: int, step: int) -> range:
    """Origins of every rolling window (index of the forecasted week)."""
    if step < 1:
        raise ValueError("step must be a positive integer")
    last_origin = n_obs - HORIZON_WEEKS
    if last_origin < min_train_size:
        return range(0)
    return range(min_train_size, last_origin + 1, step)


def _forecast_one_series(
    key_data: dict[str, object],
    demand_class: str,
    values: np.ndarray,
    period_starts: np.ndarray,
    min_train_size: int,
    step: int,
    model_names: Iterable[str] | None,
) -> list[dict[str, object]]:
    """Generate per-window forecast rows for a single series."""
    rows: list[dict[str, object]] = []
    models = build_models(demand_class=demand_class, model_names=model_names)
    if not models:
        return rows

    for window_index, origin in enumerate(
        _window_origins(len(values), min_train_size, step)
    ):
        train = values[:origin]
        actual = float(values[origin])
        period_start = period_starts[origin]
        for model in models:
            forecast = float(model.forecast(train, HORIZON_WEEKS)[0])
            rows.append(
                {
                    **key_data,
                    "demand_class": demand_class,
                    "model": model.name,
                    "window_index": window_index,
                    "train_size": int(origin),
                    "period_start": period_start,
                    "actual": actual,
                    "forecast": forecast,
                }
            )
    return rows


def evaluate_sliding_windows(
    weekly_series: pd.DataFrame,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    step: int = DEFAULT_STEP,
    model_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run rolling-origin one-step forecasts for every series and model."""
    columns = [*group_cols, *WINDOW_COLUMNS]
    if weekly_series.empty:
        return pd.DataFrame(columns=columns)

    groupby_key: str | list[str]
    groupby_key = group_cols[0] if len(group_cols) == 1 else list(group_cols)

    rows: list[dict[str, object]] = []
    for key, group in weekly_series.groupby(groupby_key, sort=False):
        group = group.sort_values("period_start")
        values = group["demand"].to_numpy(dtype=float)
        if len(values) <= min_train_size:
            continue
        period_starts = group["period_start"].to_numpy()
        key_values = key if isinstance(key, tuple) else (key,)
        key_data = dict(zip(group_cols, key_values))
        rows.extend(
            _forecast_one_series(
                key_data=key_data,
                demand_class=str(group["demand_class"].iloc[0]),
                values=values,
                period_starts=period_starts,
                min_train_size=min_train_size,
                step=step,
                model_names=model_names,
            )
        )

    return pd.DataFrame(rows, columns=columns)
