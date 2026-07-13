"""Rolling-origin forecasts for every series/model pair.

For each period series of length ``N`` we slide a window forward starting at
``min_train_size``: every step fits a model on ``y[:t]`` and predicts
``y[t:t + forecast_periods]``. The output is a long-form dataframe with one row
per forecasted period — downstream code reduces those rows to series-level and
cluster-level metrics.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from src.models.baseline.registry import build_models
from src.models.model_selection.config import (
    DEFAULT_FORECAST_PERIODS,
    DEFAULT_GROUP_COLS,
    DEFAULT_MIN_TRAIN_SIZE,
    DEFAULT_N_JOBS,
    DEFAULT_STEP,
)

WINDOW_COLUMNS = (
    "demand_class",
    "model",
    "window_index",
    "horizon_period",
    "train_size",
    "forecast_origin",
    "period_start",
    "actual",
    "forecast",
    "mase_scale",
)


@dataclass(frozen=True)
class _SeriesTask:
    key_data: dict[str, object]
    demand_class: str
    values: np.ndarray
    period_starts: np.ndarray
    min_train_size: int
    step: int
    forecast_periods: int
    model_names: tuple[str, ...] | None


class _ProgressBar:
    def __init__(self, total: int, label: str, enabled: bool) -> None:
        self.total = total
        self.label = label
        self.enabled = enabled and total > 0
        self.current = 0
        self.width = 32
        self._last_percent = -1
        if self.enabled:
            self._render(force=True)

    def update(self) -> None:
        if not self.enabled:
            return
        self.current += 1
        percent = int((self.current / self.total) * 100)
        if percent != self._last_percent or self.current == self.total:
            self._render()

    def close(self) -> None:
        if not self.enabled:
            return
        if self.current < self.total:
            self.current = self.total
            self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, force: bool = False) -> None:
        percent = int((self.current / self.total) * 100)
        if not force and percent == self._last_percent:
            return
        self._last_percent = percent
        filled = int(self.width * self.current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(
            f"\r{self.label}: [{bar}] "
            f"{self.current:,}/{self.total:,} series ({percent:3d}%)"
        )
        sys.stdout.flush()


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _window_origins(
    n_obs: int,
    min_train_size: int,
    step: int,
    forecast_periods: int,
) -> range:
    """Origins of every rolling window (index of the first forecasted period)."""
    if step < 1:
        raise ValueError("step must be a positive integer")
    last_origin = n_obs - forecast_periods
    if last_origin < min_train_size:
        return range(0)
    return range(min_train_size, last_origin + 1, step)


def _mase_scale(train: np.ndarray) -> float:
    """Mean absolute one-step naive error in the training window."""
    if len(train) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(train))))


def _forecast_one_series(
    key_data: dict[str, object],
    demand_class: str,
    values: np.ndarray,
    period_starts: np.ndarray,
    min_train_size: int,
    step: int,
    forecast_periods: int,
    model_names: Iterable[str] | None,
) -> list[dict[str, object]]:
    """Generate per-window forecast rows for a single series."""
    rows: list[dict[str, object]] = []
    models = build_models(demand_class=demand_class, model_names=model_names)
    if not models:
        return rows

    for window_index, origin in enumerate(
        _window_origins(len(values), min_train_size, step, forecast_periods)
    ):
        train = values[:origin]
        mase_scale = _mase_scale(train)
        actuals = values[origin : origin + forecast_periods]
        forecast_period_starts = period_starts[origin : origin + forecast_periods]
        forecast_origin = period_starts[origin]
        for model in models:
            forecasts = np.asarray(model.forecast(train, forecast_periods), dtype=float)
            if len(forecasts) != forecast_periods:
                raise ValueError(
                    f"{model.name} returned {len(forecasts)} forecasts for "
                    f"horizon={forecast_periods}"
                )

            for horizon_period, (period_start, actual, forecast) in enumerate(
                zip(forecast_period_starts, actuals, forecasts), start=1
            ):
                rows.append(
                    {
                        **key_data,
                        "demand_class": demand_class,
                        "model": model.name,
                        "window_index": window_index,
                        "horizon_period": horizon_period,
                        "train_size": int(origin),
                        "forecast_origin": forecast_origin,
                        "period_start": period_start,
                        "actual": float(actual),
                        "forecast": float(forecast),
                        "mase_scale": mase_scale,
                    }
                )
    return rows


def _forecast_series_task(task: _SeriesTask) -> list[dict[str, object]]:
    return _forecast_one_series(
        key_data=task.key_data,
        demand_class=task.demand_class,
        values=task.values,
        period_starts=task.period_starts,
        min_train_size=task.min_train_size,
        step=task.step,
        forecast_periods=task.forecast_periods,
        model_names=task.model_names,
    )


def _normalize_n_jobs(n_jobs: int) -> int:
    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    return n_jobs


def _series_tasks(
    period_series: pd.DataFrame,
    group_cols: tuple[str, ...],
    min_train_size: int,
    step: int,
    forecast_periods: int,
    model_names: Iterable[str] | None,
) -> list[_SeriesTask]:
    groupby_key: str | list[str]
    groupby_key = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    model_names_tuple = tuple(model_names) if model_names is not None else None

    tasks: list[_SeriesTask] = []
    for key, group in period_series.groupby(groupby_key, sort=False):
        group = group.sort_values("period_start")
        values = group["demand"].to_numpy(dtype=float)
        if len(values) < min_train_size + forecast_periods:
            continue
        period_starts = group["period_start"].to_numpy()
        key_values = key if isinstance(key, tuple) else (key,)
        key_data = dict(zip(group_cols, key_values))
        tasks.append(
            _SeriesTask(
                key_data=key_data,
                demand_class=str(group["demand_class"].iloc[0]),
                values=values,
                period_starts=period_starts,
                min_train_size=min_train_size,
                step=step,
                forecast_periods=forecast_periods,
                model_names=model_names_tuple,
            )
        )
    return tasks


def _flatten_results(
    series_results: Iterable[list[dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in series_results:
        rows.extend(result)
    return rows


def evaluate_sliding_windows(
    period_series: pd.DataFrame,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    step: int = DEFAULT_STEP,
    forecast_periods: int = DEFAULT_FORECAST_PERIODS,
    model_names: Iterable[str] | None = None,
    n_jobs: int = DEFAULT_N_JOBS,
    show_progress: bool = False,
    progress_label: str = "Evaluating baselines",
) -> pd.DataFrame:
    """Run rolling-origin forecasts for every series and model."""
    columns = [*group_cols, *WINDOW_COLUMNS]
    if period_series.empty:
        return pd.DataFrame(columns=columns)

    n_jobs = _normalize_n_jobs(n_jobs)
    forecast_periods = _positive_int(forecast_periods, "forecast_periods")
    tasks = _series_tasks(
        period_series=period_series,
        group_cols=group_cols,
        min_train_size=min_train_size,
        step=step,
        forecast_periods=forecast_periods,
        model_names=model_names,
    )
    if not tasks:
        return pd.DataFrame(columns=columns)

    progress = _ProgressBar(
        total=len(tasks),
        label=progress_label,
        enabled=show_progress,
    )
    if n_jobs == 1 or len(tasks) == 1:
        series_results = []
        for task in tasks:
            series_results.append(_forecast_series_task(task))
            progress.update()
        progress.close()
        rows = _flatten_results(series_results)
        return pd.DataFrame(rows, columns=columns)

    worker_count = min(n_jobs, len(tasks))
    chunk_size = max(1, len(tasks) // (worker_count * 4))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        series_results = []
        for result in executor.map(
            _forecast_series_task, tasks, chunksize=chunk_size
        ):
            series_results.append(result)
            progress.update()
    progress.close()
    rows = _flatten_results(series_results)
    return pd.DataFrame(rows, columns=columns)
