"""Reduce sliding-window forecasts to series-level and cluster-level metrics.

The pipeline is:

1. ``per_series_metrics`` — aggregate windows for each (series, model).
2. ``per_cluster_metrics`` — aggregate series for each (demand_class, model),
   reporting both ``wape_median`` (typical-series accuracy) and ``wape_pooled``
   (cluster-total accuracy).

These steps purely compute numbers; they make no selection decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baseline.registry import DEMAND_CLASSES
from src.models.model_selection.config import DEFAULT_GROUP_COLS

PER_SERIES_COLUMNS = (
    "demand_class",
    "model",
    "n_windows",  # Number of rolling forecast windows for this series/model.
    "abs_error_sum",  # Total absolute forecast error over all forecasted periods.
    "actual_sum",  # Total absolute actual demand over all forecasted periods.
    "forecast_sum",  # Total forecasted demand over all forecasted periods.
    "mae",
    "rmse",
    "bias",
    "wape",
    "forecast_to_actual_ratio",  # Forecasted volume divided by actual volume.
)

PER_CLUSTER_COLUMNS = (
    "demand_class",
    "model",
    "wape_median",  # Median per-series WAPE; typical relative error.
    "wape_pooled",  # Cluster-level WAPE from pooled total errors and actuals.
    "n_series",  # Number of product-store series evaluated in the cluster.
    "n_windows",  # Total rolling forecast windows across all series.
    "mae_mean",
    "mae_median",
    "rmse_mean",
    "bias_mean",  # Average per-series direction of forecast bias.
    "abs_error_sum",  # Total absolute error across the whole cluster.
    "actual_sum",  # Total absolute actual demand across the whole cluster.
    "forecast_sum",  # Total forecasted demand across the whole cluster.
    "forecast_to_actual_ratio",  # Cluster forecast volume divided by actual volume.
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio that returns 0 for 0/0 and +inf for x/0 with x > 0."""
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 else np.inf


def per_series_metrics(
    window_scores: pd.DataFrame,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    """Collapse window rows to one row per (series, model)."""
    columns = [*group_cols, *PER_SERIES_COLUMNS]
    if window_scores.empty:
        return pd.DataFrame(columns=columns)

    errors = window_scores["forecast"] - window_scores["actual"]
    enriched = window_scores.assign(
        abs_error=errors.abs(),
        squared_error=errors.pow(2),
        signed_error=errors,
        abs_actual=window_scores["actual"].abs(),
    )

    grouped = enriched.groupby(
        [*group_cols, "demand_class", "model"], observed=True, sort=False
    ).agg(
        n_windows=("window_index", "nunique"),
        abs_error_sum=("abs_error", "sum"),
        actual_sum=("abs_actual", "sum"),
        forecast_sum=("forecast", "sum"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),  # Intermediate value for RMSE.
        bias=("signed_error", "mean"),
    ).reset_index()

    grouped["rmse"] = np.sqrt(grouped["mse"])
    grouped["wape"] = [
        _safe_ratio(num, den)
        for num, den in zip(grouped["abs_error_sum"], grouped["actual_sum"])
    ]
    grouped["forecast_to_actual_ratio"] = [
        _safe_ratio(num, den)
        for num, den in zip(grouped["forecast_sum"], grouped["actual_sum"])
    ]
    return grouped[columns]


def per_cluster_metrics(series_scores: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-series rows to one row per (demand_class, model)."""
    if series_scores.empty:
        return pd.DataFrame(columns=list(PER_CLUSTER_COLUMNS))

    summary = (
        series_scores.groupby(["demand_class", "model"], observed=True)
        .agg(
            n_series=("mae", "size"),
            n_windows=("n_windows", "sum"),
            mae_mean=("mae", "mean"),
            mae_median=("mae", "median"),
            rmse_mean=("rmse", "mean"),
            bias_mean=("bias", "mean"),
            wape_median=("wape", "median"),
            abs_error_sum=("abs_error_sum", "sum"),
            actual_sum=("actual_sum", "sum"),
            forecast_sum=("forecast_sum", "sum"),
        )
        .reset_index()
    )

    summary["wape_pooled"] = [
        _safe_ratio(num, den)
        for num, den in zip(summary["abs_error_sum"], summary["actual_sum"])
    ]
    summary["forecast_to_actual_ratio"] = [
        _safe_ratio(num, den)
        for num, den in zip(summary["forecast_sum"], summary["actual_sum"])
    ]

    summary["demand_class"] = pd.Categorical(
        summary["demand_class"], categories=DEMAND_CLASSES, ordered=True
    )
    return (
        summary[list(PER_CLUSTER_COLUMNS)]
        .sort_values(["demand_class", "model"])
        .reset_index(drop=True)
    )
