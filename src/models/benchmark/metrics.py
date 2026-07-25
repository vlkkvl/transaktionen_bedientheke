"""Metric reducers for mature-series benchmark forecasts."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


SERIES_COLS = ("ARTIKEL_ID", "MARKT_ID")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else np.nan


def _enrich(forecasts: pd.DataFrame) -> pd.DataFrame:
    enriched = forecasts.loc[
        forecasts["is_active"].fillna(False)
        if "is_active" in forecasts
        else pd.Series(True, index=forecasts.index)
    ].copy()
    enriched["error"] = enriched["forecast"] - enriched["actual"]
    enriched["abs_error"] = enriched["error"].abs()
    valid_scale = enriched["seasonal_mase_scale"].gt(0)
    enriched["seasonal_scaled_abs_error"] = np.where(
        valid_scale,
        enriched["abs_error"] / enriched["seasonal_mase_scale"],
        np.nan,
    )
    return enriched


def summarize_models(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return the requested primary and secondary metrics for every model."""
    columns = [
        "model",
        "pooled_wape",
        "relative_bias",
        "forecast_to_actual_ratio",
        "median_series_wape",
        "seasonal_mase",
        "mae_kg",
        "mean_bias_kg",
        "actual_kg",
        "forecast_kg",
        "n_series",
        "n_origins",
        "n_forecast_rows",
        "seasonal_mase_valid_share",
    ]
    if forecasts.empty:
        return pd.DataFrame(columns=columns)

    enriched = _enrich(forecasts)
    series = (
        enriched.groupby([*SERIES_COLS, "model"], observed=True)
        .agg(abs_error_sum=("abs_error", "sum"), actual_sum=("actual", "sum"))
        .reset_index()
    )
    series["series_wape"] = [
        _ratio(error, actual)
        for error, actual in zip(series["abs_error_sum"], series["actual_sum"])
    ]
    median_wape = series.groupby("model", observed=True)["series_wape"].median()

    rows = []
    for model, group in enriched.groupby("model", observed=True, sort=False):
        actual = float(group["actual"].sum())
        forecast = float(group["forecast"].sum())
        error = forecast - actual
        valid_mase = group["seasonal_scaled_abs_error"].notna()
        rows.append(
            {
                "model": model,
                "pooled_wape": _ratio(float(group["abs_error"].sum()), actual),
                "relative_bias": _ratio(error, actual),
                "forecast_to_actual_ratio": _ratio(forecast, actual),
                "median_series_wape": median_wape.get(model, np.nan),
                "seasonal_mase": group["seasonal_scaled_abs_error"].mean(),
                "mae_kg": group["abs_error"].mean(),
                "mean_bias_kg": group["error"].mean(),
                "actual_kg": actual,
                "forecast_kg": forecast,
                "n_series": len(group[list(SERIES_COLS)].drop_duplicates()),
                "n_origins": group["origin"].nunique(),
                "n_forecast_rows": len(group),
                "seasonal_mase_valid_share": valid_mase.mean(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def segment_wape(
    forecasts: pd.DataFrame,
    segment_cols: str | Iterable[str],
) -> pd.DataFrame:
    """Compute pooled WAPE and volume diagnostics within supplied segments."""
    segments = [segment_cols] if isinstance(segment_cols, str) else list(segment_cols)
    missing = sorted(set(segments) - set(forecasts.columns))
    if missing:
        raise KeyError(f"Forecasts are missing segment columns: {missing}")
    columns = [
        *segments,
        "model",
        "pooled_wape",
        "relative_bias",
        "forecast_to_actual_ratio",
        "actual_kg",
        "n_series",
        "n_origins",
    ]
    if forecasts.empty:
        return pd.DataFrame(columns=columns)

    enriched = _enrich(forecasts)
    rows = []
    for keys, group in enriched.groupby([*segments, "model"], observed=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_data = dict(zip([*segments, "model"], key_values))
        actual = float(group["actual"].sum())
        forecast = float(group["forecast"].sum())
        rows.append(
            {
                **key_data,
                "pooled_wape": _ratio(float(group["abs_error"].sum()), actual),
                "relative_bias": _ratio(forecast - actual, actual),
                "forecast_to_actual_ratio": _ratio(forecast, actual),
                "actual_kg": actual,
                "n_series": len(group[list(SERIES_COLS)].drop_duplicates()),
                "n_origins": group["origin"].nunique(),
            }
        )
    return pd.DataFrame(rows, columns=columns)
