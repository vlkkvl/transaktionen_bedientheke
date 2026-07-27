"""Utilities shared by the separate LightGBM model implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.lightgbm.base import (
    BaseLightGBMModel,
    LightGBMVariantResult,
)
from src.models.lightgbm.features.builder import (
    DIAGNOSTIC_COLUMNS,
    FORECAST_ID_COLUMNS,
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
)

GlobalLightGBMModel = BaseLightGBMModel


@dataclass(frozen=True)
class GlobalLightGBMResult:
    """Fitted daily L2 model, evaluation forecasts, and diagnostics."""

    model: GlobalLightGBMModel
    forecasts: pd.DataFrame
    feature_importance: pd.DataFrame
    training_summary: pd.DataFrame
    evaluation_history: dict[str, Any]
    allocation_audit: pd.DataFrame | None = None
    validation_predictions: pd.DataFrame | None = None


def origin_frame_sequence(
    frames: GlobalLightGBMFrames,
) -> tuple[GlobalLightGBMFrames, ...]:
    """Return one independently fitted frame bundle per four-week test block."""
    return frames.origin_frames or (frames,)


def training_summary_fields(frames: GlobalLightGBMFrames) -> dict[str, Any]:
    """Return common diagnostics for one expanding-window model refit."""
    evaluation_origins = pd.DatetimeIndex(
        pd.to_datetime(frames.evaluation["origin"]).unique()
    ).sort_values()
    if frames.evaluation_origin is not None:
        evaluation_origin = pd.Timestamp(frames.evaluation_origin).normalize()
    else:
        evaluation_origin = pd.Timestamp(evaluation_origins.min()).normalize()

    fit_origins = pd.DatetimeIndex(
        pd.to_datetime(frames.training["origin"]).unique()
    )
    validation_origins = pd.DatetimeIndex(
        pd.to_datetime(frames.validation["origin"]).unique()
    )
    return {
        "evaluation_origin": evaluation_origin,
        "evaluation_end": evaluation_origins.max(),
        "training_start": fit_origins.min(),
        "training_end": fit_origins.max(),
        "training_origins": len(fit_origins),
        "validation_start": validation_origins.min(),
        "validation_end": validation_origins.max(),
        "validation_origins": len(validation_origins),
        "available_origins": len(frames.training_origins),
        "fit_rows": len(frames.training),
        "validation_rows": len(frames.validation),
        "evaluation_rows": len(frames.evaluation),
        "evaluation_origins": len(evaluation_origins),
    }


def combine_origin_results(
    results: list[GlobalLightGBMResult | LightGBMVariantResult],
    result_class: type[GlobalLightGBMResult] | type[LightGBMVariantResult],
) -> GlobalLightGBMResult | LightGBMVariantResult:
    """Combine per-refit diagnostics while retaining the latest fitted model."""
    if len(results) == 1:
        return results[0]

    importance_parts = []
    for result in results:
        importance = result.feature_importance.copy()
        importance.insert(
            0,
            "evaluation_origin",
            pd.Timestamp(result.training_summary.iloc[0]["evaluation_origin"]),
        )
        importance_parts.append(importance)
    histories = {
        pd.Timestamp(result.training_summary.iloc[0]["evaluation_origin"])
        .date()
        .isoformat(): result.evaluation_history
        for result in results
    }
    audits = [
        result.allocation_audit
        for result in results
        if result.allocation_audit is not None
    ]
    validation_predictions = [
        result.validation_predictions
        for result in results
        if result.validation_predictions is not None
    ]
    return result_class(
        model=results[-1].model,
        forecasts=pd.concat([result.forecasts for result in results], ignore_index=True),
        feature_importance=pd.concat(importance_parts, ignore_index=True),
        training_summary=pd.concat(
            [result.training_summary for result in results], ignore_index=True
        ),
        evaluation_history=histories,
        allocation_audit=(pd.concat(audits, ignore_index=True) if audits else None),
        validation_predictions=(
            pd.concat(validation_predictions, ignore_index=True)
            if validation_predictions
            else None
        ),
    )


def base_model_params(config: GlobalLightGBMConfig) -> dict[str, Any]:
    """Return tuned defaults shared by the daily architectures."""
    return {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "seed": config.random_state,
        "feature_fraction_seed": config.random_state,
        "bagging_seed": config.random_state,
        "num_threads": config.num_threads,
        "verbosity": -1,
    }


def daily_forecasts(
    frames: GlobalLightGBMFrames,
    model_name: str,
    prediction: np.ndarray,
) -> pd.DataFrame:
    """Build the shared daily forecast output schema."""
    forecasts = frames.evaluation.loc[
        :, [*FORECAST_ID_COLUMNS, *DIAGNOSTIC_COLUMNS]
    ].copy()
    forecasts["model"] = model_name
    forecasts["forecast"] = np.where(
        forecasts["is_active"], np.maximum(prediction, 0.0), np.nan
    )
    return forecasts
