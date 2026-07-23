"""LightGBM architecture variants for demand forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

from src.models.benchmark.config import BenchmarkDesign, DEFAULT_DATA_DIR
from src.models.machine_learning.lightgbm.gbm_base import (
    BaseLightGBMModel,
    LightGBMVariantResult,
    TwoStageLightGBMModel,
    fit_lightgbm_model,
    lightgbm_feature_importance,
    wape_feval,
)
from src.models.machine_learning.lightgbm.lightgbm_features import (
    CATEGORICAL_FEATURES,
    DIAGNOSTIC_COLUMNS,
    FEATURE_COLUMNS,
    FORECAST_ID_COLUMNS,
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
    prepare_global_lightgbm_frames,
)

MODEL_NAME = "global_lightgbm"
TWEEDIE_MODEL_NAME = "global_lightgbm_tweedie_daily"
TWO_STAGE_MODEL_NAME = "global_lightgbm_two_stage"
WEEKLY_MODEL_NAME = "global_lightgbm_weekly_total"

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


WEEKLY_FEATURE_COLUMNS = (
    "ARTIKEL_ID",
    "MARKT_ID",
    "sourcing_group",
    "category_id",
    "week_iso_week",
    "week_month",
    "target_days",
    "event_window_days",
    "min_abs_days_to_event",
    "action_on_forecast_day",
    "action_during_horizon",
    "days_since_last_action",
    "actions_last_28d",
    "mean_action_lift_in_sourcing_group",
    "active_days_before_origin",
    "demand_days_before_origin",
    "demand_day_ratio",
    "days_since_last_positive_sale",
    "maturity_segment",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_7_mean",
    "rolling_28_mean",
    "rolling_28_demand_rate",
    "rolling_28_positive_mean",
    "ADI",
    "CV2",
    "zero_share",
    "product_cross_store_mean_28",
    "store_category_mean_28",
    "same_weekday_mean_4_total",
    "same_weekday_mean_8_total",
    "product_weekday_profile_sum",
    "recent_mean_28_total_forecast",
    "same_weekday_ma_4_total_forecast",
    "occurrence_positive_quantity_total_forecast",
)

WEEKLY_CATEGORICAL_FEATURES = (
    "ARTIKEL_ID",
    "MARKT_ID",
    "sourcing_group",
    "category_id",
    "week_iso_week",
    "week_month",
    "maturity_segment",
)

WEEK_KEYS = ("ARTIKEL_ID", "MARKT_ID", "sourcing_group", "category_id", "origin")

LIGHTGBM_MODEL_LABELS = {
    MODEL_NAME: "Daily direct LightGBM (L2)",
    TWEEDIE_MODEL_NAME: "Daily direct LightGBM (Tweedie)",
    TWO_STAGE_MODEL_NAME: "Two-stage LightGBM (occurrence × quantity)",
    WEEKLY_MODEL_NAME: "7-day total LightGBM × weekday allocation",
}


def _origin_frame_sequence(
    frames: GlobalLightGBMFrames,
) -> tuple[GlobalLightGBMFrames, ...]:
    """Return one independently fitted frame bundle per evaluation origin."""
    return frames.origin_frames or (frames,)


def _evaluation_origin(frames: GlobalLightGBMFrames) -> pd.Timestamp:
    if frames.evaluation_origin is not None:
        return pd.Timestamp(frames.evaluation_origin).normalize()
    origins = pd.DatetimeIndex(pd.to_datetime(frames.evaluation["origin"]).unique())
    if len(origins) != 1:
        raise ValueError("A fitted frame bundle must contain exactly one evaluation origin")
    return pd.Timestamp(origins[0]).normalize()


def _training_summary_fields(frames: GlobalLightGBMFrames) -> dict[str, Any]:
    fit_origins = pd.DatetimeIndex(pd.to_datetime(frames.training["origin"]).unique())
    validation_origins = pd.DatetimeIndex(
        pd.to_datetime(frames.validation["origin"]).unique()
    )
    return {
        "evaluation_origin": _evaluation_origin(frames),
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
        "evaluation_origins": frames.evaluation["origin"].nunique(),
    }


def _combine_origin_results(
    results: list[GlobalLightGBMResult | LightGBMVariantResult],
    result_class: type[GlobalLightGBMResult] | type[LightGBMVariantResult],
) -> GlobalLightGBMResult | LightGBMVariantResult:
    """Combine per-origin diagnostics while retaining the latest fitted model."""
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
    combined = result_class(
        model=results[-1].model,
        forecasts=pd.concat([result.forecasts for result in results], ignore_index=True),
        feature_importance=pd.concat(importance_parts, ignore_index=True),
        training_summary=pd.concat(
            [result.training_summary for result in results], ignore_index=True
        ),
        evaluation_history=histories,
        allocation_audit=(pd.concat(audits, ignore_index=True) if audits else None),
    )
    return combined


def _base_model_params(config: GlobalLightGBMConfig) -> dict[str, Any]:
    """Shared tuned LightGBM defaults for all direct and daily variants."""
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


def _daily_forecasts(
    frames: GlobalLightGBMFrames,
    model_name: str,
    prediction: np.ndarray,
) -> pd.DataFrame:
    forecasts = frames.evaluation.loc[:, [*FORECAST_ID_COLUMNS, *DIAGNOSTIC_COLUMNS]].copy()
    forecasts["model"] = model_name
    forecasts["forecast"] = np.maximum(prediction, 0.0)
    return forecasts


def _fit_daily_model(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: GlobalLightGBMConfig,
    *,
    objective: str,
    metric: str | None = None,
    labels: tuple[pd.Series, pd.Series],
) -> tuple[BaseLightGBMModel, dict[str, dict[str, list[float]]], int]:
    return fit_lightgbm_model(
        training=training,
        validation=validation,
        labels=labels,
        evaluation=evaluation,
        params={
            **_base_model_params(config),
            "objective": objective,
            **({"metric": metric} if metric else {}),
        },
        feature_columns=FEATURE_COLUMNS,
        categorical_features=CATEGORICAL_FEATURES,
        config=config,
        feval=wape_feval,
    )


def _fit_global_lightgbm_origin(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig,
) -> GlobalLightGBMResult:
    """Fit the direct daily L2 model for one evaluation origin."""
    model, history, best_iteration = _fit_daily_model(
        frames.training,
        frames.validation,
        frames.evaluation,
        config,
        objective="regression_l2",
        metric="l2",
        labels=(frames.training["actual"], frames.validation["actual"]),
    )
    forecasts = _daily_forecasts(frames, MODEL_NAME, model.predict(frames.evaluation))
    training_summary = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                **_training_summary_fields(frames),
                "best_iteration": best_iteration,
                "objective": "regression_l2",
                "features": len(FEATURE_COLUMNS),
            }
        ]
    )
    return GlobalLightGBMResult(
        model=model,
        forecasts=forecasts,
        feature_importance=lightgbm_feature_importance(model),
        training_summary=training_summary,
        evaluation_history=history,
    )


def fit_global_lightgbm_frames(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
) -> GlobalLightGBMResult:
    """Refit the direct daily L2 model independently at every origin."""
    config = GlobalLightGBMConfig() if config is None else config
    results = [
        _fit_global_lightgbm_origin(origin_frames, config)
        for origin_frames in _origin_frame_sequence(frames)
    ]
    combined = _combine_origin_results(results, GlobalLightGBMResult)
    assert isinstance(combined, GlobalLightGBMResult)
    return combined


def run_global_lightgbm(
    *,
    design: BenchmarkDesign,
    evaluation_origins: Iterable[object],
    connection: duckdb.DuckDBPyConnection | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    config: GlobalLightGBMConfig | None = None,
) -> GlobalLightGBMResult:
    """Refit the global model at each requested origin and combine forecasts."""
    config = GlobalLightGBMConfig() if config is None else config
    frames = prepare_global_lightgbm_frames(
        design=design,
        evaluation_origins=evaluation_origins,
        connection=connection,
        data_dir=data_dir,
        config=config,
    )
    return fit_global_lightgbm_frames(frames, config)


def _fit_tweedie_daily_origin(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig,
) -> LightGBMVariantResult:
    """Fit a direct daily Tweedie model for one evaluation origin."""
    model, history, best_iteration = fit_lightgbm_model(
        training=frames.training,
        validation=frames.validation,
        labels=(frames.training["actual"], frames.validation["actual"]),
        evaluation=frames.evaluation,
        params={
            **_base_model_params(config),
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "metric": "None",
        },
        feature_columns=FEATURE_COLUMNS,
        categorical_features=CATEGORICAL_FEATURES,
        config=config,
        feval=wape_feval,
    )
    summary = pd.DataFrame(
        [
            {
                "model": TWEEDIE_MODEL_NAME,
                **_training_summary_fields(frames),
                "best_iteration": best_iteration,
                "objective": "tweedie_1.5",
                "features": len(FEATURE_COLUMNS),
            }
        ]
    )
    return LightGBMVariantResult(
        model=model,
        forecasts=_daily_forecasts(frames, TWEEDIE_MODEL_NAME, model.predict(frames.evaluation)),
        feature_importance=lightgbm_feature_importance(model),
        training_summary=summary,
        evaluation_history=history,
    )


def fit_tweedie_daily(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
) -> LightGBMVariantResult:
    """Refit the daily Tweedie model independently at every origin."""
    config = GlobalLightGBMConfig() if config is None else config
    results = [
        _fit_tweedie_daily_origin(origin_frames, config)
        for origin_frames in _origin_frame_sequence(frames)
    ]
    combined = _combine_origin_results(results, LightGBMVariantResult)
    assert isinstance(combined, LightGBMVariantResult)
    return combined


def _train_daily_stage(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: GlobalLightGBMConfig,
    *,
    objective: str,
    metric: str | None = None,
    labels: tuple[pd.Series, pd.Series],
) -> tuple[BaseLightGBMModel, dict[str, dict[str, list[float]]], int]:
    return fit_lightgbm_model(
        training=training,
        validation=validation,
        labels=labels,
        evaluation=evaluation,
        params={
            **_base_model_params(config),
            "objective": objective,
            **({"metric": metric} if metric else {}),
        },
        feature_columns=FEATURE_COLUMNS,
        categorical_features=CATEGORICAL_FEATURES,
        config=config,
    )


def _fit_two_stage_origin(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig,
) -> LightGBMVariantResult:
    """Fit occurrence and conditional quantity models for one origin."""
    occurrence, occurrence_history, occurrence_iteration = _train_daily_stage(
        frames.training,
        frames.validation,
        frames.evaluation,
        config,
        objective="binary",
        metric="binary_logloss",
        labels=(
            frames.training["actual"].gt(0).astype(np.int8),
            frames.validation["actual"].gt(0).astype(np.int8),
        ),
    )

    positive_training = frames.training.loc[frames.training["actual"].gt(0)].copy()
    positive_validation = frames.validation.loc[frames.validation["actual"].gt(0)].copy()
    if positive_training.empty or positive_validation.empty:
        raise RuntimeError(
            "Positive-only train/validation splits are empty; two-stage model cannot be fit."
        )

    quantity, quantity_history, quantity_iteration = _train_daily_stage(
        positive_training,
        positive_validation,
        frames.evaluation,
        config,
        objective="regression_l1",
        metric="l1",
        labels=(positive_training["actual"], positive_validation["actual"]),
    )
    model = TwoStageLightGBMModel(occurrence=occurrence, quantity=quantity)
    occurrence_prediction, quantity_prediction, prediction = model.predict_components(
        frames.evaluation
    )
    forecasts = _daily_forecasts(frames, TWO_STAGE_MODEL_NAME, prediction)
    forecasts["occurrence_probability"] = occurrence_prediction
    forecasts["positive_quantity_forecast"] = quantity_prediction

    occurrence_importance = lightgbm_feature_importance(occurrence)
    occurrence_importance.insert(0, "stage", "occurrence")
    quantity_importance = lightgbm_feature_importance(quantity)
    quantity_importance.insert(0, "stage", "positive_quantity")
    importance = pd.concat([occurrence_importance, quantity_importance], ignore_index=True)

    summary = pd.DataFrame(
        [
            {
                "model": TWO_STAGE_MODEL_NAME,
                "stage": "occurrence",
                **_training_summary_fields(frames),
                "best_iteration": occurrence_iteration,
                "objective": "binary_logloss",
                "features": len(FEATURE_COLUMNS),
            },
            {
                "model": TWO_STAGE_MODEL_NAME,
                "stage": "positive_quantity",
                **_training_summary_fields(frames),
                "fit_rows": len(positive_training),
                "validation_rows": len(positive_validation),
                "best_iteration": quantity_iteration,
                "objective": "regression_l1_positive_rows",
                "features": len(FEATURE_COLUMNS),
            },
        ]
    )
    return LightGBMVariantResult(
        model=model,
        forecasts=forecasts,
        feature_importance=importance,
        training_summary=summary,
        evaluation_history={
            "occurrence": occurrence_history,
            "positive_quantity": quantity_history,
        },
    )


def fit_two_stage(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
) -> LightGBMVariantResult:
    """Refit both two-stage boosters independently at every origin."""
    config = GlobalLightGBMConfig() if config is None else config
    results = [
        _fit_two_stage_origin(origin_frames, config)
        for origin_frames in _origin_frame_sequence(frames)
    ]
    combined = _combine_origin_results(results, LightGBMVariantResult)
    assert isinstance(combined, LightGBMVariantResult)
    return combined


def make_weekly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily targets into a single seven-day total."""
    prepared = frame.copy()
    prepared["event_window_day"] = prepared["holiday_event_window"].ne("none").astype(int)
    prepared["abs_days_to_event"] = prepared["days_to_nearest_event"].abs()
    grouped = prepared.groupby(list(WEEK_KEYS), observed=True, sort=False)
    weekly = grouped.agg(
        weekly_actual=("actual", "sum"),
        target_days=("actual", "size"),
        week_iso_week=("iso_week", "first"),
        week_month=("month", "first"),
        event_window_days=("event_window_day", "sum"),
        min_abs_days_to_event=("abs_days_to_event", "min"),
        action_on_forecast_day=("action_on_forecast_day", "sum"),
        action_during_horizon=("action_during_horizon", "max"),
        days_since_last_action=("days_since_last_action", "first"),
        actions_last_28d=("actions_last_28d", "first"),
        mean_action_lift_in_sourcing_group=(
            "mean_action_lift_in_sourcing_group",
            "first",
        ),
        active_days_before_origin=("active_days_before_origin", "first"),
        demand_days_before_origin=("demand_days_before_origin", "first"),
        demand_day_ratio=("demand_day_ratio", "first"),
        days_since_last_positive_sale=("days_since_last_positive_sale", "first"),
        maturity_segment=("maturity_segment", "first"),
        lag_1=("lag_1", "first"),
        lag_7=("lag_7", "first"),
        lag_14=("lag_14", "first"),
        rolling_7_mean=("rolling_7_mean", "first"),
        rolling_28_mean=("rolling_28_mean", "first"),
        rolling_28_demand_rate=("rolling_28_demand_rate", "first"),
        rolling_28_positive_mean=("rolling_28_positive_mean", "first"),
        ADI=("ADI", "first"),
        CV2=("CV2", "first"),
        zero_share=("zero_share", "first"),
        product_cross_store_mean_28=("product_cross_store_mean_28", "first"),
        store_category_mean_28=("store_category_mean_28", "first"),
        same_weekday_mean_4_total=("same_weekday_mean_4", "sum"),
        same_weekday_mean_8_total=("same_weekday_mean_8", "sum"),
        product_weekday_profile_sum=("product_weekday_profile_value", "sum"),
        recent_mean_28_total_forecast=("recent_mean_28_forecast", "sum"),
        same_weekday_ma_4_total_forecast=("same_weekday_ma_4_forecast", "sum"),
        occurrence_positive_quantity_total_forecast=(
            "occurrence_positive_quantity_forecast",
            "sum",
        ),
    ).reset_index()
    weekly["actual"] = weekly.pop("weekly_actual")
    return weekly


def _fit_weekly_booster(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: GlobalLightGBMConfig,
) -> tuple[BaseLightGBMModel, dict[str, dict[str, list[float]]], int]:
    model, history, best_iteration = fit_lightgbm_model(
        training=training,
        validation=validation,
        labels=(training["actual"], validation["actual"]),
        evaluation=evaluation,
        params={
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "max_bin": 127,
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "metric": "None",
            "seed": config.random_state,
            "feature_fraction_seed": config.random_state,
            "bagging_seed": config.random_state,
            "num_threads": config.num_threads,
            "verbosity": -1,
        },
        feature_columns=WEEKLY_FEATURE_COLUMNS,
        categorical_features=WEEKLY_CATEGORICAL_FEATURES,
        config=config,
        feval=wape_feval,
    )
    return model, history, best_iteration


def _allocate_weekly_forecasts(
    evaluation: pd.DataFrame,
    weekly: pd.DataFrame,
    weekly_prediction: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = weekly.loc[:, list(WEEK_KEYS)].copy()
    totals["weekly_forecast"] = weekly_prediction
    daily = evaluation.copy()
    daily["local_allocation_signal"] = daily["same_weekday_mean_8"].fillna(0).clip(lower=0)
    daily["product_allocation_signal"] = (
        daily["product_weekday_profile_value"].fillna(0).clip(lower=0)
    )
    grouped = daily.groupby(list(WEEK_KEYS), observed=True)
    local_total = grouped["local_allocation_signal"].transform("sum")
    product_total = grouped["product_allocation_signal"].transform("sum")
    target_days = grouped["actual"].transform("size")
    daily["weekday_share"] = np.where(
        local_total.gt(0),
        daily["local_allocation_signal"] / local_total,
        np.where(
            product_total.gt(0),
            daily["product_allocation_signal"] / product_total,
            1.0 / target_days,
        ),
    )
    daily = daily.merge(totals, on=list(WEEK_KEYS), how="left", validate="m:1")
    daily["forecast"] = daily["weekly_forecast"] * daily["weekday_share"]
    allocated = (
        daily.groupby(list(WEEK_KEYS), observed=True)
        .agg(
            weekly_forecast=("weekly_forecast", "first"),
            allocated_forecast=("forecast", "sum"),
            weekday_share_sum=("weekday_share", "sum"),
        )
        .reset_index()
    )
    allocated["allocation_error"] = (
        allocated["allocated_forecast"] - allocated["weekly_forecast"]
    ).abs()
    return daily, allocated


def _fit_weekly_total_origin(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig,
) -> LightGBMVariantResult:
    """Fit weekly totals for one origin and allocate them to target days."""
    weekly_training = make_weekly_frame(frames.training)
    weekly_validation = make_weekly_frame(frames.validation)
    weekly_evaluation = make_weekly_frame(frames.evaluation)

    model, history, best_iteration = _fit_weekly_booster(
        weekly_training,
        weekly_validation,
        weekly_evaluation,
        config,
    )
    daily, audit = _allocate_weekly_forecasts(
        frames.evaluation,
        weekly_evaluation,
        model.predict(weekly_evaluation),
    )
    forecasts = daily.loc[:, [*FORECAST_ID_COLUMNS, *DIAGNOSTIC_COLUMNS]].copy()
    forecasts["model"] = WEEKLY_MODEL_NAME
    forecasts["forecast"] = daily["forecast"].to_numpy()
    forecasts["weekday_share"] = daily["weekday_share"].to_numpy()

    summary = pd.DataFrame(
        [
            {
                "model": WEEKLY_MODEL_NAME,
                **_training_summary_fields(frames),
                "fit_rows": len(weekly_training),
                "validation_rows": len(weekly_validation),
                "best_iteration": best_iteration,
                "objective": "weekly_tweedie_1.5",
                "features": len(WEEKLY_FEATURE_COLUMNS),
            }
        ]
    )
    return LightGBMVariantResult(
        model=model,
        forecasts=forecasts,
        feature_importance=lightgbm_feature_importance(model),
        training_summary=summary,
        evaluation_history=history,
        allocation_audit=audit,
    )


def fit_weekly_total(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
) -> LightGBMVariantResult:
    """Refit the weekly-total model independently at every origin."""
    config = GlobalLightGBMConfig() if config is None else config
    results = [
        _fit_weekly_total_origin(origin_frames, config)
        for origin_frames in _origin_frame_sequence(frames)
    ]
    combined = _combine_origin_results(results, LightGBMVariantResult)
    assert isinstance(combined, LightGBMVariantResult)
    return combined


def fit_all_lightgbm_models(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
) -> dict[str, GlobalLightGBMResult | LightGBMVariantResult]:
    """Refit all registered LightGBM variants at every prepared origin."""
    config = GlobalLightGBMConfig() if config is None else config
    return {
        MODEL_NAME: fit_global_lightgbm_frames(frames, config),
        TWEEDIE_MODEL_NAME: fit_tweedie_daily(frames, config),
        TWO_STAGE_MODEL_NAME: fit_two_stage(frames, config),
        WEEKLY_MODEL_NAME: fit_weekly_total(frames, config),
    }
