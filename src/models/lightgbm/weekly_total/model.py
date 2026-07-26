"""Weekly-total LightGBM model with daily weekday allocation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.lightgbm.base import (
    BaseLightGBMModel,
    LightGBMVariantResult,
    fit_lightgbm_model,
    lightgbm_feature_importance,
    wape_feval,
)
from src.models.lightgbm.features.builder import (
    DIAGNOSTIC_COLUMNS,
    FORECAST_ID_COLUMNS,
    NORMALIZED_TARGET_COLUMN,
    TARGET_SCALE_COLUMN,
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
)
from src.models.lightgbm.common import (
    combine_origin_results,
    origin_frame_sequence,
    training_summary_fields,
)
from src.models.lightgbm.config import LightGBMModelConfig

WEEKLY_MODEL_NAME = "global_lightgbm_weekly_total"


@dataclass(frozen=True)
class WeeklyTotalHyperparameters:
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l1: float = 0.1
    lambda_l2: float = 1.0
    max_bin: int = 127
    tweedie_variance_power: float = 1.5


@dataclass(frozen=True)
class WeeklyTotalConfig(LightGBMModelConfig):
    hyperparameters: WeeklyTotalHyperparameters = field(
        default_factory=WeeklyTotalHyperparameters
    )

    def parameters(self) -> dict[str, Any]:
        return {
            **self.hyperparameters.__dict__,
            "objective": "tweedie",
            "metric": "None",
            **self.seeded_parameters(),
        }

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
    "active_zero_demand_gap",
    "demand_days_last_7",
    "demand_days_last_28",
    "demand_days_last_60",
    "historical_p90_gap",
    "current_gap_over_historical_p90_gap",
    "same_weekday_lag_7",
    "same_weekday_lag_14",
    "rolling_7_mean",
    "rolling_28_mean",
    "rolling_28_demand_rate",
    "has_annual_history",
    "lag_364",
    "lag_371",
    "same_weekday_last_year_mean",
    "same_week_last_year_mean",
    "product_cross_store_same_weekday_last_year_mean",
    "same_event_offset_last_year_mean",
    "ADI",
    "CV2",
    "product_cross_store_mean_28",
    "store_category_mean_28",
    "same_weekday_mean_4_total",
    "same_weekday_mean_8_total",
    "product_weekday_profile_sum",
)

WEEKLY_CATEGORICAL_FEATURES = (
    "ARTIKEL_ID",
    "MARKT_ID",
    "sourcing_group",
    "category_id",
    "week_iso_week",
    "week_month",
)

WEEK_KEYS = ("ARTIKEL_ID", "MARKT_ID", "sourcing_group", "category_id", "origin")


def make_weekly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily targets into a single seven-day total."""
    prepared = frame.copy()
    active_only_signals = (
        "same_weekday_mean_4",
        "same_weekday_mean_8",
        "product_weekday_profile_value",
        "same_weekday_lag_7",
        "same_weekday_lag_14",
        "lag_364",
        "lag_371",
        "same_weekday_last_year_mean",
        "product_cross_store_same_weekday_last_year_mean",
        "same_event_offset_last_year_mean",
    )
    prepared.loc[~prepared["is_active"], list(active_only_signals)] = 0.0
    prepared["event_window_day"] = (
        prepared["holiday_event_window"].ne("none").astype(int)
    )
    prepared["abs_days_to_event"] = prepared["days_to_nearest_event"].abs()
    grouped = prepared.groupby(list(WEEK_KEYS), observed=True, sort=False)
    weekly = grouped.agg(
        weekly_actual=("actual", "sum"),
        normalized_actual=(NORMALIZED_TARGET_COLUMN, "sum"),
        target_mean=(TARGET_SCALE_COLUMN, "first"),
        target_days=("is_active", "sum"),
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
        active_zero_demand_gap=("active_zero_demand_gap", "first"),
        demand_days_last_7=("demand_days_last_7", "first"),
        demand_days_last_28=("demand_days_last_28", "first"),
        demand_days_last_60=("demand_days_last_60", "first"),
        historical_p90_gap=("historical_p90_gap", "first"),
        current_gap_over_historical_p90_gap=(
            "current_gap_over_historical_p90_gap",
            "first",
        ),
        same_weekday_lag_7=("same_weekday_lag_7", "sum"),
        same_weekday_lag_14=("same_weekday_lag_14", "sum"),
        rolling_7_mean=("rolling_7_mean", "first"),
        rolling_28_mean=("rolling_28_mean", "first"),
        rolling_28_demand_rate=("rolling_28_demand_rate", "first"),
        has_annual_history=("has_annual_history", "min"),
        lag_364=("lag_364", "sum"),
        lag_371=("lag_371", "sum"),
        same_weekday_last_year_mean=("same_weekday_last_year_mean", "sum"),
        same_week_last_year_mean=("same_week_last_year_mean", "first"),
        product_cross_store_same_weekday_last_year_mean=(
            "product_cross_store_same_weekday_last_year_mean",
            "sum",
        ),
        same_event_offset_last_year_mean=(
            "same_event_offset_last_year_mean",
            "sum",
        ),
        ADI=("ADI", "first"),
        CV2=("CV2", "first"),
        product_cross_store_mean_28=("product_cross_store_mean_28", "first"),
        store_category_mean_28=("store_category_mean_28", "first"),
        same_weekday_mean_4_total=("same_weekday_mean_4", "sum"),
        same_weekday_mean_8_total=("same_weekday_mean_8", "sum"),
        product_weekday_profile_sum=("product_weekday_profile_value", "sum"),
    ).reset_index()
    weekly["actual"] = weekly.pop("weekly_actual")
    return weekly


def _fit_booster(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: GlobalLightGBMConfig,
    params: dict[str, Any] | None = None,
) -> tuple[BaseLightGBMModel, dict[str, dict[str, list[float]]], int]:
    return fit_lightgbm_model(
        training=training,
        validation=validation,
        labels=(
            training[NORMALIZED_TARGET_COLUMN],
            validation[NORMALIZED_TARGET_COLUMN],
        ),
        evaluation=evaluation,
        params=(
            {
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
            }
            if params is None
            else params
        ),
        feature_columns=WEEKLY_FEATURE_COLUMNS,
        categorical_features=WEEKLY_CATEGORICAL_FEATURES,
        config=config,
        feval=wape_feval,
        prediction_scale_column=TARGET_SCALE_COLUMN,
    )


def _allocate_forecasts(
    evaluation: pd.DataFrame,
    weekly: pd.DataFrame,
    weekly_prediction: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = weekly.loc[:, list(WEEK_KEYS)].copy()
    totals["weekly_forecast"] = weekly_prediction
    daily = evaluation.copy()
    daily["local_allocation_signal"] = (
        daily["same_weekday_mean_8"].fillna(0).clip(lower=0)
        * daily["is_active"].astype(float)
    )
    daily["product_allocation_signal"] = (
        daily["product_weekday_profile_value"].fillna(0).clip(lower=0)
        * daily["is_active"].astype(float)
    )
    grouped = daily.groupby(list(WEEK_KEYS), observed=True)
    local_total = grouped["local_allocation_signal"].transform("sum")
    product_total = grouped["product_allocation_signal"].transform("sum")
    target_days = grouped["is_active"].transform("sum")
    daily["weekday_share"] = np.where(
        local_total.gt(0),
        daily["local_allocation_signal"] / local_total,
        np.where(
            product_total.gt(0),
            daily["product_allocation_signal"] / product_total,
            daily["is_active"].astype(float) / target_days,
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


def _fit_origin(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig,
    params: dict[str, Any] | None = None,
) -> LightGBMVariantResult:
    weekly_training = make_weekly_frame(frames.training)
    weekly_validation = make_weekly_frame(frames.validation)
    weekly_evaluation = make_weekly_frame(frames.evaluation)
    model, history, best_iteration = _fit_booster(
        weekly_training,
        weekly_validation,
        weekly_evaluation,
        config,
        params,
    )
    daily, audit = _allocate_forecasts(
        frames.evaluation,
        weekly_evaluation,
        model.predict(weekly_evaluation),
    )
    forecasts = daily.loc[:, [*FORECAST_ID_COLUMNS, *DIAGNOSTIC_COLUMNS]].copy()
    forecasts["model"] = WEEKLY_MODEL_NAME
    forecasts["forecast"] = daily["forecast"].to_numpy()
    forecasts["weekday_share"] = daily["weekday_share"].to_numpy()
    return LightGBMVariantResult(
        model=model,
        forecasts=forecasts,
        feature_importance=lightgbm_feature_importance(model),
        training_summary=pd.DataFrame(
            [
                {
                    "model": WEEKLY_MODEL_NAME,
                    **training_summary_fields(frames),
                    "fit_rows": len(weekly_training),
                    "validation_rows": len(weekly_validation),
                    "best_iteration": best_iteration,
                    "objective": "weekly_tweedie_1.5",
                    "features": len(WEEKLY_FEATURE_COLUMNS),
                }
            ]
        ),
        evaluation_history=history,
        allocation_audit=audit,
    )


def fit_weekly_total(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
    *,
    params: dict[str, Any] | None = None,
) -> LightGBMVariantResult:
    """Refit the weekly-total model for each four-week test block."""
    config = GlobalLightGBMConfig() if config is None else config
    results = [
        _fit_origin(item, config, params) for item in origin_frame_sequence(frames)
    ]
    combined = combine_origin_results(results, LightGBMVariantResult)
    assert isinstance(combined, LightGBMVariantResult)
    return combined


def fit(frames: object, config: WeeklyTotalConfig) -> object:
    """Fit weekly total with its independently owned parameters."""
    return fit_weekly_total(
        frames, config.execution_config(), params=config.parameters()
    )
