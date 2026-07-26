"""Feature-set definitions and fitting for the daily Tweedie ablation."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.lightgbm.base import (
    fit_lightgbm_model,
    lightgbm_feature_importance,
    wape_feval,
)
from src.models.lightgbm.features.builder import (
    CATEGORICAL_FEATURES,
    DIAGNOSTIC_COLUMNS,
    FEATURE_COLUMNS,
    FORECAST_ID_COLUMNS,
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
)
from src.models.lightgbm.common import base_model_params


BASE_GROUPS = OrderedDict(
    {
        "IDs": ("ARTIKEL_ID", "MARKT_ID", "sourcing_group", "category_id"),
        "Calendar and forecast position": (
            "target_weekday",
            "iso_week",
            "month",
            "days_to_nearest_event",
            "holiday_event_window",
            "event_name",
        ),
        "Local demand history": (
            "same_weekday_lag_7",
            "same_weekday_lag_14",
            "rolling_7_mean",
            "rolling_28_mean",
            "rolling_28_demand_rate",
            "same_weekday_mean_4",
            "same_weekday_mean_8",
        ),
    }
)

CANDIDATE_GROUPS = OrderedDict(
    {
        "Known action schedule": (
            "action_on_forecast_day",
            "action_during_horizon",
        ),
        "Historical action behavior": (
            "days_since_last_action",
            "actions_last_28d",
            "mean_action_lift_in_sourcing_group",
        ),
        "Maturity and demand gaps": (
            "active_days_before_origin",
            "demand_days_before_origin",
            "demand_day_ratio",
            "active_zero_demand_gap",
            "demand_days_last_7",
            "demand_days_last_28",
            "demand_days_last_60",
            "historical_p90_gap",
            "current_gap_over_historical_p90_gap",
        ),
        "Annual demand history": (
            "has_annual_history",
            "lag_364",
            "lag_371",
            "same_weekday_last_year_mean",
            "same_week_last_year_mean",
            "product_cross_store_same_weekday_last_year_mean",
            "same_event_offset_last_year_mean",
        ),
        "Demand regime": ("ADI", "CV2"),
        "Cross-sectional context": (
            "product_cross_store_mean_28",
            "product_weekday_profile_value",
            "store_category_mean_28",
        ),
    }
)


@dataclass(frozen=True)
class LightGBMAblationResult:
    """Combined forecasts and diagnostics for all ablation specifications."""

    forecasts: pd.DataFrame
    training_summary: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_design: pd.DataFrame


def build_feature_sets() -> OrderedDict[str, tuple[str, ...]]:
    """Build the base, one-group-at-a-time, and all-feature specifications."""
    candidate_groups = OrderedDict(CANDIDATE_GROUPS)
    declared = {
        feature
        for features in [*BASE_GROUPS.values(), *candidate_groups.values()]
        for feature in features
    }
    unknown = declared.difference(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Feature groups contain unknown columns: {sorted(unknown)}")
    operational = tuple(
        feature for feature in FEATURE_COLUMNS if feature not in declared
    )
    if operational:
        candidate_groups["Operational supply signals"] = operational

    base = tuple(
        feature
        for feature in FEATURE_COLUMNS
        if any(feature in group for group in BASE_GROUPS.values())
    )
    feature_sets = OrderedDict({"Base": base})
    for group_name, group_features in candidate_groups.items():
        allowed = set(base).union(group_features)
        feature_sets[f"Base + {group_name}"] = tuple(
            feature for feature in FEATURE_COLUMNS if feature in allowed
        )
    feature_sets["All features"] = tuple(FEATURE_COLUMNS)
    return feature_sets


def _model_id(specification: str) -> str:
    suffix = specification.lower().replace(" + ", "_").replace(" ", "_")
    return f"tweedie_ablation_{suffix}"


def _fit_specification(
    specification: str,
    feature_columns: tuple[str, ...],
    origin_frames: Sequence[GlobalLightGBMFrames],
    config: GlobalLightGBMConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_id = _model_id(specification)
    categorical_features = tuple(
        feature for feature in CATEGORICAL_FEATURES if feature in feature_columns
    )
    forecast_parts = []
    summary_rows = []
    importance_parts = []
    for frames in origin_frames:
        model, _, best_iteration = fit_lightgbm_model(
            training=frames.training,
            validation=frames.validation,
            labels=(frames.training["actual"], frames.validation["actual"]),
            evaluation=frames.evaluation,
            params={
                **base_model_params(config),
                "objective": "tweedie",
                "tweedie_variance_power": 1.5,
                "metric": "None",
            },
            feature_columns=feature_columns,
            categorical_features=categorical_features,
            config=config,
            feval=wape_feval,
        )
        evaluation = frames.evaluation
        forecasts = evaluation.loc[
            :, [*FORECAST_ID_COLUMNS, *DIAGNOSTIC_COLUMNS]
        ].copy()
        forecasts["model"] = model_id
        forecasts["forecast"] = np.where(
            evaluation["is_active"], model.predict(evaluation), 0.0
        )
        forecast_parts.append(forecasts)

        evaluation_origin = pd.Timestamp(evaluation["origin"].iloc[0])
        summary_rows.append(
            {
                "model": model_id,
                "specification": specification,
                "evaluation_origin": evaluation_origin,
                "fit_rows": len(frames.training),
                "validation_rows": len(frames.validation),
                "evaluation_rows": len(evaluation),
                "best_iteration": best_iteration,
                "features": len(feature_columns),
            }
        )
        importance = lightgbm_feature_importance(model)
        importance["model"] = model_id
        importance["specification"] = specification
        importance["evaluation_origin"] = evaluation_origin
        importance_parts.append(importance)
    return (
        pd.concat(forecast_parts, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.concat(importance_parts, ignore_index=True),
    )


def fit_tweedie_feature_ablation(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
    feature_sets: Mapping[str, tuple[str, ...]] | None = None,
) -> LightGBMAblationResult:
    """Fit every feature specification independently at each origin."""
    config = GlobalLightGBMConfig() if config is None else config
    specifications = OrderedDict(
        build_feature_sets() if feature_sets is None else feature_sets
    )
    origin_frames = tuple(frames.origin_frames or (frames,))
    fitted = [
        _fit_specification(name, columns, origin_frames, config)
        for name, columns in specifications.items()
    ]
    feature_design = pd.DataFrame(
        [
            {
                "model": _model_id(name),
                "specification": name,
                "n_features": len(columns),
                "features": ", ".join(columns),
            }
            for name, columns in specifications.items()
        ]
    )
    return LightGBMAblationResult(
        forecasts=pd.concat([item[0] for item in fitted], ignore_index=True),
        training_summary=pd.concat([item[1] for item in fitted], ignore_index=True),
        feature_importance=pd.concat([item[2] for item in fitted], ignore_index=True),
        feature_design=feature_design,
    )
