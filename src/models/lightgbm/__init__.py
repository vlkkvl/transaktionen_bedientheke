"""Shared-feature LightGBM forecasting models."""

from src.models.lightgbm.base import (
    BaseLightGBMModel,
    LightGBMVariantResult,
)
from src.models.lightgbm.common import GlobalLightGBMResult
from src.models.lightgbm.features.builder import (
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
    create_feature_tables,
    get_last_year_offset,
    historical_training_origins,
    iter_lightgbm_origin_windows,
    make_feature_frame,
    materialize_features_for_origins,
    prepare_global_lightgbm_frames,
)
from src.models.lightgbm.features.definition import (
    CATEGORICAL_FEATURES,
    DIAGNOSTIC_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    FORECAST_ID_COLUMNS,
    NORMALIZED_TARGET_COLUMN,
    TARGET_COLUMN,
    TARGET_SCALE_COLUMN,
)
from src.models.lightgbm.l2.model import (
    MODEL_NAME,
    GlobalLightGBMModel,
    fit_global_lightgbm_frames,
    run_global_lightgbm,
)
from src.models.lightgbm.registry import (
    LIGHTGBM_MODEL_LABELS,
    LIGHTGBM_MODELS,
    fit_all_lightgbm_models,
)
from src.models.lightgbm.tweedie.model import (
    TWEEDIE_MODEL_NAME,
    fit_tweedie_daily,
)
from src.models.lightgbm.two_stage.model import (
    TWO_STAGE_MODEL_NAME,
    fit_two_stage,
)
from src.models.lightgbm.weekly_total.model import (
    WEEKLY_CATEGORICAL_FEATURES,
    WEEKLY_FEATURE_COLUMNS,
    WEEKLY_MODEL_NAME,
    fit_weekly_total,
    make_weekly_frame,
)

__all__ = [
    "BaseLightGBMModel",
    "CATEGORICAL_FEATURES",
    "DIAGNOSTIC_COLUMNS",
    "FEATURE_COLUMNS",
    "FEATURE_DESCRIPTIONS",
    "FORECAST_ID_COLUMNS",
    "GlobalLightGBMConfig",
    "GlobalLightGBMFrames",
    "GlobalLightGBMModel",
    "GlobalLightGBMResult",
    "LIGHTGBM_MODEL_LABELS",
    "LIGHTGBM_MODELS",
    "LightGBMVariantResult",
    "MODEL_NAME",
    "NORMALIZED_TARGET_COLUMN",
    "TARGET_COLUMN",
    "TARGET_SCALE_COLUMN",
    "TWEEDIE_MODEL_NAME",
    "TWO_STAGE_MODEL_NAME",
    "WEEKLY_CATEGORICAL_FEATURES",
    "WEEKLY_FEATURE_COLUMNS",
    "WEEKLY_MODEL_NAME",
    "create_feature_tables",
    "fit_all_lightgbm_models",
    "fit_global_lightgbm_frames",
    "fit_tweedie_daily",
    "fit_two_stage",
    "fit_weekly_total",
    "get_last_year_offset",
    "historical_training_origins",
    "iter_lightgbm_origin_windows",
    "make_feature_frame",
    "make_weekly_frame",
    "materialize_features_for_origins",
    "prepare_global_lightgbm_frames",
    "run_global_lightgbm",
]
