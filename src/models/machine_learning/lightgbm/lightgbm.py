"""Compatibility shim for global LightGBM APIs.

Feature engineering now lives in :mod:`lightgbm_features`, while all model
implementations live in :mod:`lightgbm_variants`. This module re-exports the
stable objects that existing notebooks/tests import.
"""
from __future__ import annotations

from src.models.machine_learning.lightgbm.lightgbm_features import (
    CATEGORICAL_FEATURES,
    DIAGNOSTIC_COLUMNS,
    FEATURE_COLUMNS,
    FORECAST_ID_COLUMNS,
    NORMALIZED_TARGET_COLUMN,
    TARGET_COLUMN,
    TARGET_SCALE_COLUMN,
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
    create_feature_tables,
    historical_training_origins,
    make_feature_frame,
    prepare_global_lightgbm_frames,
)
from src.models.machine_learning.lightgbm.lightgbm_variants import (
    MODEL_NAME,
    GlobalLightGBMModel,
    GlobalLightGBMResult,
    fit_global_lightgbm_frames,
    run_global_lightgbm,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "DIAGNOSTIC_COLUMNS",
    "FEATURE_COLUMNS",
    "FORECAST_ID_COLUMNS",
    "MODEL_NAME",
    "NORMALIZED_TARGET_COLUMN",
    "TARGET_COLUMN",
    "TARGET_SCALE_COLUMN",
    "GlobalLightGBMConfig",
    "GlobalLightGBMFrames",
    "GlobalLightGBMModel",
    "GlobalLightGBMResult",
    "create_feature_tables",
    "fit_global_lightgbm_frames",
    "historical_training_origins",
    "make_feature_frame",
    "prepare_global_lightgbm_frames",
    "run_global_lightgbm",
]
