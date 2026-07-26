"""Two-stage occurrence and positive-quantity LightGBM model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.lightgbm.base import (
    BaseLightGBMModel,
    LightGBMVariantResult,
    TwoStageLightGBMModel,
    fit_lightgbm_model,
    lightgbm_feature_importance,
)
from src.models.lightgbm.features.builder import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NORMALIZED_TARGET_COLUMN,
    TARGET_SCALE_COLUMN,
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
)
from src.models.lightgbm.common import (
    base_model_params,
    combine_origin_results,
    daily_forecasts,
    origin_frame_sequence,
    training_summary_fields,
)
from src.models.lightgbm.config import LightGBMModelConfig

TWO_STAGE_MODEL_NAME = "global_lightgbm_two_stage"


@dataclass(frozen=True)
class OccurrenceHyperparameters:
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l1: float = 0.1
    lambda_l2: float = 1.0
    max_bin: int = 127


@dataclass(frozen=True)
class QuantityHyperparameters:
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l1: float = 0.1
    lambda_l2: float = 1.0
    max_bin: int = 127


@dataclass(frozen=True)
class TwoStageConfig(LightGBMModelConfig):
    occurrence: OccurrenceHyperparameters = field(
        default_factory=OccurrenceHyperparameters
    )
    quantity: QuantityHyperparameters = field(
        default_factory=QuantityHyperparameters
    )

    def occurrence_parameters(self) -> dict[str, Any]:
        return {
            **self.occurrence.__dict__,
            "objective": "binary",
            "metric": "binary_logloss",
            **self.seeded_parameters(),
        }

    def quantity_parameters(self) -> dict[str, Any]:
        return {
            **self.quantity.__dict__,
            "objective": "regression_l1",
            "metric": "l1",
            **self.seeded_parameters(),
        }


def _train_stage(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: GlobalLightGBMConfig,
    *,
    objective: str,
    metric: str,
    labels: tuple[pd.Series, pd.Series],
    restore_target_scale: bool = False,
    params: dict[str, Any] | None = None,
) -> tuple[BaseLightGBMModel, dict[str, dict[str, list[float]]], int]:
    return fit_lightgbm_model(
        training=training,
        validation=validation,
        labels=labels,
        evaluation=evaluation,
        params=(
            {
                **base_model_params(config),
                "objective": objective,
                "metric": metric,
            }
            if params is None
            else params
        ),
        feature_columns=FEATURE_COLUMNS,
        categorical_features=CATEGORICAL_FEATURES,
        config=config,
        prediction_scale_column=(
            TARGET_SCALE_COLUMN if restore_target_scale else None
        ),
    )


def _fit_origin(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig,
    occurrence_params: dict[str, Any] | None = None,
    quantity_params: dict[str, Any] | None = None,
) -> LightGBMVariantResult:
    occurrence, occurrence_history, occurrence_iteration = _train_stage(
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
        params=occurrence_params,
    )
    positive_training = frames.training.loc[frames.training["actual"].gt(0)].copy()
    positive_validation = frames.validation.loc[
        frames.validation["actual"].gt(0)
    ].copy()
    if positive_training.empty or positive_validation.empty:
        raise RuntimeError(
            "Positive-only train/validation splits are empty; "
            "two-stage model cannot be fit."
        )

    quantity, quantity_history, quantity_iteration = _train_stage(
        positive_training,
        positive_validation,
        frames.evaluation,
        config,
        objective="regression_l1",
        metric="l1",
        labels=(
            positive_training[NORMALIZED_TARGET_COLUMN],
            positive_validation[NORMALIZED_TARGET_COLUMN],
        ),
        restore_target_scale=True,
        params=quantity_params,
    )
    model = TwoStageLightGBMModel(occurrence=occurrence, quantity=quantity)
    occurrence_prediction, quantity_prediction, prediction = model.predict_components(
        frames.evaluation
    )
    forecasts = daily_forecasts(frames, TWO_STAGE_MODEL_NAME, prediction)
    forecasts["occurrence_probability"] = occurrence_prediction
    forecasts["positive_quantity_forecast"] = quantity_prediction

    occurrence_importance = lightgbm_feature_importance(occurrence)
    occurrence_importance.insert(0, "stage", "occurrence")
    quantity_importance = lightgbm_feature_importance(quantity)
    quantity_importance.insert(0, "stage", "positive_quantity")
    importance = pd.concat(
        [occurrence_importance, quantity_importance], ignore_index=True
    )
    common_summary = training_summary_fields(frames)
    summary = pd.DataFrame(
        [
            {
                "model": TWO_STAGE_MODEL_NAME,
                "stage": "occurrence",
                **common_summary,
                "best_iteration": occurrence_iteration,
                "objective": "binary_logloss",
                "features": len(FEATURE_COLUMNS),
            },
            {
                "model": TWO_STAGE_MODEL_NAME,
                "stage": "positive_quantity",
                **common_summary,
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
    *,
    occurrence_params: dict[str, Any] | None = None,
    quantity_params: dict[str, Any] | None = None,
) -> LightGBMVariantResult:
    """Refit both two-stage boosters for each four-week test block."""
    config = GlobalLightGBMConfig() if config is None else config
    results = [
        _fit_origin(item, config, occurrence_params, quantity_params)
        for item in origin_frame_sequence(frames)
    ]
    combined = combine_origin_results(results, LightGBMVariantResult)
    assert isinstance(combined, LightGBMVariantResult)
    return combined


def fit(frames: object, config: TwoStageConfig) -> object:
    """Fit both stages with their independently owned parameter sets."""
    return fit_two_stage(
        frames,
        config.execution_config(),
        occurrence_params=config.occurrence_parameters(),
        quantity_params=config.quantity_parameters(),
    )
