"""Two-stage model with pinball-loss quantile quantity boosters.

Same architecture as the two-stage model (binary occurrence stage, quantity
stage trained on positive rows only, stage-specific feature routing), but the
quantity stage is fit once per requested quantile level with LightGBM's
``quantile`` objective instead of once with ``gamma``. The persisted forecasts
carry the occurrence probability and one positive-quantity column per level,
so predictive intervals can be evaluated without refitting.
"""
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
)
from src.models.lightgbm.features.builder import (
    CATEGORICAL_FEATURES,
    DIAGNOSTIC_COLUMNS,
    FEATURE_COLUMNS,
    FORECAST_ID_COLUMNS,
    NORMALIZED_TARGET_COLUMN,
    TARGET_SCALE_COLUMN,
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
)
from src.models.lightgbm.common import (
    combine_origin_results,
    daily_forecasts,
    origin_frame_sequence,
    training_summary_fields,
)
from src.models.lightgbm.config import LightGBMModelConfig

TWO_STAGE_QUANTILE_MODEL_NAME = "global_lightgbm_two_stage_quantile"
QUANTILE_LEVELS = (0.1, 0.5, 0.9)

_OCCURRENCE_EXCLUDED = {
    "event_lift_pooled_quantity",
    "event_lift_pooled_total",
    "event_position_lift_quantity",
    "event_position_lift_total",
}
_QUANTITY_EXCLUDED = {
    "event_lift_pooled_occurrence",
    "event_lift_pooled_total",
    "event_position_lift_occurrence",
    "event_position_lift_total",
}


def quantile_column_name(level: float) -> str:
    """Stable forecast column name for one quantile level."""
    return f"positive_quantity_p{int(round(level * 100)):02d}"


def _validate_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    levels = tuple(levels)
    if not levels:
        raise ValueError("At least one quantile level is required")
    if any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("Quantile levels must lie strictly between 0 and 1")
    if list(levels) != sorted(set(levels)):
        raise ValueError("Quantile levels must be strictly increasing")
    return levels


@dataclass(frozen=True)
class QuantileOccurrenceHyperparameters:
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
class QuantileQuantityHyperparameters:
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
class TwoStageQuantileConfig(LightGBMModelConfig):
    occurrence: QuantileOccurrenceHyperparameters = field(
        default_factory=QuantileOccurrenceHyperparameters
    )
    quantity: QuantileQuantityHyperparameters = field(
        default_factory=QuantileQuantityHyperparameters
    )
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS

    def occurrence_parameters(self) -> dict[str, Any]:
        return {
            **self.occurrence.__dict__,
            "objective": "binary",
            "metric": "binary_logloss",
            **self.seeded_parameters(),
        }

    def quantity_parameters(self, level: float) -> dict[str, Any]:
        return {
            **self.quantity.__dict__,
            "objective": "quantile",
            "metric": "quantile",
            "alpha": float(level),
            **self.seeded_parameters(),
        }


@dataclass(frozen=True)
class TwoStageQuantileLightGBMModel:
    """Occurrence classifier × per-level positive-quantity quantile boosters."""

    occurrence: BaseLightGBMModel
    quantity_by_level: tuple[tuple[float, BaseLightGBMModel], ...]

    def predict_components(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (occurrence probability, level × row quantile matrix).

        The per-level boosters are fit independently, so their raw
        predictions may cross; rows are monotonically rearranged (sorted)
        so the k-th output row is a valid k-th quantile estimate.
        """
        occurrence = np.clip(
            self.occurrence.predict(frame, clip_non_negative=False), 0.0, 1.0
        )
        raw = np.vstack(
            [model.predict(frame) for _, model in self.quantity_by_level]
        )
        return occurrence, np.sort(raw, axis=0)

    def _median_index(self) -> int:
        levels = [level for level, _ in self.quantity_by_level]
        return int(np.argmin(np.abs(np.asarray(levels) - 0.5)))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        occurrence, quantiles = self.predict_components(frame)
        return occurrence * quantiles[self._median_index()]


def _train_stage(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: GlobalLightGBMConfig,
    *,
    params: dict[str, Any],
    labels: tuple[pd.Series, pd.Series],
    feature_columns: tuple[str, ...],
    restore_target_scale: bool = False,
) -> tuple[
    BaseLightGBMModel,
    dict[str, dict[str, list[float]]],
    int,
    np.ndarray,
]:
    return fit_lightgbm_model(
        training=training,
        validation=validation,
        labels=labels,
        evaluation=evaluation,
        params=params,
        feature_columns=feature_columns,
        categorical_features=tuple(
            feature
            for feature in CATEGORICAL_FEATURES
            if feature in feature_columns
        ),
        config=config,
        prediction_scale_column=(
            TARGET_SCALE_COLUMN if restore_target_scale else None
        ),
    )


def _fit_origin(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig,
    occurrence_params: dict[str, Any],
    quantity_params_by_level: tuple[tuple[float, dict[str, Any]], ...],
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> LightGBMVariantResult:
    occurrence_features = tuple(
        feature for feature in feature_columns
        if feature not in _OCCURRENCE_EXCLUDED
    )
    quantity_features = tuple(
        feature for feature in feature_columns
        if feature not in _QUANTITY_EXCLUDED
    )
    (
        occurrence,
        occurrence_history,
        occurrence_iteration,
        validation_occurrence_probability,
    ) = _train_stage(
        frames.training,
        frames.validation,
        frames.evaluation,
        config,
        params=occurrence_params,
        labels=(
            frames.training["actual"].gt(0).astype(np.int8),
            frames.validation["actual"].gt(0).astype(np.int8),
        ),
        feature_columns=occurrence_features,
    )
    positive_training = frames.training.loc[frames.training["actual"].gt(0)].copy()
    positive_validation = frames.validation.loc[
        frames.validation["actual"].gt(0)
    ].copy()
    if positive_training.empty or positive_validation.empty:
        raise RuntimeError(
            "Positive-only train/validation splits are empty; "
            "two-stage quantile model cannot be fit."
        )

    common_summary = training_summary_fields(frames)
    evaluation_history: dict[str, Any] = {"occurrence": occurrence_history}
    importance_frames = [lightgbm_feature_importance(occurrence)]
    importance_frames[0].insert(0, "stage", "occurrence")
    summary_rows: list[dict[str, Any]] = [
        {
            "model": TWO_STAGE_QUANTILE_MODEL_NAME,
            "stage": "occurrence",
            **common_summary,
            "best_iteration": occurrence_iteration,
            "objective": "binary",
            "early_stopping_metric": "binary_logloss",
            "features": len(occurrence_features),
        }
    ]
    quantity_by_level: list[tuple[float, BaseLightGBMModel]] = []
    for level, params in quantity_params_by_level:
        stage_name = quantile_column_name(level)
        quantity, quantity_history, quantity_iteration, _ = _train_stage(
            positive_training,
            positive_validation,
            frames.evaluation,
            config,
            params=params,
            labels=(
                positive_training[NORMALIZED_TARGET_COLUMN],
                positive_validation[NORMALIZED_TARGET_COLUMN],
            ),
            feature_columns=quantity_features,
            restore_target_scale=True,
        )
        quantity_by_level.append((level, quantity))
        evaluation_history[stage_name] = quantity_history
        stage_importance = lightgbm_feature_importance(quantity)
        stage_importance.insert(0, "stage", stage_name)
        importance_frames.append(stage_importance)
        summary_rows.append(
            {
                "model": TWO_STAGE_QUANTILE_MODEL_NAME,
                "stage": stage_name,
                **common_summary,
                "fit_rows": len(positive_training),
                "validation_rows": len(positive_validation),
                "best_iteration": quantity_iteration,
                "objective": "quantile",
                "early_stopping_metric": "quantile",
                "features": len(quantity_features),
            }
        )

    model = TwoStageQuantileLightGBMModel(
        occurrence=occurrence, quantity_by_level=tuple(quantity_by_level)
    )
    occurrence_prediction, quantile_predictions = model.predict_components(
        frames.evaluation
    )
    prediction = occurrence_prediction * quantile_predictions[model._median_index()]
    forecasts = daily_forecasts(frames, TWO_STAGE_QUANTILE_MODEL_NAME, prediction)
    forecasts["occurrence_probability"] = occurrence_prediction
    for position, (level, _) in enumerate(quantity_by_level):
        forecasts[quantile_column_name(level)] = quantile_predictions[position]

    validation_predictions = frames.validation.loc[
        :, [*FORECAST_ID_COLUMNS, *DIAGNOSTIC_COLUMNS]
    ].copy()
    validation_predictions["model"] = TWO_STAGE_QUANTILE_MODEL_NAME
    validation_predictions["evaluation_origin"] = common_summary[
        "evaluation_origin"
    ]
    validation_predictions["best_iteration"] = occurrence_iteration
    validation_predictions["occurrence_probability"] = np.clip(
        validation_occurrence_probability,
        0.0,
        1.0,
    )
    return LightGBMVariantResult(
        model=model,
        forecasts=forecasts,
        feature_importance=pd.concat(importance_frames, ignore_index=True),
        training_summary=pd.DataFrame(summary_rows),
        evaluation_history=evaluation_history,
        validation_predictions=validation_predictions,
    )


def fit_two_stage_quantile(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
    *,
    occurrence_params: dict[str, Any] | None = None,
    quantity_params_by_level: tuple[tuple[float, dict[str, Any]], ...] | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> LightGBMVariantResult:
    """Refit the occurrence stage plus one quantile booster per level."""
    config = GlobalLightGBMConfig() if config is None else config
    feature_columns = tuple(feature_columns)
    if not feature_columns:
        raise ValueError("At least one feature is required")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature columns must be unique")
    unknown_features = sorted(set(feature_columns) - set(FEATURE_COLUMNS))
    if unknown_features:
        raise KeyError(
            "Unknown two-stage quantile features: " + ", ".join(unknown_features)
        )
    if occurrence_params is None or quantity_params_by_level is None:
        defaults = TwoStageQuantileConfig()
        if occurrence_params is None:
            occurrence_params = defaults.occurrence_parameters()
        if quantity_params_by_level is None:
            quantity_params_by_level = tuple(
                (level, defaults.quantity_parameters(level))
                for level in defaults.quantile_levels
            )
    _validate_levels(tuple(level for level, _ in quantity_params_by_level))
    results = [
        _fit_origin(
            item,
            config,
            occurrence_params,
            tuple(quantity_params_by_level),
            feature_columns,
        )
        for item in origin_frame_sequence(frames)
    ]
    combined = combine_origin_results(results, LightGBMVariantResult)
    assert isinstance(combined, LightGBMVariantResult)
    return combined


def fit(frames: object, config: TwoStageQuantileConfig) -> object:
    """Fit the occurrence stage and every configured quantile level."""
    levels = _validate_levels(config.quantile_levels)
    return fit_two_stage_quantile(
        frames,
        config.execution_config(),
        occurrence_params=config.occurrence_parameters(),
        quantity_params_by_level=tuple(
            (level, config.quantity_parameters(level)) for level in levels
        ),
    )
