"""Shared LightGBM training and prediction building blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd


Feval = Callable[[np.ndarray, lgb.Dataset], tuple[str, float, bool]]


@dataclass(frozen=True)
class BaseLightGBMModel:
    """Common predictor wrapper for all LightGBM-based models."""

    booster: lgb.Booster
    category_levels: dict[str, list[Any]]
    feature_columns: tuple[str, ...]
    categorical_features: tuple[str, ...]
    prediction_scale_column: str | None = None

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise KeyError(f"Feature frame is missing: {', '.join(missing)}")
        matrix = frame.loc[:, self.feature_columns].copy()
        for column in self.categorical_features:
            matrix[column] = matrix[column].astype(
                pd.CategoricalDtype(self.category_levels[column])
            )
        return matrix

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        num_iteration: int | None = None,
        clip_non_negative: bool = True,
    ) -> np.ndarray:
        """Predict with a stable feature transformation."""
        if num_iteration is None:
            num_iteration = self.booster.best_iteration
            if not num_iteration:
                num_iteration = None
        prediction = self.booster.predict(
            self.matrix(frame),
            num_iteration=num_iteration,
        )
        prediction = np.nan_to_num(prediction, nan=0.0)
        if self.prediction_scale_column is not None:
            if self.prediction_scale_column not in frame:
                raise KeyError(
                    "Feature frame is missing prediction scale column: "
                    f"{self.prediction_scale_column}"
                )
            prediction = prediction * frame[self.prediction_scale_column].to_numpy(
                dtype=float
            )
        return np.maximum(prediction, 0.0) if clip_non_negative else prediction


@dataclass(frozen=True)
class TwoStageLightGBMModel:
    """Occurrence × positive-quantity architecture."""

    occurrence: BaseLightGBMModel
    quantity: BaseLightGBMModel

    def predict_components(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        occurrence = self.occurrence.predict(frame, clip_non_negative=False)
        quantity = self.quantity.predict(frame)
        return (
            np.clip(occurrence, 0.0, 1.0),
            quantity,
            np.clip(occurrence, 0.0, 1.0) * quantity,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_components(frame)[2]


@dataclass(frozen=True)
class LightGBMVariantResult:
    """Unified output container for a LightGBM architecture."""

    model: Any
    forecasts: pd.DataFrame
    feature_importance: pd.DataFrame
    training_summary: pd.DataFrame
    evaluation_history: dict[str, Any]
    allocation_audit: pd.DataFrame | None = None
    validation_predictions: pd.DataFrame | None = None


def build_category_levels(
    categorical_features: tuple[str, ...], *frames: pd.DataFrame
) -> dict[str, list[Any]]:
    """Create a stable pandas category level map across all supplied frames."""
    levels: dict[str, list[Any]] = {}
    for column in categorical_features:
        values = (
            pd.concat([frame[column] for frame in frames], ignore_index=True)
            .dropna()
            .drop_duplicates()
        )
        levels[column] = values.sort_values().tolist()
    return levels


def build_matrix(
    frame: pd.DataFrame,
    category_levels: dict[str, list[Any]],
    feature_columns: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> pd.DataFrame:
    """Apply deterministic categorical encoding to matrix columns."""
    matrix = frame.loc[:, feature_columns].copy()
    for column in categorical_features:
        matrix[column] = matrix[column].astype(
            pd.CategoricalDtype(category_levels[column])
        )
    return matrix


def wape_feval(prediction: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    """WAPE custom metric used across LightGBM implementations."""
    actual = dataset.get_label()
    denominator = float(np.sum(actual))
    value = (
        float(np.sum(np.abs(prediction - actual)) / denominator)
        if denominator > 0
        else float("nan")
    )
    return "wape", value, False


def lightgbm_feature_importance(model: BaseLightGBMModel) -> pd.DataFrame:
    """Feature importance with normalized gain share."""
    importance = pd.DataFrame(
        {
            "feature": model.booster.feature_name(),
            "gain": model.booster.feature_importance(importance_type="gain"),
            "splits": model.booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False, ignore_index=True)
    total_gain = float(importance["gain"].sum())
    importance["gain_share"] = importance["gain"] / total_gain if total_gain > 0 else 0.0
    return importance


def fit_lightgbm_model(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    labels: tuple[pd.Series, pd.Series],
    evaluation: pd.DataFrame,
    params: dict[str, Any],
    feature_columns: tuple[str, ...],
    categorical_features: tuple[str, ...],
    config: Any,
    *,
    feval: Feval | None = None,
    model_class: type[BaseLightGBMModel] = BaseLightGBMModel,
    prediction_scale_column: str | None = None,
) -> tuple[
    BaseLightGBMModel,
    dict[str, dict[str, list[float]]],
    int,
    np.ndarray,
]:
    """Fit a model and refit a final booster on train+validation."""
    levels = build_category_levels(categorical_features, training, validation, evaluation)
    train_x = build_matrix(training, levels, feature_columns, categorical_features)
    validation_x = build_matrix(validation, levels, feature_columns, categorical_features)

    history: dict[str, dict[str, list[float]]] = {}
    booster = lgb.train(
        params,
        lgb.Dataset(
            train_x,
            label=labels[0],
            categorical_feature=list(categorical_features),
            free_raw_data=False,
        ),
        num_boost_round=int(config.num_boost_round),
        valid_sets=[
            lgb.Dataset(
                validation_x,
                label=labels[1],
                categorical_feature=list(categorical_features),
                free_raw_data=False,
            )
        ],
        valid_names=["validation"],
        feval=feval,
        callbacks=[
            lgb.early_stopping(int(config.early_stopping_rounds), verbose=False),
            lgb.record_evaluation(history),
        ],
    )
    best_iteration = int(booster.best_iteration or int(config.num_boost_round))
    validation_prediction = booster.predict(
        validation_x,
        num_iteration=best_iteration,
    )

    all_training = pd.concat([training, validation], ignore_index=True)
    all_labels = pd.concat([labels[0], labels[1]], ignore_index=True)
    final = lgb.train(
        params,
        lgb.Dataset(
            build_matrix(all_training, levels, feature_columns, categorical_features),
            label=all_labels,
            categorical_feature=list(categorical_features),
            free_raw_data=False,
        ),
        num_boost_round=best_iteration,
    )
    return (
        model_class(
            final,
            levels,
            feature_columns,
            categorical_features,
            prediction_scale_column,
        ),
        history,
        best_iteration,
        np.asarray(validation_prediction),
    )
