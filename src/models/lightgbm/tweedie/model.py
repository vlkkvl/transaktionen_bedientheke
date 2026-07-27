"""Direct daily LightGBM model with a Tweedie objective."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.models.lightgbm.base import (
    LightGBMVariantResult,
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

TWEEDIE_MODEL_NAME = "global_lightgbm_tweedie_daily"


@dataclass(frozen=True)
class TweedieHyperparameters:
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l1: float = 0.1
    lambda_l2: float = 1.0
    max_bin: int = 127
    tweedie_variance_power: float = 1.5


@dataclass(frozen=True)
class TweedieConfig(LightGBMModelConfig):
    hyperparameters: TweedieHyperparameters = field(
        default_factory=TweedieHyperparameters
    )

    def parameters(self) -> dict[str, Any]:
        return {
            **self.hyperparameters.__dict__,
            "objective": "tweedie",
            "metric": "tweedie",
            **self.seeded_parameters(),
        }


def _fit_origin(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig,
    params: dict[str, Any] | None = None,
) -> LightGBMVariantResult:
    model, history, best_iteration, _ = fit_lightgbm_model(
        training=frames.training,
        validation=frames.validation,
        labels=(
            frames.training[NORMALIZED_TARGET_COLUMN],
            frames.validation[NORMALIZED_TARGET_COLUMN],
        ),
        evaluation=frames.evaluation,
        params=(
            {
                **base_model_params(config),
                "objective": "tweedie",
                "tweedie_variance_power": 1.5,
                "metric": "tweedie",
            }
            if params is None
            else params
        ),
        feature_columns=FEATURE_COLUMNS,
        categorical_features=CATEGORICAL_FEATURES,
        config=config,
        prediction_scale_column=TARGET_SCALE_COLUMN,
    )
    return LightGBMVariantResult(
        model=model,
        forecasts=daily_forecasts(
            frames, TWEEDIE_MODEL_NAME, model.predict(frames.evaluation)
        ),
        feature_importance=lightgbm_feature_importance(model),
        training_summary=pd.DataFrame(
            [
                {
                    "model": TWEEDIE_MODEL_NAME,
                    **training_summary_fields(frames),
                    "best_iteration": best_iteration,
                    "objective": "tweedie",
                    "early_stopping_metric": "tweedie_deviance",
                    "features": len(FEATURE_COLUMNS),
                }
            ]
        ),
        evaluation_history=history,
    )


def fit_tweedie_daily(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
    *,
    params: dict[str, Any] | None = None,
) -> LightGBMVariantResult:
    """Refit the daily Tweedie model for each four-week test block."""
    config = GlobalLightGBMConfig() if config is None else config
    results = [
        _fit_origin(item, config, params) for item in origin_frame_sequence(frames)
    ]
    combined = combine_origin_results(results, LightGBMVariantResult)
    assert isinstance(combined, LightGBMVariantResult)
    return combined


def fit(frames: object, config: TweedieConfig) -> object:
    """Fit Tweedie with its independently owned parameters."""
    return fit_tweedie_daily(
        frames, config.execution_config(), params=config.parameters()
    )
