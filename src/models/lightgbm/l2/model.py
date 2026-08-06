"""Direct daily LightGBM model with an L2 objective."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable

import duckdb
import pandas as pd

from src.models.benchmark.config import BenchmarkDesign
from src.models.lightgbm.base import (
    BaseLightGBMModel,
    fit_lightgbm_model,
    lightgbm_feature_importance,
)
from src.models.lightgbm.features.builder import (
    CATEGORICAL_FEATURES,
    DEFAULT_FEATURES_PATH,
    DIRECT_FEATURE_COLUMNS,
    NORMALIZED_TARGET_COLUMN,
    TARGET_SCALE_COLUMN,
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
    prepare_global_lightgbm_frames,
)
from src.models.lightgbm.common import (
    GlobalLightGBMResult,
    base_model_params,
    combine_origin_results,
    daily_forecasts,
    origin_frame_sequence,
    training_summary_fields,
)
from src.models.lightgbm.config import LightGBMModelConfig

MODEL_NAME = "global_lightgbm"


@dataclass(frozen=True)
class L2Hyperparameters:
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
class L2Config(LightGBMModelConfig):
    hyperparameters: L2Hyperparameters = field(default_factory=L2Hyperparameters)

    def parameters(self) -> dict[str, Any]:
        return {
            **self.hyperparameters.__dict__,
            "objective": "regression_l2",
            "metric": "rmse",
            **self.seeded_parameters(),
        }


def _fit_origin(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig,
    params: dict[str, Any] | None = None,
) -> GlobalLightGBMResult:
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
                "objective": "regression_l2",
                "metric": "rmse",
            }
            if params is None
            else params
        ),
        feature_columns=DIRECT_FEATURE_COLUMNS,
        categorical_features=CATEGORICAL_FEATURES,
        config=config,
        prediction_scale_column=TARGET_SCALE_COLUMN,
    )
    return GlobalLightGBMResult(
        model=model,
        forecasts=daily_forecasts(
            frames, MODEL_NAME, model.predict(frames.evaluation)
        ),
        feature_importance=lightgbm_feature_importance(model),
        training_summary=pd.DataFrame(
            [
                {
                    "model": MODEL_NAME,
                    **training_summary_fields(frames),
                    "best_iteration": best_iteration,
                    "objective": "regression_l2",
                    "early_stopping_metric": "rmse",
                    "features": len(DIRECT_FEATURE_COLUMNS),
                }
            ]
        ),
        evaluation_history=history,
    )


def fit_global_lightgbm_frames(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
    *,
    params: dict[str, Any] | None = None,
) -> GlobalLightGBMResult:
    """Refit the direct daily L2 model for each four-week test block."""
    config = GlobalLightGBMConfig() if config is None else config
    results = [
        _fit_origin(item, config, params) for item in origin_frame_sequence(frames)
    ]
    combined = combine_origin_results(results, GlobalLightGBMResult)
    assert isinstance(combined, GlobalLightGBMResult)
    return combined


def run_global_lightgbm(
    *,
    design: BenchmarkDesign,
    evaluation_origins: Iterable[object],
    connection: duckdb.DuckDBPyConnection | None = None,
    data_dir: Path | None = None,
    config: GlobalLightGBMConfig | None = None,
    feature_dataset_path: Path | str = DEFAULT_FEATURES_PATH,
    force_feature_recompute: bool = False,
) -> GlobalLightGBMResult:
    """Prepare weekly features and refit the L2 model every four origins."""
    config = GlobalLightGBMConfig() if config is None else config
    print("[global_lightgbm] Preparing shared feature frames...", flush=True)
    frames = prepare_global_lightgbm_frames(
        design=design,
        evaluation_origins=evaluation_origins,
        connection=connection,
        data_dir=data_dir,
        config=config,
        feature_dataset_path=feature_dataset_path,
        force_feature_recompute=force_feature_recompute,
    )
    print("[global_lightgbm] Fitting expanding-window L2 models...", flush=True)
    result = fit_global_lightgbm_frames(frames, config)
    print(
        f"[global_lightgbm] Completed: {len(result.forecasts):,} forecasts",
        flush=True,
    )
    return result


GlobalLightGBMModel = BaseLightGBMModel


def fit(frames: object, config: L2Config) -> object:
    """Fit the L2 model with its independently owned parameters."""
    return fit_global_lightgbm_frames(
        frames, config.execution_config(), params=config.parameters()
    )
