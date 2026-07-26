"""Composed configuration shared in shape by LightGBM model packages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models.core.config import FitControl, ForecastWindow, RuntimeConfig
from src.models.lightgbm.features.builder import GlobalLightGBMConfig


@dataclass(frozen=True)
class LightGBMModelConfig:
    """Non-statistical settings composed into every model-specific config."""

    window: ForecastWindow = field(default_factory=ForecastWindow)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    fit_control: FitControl = field(default_factory=FitControl)

    def execution_config(self) -> GlobalLightGBMConfig:
        """Adapt the new composed config to the existing frame-building API."""
        return GlobalLightGBMConfig(
            training_origins=self.window.training_origins,
            validation_origins=self.window.validation_origins,
            test_origins=self.window.test_origins,
            refit_interval_origins=self.window.refit_interval_origins,
            origin_spacing_days=self.window.origin_spacing_days,
            num_boost_round=self.fit_control.num_boost_round,
            early_stopping_rounds=self.fit_control.early_stopping_rounds,
            random_state=self.runtime.random_state,
            num_threads=self.runtime.num_threads,
        )

    def seeded_parameters(self) -> dict[str, Any]:
        seed = self.runtime.random_state
        return {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "num_threads": self.runtime.num_threads,
            "verbosity": -1,
        }
