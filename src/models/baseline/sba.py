"""Syntetos-Boylan bias adjustment for Croston forecasts."""
from __future__ import annotations

import numpy as np

from src.models.baseline.base import ArrayLike, ForecastModel, validate_alpha
from src.models.baseline.croston import DEFAULT_CROSTON_ALPHA, croston_level


class SBAForecast(ForecastModel):
    """Croston forecast multiplied by its standard ``1 - alpha / 2`` factor."""

    name = "sba"

    def __init__(self, alpha: float = DEFAULT_CROSTON_ALPHA) -> None:
        self.alpha = validate_alpha(alpha)
        self.level_: float | None = None

    def fit(self, y: ArrayLike) -> "SBAForecast":
        values = self.as_array(y)
        if len(values) == 0:
            raise ValueError("SBAForecast requires at least one observation")
        self.level_ = (1.0 - self.alpha / 2.0) * croston_level(values, self.alpha)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if self.level_ is None:
            raise RuntimeError("fit must be called before predict")
        return np.full(horizon, self.level_, dtype=np.float64)
