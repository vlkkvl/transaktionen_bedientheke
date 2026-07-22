"""Teunter-Syntetos-Babai intermittent-demand benchmark."""
from __future__ import annotations

import numpy as np

from src.models.baseline.base import (
    ArrayLike,
    ForecastModel,
    ses_level,
    validate_alpha,
)


DEFAULT_TSB_DEMAND_ALPHA = 0.1
DEFAULT_TSB_PROBABILITY_ALPHA = 0.1


def tsb_level(values: np.ndarray, alpha_d: float, alpha_p: float) -> float:
    """Estimate occurrence probability multiplied by positive-demand size."""
    positive = values[values > 0]
    if len(positive) == 0:
        return 0.0
    probability = ses_level((values > 0).astype(np.float64), alpha_p)
    positive_size = ses_level(positive, alpha_d)
    return max(probability * positive_size, 0.0)


class TSBForecast(ForecastModel):
    """TSB with separately fixed smoothing for demand size and occurrence."""

    name = "tsb"

    def __init__(
        self,
        alpha_d: float = DEFAULT_TSB_DEMAND_ALPHA,
        alpha_p: float = DEFAULT_TSB_PROBABILITY_ALPHA,
    ) -> None:
        self.alpha_d = validate_alpha(alpha_d, "alpha_d")
        self.alpha_p = validate_alpha(alpha_p, "alpha_p")
        self.level_: float | None = None

    def fit(self, y: ArrayLike) -> "TSBForecast":
        values = self.as_array(y)
        if len(values) == 0:
            raise ValueError("TSBForecast requires at least one observation")
        self.level_ = tsb_level(values, self.alpha_d, self.alpha_p)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if self.level_ is None:
            raise RuntimeError("fit must be called before predict")
        return np.full(horizon, self.level_, dtype=np.float64)
