"""Croston intermittent-demand benchmark."""
from __future__ import annotations

import numpy as np

from src.models.baseline.base import ArrayLike, ForecastModel, ses_level, validate_alpha


DEFAULT_CROSTON_ALPHA = 0.1


def croston_level(values: np.ndarray, alpha: float = DEFAULT_CROSTON_ALPHA) -> float:
    """Estimate demand size divided by the inter-demand interval."""
    positive_positions = np.flatnonzero(values > 0)
    if len(positive_positions) == 0:
        return 0.0
    sizes = values[positive_positions]
    intervals = np.diff(positive_positions + 1, prepend=0).astype(np.float64)
    size_level = ses_level(sizes, alpha)
    interval_level = ses_level(intervals, alpha)
    return max(size_level / interval_level if interval_level else size_level, 0.0)


class CrostonForecast(ForecastModel):
    """Classic Croston with fixed smoothing for size and interval states."""

    name = "croston"

    def __init__(self, alpha: float = DEFAULT_CROSTON_ALPHA) -> None:
        self.alpha = validate_alpha(alpha)
        self.level_: float | None = None

    def fit(self, y: ArrayLike) -> "CrostonForecast":
        values = self.as_array(y)
        if len(values) == 0:
            raise ValueError("CrostonForecast requires at least one observation")
        self.level_ = croston_level(values, self.alpha)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if self.level_ is None:
            raise RuntimeError("fit must be called before predict")
        return np.full(horizon, self.level_, dtype=np.float64)
