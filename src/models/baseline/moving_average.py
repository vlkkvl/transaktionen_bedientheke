"""Moving-average forecast baseline."""
from __future__ import annotations

import numpy as np

from src.models.baseline.base import ArrayLike, ForecastModel


class MovingAverageForecast(ForecastModel):
    """Forecast with the mean of the most recent ``window`` observations."""

    name = "moving_average_4"

    def __init__(self, window: int = 4) -> None:
        if window < 1:
            raise ValueError("window must be a positive integer")
        self.window = int(window)
        self.average_: float = 0.0

    def fit(self, y: ArrayLike) -> "MovingAverageForecast":
        values = self.as_array(y)
        window_values = values[-self.window :] if len(values) else values
        self.average_ = float(np.mean(window_values)) if len(window_values) else 0.0
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        return self.nonnegative(np.full(horizon, self.average_))
