"""Naive last-observation baseline."""
from __future__ import annotations

import numpy as np

from src.models.baseline.base import ArrayLike, ForecastModel


class NaiveForecast(ForecastModel):
    """Forecast every future period with the last observed demand value."""

    name = "naive"

    def __init__(self) -> None:
        self.last_value_: float = 0.0

    def fit(self, y: ArrayLike) -> "NaiveForecast":
        values = self.as_array(y)
        self.last_value_ = float(values[-1]) if len(values) else 0.0
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        return self.nonnegative(np.full(horizon, self.last_value_))
