"""Historical mean forecast baseline."""
from __future__ import annotations

import numpy as np

from src.models.baseline.base import ArrayLike, ForecastModel


class MeanForecast(ForecastModel):
    """Forecast every future period with the historical mean demand."""

    name = "mean"

    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, y: ArrayLike) -> "MeanForecast":
        values = self.as_array(y)
        self.mean_ = float(np.mean(values)) if len(values) else 0.0
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        return self.nonnegative(np.full(horizon, self.mean_))
