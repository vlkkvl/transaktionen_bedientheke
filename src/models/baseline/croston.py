"""Croston intermittent-demand baseline backed by statsforecast."""
from __future__ import annotations

import numpy as np
from statsforecast.models import CrostonClassic

from src.models.baseline.base import ArrayLike, ForecastModel


class CrostonForecast(ForecastModel):
    """Croston forecast for intermittent demand using statsforecast."""

    name = "croston"

    def __init__(self) -> None:
        self.model_: CrostonClassic | None = None

    def fit(self, y: ArrayLike) -> "CrostonForecast":
        values = self.as_array(y)
        if len(values) == 0:
            raise ValueError("CrostonForecast requires at least one observation")
        self.model_ = CrostonClassic(alias=self.name).fit(values)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if self.model_ is None:
            raise RuntimeError("CrostonForecast must be fit before predict")
        forecast = self.model_.predict(horizon)["mean"]
        return self.nonnegative(forecast)
