"""Syntetos-Boylan approximation baseline backed by statsforecast."""
from __future__ import annotations

import numpy as np
from statsforecast.models import CrostonSBA

from src.models.baseline.base import ArrayLike, ForecastModel


class SBAForecast(ForecastModel):
    """Bias-corrected Croston forecast using statsforecast's CrostonSBA."""

    name = "sba"

    def __init__(self) -> None:
        self.model_: CrostonSBA | None = None

    def fit(self, y: ArrayLike) -> "SBAForecast":
        values = self.as_array(y)
        if len(values) == 0:
            raise ValueError("SBAForecast requires at least one observation")
        self.model_ = CrostonSBA(alias=self.name).fit(values)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if self.model_ is None:
            raise RuntimeError("SBAForecast must be fit before predict")
        forecast = self.model_.predict(horizon)["mean"]
        return self.nonnegative(forecast)
