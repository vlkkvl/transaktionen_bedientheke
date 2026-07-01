"""Teunter-Syntetos-Babai intermittent-demand baseline backed by statsforecast."""
from __future__ import annotations

import numpy as np
from statsforecast.models import TSB

from src.models.baseline.base import ArrayLike, ForecastModel


class TSBForecast(ForecastModel):
    """TSB method with separate demand occurrence and size smoothing."""

    name = "tsb"

    def __init__(self, alpha_d: float = 0.1, alpha_p: float = 0.1) -> None:
        if not 0 < alpha_d <= 1:
            raise ValueError("alpha_d must be in (0, 1]")
        if not 0 < alpha_p <= 1:
            raise ValueError("alpha_p must be in (0, 1]")
        self.alpha_d = float(alpha_d)
        self.alpha_p = float(alpha_p)
        self.model_: TSB | None = None

    def fit(self, y: ArrayLike) -> "TSBForecast":
        values = self.as_array(y)
        if len(values) == 0:
            raise ValueError("TSBForecast requires at least one observation")
        self.model_ = TSB(
            alpha_d=self.alpha_d,
            alpha_p=self.alpha_p,
            alias=self.name,
        ).fit(values)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if self.model_ is None:
            raise RuntimeError("TSBForecast must be fit before predict")
        forecast = self.model_.predict(horizon)["mean"]
        return self.nonnegative(forecast)
