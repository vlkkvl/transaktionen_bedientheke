"""ETS(A,N,N), also known as simple exponential smoothing."""
from __future__ import annotations

import numpy as np

from src.models.baseline.base import (
    ArrayLike,
    ForecastModel,
    ses_level,
    validate_alpha,
)


DEFAULT_SES_ALPHA = 0.2


class SimpleExponentialSmoothingForecast(ForecastModel):
    """Level-only ETS benchmark with a fixed, preregistered smoothing value."""

    name = "simple_exponential_smoothing"

    def __init__(self, alpha: float = DEFAULT_SES_ALPHA) -> None:
        self.alpha = validate_alpha(alpha)
        self.level_: float | None = None

    def fit(self, y: ArrayLike) -> "SimpleExponentialSmoothingForecast":
        values = self.as_array(y)
        self.level_ = ses_level(values, self.alpha)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if self.level_ is None:
            raise RuntimeError("fit must be called before predict")
        return np.full(horizon, self.level_, dtype=np.float64)
