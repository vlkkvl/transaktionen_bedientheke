"""Shared interface for univariate forecasting baselines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


class ForecastModel(ABC):
    """Minimal interface shared by all baseline forecasting models."""

    name: str

    @abstractmethod
    def fit(self, y: ArrayLike) -> "ForecastModel":
        """Fit the model on a single demand series."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` periods."""

    def forecast(self, y: ArrayLike, horizon: int) -> np.ndarray:
        """Fit the model and immediately return a forecast."""
        return self.fit(y).predict(horizon)

    @staticmethod
    def as_array(y: ArrayLike) -> np.ndarray:
        """Convert model input to a clean one-dimensional float array."""
        values = np.asarray(y, dtype=float)
        if values.ndim != 1:
            values = values.reshape(-1)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def validate_horizon(horizon: int) -> int:
        """Validate and normalize the requested forecast horizon."""
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return horizon

    @staticmethod
    def nonnegative(values: Any) -> np.ndarray:
        """Return forecasts clipped to the non-negative demand domain."""
        return np.maximum(np.asarray(values, dtype=float), 0.0)
