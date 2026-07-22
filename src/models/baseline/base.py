"""Shared interface and validation for univariate baseline forecasts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


class ForecastModel(ABC):
    """Minimal fit/predict interface shared by scalar baseline models."""

    name: str

    @abstractmethod
    def fit(self, y: ArrayLike) -> "ForecastModel":
        """Fit the model on one observed demand history."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return a nonnegative forecast for the requested horizon."""

    def forecast(self, y: ArrayLike, horizon: int) -> np.ndarray:
        """Fit the model and immediately forecast."""
        return self.fit(y).predict(horizon)

    @staticmethod
    def as_array(y: ArrayLike) -> np.ndarray:
        values = np.asarray(y, dtype=np.float64).reshape(-1)
        return np.maximum(
            np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0
        )

    @staticmethod
    def validate_horizon(horizon: int) -> int:
        normalized = int(horizon)
        if normalized < 1:
            raise ValueError("horizon must be a positive integer")
        return normalized


def validate_alpha(alpha: float, name: str = "alpha") -> float:
    """Return a smoothing parameter after validating its unit interval."""
    normalized = float(alpha)
    if not 0 < normalized <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return normalized


def ses_level(values: np.ndarray, alpha: float) -> float:
    """Return the final simple-exponential-smoothing level."""
    if len(values) == 0:
        raise ValueError("at least one observation is required")
    level = float(values[0])
    for value in values[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return max(level, 0.0)
