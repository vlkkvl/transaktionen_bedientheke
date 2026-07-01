"""Simple exponential smoothing baseline backed by statsmodels."""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.models.baseline.base import ArrayLike, ForecastModel

DEFAULT_ALPHA = 0.2


class SimpleExponentialSmoothingForecast(ForecastModel):
    """Simple exponential smoothing without seasonality using statsmodels."""

    name = "simple_exponential_smoothing"

    def __init__(
        self,
        alpha: float | None = None,
        optimized: bool = True,
        remove_bias: bool = False,
    ) -> None:
        if alpha is not None and not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not optimized and alpha is None:
            raise ValueError("alpha is required when optimized=False")
        self.alpha = float(alpha) if alpha is not None else None
        self.optimized = optimized
        self.remove_bias = remove_bias
        self.level_: float = 0.0
        self.alpha_: float | None = self.alpha
        self.optimization_converged_: bool | None = None
        self._fit_result = None

    def fit(self, y: ArrayLike) -> "SimpleExponentialSmoothingForecast":
        values = self.as_array(y)
        if len(values) == 0:
            raise ValueError(
                "SimpleExponentialSmoothingForecast requires at least one observation"
            )

        fit_kwargs = {
            "optimized": self.optimized,
            "remove_bias": self.remove_bias,
        }
        if self.alpha is not None:
            fit_kwargs["smoothing_level"] = self.alpha

        self._fit_result = self._fit_statsmodels(values, fit_kwargs)
        self.alpha_ = float(self._fit_result.params["smoothing_level"])
        self.level_ = float(np.asarray(self._fit_result.level)[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if self._fit_result is None:
            raise RuntimeError(
                "SimpleExponentialSmoothingForecast must be fit before predict"
            )
        return self.nonnegative(self._fit_result.forecast(horizon))

    def _fit_statsmodels(self, values: np.ndarray, fit_kwargs: dict):
        model = SimpleExpSmoothing(values, initialization_method="estimated")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                result = model.fit(**fit_kwargs)
            self.optimization_converged_ = True if self.optimized else None
            return result
        except ConvergenceWarning:
            fixed_alpha = self.alpha if self.alpha is not None else DEFAULT_ALPHA
            self.optimization_converged_ = False
            return model.fit(
                optimized=False,
                smoothing_level=fixed_alpha,
                remove_bias=self.remove_bias,
            )
