"""Damped-trend exponential smoothing baseline backed by statsmodels."""
from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.models.baseline.base import ArrayLike, ForecastModel

DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.1
DEFAULT_PHI = 0.9


class DampedTrendETSForecast(ForecastModel):
    """Additive damped-trend ETS without seasonality using statsmodels."""

    name = "damped_trend_ets"

    def __init__(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        phi: float | None = None,
        optimized: bool = True,
        remove_bias: bool = False,
    ) -> None:
        self._validate_optional_params(alpha, beta, phi)
        if not optimized and any(value is None for value in (alpha, beta, phi)):
            raise ValueError("alpha, beta, and phi are required when optimized=False")
        self.alpha = float(alpha) if alpha is not None else None
        self.beta = float(beta) if beta is not None else None
        self.phi = float(phi) if phi is not None else None
        self.optimized = optimized
        self.remove_bias = remove_bias
        self.level_: float = 0.0
        self.trend_: float = 0.0
        self.alpha_: float | None = self.alpha
        self.beta_: float | None = self.beta
        self.phi_: float | None = self.phi
        self.optimization_converged_: bool | None = None
        self._fit_result = None

    def fit(self, y: ArrayLike) -> "DampedTrendETSForecast":
        values = self.as_array(y)
        if len(values) < 2:
            raise ValueError("DampedTrendETSForecast requires at least two observations")

        fit_kwargs = {
            "optimized": self.optimized,
            "remove_bias": self.remove_bias,
        }
        if self.alpha is not None:
            fit_kwargs["smoothing_level"] = self.alpha
        if self.beta is not None:
            fit_kwargs["smoothing_trend"] = self.beta
        if self.phi is not None:
            fit_kwargs["damping_trend"] = self.phi

        self._fit_result = self._fit_statsmodels(values, fit_kwargs)

        params = self._fit_result.params
        self.alpha_ = self._get_optional_float(params, "smoothing_level", self.alpha)
        self.beta_ = self._get_optional_float(params, "smoothing_trend", self.beta)
        self.phi_ = self._get_optional_float(params, "damping_trend", self.phi)
        self.level_ = float(np.asarray(self._fit_result.level)[-1])
        trend_values = getattr(
            self._fit_result,
            "trend",
            getattr(self._fit_result, "slope", None),
        )
        self.trend_ = float(np.asarray(trend_values)[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if self._fit_result is None:
            raise RuntimeError("DampedTrendETSForecast must be fit before predict")
        return self.nonnegative(self._fit_result.forecast(horizon))

    def _fit_statsmodels(self, values: np.ndarray, fit_kwargs: dict):
        model = ExponentialSmoothing(
            values,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                result = model.fit(**fit_kwargs)
            self.optimization_converged_ = True if self.optimized else None
            return result
        except ConvergenceWarning:
            self.optimization_converged_ = False
            return model.fit(
                optimized=False,
                smoothing_level=self.alpha if self.alpha is not None else DEFAULT_ALPHA,
                smoothing_trend=self.beta if self.beta is not None else DEFAULT_BETA,
                damping_trend=self.phi if self.phi is not None else DEFAULT_PHI,
                remove_bias=self.remove_bias,
            )

    @staticmethod
    def _validate_optional_params(
        alpha: float | None,
        beta: float | None,
        phi: float | None,
    ) -> None:
        if alpha is not None and not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if beta is not None and not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        if phi is not None and not 0 < phi <= 1:
            raise ValueError("phi must be in (0, 1]")

    @staticmethod
    def _get_optional_float(
        params: Mapping[str, object],
        name: str,
        default: float | None,
    ) -> float | None:
        value = params.get(name, default)
        return float(value) if value is not None else None
