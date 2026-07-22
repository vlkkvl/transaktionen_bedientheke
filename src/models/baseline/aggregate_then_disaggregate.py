"""Weekly aggregate forecast distributed through a historical weekday profile."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.models.baseline.base import ArrayLike, ForecastModel


LEVEL_WEEKS = 4
PROFILE_WEEKS = 8


class AggregateThenDisaggregateForecast(ForecastModel):
    """Forecast a seven-day total and distribute it by pre-origin weekday share."""

    name = "aggregate_then_disaggregate"

    def __init__(
        self,
        level_weeks: int = LEVEL_WEEKS,
        profile_weeks: int = PROFILE_WEEKS,
    ) -> None:
        if int(level_weeks) < 1 or int(profile_weeks) < 1:
            raise ValueError("lookback weeks must be positive integers")
        self.level_weeks = int(level_weeks)
        self.profile_weeks = int(profile_weeks)
        self.weekly_level_: float | None = None
        self.weekday_profile_: np.ndarray | None = None
        self.origin_: pd.Timestamp | None = None

    def fit(  # type: ignore[override]
        self,
        y: ArrayLike,
        dates: Sequence[object] | pd.DatetimeIndex,
        origin: object,
    ) -> "AggregateThenDisaggregateForecast":
        values = self.as_array(y)
        observed_dates = pd.DatetimeIndex(dates).normalize()
        origin_date = pd.Timestamp(origin).normalize()
        if len(values) != len(observed_dates):
            raise ValueError("y and dates must have equal length")
        if (observed_dates >= origin_date).any():
            raise ValueError("all fitted dates must be strictly before origin")

        level_start = origin_date - pd.Timedelta(days=7 * self.level_weeks)
        in_level = observed_dates >= level_start
        self.weekly_level_ = float(values[in_level].sum() / self.level_weeks)

        profile_start = origin_date - pd.Timedelta(days=7 * self.profile_weeks)
        in_profile = observed_dates >= profile_start
        weekday_totals = np.bincount(
            observed_dates[in_profile].weekday,
            weights=values[in_profile],
            minlength=7,
        ).astype(np.float64)
        profile_total = float(weekday_totals.sum())
        self.weekday_profile_ = (
            weekday_totals / profile_total
            if profile_total > 0
            else np.full(7, 1.0 / 7.0)
        )
        self.origin_ = origin_date
        return self

    def predict(  # type: ignore[override]
        self,
        horizon: int,
        target_dates: Sequence[object] | pd.DatetimeIndex | None = None,
    ) -> np.ndarray:
        horizon = self.validate_horizon(horizon)
        if horizon != 7:
            raise ValueError("aggregate-then-disaggregate is defined for a 7-day horizon")
        if (
            self.weekly_level_ is None
            or self.weekday_profile_ is None
            or self.origin_ is None
        ):
            raise RuntimeError("fit must be called before predict")
        weekdays = (
            pd.date_range(self.origin_, periods=horizon, freq="D").weekday.to_numpy()
            if target_dates is None
            else pd.DatetimeIndex(target_dates).weekday.to_numpy()
        )
        if len(weekdays) != horizon:
            raise ValueError("target_dates must match horizon")
        return self.weekly_level_ * self.weekday_profile_[weekdays]

    def forecast(  # type: ignore[override]
        self,
        y: ArrayLike,
        horizon: int,
        *,
        dates: Sequence[object] | pd.DatetimeIndex,
        origin: object,
        target_dates: Sequence[object] | pd.DatetimeIndex | None = None,
    ) -> np.ndarray:
        return self.fit(y, dates, origin).predict(horizon, target_dates)
