from __future__ import annotations

import unittest

import duckdb
import numpy as np
import pandas as pd

from src.models.baseline.aggregate_then_disaggregate import (
    AggregateThenDisaggregateForecast,
)
from src.models.baseline.batch import SCALAR_MODEL_NAMES, forecast_scalar_levels
from src.models.baseline.evaluation import (
    forecast_aggregate_then_disaggregate,
    forecast_scalar_baselines,
)
from src.models.baseline.croston import CrostonForecast
from src.models.baseline.sba import SBAForecast
from src.models.baseline.simple_exponential_smoothing import (
    SimpleExponentialSmoothingForecast,
)
from src.models.baseline.tsb import TSBForecast
from src.models.benchmark.config import BenchmarkDesign
from src.models.benchmark.evaluation import _create_assessed_origins
from src.models.benchmark.models import create_history_features


class BaselineFormulaTest(unittest.TestCase):
    def test_batch_and_reusable_classes_agree(self) -> None:
        history = np.array([0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0])
        expected = [
            SimpleExponentialSmoothingForecast().forecast(history, 1)[0],
            CrostonForecast().forecast(history, 1)[0],
            SBAForecast().forecast(history, 1)[0],
            TSBForecast().forecast(history, 1)[0],
        ]

        actual = forecast_scalar_levels([history])[0]

        np.testing.assert_allclose(actual, expected)
        self.assertEqual(len(actual), len(SCALAR_MODEL_NAMES))

    def test_aggregate_forecast_conserves_weekly_level(self) -> None:
        origin = pd.Timestamp("2025-03-03")
        dates = pd.date_range(origin - pd.Timedelta(days=56), periods=56, freq="D")
        demand = np.where(dates.weekday == 0, 2.0, 1.0)
        model = AggregateThenDisaggregateForecast().fit(demand, dates, origin)

        forecast = model.predict(7, pd.date_range(origin, periods=7, freq="D"))

        self.assertAlmostEqual(forecast.sum(), model.weekly_level_)
        self.assertAlmostEqual(forecast[0] / forecast[1], 2.0)

    def test_aggregate_zero_profile_uses_equal_shares(self) -> None:
        origin = pd.Timestamp("2025-03-03")
        dates = pd.date_range(origin - pd.Timedelta(days=56), periods=56, freq="D")
        model = AggregateThenDisaggregateForecast().fit(
            np.zeros(56), dates, origin
        )

        np.testing.assert_allclose(model.weekday_profile_, np.full(7, 1 / 7))


class BaselineEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect()
        self.con.execute(
            """
            CREATE TEMP TABLE benchmark_daily_rows (
                ARTIKEL_ID BIGINT,
                MARKT_ID BIGINT,
                period DATE,
                demand DOUBLE,
                is_active BOOLEAN,
                reason_closed VARCHAR,
                sourcing_group VARCHAR,
                category_id INTEGER
            )
            """
        )
        dates = pd.date_range("2025-01-01", periods=77, freq="D")
        rows = [
            (
                1,
                10,
                date.date(),
                1000.0 if index >= 70 else 1.0,
                True,
                None,
                "FCM",
                890,
            )
            for index, date in enumerate(dates)
        ]
        self.con.executemany(
            "INSERT INTO benchmark_daily_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        create_history_features(self.con)
        self.origin = pd.Timestamp("2025-03-12")
        design = BenchmarkDesign(61, self.origin, 28, 7)
        _create_assessed_origins(
            self.con, pd.DatetimeIndex([self.origin]), design
        )

    def test_scalar_models_ignore_origin_and_future_rows(self) -> None:
        forecasts = forecast_scalar_baselines(self.con, horizon_days=7)

        self.assertEqual(set(forecasts["model"]), set(SCALAR_MODEL_NAMES))
        self.assertEqual(len(forecasts), 4 * 7)
        levels = forecasts.groupby("model")["forecast"].first()
        self.assertAlmostEqual(levels["simple_exponential_smoothing"], 1.0)
        self.assertAlmostEqual(levels["croston"], 1.0)
        self.assertAlmostEqual(levels["sba"], 0.95)
        self.assertAlmostEqual(levels["tsb"], 1.0)

    def test_aggregate_model_uses_only_pre_origin_history(self) -> None:
        forecasts = forecast_aggregate_then_disaggregate(
            self.con, horizon_days=7
        )

        self.assertEqual(len(forecasts), 7)
        np.testing.assert_allclose(forecasts["forecast"], 1.0)


if __name__ == "__main__":
    unittest.main()
