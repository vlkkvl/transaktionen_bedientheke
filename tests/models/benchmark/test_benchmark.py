from __future__ import annotations

import unittest

import duckdb
import pandas as pd

from src.models.benchmark.config import BenchmarkDesign
from src.models.benchmark.evaluation import (
    _create_eligible_origins,
    _forecast_rows,
)
from src.models.benchmark.metrics import segment_wape, summarize_models
from src.models.benchmark.models import PRIMARY_MODEL, create_history_features


class BenchmarkDesignTest(unittest.TestCase):
    def test_origins_only_include_complete_horizons(self) -> None:
        design = BenchmarkDesign(61, 10, pd.Timestamp("2025-12-01"), 28, 7)

        origins = design.origins_through("2026-01-04")

        self.assertEqual(
            list(origins),
            [pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-29")],
        )


class ForecastEvaluationTest(unittest.TestCase):
    def test_maturity_and_forecasts_use_only_pre_origin_rows(self) -> None:
        con = duckdb.connect()
        con.execute(
            """
            CREATE TEMP TABLE benchmark_daily_rows (
                ARTIKEL_ID BIGINT,
                MARKT_ID BIGINT,
                period DATE,
                demand DOUBLE,
                sourcing_group VARCHAR,
                category_id INTEGER
            )
            """
        )
        dates = pd.date_range("2025-01-01", periods=77, freq="D")
        rows = [
            (1, 10, date.date(), float(index % 7 == 0), "FCM", 890)
            for index, date in enumerate(dates)
        ]
        con.executemany(
            "INSERT INTO benchmark_daily_rows VALUES (?, ?, ?, ?, ?, ?)", rows
        )
        create_history_features(con)
        origin = pd.Timestamp("2025-03-12")
        design = BenchmarkDesign(70, 10, origin, 28, 7)

        origin_summary = _create_eligible_origins(
            con, pd.DatetimeIndex([origin]), design
        )
        forecasts = _forecast_rows(con, horizon_days=7)

        self.assertEqual(int(origin_summary.loc[0, "mature_series"]), 1)
        self.assertEqual(forecasts["active_days_before_origin"].unique().tolist(), [70])
        self.assertEqual(forecasts["demand_days_before_origin"].unique().tolist(), [10])
        self.assertEqual(forecasts["days_since_last_demand"].unique().tolist(), [7])
        self.assertAlmostEqual(forecasts["recent_occurrence_rate"].iloc[0], 1 / 7)
        primary = forecasts[forecasts["model"].eq(PRIMARY_MODEL)]
        self.assertEqual(len(primary), 7)
        self.assertEqual(primary.iloc[0]["forecast"], 1.0)

class BenchmarkMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.forecasts = pd.DataFrame(
            {
                "ARTIKEL_ID": [1, 2],
                "MARKT_ID": [10, 10],
                "sourcing_group": ["FCM", "Pseudo"],
                "origin": pd.to_datetime(["2025-12-01", "2025-12-01"]),
                "model": [PRIMARY_MODEL, PRIMARY_MODEL],
                "actual": [10.0, 0.0],
                "forecast": [8.0, 1.0],
                "seasonal_mase_scale": [2.0, 2.0],
            }
        )

    def test_requested_summary_metrics(self) -> None:
        summary = summarize_models(self.forecasts).iloc[0]

        self.assertAlmostEqual(summary["pooled_wape"], 0.3)
        self.assertAlmostEqual(summary["relative_bias"], -0.1)
        self.assertAlmostEqual(summary["forecast_to_actual_ratio"], 0.9)
        self.assertAlmostEqual(summary["median_series_wape"], 0.2)
        self.assertAlmostEqual(summary["seasonal_mase"], 0.75)
        self.assertAlmostEqual(summary["mae_kg"], 1.5)

    def test_segment_wape_uses_pooled_segment_volume(self) -> None:
        segments = segment_wape(self.forecasts, "sourcing_group").set_index(
            "sourcing_group"
        )

        self.assertAlmostEqual(segments.loc["FCM", "pooled_wape"], 0.2)
        self.assertTrue(pd.isna(segments.loc["Pseudo", "pooled_wape"]))


if __name__ == "__main__":
    unittest.main()
