from __future__ import annotations

import unittest

import duckdb
import pandas as pd

from src.models.benchmark.config import BenchmarkDesign
from src.models.benchmark.evaluation import (
    _create_assessed_origins,
    _forecast_rows,
)
from src.models.benchmark.metrics import segment_wape, summarize_models
from src.models.benchmark.models import PRIMARY_MODEL, create_history_features


class BenchmarkDesignTest(unittest.TestCase):
    def test_origins_only_include_complete_horizons(self) -> None:
        design = BenchmarkDesign(28, pd.Timestamp("2026-03-02"), 7, 7, 20)

        origins = design.origins_through("2026-07-21")

        self.assertEqual(len(origins), 20)
        self.assertEqual(origins[0], pd.Timestamp("2026-03-02"))
        self.assertEqual(origins[-1], pd.Timestamp("2026-07-13"))
        self.assertTrue((origins.weekday == 0).all())


class ForecastEvaluationTest(unittest.TestCase):
    def test_series_with_28_active_days_is_assessed_without_demand(self) -> None:
        con = duckdb.connect()
        con.execute(
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
        dates = pd.date_range("2025-01-01", periods=35, freq="D")
        con.executemany(
            "INSERT INTO benchmark_daily_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(1, 10, date.date(), 0.0, True, None, "FCM", 890) for date in dates],
        )
        create_history_features(con)
        origin = pd.Timestamp("2025-01-29")
        design = BenchmarkDesign(28, origin, 7, 7)

        summary = _create_assessed_origins(con, pd.DatetimeIndex([origin]), design)

        self.assertEqual(int(summary.loc[0, "assessed_series"]), 1)

    def test_maturity_and_forecasts_use_only_pre_origin_rows(self) -> None:
        con = duckdb.connect()
        con.execute(
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
                float(index % 7 == 0 and index != 63),
                index not in {63, 74},
                "Holiday" if index == 63 else "Sunday" if index == 74 else None,
                "FCM",
                890,
            )
            for index, date in enumerate(dates)
        ]
        con.executemany(
            "INSERT INTO benchmark_daily_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        create_history_features(con)
        origin = pd.Timestamp("2025-03-12")
        design = BenchmarkDesign(69, origin, 28, 7)

        origin_summary = _create_assessed_origins(
            con, pd.DatetimeIndex([origin]), design
        )
        forecasts = _forecast_rows(con, horizon_days=7)

        self.assertEqual(int(origin_summary.loc[0, "assessed_series"]), 1)
        self.assertEqual(forecasts["active_days_before_origin"].unique().tolist(), [69])
        self.assertEqual(forecasts["demand_days_before_origin"].unique().tolist(), [9])
        self.assertEqual(
            forecasts["calendar_days_since_last_demand"].unique().tolist(), [14]
        )
        self.assertAlmostEqual(forecasts["recent_occurrence_rate"].iloc[0], 1 / 9)
        primary = forecasts[forecasts["model"].eq(PRIMARY_MODEL)]
        self.assertEqual(len(primary), 7)
        self.assertEqual(primary.iloc[0]["forecast"], 1.0)
        closed = primary.loc[~primary["is_active"]].iloc[0]
        self.assertEqual(closed["reason_closed"], "Sunday")
        self.assertTrue(pd.isna(closed["forecast"]))

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

    def test_metric_reducers_exclude_closed_rows_when_flag_is_available(self) -> None:
        forecasts = self.forecasts.assign(is_active=[True, False])

        summary = summarize_models(forecasts).iloc[0]

        self.assertAlmostEqual(summary["pooled_wape"], 0.2)
        self.assertEqual(summary["n_forecast_rows"], 1)


if __name__ == "__main__":
    unittest.main()
