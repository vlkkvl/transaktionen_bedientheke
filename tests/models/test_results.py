from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.models.benchmark.config import BenchmarkDesign
from src.models.results import (
    read_forecasts,
    read_result,
    result_path,
    write_forecasts,
    write_result,
)


class ResultIOTest(unittest.TestCase):
    def setUp(self) -> None:
        self.design = BenchmarkDesign(28, pd.Timestamp("2025-12-01"), 7, 7)

    def test_lightgbm_tweedie_uses_requested_result_name(self) -> None:
        path = result_path("global_lightgbm_tweedie_daily", self.design)

        self.assertEqual(
            path.name, "artikel_markt_multi7days_lightgbm_tweedie.csv"
        )

    def test_forecast_round_trip_splits_models_and_restores_dates(self) -> None:
        forecasts = pd.DataFrame(
            {
                "origin": pd.to_datetime(["2025-12-01", "2025-12-01"]),
                "period": pd.to_datetime(["2025-12-01", "2025-12-01"]),
                "model": ["recent_mean", "croston"],
                "actual": [1.0, 1.0],
                "forecast": [0.5, 0.75],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            write_forecasts(forecasts, self.design, results_dir=directory)
            restored = read_forecasts(
                ["recent_mean", "croston"],
                self.design,
                results_dir=directory,
            )

        self.assertEqual(set(restored["model"]), {"recent_mean", "croston"})
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(restored["origin"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(restored["period"]))

    def test_training_summary_restores_refit_block_end_date(self) -> None:
        summary = pd.DataFrame(
            {
                "evaluation_origin": [pd.Timestamp("2025-12-01")],
                "evaluation_end": [pd.Timestamp("2025-12-22")],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            write_result(
                summary,
                "global_lightgbm",
                self.design,
                artifact="training_summaries",
                results_dir=directory,
            )
            restored = read_result(
                "global_lightgbm",
                self.design,
                artifact="training_summaries",
                results_dir=directory,
            )

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(restored["evaluation_end"])
        )


if __name__ == "__main__":
    unittest.main()
