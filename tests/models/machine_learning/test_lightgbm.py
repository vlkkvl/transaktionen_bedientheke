from __future__ import annotations

import unittest

import duckdb
import numpy as np
import pandas as pd

from src.models.benchmark.config import BenchmarkDesign
from src.models.benchmark.evaluation import _create_eligible_origins
from src.models.benchmark.models import create_history_features
from src.models.machine_learning import (
    FEATURE_COLUMNS,
    GlobalLightGBMConfig,
    create_feature_tables,
    make_feature_frame,
    prepare_global_lightgbm_frames,
    run_global_lightgbm,
)
from src.models.machine_learning import (
    WEEKLY_MODEL_NAME,
    fit_all_lightgbm_models,
)


class GlobalLightGBMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect()
        self.con.execute(
            """
            CREATE TEMP TABLE benchmark_daily_rows (
                ARTIKEL_ID BIGINT,
                MARKT_ID BIGINT,
                period DATE,
                demand DOUBLE,
                action_flag TINYINT,
                sourcing_group VARCHAR,
                category_id INTEGER
            )
            """
        )
        dates = pd.date_range("2024-01-01", periods=500, freq="D")
        rows = []
        for article in (1, 2):
            for market in (10, 11):
                rows.extend(
                    (
                        article,
                        market,
                        date.date(),
                        float((index + article + market) % 7 == 0) * (article + 1),
                        int(index % 30 == 0 or index == 282),
                        "FCM" if article == 1 else "Pseudo",
                        890 if article == 1 else 900,
                    )
                    for index, date in enumerate(dates)
                )
        self.con.executemany(
            "INSERT INTO benchmark_daily_rows VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )
        self.design = BenchmarkDesign(61, 8, pd.Timestamp("2025-01-01"), 28, 7)
        create_history_features(self.con)
        _create_eligible_origins(
            self.con, pd.DatetimeIndex([self.design.first_origin]), self.design
        )

    def test_features_are_anchored_strictly_before_origin(self) -> None:
        create_feature_tables(self.con)
        origin = pd.Timestamp("2024-10-07")
        frame = make_feature_frame(self.con, [origin], self.design)
        first = frame.loc[
            frame.ARTIKEL_ID.eq(1) & frame.MARKT_ID.eq(10)
        ].iloc[0]
        history = self.con.execute(
            """
            SELECT demand FROM benchmark_daily_rows
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10 AND period < ?
            ORDER BY period
            """,
            [origin.date()],
        ).fetchnumpy()["demand"]

        self.assertEqual(set(FEATURE_COLUMNS) - set(frame.columns), set())
        self.assertEqual(first.lag_1, history[-1])
        self.assertEqual(first.lag_7, history[-7])
        self.assertAlmostEqual(first.rolling_28_mean, np.mean(history[-28:]))
        self.assertEqual(first.days_since_last_action, 10)
        self.assertEqual(first.actions_last_28d, 1)
        self.assertEqual(frame["action_during_horizon"].unique().tolist(), [1])
        self.assertEqual(
            frame.loc[frame["horizon_day"].eq(3), "action_on_forecast_day"]
            .unique()
            .tolist(),
            [1],
        )
        self.assertTrue(frame["mean_action_lift_in_sourcing_group"].notna().all())
        self.assertNotIn("demand_class", frame.columns)

    def test_end_to_end_model_returns_nonnegative_matching_rows(self) -> None:
        result = run_global_lightgbm(
            design=self.design,
            evaluation_origins=[self.design.first_origin],
            connection=self.con,
            config=GlobalLightGBMConfig(
                max_training_origins=4,
                validation_origins=1,
                num_boost_round=10,
                early_stopping_rounds=3,
                num_threads=2,
            ),
        )

        self.assertFalse(result.forecasts.empty)
        self.assertTrue(result.forecasts.forecast.ge(0).all())
        self.assertEqual(result.forecasts.model.unique().tolist(), ["global_lightgbm"])
        self.assertEqual(result.training_summary.loc[0, "evaluation_origins"], 1)

    def test_origin_splits_expand_fit_history_and_keep_two_for_validation(self) -> None:
        evaluation_origins = pd.date_range(
            self.design.first_origin
            + pd.Timedelta(days=2 * self.design.origin_spacing_days),
            periods=3,
            freq=f"{self.design.origin_spacing_days}D",
        )
        frames = prepare_global_lightgbm_frames(
            design=self.design,
            evaluation_origins=evaluation_origins,
            connection=self.con,
            config=GlobalLightGBMConfig(
                num_boost_round=5,
                early_stopping_rounds=2,
                num_threads=2,
            ),
        )

        self.assertEqual(len(frames.origin_frames), 3)
        self.assertEqual(
            [frame.training["origin"].nunique() for frame in frames.origin_frames],
            [10, 11, 12],
        )
        self.assertEqual(
            [frame.validation["origin"].nunique() for frame in frames.origin_frames],
            [2, 2, 2],
        )
        self.assertEqual(
            [len(frame.training_origins) for frame in frames.origin_frames],
            [12, 13, 14],
        )
        for expected_origin, frame in zip(evaluation_origins, frames.origin_frames):
            self.assertEqual(frame.evaluation["origin"].nunique(), 1)
            self.assertEqual(frame.evaluation_origin, expected_origin)
            self.assertLess(
                pd.to_datetime(frame.validation["origin"]).max(),
                expected_origin,
            )

    def test_all_variants_match_rows_and_weekly_allocation_conserves_total(self) -> None:
        config = GlobalLightGBMConfig(
            max_training_origins=4,
            validation_origins=1,
            num_boost_round=10,
            early_stopping_rounds=3,
            num_threads=2,
        )
        evaluation_origins = pd.date_range(
            self.design.first_origin,
            periods=2,
            freq=f"{self.design.origin_spacing_days}D",
        )
        frames = prepare_global_lightgbm_frames(
            design=self.design,
            evaluation_origins=evaluation_origins,
            connection=self.con,
            config=config,
        )
        results = fit_all_lightgbm_models(frames, config)

        self.assertEqual(len(results), 4)
        for result in results.values():
            self.assertEqual(len(result.forecasts), len(frames.evaluation))
            self.assertTrue(result.forecasts.forecast.ge(0).all())
            self.assertEqual(result.training_summary["evaluation_origin"].nunique(), 2)
        weekly_audit = results[WEEKLY_MODEL_NAME].allocation_audit
        self.assertIsNotNone(weekly_audit)
        assert weekly_audit is not None
        self.assertLess(weekly_audit.allocation_error.max(), 1e-9)
        self.assertTrue(np.allclose(weekly_audit.weekday_share_sum, 1.0))


if __name__ == "__main__":
    unittest.main()
