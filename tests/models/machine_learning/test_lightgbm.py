from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

from src.models.benchmark.config import BenchmarkDesign
from src.models.benchmark.evaluation import _create_assessed_origins
from src.models.benchmark.models import create_history_features
from src.models.lightgbm import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    NORMALIZED_TARGET_COLUMN,
    TARGET_SCALE_COLUMN,
    TWO_STAGE_MODEL_NAME,
    WEEKLY_MODEL_NAME,
    GlobalLightGBMConfig,
    create_feature_tables,
    fit_all_lightgbm_models,
    make_feature_frame,
    make_weekly_frame,
    prepare_global_lightgbm_frames,
    run_global_lightgbm,
)
from src.models.lightgbm.base import BaseLightGBMModel
from src.models.lightgbm.features.builder import (
    iter_lightgbm_origin_windows,
    materialize_features_for_origins,
)
from src.models.lightgbm.features.definition import (
    DIAGNOSTIC_COLUMNS,
    FORECAST_ID_COLUMNS,
)
from src.models.lightgbm.weekly_total.model import (
    WEEKLY_FEATURE_COLUMNS,
)


class GlobalLightGBMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.feature_dataset_path = Path(self.temp_dir.name) / "features.parquet"
        self.con = duckdb.connect()
        self.addCleanup(self.con.close)
        self.con.execute(
            """
            CREATE TEMP TABLE benchmark_daily_rows (
                ARTIKEL_ID BIGINT,
                MARKT_ID BIGINT,
                period DATE,
                demand DOUBLE,
                is_active BOOLEAN,
                reason_closed VARCHAR,
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
                        True,
                        None,
                        int(index % 30 == 0 or index == 282),
                        "FCM" if article == 1 else "Pseudo",
                        890 if article == 1 else 900,
                    )
                    for index, date in enumerate(dates)
                )
        self.con.executemany(
            "INSERT INTO benchmark_daily_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.design = BenchmarkDesign(61, pd.Timestamp("2025-01-06"), 7, 7)
        create_history_features(self.con)
        _create_assessed_origins(
            self.con, pd.DatetimeIndex([self.design.first_origin]), self.design
        )

    def test_default_design_has_five_expanding_four_week_refits(self) -> None:
        config = GlobalLightGBMConfig()
        history = pd.date_range("2024-01-01", periods=52, freq="7D")
        evaluation = pd.date_range("2024-12-30", periods=20, freq="7D")

        windows = tuple(
            iter_lightgbm_origin_windows(history, evaluation, config)
        )

        self.assertEqual(config.training_origins, 48)
        self.assertEqual(config.validation_origins, 4)
        self.assertEqual(config.test_origins, 20)
        self.assertEqual(config.refit_interval_days, 28)
        self.assertEqual(len(windows), 5)
        self.assertEqual(
            [len(window.training) for window in windows],
            [48, 52, 56, 60, 64],
        )
        self.assertEqual([len(window.validation) for window in windows], [4] * 5)
        self.assertEqual([len(window.evaluation) for window in windows], [4] * 5)
        self.assertTrue(
            all(
                (window.evaluation[1:] - window.evaluation[:-1]).days.tolist()
                == [7, 7, 7]
                for window in windows
            )
        )
        self.assertEqual(
            [window.evaluation.min() for window in windows],
            list(evaluation[::4]),
        )
        self.assertEqual(list(windows[0].training), list(history[:48]))
        self.assertEqual(list(windows[0].validation), list(history[48:]))
        self.assertEqual(list(windows[1].validation), list(evaluation[:4]))
        self.assertEqual(windows[1].training.min(), history[0])
        self.assertEqual(list(windows[1].training), list(history))

    def test_features_are_anchored_strictly_before_origin(self) -> None:
        origin = pd.Timestamp("2024-10-07")
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET is_active = FALSE, reason_closed = 'test closure'
            WHERE period = DATE '2024-10-05'
            """
        )
        create_feature_tables(self.con)
        frame = make_feature_frame(self.con, [origin], self.design)
        first = frame.loc[
            frame.ARTIKEL_ID.eq(1) & frame.MARKT_ID.eq(10)
        ].iloc[0]
        history = self.con.execute(
            """
            SELECT period, demand, is_active FROM benchmark_daily_rows
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10 AND period < ?
            ORDER BY period
            """,
            [origin.date()],
        ).fetchdf()
        active_history = history.loc[history["is_active"]].reset_index(drop=True)
        sale_positions = np.flatnonzero(active_history["demand"].to_numpy() > 0)
        completed_gaps = np.diff(sale_positions) - 1
        current_gap = len(active_history) - int(sale_positions[-1]) - 1

        self.assertEqual(set(FEATURE_COLUMNS) - set(frame.columns), set())
        self.assertEqual(set(FEATURE_DESCRIPTIONS), set(FEATURE_COLUMNS))
        self.assertNotIn("days_since_last_positive_sale", frame.columns)
        self.assertNotIn("maturity_segment", frame.columns)
        removed_features = {
            "recent_mean_28_forecast",
            "same_weekday_ma_4_forecast",
            "occurrence_positive_quantity_forecast",
            "rolling_28_positive_mean",
            "days_since_first_sale",
            "demand_sum_last_7",
            "demand_sum_last_28",
            "demand_sum_last_60",
            "horizon_day",
            "lag_1",
            "lag_7",
            "lag_14",
            "zero_share",
        }
        self.assertTrue(removed_features.isdisjoint(frame.columns))
        self.assertTrue(
            {"is_active", "reason_closed", "is_public_holiday"}.isdisjoint(
                FEATURE_COLUMNS
            )
        )
        self.assertTrue(
            {"is_active", "reason_closed"}.issubset(FORECAST_ID_COLUMNS)
        )
        self.assertIn("is_public_holiday", DIAGNOSTIC_COLUMNS)
        self.assertEqual(
            first.same_weekday_lag_7,
            history.loc[history["period"].eq(origin - pd.Timedelta(days=7)), "demand"]
            .iloc[0],
        )
        self.assertEqual(
            first.same_weekday_lag_14,
            history.loc[
                history["period"].eq(origin - pd.Timedelta(days=14)), "demand"
            ].iloc[0],
        )
        self.assertAlmostEqual(
            first.rolling_28_mean, history["demand"].iloc[-28:].mean()
        )
        self.assertAlmostEqual(
            first[TARGET_SCALE_COLUMN], active_history["demand"].mean()
        )
        self.assertAlmostEqual(
            first[NORMALIZED_TARGET_COLUMN],
            first.actual / active_history["demand"].mean(),
        )
        for days in (7, 28, 60):
            trailing = history.iloc[-days:]
            self.assertEqual(
                first[f"demand_days_last_{days}"],
                int((trailing["is_active"] & trailing["demand"].gt(0)).sum()),
            )
        self.assertEqual(first.active_zero_demand_gap, current_gap)
        historical_p90_gap = float(np.quantile(completed_gaps, 0.9))
        self.assertAlmostEqual(first.historical_p90_gap, historical_p90_gap)
        self.assertAlmostEqual(
            first.current_gap_over_historical_p90_gap,
            current_gap / historical_p90_gap,
        )
        weekly = make_weekly_frame(frame)
        self.assertEqual(set(WEEKLY_FEATURE_COLUMNS) - set(weekly.columns), set())
        self.assertEqual(first.days_since_last_action, 10)
        self.assertEqual(first.actions_last_28d, 1)
        self.assertEqual(frame["action_during_horizon"].unique().tolist(), [1])
        self.assertEqual(
            frame.loc[
                frame["period"].eq(origin + pd.Timedelta(days=2)),
                "action_on_forecast_day",
            ]
            .unique()
            .tolist(),
            [1],
        )
        self.assertTrue(frame["mean_action_lift_in_sourcing_group"].notna().all())
        self.assertNotIn("demand_class", frame.columns)

    def test_annual_features_use_calendar_aligned_history(self) -> None:
        origin = pd.Timestamp("2025-02-03")
        create_feature_tables(self.con)
        frame = make_feature_frame(self.con, [origin], self.design)
        first = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.eq(origin)
        ].iloc[0]
        history = self.con.execute(
            """
            SELECT period, demand
            FROM benchmark_daily_rows
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
            ORDER BY period
            """
        ).fetchdf()
        demand_by_date = history.set_index("period")["demand"]
        annual_reference = origin - pd.Timedelta(days=364)
        corresponding_weekdays = [
            annual_reference + pd.Timedelta(days=offset)
            for offset in (-14, -7, 0, 7, 14)
        ]
        prior_week = pd.date_range(annual_reference, periods=7, freq="D")

        self.assertEqual(first.has_annual_history, 1)
        self.assertEqual(first.lag_364, demand_by_date.loc[annual_reference])
        self.assertEqual(
            first.lag_371,
            demand_by_date.loc[origin - pd.Timedelta(days=371)],
        )
        self.assertAlmostEqual(
            first.same_weekday_last_year_mean,
            demand_by_date.loc[corresponding_weekdays].mean(),
        )
        self.assertAlmostEqual(
            first.same_week_last_year_mean,
            demand_by_date.loc[prior_week].mean(),
        )

        cross_store = self.con.execute(
            """
            SELECT AVG(daily_mean)
            FROM (
                SELECT period, AVG(demand) AS daily_mean
                FROM benchmark_daily_rows
                WHERE ARTIKEL_ID = 1
                    AND period IN (?, ?, ?, ?, ?)
                    AND is_active
                GROUP BY period
            )
            """,
            [date.date() for date in corresponding_weekdays],
        ).fetchone()[0]
        self.assertAlmostEqual(
            first.product_cross_store_same_weekday_last_year_mean,
            cross_store,
        )

        event_reference = self.con.execute(
            """
            SELECT previous_event_offset_date
            FROM ml_calendar
            WHERE period = ?
            """,
            [origin.date()],
        ).fetchone()[0]
        event_mean = self.con.execute(
            """
            SELECT AVG(demand)
            FROM benchmark_daily_rows
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period BETWEEN ? - INTERVAL 3 DAY AND ? + INTERVAL 3 DAY
            """,
            [event_reference, event_reference],
        ).fetchone()[0]
        self.assertAlmostEqual(first.same_event_offset_last_year_mean, event_mean)

    def test_feature_materialization_writes_origin_batches(self) -> None:
        origins = pd.DatetimeIndex(["2025-01-06", "2025-01-13"])
        create_feature_tables(self.con)

        result = materialize_features_for_origins(
            self.con,
            origins=origins,
            design=self.design,
            feature_path=self.feature_dataset_path,
            origin_batch_size=1,
            return_frame=False,
        )

        self.assertIsNone(result)
        materialized = pd.read_parquet(self.feature_dataset_path)
        self.assertEqual(
            pd.DatetimeIndex(materialized["origin"].unique()).sort_values().tolist(),
            origins.tolist(),
        )
        self.assertFalse(materialized.empty)

    def test_model_prediction_is_restored_to_each_series_scale(self) -> None:
        class Booster:
            best_iteration = 0

            @staticmethod
            def predict(
                matrix: pd.DataFrame, *, num_iteration: int | None
            ) -> np.ndarray:
                return np.array([0.5, 2.0])

        model = BaseLightGBMModel(
            booster=Booster(),  # type: ignore[arg-type]
            category_levels={},
            feature_columns=(),
            categorical_features=(),
            prediction_scale_column=TARGET_SCALE_COLUMN,
        )
        frame = pd.DataFrame({TARGET_SCALE_COLUMN: [4.0, 3.0]})

        self.assertTrue(np.array_equal(model.predict(frame), np.array([2.0, 6.0])))

    def test_end_to_end_model_returns_nonnegative_matching_rows(self) -> None:
        result = run_global_lightgbm(
            design=self.design,
            evaluation_origins=[self.design.first_origin],
            connection=self.con,
            feature_dataset_path=self.feature_dataset_path,
            config=GlobalLightGBMConfig(
                training_origins=4,
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

    def test_origin_splits_expand_and_refit_after_four_weekly_forecasts(self) -> None:
        evaluation_origins = pd.date_range(
            self.design.first_origin
            + pd.Timedelta(days=2 * self.design.origin_spacing_days),
            periods=5,
            freq=f"{self.design.origin_spacing_days}D",
        )
        frames = prepare_global_lightgbm_frames(
            design=self.design,
            evaluation_origins=evaluation_origins,
            connection=self.con,
            feature_dataset_path=self.feature_dataset_path,
            config=GlobalLightGBMConfig(
                training_origins=4,
                validation_origins=2,
                refit_interval_origins=4,
                num_boost_round=5,
                early_stopping_rounds=2,
                num_threads=2,
            ),
        )

        self.assertEqual(len(frames.origin_frames), 2)
        self.assertEqual(
            [frame.training["origin"].nunique() for frame in frames.origin_frames],
            [4, 8],
        )
        self.assertEqual(
            [frame.validation["origin"].nunique() for frame in frames.origin_frames],
            [2, 2],
        )
        self.assertEqual(
            [len(frame.training_origins) for frame in frames.origin_frames],
            [6, 10],
        )
        self.assertEqual(
            [frame.evaluation["origin"].nunique() for frame in frames.origin_frames],
            [4, 1],
        )
        self.assertEqual(
            [frame.evaluation_origin for frame in frames.origin_frames],
            [evaluation_origins[0], evaluation_origins[4]],
        )
        self.assertEqual(
            (
                frames.origin_frames[1].evaluation_origin
                - frames.origin_frames[0].evaluation_origin
            ).days,
            28,
        )
        for frame in frames.origin_frames:
            self.assertLess(
                pd.to_datetime(frame.validation["origin"]).max(),
                pd.to_datetime(frame.evaluation["origin"]).min(),
            )

    def test_all_variants_match_rows_and_weekly_allocation_conserves_total(self) -> None:
        config = GlobalLightGBMConfig(
            training_origins=4,
            validation_origins=1,
            refit_interval_origins=2,
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
            feature_dataset_path=self.feature_dataset_path,
            config=config,
        )
        results = fit_all_lightgbm_models(frames, config)

        self.assertEqual(len(results), 4)
        for model_name, result in results.items():
            self.assertEqual(len(result.forecasts), len(frames.evaluation))
            self.assertTrue(result.forecasts.forecast.ge(0).all())
            expected_summary_rows = 2 if model_name == TWO_STAGE_MODEL_NAME else 1
            self.assertEqual(len(result.training_summary), expected_summary_rows)
            self.assertEqual(result.training_summary.loc[0, "evaluation_origins"], 2)
        weekly_audit = results[WEEKLY_MODEL_NAME].allocation_audit
        self.assertIsNotNone(weekly_audit)
        assert weekly_audit is not None
        self.assertLess(weekly_audit.allocation_error.max(), 1e-9)
        self.assertTrue(np.allclose(weekly_audit.weekday_share_sum, 1.0))


if __name__ == "__main__":
    unittest.main()
