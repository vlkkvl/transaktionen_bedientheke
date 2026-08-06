from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

from src.models.benchmark.config import BenchmarkDesign
from src.models.benchmark.evaluation import _create_assessed_origins
from src.models.benchmark.models import create_history_features
from src.models.lightgbm import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    DIRECT_FEATURE_COLUMNS,
    MODEL_NAME,
    NORMALIZED_TARGET_COLUMN,
    OCCURRENCE_FEATURE_COLUMNS,
    QUANTITY_FEATURE_COLUMNS,
    TARGET_SCALE_COLUMN,
    TWEEDIE_MODEL_NAME,
    TWO_STAGE_MODEL_NAME,
    WEEKLY_MODEL_NAME,
    GlobalLightGBMConfig,
    create_feature_tables,
    fit_all_lightgbm_models,
    fit_two_stage,
    make_feature_frame,
    make_weekly_frame,
    prepare_global_lightgbm_frames,
    run_global_lightgbm,
)
from src.models.lightgbm.base import BaseLightGBMModel
from src.models.lightgbm.features.builder import (
    EVENT_MIN_ARTICLE_BASELINE_OBSERVATIONS,
    EVENT_MIN_POOLED_CELL_DATES,
    _annual_anchor_calendar,
    _holiday_calendar,
    anchor_date,
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


EXPECTED_NI_HOLIDAYS = {
    "2023-01-01": "Neujahr",
    "2023-04-07": "Karfreitag",
    "2023-04-10": "Ostermontag",
    "2023-05-01": "Erster Mai",
    "2023-05-18": "Christi Himmelfahrt",
    "2023-05-29": "Pfingstmontag",
    "2023-10-03": "Tag der Deutschen Einheit",
    "2023-10-31": "Reformationstag",
    "2023-12-25": "Erster Weihnachtstag",
    "2023-12-26": "Zweiter Weihnachtstag",
    "2024-01-01": "Neujahr",
    "2024-03-29": "Karfreitag",
    "2024-04-01": "Ostermontag",
    "2024-05-01": "Erster Mai",
    "2024-05-09": "Christi Himmelfahrt",
    "2024-05-20": "Pfingstmontag",
    "2024-10-03": "Tag der Deutschen Einheit",
    "2024-10-31": "Reformationstag",
    "2024-12-25": "Erster Weihnachtstag",
    "2024-12-26": "Zweiter Weihnachtstag",
    "2025-01-01": "Neujahr",
    "2025-04-18": "Karfreitag",
    "2025-04-21": "Ostermontag",
    "2025-05-01": "Erster Mai",
    "2025-05-29": "Christi Himmelfahrt",
    "2025-06-09": "Pfingstmontag",
    "2025-10-03": "Tag der Deutschen Einheit",
    "2025-10-31": "Reformationstag",
    "2025-12-25": "Erster Weihnachtstag",
    "2025-12-26": "Zweiter Weihnachtstag",
    "2026-01-01": "Neujahr",
    "2026-04-03": "Karfreitag",
    "2026-04-06": "Ostermontag",
    "2026-05-01": "Erster Mai",
    "2026-05-14": "Christi Himmelfahrt",
    "2026-05-25": "Pfingstmontag",
    "2026-10-03": "Tag der Deutschen Einheit",
    "2026-10-31": "Reformationstag",
    "2026-12-25": "Erster Weihnachtstag",
    "2026-12-26": "Zweiter Weihnachtstag",
}
EXPECTED_MOTHERS_DAYS = {
    "2023-05-14": "Muttertag",
    "2024-05-12": "Muttertag",
    "2025-05-11": "Muttertag",
    "2026-05-10": "Muttertag",
}
EXPECTED_EVENT_NAMES = {
    "none",
    "Neujahr",
    "Karfreitag",
    "Ostermontag",
    "Erster Mai",
    "Christi Himmelfahrt",
    "Pfingstmontag",
    "Tag der Deutschen Einheit",
    "Reformationstag",
    "Erster Weihnachtstag",
    "Zweiter Weihnachtstag",
    "Muttertag",
}


class HolidayCalendarTest(unittest.TestCase):
    START = pd.Timestamp("2023-07-22")
    END = pd.Timestamp("2026-07-21")

    def test_public_holiday_dates_match_known_lower_saxony_dates(self) -> None:
        calendar = _holiday_calendar("2023-01-01", "2026-12-31")
        actual = {
            str(row.period): row.event_name
            for row in calendar.loc[calendar.is_public_holiday].itertuples()
        }
        self.assertEqual(actual, EXPECTED_NI_HOLIDAYS)

    def test_event_name_categories_are_complete_and_exact(self) -> None:
        calendar = _holiday_calendar("2023-01-01", "2026-12-31")
        self.assertEqual(set(calendar.event_name), EXPECTED_EVENT_NAMES)

    def test_calendar_invariants_and_sign_convention(self) -> None:
        calendar = _holiday_calendar(self.START, self.END)
        near_event = calendar.days_to_nearest_event.abs().le(3)
        in_event_window = calendar.holiday_event_window.ne("none")
        named_event = calendar.event_name.ne("none")
        pd.testing.assert_series_equal(
            near_event,
            in_event_window,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            in_event_window,
            named_event,
            check_names=False,
        )

        by_date = calendar.set_index("period")
        expected = {
            "2026-04-30": (1, "before_holiday_1_3d", "Erster Mai"),
            "2026-05-02": (-1, "after_holiday_1_3d", "Erster Mai"),
            "2026-05-09": (1, "before_holiday_1_3d", "Muttertag"),
            "2026-05-12": (-2, "after_holiday_1_3d", "Muttertag"),
        }
        for date, values in expected.items():
            with self.subTest(date=date):
                row = by_date.loc[pd.Timestamp(date).date()]
                self.assertEqual(
                    (
                        row.days_to_nearest_event,
                        row.holiday_event_window,
                        row.event_name,
                    ),
                    values,
                )

    def test_all_equidistant_event_ties_resolve_to_earlier_date(self) -> None:
        calendar = _holiday_calendar(self.START, self.END).set_index("period")
        events = {
            pd.Timestamp(date): name
            for date, name in {
                **EXPECTED_NI_HOLIDAYS,
                **EXPECTED_MOTHERS_DAYS,
            }.items()
        }
        tie_count = 0
        for target in pd.date_range(self.START, self.END):
            distances = {
                event_date: abs((event_date - target).days)
                for event_date in events
            }
            nearest_distance = min(distances.values())
            nearest_dates = sorted(
                event_date
                for event_date, distance in distances.items()
                if distance == nearest_distance
            )
            if len(nearest_dates) == 1:
                continue
            self.assertEqual(len(nearest_dates), 2)
            tie_count += 1
            expected_event_date = nearest_dates[0]
            expected_offset = int((expected_event_date - target).days)
            row = calendar.loc[target.date()]
            with self.subTest(target=str(target.date())):
                self.assertEqual(row.days_to_nearest_event, expected_offset)
                if nearest_distance <= 3:
                    self.assertEqual(row.event_name, events[expected_event_date])
        self.assertEqual(tie_count, 17)


class GlobalLightGBMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_min_article_baseline_observations = 30
        article_threshold = patch(
            "src.models.lightgbm.features.builder."
            "EVENT_MIN_ARTICLE_BASELINE_OBSERVATIONS",
            self.test_min_article_baseline_observations,
        )
        article_threshold.start()
        self.addCleanup(article_threshold.stop)
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
        dates = pd.date_range("2024-01-01", periods=600, freq="D")
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

    def test_event_lift_thresholds_match_their_aggregation_levels(self) -> None:
        self.assertEqual(EVENT_MIN_ARTICLE_BASELINE_OBSERVATIONS, 30)
        self.assertEqual(EVENT_MIN_POOLED_CELL_DATES, 4)

    def test_features_are_anchored_strictly_before_origin(self) -> None:
        origin = pd.Timestamp("2024-10-07")
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET is_active = FALSE, reason_closed = 'test closure'
            WHERE MARKT_ID = 10
                AND period IN (
                    DATE '2024-10-05',
                    DATE '2024-10-08',
                    DATE '2024-10-10'
                )
            """
        )
        create_feature_tables(self.con, origins=[origin], design=self.design)
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
        positive_history = active_history.loc[active_history["demand"].gt(0)]
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
            "lag_364",
            "lag_371",
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
            first[
                [
                    "closed_days_next_1",
                    "closed_days_next_2",
                    "closed_days_next_3",
                    "closed_days_prev_1",
                    "closed_days_prev_2",
                    "closed_days_prev_3",
                ]
            ].tolist(),
            [1, 1, 2, 0, 1, 1],
        )
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
        self.assertAlmostEqual(first.rolling_6_mean, active_history.iloc[-6:].demand.mean())
        self.assertAlmostEqual(
            first.rolling_24_mean,
            active_history.iloc[-24:].demand.mean(),
        )
        self.assertAlmostEqual(
            first[TARGET_SCALE_COLUMN], positive_history["demand"].mean()
        )
        self.assertAlmostEqual(
            first[NORMALIZED_TARGET_COLUMN],
            first.actual / positive_history["demand"].mean(),
        )
        for active_rows in (6, 12, 24):
            trailing = active_history.iloc[-active_rows:]
            self.assertAlmostEqual(
                first[f"demand_rate_last_{active_rows}"],
                trailing["demand"].gt(0).mean(),
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
        self.assertTrue({"lag_364", "lag_371"}.isdisjoint(WEEKLY_FEATURE_COLUMNS))
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

    def test_lags_and_rolling_means_exclude_inactive_calendar_rows(self) -> None:
        origin = pd.Timestamp("2024-10-07")
        window_start = origin - pd.Timedelta(days=28)
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 10.0, is_active = TRUE, reason_closed = NULL
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period >= ? AND period < ?
            """,
            [window_start.date(), origin.date()],
        )
        inactive_lookup_dates = [
            origin - pd.Timedelta(days=7),
            origin - pd.Timedelta(days=14),
        ]
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 123.0, is_active = FALSE, reason_closed = 'test closure'
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period IN (?, ?)
            """,
            [date.date() for date in inactive_lookup_dates],
        )

        create_feature_tables(self.con, origins=[origin], design=self.design)
        frame = make_feature_frame(self.con, [origin], self.design)
        first = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.eq(origin)
        ].iloc[0]

        self.assertTrue(pd.isna(first.same_weekday_lag_7))
        self.assertTrue(pd.isna(first.same_weekday_lag_14))
        self.assertAlmostEqual(first.rolling_6_mean, 10.0)
        self.assertAlmostEqual(first.rolling_24_mean, 10.0)

    def test_demand_rates_use_fixed_active_row_windows(self) -> None:
        origin = pd.Timestamp("2024-10-07")
        window_start = origin - pd.Timedelta(days=35)
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 0.0, is_active = TRUE, reason_closed = NULL
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period >= ? AND period < ?
            """,
            [window_start.date(), origin.date()],
        )
        inactive_dates = [
            origin - pd.Timedelta(days=1),
            origin - pd.Timedelta(days=4),
            origin - pd.Timedelta(days=11),
        ]
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 999.0, is_active = FALSE, reason_closed = 'test closure'
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period IN (?, ?, ?)
            """,
            [date.date() for date in inactive_dates],
        )
        active_dates = self.con.execute(
            """
            SELECT period
            FROM benchmark_daily_rows
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period < ? AND is_active
            ORDER BY period DESC
            LIMIT 24
            """,
            [origin.date()],
        ).fetchdf()["period"]
        positive_positions = (1, 3, 6, 7, 12, 13, 20, 24)
        positive_dates = [active_dates.iloc[position - 1] for position in positive_positions]
        self.con.execute(
            f"""
            UPDATE benchmark_daily_rows
            SET demand = 5.0
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period IN ({", ".join("?" for _ in positive_dates)})
            """,
            [date.date() for date in positive_dates],
        )

        create_feature_tables(self.con, origins=[origin], design=self.design)
        frame = make_feature_frame(self.con, [origin], self.design)
        first = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.eq(origin)
        ].iloc[0]

        self.assertAlmostEqual(first.demand_rate_last_6, 3 / 6)
        self.assertAlmostEqual(first.demand_rate_last_12, 5 / 12)
        self.assertAlmostEqual(first.demand_rate_last_24, 8 / 24)
        self.assertAlmostEqual(first.rolling_6_mean, 15 / 6)
        self.assertAlmostEqual(first.rolling_24_mean, 40 / 24)

    def test_same_weekday_means_exclude_event_window_dates(self) -> None:
        origin = pd.Timestamp("2025-01-06")
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 10.0, is_active = TRUE, reason_closed = NULL
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period < ?
                AND EXTRACT(ISODOW FROM period) = 1
            """,
            [origin.date()],
        )
        anomalous_mondays = [
            pd.Timestamp("2024-12-23"),
            pd.Timestamp("2024-12-30"),
        ]
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 999.0
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period IN (?, ?)
            """,
            [date.date() for date in anomalous_mondays],
        )

        create_feature_tables(self.con, origins=[origin], design=self.design)
        contexts = self.con.execute(
            """
            SELECT holiday_event_window
            FROM ml_calendar
            WHERE period IN (?, ?)
            ORDER BY period
            """,
            [date.date() for date in anomalous_mondays],
        ).fetchall()
        frame = make_feature_frame(self.con, [origin], self.design)
        row = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.eq(origin)
        ].iloc[0]

        self.assertTrue(all(window != "none" for (window,) in contexts))
        self.assertEqual(row.same_weekday_mean_4, 10.0)
        self.assertEqual(row.same_weekday_mean_8, 10.0)

    def test_product_weekday_profile_uses_open_non_event_cross_store_means(
        self,
    ) -> None:
        origin = pd.Timestamp("2025-01-06")
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 100000.0
            WHERE ARTIKEL_ID = 1
                AND period BETWEEN DATE '2024-12-22' AND DATE '2025-01-04'
            """
        )
        closed_dates = [pd.Timestamp("2024-11-30"), pd.Timestamp("2024-12-07")]
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 100000.0,
                is_active = FALSE,
                reason_closed = 'test closure'
            WHERE ARTIKEL_ID = 1 AND period IN (?, ?)
            """,
            [date.date() for date in closed_dates],
        )

        create_feature_tables(self.con, origins=[origin], design=self.design)
        frame = make_feature_frame(self.con, [origin], self.design)
        eligible = self.con.execute(
            """
            SELECT
                product.period,
                EXTRACT(ISODOW FROM product.period)::INTEGER AS target_weekday,
                AVG(product.demand) FILTER (WHERE product.is_active)
                    AS cross_store_mean
            FROM benchmark_daily_rows AS product
            INNER JOIN ml_calendar AS calendar ON product.period = calendar.period
            WHERE product.ARTIKEL_ID = 1
                AND product.period < ?
                AND calendar.holiday_event_window = 'none'
            GROUP BY product.period
            HAVING COUNT_IF(product.is_active) > 0
            ORDER BY product.period
            """,
            [origin.date()],
        ).fetchdf()
        denominator = eligible.tail(48)["cross_store_mean"].mean()
        article_rows = frame.loc[frame.ARTIKEL_ID.eq(1)]
        for row in article_rows.itertuples():
            numerator = (
                eligible.loc[eligible.target_weekday.eq(row.target_weekday)]
                .tail(8)["cross_store_mean"]
                .mean()
            )
            self.assertAlmostEqual(
                row.product_weekday_profile_value,
                numerator / denominator,
            )

        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 0.0
            WHERE ARTIKEL_ID = 1 AND period < ?
            """,
            [origin.date()],
        )
        create_feature_tables(self.con, origins=[origin], design=self.design)
        rebuilt = make_feature_frame(self.con, [origin], self.design)
        self.assertTrue(
            rebuilt.loc[
                rebuilt.ARTIKEL_ID.eq(1), "product_weekday_profile_value"
            ].isna().all()
        )

    def test_annual_features_use_calendar_aligned_history(self) -> None:
        origin = pd.Timestamp("2025-04-07")
        annual_reference = anchor_date(origin)
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 100.0
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period BETWEEN ? - INTERVAL 14 DAY AND ? + INTERVAL 14 DAY
            """,
            [annual_reference.date(), annual_reference.date()],
        )
        create_feature_tables(self.con, origins=[origin], design=self.design)
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
        corresponding_weekdays = [
            annual_reference + pd.Timedelta(days=offset)
            for offset in (-14, -7, 0, 7, 14)
        ]
        prior_week = pd.date_range(annual_reference, periods=7, freq="D")

        self.assertEqual((origin - annual_reference).days, 385)
        self.assertGreaterEqual(first.annual_lookup_days_available, 1)
        self.assertTrue({"lag_364", "lag_371"}.isdisjoint(frame.columns))
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
                SELECT history.period, AVG(history.demand) AS daily_mean
                FROM benchmark_daily_rows AS history
                INNER JOIN ml_calendar AS calendar
                    ON history.period = calendar.period
                WHERE history.ARTIKEL_ID = 1
                    AND history.period IN (?, ?, ?, ?, ?)
                    AND history.is_active
                    AND calendar.holiday_event_window = 'none'
                GROUP BY history.period
            )
            """,
            [date.date() for date in corresponding_weekdays],
        ).fetchone()[0]
        self.assertAlmostEqual(
            first.product_cross_store_same_weekday_last_year_mean,
            cross_store,
        )

        self.assertEqual(first.annual_lookup_days_available, 4)
        self.assertTrue(pd.isna(first.event_lift_series))
        self.assertTrue(pd.isna(first.event_lift_pooled_occurrence))
        self.assertTrue(pd.isna(first.event_lift_pooled_quantity))
        self.assertTrue(pd.isna(first.event_lift_pooled_total))

    def test_annual_means_exclude_inactive_lookup_dates(self) -> None:
        origin = pd.Timestamp("2025-04-07")
        annual_reference = anchor_date(origin)
        corresponding_weekdays = [
            annual_reference + pd.Timedelta(days=offset)
            for offset in (-14, -7, 0, 7, 14)
        ]
        reference_week = pd.date_range(annual_reference, periods=7, freq="D")

        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 10.0, is_active = TRUE, reason_closed = NULL
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period BETWEEN ? - INTERVAL 14 DAY AND ? + INTERVAL 14 DAY
            """,
            [
                annual_reference.date(),
                annual_reference.date(),
            ],
        )
        inactive_dates = {
            annual_reference,
            annual_reference + pd.Timedelta(days=2),
            annual_reference + pd.Timedelta(days=7),
        }
        self.con.execute(
            f"""
            UPDATE benchmark_daily_rows
            SET demand = 0.0, is_active = FALSE, reason_closed = 'test closure'
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period IN ({", ".join("?" for _ in inactive_dates)})
            """,
            [date.date() for date in sorted(inactive_dates)],
        )

        create_feature_tables(self.con, origins=[origin], design=self.design)
        frame = make_feature_frame(self.con, [origin], self.design)
        annual_row = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.eq(origin)
        ].iloc[0]
        weekday_mean, active_weekdays = self.con.execute(
            f"""
            SELECT
                AVG(history.demand) FILTER (WHERE history.is_active),
                COUNT_IF(history.is_active)
            FROM benchmark_daily_rows AS history
            INNER JOIN ml_calendar AS calendar ON history.period = calendar.period
            WHERE history.ARTIKEL_ID = 1 AND history.MARKT_ID = 10
                AND history.period IN ({", ".join("?" for _ in corresponding_weekdays)})
                AND calendar.holiday_event_window = 'none'
            """,
            [date.date() for date in corresponding_weekdays],
        ).fetchone()
        week_mean, active_week_dates = self.con.execute(
            """
            SELECT AVG(demand) FILTER (WHERE is_active), COUNT_IF(is_active)
            FROM benchmark_daily_rows
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period BETWEEN ? AND ?
            """,
            [reference_week.min().date(), reference_week.max().date()],
        ).fetchone()
        self.assertEqual(active_weekdays, 2)
        self.assertLess(active_week_dates, 7)
        self.assertEqual(annual_row.annual_lookup_days_available, active_weekdays)
        self.assertAlmostEqual(annual_row.same_weekday_last_year_mean, weekday_mean)
        self.assertAlmostEqual(annual_row.same_week_last_year_mean, week_mean)

    def test_fixed_event_anchor_uses_one_date_and_event_lifts(self) -> None:
        origin = pd.Timestamp("2025-05-05")
        target = pd.Timestamp("2025-05-10")
        annual_reference = anchor_date(target)
        self.assertEqual(annual_reference, pd.Timestamp("2024-05-11"))

        previous_active_dates = self.con.execute(
            """
            SELECT period
            FROM benchmark_daily_rows
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period < ? AND is_active
            ORDER BY period DESC
            LIMIT 24
            """,
            [annual_reference.date()],
        ).fetchdf()["period"]
        self.con.execute(
            f"""
            UPDATE benchmark_daily_rows
            SET demand = 10.0
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period IN ({", ".join("?" for _ in previous_active_dates)})
            """,
            [date.date() for date in previous_active_dates],
        )
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 60.0, is_active = TRUE, reason_closed = NULL
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10 AND period = ?
            """,
            [annual_reference.date()],
        )
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 999.0
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                AND period IN (? - INTERVAL 14 DAY, ? - INTERVAL 7 DAY,
                    ? + INTERVAL 7 DAY, ? + INTERVAL 14 DAY)
            """,
            [annual_reference.date()] * 4,
        )
        thin_article_dates = pd.date_range("2025-01-01", "2025-05-04", freq="D")
        self.con.executemany(
            "INSERT INTO benchmark_daily_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    3,
                    12,
                    date.date(),
                    20.0 if index % 3 == 0 else 0.0,
                    True,
                    None,
                    0,
                    "FCM",
                    890,
                )
                for index, date in enumerate(thin_article_dates)
            ],
        )

        create_feature_tables(self.con, origins=[origin], design=self.design)
        frame = make_feature_frame(self.con, [origin], self.design)
        row = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.eq(target)
        ].iloc[0]
        baseline_mean = self.con.execute(
            """
            SELECT AVG(demand)
            FROM (
                SELECT demand
                FROM benchmark_daily_rows
                WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10
                    AND period < ? AND is_active
                ORDER BY period DESC
                LIMIT 24
            )
            """,
            [annual_reference.date()],
        ).fetchone()[0]

        self.assertEqual(row.annual_lookup_days_available, 1)
        self.assertEqual(row.same_weekday_last_year_mean, 60.0)
        self.assertTrue(pd.isna(row.same_week_last_year_mean))
        self.assertAlmostEqual(row.event_lift_series, 60.0 / baseline_mean)
        self.assertTrue(pd.notna(row.event_lift_pooled_occurrence))
        self.assertTrue(pd.notna(row.event_lift_pooled_quantity))
        self.assertAlmostEqual(
            row.event_lift_pooled_total,
            row.event_lift_pooled_occurrence * row.event_lift_pooled_quantity,
        )

        expected_occurrence, expected_quantity = self.con.execute(
            f"""
            WITH active_history AS (
                SELECT
                    ARTIKEL_ID,
                    MARKT_ID,
                    sourcing_group,
                    category_id,
                    period,
                    demand,
                    COUNT(*) OVER previous_24 AS previous_active_days
                FROM benchmark_daily_rows
                WHERE is_active
                WINDOW previous_24 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
                )
            ),
            params AS (
                SELECT ?::DATE AS target, ?::DATE AS origin
            ),
            target_context AS (
                SELECT
                    calendar_event_key,
                    calendar_event_closure_block_length
                FROM ml_calendar
                CROSS JOIN params
                WHERE period = params.target
            ),
            article_baseline AS (
                SELECT
                    EXTRACT(ISODOW FROM history.period)::INTEGER AS weekday,
                    history.ARTIKEL_ID,
                    COUNT(*) AS observation_count,
                    COUNT_IF(history.demand > 0) AS positive_count,
                    SUM(history.demand) FILTER (WHERE history.demand > 0)
                        AS positive_sum
                FROM active_history AS history
                INNER JOIN ml_calendar AS calendar
                    ON history.period = calendar.period
                CROSS JOIN params
                WHERE history.period < params.origin
                    AND history.previous_active_days = 24
                    AND calendar.holiday_event_window = 'none'
                GROUP BY weekday, history.ARTIKEL_ID
            ),
            coarse_baseline AS (
                SELECT
                    EXTRACT(ISODOW FROM history.period)::INTEGER AS weekday,
                    history.sourcing_group,
                    history.category_id,
                    COUNT(*) AS observation_count,
                    COUNT_IF(history.demand > 0) AS positive_count,
                    SUM(history.demand) FILTER (WHERE history.demand > 0)
                        AS positive_sum
                FROM active_history AS history
                INNER JOIN ml_calendar AS calendar
                    ON history.period = calendar.period
                CROSS JOIN params
                WHERE history.period < params.origin
                    AND history.previous_active_days = 24
                    AND calendar.holiday_event_window = 'none'
                GROUP BY weekday, history.sourcing_group, history.category_id
            ),
            eligible_events AS (
                SELECT history.*, calendar.calendar_event_key
                FROM active_history AS history
                INNER JOIN ml_calendar AS calendar
                    ON history.period = calendar.period
                CROSS JOIN params
                CROSS JOIN target_context
                WHERE history.period < params.origin
                    AND history.previous_active_days = 24
                    AND calendar.holiday_event_window <> 'none'
                    AND calendar.calendar_event_key
                        = target_context.calendar_event_key
                    AND calendar.calendar_event_closure_block_length
                        = target_context.calendar_event_closure_block_length
            ),
            resolved AS (
                SELECT
                    event.*,
                    CASE
                        WHEN article.observation_count
                                >= {self.test_min_article_baseline_observations}
                            AND article.positive_count > 0
                        THEN article.positive_count::DOUBLE
                            / article.observation_count
                        ELSE coarse.positive_count::DOUBLE
                            / NULLIF(coarse.observation_count, 0)
                    END AS baseline_occurrence_rate,
                    CASE
                        WHEN article.observation_count
                                >= {self.test_min_article_baseline_observations}
                            AND article.positive_count > 0
                            AND article.positive_sum > 0
                        THEN article.positive_sum / article.positive_count
                        ELSE coarse.positive_sum
                            / NULLIF(coarse.positive_count, 0)
                    END AS baseline_positive_mean
                FROM eligible_events AS event
                LEFT JOIN article_baseline AS article
                    ON event.ARTIKEL_ID = article.ARTIKEL_ID
                    AND EXTRACT(ISODOW FROM event.period) = article.weekday
                LEFT JOIN coarse_baseline AS coarse
                    ON event.sourcing_group = coarse.sourcing_group
                    AND event.category_id = coarse.category_id
                    AND EXTRACT(ISODOW FROM event.period) = coarse.weekday
            )
            SELECT
                SUM(CASE WHEN baseline_occurrence_rate > 0
                    THEN (demand > 0)::INTEGER END)::DOUBLE
                    / SUM(CASE WHEN baseline_occurrence_rate > 0
                        THEN baseline_occurrence_rate END),
                SUM(CASE WHEN demand > 0 AND baseline_positive_mean > 0
                    THEN demand END)
                    / SUM(CASE WHEN demand > 0 AND baseline_positive_mean > 0
                        THEN baseline_positive_mean END)
            FROM resolved
            """,
            [target.date(), origin.date()],
        ).fetchone()
        self.assertAlmostEqual(
            row.event_lift_pooled_occurrence,
            expected_occurrence,
        )
        self.assertAlmostEqual(
            row.event_lift_pooled_quantity,
            expected_quantity,
        )
        audit = self.con.execute(
            """
            SELECT
                event_lift_pooled_cell_row_count,
                event_lift_pooled_cell_date_count
            FROM ml_origin_event_lifts
            WHERE origin = ? AND target_period = ?
            """,
            [origin.date(), target.date()],
        ).fetchone()
        self.assertGreater(audit[0], audit[1])
        self.assertGreaterEqual(audit[1], EVENT_MIN_POOLED_CELL_DATES)
        baseline_counts = self.con.execute(
            """
            WITH active_history AS (
                SELECT
                    ARTIKEL_ID,
                    MARKT_ID,
                    sourcing_group,
                    category_id,
                    period,
                    COUNT(*) OVER previous_24 AS previous_active_days
                FROM benchmark_daily_rows
                WHERE is_active
                WINDOW previous_24 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
                )
            ),
            eligible AS (
                SELECT history.*
                FROM active_history AS history
                INNER JOIN ml_calendar AS calendar USING (period)
                WHERE history.period < ?
                    AND history.previous_active_days = 24
                    AND calendar.holiday_event_window = 'none'
            ),
            article_counts AS (
                SELECT
                    EXTRACT(ISODOW FROM period) AS weekday,
                    COUNT(*) AS observation_count
                FROM eligible
                WHERE ARTIKEL_ID = 3
                GROUP BY weekday
            ),
            coarse_counts AS (
                SELECT
                    EXTRACT(ISODOW FROM period) AS weekday,
                    COUNT(*) AS observation_count
                FROM eligible
                WHERE sourcing_group = 'FCM' AND category_id = 890
                GROUP BY weekday
            )
            SELECT
                MAX(article_counts.observation_count),
                MIN(coarse_counts.observation_count)
            FROM article_counts
            INNER JOIN coarse_counts USING (weekday)
            """,
            [origin.date()],
        ).fetchone()
        self.assertLess(
            baseline_counts[0], self.test_min_article_baseline_observations
        )
        self.assertGreaterEqual(
            baseline_counts[1], self.test_min_article_baseline_observations
        )

        pooled_before = row[
            [
                "event_lift_pooled_occurrence",
                "event_lift_pooled_quantity",
                "event_lift_pooled_total",
            ]
        ].astype(float)
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = 1000000.0
            WHERE period >= ?
            """,
            [origin.date()],
        )
        create_feature_tables(self.con, origins=[origin], design=self.design)
        rebuilt = make_feature_frame(self.con, [origin], self.design)
        rebuilt_row = rebuilt.loc[
            rebuilt.ARTIKEL_ID.eq(1)
            & rebuilt.MARKT_ID.eq(10)
            & rebuilt.period.eq(target)
        ].iloc[0]
        pd.testing.assert_series_equal(
            rebuilt_row[
                [
                    "event_lift_pooled_occurrence",
                    "event_lift_pooled_quantity",
                    "event_lift_pooled_total",
                ]
            ].astype(float),
            pooled_before,
            check_names=False,
        )

    def test_pooled_event_lifts_recompute_at_each_forecast_origin(self) -> None:
        origins = pd.DatetimeIndex(["2025-04-28", "2025-04-29"])
        target = pd.Timestamp("2025-05-01")
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET demand = CASE
                WHEN period = DATE '2025-04-28' THEN 777.0
                ELSE 5.0
            END
            WHERE period IN (DATE '2025-04-14', DATE '2025-04-28')
            """
        )

        create_feature_tables(self.con, origins=origins, design=self.design)
        lifts = self.con.execute(
            """
            SELECT
                origin,
                event_lift_pooled_cell_row_count AS cell_rows,
                event_lift_pooled_cell_date_count AS cell_dates,
                event_lift_pooled_occurrence AS occurrence,
                event_lift_pooled_quantity AS quantity
            FROM ml_origin_event_lifts
            WHERE target_period = ?
            ORDER BY origin
            """,
            [target.date()],
        ).fetchdf()

        self.assertEqual(lifts["origin"].tolist(), list(origins))
        self.assertEqual(
            lifts.loc[1, "cell_rows"], lifts.loc[0, "cell_rows"] + 4
        )
        self.assertEqual(
            lifts.loc[1, "cell_dates"], lifts.loc[0, "cell_dates"] + 1
        )
        self.assertFalse(
            np.allclose(
                lifts.loc[0, ["occurrence", "quantity"]].astype(float),
                lifts.loc[1, ["occurrence", "quantity"]].astype(float),
            ),
            lifts,
        )

    def test_thin_event_cell_is_nan_without_cross_event_fallback(self) -> None:
        origin = pd.Timestamp("2025-05-05")
        target = pd.Timestamp("2025-05-10")
        calendar = _holiday_calendar("2024-01-01", origin)
        event_dates = sorted(
            calendar.loc[
                calendar["calendar_event_key"].eq("muttertag"), "period"
            ].tolist()
        )
        self.assertGreater(len(event_dates), 2)
        inactive_dates = event_dates[2:]
        self.con.execute(
            f"""
            UPDATE benchmark_daily_rows
            SET is_active = FALSE, reason_closed = 'thin event test'
            WHERE period IN ({", ".join("?" for _ in inactive_dates)})
            """,
            inactive_dates,
        )

        create_feature_tables(self.con, origins=[origin], design=self.design)
        audit = self.con.execute(
            """
            SELECT
                event_lift_pooled_cell_row_count,
                event_lift_pooled_cell_date_count,
                event_lift_pooled_occurrence,
                event_lift_pooled_quantity,
                event_lift_pooled_total
            FROM ml_origin_event_lifts
            WHERE origin = ? AND target_period = ?
            """,
            [origin.date(), target.date()],
        ).fetchone()
        other_event_dates = self.con.execute(
            """
            SELECT COUNT(DISTINCT period)
            FROM ml_calendar
            WHERE period < ?
                AND holiday_event_window <> 'none'
                AND calendar_event_key <> 'muttertag'
                AND calendar_event_closure_block_length = 1
            """,
            [origin.date()],
        ).fetchone()[0]

        self.assertEqual(audit[1], 2)
        self.assertGreater(audit[0], audit[1])
        self.assertGreaterEqual(other_event_dates, EVENT_MIN_POOLED_CELL_DATES)
        self.assertTrue(all(pd.isna(value) for value in audit[2:]))

    def test_pfingstmontag_event_lifts_are_not_gated_out(self) -> None:
        origin = pd.Timestamp("2025-06-02")
        target = pd.Timestamp("2025-06-06")
        self.assertEqual(anchor_date(target), pd.Timestamp("2024-05-17"))
        _create_assessed_origins(self.con, pd.DatetimeIndex([origin]), self.design)

        create_feature_tables(self.con, origins=[origin], design=self.design)
        frame = make_feature_frame(self.con, [origin], self.design)
        row = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.eq(target)
        ].iloc[0]
        context = self.con.execute(
            """
            SELECT holiday_event_window, event_name, calendar_event_key
            FROM ml_calendar
            WHERE period = ?
            """,
            [target.date()],
        ).fetchone()

        self.assertEqual(
            context,
            ("before_holiday_1_3d", "Pfingstmontag", "pfingstmontag"),
        )
        self.assertTrue(pd.notna(row.event_lift_series))
        self.assertTrue(pd.notna(row.event_lift_pooled_occurrence))
        self.assertTrue(pd.notna(row.event_lift_pooled_quantity))
        self.assertTrue(pd.notna(row.event_lift_pooled_total))

        pooled_columns = [
            "event_lift_pooled_occurrence",
            "event_lift_pooled_quantity",
            "event_lift_pooled_total",
        ]
        pfingst_context = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.between("2025-06-06", "2025-06-08"),
            pooled_columns,
        ]
        self.assertEqual(len(pfingst_context), 3)
        self.assertTrue(pfingst_context.notna().all().all())
        self.assertTrue(pfingst_context.nunique().eq(1).all())

    def test_annual_candidates_must_match_target_event_context(self) -> None:
        origin = pd.Timestamp("2025-04-14")
        target = pd.Timestamp("2025-04-15")
        annual_reference = anchor_date(target)
        self.assertEqual(annual_reference, pd.Timestamp("2024-03-26"))
        candidates = [
            annual_reference + pd.Timedelta(days=offset)
            for offset in (-14, -7, 0, 7, 14)
        ]
        self.con.execute(
            f"""
            UPDATE benchmark_daily_rows
            SET demand = CASE WHEN period = ? THEN 20.0 ELSE 100.0 END,
                is_active = TRUE,
                reason_closed = NULL
            WHERE ARTIKEL_ID = 1
                AND period IN ({", ".join("?" for _ in candidates)})
            """,
            [annual_reference.date(), *[date.date() for date in candidates]],
        )

        create_feature_tables(self.con, origins=[origin], design=self.design)
        frame = make_feature_frame(self.con, [origin], self.design)
        row = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.eq(target)
        ].iloc[0]

        target_context = self.con.execute(
            """
            SELECT holiday_event_window, event_name
            FROM ml_calendar WHERE period = ?
            """,
            [target.date()],
        ).fetchone()
        matching_candidates = self.con.execute(
            f"""
            SELECT COUNT(*)
            FROM ml_calendar
            WHERE period IN ({", ".join("?" for _ in candidates)})
                AND (holiday_event_window, event_name) = (?, ?)
            """,
            [*[date.date() for date in candidates], *target_context],
        ).fetchone()[0]
        self.assertEqual(matching_candidates, 1)
        self.assertEqual(row.annual_lookup_days_available, 1)
        self.assertEqual(row.same_weekday_last_year_mean, 20.0)
        self.assertEqual(
            row.product_cross_store_same_weekday_last_year_mean,
            20.0,
        )

        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET is_active = FALSE, reason_closed = 'test closure'
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10 AND period = ?
            """,
            [annual_reference.date()],
        )
        create_feature_tables(self.con, origins=[origin], design=self.design)
        rebuilt = make_feature_frame(self.con, [origin], self.design)
        rebuilt_row = rebuilt.loc[
            rebuilt.ARTIKEL_ID.eq(1)
            & rebuilt.MARKT_ID.eq(10)
            & rebuilt.period.eq(target)
        ].iloc[0]
        self.assertEqual(rebuilt_row.annual_lookup_days_available, 0)
        self.assertTrue(pd.isna(rebuilt_row.same_weekday_last_year_mean))

    def test_annual_anchor_uses_inclusive_easter_window(self) -> None:
        expected = {
            "2026-04-30": ("2025-04-30", "event"),
            "2026-05-22": ("2025-06-06", "easter"),
            "2026-05-23": ("2025-06-07", "easter"),
            "2026-05-09": ("2025-05-10", "event"),
            "2026-04-13": ("2025-04-28", "easter"),
            "2026-04-07": ("2025-04-22", "easter"),
            "2026-02-10": ("2025-02-11", "regular"),
        }
        contexts = _annual_anchor_calendar("2026-02-10", "2026-05-23").set_index(
            "target_period"
        )
        for target, (expected_anchor, expected_kind) in expected.items():
            with self.subTest(target=target):
                context = contexts.loc[pd.Timestamp(target).date()]
                self.assertEqual(
                    pd.Timestamp(context.anchor_period),
                    pd.Timestamp(expected_anchor),
                )
                self.assertEqual(context.anchor_kind, expected_kind)
                self.assertEqual(anchor_date(target), pd.Timestamp(expected_anchor))

    def test_annual_anchor_easter_window_boundaries_are_inclusive(self) -> None:
        self.assertEqual(anchor_date("2025-03-29"), pd.Timestamp("2024-03-30"))
        self.assertEqual(anchor_date("2025-03-30"), pd.Timestamp("2024-03-10"))
        self.assertEqual(anchor_date("2025-06-23"), pd.Timestamp("2024-06-03"))
        self.assertEqual(anchor_date("2025-06-24"), pd.Timestamp("2024-06-25"))

    def test_annual_lookup_availability_uses_easter_anchor(self) -> None:
        origin = pd.Timestamp("2025-04-07")
        annual_reference = anchor_date(origin)
        fixed_reference = origin - pd.Timedelta(days=364)
        self.con.execute(
            """
            DELETE FROM benchmark_daily_rows
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10 AND period = ?
            """,
            [annual_reference.date()],
        )
        create_feature_tables(self.con, origins=[origin], design=self.design)

        fixed_reference_exists = self.con.execute(
            """
            SELECT COUNT(*)
            FROM benchmark_daily_rows
            WHERE ARTIKEL_ID = 1 AND MARKT_ID = 10 AND period = ?
            """,
            [fixed_reference.date()],
        ).fetchone()[0]
        frame = make_feature_frame(self.con, [origin], self.design)
        first = frame.loc[
            frame.ARTIKEL_ID.eq(1)
            & frame.MARKT_ID.eq(10)
            & frame.period.eq(origin)
        ].iloc[0]

        self.assertEqual(fixed_reference_exists, 1)
        self.assertEqual(first.annual_lookup_days_available, 3)

    def test_feature_materialization_writes_origin_batches(self) -> None:
        origins = pd.DatetimeIndex(["2025-01-06", "2025-01-13"])
        create_feature_tables(self.con, origins=origins, design=self.design)

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
        self.con.execute(
            """
            UPDATE benchmark_daily_rows
            SET is_active = FALSE, reason_closed = 'test closure'
            WHERE MARKT_ID = 10 AND period = ?
            """,
            [self.design.first_origin.date()],
        )
        frames = prepare_global_lightgbm_frames(
            design=self.design,
            evaluation_origins=evaluation_origins,
            connection=self.con,
            feature_dataset_path=self.feature_dataset_path,
            config=config,
        )
        progress = StringIO()
        with redirect_stdout(progress):
            results = fit_all_lightgbm_models(frames, config)

        self.assertEqual(len(results), 4)
        self.assertIn("[model 1/4] Fitting", progress.getvalue())
        self.assertIn("[model 4/4] Completed", progress.getvalue())
        expected_training_settings = {
            MODEL_NAME: [("regression_l2", "rmse")],
            TWEEDIE_MODEL_NAME: [("tweedie", "tweedie_deviance")],
            TWO_STAGE_MODEL_NAME: [
                ("binary", "binary_logloss"),
                ("gamma", "gamma_deviance"),
            ],
            WEEKLY_MODEL_NAME: [
                ("tweedie", "tweedie_deviance_weekly_totals")
            ],
        }
        for model_name, result in results.items():
            self.assertEqual(len(result.forecasts), len(frames.evaluation))
            active = result.forecasts["is_active"]
            self.assertTrue(result.forecasts.loc[active, "forecast"].ge(0).all())
            self.assertTrue(result.forecasts.loc[~active, "forecast"].isna().all())
            expected_summary_rows = 2 if model_name == TWO_STAGE_MODEL_NAME else 1
            self.assertEqual(len(result.training_summary), expected_summary_rows)
            self.assertEqual(result.training_summary.loc[0, "evaluation_origins"], 2)
            self.assertEqual(
                list(
                    result.training_summary[
                        ["objective", "early_stopping_metric"]
                    ].itertuples(index=False, name=None)
                ),
                expected_training_settings[model_name],
            )
        weekly_audit = results[WEEKLY_MODEL_NAME].allocation_audit
        self.assertIsNotNone(weekly_audit)
        assert weekly_audit is not None
        self.assertLess(weekly_audit.allocation_error.max(), 1e-9)
        self.assertTrue(np.allclose(weekly_audit.weekday_share_sum, 1.0))

        two_stage = results[TWO_STAGE_MODEL_NAME]
        self.assertEqual(
            results[MODEL_NAME].model.feature_columns,
            DIRECT_FEATURE_COLUMNS,
        )
        self.assertEqual(
            results[TWEEDIE_MODEL_NAME].model.feature_columns,
            DIRECT_FEATURE_COLUMNS,
        )
        self.assertTrue(
            {
                "event_lift_pooled_occurrence",
                "event_lift_pooled_quantity",
                "event_lift_pooled_total",
            }.isdisjoint(WEEKLY_FEATURE_COLUMNS)
        )
        self.assertIn("event_lift_pooled_total", DIRECT_FEATURE_COLUMNS)
        self.assertNotIn("event_lift_pooled_occurrence", DIRECT_FEATURE_COLUMNS)
        self.assertNotIn("event_lift_pooled_quantity", DIRECT_FEATURE_COLUMNS)
        validation_predictions = two_stage.validation_predictions
        self.assertIsNotNone(validation_predictions)
        assert validation_predictions is not None
        self.assertEqual(len(validation_predictions), len(frames.validation))
        self.assertTrue(
            validation_predictions.occurrence_probability.between(0, 1).all()
        )
        occurrence_iteration = int(
            two_stage.training_summary.loc[
                two_stage.training_summary.stage.eq("occurrence"),
                "best_iteration",
            ].iloc[0]
        )
        self.assertEqual(
            validation_predictions.best_iteration.unique().tolist(),
            [occurrence_iteration],
        )
        self.assertEqual(
            pd.DatetimeIndex(validation_predictions.origin.unique()).tolist(),
            pd.DatetimeIndex(frames.validation.origin.unique()).tolist(),
        )
        self.assertEqual(
            two_stage.model.occurrence.feature_columns,
            OCCURRENCE_FEATURE_COLUMNS,
        )
        self.assertEqual(
            two_stage.model.quantity.feature_columns,
            QUANTITY_FEATURE_COLUMNS,
        )

        feature_subset = tuple(FEATURE_COLUMNS[:-4])
        subset_result = fit_two_stage(
            frames,
            config,
            feature_columns=feature_subset,
        )
        self.assertEqual(
            subset_result.model.occurrence.feature_columns,
            tuple(
                feature
                for feature in feature_subset
                if feature
                not in {
                    "event_lift_pooled_quantity",
                    "event_lift_pooled_total",
                }
            ),
        )
        self.assertEqual(
            subset_result.model.quantity.feature_columns,
            tuple(
                feature
                for feature in feature_subset
                if feature
                not in {
                    "event_lift_pooled_occurrence",
                    "event_lift_pooled_total",
                }
            ),
        )
        self.assertEqual(
            set(OCCURRENCE_FEATURE_COLUMNS)
            - {"event_lift_pooled_occurrence"},
            set(QUANTITY_FEATURE_COLUMNS)
            - {"event_lift_pooled_quantity"},
        )


if __name__ == "__main__":
    unittest.main()
