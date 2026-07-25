"""Baseline definitions shared with notebook 02_01."""
from __future__ import annotations


RECENT_MEAN_DAYS = 28
SAME_WEEKDAY_OCCURRENCES = 4
POSITIVE_QUANTITY_OBSERVATIONS = 10

MODEL_COLUMNS = {
    "recent_mean": "Recent mean (28 observations)",
    "same_weekday_moving_average": (
        "Same-weekday moving average (4 occurrences; recent-mean fallback)"
    ),
    "occurrence_x_positive_quantity": "Occurrence x positive quantity",
}
PRIMARY_MODEL = "same_weekday_moving_average"


def create_history_features(con: object) -> None:
    """Create leakage-safe trailing features from the prepared daily rows.

    Every feature on a row uses that row and earlier observations. The benchmark
    later performs a strict as-of join (feature date < origin), so the origin and
    forecast horizon can never enter a model's history.
    """
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE benchmark_row_features AS
        WITH row_features AS (
            SELECT
                d.*,
                EXTRACT(DOW FROM d.period)::INTEGER AS weekday,
                COUNT_IF(d.is_active) OVER series_to_date AS active_days_to_date,
                COUNT_IF(d.is_active AND d.demand > 0) OVER series_to_date
                    AS demand_days_to_date,
                MAX(CASE WHEN d.is_active AND d.demand > 0 THEN d.period END)
                    OVER series_to_date AS last_positive_period,
                AVG(d.demand) FILTER (WHERE d.is_active) OVER series_to_date
                    AS store_scale,
                AVG(d.demand) OVER recent_rows AS recent_mean,
                AVG((d.demand > 0)::INTEGER) FILTER (WHERE d.is_active)
                    OVER recent_rows AS occurrence_rate,
                CASE
                    WHEN d.is_active AND previous_week.is_active
                    THEN ABS(d.demand - previous_week.demand)
                END AS seasonal_abs_error
            FROM benchmark_daily_rows AS d
            LEFT JOIN benchmark_daily_rows AS previous_week
                ON previous_week.ARTIKEL_ID = d.ARTIKEL_ID
                AND previous_week.MARKT_ID = d.MARKT_ID
                AND previous_week.period = d.period - INTERVAL 7 DAY
            WINDOW
                series_to_date AS (
                    PARTITION BY d.ARTIKEL_ID, d.MARKT_ID
                    ORDER BY d.period ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ),
                recent_rows AS (
                    PARTITION BY d.ARTIKEL_ID, d.MARKT_ID
                    ORDER BY d.period ROWS BETWEEN {RECENT_MEAN_DAYS - 1} PRECEDING
                        AND CURRENT ROW
                )
        )
        SELECT
            *,
            SUM(seasonal_abs_error) OVER series_to_date
                AS seasonal_abs_error_sum,
            COUNT(seasonal_abs_error) OVER series_to_date
                AS seasonal_pair_count
        FROM row_features
        WINDOW series_to_date AS (
            PARTITION BY ARTIKEL_ID, MARKT_ID
            ORDER BY period ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE benchmark_weekday_features AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            period,
            EXTRACT(DOW FROM period)::INTEGER AS weekday,
            AVG(demand) OVER weekday_rows AS same_weekday_mean
        FROM benchmark_daily_rows
        WHERE is_active
        WINDOW weekday_rows AS (
            PARTITION BY ARTIKEL_ID, MARKT_ID, EXTRACT(DOW FROM period)
            ORDER BY period ROWS BETWEEN {SAME_WEEKDAY_OCCURRENCES - 1}
                PRECEDING AND CURRENT ROW
        )
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE benchmark_positive_features AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            period,
            AVG(demand) OVER (
                PARTITION BY ARTIKEL_ID, MARKT_ID
                ORDER BY period ROWS BETWEEN {POSITIVE_QUANTITY_OBSERVATIONS - 1}
                    PRECEDING AND CURRENT ROW
            ) AS positive_quantity_mean
        FROM benchmark_daily_rows
        WHERE is_active AND demand > 0
        """
    )
