"""Maturity-aware rolling-origin evaluation for the extended baselines."""
from __future__ import annotations

import duckdb
import pandas as pd

from src.models.baseline.aggregate_then_disaggregate import (
    LEVEL_WEEKS,
    PROFILE_WEEKS,
)
from src.models.baseline.batch import SCALAR_MODEL_NAMES, forecast_scalar_levels


BASELINE_MODEL_LABELS = {
    "simple_exponential_smoothing": "ETS (A,N,N) / simple exponential smoothing",
    "croston": "Croston",
    "sba": "SBA (bias-adjusted Croston)",
    "tsb": "TSB",
    "aggregate_then_disaggregate": (
        "Aggregate then disaggregate (7-day total x weekday profile)"
    ),
}


def _target_rows(
    con: duckdb.DuckDBPyConnection,
    origin: object,
    horizon_days: int,
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            e.ARTIKEL_ID,
            e.MARKT_ID,
            e.sourcing_group,
            e.category_id,
            e.origin,
            t.period,
            DATE_DIFF('day', e.origin, t.period) + 1 AS horizon_day,
            e.active_days_before_origin,
            e.demand_days_before_origin,
            e.recent_occurrence_rate,
            e.calendar_days_since_last_demand,
            e.seasonal_mase_scale,
            t.demand AS actual,
            t.is_active,
            t.reason_closed
        FROM benchmark_assessed_origins AS e
        INNER JOIN benchmark_daily_rows AS t
            ON t.ARTIKEL_ID = e.ARTIKEL_ID
            AND t.MARKT_ID = e.MARKT_ID
            AND t.period >= e.origin
            AND t.period < e.origin + ? * INTERVAL 1 DAY
        WHERE e.origin = ?
        ORDER BY e.ARTIKEL_ID, e.MARKT_ID, t.period
        """,
        [int(horizon_days), pd.Timestamp(origin).date()],
    ).fetchdf()


def forecast_scalar_baselines(
    con: duckdb.DuckDBPyConnection,
    horizon_days: int,
) -> pd.DataFrame:
    """Fit SES and intermittent baselines on histories strictly before each origin."""
    origins = con.execute(
        "SELECT DISTINCT origin FROM benchmark_assessed_origins ORDER BY origin"
    ).fetchnumpy()["origin"]
    outputs: list[pd.DataFrame] = []
    for origin in origins:
        histories = con.execute(
            """
            SELECT
                e.ARTIKEL_ID,
                e.MARKT_ID,
                LIST(d.demand ORDER BY d.period) AS history
            FROM benchmark_assessed_origins AS e
            INNER JOIN benchmark_daily_rows AS d
                ON d.ARTIKEL_ID = e.ARTIKEL_ID
                AND d.MARKT_ID = e.MARKT_ID
                AND d.period < e.origin
            WHERE e.origin = ?
            GROUP BY e.ARTIKEL_ID, e.MARKT_ID
            ORDER BY e.ARTIKEL_ID, e.MARKT_ID
            """,
            [pd.Timestamp(origin).date()],
        ).fetchdf()
        levels = forecast_scalar_levels(histories["history"].tolist())
        level_frame = histories[["ARTIKEL_ID", "MARKT_ID"]].copy()
        for index, model in enumerate(SCALAR_MODEL_NAMES):
            level_frame[model] = levels[:, index]

        targets = _target_rows(con, origin, horizon_days)
        wide = targets.merge(
            level_frame, on=["ARTIKEL_ID", "MARKT_ID"], how="inner", validate="m:1"
        )
        id_cols = list(targets.columns)
        outputs.append(
            wide.melt(
                id_vars=id_cols,
                value_vars=list(SCALAR_MODEL_NAMES),
                var_name="model",
                value_name="forecast",
            ).assign(
                forecast=lambda frame: frame["forecast"].where(
                    frame["is_active"], 0.0
                )
            )
        )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def forecast_aggregate_then_disaggregate(
    con: duckdb.DuckDBPyConnection,
    horizon_days: int,
) -> pd.DataFrame:
    """Forecast a recent weekly level and apply an eight-week weekday profile."""
    if int(horizon_days) != 7:
        raise ValueError("aggregate-then-disaggregate requires a 7-day horizon")
    return con.execute(
        """
        WITH history AS (
            SELECT
                e.origin,
                e.ARTIKEL_ID,
                e.MARKT_ID,
                SUM(d.demand) FILTER (
                    WHERE d.period >= e.origin - ? * 7 * INTERVAL 1 DAY
                ) / ? AS weekly_level,
                SUM(d.demand) FILTER (
                    WHERE d.period >= e.origin - ? * 7 * INTERVAL 1 DAY
                ) AS profile_total
            FROM benchmark_assessed_origins AS e
            INNER JOIN benchmark_daily_rows AS d
                ON d.ARTIKEL_ID = e.ARTIKEL_ID
                AND d.MARKT_ID = e.MARKT_ID
                AND d.period < e.origin
                AND d.period >= e.origin - ? * 7 * INTERVAL 1 DAY
            GROUP BY e.origin, e.ARTIKEL_ID, e.MARKT_ID
        ),
        weekday_history AS (
            SELECT
                e.origin,
                e.ARTIKEL_ID,
                e.MARKT_ID,
                EXTRACT(DOW FROM d.period)::INTEGER AS weekday,
                SUM(d.demand) AS weekday_demand
            FROM benchmark_assessed_origins AS e
            INNER JOIN benchmark_daily_rows AS d
                ON d.ARTIKEL_ID = e.ARTIKEL_ID
                AND d.MARKT_ID = e.MARKT_ID
                AND d.period < e.origin
                AND d.period >= e.origin - ? * 7 * INTERVAL 1 DAY
            GROUP BY e.origin, e.ARTIKEL_ID, e.MARKT_ID, weekday
        ),
        targets AS (
            SELECT
                e.*,
                t.period,
                t.demand AS actual,
                t.is_active,
                t.reason_closed,
                EXTRACT(DOW FROM t.period)::INTEGER AS target_weekday
            FROM benchmark_assessed_origins AS e
            INNER JOIN benchmark_daily_rows AS t
                ON t.ARTIKEL_ID = e.ARTIKEL_ID
                AND t.MARKT_ID = e.MARKT_ID
                AND t.period >= e.origin
                AND t.period < e.origin + 7 * INTERVAL 1 DAY
        )
        SELECT
            t.ARTIKEL_ID,
            t.MARKT_ID,
            t.sourcing_group,
            t.category_id,
            t.origin,
            t.period,
            DATE_DIFF('day', t.origin, t.period) + 1 AS horizon_day,
            t.active_days_before_origin,
            t.demand_days_before_origin,
            t.recent_occurrence_rate,
            t.calendar_days_since_last_demand,
            t.seasonal_mase_scale,
            t.actual,
            t.is_active,
            t.reason_closed,
            'aggregate_then_disaggregate' AS model,
            CASE
                WHEN NOT t.is_active THEN 0.0
                ELSE GREATEST(
                    0,
                    h.weekly_level * COALESCE(
                        CASE
                            WHEN h.profile_total > 0 THEN
                                COALESCE(w.weekday_demand, 0) / NULLIF(
                                    SUM(
                                        CASE
                                            WHEN t.is_active
                                                THEN COALESCE(w.weekday_demand, 0)
                                            ELSE 0
                                        END
                                    ) OVER (
                                        PARTITION BY t.origin, t.ARTIKEL_ID,
                                            t.MARKT_ID
                                    ),
                                    0
                                )
                        END,
                        1.0 / COUNT_IF(t.is_active) OVER (
                            PARTITION BY t.origin, t.ARTIKEL_ID, t.MARKT_ID
                        )
                    )
                )
            END AS forecast
        FROM targets AS t
        INNER JOIN history AS h
            ON h.origin = t.origin
            AND h.ARTIKEL_ID = t.ARTIKEL_ID
            AND h.MARKT_ID = t.MARKT_ID
        LEFT JOIN weekday_history AS w
            ON w.origin = t.origin
            AND w.ARTIKEL_ID = t.ARTIKEL_ID
            AND w.MARKT_ID = t.MARKT_ID
            AND w.weekday = t.target_weekday
        ORDER BY t.origin, t.ARTIKEL_ID, t.MARKT_ID, t.period
        """,
        [LEVEL_WEEKS, LEVEL_WEEKS, PROFILE_WEEKS, PROFILE_WEEKS, PROFILE_WEEKS],
    ).fetchdf()


def forecast_extended_baselines(
    con: duckdb.DuckDBPyConnection,
    horizon_days: int,
) -> pd.DataFrame:
    """Return every extended baseline in the shared benchmark row schema."""
    scalar = forecast_scalar_baselines(con, horizon_days)
    aggregate = forecast_aggregate_then_disaggregate(con, horizon_days)
    return pd.concat([scalar, aggregate], ignore_index=True)
