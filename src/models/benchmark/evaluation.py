"""Run the rolling mature-series benchmark on daily demand data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from src.models.benchmark.config import (
    BenchmarkDesign,
    DEFAULT_DATA_DIR,
    load_benchmark_design,
)
from src.models.benchmark.models import MODEL_COLUMNS, create_history_features
from src.models.baseline.evaluation import forecast_extended_baselines


GROUP_COLS = ("ARTIKEL_ID", "MARKT_ID")
DEFAULT_DEMAND_COL = "ABVERKAUFTE_MENGE_KG"


@dataclass(frozen=True)
class BenchmarkResult:
    """Forecast rows plus diagnostics for one mature-series benchmark run."""

    design: BenchmarkDesign
    forecasts: pd.DataFrame
    origin_summary: pd.DataFrame
    data_audit: pd.DataFrame


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _parquet_glob(data_dir: Path) -> str:
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return str(Path(data_dir) / "*.parquet")


def prepare_daily_rows(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path = DEFAULT_DATA_DIR,
    demand_col: str = DEFAULT_DEMAND_COL,
) -> pd.DataFrame:
    """Load the benchmark population and validate stable series attributes."""
    parquet_glob = _parquet_glob(data_dir)
    demand = _quote_identifier(demand_col)
    con.read_parquet(parquet_glob).create_view("benchmark_parquet", replace=True)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW benchmark_source_rows AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            CAST(DATE AS DATE) AS period,
            CAST(COALESCE({demand}, 0) AS DOUBLE) AS demand,
            CAST(is_active AS BOOLEAN) AS is_active,
            reason_closed,
            CASE
                WHEN COALESCE(AKTION_KENNZEICHEN, 0) = 1 THEN 1
                ELSE 0
            END::TINYINT AS action_flag,
            is_fcm,
            is_pseudo,
            WGR_ID::INTEGER AS category_id,
            CASE WHEN is_fcm THEN 'FCM' WHEN is_pseudo THEN 'Pseudo' END
                AS sourcing_group
        FROM benchmark_parquet
        WHERE (is_fcm OR is_pseudo)
          AND WGR_ID IN (890, 900)
        """
    )
    audit = con.execute(
        """
        WITH series_attributes AS (
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                COUNT(DISTINCT sourcing_group) AS source_groups,
                COUNT(DISTINCT category_id) AS categories
            FROM benchmark_source_rows
            GROUP BY ALL
        )
        SELECT
            COUNT(*) AS source_rows,
            COUNT(DISTINCT (ARTIKEL_ID, MARKT_ID)) AS series,
            MIN(period) AS first_date,
            MAX(period) AS last_date,
            COUNT_IF(is_fcm AND is_pseudo) AS overlapping_source_rows,
            COUNT_IF(is_active IS NULL) AS null_active_flags,
            COUNT_IF(is_active AND reason_closed IS NOT NULL)
                AS active_rows_with_closure_reason,
            COUNT_IF(NOT is_active AND reason_closed IS NULL)
                AS closed_rows_without_reason,
            COUNT_IF(NOT is_active AND demand <> 0) AS closed_rows_with_demand,
            (SELECT COUNT(*) FROM series_attributes
                WHERE source_groups > 1) AS changing_source_series,
            (SELECT COUNT(*) FROM series_attributes
                WHERE categories > 1) AS changing_category_series
        FROM benchmark_source_rows
        """
    ).fetchdf()
    failure_cols = [
        "overlapping_source_rows",
        "null_active_flags",
        "active_rows_with_closure_reason",
        "closed_rows_without_reason",
        "closed_rows_with_demand",
        "changing_source_series",
        "changing_category_series",
    ]
    if audit[failure_cols].sum(axis=1).iloc[0] > 0:
        raise RuntimeError("Benchmark source/category audit failed; inspect data_audit.")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE benchmark_daily_rows AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            period,
            SUM(demand) AS demand,
            BOOL_OR(is_active) AS is_active,
            CASE
                WHEN BOOL_OR(is_active) THEN NULL
                ELSE ANY_VALUE(reason_closed)
            END AS reason_closed,
            MAX(action_flag)::TINYINT AS action_flag,
            ANY_VALUE(sourcing_group) AS sourcing_group,
            ANY_VALUE(category_id) AS category_id
        FROM benchmark_source_rows
        GROUP BY ARTIKEL_ID, MARKT_ID, period
        ORDER BY ARTIKEL_ID, MARKT_ID, period
        """
    )
    return audit


def _create_eligible_origins(
    con: duckdb.DuckDBPyConnection,
    origins: pd.DatetimeIndex,
    design: BenchmarkDesign,
) -> pd.DataFrame:
    origin_frame = pd.DataFrame({"origin": origins.date})
    con.register("benchmark_origins", origin_frame)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE benchmark_series AS
        SELECT DISTINCT
            ARTIKEL_ID, MARKT_ID, sourcing_group, category_id
        FROM benchmark_daily_rows
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE benchmark_origin_history AS
        WITH origin_series AS (
            SELECT s.*, o.origin
            FROM benchmark_series AS s
            CROSS JOIN benchmark_origins AS o
        )
        SELECT
            os.*,
            f.active_days_to_date AS active_days_before_origin,
            f.demand_days_to_date AS demand_days_before_origin,
            f.store_scale,
            f.recent_mean,
            f.occurrence_rate AS recent_occurrence_rate,
            DATE_DIFF('day', f.last_positive_period, os.origin)
                AS days_since_last_demand,
            f.seasonal_abs_error_sum
                / NULLIF(f.seasonal_pair_count, 0) AS seasonal_mase_scale
        FROM origin_series AS os
        ASOF LEFT JOIN benchmark_row_features AS f
            ON os.ARTIKEL_ID = f.ARTIKEL_ID
            AND os.MARKT_ID = f.MARKT_ID
            AND os.origin > f.period
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE benchmark_eligible_origins AS
        SELECT *
        FROM benchmark_origin_history
        WHERE active_days_before_origin >= ?
          AND demand_days_before_origin >= ?
        """,
        [design.min_active_days, design.min_demand_days],
    )
    return con.execute(
        """
        SELECT
            h.origin,
            COUNT_IF(h.active_days_before_origin IS NOT NULL) AS known_series,
            COUNT(e.ARTIKEL_ID) AS mature_series,
            COUNT_IF(e.sourcing_group = 'FCM') AS mature_fcm_series,
            COUNT_IF(e.sourcing_group = 'Pseudo') AS mature_pseudo_series
        FROM benchmark_origin_history AS h
        LEFT JOIN benchmark_eligible_origins AS e
            ON e.ARTIKEL_ID = h.ARTIKEL_ID
            AND e.MARKT_ID = h.MARKT_ID
            AND e.origin = h.origin
        GROUP BY h.origin
        ORDER BY h.origin
        """
    ).fetchdf()


def _forecast_rows(
    con: duckdb.DuckDBPyConnection,
    horizon_days: int,
) -> pd.DataFrame:
    wide = con.execute(
        """
        WITH targets AS (
            SELECT
                e.*,
                t.period,
                t.demand AS actual,
                t.is_active,
                t.reason_closed,
                EXTRACT(DOW FROM t.period)::INTEGER AS target_weekday
            FROM benchmark_eligible_origins AS e
            INNER JOIN benchmark_daily_rows AS t
                ON t.ARTIKEL_ID = e.ARTIKEL_ID
                AND t.MARKT_ID = e.MARKT_ID
                AND t.period >= e.origin
                AND t.period < e.origin + ? * INTERVAL 1 DAY
        ),
        with_weekday AS (
            SELECT t.*, w.same_weekday_mean
            FROM targets AS t
            ASOF LEFT JOIN benchmark_weekday_features AS w
                ON t.ARTIKEL_ID = w.ARTIKEL_ID
                AND t.MARKT_ID = w.MARKT_ID
                AND t.target_weekday = w.weekday
                AND t.origin > w.period
        ),
        with_positive_quantity AS (
            SELECT w.*, p.positive_quantity_mean
            FROM with_weekday AS w
            ASOF LEFT JOIN benchmark_positive_features AS p
                ON w.ARTIKEL_ID = p.ARTIKEL_ID
                AND w.MARKT_ID = p.MARKT_ID
                AND w.origin > p.period
        )
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            sourcing_group,
            category_id,
            origin,
            period,
            DATE_DIFF('day', origin, period) + 1 AS horizon_day,
            active_days_before_origin,
            demand_days_before_origin,
            recent_occurrence_rate,
            days_since_last_demand,
            seasonal_mase_scale,
            actual,
            is_active,
            reason_closed,
            CASE WHEN is_active THEN recent_mean ELSE 0.0 END AS recent_mean,
            CASE
                WHEN is_active THEN COALESCE(same_weekday_mean, recent_mean)
                ELSE 0.0
            END AS same_weekday_moving_average,
            CASE
                WHEN is_active
                    THEN recent_occurrence_rate * COALESCE(positive_quantity_mean, 0)
                ELSE 0.0
            END AS occurrence_x_positive_quantity
        FROM with_positive_quantity
        ORDER BY origin, ARTIKEL_ID, MARKT_ID, period
        """,
        [horizon_days],
    ).fetchdf()
    id_cols = [
        "ARTIKEL_ID",
        "MARKT_ID",
        "sourcing_group",
        "category_id",
        "origin",
        "period",
        "horizon_day",
        "active_days_before_origin",
        "demand_days_before_origin",
        "recent_occurrence_rate",
        "days_since_last_demand",
        "seasonal_mase_scale",
        "actual",
        "is_active",
        "reason_closed",
    ]
    return wide.melt(
        id_vars=id_cols,
        value_vars=list(MODEL_COLUMNS),
        var_name="model",
        value_name="forecast",
    )


def run_benchmark(
    data_dir: Path = DEFAULT_DATA_DIR,
    design: BenchmarkDesign | None = None,
    demand_col: str = DEFAULT_DEMAND_COL,
    connection: duckdb.DuckDBPyConnection | None = None,
    include_extended_baselines: bool = False,
) -> BenchmarkResult:
    """Run configured baselines at every complete forecast origin."""
    design = load_benchmark_design() if design is None else design
    con = duckdb.connect() if connection is None else connection
    con.execute("PRAGMA threads=4")
    audit = prepare_daily_rows(con, data_dir=data_dir, demand_col=demand_col)
    create_history_features(con)
    last_observed = audit.loc[0, "last_date"]
    origins = design.origins_through(last_observed)
    if len(origins) == 0:
        raise RuntimeError("No configured origin has a complete forecast horizon.")
    origin_summary = _create_eligible_origins(con, origins, design)
    forecasts = _forecast_rows(con, design.forecast_horizon_days)
    if include_extended_baselines:
        extended_forecasts = forecast_extended_baselines(
            con, design.forecast_horizon_days
        )
        forecasts = pd.concat([forecasts, extended_forecasts], ignore_index=True)
    evaluated = (
        forecasts[["origin", *GROUP_COLS]].drop_duplicates()
        .groupby("origin")
        .size()
        .rename("evaluated_series")
    )
    origin_summary = origin_summary.merge(
        evaluated, how="left", left_on="origin", right_index=True
    )
    origin_summary["evaluated_series"] = (
        origin_summary["evaluated_series"].fillna(0).astype(int)
    )
    return BenchmarkResult(
        design=design,
        forecasts=forecasts,
        origin_summary=origin_summary,
        data_audit=audit,
    )
