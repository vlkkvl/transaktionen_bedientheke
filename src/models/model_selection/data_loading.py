"""Load demand series and compute ADI/CV2 demand classes from parquet."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.models.baseline.registry import DEMAND_CLASSES, add_demand_class
from src.models.model_selection.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_DEMAND_COL,
    DEFAULT_FORECAST_PERIODS,
    DEFAULT_GROUP_COLS,
    DEFAULT_MIN_TRAIN_SIZE,
    MIN_DEMAND_PERIODS_UNTIL_ORIGIN,
)


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def parquet_glob(data_dir: Path) -> str:
    """Return the parquet glob after validating the directory exists."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return str(data_dir / "*.parquet")


def _new_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    return con


def compute_series_metrics(
    data_dir: Path = DEFAULT_DATA_DIR,
    demand_col: str = DEFAULT_DEMAND_COL,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_demand_periods_until_origin: int = MIN_DEMAND_PERIODS_UNTIL_ORIGIN,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
) -> pd.DataFrame:
    """Compute initial-window ADI/CV2 and demand class for every period series."""
    group_sql = ", ".join(quote_identifier(col) for col in group_cols)
    demand_sql = quote_identifier(demand_col)
    date_sql = quote_identifier("DATE")
    glob = parquet_glob(data_dir)
    initial_window_filter = f"period_number <= {int(min_train_size)}"

    query = f"""
    WITH period_data AS (
        SELECT
            {group_sql},
            CAST({date_sql} AS DATE) AS period_start,
            SUM(CAST(COALESCE({demand_sql}, 0) AS DOUBLE)) AS demand
        FROM read_parquet(?)
        GROUP BY {group_sql}, period_start
    ), ordered_period_data AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY {group_sql}
                ORDER BY period_start
            ) AS period_number
        FROM period_data
    ), series AS (
        SELECT
            {group_sql},
            COUNT(*) AS active_periods,
            SUM(CASE WHEN demand > 0 THEN 1 ELSE 0 END) AS demand_periods,
            SUM(demand) AS series_total_demand,
            AVG(CASE WHEN demand > 0 THEN demand END) AS mean_nonzero_demand,
            VAR_SAMP(CASE WHEN demand > 0 THEN demand END) AS var_nonzero_demand,
            MIN(period_start) AS first_active_period,
            MAX(period_start) AS last_active_period
        FROM ordered_period_data
        GROUP BY {group_sql}
    ), initial_window AS (
        SELECT
            {group_sql},
            SUM(CASE WHEN {initial_window_filter} THEN 1 ELSE 0 END)
                AS active_periods_until_origin,
            SUM(
                CASE
                    WHEN {initial_window_filter} AND demand > 0 THEN 1
                    ELSE 0
                END
            ) AS demand_periods_until_origin,
            SUM(
                CASE
                    WHEN {initial_window_filter} THEN demand
                    ELSE 0
                END
            ) AS series_total_demand_until_origin,
            AVG(
                CASE
                    WHEN {initial_window_filter} AND demand > 0 THEN demand
                    ELSE NULL
                END
            ) AS mean_nonzero_demand_until_origin,
            VAR_SAMP(
                CASE
                    WHEN {initial_window_filter} AND demand > 0 THEN demand
                    ELSE NULL
                END
            ) AS var_nonzero_demand_until_origin
        FROM ordered_period_data
        GROUP BY {group_sql}
    )
    SELECT
        s.*,
        i.active_periods_until_origin,
        i.demand_periods_until_origin,
        i.series_total_demand_until_origin,
        i.mean_nonzero_demand_until_origin,
        i.var_nonzero_demand_until_origin,
        i.active_periods_until_origin
            / NULLIF(i.demand_periods_until_origin, 0) AS ADI,
        CASE
            WHEN i.demand_periods_until_origin > 1
                AND i.mean_nonzero_demand_until_origin > 0
                THEN i.var_nonzero_demand_until_origin
                    / (
                        i.mean_nonzero_demand_until_origin
                        * i.mean_nonzero_demand_until_origin
                    )
            WHEN i.demand_periods_until_origin = 1 THEN 0.0
            ELSE NULL
        END AS CV2
    FROM series AS s
    INNER JOIN initial_window AS i
        USING ({group_sql})
    WHERE s.demand_periods > 0
    """

    con = _new_connection()
    metrics = con.execute(query, [glob]).fetchdf()
    metrics = add_demand_class(metrics)
    filtered = metrics[
        metrics["demand_periods_until_origin"] >= min_demand_periods_until_origin
    ].reset_index(drop=True)
    filtered.attrs["series_before_min_demand_filter"] = len(metrics)
    filtered.attrs["excluded_by_min_demand_periods_until_origin"] = (
        len(metrics) - len(filtered)
    )
    return filtered


def select_evaluation_keys(
    series_metrics: pd.DataFrame,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    forecast_periods: int = DEFAULT_FORECAST_PERIODS,
    max_series_per_class: int | None = None,
) -> pd.DataFrame:
    """Pick recent series long enough for at least one sliding-window forecast."""
    min_periods = min_train_size + forecast_periods
    eligible = series_metrics[
        series_metrics["demand_class"].isin(DEMAND_CLASSES)
        & (series_metrics["active_periods"] >= min_periods)
    ].copy()

    if max_series_per_class is None:
        return eligible.sort_values(["demand_class", *group_cols]).reset_index(drop=True)

    if max_series_per_class < 1:
        raise ValueError("max_series_per_class must be positive or None")

    if eligible.empty:
        return eligible.reset_index(drop=True)

    sort_cols = [
        "demand_class",
        "last_active_period",
        "demand_periods_until_origin",
        "active_periods",
        "series_total_demand",
        *group_cols,
    ]
    ascending = [True, False, False, False, False, *([True] * len(group_cols))]
    ranked = eligible.sort_values(sort_cols, ascending=ascending)
    top_parts = [
        class_df.head(max_series_per_class)
        for _, class_df in ranked.groupby("demand_class", sort=False)
    ]
    selected = pd.concat(top_parts, ignore_index=True)
    return selected.sort_values(["demand_class", *group_cols]).reset_index(drop=True)


def load_period_series(
    data_dir: Path,
    selected_keys: pd.DataFrame,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    demand_col: str = DEFAULT_DEMAND_COL,
) -> pd.DataFrame:
    """Load demand rows for the selected product-store series."""
    if selected_keys.empty:
        return pd.DataFrame(
            columns=[*group_cols, "period_start", "demand", "demand_class"]
        )

    glob = parquet_glob(data_dir)
    group_sql = ", ".join(quote_identifier(col) for col in group_cols)
    p_group_sql = ", ".join(f"p.{quote_identifier(col)}" for col in group_cols)
    join_sql = " AND ".join(
        f"p.{quote_identifier(col)} = k.{quote_identifier(col)}" for col in group_cols
    )
    demand_sql = quote_identifier(demand_col)
    date_sql = quote_identifier("DATE")

    key_cols = selected_keys[[*group_cols, "demand_class"]].drop_duplicates()

    con = _new_connection()
    con.register("selected_keys", key_cols)
    query = f"""
    WITH period_data AS (
        SELECT
            {group_sql},
            CAST({date_sql} AS DATE) AS period_start,
            SUM(CAST(COALESCE({demand_sql}, 0) AS DOUBLE)) AS demand
        FROM read_parquet(?)
        GROUP BY {group_sql}, period_start
    )
    SELECT
        {p_group_sql},
        p.period_start,
        p.demand,
        k.demand_class
    FROM period_data AS p
    INNER JOIN selected_keys AS k
        ON {join_sql}
    ORDER BY {p_group_sql}, p.period_start
    """
    return con.execute(query, [glob]).fetchdf()


load_weekly_series = load_period_series
