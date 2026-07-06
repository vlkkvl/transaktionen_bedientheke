"""Load weekly demand series and compute ADI/CV2 demand classes from parquet."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.models.baseline.registry import DEMAND_CLASSES, add_demand_class
from src.models.model_selection.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_DEMAND_COL,
    DEFAULT_GROUP_COLS,
    DEFAULT_MIN_DEMAND_WEEKS,
    DEFAULT_MIN_TRAIN_SIZE,
    HORIZON_WEEKS,
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
    min_demand_weeks: int = DEFAULT_MIN_DEMAND_WEEKS,
) -> pd.DataFrame:
    """Compute ADI/CV2 and demand class for every weekly series."""
    group_sql = ", ".join(quote_identifier(col) for col in group_cols)
    demand_sql = quote_identifier(demand_col)
    date_sql = quote_identifier("DATE")
    glob = parquet_glob(data_dir)

    query = f"""
    WITH period_data AS (
        SELECT
            {group_sql},
            CAST({date_sql} AS DATE) AS period_start,
            SUM(CAST(COALESCE({demand_sql}, 0) AS DOUBLE)) AS demand
        FROM read_parquet(?)
        GROUP BY {group_sql}, period_start
    ), series AS (
        SELECT
            {group_sql},
            COUNT(*) AS active_weeks,
            SUM(CASE WHEN demand > 0 THEN 1 ELSE 0 END) AS demand_weeks,
            SUM(demand) AS series_total_demand,
            AVG(CASE WHEN demand > 0 THEN demand END) AS mean_nonzero_demand,
            VAR_SAMP(CASE WHEN demand > 0 THEN demand END) AS var_nonzero_demand,
            MIN(period_start) AS first_active_week,
            MAX(period_start) AS last_active_week
        FROM period_data
        GROUP BY {group_sql}
    )
    SELECT
        *,
        active_weeks / NULLIF(demand_weeks, 0) AS ADI,
        CASE
            WHEN demand_weeks > 1 AND mean_nonzero_demand > 0
                THEN var_nonzero_demand
                    / (mean_nonzero_demand * mean_nonzero_demand)
            WHEN demand_weeks = 1 THEN 0.0
            ELSE NULL
        END AS CV2
    FROM series
    WHERE demand_weeks > 0
    """

    con = _new_connection()
    metrics = con.execute(query, [glob]).fetchdf()
    metrics = add_demand_class(metrics)
    return metrics[metrics["demand_weeks"] >= min_demand_weeks].reset_index(drop=True)


def select_evaluation_keys(
    series_metrics: pd.DataFrame,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    max_series_per_class: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Pick series long enough for at least one sliding-window forecast."""
    min_periods = min_train_size + HORIZON_WEEKS
    eligible = series_metrics[
        series_metrics["demand_class"].isin(DEMAND_CLASSES)
        & (series_metrics["active_weeks"] >= min_periods)
    ].copy()

    if max_series_per_class is None:
        return eligible.sort_values(["demand_class", *group_cols]).reset_index(drop=True)

    if max_series_per_class < 1:
        raise ValueError("max_series_per_class must be positive or None")

    if eligible.empty:
        return eligible.reset_index(drop=True)

    sampled_parts = [
        class_df.sample(
            n=min(max_series_per_class, len(class_df)),
            random_state=random_state,
        )
        for _, class_df in eligible.groupby("demand_class", sort=False)
    ]
    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sort_values(["demand_class", *group_cols]).reset_index(drop=True)


def load_weekly_series(
    data_dir: Path,
    selected_keys: pd.DataFrame,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    demand_col: str = DEFAULT_DEMAND_COL,
) -> pd.DataFrame:
    """Load weekly demand rows for the selected product-store series."""
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
