"""Remove stale and very sparse product-store series.

The reference date is the maximum ``DATE`` present in the input dataset. A
series is removed when its last positive-demand day is more than 365 days before
that reference date. Series with no positive sale at all are also removed, as
are series where positive-demand active days make up less than 10% of active
days in the last dataset year. Series with fewer than 15 positive-demand days
over their full active history are also removed.

Input:  data/interim/transactions_dst_daily_no_outliers/*.parquet
Output: data/interim/transactions_dst_daily_no_outliers_no_stale/*.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import (
    ROOT,
    clear_parquet_outputs,
    configure_duckdb,
    read_parquet_expr,
    require_parquet_files,
    sql_literal,
)


IN_DIR = ROOT / "data" / "interim" / "transactions_dst_daily_no_outliers"
OUT_DIR = ROOT / "data" / "interim" / "transactions_dst_daily_no_outliers_no_stale"

DEMAND_COL = "ABVERKAUFTE_MENGE_KG"
MAX_DAYS_SINCE_LAST_SALE = 365
MIN_LAST_YEAR_DEMAND_ACTIVE_DAY_SHARE = 0.10
MIN_DEMAND_DAYS = 15


def temp_output_path(path: Path) -> Path:
    """Return the temporary path used for an atomic yearly output write."""
    return path.with_name(f".{path.stem}.no_stale.tmp{path.suffix}")


def create_stale_series_table(
    con: duckdb.DuckDBPyConnection,
    input_glob: Path,
    *,
    max_days_since_last_sale: int = MAX_DAYS_SINCE_LAST_SALE,
    min_last_year_demand_active_day_share: float = (
        MIN_LAST_YEAR_DEMAND_ACTIVE_DAY_SHARE
    ),
    min_demand_days: int = MIN_DEMAND_DAYS,
) -> None:
    """Materialize product-store series excluded by recency or demand share."""
    if max_days_since_last_sale < 1:
        raise ValueError("Maximum days since last sale must be positive")
    if not 0 <= min_last_year_demand_active_day_share <= 1:
        raise ValueError(
            "Minimum last-year demand/active day share must be between 0 and 1"
        )
    if min_demand_days < 1:
        raise ValueError("Minimum demand days must be positive")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE stale_series AS
        WITH dataset_bounds AS (
            SELECT MAX(CAST(DATE AS DATE)) AS max_dataset_date
            FROM {read_parquet_expr(input_glob)}
        ),
        source AS (
            SELECT t.*, b.max_dataset_date
            FROM {read_parquet_expr(input_glob)} t
            CROSS JOIN dataset_bounds b
        ),
        series_sales AS (
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                COUNT(*)::BIGINT AS active_days,
                COUNT_IF(COALESCE({DEMAND_COL}, 0) > 0)::BIGINT AS demand_days,
                COUNT(*) FILTER (
                    WHERE CAST(DATE AS DATE)
                        >= max_dataset_date - INTERVAL {max_days_since_last_sale} DAY
                )::BIGINT AS last_year_active_days,
                COUNT_IF(
                    CAST(DATE AS DATE)
                        >= max_dataset_date - INTERVAL {max_days_since_last_sale} DAY
                    AND COALESCE({DEMAND_COL}, 0) > 0
                )::BIGINT AS last_year_demand_days,
                MAX(CAST(DATE AS DATE)) FILTER (
                    WHERE COALESCE({DEMAND_COL}, 0) > 0
                ) AS last_sale_date
            FROM source
            WHERE is_active
            GROUP BY ARTIKEL_ID, MARKT_ID
        )
        SELECT
            s.ARTIKEL_ID,
            s.MARKT_ID,
            s.active_days,
            s.demand_days,
            s.last_year_active_days,
            s.last_year_demand_days,
            s.last_year_demand_days::DOUBLE / NULLIF(s.last_year_active_days, 0)
                AS last_year_demand_active_day_share,
            s.last_sale_date,
            b.max_dataset_date,
            DATE_DIFF('day', s.last_sale_date, b.max_dataset_date)
                AS days_since_last_sale,
            CASE
                WHEN s.last_sale_date IS NULL THEN 'no_positive_sale'
                WHEN DATE_DIFF('day', s.last_sale_date, b.max_dataset_date) > ?
                    THEN 'stale_last_sale'
                WHEN s.last_year_demand_days::DOUBLE
                    / NULLIF(s.last_year_active_days, 0) < ?
                    THEN 'low_last_year_demand_active_day_share'
                WHEN s.demand_days < ?
                    THEN 'low_demand_days'
            END AS exclusion_reason
        FROM series_sales s
        CROSS JOIN dataset_bounds b
        WHERE
            s.last_sale_date IS NULL
            OR DATE_DIFF('day', s.last_sale_date, b.max_dataset_date) > ?
            OR s.last_year_demand_days::DOUBLE
                / NULLIF(s.last_year_active_days, 0) < ?
            OR s.demand_days < ?
        """,
        [
            max_days_since_last_sale,
            min_last_year_demand_active_day_share,
            min_demand_days,
            max_days_since_last_sale,
            min_last_year_demand_active_day_share,
            min_demand_days,
        ],
    )


def _share(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def print_filter_summary(con: duckdb.DuckDBPyConnection) -> None:
    """Print the stale/sparse-series exclusion count."""
    stale_series, stale_rows, stale_demand_kg = con.execute(
        f"""
        SELECT
            COUNT(DISTINCT (t.ARTIKEL_ID, t.MARKT_ID)) AS stale_series,
            COUNT(*) AS stale_rows,
            SUM(COALESCE(t.{DEMAND_COL}, 0)) AS stale_demand_kg
        FROM source_rows t
        INNER JOIN stale_series stale USING (ARTIKEL_ID, MARKT_ID)
        """
    ).fetchone()
    total_series, total_rows, total_demand_kg = con.execute(
        f"""
        SELECT
            COUNT(DISTINCT (ARTIKEL_ID, MARKT_ID)) AS total_series,
            COUNT(*) AS total_rows,
            SUM(COALESCE({DEMAND_COL}, 0)) AS total_demand_kg
        FROM source_rows
        """
    ).fetchone()
    stale_demand_kg = stale_demand_kg or 0.0
    total_demand_kg = total_demand_kg or 0.0
    print("\nStale/sparse-series filter summary")
    print(
        f"  {stale_series:,}/{total_series:,} series removed "
        f"({_share(stale_series, total_series):.2%}); "
        f"{stale_rows:,}/{total_rows:,} rows removed "
        f"({_share(stale_rows, total_rows):.2%}); "
        f"{stale_demand_kg:,.1f}/{total_demand_kg:,.1f} kg removed "
        f"({_share(stale_demand_kg, total_demand_kg):.3%})"
    )
    reason_summary = con.execute(
        """
        SELECT exclusion_reason, COUNT(*) AS series
        FROM stale_series
        GROUP BY exclusion_reason
        ORDER BY series DESC, exclusion_reason
        """
    ).fetchall()
    for reason, series in reason_summary:
        print(f"  {reason}: {series:,} series")


def output_select_sql(path: Path) -> str:
    """Return the row-preserving selection for one yearly input file."""
    return f"""
        SELECT t.*
        FROM {read_parquet_expr(path)} t
        LEFT JOIN stale_series stale USING (ARTIKEL_ID, MARKT_ID)
        WHERE stale.ARTIKEL_ID IS NULL
    """


def write_outputs(
    con: duckdb.DuckDBPyConnection,
    input_files: list[Path],
    out_dir: Path,
) -> None:
    """Write one stale-filtered parquet per input year."""
    clear_parquet_outputs(out_dir)
    total_input_rows = 0
    total_output_rows = 0

    for path in input_files:
        out_path = out_dir / path.name
        tmp_path = temp_output_path(out_path)
        if tmp_path.exists():
            tmp_path.unlink()

        input_rows = con.execute(
            f"SELECT COUNT(*) FROM {read_parquet_expr(path)}"
        ).fetchone()[0]
        output_rows = con.execute(
            f"SELECT COUNT(*) FROM ({output_select_sql(path)})"
        ).fetchone()[0]

        con.execute(
            f"""
            COPY ({output_select_sql(path)})
            TO {sql_literal(tmp_path)} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        tmp_path.replace(out_path)

        total_input_rows += input_rows
        total_output_rows += output_rows
        print(f"Wrote {out_path.name}: {output_rows:,}/{input_rows:,} rows retained")

    print(f"Total rows retained: {total_output_rows:,}/{total_input_rows:,}")


def audit_outputs(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    """Verify no stale series leaked into the output."""
    leaked_series = con.execute(
        f"""
        SELECT COUNT(DISTINCT (t.ARTIKEL_ID, t.MARKT_ID))
        FROM {read_parquet_expr(out_dir / 'transactions_year_*.parquet')} t
        INNER JOIN stale_series stale USING (ARTIKEL_ID, MARKT_ID)
        """
    ).fetchone()[0]
    if leaked_series:
        raise RuntimeError(
            f"Stale-series output audit failed: {leaked_series:,} stale series leaked"
        )


def main(
    in_dir: Path = IN_DIR,
    out_dir: Path = OUT_DIR,
    *,
    max_days_since_last_sale: int = MAX_DAYS_SINCE_LAST_SALE,
    min_last_year_demand_active_day_share: float = (
        MIN_LAST_YEAR_DEMAND_ACTIVE_DAY_SHARE
    ),
    min_demand_days: int = MIN_DEMAND_DAYS,
    threads: int = 8,
) -> None:
    """Remove stale/sparse product-store series and write yearly parquet outputs."""
    input_files = require_parquet_files(in_dir)
    input_glob = in_dir / "transactions_year_*.parquet"
    con = configure_duckdb(threads)

    print(f"Reading daily data from {input_glob}")
    print(f"Writing stale-filtered data to {out_dir}")
    print(
        "Removing series with last positive sale more than "
        f"{max_days_since_last_sale:,} days before the dataset max date"
    )
    print(
        "Removing series with last-year demand/active day share below "
        f"{min_last_year_demand_active_day_share:.1%}"
    )
    print(f"Removing series with fewer than {min_demand_days:,} demand days")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW source_rows AS
        SELECT * FROM {read_parquet_expr(input_glob)}
        """
    )
    create_stale_series_table(
        con,
        input_glob,
        max_days_since_last_sale=max_days_since_last_sale,
        min_last_year_demand_active_day_share=(
            min_last_year_demand_active_day_share
        ),
        min_demand_days=min_demand_days,
    )
    print_filter_summary(con)
    write_outputs(con, input_files, out_dir)
    audit_outputs(con, out_dir)
    print("Stale-series output audit passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--max-days-since-last-sale",
        type=int,
        default=MAX_DAYS_SINCE_LAST_SALE,
    )
    parser.add_argument(
        "--min-last-year-demand-active-day-share",
        type=float,
        default=MIN_LAST_YEAR_DEMAND_ACTIVE_DAY_SHARE,
    )
    parser.add_argument("--min-demand-days", type=int, default=MIN_DEMAND_DAYS)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        max_days_since_last_sale=args.max_days_since_last_sale,
        min_last_year_demand_active_day_share=(
            args.min_last_year_demand_active_day_share
        ),
        min_demand_days=args.min_demand_days,
        threads=args.threads,
    )
