"""Keep only daily product-store series with sufficient demand evidence.

The source-specific thresholds are the decisions documented in
``notebooks/00_data_cleaning/00_03_minimum_demand.ipynb``:

- FCM: at least 30 observed active days and 10 positive-demand days.
- Pseudo: at least 120 observed active days and 30 positive-demand days.

Input:  data/interim/transactions_dst_daily_no_tail/*.parquet
Output: data/interim/transactions_dst_daily_min_demand/*.parquet
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


IN_DIR = ROOT / "data" / "interim" / "transactions_dst_daily_no_tail"
OUT_DIR = ROOT / "data" / "interim" / "transactions_dst_daily_min_demand"

DEMAND_COL = "ABVERKAUFTE_MENGE_KG"
FCM_MIN_ACTIVE_DAYS = 30
FCM_MIN_DEMAND_DAYS = 10
PSEUDO_MIN_ACTIVE_DAYS = 120
PSEUDO_MIN_DEMAND_DAYS = 30


def temp_output_path(path: Path) -> Path:
    """Return the temporary path used for an atomic yearly output write."""
    return path.with_name(f".{path.stem}.min_demand.tmp{path.suffix}")


def create_source_view(con: duckdb.DuckDBPyConnection, input_glob: Path) -> None:
    """Create a normalized view with an explicit source class."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW source AS
        SELECT
            *,
            CAST(DATE AS DATE) AS period,
            CAST(COALESCE({DEMAND_COL}, 0) AS DOUBLE) AS demand,
            CASE
                WHEN is_fcm THEN 'FCM'
                WHEN is_pseudo THEN 'Pseudo'
            END AS source_class
        FROM {read_parquet_expr(input_glob)}
        """
    )


def audit_source(con: duckdb.DuckDBPyConnection) -> None:
    """Reject ambiguous flags or source classes that change within a series."""
    row_audit = con.execute(
        """
        SELECT
            COUNT_IF(ARTIKEL_ID IS NULL OR MARKT_ID IS NULL OR period IS NULL),
            COUNT_IF(is_fcm IS NULL OR is_pseudo IS NULL),
            COUNT_IF(is_fcm AND is_pseudo),
            COUNT_IF(source_class IS NULL)
        FROM source
        """
    ).fetchone()
    if any(row_audit):
        raise RuntimeError(
            "Minimum-demand source audit failed: null keys/flags, overlapping "
            "source flags, or unclassified rows are present"
        )

    changing_source_series = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT ARTIKEL_ID, MARKT_ID
            FROM source
            GROUP BY ALL
            HAVING COUNT(DISTINCT source_class) > 1
        )
        """
    ).fetchone()[0]
    if changing_source_series:
        raise RuntimeError(
            f"Source class changes within {changing_source_series:,} series"
        )


def create_series_tables(
    con: duckdb.DuckDBPyConnection,
    *,
    fcm_min_active_days: int,
    fcm_min_demand_days: int,
    pseudo_min_active_days: int,
    pseudo_min_demand_days: int,
) -> None:
    """Calculate whole-series metrics and materialize eligible keys."""
    thresholds = (
        fcm_min_active_days,
        fcm_min_demand_days,
        pseudo_min_active_days,
        pseudo_min_demand_days,
    )
    if any(value < 1 for value in thresholds):
        raise ValueError("All minimum active-day and demand-day thresholds must be positive")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE series_metrics AS
        SELECT
            source_class,
            ARTIKEL_ID,
            MARKT_ID,
            COUNT(*)::BIGINT AS active_days,
            COUNT_IF(demand > 0)::BIGINT AS demand_days,
            SUM(demand)::DOUBLE AS total_demand_kg
        FROM source
        GROUP BY source_class, ARTIKEL_ID, MARKT_ID
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE eligible_series AS
        SELECT ARTIKEL_ID, MARKT_ID
        FROM series_metrics
        WHERE
            (source_class = 'FCM' AND active_days >= ? AND demand_days >= ?)
            OR
            (source_class = 'Pseudo' AND active_days >= ? AND demand_days >= ?)
        """,
        thresholds,
    )


def print_filter_summary(con: duckdb.DuckDBPyConnection) -> None:
    """Print source-specific series and demand-volume retention."""
    summary = con.execute(
        """
        SELECT
            m.source_class,
            COUNT(*) AS series_before,
            COUNT(e.ARTIKEL_ID) AS series_after,
            COUNT(e.ARTIKEL_ID)::DOUBLE / COUNT(*) AS series_retained,
            SUM(m.total_demand_kg) AS demand_kg_before,
            SUM(m.total_demand_kg) FILTER (WHERE e.ARTIKEL_ID IS NOT NULL)
                AS demand_kg_after,
            SUM(m.total_demand_kg) FILTER (WHERE e.ARTIKEL_ID IS NOT NULL)
                / SUM(m.total_demand_kg) AS demand_kg_retained
        FROM series_metrics m
        LEFT JOIN eligible_series e USING (ARTIKEL_ID, MARKT_ID)
        GROUP BY m.source_class
        ORDER BY CASE m.source_class WHEN 'FCM' THEN 1 ELSE 2 END
        """
    ).fetchall()

    print("\nMinimum-demand filter summary")
    for source_class, before, after, series_share, kg_before, kg_after, kg_share in summary:
        print(
            f"  {source_class}: {after:,}/{before:,} series retained "
            f"({series_share:.2%}); {kg_after:,.1f}/{kg_before:,.1f} kg "
            f"retained ({kg_share:.3%})"
        )


def output_select_sql(path: Path) -> str:
    """Return the row-preserving selection for one yearly input file."""
    return f"""
        SELECT t.*
        FROM {read_parquet_expr(path)} t
        INNER JOIN eligible_series e
          ON t.ARTIKEL_ID = e.ARTIKEL_ID
         AND t.MARKT_ID = e.MARKT_ID
    """


def write_outputs(
    con: duckdb.DuckDBPyConnection,
    input_files: list[Path],
    out_dir: Path,
) -> None:
    """Write one filtered parquet per input year."""
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
        print(
            f"Wrote {out_path.name}: {output_rows:,}/{input_rows:,} rows retained"
        )

    print(f"Total rows retained: {total_output_rows:,}/{total_input_rows:,}")


def audit_outputs(
    con: duckdb.DuckDBPyConnection,
    out_dir: Path,
    *,
    fcm_min_active_days: int,
    fcm_min_demand_days: int,
    pseudo_min_active_days: int,
    pseudo_min_demand_days: int,
) -> None:
    """Verify every written series satisfies its source-specific rule."""
    output_glob = out_dir / "transactions_year_*.parquet"
    violations = con.execute(
        f"""
        WITH metrics AS (
            SELECT
                CASE WHEN is_fcm THEN 'FCM' WHEN is_pseudo THEN 'Pseudo' END
                    AS source_class,
                ARTIKEL_ID,
                MARKT_ID,
                COUNT(*) AS active_days,
                COUNT_IF(COALESCE({DEMAND_COL}, 0) > 0) AS demand_days
            FROM {read_parquet_expr(output_glob)}
            GROUP BY source_class, ARTIKEL_ID, MARKT_ID
        )
        SELECT COUNT(*)
        FROM metrics
        WHERE
            (source_class = 'FCM'
             AND (active_days < ? OR demand_days < ?))
            OR
            (source_class = 'Pseudo'
             AND (active_days < ? OR demand_days < ?))
            OR source_class IS NULL
        """,
        (
            fcm_min_active_days,
            fcm_min_demand_days,
            pseudo_min_active_days,
            pseudo_min_demand_days,
        ),
    ).fetchone()[0]
    if violations:
        raise RuntimeError(
            f"Output audit failed: {violations:,} series violate the thresholds"
        )


def main(
    in_dir: Path = IN_DIR,
    out_dir: Path = OUT_DIR,
    *,
    fcm_min_active_days: int = FCM_MIN_ACTIVE_DAYS,
    fcm_min_demand_days: int = FCM_MIN_DEMAND_DAYS,
    pseudo_min_active_days: int = PSEUDO_MIN_ACTIVE_DAYS,
    pseudo_min_demand_days: int = PSEUDO_MIN_DEMAND_DAYS,
) -> None:
    input_files = require_parquet_files(in_dir)
    input_glob = in_dir / "transactions_year_*.parquet"
    con = configure_duckdb()

    print(f"Reading cleaned daily data from {input_glob}")
    print(f"Writing minimum-demand-filtered data to {out_dir}")
    print(
        "Thresholds: "
        f"FCM active>={fcm_min_active_days}, demand>={fcm_min_demand_days}; "
        f"Pseudo active>={pseudo_min_active_days}, demand>={pseudo_min_demand_days}"
    )

    create_source_view(con, input_glob)
    audit_source(con)
    create_series_tables(
        con,
        fcm_min_active_days=fcm_min_active_days,
        fcm_min_demand_days=fcm_min_demand_days,
        pseudo_min_active_days=pseudo_min_active_days,
        pseudo_min_demand_days=pseudo_min_demand_days,
    )
    print_filter_summary(con)
    write_outputs(con, input_files, out_dir)
    audit_outputs(
        con,
        out_dir,
        fcm_min_active_days=fcm_min_active_days,
        fcm_min_demand_days=fcm_min_demand_days,
        pseudo_min_active_days=pseudo_min_active_days,
        pseudo_min_demand_days=pseudo_min_demand_days,
    )
    print("Minimum-demand output audit passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fcm-min-active-days", type=int, default=FCM_MIN_ACTIVE_DAYS)
    parser.add_argument("--fcm-min-demand-days", type=int, default=FCM_MIN_DEMAND_DAYS)
    parser.add_argument(
        "--pseudo-min-active-days", type=int, default=PSEUDO_MIN_ACTIVE_DAYS
    )
    parser.add_argument(
        "--pseudo-min-demand-days", type=int, default=PSEUDO_MIN_DEMAND_DAYS
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        fcm_min_active_days=args.fcm_min_active_days,
        fcm_min_demand_days=args.fcm_min_demand_days,
        pseudo_min_active_days=args.pseudo_min_active_days,
        pseudo_min_demand_days=args.pseudo_min_demand_days,
    )
