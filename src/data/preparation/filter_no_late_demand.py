"""Filter daily FCM series whose current no-demand tail is historically unusual.

The input is the active-day FCM dataset produced by
``distribute_sales_over_active_days.py``. A series is kept only when it has at
least ``MIN_DEMAND_PERIODS_OVERALL`` positive demand days overall and its
current gap since the last sale is not larger than a factor times its own
maximum historical gap between sales.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import duckdb

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import ROOT, clear_parquet_outputs, sql_literal, step


IN_DIR = ROOT / "data" / "processed" / "transactions_dst_over_days_fcm"
OUT_DIR = ROOT / "data" / "interim" / "transactions_dst_daily_no_tail_fcm"

MIN_DEMAND_PERIODS_OVERALL = 10

DEMAND_COL = "ABVERKAUFTE_MENGE_KG"
GROUP_COLS = ["ARTIKEL_ID", "MARKT_ID"]
KEY_COLS = [*GROUP_COLS, "DATE"]
SUM_COLS = ["UMS_MENGE", DEMAND_COL, "UMS_VK_WERT"]
FLAG_COLS = ["AKTION_KENNZEICHEN", "RABATT", "ARTIKELRABATT"]
STATIC_COLS = [
    "ARTIKEL_BEZ",
    "ARTIKEL_INHALT",
    "VERKAUFSEINHEIT",
    "GEWICHT_FLAG",
    "GEWICHTSARTIKEL",
    "MARKT_NR",
    "MANDANT_ID",
    "LEH_SEH",
    "WGR_ID",
    "N_WARENKLASSE_KBEZ",
]
OUTPUT_COLS = KEY_COLS + SUM_COLS + FLAG_COLS + STATIC_COLS

DEFAULT_CURRENT_TO_HISTORICAL_GAP_FACTOR = 3.75


def validate_input_schema(con: duckdb.DuckDBPyConnection, input_glob: str) -> None:
    """Ensure all columns required by the filtered output exist in the input."""
    columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_literal(input_glob)})"
        ).fetchall()
    }
    missing = sorted(set(OUTPUT_COLS) - columns)
    if missing:
        raise ValueError(f"Input parquet files are missing columns: {missing}")


def create_source_view(con: duckdb.DuckDBPyConnection, input_glob: str) -> None:
    """Create a normalized DuckDB view over daily active-day FCM parquet files."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW source AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            CAST(DATE AS DATE) AS DATE_D,
            UMS_MENGE,
            {DEMAND_COL},
            UMS_VK_WERT,
            AKTION_KENNZEICHEN,
            RABATT,
            ARTIKELRABATT,
            {", ".join(STATIC_COLS)}
        FROM read_parquet({sql_literal(input_glob)})
        WHERE ARTIKEL_ID IS NOT NULL
          AND MARKT_ID IS NOT NULL
          AND DATE IS NOT NULL
        """
    )


def create_series_tables(
    con: duckdb.DuckDBPyConnection,
    *,
    min_demand_periods: int,
    gap_factor: float,
) -> None:
    """Create tables with eligible, removed, and kept series."""
    global_data_stand = con.execute("SELECT MAX(DATE_D) FROM source").fetchone()[0]
    if global_data_stand is None:
        raise ValueError("Input parquet files do not contain any usable rows")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE series_metrics AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            MIN(DATE_D) AS erster_tag,
            MAX(DATE_D) AS letzter_tag,
            MIN(CASE WHEN COALESCE({DEMAND_COL}, 0.0) > 0 THEN DATE_D END)
                AS erster_verkauf,
            MAX(CASE WHEN COALESCE({DEMAND_COL}, 0.0) > 0 THEN DATE_D END)
                AS letzter_verkauf,
            COUNT(*)::BIGINT AS aktive_tage,
            SUM(
                CASE
                    WHEN COALESCE({DEMAND_COL}, 0.0) > 0 THEN 1
                    ELSE 0
                END
            )::BIGINT AS nachfrageperioden
        FROM source
        GROUP BY ARTIKEL_ID, MARKT_ID
        HAVING letzter_verkauf IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE eligible_series AS
        SELECT *
        FROM series_metrics
        WHERE nachfrageperioden >= {int(min_demand_periods)}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE positive_days AS
        SELECT DISTINCT
            s.ARTIKEL_ID,
            s.MARKT_ID,
            s.DATE_D AS verkaufstag
        FROM source s
        INNER JOIN eligible_series e
            ON s.ARTIKEL_ID = e.ARTIKEL_ID
           AND s.MARKT_ID = e.MARKT_ID
        WHERE COALESCE(s.{DEMAND_COL}, 0.0) > 0
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE historical_gaps AS
        WITH ordered AS (
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                verkaufstag,
                LAG(verkaufstag) OVER (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY verkaufstag
                ) AS vorheriger_verkaufstag
            FROM positive_days
        )
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            (date_diff('day', vorheriger_verkaufstag, verkaufstag) - 1)::DOUBLE
                AS historische_luecke_tage
        FROM ordered
        WHERE vorheriger_verkaufstag IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE historical_gap_summary AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            COUNT(*)::BIGINT AS historische_luecken,
            MAX(historische_luecke_tage)::DOUBLE AS maximale_historische_luecke
        FROM historical_gaps
        GROUP BY ARTIKEL_ID, MARKT_ID
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE series_gap_filter AS
        SELECT
            e.*,
            date_diff('day', e.letzter_verkauf, DATE {sql_literal(str(global_data_stand))})
                ::DOUBLE AS aktuelle_luecke_tage,
            COALESCE(h.historische_luecken, 0)::BIGINT AS historische_luecken,
            h.maximale_historische_luecke,
            CASE
                WHEN h.maximale_historische_luecke > 0
                    THEN date_diff(
                        'day',
                        e.letzter_verkauf,
                        DATE {sql_literal(str(global_data_stand))}
                    )::DOUBLE / h.maximale_historische_luecke
                WHEN date_diff(
                        'day',
                        e.letzter_verkauf,
                        DATE {sql_literal(str(global_data_stand))}
                    ) > 0
                    THEN 'Infinity'::DOUBLE
                ELSE 0.0
            END AS luecken_faktor,
            (
                date_diff(
                    'day',
                    e.letzter_verkauf,
                    DATE {sql_literal(str(global_data_stand))}
                )::DOUBLE
                > {float(gap_factor)} * COALESCE(h.maximale_historische_luecke, 0.0)
            ) AS entfernen
        FROM eligible_series e
        LEFT JOIN historical_gap_summary h
            ON e.ARTIKEL_ID = h.ARTIKEL_ID
           AND e.MARKT_ID = h.MARKT_ID
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE kept_series AS
        SELECT ARTIKEL_ID, MARKT_ID
        FROM series_gap_filter
        WHERE NOT entfernen
        """
    )


def source_years(con: duckdb.DuckDBPyConnection) -> list[int]:
    """Return calendar years present in the input source rows."""
    return [
        row[0]
        for row in con.execute(
            """
            SELECT DISTINCT EXTRACT(year FROM DATE_D)::INTEGER AS YEAR
            FROM source
            ORDER BY YEAR
            """
        ).fetchall()
    ]


def output_select_sql(year: int) -> str:
    """Build SQL for one yearly no-tail parquet output."""
    static_cols = ",\n            ".join(f"s.{col}" for col in STATIC_COLS)
    return f"""
        SELECT
            s.ARTIKEL_ID,
            s.MARKT_ID,
            strftime(s.DATE_D, '%Y-%m-%d') AS DATE,
            s.UMS_MENGE,
            s.{DEMAND_COL},
            s.UMS_VK_WERT,
            s.AKTION_KENNZEICHEN,
            s.RABATT,
            s.ARTIKELRABATT,
            {static_cols}
        FROM source s
        INNER JOIN kept_series k
            ON s.ARTIKEL_ID = k.ARTIKEL_ID
           AND s.MARKT_ID = k.MARKT_ID
        WHERE EXTRACT(year FROM s.DATE_D)::INTEGER = {year}
        """


def write_outputs(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    """Write one filtered parquet file per input year."""
    clear_parquet_outputs(out_dir)
    for year in source_years(con):
        out_file = out_dir / f"transactions_year_{year}.parquet"
        t0 = perf_counter()
        con.execute(
            f"""
            COPY ({output_select_sql(year)})
            TO {sql_literal(out_file)}
            (FORMAT PARQUET)
            """
        )
        row_count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_literal(out_file)})"
        ).fetchone()[0]
        print(f"{out_file.name}: {row_count:,} rows ({perf_counter() - t0:.1f}s)")


def print_filter_summary(con: duckdb.DuckDBPyConnection) -> None:
    """Print the series-level effect of the min-demand and no-tail filters."""
    rows_before = con.execute("SELECT COUNT(*) FROM series_metrics").fetchone()[0]
    eligible = con.execute("SELECT COUNT(*) FROM eligible_series").fetchone()[0]
    removed = con.execute(
        "SELECT COUNT(*) FROM series_gap_filter WHERE entfernen"
    ).fetchone()[0]
    kept = con.execute("SELECT COUNT(*) FROM kept_series").fetchone()[0]
    median_factor = con.execute(
        """
        SELECT median(luecken_faktor)
        FROM series_gap_filter
        WHERE entfernen
          AND isfinite(luecken_faktor)
        """
    ).fetchone()[0]
    print("\nSeries filter summary")
    print(f"  before min-demand filter: {rows_before:,}")
    print(f"  after min-demand filter:  {eligible:,}")
    print(f"  removed by no-tail rule:  {removed:,}")
    print(f"  kept for output:          {kept:,}")
    if median_factor is not None:
        print(f"  median removed factor:    {median_factor:.2f}")


def main(
    in_dir: Path = IN_DIR,
    out_dir: Path = OUT_DIR,
    *,
    min_demand_periods: int = MIN_DEMAND_PERIODS_OVERALL,
    gap_factor: float = DEFAULT_CURRENT_TO_HISTORICAL_GAP_FACTOR,
    threads: int = 8,
) -> None:
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {in_dir}")

    input_glob = str(in_dir / "*.parquet")
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("SET preserve_insertion_order=false")

    t0 = perf_counter()
    print(f"Reading daily FCM active-day data from {input_glob}")
    print(f"Writing no-tail daily FCM data to {out_dir}")
    print(f"Minimum overall demand periods: {min_demand_periods}")
    print(f"Current-to-historical gap factor: {gap_factor:g}")
    validate_input_schema(con, input_glob)
    create_source_view(con, input_glob)
    t0 = step("Created source view", t0)

    create_series_tables(
        con,
        min_demand_periods=min_demand_periods,
        gap_factor=gap_factor,
    )
    t0 = step("Computed series-level no-tail filter", t0)
    print_filter_summary(con)

    write_outputs(con, out_dir)
    step("Wrote no-tail daily FCM outputs", t0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--min-demand-periods",
        type=int,
        default=MIN_DEMAND_PERIODS_OVERALL,
        help=(
            "Minimum positive demand days required over the complete observed "
            "series before applying the tail filter."
        ),
    )
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=DEFAULT_CURRENT_TO_HISTORICAL_GAP_FACTOR,
        help=(
            "Remove a series when current_gap > gap_factor * "
            "max_historical_gap for that same series."
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        min_demand_periods=args.min_demand_periods,
        gap_factor=args.gap_factor,
        threads=args.threads,
    )
