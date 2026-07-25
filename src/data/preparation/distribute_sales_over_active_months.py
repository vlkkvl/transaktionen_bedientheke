"""Aggregate calendar-day sales to active months.

The input is the final daily calendar dataset after tail, minimum-demand,
and outlier filtering.
Because that dataset contains every calendar date plus an ``is_active`` flag, this
script keeps active months with no sales as zero-demand monthly rows.

Output ``DATE`` is the first day of the month.
"""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import duckdb

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import ROOT, sql_literal, step


IN_DIR = ROOT / "data" / "interim" / "transactions_dst_daily_min_demand_no_outliers"
OUT_DIR = ROOT / "data" / "interim" / "transactions_dst_over_months"

KEY_COLS = ["ARTIKEL_ID", "MARKT_ID", "DATE"]
SUM_COLS = ["UMS_MENGE", "ABVERKAUFTE_MENGE_KG", "UMS_VK_WERT"]
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
TYPE_COLS = ["is_fcm", "is_pseudo"]
CALENDAR_COLS = ["is_active", "reason_closed"]
INPUT_COLS = KEY_COLS + CALENDAR_COLS + SUM_COLS + TYPE_COLS + FLAG_COLS + STATIC_COLS

PERIOD_START_SQL = "date_trunc('month', DATE_D)::DATE"


def validate_input_schema(con: duckdb.DuckDBPyConnection, input_glob: str) -> None:
    """Ensure all columns required by the monthly output exist in the inputs."""
    columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_literal(input_glob)})"
        ).fetchall()
    }
    missing = sorted(set(INPUT_COLS) - columns)
    if missing:
        raise ValueError(f"Input parquet files are missing columns: {missing}")


def create_source_view(con: duckdb.DuckDBPyConnection, input_glob: str) -> None:
    """Create a normalized DuckDB view over the daily calendar parquet inputs."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW source AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            CAST(DATE AS DATE) AS DATE_D,
            is_active,
            reason_closed,
            UMS_MENGE,
            ABVERKAUFTE_MENGE_KG,
            UMS_VK_WERT,
            is_fcm,
            is_pseudo,
            AKTION_KENNZEICHEN,
            RABATT,
            ARTIKELRABATT,
            {", ".join(STATIC_COLS)}
        FROM read_parquet({sql_literal(input_glob)})
        """
    )


def period_years(con: duckdb.DuckDBPyConnection) -> list[int]:
    """Return the years present in the monthly period-start dates."""
    return [
        row[0]
        for row in con.execute(
            f"""
            SELECT DISTINCT
                EXTRACT(year FROM {PERIOD_START_SQL})::INTEGER AS PERIOD_YEAR
            FROM source
            ORDER BY PERIOD_YEAR
            """
        ).fetchall()
    ]


def output_select_sql(year: int) -> str:
    """Build the yearly SQL query for the monthly parquet output."""
    static_select = ",\n            ".join(
        f"arg_min({col}, DATE_D) AS {col}" for col in STATIC_COLS
    )
    return f"""
        WITH normalized AS (
            SELECT
                *,
                {PERIOD_START_SQL} AS PERIOD_START
            FROM source
            WHERE EXTRACT(year FROM {PERIOD_START_SQL})::INTEGER = {year}
        )
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            strftime(PERIOD_START, '%Y-%m-%d') AS DATE,
            SUM(COALESCE(UMS_MENGE, 0.0))::DOUBLE AS UMS_MENGE,
            SUM(COALESCE(ABVERKAUFTE_MENGE_KG, 0.0))::DOUBLE AS ABVERKAUFTE_MENGE_KG,
            SUM(COALESCE(UMS_VK_WERT, 0.0))::DOUBLE AS UMS_VK_WERT,
            BOOL_OR(is_fcm) AS is_fcm,
            BOOL_OR(is_pseudo) AS is_pseudo,
            MAX(COALESCE(AKTION_KENNZEICHEN, 0))::TINYINT AS AKTION_KENNZEICHEN,
            MAX(COALESCE(RABATT, 0))::TINYINT AS RABATT,
            MAX(COALESCE(ARTIKELRABATT, 0))::TINYINT AS ARTIKELRABATT,
            COUNT_IF(is_active)::INTEGER AS active_days_in_month,
            COUNT_IF(NOT is_active)::INTEGER AS closed_days_in_month,
            {static_select}
        FROM normalized
        GROUP BY ARTIKEL_ID, MARKT_ID, PERIOD_START
        """


def write_outputs(con: duckdb.DuckDBPyConnection) -> None:
    """Write one monthly parquet file per period-start year."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUT_DIR.glob("transactions_year_*.parquet"):
        old_file.unlink()

    for year in period_years(con):
        out_file = OUT_DIR / f"transactions_year_{year}.parquet"
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


def main() -> None:
    files = sorted(IN_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {IN_DIR}")

    input_glob = str(IN_DIR / "*.parquet")
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")

    t0 = perf_counter()
    print(f"Reading daily calendar data from {input_glob}")
    validate_input_schema(con, input_glob)
    create_source_view(con, input_glob)
    t0 = step("Created source view", t0)

    write_outputs(con)
    step("Wrote active-month outputs", t0)


if __name__ == "__main__":
    main()
