"""Aggregate active-day sales to active weeks.

The input is the final daily active-day dataset after tail, minimum-demand,
and outlier filtering.
Because that dataset already contains zero-sale rows for active open days, this
script keeps active weeks with no sales as zero-demand weekly rows.

Output ``DATE`` is the Monday week-start date.
The weekly output also keeps daily-count diagnostics:
``active_days_in_week``, ``demand_days_in_week``, and
``zero_days_in_week``.
** active_days_in_week = demand_days_in_week + zero_days_in_week
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
OUT_DIR = ROOT / "data" / "interim" / "transactions_dst_over_weeks"

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
INPUT_COLS = KEY_COLS + SUM_COLS + TYPE_COLS + FLAG_COLS + STATIC_COLS

PERIOD_START_SQL = "date_trunc('week', DATE_D)::DATE"


def validate_input_schema(con: duckdb.DuckDBPyConnection, input_glob: str) -> None:
    """Ensure all columns required by the weekly aggregation exist in the inputs."""
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
    """Create a normalized DuckDB view over the daily active-day parquet inputs."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW source AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            CAST(DATE AS DATE) AS DATE_D,
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
    """Return the years present in the weekly period-start dates."""
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
    """Build the yearly SQL query for the weekly parquet output."""
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
            COUNT(*)::INTEGER AS active_days_in_week,
            SUM(
                CASE
                    WHEN COALESCE(ABVERKAUFTE_MENGE_KG, 0.0) > 0 THEN 1
                    ELSE 0
                END
            )::INTEGER AS demand_days_in_week,
            (
                COUNT(*) - SUM(
                    CASE
                        WHEN COALESCE(ABVERKAUFTE_MENGE_KG, 0.0) > 0 THEN 1
                        ELSE 0
                    END
                )
            )::INTEGER AS zero_days_in_week,
            {static_select}
        FROM normalized
        GROUP BY ARTIKEL_ID, MARKT_ID, PERIOD_START
        """


def write_outputs(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    """Write one weekly parquet file per period-start year."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("transactions_year_*.parquet"):
        old_file.unlink()

    for year in period_years(con):
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


def main(in_dir: Path = IN_DIR, out_dir: Path = OUT_DIR) -> None:
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {in_dir}")

    input_glob = str(in_dir / "*.parquet")
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")

    t0 = perf_counter()
    print(f"Reading daily active-day data from {input_glob}")
    validate_input_schema(con, input_glob)
    create_source_view(con, input_glob)
    t0 = step("Created source view", t0)

    write_outputs(con, out_dir)
    step("Wrote active-week outputs", t0)


if __name__ == "__main__":
    main()
