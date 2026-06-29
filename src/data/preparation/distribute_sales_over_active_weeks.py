"""Aggregate active-day sales to active weeks.

The input is the daily active-day dataset produced by
``distribute_sales_over_active_days.py``. Because that dataset already contains
zero-sale rows for active open days, this script keeps active weeks with no
sales as zero-demand weekly rows.

Output ``DATE`` is the Monday week-start date.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import duckdb


def find_project_root() -> Path:
    """Find the repository root from this nested script location."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = find_project_root()
IN_DIR = ROOT / "data" / "processed" / "transactions_dst_over_days"
OUT_DIR = ROOT / "data" / "processed" / "transactions_dst_over_weeks"

KEY_COLS = ["ARTIKEL_ID", "MARKT_ID", "DATE"]
SUM_COLS = ["UMS_MENGE", "ABVERKAUFTE_MENGE", "UMS_VK_WERT"]
FLAG_COLS = ["AKTION_KENNZEICHEN", "RABATT", "ARTIKELRABATT"]
STATIC_COLS = [
    "EAN_ID",
    "ARTIKEL_BEZ",
    "ARTIKEL_INHALT",
    "VERKAUFSEINHEIT",
    "GEWICHTSARTIKEL",
    "MARKT_NR",
    "MANDANT_ID",
    "LEH_SEH",
    "WGR_ID",
    "N_WARENKLASSE_KBEZ",
]
OUTPUT_COLS = KEY_COLS + SUM_COLS + FLAG_COLS + STATIC_COLS

PERIOD_START_SQL = "date_trunc('week', DATE_D)::DATE"


def sql_literal(value: str | Path) -> str:
    """Escape a value for use as a single-quoted DuckDB SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def step(message: str, t0: float) -> float:
    """Print an elapsed-time message and return the current timestamp."""
    t1 = perf_counter()
    print(f"{message} ({t1 - t0:.1f}s)")
    return t1


def validate_input_schema(con: duckdb.DuckDBPyConnection, input_glob: str) -> None:
    """Ensure all columns required by the weekly output exist in the inputs."""
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
    """Create a normalized DuckDB view over the daily active-day parquet inputs."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW source AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            CAST(DATE AS DATE) AS DATE_D,
            UMS_MENGE,
            ABVERKAUFTE_MENGE,
            UMS_VK_WERT,
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
            SUM(COALESCE(ABVERKAUFTE_MENGE, 0.0))::DOUBLE AS ABVERKAUFTE_MENGE,
            SUM(COALESCE(UMS_VK_WERT, 0.0))::DOUBLE AS UMS_VK_WERT,
            MAX(COALESCE(AKTION_KENNZEICHEN, 0))::TINYINT AS AKTION_KENNZEICHEN,
            MAX(COALESCE(RABATT, 0))::TINYINT AS RABATT,
            MAX(COALESCE(ARTIKELRABATT, 0))::TINYINT AS ARTIKELRABATT,
            {static_select}
        FROM normalized
        GROUP BY ARTIKEL_ID, MARKT_ID, PERIOD_START
        """


def write_outputs(con: duckdb.DuckDBPyConnection) -> None:
    """Write one weekly parquet file per period-start year."""
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
    print(f"Reading daily active-day data from {input_glob}")
    validate_input_schema(con, input_glob)
    create_source_view(con, input_glob)
    t0 = step("Created source view", t0)

    write_outputs(con)
    step("Wrote active-week outputs", t0)


if __name__ == "__main__":
    main()
