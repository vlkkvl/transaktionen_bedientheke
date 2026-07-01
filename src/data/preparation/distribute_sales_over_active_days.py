"""Expand daily aggregate sales over every active open day.

The input is daily sales per (ARTIKEL_ID, MARKT_ID, DATE).  For every
article/store pair, this script creates rows for every open day between its
first and last observed sale date.  Missing sales days are filled with zero
sales and zero flags.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import duckdb
import pandas as pd
import holidays


def find_project_root() -> Path:
    """Find the repository root from this nested script location."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = find_project_root()
IN_DIR = ROOT / "data" / "processed" / "transactions_daily_agg_no_outliers"
OUT_DIR = ROOT / "data" / "processed" / "transactions_dst_over_days"

KEY_COLS = ["ARTIKEL_ID", "MARKT_ID", "DATE"]
SUM_COLS = ["UMS_MENGE", "ABVERKAUFTE_MENGE", "UMS_VK_WERT"]
FLAG_COLS = ["AKTION_KENNZEICHEN", "RABATT", "ARTIKELRABATT"]
STATIC_COLS = [
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

SUNDAY_OPEN_MARKT_IDS = (1100084, 1100079)
HOLIDAY_COUNTRY = "DE"
HOLIDAY_SUBDIVISION = "NI"  # Niedersachsen


def sql_literal(value: str | Path) -> str:
    """Escape a value for use as a single-quoted DuckDB SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def create_germany_ni_holidays(years: range):
    """Create the German Niedersachsen holiday calendar for the requested years."""
    if hasattr(holidays, "country_holidays"):
        return holidays.country_holidays(
            HOLIDAY_COUNTRY,
            subdiv=HOLIDAY_SUBDIVISION,
            years=years,
        )
    return holidays.Germany(subdiv=HOLIDAY_SUBDIVISION, years=years)


def build_calendar(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Return non-holiday dates plus a Sunday flag for store-specific filtering."""
    all_dates = pd.date_range(start_date, end_date, freq="D")
    years = range(start_date.year, end_date.year + 1)
    holiday_dates = set(create_germany_ni_holidays(years).keys())

    calendar = pd.DataFrame({"DATE_D": all_dates})
    calendar["IS_SUNDAY"] = calendar["DATE_D"].dt.dayofweek == 6
    calendar = calendar[~calendar["DATE_D"].dt.date.isin(holiday_dates)].copy()
    calendar["YEAR"] = calendar["DATE_D"].dt.year
    calendar["DATE_D"] = calendar["DATE_D"].dt.date
    return calendar


def step(message: str, t0: float) -> float:
    """Print an elapsed-time message and return the current timestamp."""
    t1 = perf_counter()
    print(f"{message} ({t1 - t0:.1f}s)")
    return t1


def create_source_view(con: duckdb.DuckDBPyConnection, input_glob: str) -> None:
    """Create a normalized DuckDB view over the daily aggregate parquet inputs."""
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


def validate_input_schema(con: duckdb.DuckDBPyConnection, input_glob: str) -> None:
    """Ensure all columns required by the expanded output exist in the inputs."""
    columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_literal(input_glob)})"
        ).fetchall()
    }
    missing = sorted(set(OUTPUT_COLS) - columns)
    if missing:
        raise ValueError(f"Input parquet files are missing columns: {missing}")


def create_series_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create one active date range per article/store pair."""
    static_select = ",\n            ".join(
        f"arg_min({col}, DATE_D) AS {col}" for col in STATIC_COLS
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE series AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            MIN(DATE_D) AS START_DATE,
            MAX(DATE_D) AS END_DATE,
            {static_select}
        FROM source
        GROUP BY ARTIKEL_ID, MARKT_ID
        """
    )


def output_select_sql(year: int) -> str:
    """Build the yearly SQL query that fills missing open days with zero sales."""
    sunday_open_ids = ", ".join(str(x) for x in SUNDAY_OPEN_MARKT_IDS)
    static_cols = ",\n            ".join(f"s.{col}" for col in STATIC_COLS)
    return f"""
        SELECT
            s.ARTIKEL_ID,
            s.MARKT_ID,
            strftime(c.DATE_D, '%Y-%m-%d') AS DATE,
            COALESCE(src.UMS_MENGE, 0.0)::DOUBLE AS UMS_MENGE,
            COALESCE(src.ABVERKAUFTE_MENGE, 0.0)::DOUBLE AS ABVERKAUFTE_MENGE,
            COALESCE(src.UMS_VK_WERT, 0.0)::DOUBLE AS UMS_VK_WERT,
            COALESCE(src.AKTION_KENNZEICHEN, 0)::TINYINT AS AKTION_KENNZEICHEN,
            COALESCE(src.RABATT, 0)::TINYINT AS RABATT,
            COALESCE(src.ARTIKELRABATT, 0)::TINYINT AS ARTIKELRABATT,
            {static_cols}
        FROM series s
        JOIN calendar c
            ON c.DATE_D BETWEEN s.START_DATE AND s.END_DATE
            AND (NOT c.IS_SUNDAY OR s.MARKT_ID IN ({sunday_open_ids}))
        LEFT JOIN source src
            ON src.ARTIKEL_ID = s.ARTIKEL_ID
            AND src.MARKT_ID = s.MARKT_ID
            AND src.DATE_D = c.DATE_D
        WHERE c.YEAR = {year}
        """


def write_outputs(con: duckdb.DuckDBPyConnection, years: list[int]) -> None:
    """Write one expanded parquet file per calendar year."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUT_DIR.glob("transactions_year_*.parquet"):
        old_file.unlink()

    for year in years:
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
    con.execute("PRAGMA threads=8")

    t0 = perf_counter()
    print(f"Reading daily aggregates from {input_glob}")
    validate_input_schema(con, input_glob)
    create_source_view(con, input_glob)
    t0 = step("Created source view", t0)

    min_date, max_date = con.execute(
        "SELECT MIN(DATE_D), MAX(DATE_D) FROM source"
    ).fetchone()
    if min_date is None or max_date is None:
        raise ValueError("Input parquet files do not contain any rows")

    calendar = build_calendar(pd.Timestamp(min_date), pd.Timestamp(max_date))
    con.register("calendar_df", calendar)
    con.execute("CREATE OR REPLACE TEMP TABLE calendar AS SELECT * FROM calendar_df")
    years = sorted(calendar["YEAR"].unique().tolist())
    t0 = step(f"Created Niedersachsen open-day calendar for {len(years)} years", t0)

    create_series_table(con)
    series_count = con.execute("SELECT COUNT(*) FROM series").fetchone()[0]
    t0 = step(f"Created {series_count:,} article/store active periods", t0)

    write_outputs(con, years)


if __name__ == "__main__":
    main()
