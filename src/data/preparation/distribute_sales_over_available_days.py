"""Expand daily aggregate sales over replenishment-supported availability days.

The regular active-day preparation assumes every open day between the first and
last observed sale is an active no-sale day.  This variant uses goods receipts
(``Wareneingaenge``) as an availability proxy:

* days with positive observed sales are kept;
* missing sales days are filled with zero only when the product-store pair had a
  positive goods receipt within ``REPLENISHMENT_LOOKBACK_DAYS`` days;
* days without a positive sale and without recent replenishment are not emitted.
* holidays/weekends are emitted.

The output is intentionally written to a separate directory so notebooks can
compare the broad active-window assumption against the availability proxy.
"""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import duckdb
import holidays
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import ROOT, sql_literal, step


IN_DIR = ROOT / "data" / "interim" / "transactions_daily_agg_no_outliers"
OUT_DIR = ROOT / "data" / "processed" / "transactions_dst_over_days_availability_proxy"
REPLENISHMENT_DIR = ROOT / "data" / "raw" / "wareneingaenge"
REPLENISHMENT_LOOKBACK_DAYS = 7
REPLENISHMENT_FILE_GLOB = "*.csv*"

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
INPUT_COLS = KEY_COLS + SUM_COLS + FLAG_COLS + STATIC_COLS
REPLENISHMENT_COLS = ["ARTIKEL_ID", "MARKT_ID", "DATE", "WE_MENGE_VKE"]

SUNDAY_OPEN_MARKT_IDS = (1100084, 1100079)
HOLIDAY_COUNTRY = "DE"
HOLIDAY_SUBDIVISION = "NI"  # Niedersachsen


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


def validate_input_schema(con: duckdb.DuckDBPyConnection, input_glob: str) -> None:
    """Ensure all columns required by the expanded output exist in the inputs."""
    columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_literal(input_glob)})"
        ).fetchall()
    }
    missing = sorted(set(INPUT_COLS) - columns)
    if missing:
        raise ValueError(f"Input parquet files are missing columns: {missing}")


def validate_replenishment_schema(
    con: duckdb.DuckDBPyConnection,
    replenishment_glob: str,
) -> None:
    """Ensure the goods-receipt files contain the required availability columns."""
    columns = {
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE SELECT *
            FROM read_csv_auto(
                {sql_literal(replenishment_glob)},
                delim=',',
                header=true,
                union_by_name=true
            )
            """
        ).fetchall()
    }
    missing = sorted(set(REPLENISHMENT_COLS) - columns)
    if missing:
        raise ValueError(f"Goods-receipt files are missing columns: {missing}")


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
            ABVERKAUFTE_MENGE_KG,
            UMS_VK_WERT,
            AKTION_KENNZEICHEN,
            RABATT,
            ARTIKELRABATT,
            {", ".join(STATIC_COLS)}
        FROM read_parquet({sql_literal(input_glob)})
        """
    )


def create_requested_pairs_table(con: duckdb.DuckDBPyConnection) -> str:
    """Create requested pairs from all product-store pairs in the source data."""
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE requested_pairs AS
        SELECT DISTINCT ARTIKEL_ID, MARKT_ID
        FROM source
        """
    )
    return "all pairs present in daily aggregate transactions"


def create_series_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create the product-store series to process, scoped to requested pairs."""
    static_select = ",\n            ".join(
        f"arg_min({col}, DATE_D) AS {col}" for col in STATIC_COLS
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE series AS
        SELECT
            s.ARTIKEL_ID,
            s.MARKT_ID,
            MIN(s.DATE_D) AS FIRST_SOURCE_DATE,
            MAX(s.DATE_D) AS LAST_SOURCE_DATE,
            MIN(CASE WHEN s.ABVERKAUFTE_MENGE_KG > 0 THEN s.DATE_D END)
                AS FIRST_POSITIVE_SALE_DATE,
            MAX(CASE WHEN s.ABVERKAUFTE_MENGE_KG > 0 THEN s.DATE_D END)
                AS LAST_POSITIVE_SALE_DATE,
            {static_select}
        FROM source s
        SEMI JOIN requested_pairs p
            ON s.ARTIKEL_ID = p.ARTIKEL_ID
            AND s.MARKT_ID = p.MARKT_ID
        GROUP BY s.ARTIKEL_ID, s.MARKT_ID
        HAVING FIRST_POSITIVE_SALE_DATE IS NOT NULL
        """
    )


def create_replenishment_table(
    con: duckdb.DuckDBPyConnection,
    replenishment_glob: str,
) -> None:
    """Create daily positive replenishment quantities for selected series."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE replenishment_daily AS
        SELECT
            w.ARTIKEL_ID,
            w.MARKT_ID,
            CAST(w.DATE AS DATE) AS DATE_D,
            SUM(CAST(w.WE_MENGE_VKE AS DOUBLE)) AS POSITIVE_WE_MENGE_VKE
        FROM read_csv_auto(
            {sql_literal(replenishment_glob)},
            delim=',',
            header=true,
            union_by_name=true
        ) AS w
        SEMI JOIN series s
            ON w.ARTIKEL_ID = s.ARTIKEL_ID
            AND w.MARKT_ID = s.MARKT_ID
        WHERE CAST(w.WE_MENGE_VKE AS DOUBLE) > 0
        GROUP BY w.ARTIKEL_ID, w.MARKT_ID, CAST(w.DATE AS DATE)
        """
    )


def create_series_bounds_table(
    con: duckdb.DuckDBPyConnection,
    max_source_date: pd.Timestamp,
) -> None:
    """Create date bounds for candidate availability-proxy days."""
    max_source_date_sql = sql_literal(max_source_date.date().isoformat())
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE series_bounds AS
        WITH replenishment_bounds AS (
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                MIN(DATE_D) AS FIRST_POSITIVE_REPLENISHMENT_DATE,
                MAX(DATE_D) AS LAST_POSITIVE_REPLENISHMENT_DATE,
                COUNT(*) AS PAIR_POSITIVE_REPLENISHMENT_DAYS,
                SUM(POSITIVE_WE_MENGE_VKE) AS PAIR_POSITIVE_WE_MENGE_VKE
            FROM replenishment_daily
            GROUP BY ARTIKEL_ID, MARKT_ID
        )
        SELECT
            s.*,
            r.FIRST_POSITIVE_REPLENISHMENT_DATE,
            r.LAST_POSITIVE_REPLENISHMENT_DATE,
            COALESCE(r.PAIR_POSITIVE_REPLENISHMENT_DAYS, 0)
                AS PAIR_POSITIVE_REPLENISHMENT_DAYS,
            COALESCE(r.PAIR_POSITIVE_WE_MENGE_VKE, 0.0)
                AS PAIR_POSITIVE_WE_MENGE_VKE,
            LEAST(
                s.FIRST_POSITIVE_SALE_DATE,
                COALESCE(r.FIRST_POSITIVE_REPLENISHMENT_DATE, s.FIRST_POSITIVE_SALE_DATE)
            ) AS START_DATE,
            LEAST(
                GREATEST(
                    s.LAST_POSITIVE_SALE_DATE,
                    COALESCE(
                        r.LAST_POSITIVE_REPLENISHMENT_DATE
                            + INTERVAL {REPLENISHMENT_LOOKBACK_DAYS} DAY,
                        s.LAST_POSITIVE_SALE_DATE
                    )
                ),
                CAST({max_source_date_sql} AS DATE)
            ) AS END_DATE
        FROM series s
        LEFT JOIN replenishment_bounds r USING (ARTIKEL_ID, MARKT_ID)
        """
    )


def output_select_sql(year: int) -> str:
    """Build the yearly SQL query that fills availability-proxy days."""
    sunday_open_ids = ", ".join(str(x) for x in SUNDAY_OPEN_MARKT_IDS)
    static_cols = ",\n            ".join(f"cd.{col}" for col in STATIC_COLS)
    return f"""
        WITH candidate_days AS (
            SELECT
                s.*,
                c.DATE_D
            FROM series_bounds s
            JOIN calendar c
                ON c.DATE_D BETWEEN s.START_DATE AND s.END_DATE
                AND (NOT c.IS_SUNDAY OR s.MARKT_ID IN ({sunday_open_ids}))
            WHERE c.YEAR = {year}
        ),
        replenishment_availability AS (
            SELECT
                cd.ARTIKEL_ID,
                cd.MARKT_ID,
                cd.DATE_D,
                MAX(r.DATE_D) AS LAST_REPLENISHMENT_DATE,
                COUNT(*) AS POSITIVE_REPLENISHMENT_DAYS_LOOKBACK,
                SUM(r.POSITIVE_WE_MENGE_VKE) AS POSITIVE_WE_MENGE_VKE_LOOKBACK
            FROM candidate_days cd
            JOIN replenishment_daily r
                ON r.ARTIKEL_ID = cd.ARTIKEL_ID
                AND r.MARKT_ID = cd.MARKT_ID
                AND r.DATE_D BETWEEN
                    cd.DATE_D - INTERVAL {REPLENISHMENT_LOOKBACK_DAYS} DAY
                    AND cd.DATE_D
            GROUP BY cd.ARTIKEL_ID, cd.MARKT_ID, cd.DATE_D
        )
        SELECT
            cd.ARTIKEL_ID,
            cd.MARKT_ID,
            strftime(cd.DATE_D, '%Y-%m-%d') AS DATE,
            COALESCE(src.UMS_MENGE, 0.0)::DOUBLE AS UMS_MENGE,
            COALESCE(src.ABVERKAUFTE_MENGE_KG, 0.0)::DOUBLE AS ABVERKAUFTE_MENGE_KG,
            COALESCE(src.UMS_VK_WERT, 0.0)::DOUBLE AS UMS_VK_WERT,
            COALESCE(src.AKTION_KENNZEICHEN, 0)::TINYINT AS AKTION_KENNZEICHEN,
            COALESCE(src.RABATT, 0)::TINYINT AS RABATT,
            COALESCE(src.ARTIKELRABATT, 0)::TINYINT AS ARTIKELRABATT,
            {static_cols},
            CASE
                WHEN COALESCE(src.ABVERKAUFTE_MENGE_KG, 0.0) > 0 THEN 1
                ELSE 0
            END::TINYINT AS HAS_POSITIVE_SALE,
            CASE
                WHEN ra.LAST_REPLENISHMENT_DATE IS NOT NULL THEN 1
                ELSE 0
            END::TINYINT AS HAS_RECENT_REPLENISHMENT,
            ra.LAST_REPLENISHMENT_DATE,
            date_diff('day', ra.LAST_REPLENISHMENT_DATE, cd.DATE_D)::INTEGER
                AS DAYS_SINCE_REPLENISHMENT,
            COALESCE(ra.POSITIVE_REPLENISHMENT_DAYS_LOOKBACK, 0)::INTEGER
                AS POSITIVE_REPLENISHMENT_DAYS_LOOKBACK,
            COALESCE(ra.POSITIVE_WE_MENGE_VKE_LOOKBACK, 0.0)::DOUBLE
                AS POSITIVE_WE_MENGE_VKE_LOOKBACK,
            cd.FIRST_POSITIVE_REPLENISHMENT_DATE,
            cd.LAST_POSITIVE_REPLENISHMENT_DATE,
            cd.PAIR_POSITIVE_REPLENISHMENT_DAYS::INTEGER
                AS PAIR_POSITIVE_REPLENISHMENT_DAYS,
            cd.PAIR_POSITIVE_WE_MENGE_VKE::DOUBLE
                AS PAIR_POSITIVE_WE_MENGE_VKE,
            CASE
                WHEN COALESCE(src.ABVERKAUFTE_MENGE_KG, 0.0) > 0
                    AND ra.LAST_REPLENISHMENT_DATE IS NOT NULL
                    THEN 'sale_and_recent_replenishment'
                WHEN COALESCE(src.ABVERKAUFTE_MENGE_KG, 0.0) > 0
                    THEN 'positive_sale'
                ELSE 'recent_replenishment_zero_sale'
            END AS AVAILABILITY_SOURCE
        FROM candidate_days cd
        LEFT JOIN source src
            ON src.ARTIKEL_ID = cd.ARTIKEL_ID
            AND src.MARKT_ID = cd.MARKT_ID
            AND src.DATE_D = cd.DATE_D
        LEFT JOIN replenishment_availability ra
            ON ra.ARTIKEL_ID = cd.ARTIKEL_ID
            AND ra.MARKT_ID = cd.MARKT_ID
            AND ra.DATE_D = cd.DATE_D
        WHERE COALESCE(src.ABVERKAUFTE_MENGE_KG, 0.0) > 0
            OR ra.LAST_REPLENISHMENT_DATE IS NOT NULL
        """


def write_outputs(con: duckdb.DuckDBPyConnection, years: list[int]) -> None:
    """Write one availability-proxy parquet file per calendar year."""
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
    input_files = sorted(IN_DIR.glob("transactions_year_*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {IN_DIR}")

    replenishment_files = sorted(REPLENISHMENT_DIR.glob(REPLENISHMENT_FILE_GLOB))
    if not replenishment_files:
        raise FileNotFoundError(f"No goods-receipt CSV files found in {REPLENISHMENT_DIR}")

    input_glob = str(IN_DIR / "transactions_year_*.parquet")
    replenishment_glob = str(REPLENISHMENT_DIR / REPLENISHMENT_FILE_GLOB)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("SET preserve_insertion_order=false")

    t0 = perf_counter()
    print(f"Reading daily aggregates from {input_glob}")
    print(f"Reading goods receipts from {replenishment_glob}")
    print(f"Replenishment lookback: {REPLENISHMENT_LOOKBACK_DAYS} days")
    validate_input_schema(con, input_glob)
    validate_replenishment_schema(con, replenishment_glob)
    create_source_view(con, input_glob)
    t0 = step("Created source view", t0)

    pair_source = create_requested_pairs_table(con)
    requested_count = con.execute("SELECT COUNT(*) FROM requested_pairs").fetchone()[0]
    t0 = step(f"Loaded {requested_count:,} requested pairs from {pair_source}", t0)

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
    t0 = step(f"Created {series_count:,} requested article/store sale series", t0)

    create_replenishment_table(con, replenishment_glob)
    replenishment_count = con.execute(
        "SELECT COUNT(*) FROM replenishment_daily"
    ).fetchone()[0]
    replenished_pairs = con.execute(
        "SELECT COUNT(DISTINCT ARTIKEL_ID || '|' || MARKT_ID) FROM replenishment_daily"
    ).fetchone()[0]
    t0 = step(
        f"Created {replenishment_count:,} positive replenishment days "
        f"for {replenished_pairs:,} series",
        t0,
    )

    create_series_bounds_table(con, pd.Timestamp(max_date))
    write_outputs(con, years)
    step(f"Wrote availability-proxy active-day outputs to {OUT_DIR}", t0)


if __name__ == "__main__":
    main()
