"""Expand daily aggregate sales over a complete calendar.

The input is daily sales per (ARTIKEL_ID, MARKT_ID, DATE).  For every
article/store pair, this script creates one row for every calendar date from its
first positive sale through the latest date in the input transaction table.
Missing sales are filled with zero sales and zero flags. All Sundays and public
holidays are retained and marked with ``is_active`` and ``reason_closed`` so
downstream calendar lags remain date-correct while demand analyses can exclude
days on which a store was closed.
The pooled ``is_fcm`` and ``is_pseudo`` markers are copied to every generated
row in their series.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sys
from time import perf_counter

import duckdb
import pandas as pd
import holidays

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import ROOT, sql_literal, step


IN_DIR = ROOT / "data" / "interim" / "transactions_daily_agg"
OUT_DIR = ROOT / "data" / "interim" / "transactions_dst_over_days"

KEY_COLS = ["ARTIKEL_ID", "MARKT_ID", "DATE"]
DEMAND_COL = "ABVERKAUFTE_MENGE_KG"
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
TYPE_COLS = ["is_fcm", "is_pseudo"]
CALENDAR_COLS = ["is_active", "reason_closed"]
OUTPUT_COLS = KEY_COLS + CALENDAR_COLS + SUM_COLS + TYPE_COLS + FLAG_COLS + STATIC_COLS
INPUT_COLS = KEY_COLS + SUM_COLS + TYPE_COLS + FLAG_COLS + STATIC_COLS

HOLIDAY_COUNTRY = "DE"
HOLIDAY_SUBDIVISION = "NI"  # Niedersachsen — retained as the default only

# The store network spans two Bundeslaender, and their public-holiday calendars
# differ. Applying the Niedersachsen calendar to every store corrupts the data in
# both directions: on NW-only holidays (Fronleichnam, Allerheiligen) the NW stores
# are shut but marked active, and on NI-only holidays (Reformationstag) the NW
# stores are open but forced inactive, which zeroes their real sales. Measured on
# the 2026 evaluation window, the single Fronleichnam misclassification cost
# 0.241 pp of row WAPE; the Reformationstag one destroys ~110 real store-days a
# year for 57 stores whose median daily volume is 54 kg.
MARKET_PATH = ROOT / "data" / "raw" / "maerkte" / "maerkte.csv"
HOLIDAY_SUBDIVISIONS = ("NI", "NW")
# Two-digit postal prefixes that fall in Nordrhein-Westfalen.
NW_PLZ_PREFIXES = frozenset({32, 33, 48, 57, 59})
# Border municipalities whose prefix contradicts their Bundesland. Each was
# cross-checked against observed store closures on NW-only holidays.
PLZ_SUBDIVISION_OVERRIDES = {
    48499: "NI",  # Salzbergen, Emsland
    48488: "NI",  # Emsbueren, Emsland
    49469: "NW",  # Ibbenbueren, Kreis Steinfurt
}


def subdivision_for_postal_code(postal_code: object) -> str:
    """Return the Bundesland subdivision code for a German postal code."""
    code = int(postal_code)
    override = PLZ_SUBDIVISION_OVERRIDES.get(code)
    if override is not None:
        return override
    return "NW" if code // 1000 in NW_PLZ_PREFIXES else "NI"


def store_subdivisions(market_path: Path = MARKET_PATH) -> pd.DataFrame:
    """Return one Bundesland subdivision code per store."""
    markets = pd.read_csv(market_path, usecols=["MARKT_ID", "PLZ"]).dropna(
        subset=["PLZ"]
    )
    markets["subdivision"] = markets["PLZ"].map(subdivision_for_postal_code)
    return markets[["MARKT_ID", "subdivision"]].drop_duplicates("MARKT_ID")


def create_germany_ni_holidays(years: range):
    """Create the German Niedersachsen holiday calendar for the requested years."""
    if hasattr(holidays, "country_holidays"):
        return holidays.country_holidays(
            HOLIDAY_COUNTRY,
            subdiv=HOLIDAY_SUBDIVISION,
            years=years,
        )
    return holidays.Germany(subdiv=HOLIDAY_SUBDIVISION, years=years)


def create_germany_holidays(subdivision: str, years: range):
    """Create the German public-holiday calendar of one Bundesland."""
    if hasattr(holidays, "country_holidays"):
        return holidays.country_holidays(
            HOLIDAY_COUNTRY, subdiv=subdivision, years=years
        )
    return holidays.Germany(subdiv=subdivision, years=years)


def build_calendar(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Return all dates with Sunday and Niedersachsen-holiday flags."""
    all_dates = pd.date_range(start_date, end_date, freq="D")
    years = range(start_date.year, end_date.year + 1)
    holiday_dates = set(create_germany_ni_holidays(years).keys())

    calendar = pd.DataFrame({"DATE_D": all_dates})
    calendar["IS_SUNDAY"] = calendar["DATE_D"].dt.dayofweek == 6
    calendar["IS_HOLIDAY"] = calendar["DATE_D"].dt.date.isin(holiday_dates)
    calendar["YEAR"] = calendar["DATE_D"].dt.year
    calendar["DATE_D"] = calendar["DATE_D"].dt.date
    return calendar


def build_state_calendar(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    subdivisions: Iterable[str] = HOLIDAY_SUBDIVISIONS,
) -> pd.DataFrame:
    """Return one Sunday/holiday calendar per Bundesland subdivision."""
    all_dates = pd.date_range(start_date, end_date, freq="D")
    years = range(start_date.year, end_date.year + 1)
    frames = []
    for subdivision in subdivisions:
        holiday_dates = set(create_germany_holidays(subdivision, years).keys())
        calendar = pd.DataFrame({"DATE_D": all_dates})
        calendar["subdivision"] = subdivision
        calendar["IS_SUNDAY"] = calendar["DATE_D"].dt.dayofweek == 6
        calendar["IS_HOLIDAY"] = calendar["DATE_D"].dt.date.isin(holiday_dates)
        calendar["YEAR"] = calendar["DATE_D"].dt.year
        calendar["DATE_D"] = calendar["DATE_D"].dt.date
        frames.append(calendar)
    return pd.concat(frames, ignore_index=True)


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
            is_fcm,
            is_pseudo,
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
    missing = sorted(set(INPUT_COLS) - columns)
    if missing:
        raise ValueError(f"Input parquet files are missing columns: {missing}")


def create_series_table(
    con: duckdb.DuckDBPyConnection,
    global_end_date: pd.Timestamp,
) -> None:
    """Create one date range per pair from first sale to global input end date."""
    static_select = ",\n            ".join(
        f"arg_min({col}, DATE_D) AS {col}" for col in STATIC_COLS
    )
    global_end_date_sql = sql_literal(pd.Timestamp(global_end_date).date().isoformat())
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE series AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            MIN(
                CASE
                    WHEN COALESCE({DEMAND_COL}, 0.0) > 0 THEN DATE_D
                    ELSE NULL
                END
            ) AS START_DATE,
            CAST({global_end_date_sql} AS DATE) AS END_DATE,
            BOOL_OR(is_fcm) AS is_fcm,
            BOOL_OR(is_pseudo) AS is_pseudo,
            {static_select}
        FROM source
        GROUP BY ARTIKEL_ID, MARKT_ID
        HAVING START_DATE IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE series_state AS
        SELECT s.*, COALESCE(m.subdivision, 'NI') AS subdivision
        FROM series AS s
        LEFT JOIN store_subdivision AS m USING (MARKT_ID)
        """
    )


def output_select_sql(year: int) -> str:
    """Build the yearly SQL query that retains every calendar date."""
    static_cols = ",\n            ".join(f"s.{col}" for col in STATIC_COLS)
    return f"""
        SELECT
            s.ARTIKEL_ID,
            s.MARKT_ID,
            strftime(c.DATE_D, '%Y-%m-%d') AS DATE,
            (
                NOT c.IS_HOLIDAY
                AND NOT c.IS_SUNDAY
            ) AS is_active,
            CASE
                WHEN c.IS_HOLIDAY THEN 'Holiday'
                WHEN c.IS_SUNDAY THEN 'Sunday'
                ELSE NULL
            END AS reason_closed,
            CASE WHEN is_active THEN COALESCE(src.UMS_MENGE, 0.0) ELSE 0.0 END
                ::DOUBLE AS UMS_MENGE,
            CASE
                WHEN is_active THEN COALESCE(src.ABVERKAUFTE_MENGE_KG, 0.0)
                ELSE 0.0
            END::DOUBLE AS ABVERKAUFTE_MENGE_KG,
            CASE WHEN is_active THEN COALESCE(src.UMS_VK_WERT, 0.0) ELSE 0.0 END
                ::DOUBLE AS UMS_VK_WERT,
            s.is_fcm,
            s.is_pseudo,
            CASE WHEN is_active THEN COALESCE(src.AKTION_KENNZEICHEN, 0) ELSE 0 END
                ::TINYINT AS AKTION_KENNZEICHEN,
            CASE WHEN is_active THEN COALESCE(src.RABATT, 0) ELSE 0 END
                ::TINYINT AS RABATT,
            CASE WHEN is_active THEN COALESCE(src.ARTIKELRABATT, 0) ELSE 0 END
                ::TINYINT AS ARTIKELRABATT,
            {static_cols}
        FROM series_state s
        JOIN calendar c
            ON c.subdivision = s.subdivision
            AND c.DATE_D BETWEEN s.START_DATE AND s.END_DATE
        LEFT JOIN source src
            ON src.ARTIKEL_ID = s.ARTIKEL_ID
            AND src.MARKT_ID = s.MARKT_ID
            AND src.DATE_D = c.DATE_D
        WHERE c.YEAR = {year}
        """


def write_outputs(con: duckdb.DuckDBPyConnection, years: list[int], out_dir: Path) -> None:
    """Write one expanded parquet file per calendar year."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("transactions_year_*.parquet"):
        old_file.unlink()

    for year in years:
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

    subdivisions = store_subdivisions()
    con.register("store_subdivision_df", subdivisions)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE store_subdivision AS "
        "SELECT * FROM store_subdivision_df"
    )
    calendar = build_state_calendar(pd.Timestamp(min_date), pd.Timestamp(max_date))
    con.register("calendar_df", calendar)
    con.execute("CREATE OR REPLACE TEMP TABLE calendar AS SELECT * FROM calendar_df")
    years = sorted(calendar["YEAR"].unique().tolist())
    counts = subdivisions.subdivision.value_counts().to_dict()
    t0 = step(
        f"Created per-Bundesland calendars for {len(years)} years "
        f"({', '.join(f'{k}: {v} stores' for k, v in sorted(counts.items()))})",
        t0,
    )

    create_series_table(con, max_date)
    series_count = con.execute("SELECT COUNT(*) FROM series").fetchone()[0]
    t0 = step(
        f"Created {series_count:,} article/store active periods through {max_date}",
        t0,
    )

    write_outputs(con, years, out_dir)


if __name__ == "__main__":
    main()
