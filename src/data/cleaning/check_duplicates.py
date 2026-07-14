"""Export duplicate transaction rows after applying article filters.

A duplicate is a row sharing the same ARTIKEL_ID, MARKT_ID, BON_ID, DATE, TIME,
and UMS_MENGE as another row. This step writes only duplicate diagnostics; the
full de-duplicated transaction table is not materialized.
"""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import (
    ROOT,
    configure_duckdb,
    parquet_files,
    read_parquet_expr,
    sql_literal,
    step,
)
from src.data.cleaning.rules import (
    DUPLICATE_KEY_COLS,
    FCM_RULE,
    duplicate_keys_sql,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_per_year_filtered"
BASE_IN_DIR = ROOT / "data" / "interim" / "transactions_per_year_filtered_no_fcm"
DUP_OUT_DIR = ROOT / "data" / "interim" / "transactions_duplicates"
DUP_OUT_FILE = DUP_OUT_DIR / "duplicates_transactions_5_years.csv"


def default_input_dir() -> Path:
    if FCM_RULE and parquet_files(BASE_IN_DIR):
        return BASE_IN_DIR
    return IN_DIR


def main() -> None:
    in_dir = default_input_dir()
    input_glob = in_dir / "transactions_year_*.parquet"
    if not parquet_files(in_dir):
        raise FileNotFoundError(f"No parquet files found in {in_dir}")

    DUP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = configure_duckdb()
    con.execute("PRAGMA disable_progress_bar")
    read_expr = read_parquet_expr(input_glob, filename=True)
    keys_sql = duplicate_keys_sql()

    print(f"Scanning filtered transactions from {in_dir}")
    print(f"Duplicate key columns: {DUPLICATE_KEY_COLS}")

    t0 = perf_counter()
    print("\n[1/3] Counting filtered rows ...")
    total = con.execute(f"SELECT COUNT(*) FROM {read_expr}").fetchone()[0]
    t0 = step(f"Total filtered rows: {total:,}", t0)

    print("\n[2/3] Building duplicate-key table ...")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE dup_keys AS
        SELECT {keys_sql}, COUNT(*) AS n
        FROM {read_expr}
        GROUP BY {keys_sql}
        HAVING COUNT(*) > 1
        """
    )
    n_groups = con.execute("SELECT COUNT(*) FROM dup_keys").fetchone()[0]
    n_dup_rows = con.execute("SELECT COALESCE(SUM(n), 0) FROM dup_keys").fetchone()[0]
    max_rep = con.execute("SELECT COALESCE(MAX(n), 0) FROM dup_keys").fetchone()[0]
    t0 = step(
        "Duplicate scan finished: "
        f"{n_groups:,} key groups, {n_dup_rows:,} rows, max repetitions {max_rep:,}",
        t0,
    )

    if n_groups:
        print("\n[3/3] Writing duplicate diagnostics ...")
        con.execute(
            f"""
            COPY (
                SELECT
                    {duplicate_keys_sql("t")},
                    t.filename AS __source_file
                FROM {read_expr} t
                SEMI JOIN dup_keys d USING ({keys_sql})
                ORDER BY {keys_sql}
            ) TO {sql_literal(DUP_OUT_FILE)} (HEADER, DELIMITER ',')
            """
        )
        step(f"Wrote duplicates to {DUP_OUT_FILE}", t0)
    else:
        if DUP_OUT_FILE.exists():
            DUP_OUT_FILE.unlink()
        print("\n[3/3] No duplicates found. No duplicate CSV written.")

    print("\nSummary")
    print(f"  total filtered rows:       {total:,}")
    print(f"  duplicate key groups:      {n_groups:,}")
    print(f"  rows in duplicate groups:  {n_dup_rows:,}")
    print(f"  max repetitions per key:   {max_rep:,}")
    print(f"  duplicate csv:             {DUP_OUT_FILE}")


if __name__ == "__main__":
    main()
