"""Check data/interim/transactions_5_years for duplicate rows using DuckDB.

A duplicate is a row sharing the same ARTIKEL_ID, MARKT_ID, BON_ID, DATE,
TIME and UMS_MENGE as another row.
"""
from pathlib import Path
from time import perf_counter

import duckdb

ROOT = Path(__file__).resolve().parents[3]
IN_GLOB = str(ROOT / "data" / "interim" / "transactions_per_year" / "*.parquet")
OUT_DIR = ROOT / "data" / "interim" / "transactions_duplicates"
OUT_FILE = OUT_DIR / "duplicates_transactions_5_years.csv"

KEYS = ["ARTIKEL_ID", "MARKT_ID", "BON_ID", "DATE", "TIME", "UMS_MENGE"]
KEYS_SQL = ", ".join(KEYS)


def step(msg: str, t0: float) -> float:
    t1 = perf_counter()
    print(f"  ... done in {t1 - t0:.1f}s")
    print(msg)
    return t1


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {IN_GLOB}")
    print(f"Key columns: {KEYS}")

    con = duckdb.connect()
    con.execute("PRAGMA enable_progress_bar")
    con.execute("PRAGMA threads=8")

    read_expr = f"read_parquet('{IN_GLOB}', filename=true)"

    t0 = perf_counter()
    print("\n[1/4] Counting total rows ...")
    total = con.execute(
        f"SELECT COUNT(*) FROM {read_expr}"
    ).fetchone()[0]
    t0 = step(f"Total rows: {total:,}", t0)

    print("\n[2/4] Building duplicate-key table (GROUP BY + HAVING) ...")
    con.execute(
        f"""
        CREATE TEMP TABLE dup_keys AS
        SELECT {KEYS_SQL}, COUNT(*) AS n
        FROM {read_expr}
        GROUP BY {KEYS_SQL}
        HAVING COUNT(*) > 1
        """
    )
    n_groups = con.execute("SELECT COUNT(*) FROM dup_keys").fetchone()[0]
    n_dup_rows = con.execute("SELECT COALESCE(SUM(n), 0) FROM dup_keys").fetchone()[0]
    max_rep = con.execute("SELECT COALESCE(MAX(n), 0) FROM dup_keys").fetchone()[0]
    t0 = step(
        f"Distinct duplicated key groups: {n_groups:,}\n"
        f"Total rows in duplicate groups: {n_dup_rows:,}\n"
        f"Max repetitions of a single key: {max_rep:,}",
        t0,
    )

    if n_groups == 0:
        print("\nNo duplicates found. Exiting.")
        return

    print("\n[3/4] Extracting full duplicate rows (semi join) ...")
    con.execute(
        f"""
        COPY (
            SELECT t.{', t.'.join(KEYS)}, t.filename AS __source_file
            FROM {read_expr} t
            SEMI JOIN dup_keys d USING ({KEYS_SQL})
            ORDER BY {KEYS_SQL}
        ) TO '{OUT_FILE}' (HEADER, DELIMITER ',')
        """
    )
    t0 = step(f"Wrote duplicates to {OUT_FILE}", t0)

    print("\n[4/4] Preview of first 20 duplicate rows:")
    preview = con.execute(
        f"""
        SELECT t.{', t.'.join(KEYS)}, t.filename AS __source_file
        FROM {read_expr} t
        SEMI JOIN dup_keys d USING ({KEYS_SQL})
        ORDER BY {KEYS_SQL}
        LIMIT 20
        """
    ).fetchdf()
    print(preview.to_string(index=False))

    print("\nSummary")
    print(f"  total rows scanned:           {total:,}")
    print(f"  duplicate key groups:         {n_groups:,}")
    print(f"  rows in duplicate groups:     {n_dup_rows:,}")
    print(f"  max repetitions per key:      {max_rep:,}")
    print(f"  output csv:                   {OUT_FILE}")


if __name__ == "__main__":
    main()
