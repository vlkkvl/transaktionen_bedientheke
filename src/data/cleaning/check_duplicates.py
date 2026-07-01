"""Check and remove duplicate transaction rows using DuckDB.

A duplicate is a row sharing the same ARTIKEL_ID, MARKT_ID, BON_ID, DATE,
TIME and UMS_MENGE as another row.
"""
from pathlib import Path
from time import perf_counter

import duckdb

ROOT = Path(__file__).resolve().parents[3]
IN_DIR = ROOT / "data" / "interim" / "transactions_per_year"
IN_GLOB = IN_DIR / "*.parquet"
DUP_OUT_DIR = ROOT / "data" / "interim" / "transactions_duplicates"
DUP_OUT_FILE = DUP_OUT_DIR / "duplicates_transactions_5_years.csv"
CLEAN_OUT_DIR = ROOT / "data" / "interim" / "transactions_per_year_no_dups"

KEYS = ["ARTIKEL_ID", "MARKT_ID", "BON_ID", "DATE", "TIME", "UMS_MENGE"]
DROP_COLUMNS = {"EAN_ID"}
BINARY_FLAG_COLUMNS = {
    "GEWICHTSARTIKEL",
    "WAAGENARTIKEL",
    "AKTION_KENNZEICHEN",
    "PREISUEBERSCHREIBUNG",
    "BONABBRUCH",
    "STORNOART",
    "STORNOZEILE",
    "NEGATIVARTIKEL",
}


def step(msg: str, t0: float) -> float:
    t1 = perf_counter()
    print(f"  ... done in {t1 - t0:.1f}s")
    print(msg)
    return t1


def sql_literal(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


KEYS_SQL = ", ".join(ident(col) for col in KEYS)


def read_parquet_expr(path: Path | str) -> str:
    return f"read_parquet({sql_literal(path)}, filename=true)"


def get_columns(con: duckdb.DuckDBPyConnection, read_expr: str) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()
    return [row[0] for row in rows if row[0] != "filename"]


def cleaned_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if col not in DROP_COLUMNS]


def passthrough_select_sql(columns: list[str]) -> str:
    return ",\n            ".join(ident(col) for col in cleaned_columns(columns))


def deduped_select_sql(columns: list[str]) -> str:
    order_sql = f"{ident('filename')}, {ident('EAN_ID')}"
    expressions = []
    for col in cleaned_columns(columns):
        col_sql = ident(col)
        if col in KEYS:
            expressions.append(col_sql)
        elif col in BINARY_FLAG_COLUMNS:
            expressions.append(f"MAX({col_sql}) AS {col_sql}")
        else:
            expressions.append(f"FIRST({col_sql} ORDER BY {order_sql}) AS {col_sql}")

    return ",\n            ".join(expressions)


def year_from_path(path: Path) -> int:
    return int(path.stem.removeprefix("transactions_year_"))


def main():
    DUP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(IN_DIR.glob("transactions_year_*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {IN_DIR}")

    print(f"Scanning {IN_GLOB}")
    print(f"Key columns: {KEYS}")
    print(f"Dropped columns in cleaned output: {sorted(DROP_COLUMNS)}")
    print(f"Binary flag columns aggregated with MAX: {sorted(BINARY_FLAG_COLUMNS)}")

    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    con.execute("PRAGMA threads=8")
    con.execute("SET preserve_insertion_order = false")

    read_expr = read_parquet_expr(IN_GLOB)
    columns = get_columns(con, read_expr)
    passthrough_select = passthrough_select_sql(columns)
    deduped_select = deduped_select_sql(columns)

    t0 = perf_counter()
    print("\n[1/5] Counting total rows ...")
    total = con.execute(
        f"SELECT COUNT(*) FROM {read_expr}"
    ).fetchone()[0]
    t0 = step(f"Total rows: {total:,}", t0)

    print("\n[2/5] Building duplicate-key table (GROUP BY + HAVING) ...")
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

    if n_groups:
        print("\n[3/5] Extracting duplicate rows (semi join) ...")
        con.execute(
            f"""
            COPY (
                SELECT
                    {", ".join(f"t.{ident(col)}" for col in KEYS)},
                    t.filename AS __source_file
                FROM {read_expr} t
                SEMI JOIN dup_keys d USING ({KEYS_SQL})
                ORDER BY {KEYS_SQL}
            ) TO {sql_literal(DUP_OUT_FILE)} (HEADER, DELIMITER ',')
            """
        )
        t0 = step(f"Wrote duplicates to {DUP_OUT_FILE}", t0)

        print("\n[4/5] Preview of first 20 duplicate rows:")
        preview = con.execute(
            f"""
            SELECT
                {", ".join(f"t.{ident(col)}" for col in KEYS)},
                t.filename AS __source_file
            FROM {read_expr} t
            SEMI JOIN dup_keys d USING ({KEYS_SQL})
            ORDER BY {KEYS_SQL}
            LIMIT 20
            """
        ).fetchdf()
        print(preview.to_string(index=False))
    else:
        print("\n[3/5] No duplicates found. Skipping duplicate CSV export.")
        print("\n[4/5] No duplicate preview available.")

    print("\n[5/5] Writing de-duplicated yearly parquet files ...")
    expected_clean_rows = total - (n_dup_rows - n_groups)
    written_rows = 0
    for input_file in input_files:
        year = year_from_path(input_file)
        output_file = CLEAN_OUT_DIR / input_file.name
        year_read_expr = read_parquet_expr(input_file)

        con.execute(
            f"""
            COPY (
                WITH source AS (
                    SELECT *
                    FROM {year_read_expr}
                ),
                unique_rows AS (
                    SELECT
                        {passthrough_select}
                    FROM source
                    ANTI JOIN dup_keys d USING ({KEYS_SQL})
                ),
                deduped_duplicate_rows AS (
                    SELECT
                        {deduped_select}
                    FROM source
                    SEMI JOIN dup_keys d USING ({KEYS_SQL})
                    GROUP BY {KEYS_SQL}
                )
                SELECT * FROM unique_rows
                UNION ALL
                SELECT * FROM deduped_duplicate_rows
            ) TO {sql_literal(output_file)} (FORMAT PARQUET)
            """
        )
        rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_literal(output_file)})"
        ).fetchone()[0]
        written_rows += rows
        t0 = step(f"Wrote {rows:,} rows for {year} to {output_file}", t0)

    print("\nSummary")
    print(f"  total rows scanned:           {total:,}")
    print(f"  duplicate key groups:         {n_groups:,}")
    print(f"  rows in duplicate groups:     {n_dup_rows:,}")
    print(f"  max repetitions per key:      {max_rep:,}")
    print(f"  expected cleaned rows:        {expected_clean_rows:,}")
    print(f"  written cleaned rows:         {written_rows:,}")
    print(f"  duplicate csv:                {DUP_OUT_FILE}")
    print(f"  cleaned parquet dir:          {CLEAN_OUT_DIR}")


if __name__ == "__main__":
    main()
