"""Materialize filtered transactions with duplicate rows removed.

A duplicate is a row sharing the same ARTIKEL_ID, MARKT_ID, BON_ID, DATE, TIME,
and UMS_MENGE as another row. One row per duplicate key is retained in yearly
Parquet files under ``data/interim/transactions_no_dups``.
"""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import (
    ROOT,
    clear_parquet_outputs,
    columns_for_expr,
    configure_duckdb,
    ident,
    parquet_files,
    read_parquet_expr,
    sql_literal,
    step,
)
from src.data.cleaning.rules import (
    BINARY_FLAG_COLUMNS,
    DROP_COLUMNS,
    DUPLICATE_KEY_COLS,
    duplicate_keys_sql,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_per_year_filtered"
OUT_DIR = ROOT / "data" / "interim" / "transactions_no_dups"


def cleaned_columns(columns: list[str]) -> list[str]:
    """Return materialized columns, excluding technical source identifiers."""
    return [col for col in columns if col not in DROP_COLUMNS and col != "filename"]


def passthrough_select_sql(columns: list[str]) -> str:
    """Select retained columns without materializing technical identifiers."""
    return ",\n                    ".join(ident(col) for col in cleaned_columns(columns))


def deduplicated_select_sql(columns: list[str]) -> str:
    """Build a select that reduces every duplicate key to one transaction."""
    order_cols = [ident("filename")]
    if "EAN_ID" in columns:
        order_cols.append(ident("EAN_ID"))
    order_sql = ", ".join(order_cols)

    expressions = []
    for col in cleaned_columns(columns):
        col_sql = ident(col)
        if col in DUPLICATE_KEY_COLS:
            expressions.append(col_sql)
        elif col.casefold() in {flag.casefold() for flag in BINARY_FLAG_COLUMNS}:
            expressions.append(f"MAX({col_sql}) AS {col_sql}")
        else:
            expressions.append(f"FIRST({col_sql} ORDER BY {order_sql}) AS {col_sql}")
    return ",\n                    ".join(expressions)


def duplicate_key_match_sql(left_alias: str, right_alias: str) -> str:
    """Match duplicate keys null-safely across two relations."""
    return " AND ".join(
        f"{left_alias}.{ident(col)} IS NOT DISTINCT FROM "
        f"{right_alias}.{ident(col)}"
        for col in DUPLICATE_KEY_COLS
    )


def deduplicated_dataset_sql(columns: list[str], read_expr: str) -> str:
    """Stream unique rows and aggregate only rows in duplicate groups."""
    passthrough_select = passthrough_select_sql(columns)
    deduplicated_select = deduplicated_select_sql(columns)
    key_match = duplicate_key_match_sql("s", "d")
    keys_sql = duplicate_keys_sql("s")
    return f"""
        WITH source AS (
            SELECT *
            FROM {read_expr}
        ),
        unique_rows AS (
            SELECT
                {passthrough_select}
            FROM source AS s
            WHERE NOT EXISTS (
                SELECT 1
                FROM dup_keys AS d
                WHERE {key_match}
            )
        ),
        deduplicated_duplicate_rows AS (
            SELECT
                {deduplicated_select}
            FROM source AS s
            WHERE EXISTS (
                SELECT 1
                FROM dup_keys AS d
                WHERE {key_match}
            )
            GROUP BY {keys_sql}
        )
        SELECT * FROM unique_rows
        UNION ALL
        SELECT * FROM deduplicated_duplicate_rows
        """


def duplicate_stats(con, read_expr: str) -> tuple[int, int, int]:
    keys_sql = duplicate_keys_sql()
    return con.execute(
        f"""
        SELECT
            COUNT(*),
            COALESCE(SUM(n), 0),
            COALESCE(MAX(n), 0)
        FROM (
            SELECT COUNT(*) AS n
            FROM {read_expr}
            GROUP BY {keys_sql}
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()


def create_duplicate_key_table(con, read_expr: str) -> tuple[int, int, int]:
    """Materialize only duplicate keys and return their summary statistics."""
    keys_sql = duplicate_keys_sql()
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE dup_keys AS
        SELECT {keys_sql}, COUNT(*) AS n
        FROM {read_expr}
        GROUP BY {keys_sql}
        HAVING COUNT(*) > 1
        """
    )
    return con.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(n), 0), COALESCE(MAX(n), 0)
        FROM dup_keys
        """
    ).fetchone()


def main(in_dir: Path = IN_DIR, out_dir: Path = OUT_DIR) -> None:
    input_files = parquet_files(in_dir)
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {in_dir}")

    con = configure_duckdb(threads=2)
    con.execute("PRAGMA disable_progress_bar")
    con.execute("SET memory_limit = '4GB'")
    temp_dir = out_dir.parent / ".duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = {sql_literal(temp_dir)}")
    all_read_expr = read_parquet_expr(
        in_dir / "transactions_year_*.parquet",
        filename=True,
    )
    columns = columns_for_expr(con, all_read_expr)
    missing_keys = sorted(set(DUPLICATE_KEY_COLS) - set(columns))
    if missing_keys:
        raise ValueError(f"Input parquet files are missing columns: {missing_keys}")

    clear_parquet_outputs(out_dir)

    print(f"Reading filtered transactions from {in_dir}")
    print(f"Duplicate key columns: {DUPLICATE_KEY_COLS}")
    print(f"Writing de-duplicated transactions to {out_dir}")
    print("DuckDB resources: 2 threads, 4GB memory, disk spilling enabled")

    t0 = perf_counter()
    print("\n[1/3] Profiling duplicate keys ...")
    total = con.execute(f"SELECT COUNT(*) FROM {all_read_expr}").fetchone()[0]
    n_groups, n_dup_rows, max_rep = create_duplicate_key_table(
        con,
        all_read_expr,
    )
    expected_rows = total - n_dup_rows + n_groups
    t0 = step(
        "Duplicate scan finished: "
        f"{n_groups:,} key groups, {n_dup_rows:,} rows, max repetitions {max_rep:,}",
        t0,
    )

    print("\n[2/3] Writing de-duplicated yearly parquet files ...")
    written_rows = 0
    for input_file in input_files:
        output_file = out_dir / input_file.name
        read_expr = read_parquet_expr(input_file, filename=True)
        con.execute(
            f"""
            COPY ({deduplicated_dataset_sql(columns, read_expr)})
            TO {sql_literal(output_file)} (FORMAT PARQUET)
            """
        )
        rows = con.execute(
            f"SELECT COUNT(*) FROM {read_parquet_expr(output_file)}"
        ).fetchone()[0]
        written_rows += rows
        t0 = step(f"{input_file.name}: wrote {rows:,} rows", t0)

    print("\n[3/3] Validating output ...")
    output_expr = read_parquet_expr(out_dir / "transactions_year_*.parquet")
    remaining_groups, _, _ = duplicate_stats(con, output_expr)
    if written_rows != expected_rows:
        raise RuntimeError(
            "De-duplicated row count mismatch: "
            f"expected {expected_rows:,}, wrote {written_rows:,}"
        )
    if remaining_groups:
        raise RuntimeError(
            f"De-duplicated output still contains {remaining_groups:,} duplicate groups"
        )
    step("Validated row count and duplicate-key uniqueness", t0)

    print("\nSummary")
    print(f"  filtered rows scanned:       {total:,}")
    print(f"  duplicate key groups:        {n_groups:,}")
    print(f"  rows in duplicate groups:    {n_dup_rows:,}")
    print(f"  duplicate rows removed:      {total - written_rows:,}")
    print(f"  de-duplicated rows written:  {written_rows:,}")
    print(f"  output dir:                  {out_dir}")


if __name__ == "__main__":
    main()
