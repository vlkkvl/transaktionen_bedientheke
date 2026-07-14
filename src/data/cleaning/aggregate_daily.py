"""Aggregate base-filtered, de-duplicated transactions to daily demand."""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import duckdb

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
    FCM_RULE,
    duplicate_keys_sql,
    fcm_filter_condition,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_per_year_filtered"
BASE_IN_DIR = ROOT / "data" / "interim" / "transactions_per_year_filtered_no_fcm"
OUT_DIR = ROOT / "data" / "interim" / "transactions_daily_agg"
FCM_OUT_DIR = ROOT / "data" / "interim" / "transactions_daily_agg_fcm"

KEYS = ["ARTIKEL_ID", "MARKT_ID", "DATE"]
SUM_COLS = ["UMS_MENGE", "ABVERKAUFTE_MENGE_KG", "UMS_VK_WERT", "GRAMM_BON"]
QTY_EPS = 1e-3
FLAG_COLS = ["AKTION_KENNZEICHEN", "RABATT", "ARTIKELRABATT"]
FIRST_COLS = [
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
REQUIRED_COLS = sorted(set(KEYS + SUM_COLS + FLAG_COLS + FIRST_COLS + DUPLICATE_KEY_COLS))


def cleaned_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if col not in DROP_COLUMNS and col != "filename"]


def passthrough_select_sql(columns: list[str]) -> str:
    return ",\n            ".join(ident(col) for col in cleaned_columns(columns))


def deduped_select_sql(columns: list[str]) -> str:
    order_cols = [ident("filename")]
    if "EAN_ID" in columns:
        order_cols.append(ident("EAN_ID"))
    order_sql = ", ".join(order_cols)

    expressions = []
    for col in cleaned_columns(columns):
        col_sql = ident(col)
        if col in DUPLICATE_KEY_COLS:
            expressions.append(col_sql)
        elif col in BINARY_FLAG_COLUMNS:
            expressions.append(f"MAX({col_sql}) AS {col_sql}")
        else:
            expressions.append(f"FIRST({col_sql} ORDER BY {order_sql}) AS {col_sql}")

    return ",\n            ".join(expressions)


def validate_input_schema(columns: list[str]) -> None:
    missing = sorted(set(REQUIRED_COLS) - set(columns))
    if missing:
        raise ValueError(f"Input parquet files are missing columns: {missing}")


def create_duplicate_key_table(
    con: duckdb.DuckDBPyConnection,
    read_expr: str,
) -> tuple[int, int]:
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
    n_groups = con.execute("SELECT COUNT(*) FROM dup_keys").fetchone()[0]
    n_rows = con.execute("SELECT COALESCE(SUM(n), 0) FROM dup_keys").fetchone()[0]
    return n_groups, n_rows


def aggregate_select_sql(columns: list[str], read_expr: str) -> str:
    keys_sql = duplicate_keys_sql()
    passthrough_select = passthrough_select_sql(columns)
    deduped_select = deduped_select_sql(columns)
    static_select = ",\n            ".join(
        f"FIRST({ident(col)}) AS {ident(col)}" for col in FIRST_COLS
    )
    flag_select = ",\n            ".join(
        f"MAX(CASE WHEN {ident(col)} = 1 THEN 1 ELSE 0 END)::TINYINT AS {ident(col)}"
        for col in FLAG_COLS
    )

    return f"""
        WITH source AS (
            SELECT *
            FROM {read_expr}
        ),
        unique_rows AS (
            SELECT
                {passthrough_select}
            FROM source
            ANTI JOIN dup_keys d USING ({keys_sql})
        ),
        deduped_duplicate_rows AS (
            SELECT
                {deduped_select}
            FROM source
            SEMI JOIN dup_keys d USING ({keys_sql})
            GROUP BY {keys_sql}
        ),
        cleaned AS (
            SELECT * FROM unique_rows
            UNION ALL
            SELECT * FROM deduped_duplicate_rows
        )
        SELECT
            {ident("ARTIKEL_ID")},
            {ident("MARKT_ID")},
            {ident("DATE")},
            SUM(COALESCE({ident("UMS_MENGE")}, 0.0))::DOUBLE AS {ident("UMS_MENGE")},
            CASE
                WHEN ABS(SUM(COALESCE({ident("ABVERKAUFTE_MENGE_KG")}, 0.0))) < {QTY_EPS}
                    THEN 0.0
                ELSE SUM(COALESCE({ident("ABVERKAUFTE_MENGE_KG")}, 0.0))::DOUBLE
            END AS {ident("ABVERKAUFTE_MENGE_KG")},
            SUM(COALESCE({ident("UMS_VK_WERT")}, 0.0))::DOUBLE AS {ident("UMS_VK_WERT")},
            SUM(COALESCE({ident("GRAMM_BON")}, 0.0))::DOUBLE AS {ident("GRAMM_BON")},
            {flag_select},
            {static_select}
        FROM cleaned
        GROUP BY {", ".join(ident(col) for col in KEYS)}
        """


def default_input_dir() -> Path:
    if FCM_RULE and parquet_files(BASE_IN_DIR):
        return BASE_IN_DIR
    return IN_DIR


def materialize_fcm_daily_agg(
    con: duckdb.DuckDBPyConnection,
    source_dir: Path = OUT_DIR,
    out_dir: Path = FCM_OUT_DIR,
) -> None:
    source_files = parquet_files(source_dir)
    if not source_files:
        raise FileNotFoundError(f"No parquet files found in {source_dir}")

    clear_parquet_outputs(out_dir)

    total_in = 0
    total_out = 0
    print(f"\nMaterializing FCM-filtered daily aggregates to {out_dir}")
    for path in source_files:
        output_file = out_dir / path.name
        source_expr = read_parquet_expr(path)
        rows_in = con.execute(f"SELECT COUNT(*) FROM {source_expr}").fetchone()[0]
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM {source_expr}
                WHERE {fcm_filter_condition()}
            )
            TO {sql_literal(output_file)}
            (FORMAT PARQUET)
            """
        )
        rows_out = con.execute(
            f"SELECT COUNT(*) FROM {read_parquet_expr(output_file)}"
        ).fetchone()[0]
        total_in += rows_in
        total_out += rows_out
        print(f"{path.name}: in={rows_in:,} out={rows_out:,}")

    print(f"FCM daily groups written: {total_out:,} of {total_in:,}")


def main(
    in_dir: Path | None = None,
    out_dir: Path = OUT_DIR,
    *,
    write_fcm: bool | None = None,
) -> None:
    in_dir = in_dir or default_input_dir()
    input_files = sorted(in_dir.glob("transactions_year_*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {in_dir}")

    clear_parquet_outputs(out_dir)

    con = configure_duckdb()
    in_glob = in_dir / "transactions_year_*.parquet"
    all_read_expr = read_parquet_expr(in_glob, filename=True)
    columns = columns_for_expr(con, all_read_expr)
    validate_input_schema(columns)

    t0 = perf_counter()
    print(f"Reading filtered transactions from {in_dir}")
    n_groups, n_dup_rows = create_duplicate_key_table(con, all_read_expr)
    t0 = step(
        f"Prepared duplicate keys: {n_groups:,} groups, {n_dup_rows:,} rows",
        t0,
    )

    total_in = 0
    total_out = 0
    for input_file in input_files:
        output_file = out_dir / input_file.name
        read_expr = read_parquet_expr(input_file, filename=True)
        rows_in = con.execute(f"SELECT COUNT(*) FROM {read_expr}").fetchone()[0]
        con.execute(
            f"""
            COPY ({aggregate_select_sql(columns, read_expr)})
            TO {sql_literal(output_file)}
            (FORMAT PARQUET)
            """
        )
        rows_out = con.execute(
            f"SELECT COUNT(*) FROM {read_parquet_expr(output_file)}"
        ).fetchone()[0]
        total_in += rows_in
        total_out += rows_out
        t0 = step(
            f"{input_file.name}: {rows_in:,} filtered rows -> {rows_out:,} daily groups",
            t0,
        )

    print("\nSummary")
    print(f"  filtered rows scanned: {total_in:,}")
    print(f"  daily groups written:  {total_out:,}")
    print(f"  output dir:            {out_dir}")

    if write_fcm is None:
        write_fcm = FCM_RULE and out_dir == OUT_DIR
    if write_fcm:
        materialize_fcm_daily_agg(con, out_dir, FCM_OUT_DIR)


if __name__ == "__main__":
    main()
