"""Define the ABVERKAUFTE_MENGE_KG goal variable after transaction filtering."""
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
    read_parquet_expr,
    require_parquet_files,
    sql_literal,
    step,
)
from src.data.cleaning.rules import (
    GEWICHT_FLAG_COL,
    gewicht_flag_expression_sql,
    transaction_filter_condition,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_per_year"
OUT_DIR = ROOT / "data" / "interim" / "transactions_per_year_filtered"

TARGET_COL = "ABVERKAUFTE_MENGE_KG"
GEWICHT_FLAG = GEWICHT_FLAG_COL
GEWICHTSARTIKEL = "GEWICHTSARTIKEL"
ARTIKEL_INHALT = "ARTIKEL_INHALT"
GRAMM_BON = "GRAMM_BON"
UMS_MENGE = "UMS_MENGE"
REQUIRED_COLS = [GEWICHTSARTIKEL, ARTIKEL_INHALT, GRAMM_BON, UMS_MENGE]

WEIGHT_UNIT_PATTERN = (
    r"(^|[^[:alnum:]])(kil+ogramm?|kg|gramm?|gr\.?|g)([^[:alnum:]]|$)"
)


def temp_output_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.goal_variable.tmp{path.suffix}")


def validate_input_schema(columns: list[str]) -> None:
    missing = sorted(set(REQUIRED_COLS) - set(columns))
    if missing:
        raise ValueError(f"Input parquet files are missing columns: {missing}")


def contains_weight_unit_sql() -> str:
    content_sql = f"LOWER(COALESCE(CAST({ident(ARTIKEL_INHALT)} AS VARCHAR), ''))"
    return f"regexp_matches({content_sql}, {sql_literal(WEIGHT_UNIT_PATTERN)})"


def target_expression_sql() -> str:
    return f"""
        CASE
            WHEN COALESCE({ident(GEWICHTSARTIKEL)}, 0) = 1
                THEN CAST({ident(GRAMM_BON)} AS DOUBLE)
            WHEN {ident(GRAMM_BON)} IS NOT NULL
                AND {contains_weight_unit_sql()}
                THEN CAST({ident(GRAMM_BON)} AS DOUBLE)
            ELSE NULL
        END
        """


def output_select_sql(columns: list[str]) -> str:
    target_expr = f"({target_expression_sql()}) AS {ident(TARGET_COL)}"
    weight_flag_expr = (
        f"({gewicht_flag_expression_sql()})::TINYINT AS {ident(GEWICHT_FLAG)}"
    )
    expressions = []
    for col in columns:
        if col == TARGET_COL:
            expressions.append(target_expr)
        elif col == GEWICHT_FLAG:
            expressions.append(weight_flag_expr)
        else:
            expressions.append(ident(col))
    if TARGET_COL not in columns:
        expressions.append(target_expr)
    if GEWICHT_FLAG not in columns:
        expressions.append(weight_flag_expr)
    return ",\n            ".join(expressions)


def filtered_source_expr_sql(read_expr: str) -> str:
    return f"""
        (
            SELECT *
            FROM {read_expr}
            WHERE {transaction_filter_condition()}
        )
        """


def write_with_goal_variable(
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    output_path: Path,
    columns: list[str],
) -> tuple[int, int]:
    out_path = temp_output_path(output_path)
    if out_path.exists():
        out_path.unlink()

    read_expr = read_parquet_expr(input_path)
    filtered_expr = filtered_source_expr_sql(read_expr)
    rows_in = con.execute(f"SELECT COUNT(*) FROM {read_expr}").fetchone()[0]
    con.execute(
        f"""
        COPY (
            SELECT
                {output_select_sql(columns)}
            FROM {filtered_expr}
        )
        TO {sql_literal(out_path)}
        (FORMAT PARQUET)
        """
    )
    rows_out = con.execute(
        f"SELECT COUNT(*) FROM {read_parquet_expr(out_path)}"
    ).fetchone()[0]
    out_path.replace(output_path)
    return rows_in, rows_out


def main() -> None:
    input_files = require_parquet_files(IN_DIR)
    clear_parquet_outputs(OUT_DIR)
    con = configure_duckdb()

    t0 = perf_counter()
    total_in = 0
    total_out = 0
    for path in input_files:
        output_path = OUT_DIR / path.name
        read_expr = read_parquet_expr(path)
        columns = columns_for_expr(con, read_expr)
        validate_input_schema(columns)
        rows_in, rows_out = write_with_goal_variable(con, path, output_path, columns)
        total_in += rows_in
        total_out += rows_out
        print(f"{path.name}: in={rows_in:,}, filtered={rows_out:,}")

    print(f"\nTotal: in={total_in:,} out={total_out:,}")
    print(f"Output dir: {OUT_DIR}")
    step(f"Filtered transactions and defined {TARGET_COL}", t0)


if __name__ == "__main__":
    main()
