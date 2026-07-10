"""Define the ABVERKAUFTE_MENGE goal variable in yearly transaction parquet."""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import duckdb

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import (
    ROOT,
    columns_for_expr,
    configure_duckdb,
    ident,
    read_parquet_expr,
    require_parquet_files,
    sql_literal,
    step,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_per_year"

TARGET_COL = "ABVERKAUFTE_MENGE"
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
            ELSE CAST({ident(UMS_MENGE)} AS DOUBLE)
        END
        """


def output_select_sql(columns: list[str]) -> str:
    target_expr = f"({target_expression_sql()}) AS {ident(TARGET_COL)}"
    expressions = [
        target_expr if col == TARGET_COL else ident(col)
        for col in columns
    ]
    if TARGET_COL not in columns:
        expressions.append(target_expr)
    return ",\n            ".join(expressions)


def write_with_goal_variable(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    columns: list[str],
) -> tuple[int, int]:
    out_path = temp_output_path(path)
    if out_path.exists():
        out_path.unlink()

    read_expr = read_parquet_expr(path)
    rows_in = con.execute(f"SELECT COUNT(*) FROM {read_expr}").fetchone()[0]
    con.execute(
        f"""
        COPY (
            SELECT
                {output_select_sql(columns)}
            FROM {read_expr}
        )
        TO {sql_literal(out_path)}
        (FORMAT PARQUET)
        """
    )
    rows_out = con.execute(
        f"SELECT COUNT(*) FROM {read_parquet_expr(out_path)}"
    ).fetchone()[0]
    out_path.replace(path)
    return rows_in, rows_out


def main() -> None:
    input_files = require_parquet_files(IN_DIR)
    con = configure_duckdb()

    t0 = perf_counter()
    total_in = 0
    total_out = 0
    for path in input_files:
        read_expr = read_parquet_expr(path)
        columns = columns_for_expr(con, read_expr)
        validate_input_schema(columns)
        rows_in, rows_out = write_with_goal_variable(con, path, columns)
        total_in += rows_in
        total_out += rows_out
        print(f"{path.name}: rows={rows_out:,}")

    print(f"\nTotal: in={total_in:,} out={total_out:,}")
    step(f"Defined {TARGET_COL} in {IN_DIR}", t0)


if __name__ == "__main__":
    main()
