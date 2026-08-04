"""Aggregate transactions to pooled daily demand with product-type markers."""
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
    read_parquet_expr,
    sql_literal,
    step,
)
from src.data.cleaning.rules import (
    FCM_COL,
    PSEUDO_COL,
    fcm_filter_condition,
    pseudo_filter_condition,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_no_dups"
OUT_DIR = ROOT / "data" / "interim" / "transactions_daily_agg"

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
REQUIRED_COLS = sorted(set(KEYS + SUM_COLS + FLAG_COLS + FIRST_COLS))


def validate_input_schema(columns: list[str]) -> None:
    missing = sorted(set(REQUIRED_COLS) - set(columns))
    if missing:
        raise ValueError(f"Input parquet files are missing columns: {missing}")


def aggregate_select_sql(columns: list[str], read_expr: str) -> str:
    static_select = ",\n            ".join(
        f"FIRST({ident(col)}) AS {ident(col)}" for col in FIRST_COLS
    )
    flag_select = ",\n            ".join(
        f"MAX(CASE WHEN {ident(col)} = 1 THEN 1 ELSE 0 END)::TINYINT AS {ident(col)}"
        for col in FLAG_COLS
    )

    return f"""
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
            BOOL_OR({fcm_filter_condition(columns=columns)})
                AS {ident(FCM_COL)},
            BOOL_OR({pseudo_filter_condition(columns=columns)})
                AS {ident(PSEUDO_COL)},
            {flag_select},
            {static_select}
        FROM {read_expr}
        GROUP BY {", ".join(ident(col) for col in KEYS)}
        """


def main(
    in_dir: Path | None = None,
    out_dir: Path = OUT_DIR,
) -> None:
    in_dir = in_dir or IN_DIR
    input_files = sorted(in_dir.glob("transactions_year_*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {in_dir}")

    clear_parquet_outputs(out_dir)

    con = configure_duckdb()
    in_glob = in_dir / "transactions_year_*.parquet"
    all_read_expr = read_parquet_expr(in_glob)
    columns = columns_for_expr(con, all_read_expr)
    validate_input_schema(columns)

    t0 = perf_counter()
    print(f"Reading de-duplicated transactions from {in_dir}")

    total_in = 0
    total_out = 0
    for input_file in input_files:
        output_file = out_dir / input_file.name
        read_expr = read_parquet_expr(input_file)
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
            f"{input_file.name}: {rows_in:,} de-duplicated rows -> "
            f"{rows_out:,} daily groups",
            t0,
        )

    print("\nSummary")
    print(f"  de-duplicated rows scanned: {total_in:,}")
    print(f"  daily groups written:  {total_out:,}")
    print(f"  output dir:            {out_dir}")


if __name__ == "__main__":
    main()
