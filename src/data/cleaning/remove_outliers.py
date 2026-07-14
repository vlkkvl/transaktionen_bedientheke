"""Remove ABVERKAUFTE_MENGE_KG outliers per (ARTIKEL_ID, MARKT_ID) series.

For each demand series, compute Q1, Q3, IQR on ABVERKAUFTE_MENGE_KG and drop rows
above Q3 + IQR_K * IQR.

Input:  data/interim/transactions_daily_agg/*.parquet
Output: data/interim/transactions_daily_agg_no_outliers/*.parquet
"""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import (
    ROOT,
    clear_parquet_outputs,
    configure_duckdb,
    ident,
    read_parquet_expr,
    require_parquet_files,
    sql_literal,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_daily_agg"
OUT_DIR = ROOT / "data" / "interim" / "transactions_daily_agg_no_outliers"
FCM_IN_DIR = ROOT / "data" / "interim" / "transactions_daily_agg_fcm"
FCM_OUT_DIR = ROOT / "data" / "interim" / "transactions_daily_agg_fcm_no_outliers"

DEMAND_COL = "ABVERKAUFTE_MENGE_KG"
GROUP_COLS = ["ARTIKEL_ID", "MARKT_ID"]
IQR_K = 2.0


def temp_output_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.no_outliers.tmp{path.suffix}")


def main(in_dir: Path = IN_DIR, out_dir: Path = OUT_DIR) -> None:
    input_files = require_parquet_files(in_dir)
    clear_parquet_outputs(out_dir)
    group_sql = ", ".join(ident(col) for col in GROUP_COLS)
    in_glob = in_dir / "transactions_year_*.parquet"

    con = configure_duckdb()
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE bounds AS
        WITH q AS (
            SELECT
                {group_sql},
                quantile_cont({ident(DEMAND_COL)}, 0.25) AS q1,
                quantile_cont({ident(DEMAND_COL)}, 0.75) AS q3
            FROM {read_parquet_expr(in_glob)}
            GROUP BY {group_sql}
        )
        SELECT {group_sql}, q1, q3, q3 + {IQR_K} * (q3 - q1) AS upper
        FROM q
        """
    )

    n_in_total = 0
    n_out_total = 0
    for path in input_files:
        out_path = out_dir / path.name
        tmp_path = temp_output_path(out_path)
        if tmp_path.exists():
            tmp_path.unlink()

        n_in = con.execute(
            f"SELECT COUNT(*) FROM {read_parquet_expr(path)}"
        ).fetchone()[0]
        con.execute(
            f"""
            COPY (
                SELECT t.*
                FROM {read_parquet_expr(path)} t
                JOIN bounds b USING ({group_sql})
                WHERE t.{ident(DEMAND_COL)} <= b.upper
            ) TO {sql_literal(tmp_path)} (FORMAT PARQUET)
            """
        )
        n_out = con.execute(
            f"SELECT COUNT(*) FROM {read_parquet_expr(tmp_path)}"
        ).fetchone()[0]
        tmp_path.replace(out_path)

        n_in_total += n_in
        n_out_total += n_out
        print(f"{path.name}: in={n_in:,} out={n_out:,} removed={n_in - n_out:,}")

    removed = n_in_total - n_out_total
    pct = removed / n_in_total * 100 if n_in_total else 0.0
    print(
        f"\nTotal: in={n_in_total:,} out={n_out_total:,} "
        f"removed={removed:,} ({pct:.4f}%)"
    )
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
