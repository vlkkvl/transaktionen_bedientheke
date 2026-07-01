"""Remove ABVERKAUFTE_MENGE outliers per (ARTIKEL_ID, MARKT_ID) series.

For each demand series, compute Q1, Q3, IQR on ABVERKAUFTE_MENGE and drop rows
above Q3 + IQR_K * IQR (default k=3, "extreme outlier" rule). Per-series bounds
are used because products are sold in different units (kg vs piece) and series
magnitudes differ by orders of magnitude.

Input:  data/processed/transactions_daily_agg/*.parquet
Output: data/processed/transactions_daily_agg_no_outliers/*.parquet
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
IN_DIR = ROOT / "data" / "processed" / "transactions_daily_agg"
OUT_DIR = ROOT / "data" / "processed" / "transactions_daily_agg_no_outliers"

DEMAND_COL = "ABVERKAUFTE_MENGE"
GROUP_COLS = ["ARTIKEL_ID", "MARKT_ID"]
IQR_K = 3.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    group_sql = ", ".join(GROUP_COLS)
    in_glob = str(IN_DIR / "*.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    con.execute(
        f"""
        CREATE TEMP TABLE bounds AS
        WITH q AS (
            SELECT
                {group_sql},
                quantile_cont({DEMAND_COL}, 0.25) AS q1,
                quantile_cont({DEMAND_COL}, 0.75) AS q3
            FROM read_parquet('{in_glob}')
            GROUP BY {group_sql}
        )
        SELECT {group_sql}, q1, q3, q3 + {IQR_K} * (q3 - q1) AS upper
        FROM q
        """
    )

    in_files = sorted(IN_DIR.glob("*.parquet"))
    n_in_total = 0
    n_out_total = 0
    for path in in_files:
        out_path = OUT_DIR / path.name
        n_in = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{path}')"
        ).fetchone()[0]
        con.execute(
            f"""
            COPY (
                SELECT t.*
                FROM read_parquet('{path}') t
                JOIN bounds b USING ({group_sql})
                WHERE t.{DEMAND_COL} <= b.upper
            ) TO '{out_path}' (FORMAT PARQUET)
            """
        )
        n_out = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{out_path}')"
        ).fetchone()[0]
        n_in_total += n_in
        n_out_total += n_out
        print(
            f"{path.name}: in={n_in:,} out={n_out:,} removed={n_in - n_out:,}"
        )

    removed = n_in_total - n_out_total
    pct = removed / n_in_total * 100 if n_in_total else 0.0
    print(
        f"\nTotal: in={n_in_total:,} out={n_out_total:,} "
        f"removed={removed:,} ({pct:.4f}%)"
    )


if __name__ == "__main__":
    main()
