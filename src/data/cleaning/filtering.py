"""Report the shared transaction filters.

The filter itself is intentionally exposed as SQL in ``rules.py`` so downstream
steps can apply it without materializing another full transaction table.
"""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import ROOT, configure_duckdb, read_parquet_expr, step
from src.data.cleaning.rules import (
    ALLOWED_FCM_ARTICLE_IDS,
    ALLOWED_MANDANT_IDS,
    ARTICLE_ID_COL,
    FCM_RULE,
    MANDANT_RULE,
    MANDANT_ID_COL,
    MIN_UMS_MENGE,
    UMS_MENGE_COL,
    fcm_filter_condition,
    mandant_filter_condition,
    transaction_filter_condition,
    ums_menge_filter_condition,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_per_year"
IN_GLOB = IN_DIR / "transactions_year_*.parquet"


def main() -> None:
    if not list(IN_DIR.glob("transactions_year_*.parquet")):
        raise FileNotFoundError(f"No parquet files found in {IN_DIR}")

    con = configure_duckdb()
    read_expr = read_parquet_expr(IN_GLOB)

    print(f"Reading transactions from {IN_DIR}")
    print(f"MANDANT_RULE: {MANDANT_RULE}")
    if MANDANT_RULE:
        print("Allowed MANDANT_ID values:")
        for mandant_id in sorted(ALLOWED_MANDANT_IDS):
            print(f"  {mandant_id}")
    print(f"FCM_RULE: {FCM_RULE}")
    if FCM_RULE:
        print(f"Allowed FCM ARTIKEL_ID values: {len(ALLOWED_FCM_ARTICLE_IDS):,}")
    print(f"{UMS_MENGE_COL} threshold: > {MIN_UMS_MENGE}")

    t0 = perf_counter()
    total = con.execute(f"SELECT COUNT(*) FROM {read_expr}").fetchone()[0]
    kept = con.execute(
        f"SELECT COUNT(*) FROM {read_expr} WHERE {transaction_filter_condition()}"
    ).fetchone()[0]

    mandant_counts = []
    mandant_removed = None
    if MANDANT_RULE:
        mandant_counts = con.execute(
            f"""
            SELECT {MANDANT_ID_COL}, COUNT(*) AS n
            FROM {read_expr}
            WHERE {MANDANT_ID_COL} IS NULL
               OR NOT ({mandant_filter_condition()})
            GROUP BY {MANDANT_ID_COL}
            ORDER BY {MANDANT_ID_COL} NULLS FIRST
            """
        ).fetchall()
        mandant_kept = con.execute(
            f"SELECT COUNT(*) FROM {read_expr} WHERE {mandant_filter_condition()}"
        ).fetchone()[0]
        mandant_removed = total - mandant_kept

    fcm_removed = None
    if FCM_RULE:
        fcm_removed = con.execute(
            f"""
            SELECT
                COUNT(*) AS rows_removed,
                COUNT(DISTINCT {ARTICLE_ID_COL}) AS articles_removed
            FROM {read_expr}
            WHERE NOT ({fcm_filter_condition()})
            """
        ).fetchone()

    ums_menge_removed = con.execute(
        f"SELECT COUNT(*) FROM {read_expr} WHERE NOT ({ums_menge_filter_condition()})"
    ).fetchone()[0]

    print("\nSummary")
    print(f"  rows in:      {total:,}")
    print(f"  rows kept:    {kept:,}")
    print(f"  rows removed: {total - kept:,}")
    print(f"  removed by {UMS_MENGE_COL} filter: {ums_menge_removed:,}")
    if mandant_removed is None:
        print("  MANDANT_ID filter: disabled")
    else:
        print(f"  removed by MANDANT_ID filter: {mandant_removed:,}")
        for mandant_id, count in mandant_counts:
            print(f"  MANDANT_ID {mandant_id}: {count:,}")
    if fcm_removed is None:
        print("  FCM filter: disabled")
    else:
        print(
            "  removed by FCM filter: "
            f"{fcm_removed[0]:,} rows, {fcm_removed[1]:,} articles"
        )
    step("Filter scan finished", t0)


if __name__ == "__main__":
    main()
