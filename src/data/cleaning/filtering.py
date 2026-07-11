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
    WEIGHT_CONTENT_LIKE,
    WEIGHT_RULE,
    fcm_filter_condition,
    mandant_filter_condition,
    transaction_filter_condition,
    ums_menge_filter_condition,
    weight_filter_condition,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_per_year"
IN_GLOB = IN_DIR / "transactions_year_*.parquet"


def combine_conditions(left: str, right: str) -> str:
    return f"({left}) AND ({right})"


def count_rows(
    con,
    read_expr: str,
    condition: str | None = None,
) -> int:
    where_sql = f" WHERE {condition}" if condition else ""
    return con.execute(f"SELECT COUNT(*) FROM {read_expr}{where_sql}").fetchone()[0]


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
    print(f"WEIGHT_RULE: {WEIGHT_RULE}")
    if WEIGHT_RULE:
        print(f"Keeping rows with ARTIKEL_INHALT LIKE {WEIGHT_CONTENT_LIKE!r}")
    print(f"{UMS_MENGE_COL} threshold: > {MIN_UMS_MENGE}")

    t0 = perf_counter()
    total = count_rows(con, read_expr)

    current_condition = ums_menge_filter_condition()
    current_count = count_rows(con, read_expr, current_condition)
    ums_menge_removed = total - current_count
    ums_menge_remaining = current_count

    mandant_counts = []
    mandant_removed = None
    mandant_remaining = None
    if MANDANT_RULE:
        previous_condition = current_condition
        previous_count = current_count
        mandant_condition = mandant_filter_condition()
        current_condition = combine_conditions(previous_condition, mandant_condition)
        current_count = count_rows(con, read_expr, current_condition)
        mandant_removed = previous_count - current_count
        mandant_remaining = current_count
        mandant_counts = con.execute(
            f"""
            SELECT {MANDANT_ID_COL}, COUNT(*) AS n
            FROM {read_expr}
            WHERE ({previous_condition})
              AND (
                  {MANDANT_ID_COL} IS NULL
                  OR NOT ({mandant_condition})
              )
            GROUP BY {MANDANT_ID_COL}
            ORDER BY {MANDANT_ID_COL} NULLS FIRST
            """
        ).fetchall()

    fcm_removed = None
    fcm_remaining = None
    if FCM_RULE:
        previous_condition = current_condition
        previous_count = current_count
        fcm_condition = fcm_filter_condition()
        current_condition = combine_conditions(previous_condition, fcm_condition)
        current_count = count_rows(con, read_expr, current_condition)
        fcm_rows_removed = previous_count - current_count
        fcm_remaining = current_count
        fcm_articles_removed = con.execute(
            f"""
            SELECT COUNT(DISTINCT {ARTICLE_ID_COL}) AS articles_removed
            FROM {read_expr}
            WHERE ({previous_condition})
              AND NOT ({fcm_condition})
            """
        ).fetchone()[0]
        fcm_removed = (fcm_rows_removed, fcm_articles_removed)

    weight_removed = None
    weight_remaining = None
    if WEIGHT_RULE:
        previous_condition = current_condition
        previous_count = current_count
        weight_condition = weight_filter_condition()
        current_condition = combine_conditions(previous_condition, weight_condition)
        current_count = count_rows(con, read_expr, current_condition)
        weight_removed = previous_count - current_count
        weight_remaining = current_count

    kept = current_count
    shared_kept = count_rows(con, read_expr, transaction_filter_condition())
    if kept != shared_kept:
        raise RuntimeError(
            "Sequential filter report does not match shared transaction filter: "
            f"sequential={kept:,}, shared={shared_kept:,}"
        )

    print("\nSummary")
    print(f"  rows in:      {total:,}")
    print(f"  rows kept:    {kept:,}")
    print(f"  rows removed: {total - kept:,}")
    print("\nSequential filter impact")
    print(
        f"  {UMS_MENGE_COL} filter: removed {ums_menge_removed:,} "
        f"after previous filters, remaining {ums_menge_remaining:,}"
    )
    if mandant_removed is None:
        print("  MANDANT_ID filter: disabled")
    else:
        print(
            f"  MANDANT_ID filter: removed {mandant_removed:,} "
            f"after previous filters, remaining {mandant_remaining:,}"
        )
        for mandant_id, count in mandant_counts:
            print(f"  MANDANT_ID {mandant_id}: {count:,}")
    if fcm_removed is None:
        print("  FCM filter: disabled")
    else:
        print(
            f"  FCM filter: removed {fcm_removed[0]:,} rows "
            f"after previous filters, remaining {fcm_remaining:,} "
            f"({fcm_removed[1]:,} articles removed)"
        )
    if weight_removed is None:
        print("  ARTIKEL_INHALT weight filter: disabled")
    else:
        print(
            f"  ARTIKEL_INHALT weight filter: removed {weight_removed:,} "
            f"after previous filters, remaining {weight_remaining:,}"
        )
    step("Filter scan finished", t0)


if __name__ == "__main__":
    main()
