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
    ARTIKEL_BEZ_COL,
    DATE_COL,
    ALLOWED_WGR_IDS,
    EXCLUDED_ARTIKEL_BEZ_VALUES,
    EXCLUDED_GRAMM_BON_ARTICLE_IDS,
    EXTERNAL_PRODUCT_RULE,
    FCM_RULE,
    GRAMM_BON_RULE,
    MANDANT_RULE,
    MANDANT_ID_COL,
    MAX_TRANSACTION_DATE,
    MIN_UMS_MENGE,
    MIN_TRANSACTION_DATE,
    PSEUDO_ARTICLE_IDS,
    UMS_MENGE_COL,
    WEIGHT_CONTENT_LIKE,
    WEIGHT_RULE,
    WGR_ID_COL,
    WGR_RULE,
    artikel_bez_filter_condition,
    fcm_filter_condition,
    fcm_or_pseudo_filter_condition,
    gramm_bon_filter_condition,
    mandant_filter_condition,
    transaction_filter_condition,
    transaction_date_filter_condition,
    ums_menge_filter_condition,
    weight_filter_condition,
    wgr_filter_condition,
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
    print(f"EXTERNAL_PRODUCT_RULE: {EXTERNAL_PRODUCT_RULE}")
    if EXTERNAL_PRODUCT_RULE:
        print("Keeping only FCM and pseudo products")
    print(f"WEIGHT_RULE: {WEIGHT_RULE}")
    if WEIGHT_RULE:
        print(f"Keeping rows with ARTIKEL_INHALT LIKE {WEIGHT_CONTENT_LIKE!r}")
    print(f"WGR_RULE: {WGR_RULE}")
    if WGR_RULE:
        print(f"Keeping only {WGR_ID_COL} values: {sorted(ALLOWED_WGR_IDS)}")
    print(
        f"{DATE_COL} range: {MIN_TRANSACTION_DATE} through "
        f"{MAX_TRANSACTION_DATE}"
    )
    print(f"{UMS_MENGE_COL} threshold: > {MIN_UMS_MENGE}")
    print(
        f"Excluding {ARTIKEL_BEZ_COL} values: "
        f"{sorted(EXCLUDED_ARTIKEL_BEZ_VALUES)}"
    )
    print(f"GRAMM_BON_RULE: {GRAMM_BON_RULE}")
    if GRAMM_BON_RULE:
        print(
            "Excluding complete products with suspicious GRAMM_BON: "
            f"{len(EXCLUDED_GRAMM_BON_ARTICLE_IDS):,} articles"
        )
    print(
        "Pseudo-product IDs tagged downstream: "
        f"{len(PSEUDO_ARTICLE_IDS):,} articles"
    )

    t0 = perf_counter()
    total = count_rows(con, read_expr)

    current_condition = transaction_date_filter_condition()
    current_count = count_rows(con, read_expr, current_condition)
    date_removed = total - current_count
    date_remaining = current_count

    previous_condition = current_condition
    previous_count = current_count
    current_condition = combine_conditions(
        previous_condition,
        ums_menge_filter_condition(),
    )
    current_count = count_rows(con, read_expr, current_condition)
    ums_menge_removed = previous_count - current_count
    ums_menge_remaining = current_count

    previous_condition = current_condition
    previous_count = current_count
    artikel_bez_condition = artikel_bez_filter_condition()
    current_condition = combine_conditions(previous_condition, artikel_bez_condition)
    current_count = count_rows(con, read_expr, current_condition)
    artikel_bez_removed = previous_count - current_count
    artikel_bez_remaining = current_count

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

    external_product_removed = None
    external_product_remaining = None
    if EXTERNAL_PRODUCT_RULE:
        previous_condition = current_condition
        previous_count = current_count
        product_condition = fcm_or_pseudo_filter_condition()
        current_condition = combine_conditions(previous_condition, product_condition)
        current_count = count_rows(con, read_expr, current_condition)
        external_rows_removed = previous_count - current_count
        external_product_remaining = current_count
        external_articles_removed = con.execute(
            f"""
            SELECT COUNT(DISTINCT {ARTICLE_ID_COL}) AS articles_removed
            FROM {read_expr}
            WHERE ({previous_condition})
              AND NOT ({product_condition})
            """
        ).fetchone()[0]
        external_product_removed = (
            external_rows_removed,
            external_articles_removed,
        )

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

    wgr_removed = None
    wgr_remaining = None
    wgr_counts = []
    if WGR_RULE:
        previous_condition = current_condition
        previous_count = current_count
        wgr_condition = wgr_filter_condition()
        current_condition = combine_conditions(previous_condition, wgr_condition)
        current_count = count_rows(con, read_expr, current_condition)
        wgr_removed = previous_count - current_count
        wgr_remaining = current_count
        wgr_counts = con.execute(
            f"""
            SELECT {WGR_ID_COL}, COUNT(*) AS rows_kept,
                COUNT(DISTINCT {ARTICLE_ID_COL}) AS products_kept
            FROM {read_expr}
            WHERE ({previous_condition})
              AND ({wgr_condition})
            GROUP BY {WGR_ID_COL}
            ORDER BY {WGR_ID_COL}
            """
        ).fetchall()

    gramm_bon_removed = None
    gramm_bon_remaining = None
    gramm_bon_products = []
    if GRAMM_BON_RULE:
        previous_condition = current_condition
        previous_count = current_count
        gramm_bon_condition = gramm_bon_filter_condition()
        current_condition = combine_conditions(previous_condition, gramm_bon_condition)
        current_count = count_rows(con, read_expr, current_condition)
        gramm_bon_removed = previous_count - current_count
        gramm_bon_remaining = current_count
        gramm_bon_products = con.execute(
            f"""
            SELECT
                {ARTICLE_ID_COL},
                ANY_VALUE({ARTIKEL_BEZ_COL}) AS {ARTIKEL_BEZ_COL},
                COUNT(*) AS rows_removed,
                MAX(ABS(GRAMM_BON)) AS max_abs_gramm_bon
            FROM {read_expr}
            WHERE ({previous_condition})
              AND NOT ({gramm_bon_condition})
            GROUP BY {ARTICLE_ID_COL}
            ORDER BY max_abs_gramm_bon DESC, {ARTICLE_ID_COL}
            """
        ).fetchall()

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
        f"  {DATE_COL} filter: removed {date_removed:,} "
        f"after previous filters, remaining {date_remaining:,}"
    )
    print(
        f"  {UMS_MENGE_COL} filter: removed {ums_menge_removed:,} "
        f"after previous filters, remaining {ums_menge_remaining:,}"
    )
    print(
        f"  {ARTIKEL_BEZ_COL} filter: removed {artikel_bez_removed:,} "
        f"after previous filters, remaining {artikel_bez_remaining:,}"
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
    if external_product_removed is None:
        print("  External-product filter: disabled")
    else:
        print(
            "  External-product filter: removed "
            f"{external_product_removed[0]:,} rows after previous filters, "
            f"remaining {external_product_remaining:,} "
            f"({external_product_removed[1]:,} articles removed)"
        )
    if weight_removed is None:
        print("  ARTIKEL_INHALT weight filter: disabled")
    else:
        print(
            f"  ARTIKEL_INHALT weight filter: removed {weight_removed:,} "
            f"after previous filters, remaining {weight_remaining:,}"
        )
    if wgr_removed is None:
        print("  WGR_ID filter: disabled")
    else:
        print(
            f"  WGR_ID filter: removed {wgr_removed:,} "
            f"after previous filters, remaining {wgr_remaining:,}"
        )
        for wgr_id, rows_kept, products_kept in wgr_counts:
            print(
                f"    WGR_ID {wgr_id} kept: rows={rows_kept:,}; "
                f"products={products_kept:,}"
            )
    if gramm_bon_removed is None:
        print("  Suspicious GRAMM_BON product filter: disabled")
    else:
        print(
            "  Suspicious GRAMM_BON product filter: removed "
            f"{gramm_bon_removed:,} rows after previous filters, remaining "
            f"{gramm_bon_remaining:,}"
        )
        for (
            article_id,
            article_name,
            rows_removed,
            max_abs_gramm_bon,
        ) in gramm_bon_products:
            print(
                f"    {article_id}: {article_name!r}; rows={rows_removed:,}; "
                f"max_abs_GRAMM_BON={max_abs_gramm_bon:g}"
            )
    step("Filter scan finished", t0)


if __name__ == "__main__":
    main()
