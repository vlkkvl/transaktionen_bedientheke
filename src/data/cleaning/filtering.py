"""Report the additional article-level transaction filter.

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
    ARTICLE_ID_COL,
    EXCLUDED_ARTICLES,
    article_filter_condition,
    excluded_ids_sql,
)

IN_DIR = ROOT / "data" / "interim" / "transactions_per_year"
IN_GLOB = IN_DIR / "transactions_year_*.parquet"


def main() -> None:
    if not list(IN_DIR.glob("transactions_year_*.parquet")):
        raise FileNotFoundError(f"No parquet files found in {IN_DIR}")

    con = configure_duckdb()
    read_expr = read_parquet_expr(IN_GLOB)

    print(f"Reading transactions from {IN_DIR}")
    print("Excluded article IDs:")
    for article_id, reason in sorted(EXCLUDED_ARTICLES.items()):
        print(f"  {article_id}: {reason}")

    t0 = perf_counter()
    counts = con.execute(
        f"""
        SELECT {ARTICLE_ID_COL}, COUNT(*) AS n
        FROM {read_expr}
        WHERE {ARTICLE_ID_COL} IN ({excluded_ids_sql()})
        GROUP BY {ARTICLE_ID_COL}
        ORDER BY {ARTICLE_ID_COL}
        """
    ).fetchall()
    total = con.execute(f"SELECT COUNT(*) FROM {read_expr}").fetchone()[0]
    kept = con.execute(
        f"SELECT COUNT(*) FROM {read_expr} WHERE {article_filter_condition()}"
    ).fetchone()[0]

    print("\nSummary")
    print(f"  rows in:      {total:,}")
    print(f"  rows kept:    {kept:,}")
    print(f"  rows removed: {total - kept:,}")
    for article_id, count in counts:
        print(f"  {article_id}:     {count:,}")
    step("Filter scan finished", t0)


if __name__ == "__main__":
    main()
