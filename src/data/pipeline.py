"""Run the transaction data pipeline end to end."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.common import ROOT
from src.data.cleaning import (
    aggregate_daily,
    check_duplicates,
    csv_gzip_to_parquet,
    filtering,
    remove_outliers,
)
from src.data.preparation import (
    distribute_sales_over_active_days,
    distribute_sales_over_active_months,
    distribute_sales_over_active_weeks,
)

YEARLY_PARQUET_DIR = ROOT / "data" / "interim" / "transactions_per_year"


def has_yearly_parquet() -> bool:
    return any(YEARLY_PARQUET_DIR.glob("transactions_year_*.parquet"))


def progress_header(index: int, total: int, title: str) -> None:
    width = 24
    done = int(width * index / max(total - 1, 1))
    bar = "#" * done + "." * (width - done)
    print("\n" + "=" * 72)
    print(f"[{index}/{total - 1}] {title}")
    print(f"[{bar}]")


def run_stage(index: int, total: int, title: str, fn: Callable[[], None]) -> None:
    progress_header(index, total, title)
    started_at = perf_counter()
    fn()
    print(f"Finished {title} in {perf_counter() - started_at:.1f}s")


def make_csv_stage(force: bool) -> Callable[[], None]:
    def stage() -> None:
        if force or not has_yearly_parquet():
            csv_gzip_to_parquet.main()
            return

        print(f"Skipping CSV conversion; found yearly parquet files in {YEARLY_PARQUET_DIR}")
        print("Use --force-csv-conversion to rebuild them from raw CSV gzip files.")

    return stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-csv-conversion",
        action="store_true",
        help="Rebuild data/interim/transactions_per_year from raw CSV gzip files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages: list[tuple[str, Callable[[], None]]] = [
        ("CSV gzip to yearly parquet", make_csv_stage(args.force_csv_conversion)),
        ("Apply article filter report", filtering.main),
        ("Export duplicate diagnostics", check_duplicates.main),
        ("Aggregate daily transactions (filtered + deduplicated)", aggregate_daily.main),
        ("Remove daily outliers", remove_outliers.main),
        ("Distribute sales over active days", distribute_sales_over_active_days.main),
        ("Aggregate active weeks", distribute_sales_over_active_weeks.main),
        ("Aggregate active months", distribute_sales_over_active_months.main),
    ]

    started_at = perf_counter()
    for index, (title, fn) in enumerate(stages):
        run_stage(index, len(stages), title, fn)

    print("\n" + "=" * 72)
    print(f"Pipeline finished in {perf_counter() - started_at:.1f}s")


if __name__ == "__main__":
    main()
