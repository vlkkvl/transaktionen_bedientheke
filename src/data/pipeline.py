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
    define_goal_variable,
    filtering,
    remove_outliers,
)
from src.data.cleaning.convert import csv_gzip_to_parquet, csv_to_parquet
from src.data.preparation import (
    distribute_sales_over_active_days,
    distribute_sales_over_active_months,
    distribute_sales_over_active_weeks,
)

RAW_TRANSACTIONS_DIR = ROOT / "data" / "raw" / "transactions_5_years"
YEARLY_PARQUET_DIR = ROOT / "data" / "interim" / "transactions_per_year"


def has_yearly_parquet() -> bool:
    return any(YEARLY_PARQUET_DIR.glob("transactions_year_*.parquet"))


def raw_files_by_type() -> dict[str, list[Path]]:
    return {
        "csv": sorted(RAW_TRANSACTIONS_DIR.glob("*.csv")),
        "csv.gz": sorted(RAW_TRANSACTIONS_DIR.glob("*.csv.gz")),
        "parquet": sorted(RAW_TRANSACTIONS_DIR.glob("*.parquet")),
    }


def select_raw_converter() -> tuple[str, Callable[[], None]]:
    available = {
        file_type: files
        for file_type, files in raw_files_by_type().items()
        if files
    }
    if not available:
        raise FileNotFoundError(
            f"No raw transaction files found in {RAW_TRANSACTIONS_DIR}"
        )
    if len(available) > 1:
        counts = ", ".join(
            f"{file_type}={len(files)}" for file_type, files in available.items()
        )
        raise ValueError(
            f"Mixed raw transaction file types in {RAW_TRANSACTIONS_DIR}: {counts}"
        )

    file_type = next(iter(available))
    if file_type == "csv":
        return "CSV to yearly parquet", csv_to_parquet.main
    if file_type == "csv.gz":
        return "CSV gzip to yearly parquet", csv_gzip_to_parquet.main

    raise ValueError(
        "Raw parquet files are already parquet. Place yearly parquet files in "
        f"{YEARLY_PARQUET_DIR} or add a parquet passthrough converter."
    )


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


def make_raw_conversion_stage(force: bool) -> Callable[[], None]:
    def stage() -> None:
        if not force and has_yearly_parquet():
            print(
                "Skipping raw conversion; found yearly parquet files in "
                f"{YEARLY_PARQUET_DIR}"
            )
            print("Use --force-csv-conversion to rebuild them from raw files.")
            return

        title, converter = select_raw_converter()
        print(f"Using converter: {title}")
        converter()

    return stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-csv-conversion",
        action="store_true",
        help="Rebuild data/interim/transactions_per_year from raw transaction files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages: list[tuple[str, Callable[[], None]]] = [
        (
            "Raw files to yearly parquet",
            make_raw_conversion_stage(args.force_csv_conversion),
        ),
        ("Apply article filter report", filtering.main),
        ("Filter transactions and define ABVERKAUFTE_MENGE_KG", define_goal_variable.main),
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
