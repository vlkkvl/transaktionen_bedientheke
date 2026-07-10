"""Convert CSV files in data/raw/transactions_5_years to per-year parquet files."""
from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.data.common import ROOT, clear_parquet_outputs

RAW_DIR = ROOT / "data" / "raw" / "transactions_5_years"
OUT_DIR = ROOT / "data" / "interim" / "transactions_per_year"


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    clear_parquet_outputs(OUT_DIR)
    writers: dict[int, pq.ParquetWriter] = {}
    try:
        for i, path in enumerate(files, 1):
            df = pd.read_csv(
                path,
                sep=";",
                low_memory=False,
                escapechar="\\",
                dtype={"DEZENTRALE_MARKTAKTIONS_NR": "string"},
            )
            df["YEAR"] = pd.to_datetime(df["DATE"]).dt.year
            for year, group in df.groupby("YEAR"):
                group = group.drop(columns=["YEAR"])
                if year not in writers:
                    table = pa.Table.from_pandas(group, preserve_index=False)
                    out_path = OUT_DIR / f"transactions_year_{year}.parquet"
                    writers[year] = pq.ParquetWriter(out_path, table.schema)
                else:
                    table = pa.Table.from_pandas(
                        group, schema=writers[year].schema, preserve_index=False
                    )
                writers[year].write_table(table)
            print(f"[{i}/{len(files)}] {path.name}")
    finally:
        for w in writers.values():
            w.close()


if __name__ == "__main__":
    main()
