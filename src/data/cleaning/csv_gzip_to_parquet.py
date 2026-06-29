"""Convert CSV.gz files in data/raw/transactions_5_years to per-year parquet files."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "transactions_5_years"
OUT_DIR = ROOT / "data" / "interim" / "transactions_per_year"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writers: dict[int, pq.ParquetWriter] = {}
    try:
        files = sorted(RAW_DIR.glob("*.csv.gz"))
        for i, path in enumerate(files, 1):
            df = pd.read_csv(path, compression="gzip", low_memory=False, escapechar="\\")
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
