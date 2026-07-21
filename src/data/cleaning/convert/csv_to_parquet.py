"""Convert CSV files in data/raw/transactions to per-year Parquet files."""

from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.data.common import ROOT, clear_parquet_outputs

RAW_DIR = ROOT / "data" / "raw" / "transactions"
OUT_DIR = ROOT / "data" / "interim" / "transactions_per_year"

CHUNK_SIZE = 250_000


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_parquet_outputs(OUT_DIR)

    writers: dict[int, pq.ParquetWriter] = {}

    try:
        for file_index, path in enumerate(files, start=1):
            chunks = pd.read_csv(
                path,
                sep=",",
                chunksize=CHUNK_SIZE,
                low_memory=False,
                escapechar="\\",
                dtype={
                    "DEZENTRALE_MARKTAKTIONS_NR": "string",
                },
            )

            for chunk_index, chunk in enumerate(chunks, start=1):
                print(chunk.columns.tolist())
                dates = pd.to_datetime(
                    chunk["DATE"],
                    errors="coerce",
                )


                invalid_dates = dates.isna()
                if invalid_dates.any():
                    invalid_count = int(invalid_dates.sum())
                    raise ValueError(
                        f"{path.name}, chunk {chunk_index}: "
                        f"{invalid_count} invalid DATE values"
                    )

                chunk["YEAR"] = dates.dt.year.astype("int16")

                for year, group in chunk.groupby("YEAR", sort=False):
                    year = int(year)
                    group = group.drop(columns="YEAR")

                    if year not in writers:
                        table = pa.Table.from_pandas(
                            group,
                            preserve_index=False,
                        )

                        out_path = (
                            OUT_DIR
                            / f"transactions_year_{year}.parquet"
                        )

                        writers[year] = pq.ParquetWriter(
                            out_path,
                            table.schema,
                            compression="zstd",
                        )
                    else:
                        table = pa.Table.from_pandas(
                            group,
                            schema=writers[year].schema,
                            preserve_index=False,
                            safe=True,
                        )

                    writers[year].write_table(table)

                print(
                    f"[{file_index}/{len(files)}] "
                    f"{path.name} — chunk {chunk_index}"
                )

    finally:
        for writer in writers.values():
            writer.close()


if __name__ == "__main__":
    main()