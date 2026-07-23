"""Convert raw write-off CSV.gz files to yearly Parquet files."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.data.common import ROOT, clear_parquet_outputs

RAW_DIR = ROOT / "data" / "raw" / "abschriften"
OUT_DIR = ROOT / "data" / "interim" / "abschriften"
OUTPUT_PATTERN = "abschriften_year_*.parquet"
CHUNK_SIZE = 250_000

DTYPES = {
    "ARTIKEL_ID": "Int64",
    "ARTIKEL_BEZ": "string",
    "WGR_ID": "Int64",
    "MANDANT_ID": "Int64",
    "MARKT_ID": "Int64",
    "ABSCHRIFT_ART": "string",
    "ABSCHRIFT_ART_BEZ": "string",
    "AKTIONSNUMMER": "Int64",
    "IST_ABSCHRIFTEN_MENGE": "float64",
    "ABSCHRIFTEN_IST_WERT_NETTO": "float64",
    "IST_ABSCHRIFTEN_WERT": "float64",
    "DATE": "string",
}

SCHEMA = pa.schema(
    [
        ("ARTIKEL_ID", pa.int64()),
        ("ARTIKEL_BEZ", pa.string()),
        ("WGR_ID", pa.int64()),
        ("MANDANT_ID", pa.int64()),
        ("MARKT_ID", pa.int64()),
        ("ABSCHRIFT_ART", pa.string()),
        ("ABSCHRIFT_ART_BEZ", pa.string()),
        ("AKTIONSNUMMER", pa.int64()),
        ("IST_ABSCHRIFTEN_MENGE", pa.float64()),
        ("ABSCHRIFTEN_IST_WERT_NETTO", pa.float64()),
        ("IST_ABSCHRIFTEN_WERT", pa.float64()),
        ("DATE", pa.string()),
    ]
)


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No CSV gzip files found in {RAW_DIR}")

    clear_parquet_outputs(OUT_DIR, pattern=OUTPUT_PATTERN)
    writers: dict[int, pq.ParquetWriter] = {}

    try:
        for file_index, path in enumerate(files, start=1):
            chunks = pd.read_csv(
                path,
                compression="gzip",
                sep=",",
                chunksize=CHUNK_SIZE,
                escapechar="\\",
                dtype=DTYPES,
            )

            for chunk_index, chunk in enumerate(chunks, start=1):
                dates = pd.to_datetime(
                    chunk["DATE"], format="%Y-%m-%d", errors="coerce"
                )
                invalid_dates = dates.isna()
                if invalid_dates.any():
                    raise ValueError(
                        f"{path.name}, chunk {chunk_index}: "
                        f"{int(invalid_dates.sum())} invalid DATE values"
                    )

                chunk["YEAR"] = dates.dt.year.astype("int16")
                for year, group in chunk.groupby("YEAR", sort=False):
                    year = int(year)
                    table = pa.Table.from_pandas(
                        group.drop(columns="YEAR"),
                        schema=SCHEMA,
                        preserve_index=False,
                        safe=True,
                    )
                    if year not in writers:
                        out_path = OUT_DIR / f"abschriften_year_{year}.parquet"
                        writers[year] = pq.ParquetWriter(
                            out_path,
                            SCHEMA,
                            compression="zstd",
                        )
                    writers[year].write_table(table)

                print(
                    f"[{file_index}/{len(files)}] {path.name} -- chunk {chunk_index}"
                )
    finally:
        for writer in writers.values():
            writer.close()


if __name__ == "__main__":
    main()
