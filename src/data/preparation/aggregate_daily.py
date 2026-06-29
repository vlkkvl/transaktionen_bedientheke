"""Aggregate transactions to daily per (ARTIKEL_ID, MARKT_ID, DATE)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
IN_DIR = ROOT / "data" / "interim" / "transactions_per_year"
OUT_DIR = ROOT / "data" / "processed" / "transactions_daily_agg"

KEYS = ["ARTIKEL_ID", "MARKT_ID", "DATE"]
SUM_COLS = ["UMS_MENGE", "ABVERKAUFTE_MENGE", "UMS_VK_WERT"]
FLAG_COLS = ["AKTION_KENNZEICHEN", "RABATT", "ARTIKELRABATT"]
FIRST_COLS = [
    "EAN_ID",
    "ARTIKEL_BEZ",
    "ARTIKEL_INHALT",
    "VERKAUFSEINHEIT",
    "GEWICHTSARTIKEL",
    "MARKT_NR",
    "MANDANT_ID",
    "LEH_SEH",
    "WGR_ID",
    "N_WARENKLASSE_KBEZ",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(IN_DIR.glob("transactions_year_*.parquet")):
        df = pd.read_parquet(src, columns=KEYS + SUM_COLS + FLAG_COLS + FIRST_COLS)
        for c in FLAG_COLS:
            df[c] = (df[c] == 1).astype("int8")
        agg = df.groupby(KEYS, as_index=False).agg(
            {
                **{c: "sum" for c in SUM_COLS},
                **{c: "max" for c in FLAG_COLS},
                **{c: "first" for c in FIRST_COLS},
            }
        )
        out = OUT_DIR / src.name
        agg.to_parquet(out, index=False)
        print(f"{src.name}: {len(df):,} rows -> {len(agg):,} groups")


if __name__ == "__main__":
    main()
