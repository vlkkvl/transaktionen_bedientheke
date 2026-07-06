"""Shared transaction cleaning rules."""
from __future__ import annotations

from pathlib import Path

from src.data.common import ident, read_parquet_expr

ARTICLE_ID_COL = "ARTIKEL_ID"
EXCLUDED_ARTICLES = {
    1103534: "Geflügel Gewichtseingabe",
    1325959: "Fleisch/Wurst (unklar was verkauft wird, seit 2022 nicht mehr verkauft)",
}

DUPLICATE_KEY_COLS = ["ARTIKEL_ID", "MARKT_ID", "BON_ID", "DATE", "TIME", "UMS_MENGE"]
DROP_COLUMNS = {"EAN_ID"}
BINARY_FLAG_COLUMNS = {
    "GEWICHTSARTIKEL",
    "WAAGENARTIKEL",
    "AKTION_KENNZEICHEN",
    "PREISUEBERSCHREIBUNG",
    "BONABBRUCH",
    "STORNOART",
    "STORNOZEILE",
    "NEGATIVARTIKEL",
}


def excluded_ids_sql() -> str:
    return ", ".join(str(article_id) for article_id in sorted(EXCLUDED_ARTICLES))


def article_filter_condition(alias: str | None = None) -> str:
    col = ident(ARTICLE_ID_COL)
    if alias:
        col = f"{alias}.{col}"
    return f"{col} IS NULL OR {col} NOT IN ({excluded_ids_sql()})"


def duplicate_keys_sql(alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    return ", ".join(f"{prefix}{ident(col)}" for col in DUPLICATE_KEY_COLS)


def filtered_transactions_expr(path: Path | str, *, filename: bool = False) -> str:
    return f"""
        (
            SELECT *
            FROM {read_parquet_expr(path, filename=filename)}
            WHERE {article_filter_condition()}
        )
        """
